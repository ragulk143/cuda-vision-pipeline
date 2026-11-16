#include "ring_buffer.hpp"
#include "vision_kernels.cuh"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
namespace {
std::atomic<bool> g_sigint_received{false};
void handle_sigint(int) { g_sigint_received.store(true); }
}
struct FramePacket {
    int frame_id = 0;
    int slot_id = 0;
    Clock::time_point t_capture;
};
class SlotPool {
public:
    explicit SlotPool(int n) : busy_(n, false) {}
    bool acquire(int s, const std::atomic<bool>& stop) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait(lk, [&]() { return !busy_[s] || stop.load() || g_sigint_received.load(); });
        if (stop.load() || g_sigint_received.load()) return false;
        busy_[s] = true;
        return true;
    }
    void release(int s) { { std::lock_guard<std::mutex> lk(mtx_); busy_[s] = false; } cv_.notify_all(); }
    void wake_all() { cv_.notify_all(); }
private:
    std::vector<bool> busy_;
    std::mutex mtx_;
    std::condition_variable cv_;
};
struct Config {
    std::string source = "synthetic";
    int max_frames = 0;
    bool headless = false;
    bool loop_video = false;
};
Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--source" && i + 1 < argc) cfg.source = argv[++i];
        else if (a == "--frames" && i + 1 < argc) cfg.max_frames = std::stoi(argv[++i]);
        else if (a == "--headless") cfg.headless = true;
        else if (a == "--loop") cfg.loop_video = true;
    }
    if (cfg.source == "synthetic" && cfg.max_frames == 0) cfg.max_frames = 200;
    return cfg;
}
bool is_cam(const std::string& s, int& idx) {
    if (s.empty()) return false;
    for (char c : s) if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    idx = std::stoi(s); return true;
}
int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);
    Config cfg = parse_args(argc, argv);
    cv::VideoCapture cap;
    int width = 1920, height = 1080;
    bool is_syn = (cfg.source == "synthetic");
    if (!is_syn) {
        int idx = 0;
        if (is_cam(cfg.source, idx)) {
            cap.open(idx, cv::CAP_V4L2); if (!cap.isOpened()) cap.open(idx);
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
            cap.set(cv::CAP_PROP_FPS, 30); cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        } else { cap.open(cfg.source); }
        if (!cap.isOpened()) { std::cerr << "[ERROR] Cannot open: " << cfg.source << "\n"; return 1; }
        width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    } else { cfg.headless = true; }
    std::cout << "=== cuda-vision-pipeline v4 ===\nSource: " << cfg.source << " | " << width << "x" << height << "\n";
    CudaVisionEngine engine(width, height);
    SlotPool pool(CudaVisionEngine::NUM_SLOTS);
    RingBuffer<FramePacket> q_in(CudaVisionEngine::NUM_SLOTS), q_out(CudaVisionEngine::NUM_SLOTS);
    std::atomic<bool> stop{false};
    std::vector<double> e2e_ms; std::vector<float> kern_ms;
    e2e_ms.reserve(100000); kern_ms.reserve(100000);
    engine.enqueue_frame(0); engine.sync_and_get_kernel_ms(0);
    auto t0 = Clock::now();
    std::thread prod([&]() {
        for (int id = 0; !stop.load() && !g_sigint_received.load(); ++id) {
            if (cfg.max_frames > 0 && id >= cfg.max_frames) break;
            int s = id % CudaVisionEngine::NUM_SLOTS;
            if (!pool.acquire(s, stop)) break;
            cv::Mat v = engine.get_pinned_input_mat(s); uint8_t* ptr = v.data;
            if (is_syn) { v.setTo(cv::Scalar(20, 25, 30)); cv::circle(v, {width/2, height/2}, 90, {0, 220, 255}, -1); }
            else if (!cap.read(v) || v.empty()) {
                if (cfg.loop_video) { cap.set(cv::CAP_PROP_POS_FRAMES, 0); pool.release(s); --id; continue; }
                pool.release(s); break;
            } else if (v.data != ptr || v.cols != width || v.rows != height) {
                cv::resize(v, engine.get_pinned_input_mat(s), {width, height});
            }
            if (q_in.push(FramePacket{id, s, Clock::now()})) pool.release(s);
        }
        q_in.close();
    });
    std::thread gpu([&]() {
        while (auto p = q_in.pop()) { engine.enqueue_frame(p->slot_id); if (q_out.push(std::move(*p))) pool.release(p->slot_id); }
        q_out.close();
    });
    std::thread cons([&]() {
        cv::Mat comb(height, width * 2, CV_8UC3), disp;
        cv::Mat l_roi = comb({0, 0, width, height}), r_roi = comb({width, 0, width, height});
        double r_fps = 0.0; auto t_hud = Clock::now(); int cnt = 0;
        while (auto p = q_out.pop()) {
            float km = engine.sync_and_get_kernel_ms(p->slot_id); auto now = Clock::now();
            e2e_ms.push_back(std::chrono::duration<double, std::milli>(now - p->t_capture).count()); kern_ms.push_back(km); ++cnt;
            double dt = std::chrono::duration<double>(now - t_hud).count();
            if (dt >= 0.25) { r_fps = cnt / dt; cnt = 0; t_hud = now; }
            if (!cfg.headless) {
                engine.get_pinned_input_mat(p->slot_id).copyTo(l_roi);
                cv::cvtColor(engine.get_pinned_output_mat(p->slot_id), r_roi, cv::COLOR_GRAY2BGR);
                if (comb.cols > 1280) cv::pyrDown(comb, disp); else disp = comb;
                cv::putText(disp, "v4 | FPS: " + std::to_string(int(r_fps)) + " | Kernel: " + std::to_string(km).substr(0,4) + "ms", {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.65, {0,255,140}, 2);
                cv::imshow("CUDA Vision Engine v4", disp);
                int k = cv::waitKey(1) & 0xFF; if (k == 'q' || k == 27) { stop.store(true); pool.wake_all(); }
            }
            pool.release(p->slot_id);
        }
        if (!cfg.headless) cv::destroyAllWindows();
    });
    prod.join(); gpu.join(); cons.join();
    double sec = std::chrono::duration<double>(Clock::now() - t0).count();
    std::sort(e2e_ms.begin(), e2e_ms.end()); std::sort(kern_ms.begin(), kern_ms.end());
    size_t n = e2e_ms.size();
    std::cout << std::fixed << std::setprecision(3) << "\n--- v4 Telemetry Summary ---\nFrames Processed : " << n
              << "\nDropped Frames   : " << (q_in.dropped_count() + q_out.dropped_count())
              << "\nGPU Kernel p50   : " << (n ? kern_ms[n/2] : 0.0f) << " ms\nE2E Latency p50  : " << (n ? e2e_ms[n/2] : 0.0)
              << " ms\nE2E Latency p95  : " << (n ? e2e_ms[size_t(n*0.95)] : 0.0)
              << " ms\nE2E Throughput   : " << std::setprecision(2) << (sec > 0 ? n / sec : 0.0) << " FPS\n";
    return 0;
}
