#include "ring_buffer.hpp"
#include "vision_kernels.cuh"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

struct FramePacket {
    int frame_id = 0;
    int slot_id = 0;
    Clock::time_point t_capture;
};

// Lightweight slot-pool synchronizer so Producer never overwrites an in-flight pinned buffer
class SlotPool {
public:
    explicit SlotPool(int num_slots) : busy_(num_slots, false) {}

    void acquire(int slot) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [&]() { return !busy_[slot]; });
        busy_[slot] = true;
    }

    void release(int slot) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            busy_[slot] = false;
        }
        cv_.notify_all();
    }

private:
    std::vector<bool> busy_;
    std::mutex mtx_;
    std::condition_variable cv_;
};

int main() {
    constexpr int WIDTH = 1920;
    constexpr int HEIGHT = 1080;
    constexpr int TOTAL_FRAMES = 200;

    std::cout << "=== cuda-vision-pipeline v3 (Async Streams + Zero-Copy Pinned Host) ===\n";
    std::cout << "Resolution: " << WIDTH << "x" << HEIGHT
              << " | Frames: " << TOTAL_FRAMES
              << " | Async Slots: " << CudaVisionEngine::NUM_SLOTS << "\n\n";

    CudaVisionEngine engine(WIDTH, HEIGHT);
    SlotPool slot_pool(CudaVisionEngine::NUM_SLOTS);

    RingBuffer<FramePacket> capture_to_gpu(CudaVisionEngine::NUM_SLOTS);
    RingBuffer<FramePacket> gpu_to_consumer(CudaVisionEngine::NUM_SLOTS);

    std::vector<double> e2e_latencies_ms;
    std::vector<float> kernel_latencies_ms;
    e2e_latencies_ms.reserve(TOTAL_FRAMES);
    kernel_latencies_ms.reserve(TOTAL_FRAMES);

    // Warmup GPU context & driver before timing
    engine.enqueue_frame(0);
    engine.sync_and_get_kernel_ms(0);

    auto t_pipeline_start = Clock::now();

    // Thread 1: Producer (Writes directly into zero-copy pinned cv::Mat)
    std::thread producer([&]() {
        for (int i = 0; i < TOTAL_FRAMES; ++i) {
            int slot = i % CudaVisionEngine::NUM_SLOTS;
            slot_pool.acquire(slot);

            cv::Mat pinned_frame = engine.get_pinned_input_mat(slot);

            // Synthetic moving geometric pattern (zero extra memcpy!)
            pinned_frame.setTo(cv::Scalar(20, 25, 30));
            int cx = (i * 12) % (WIDTH - 200) + 100;
            int cy = (i * 7) % (HEIGHT - 200) + 100;
            cv::rectangle(pinned_frame, cv::Rect(cx - 80, cy - 80, 160, 160),
                          cv::Scalar(0, 220, 255), -1);
            cv::circle(pinned_frame, cv::Point(WIDTH - cx, HEIGHT - cy), 90,
                       cv::Scalar(255, 100, 180), 6);

            FramePacket pkt{i, slot, Clock::now()};
            if (capture_to_gpu.push(std::move(pkt))) {
                slot_pool.release(slot); // Release if dropped
            }
        }
        capture_to_gpu.close();
    });

    // Thread 2: GPU Vision Worker (Non-blocking async enqueue)
    std::thread gpu_worker([&]() {
        while (auto pkt_opt = capture_to_gpu.pop()) {
            FramePacket pkt = std::move(*pkt_opt);
            engine.enqueue_frame(pkt.slot_id);

            if (gpu_to_consumer.push(std::move(pkt))) {
                slot_pool.release(pkt.slot_id);
            }
        }
        gpu_to_consumer.close();
    });

    // Thread 3: Consumer & Telemetry (Stream sync + latency audit)
    std::thread consumer([&]() {
        while (auto pkt_opt = gpu_to_consumer.pop()) {
            FramePacket pkt = std::move(*pkt_opt);

            // Wait only for this slot's async stream to finish D2H
            float k_ms = engine.sync_and_get_kernel_ms(pkt.slot_id);

            // Output is now ready in zero-copy pinned host memory:
            cv::Mat sobel_out = engine.get_pinned_output_mat(pkt.slot_id);
            (void)sobel_out; // Used for cv::imshow in v4

            auto t_done = Clock::now();
            double e2e_ms = std::chrono::duration<double, std::milli>(t_done - pkt.t_capture).count();

            e2e_latencies_ms.push_back(e2e_ms);
            kernel_latencies_ms.push_back(k_ms);

            // Free slot for Producer reuse
            slot_pool.release(pkt.slot_id);
        }
    });

    producer.join();
    gpu_worker.join();
    consumer.join();

    auto t_pipeline_end = Clock::now();
    double total_sec = std::chrono::duration<double>(t_pipeline_end - t_pipeline_start).count();

    // Compute p50 / p95 telemetry
    std::sort(e2e_latencies_ms.begin(), e2e_latencies_ms.end());
    std::sort(kernel_latencies_ms.begin(), kernel_latencies_ms.end());

    size_t n = e2e_latencies_ms.size();
    double e2e_p50 = n ? e2e_latencies_ms[n / 2] : 0.0;
    double e2e_p95 = n ? e2e_latencies_ms[static_cast<size_t>(n * 0.95)] : 0.0;
    float kern_p50 = n ? kernel_latencies_ms[n / 2] : 0.0f;
    double fps = n / total_sec;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "--- v3 Benchmark Results (RTX 3050 SM_86) ---\n";
    std::cout << "Frames Processed : " << n << " / " << TOTAL_FRAMES << "\n";
    std::cout << "Dropped Frames   : " << (TOTAL_FRAMES - n) << "\n";
    std::cout << "GPU Kernel p50   : " << kern_p50 << " ms\n";
    std::cout << "E2E Latency p50  : " << e2e_p50 << " ms\n";
    std::cout << "E2E Latency p95  : " << e2e_p95 << " ms\n";
    std::cout << "E2E Throughput   : " << std::setprecision(2) << fps << " FPS\n";

    return 0;
}
