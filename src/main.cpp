#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "ring_buffer.hpp"
#include "vision_kernels.cuh"

struct FramePacket {
    uint64_t frame_id;
    cv::Mat image;
    cv::Mat edges;
    float cuda_kernel_ms;
    std::chrono::steady_clock::time_point capture_ts;
};

int main() {
    const int num_frames = 200;
    const int width = 1920, height = 1080;

    // 1. CPU Baseline Measurement
    cv::Mat sample(height, width, CV_8UC3, cv::Scalar(40, 80, 160));
    cv::circle(sample, {960, 540}, 200, {0, 255, 0}, -1);
    auto cpu_t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 50; ++i) {
        cv::Mat gray, gx, gy, abs_gx, abs_gy, cpu_edges;
        cv::cvtColor(sample, gray, cv::COLOR_BGR2GRAY);
        cv::Sobel(gray, gx, CV_16S, 1, 0);
        cv::Sobel(gray, gy, CV_16S, 0, 1);
        cv::convertScaleAbs(gx, abs_gx);
        cv::convertScaleAbs(gy, abs_gy);
        cv::addWeighted(abs_gx, 0.5, abs_gy, 0.5, 0, cpu_edges);
    }
    double cpu_avg_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - cpu_t0).count() / 50.0;

    // 2. Initialize Persistent RAII CUDA Engine ONCE (Warmup included)
    CudaVisionEngine gpu_engine(width, height);
    cv::Mat dummy_edges(height, width, CV_8UC1);
    for (int w = 0; w < 5; ++w) {
        gpu_engine.process_frame(sample.data, dummy_edges.data);
    }

    // 3. Multithreaded Zero-Copy RingBuffer Pipeline
    RingBuffer<FramePacket> q_cap_to_gpu(8);
    RingBuffer<FramePacket> q_gpu_to_metrics(8);
    std::atomic<bool> cap_running{true}, gpu_running{true};
    size_t drop1 = 0, drop2 = 0;

    std::thread t_cap([&]() {
        for (int i = 0; i < num_frames; ++i) {
            cv::Mat frame(height, width, CV_8UC3, cv::Scalar(40, 80, 160));
            cv::circle(frame, {960, 540}, 150 + (i % 60), {0, 255, 0}, -1);
            FramePacket pkt{static_cast<uint64_t>(i), std::move(frame),
                            cv::Mat(height, width, CV_8UC1), 0.0f,
                            std::chrono::steady_clock::now()};
            q_cap_to_gpu.push(std::move(pkt), drop1);
            std::this_thread::sleep_for(std::chrono::milliseconds(2)); // Simulate ~500 FPS camera
        }
        cap_running = false;
        q_cap_to_gpu.notify_all();
    });

    std::thread t_gpu([&]() {
        while (true) {
            bool running = cap_running.load();
            auto pkt_opt = q_cap_to_gpu.pop(running);
            if (!pkt_opt.has_value()) break;

            FramePacket pkt = std::move(pkt_opt.value());
            pkt.cuda_kernel_ms = gpu_engine.process_frame(
                pkt.image.data, pkt.edges.data);
            q_gpu_to_metrics.push(std::move(pkt), drop2);
        }
        gpu_running = false;
        q_gpu_to_metrics.notify_all();
    });

    std::vector<float> kernel_times;
    auto start_all = std::chrono::steady_clock::now();

    std::thread t_metrics([&]() {
        while (true) {
            bool running = gpu_running.load();
            auto pkt_opt = q_gpu_to_metrics.pop(running);
            if (!pkt_opt.has_value()) break;
            kernel_times.push_back(pkt_opt->cuda_kernel_ms);
        }
    });

    t_cap.join();
    t_gpu.join();
    t_metrics.join();

    double total_sec = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - start_all).count();
    std::sort(kernel_times.begin(), kernel_times.end());
    float p50_gpu = kernel_times[kernel_times.size() / 2];

    std::cout << "\n======================================================\n"
              << " RTX 3050 (SM_86) Persistent + Shared-Mem CUDA v2\n"
              << "======================================================\n"
              << "Frames Processed     : " << kernel_times.size() << " (Dropped: " << (drop1 + drop2) << ")\n"
              << "CPU OpenCV Avg       : " << cpu_avg_ms << " ms / frame\n"
              << "RTX 3050 CUDA Kernel : " << p50_gpu << " ms / frame (p50)\n"
              << "GPU Kernel Speedup   : " << (cpu_avg_ms / p50_gpu) << "x faster than CPU\n"
              << "End-to-End Throughput: " << (kernel_times.size() / total_sec) << " FPS\n"
              << "======================================================\n";
    return 0;
}
