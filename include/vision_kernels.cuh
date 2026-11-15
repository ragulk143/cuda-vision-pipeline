#pragma once

#include <opencv2/core.hpp>
#include <cstddef>
#include <cstdint>

class CudaVisionEngine {
public:
    // 3 slots: Producer fills slot A while GPU processes slot B and Consumer reads slot C
    static constexpr int NUM_SLOTS = 3;

    CudaVisionEngine(int width, int height);
    ~CudaVisionEngine();

    // Non-copyable RAII resource owner
    CudaVisionEngine(const CudaVisionEngine&) = delete;
    CudaVisionEngine& operator=(const CudaVisionEngine&) = delete;

    // Zero-copy cv::Mat views directly over page-locked (pinned) host memory
    cv::Mat get_pinned_input_mat(int slot);
    cv::Mat get_pinned_output_mat(int slot);

    // Enqueues H2D -> BGR2Gray -> Shared-Mem Sobel -> D2H asynchronously on streams_[slot]
    void enqueue_frame(int slot);

    // Synchronizes only streams_[slot] and returns pure GPU kernel execution time (ms)
    float sync_and_get_kernel_ms(int slot);

    int width() const { return width_; }
    int height() const { return height_; }

private:
    int width_;
    int height_;
    size_t bgr_bytes_;
    size_t gray_bytes_;

    struct Impl;
    Impl* impl_;
};
