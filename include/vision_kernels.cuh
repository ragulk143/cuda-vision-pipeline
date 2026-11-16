#pragma once

#include <opencv2/core.hpp>
#include <cstddef>
#include <cstdint>

class CudaVisionEngine {
public:
    static constexpr int NUM_SLOTS = 3;
    static constexpr int MODEL_W = 224;
    static constexpr int MODEL_H = 224;
    static constexpr size_t TENSOR_ELEMS = 32 * 3 * MODEL_H * MODEL_W;

    CudaVisionEngine(int width, int height);
    ~CudaVisionEngine();

    CudaVisionEngine(const CudaVisionEngine&) = delete;
    CudaVisionEngine& operator=(const CudaVisionEngine&) = delete;

    cv::Mat get_pinned_input_mat(int slot);
    cv::Mat get_pinned_output_mat(int slot);

    void enqueue_frame(int slot);
    float sync_and_get_kernel_ms(int slot);

    // Zero-copy GPU pointers for ONNX Runtime IoBinding (v5)
    float* get_device_tensor_ptr(int slot);
    void* get_cuda_stream(int slot);

    int width() const { return width_; }
    int height() const { return height_; }

private:
    int width_;
    int height_;
    size_t bgr_bytes_;
    size_t gray_bytes_;
    size_t tensor_bytes_;

    struct Impl;
    Impl* impl_;
};
