#pragma once
#include <cstdint>
#include <cstddef>

class CudaVisionEngine {
public:
    CudaVisionEngine(int width, int height);
    ~CudaVisionEngine();

    // Disable copy to protect GPU pointers (Rule of 5 / RAII)
    CudaVisionEngine(const CudaVisionEngine&) = delete;
    CudaVisionEngine& operator=(const CudaVisionEngine&) = delete;

    // Runs H2D -> Grayscale -> Sobel -> D2H using pre-allocated Pinned + Device memory
    float process_frame(const uint8_t* src_bgr, uint8_t* dst_edges);

private:
    int width_;
    int height_;
    size_t bgr_bytes_;
    size_t gray_bytes_;

    // Page-locked (Pinned) Host memory for high-speed PCIe DMA transfers
    uint8_t* h_pinned_bgr_ = nullptr;
    uint8_t* h_pinned_edges_ = nullptr;

    // Persistent GPU VRAM buffers (allocated ONCE in constructor)
    uint8_t* d_bgr_ = nullptr;
    uint8_t* d_gray_ = nullptr;
    uint8_t* d_edges_ = nullptr;
};
