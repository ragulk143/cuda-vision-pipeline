#include "vision_kernels.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA Error: ") +              \
                                     cudaGetErrorString(err) + " at " +         \
                                     __FILE__ + ":" + std::to_string(__LINE__));\
        }                                                                       \
    } while (0)

namespace {

constexpr int TILE_W = 16;
constexpr int TILE_H = 16;
constexpr int HALO = 1;
constexpr int SMEM_W = TILE_W + 2 * HALO; // 18
constexpr int SMEM_H = TILE_H + 2 * HALO; // 18

__global__ void bgr_to_gray_kernel(const uint8_t* __restrict__ bgr,
                                   uint8_t* __restrict__ gray,
                                   int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int bgr_idx = idx * 3;
        float b = static_cast<float>(bgr[bgr_idx + 0]);
        float g = static_cast<float>(bgr[bgr_idx + 1]);
        float r = static_cast<float>(bgr[bgr_idx + 2]);
        gray[idx] = static_cast<uint8_t>(0.114f * b + 0.587f * g + 0.299f * r);
    }
}

__global__ void sobel_tiled_kernel(const uint8_t* __restrict__ gray,
                                   uint8_t* __restrict__ out,
                                   int width, int height) {
    __shared__ uint8_t smem[SMEM_H][SMEM_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * TILE_W + tx;
    int gy = blockIdx.y * TILE_H + ty;

    // Cooperative load of 18x18 halo tile into shared SRAM
    for (int dy = ty; dy < SMEM_H; dy += TILE_H) {
        int img_y = min(max(blockIdx.y * TILE_H + dy - HALO, 0), height - 1);
        for (int dx = tx; dx < SMEM_W; dx += TILE_W) {
            int img_x = min(max(blockIdx.x * TILE_W + dx - HALO, 0), width - 1);
            smem[dy][dx] = gray[img_y * width + img_x];
        }
    }
    __syncthreads();

    if (gx < width && gy < height) {
        int sx = tx + HALO;
        int sy = ty + HALO;

        int gx_val = -smem[sy - 1][sx - 1] + smem[sy - 1][sx + 1]
                     -2 * smem[sy][sx - 1] + 2 * smem[sy][sx + 1]
                     -smem[sy + 1][sx - 1] + smem[sy + 1][sx + 1];

        int gy_val = -smem[sy - 1][sx - 1] - 2 * smem[sy - 1][sx] - smem[sy - 1][sx + 1]
                     +smem[sy + 1][sx - 1] + 2 * smem[sy + 1][sx] + smem[sy + 1][sx + 1];

        int mag = min(static_cast<int>(sqrtf(static_cast<float>(gx_val * gx_val + gy_val * gy_val))), 255);
        out[gy * width + gx] = static_cast<uint8_t>(mag);
    }
}

} // namespace

struct CudaVisionEngine::Impl {
    cudaStream_t streams[NUM_SLOTS]{};
    cudaEvent_t ev_kernel_start[NUM_SLOTS]{};
    cudaEvent_t ev_kernel_stop[NUM_SLOTS]{};

    uint8_t* h_pinned_bgr[NUM_SLOTS]{};
    uint8_t* h_pinned_sobel[NUM_SLOTS]{};

    uint8_t* d_bgr[NUM_SLOTS]{};
    uint8_t* d_gray[NUM_SLOTS]{};
    uint8_t* d_sobel[NUM_SLOTS]{};
};

CudaVisionEngine::CudaVisionEngine(int width, int height)
    : width_(width),
      height_(height),
      bgr_bytes_(static_cast<size_t>(width) * height * 3),
      gray_bytes_(static_cast<size_t>(width) * height),
      impl_(new Impl()) {
    for (int i = 0; i < NUM_SLOTS; ++i) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&impl_->streams[i], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&impl_->ev_kernel_start[i]));
        CUDA_CHECK(cudaEventCreate(&impl_->ev_kernel_stop[i]));

        CUDA_CHECK(cudaMallocHost(&impl_->h_pinned_bgr[i], bgr_bytes_));
        CUDA_CHECK(cudaMallocHost(&impl_->h_pinned_sobel[i], gray_bytes_));

        CUDA_CHECK(cudaMalloc(&impl_->d_bgr[i], bgr_bytes_));
        CUDA_CHECK(cudaMalloc(&impl_->d_gray[i], gray_bytes_));
        CUDA_CHECK(cudaMalloc(&impl_->d_sobel[i], gray_bytes_));
    }
}

CudaVisionEngine::~CudaVisionEngine() {
    if (!impl_) return;
    for (int i = 0; i < NUM_SLOTS; ++i) {
        if (impl_->streams[i]) {
            cudaStreamSynchronize(impl_->streams[i]);
            cudaStreamDestroy(impl_->streams[i]);
        }
        if (impl_->ev_kernel_start[i]) cudaEventDestroy(impl_->ev_kernel_start[i]);
        if (impl_->ev_kernel_stop[i])  cudaEventDestroy(impl_->ev_kernel_stop[i]);

        if (impl_->h_pinned_bgr[i])   cudaFreeHost(impl_->h_pinned_bgr[i]);
        if (impl_->h_pinned_sobel[i]) cudaFreeHost(impl_->h_pinned_sobel[i]);

        if (impl_->d_bgr[i])   cudaFree(impl_->d_bgr[i]);
        if (impl_->d_gray[i])  cudaFree(impl_->d_gray[i]);
        if (impl_->d_sobel[i]) cudaFree(impl_->d_sobel[i]);
    }
    delete impl_;
}

cv::Mat CudaVisionEngine::get_pinned_input_mat(int slot) {
    return cv::Mat(height_, width_, CV_8UC3, impl_->h_pinned_bgr[slot]);
}

cv::Mat CudaVisionEngine::get_pinned_output_mat(int slot) {
    return cv::Mat(height_, width_, CV_8UC1, impl_->h_pinned_sobel[slot]);
}

void CudaVisionEngine::enqueue_frame(int slot) {
    cudaStream_t stream = impl_->streams[slot];

    dim3 block(TILE_W, TILE_H);
    dim3 grid((width_ + TILE_W - 1) / TILE_W, (height_ + TILE_H - 1) / TILE_H);

    // 1. Async Host-to-Device DMA
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_bgr[slot], impl_->h_pinned_bgr[slot], bgr_bytes_,
                               cudaMemcpyHostToDevice, stream));

    // 2. Timed Custom CUDA Kernels on stream
    CUDA_CHECK(cudaEventRecord(impl_->ev_kernel_start[slot], stream));
    bgr_to_gray_kernel<<<grid, block, 0, stream>>>(impl_->d_bgr[slot], impl_->d_gray[slot], width_, height_);
    sobel_tiled_kernel<<<grid, block, 0, stream>>>(impl_->d_gray[slot], impl_->d_sobel[slot], width_, height_);
    CUDA_CHECK(cudaEventRecord(impl_->ev_kernel_stop[slot], stream));

    // 3. Async Device-to-Host DMA
    CUDA_CHECK(cudaMemcpyAsync(impl_->h_pinned_sobel[slot], impl_->d_sobel[slot], gray_bytes_,
                               cudaMemcpyDeviceToHost, stream));
}

float CudaVisionEngine::sync_and_get_kernel_ms(int slot) {
    CUDA_CHECK(cudaStreamSynchronize(impl_->streams[slot]));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, impl_->ev_kernel_start[slot], impl_->ev_kernel_stop[slot]));
    return ms;
}
