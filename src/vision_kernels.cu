#include "vision_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); } while (0)

namespace {
constexpr int TILE_W = 16, TILE_H = 16, HALO = 1;
constexpr int SMEM_W = TILE_W + 2 * HALO, SMEM_H = TILE_H + 2 * HALO;
__global__ void bgr_to_gray_kernel(const uint8_t* __restrict__ bgr, uint8_t* __restrict__ gray, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        int i = y * w + x;
        gray[i] = static_cast<uint8_t>(0.114f * bgr[i * 3] + 0.587f * bgr[i * 3 + 1] + 0.299f * bgr[i * 3 + 2]);
    }
}
__global__ void bgr_resize_norm_nchw_kernel(const uint8_t* __restrict__ bgr, float* __restrict__ nchw,
                                            int src_w, int src_h, int dst_w, int dst_h) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x, dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx < dst_w && dy < dst_h) {
        int sx = min(static_cast<int>((dx + 0.5f) * src_w / dst_w), src_w - 1);
        int sy = min(static_cast<int>((dy + 0.5f) * src_h / dst_h), src_h - 1);
        int s_idx = (sy * src_w + sx) * 3, d_idx = dy * dst_w + dx, plane = dst_w * dst_h;
        nchw[0 * plane + d_idx] = (bgr[s_idx + 2] / 255.0f - 0.485f) / 0.229f;
        nchw[1 * plane + d_idx] = (bgr[s_idx + 1] / 255.0f - 0.456f) / 0.224f;
        nchw[2 * plane + d_idx] = (bgr[s_idx + 0] / 255.0f - 0.406f) / 0.225f;
    }
}
__global__ void sobel_tiled_kernel(const uint8_t* __restrict__ gray, uint8_t* __restrict__ out, int w, int h) {
    __shared__ uint8_t smem[SMEM_H][SMEM_W];
    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * TILE_W + tx, gy = blockIdx.y * TILE_H + ty;
    for (int dy = ty; dy < SMEM_H; dy += TILE_H) {
        int iy = min(max(blockIdx.y * TILE_H + dy - HALO, 0), h - 1);
        for (int dx = tx; dx < SMEM_W; dx += TILE_W) {
            int ix = min(max(blockIdx.x * TILE_W + dx - HALO, 0), w - 1);
            smem[dy][dx] = gray[iy * w + ix];
        }
    }
    __syncthreads();
    if (gx < w && gy < h) {
        int sx = tx + HALO, sy = ty + HALO;
        int vx = -smem[sy-1][sx-1] + smem[sy-1][sx+1] - 2*smem[sy][sx-1] + 2*smem[sy][sx+1] - smem[sy+1][sx-1] + smem[sy+1][sx+1];
        int vy = -smem[sy-1][sx-1] - 2*smem[sy-1][sx] - smem[sy-1][sx+1] + smem[sy+1][sx-1] + 2*smem[sy+1][sx] + smem[sy+1][sx+1];
        out[gy * w + gx] = static_cast<uint8_t>(min(static_cast<int>(sqrtf(float(vx*vx + vy*vy))), 255));
    }
}
} // namespace
struct CudaVisionEngine::Impl {
    cudaStream_t streams[NUM_SLOTS]{};
    cudaEvent_t ev_kernel_start[NUM_SLOTS]{}, ev_kernel_stop[NUM_SLOTS]{};
    uint8_t *h_pinned_bgr[NUM_SLOTS]{}, *h_pinned_sobel[NUM_SLOTS]{};
    uint8_t *d_bgr[NUM_SLOTS]{}, *d_gray[NUM_SLOTS]{}, *d_sobel[NUM_SLOTS]{};
    float* d_tensor[NUM_SLOTS]{};
};
CudaVisionEngine::CudaVisionEngine(int width, int height)
    : width_(width), height_(height), bgr_bytes_(size_t(width)*height*3),
      gray_bytes_(size_t(width)*height), tensor_bytes_(TENSOR_ELEMS * sizeof(float)), impl_(new Impl()) {
    for (int i = 0; i < NUM_SLOTS; ++i) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&impl_->streams[i], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&impl_->ev_kernel_start[i]));
        CUDA_CHECK(cudaEventCreate(&impl_->ev_kernel_stop[i]));
        CUDA_CHECK(cudaMallocHost(&impl_->h_pinned_bgr[i], bgr_bytes_));
        CUDA_CHECK(cudaMallocHost(&impl_->h_pinned_sobel[i], gray_bytes_));
        CUDA_CHECK(cudaMalloc(&impl_->d_bgr[i], bgr_bytes_));
        CUDA_CHECK(cudaMalloc(&impl_->d_gray[i], gray_bytes_));
        CUDA_CHECK(cudaMalloc(&impl_->d_sobel[i], gray_bytes_));
        CUDA_CHECK(cudaMalloc(&impl_->d_tensor[i], tensor_bytes_));
    }
}
CudaVisionEngine::~CudaVisionEngine() {
    if (!impl_) return;
    for (int i = 0; i < NUM_SLOTS; ++i) {
        if (impl_->streams[i]) { cudaStreamSynchronize(impl_->streams[i]); cudaStreamDestroy(impl_->streams[i]); }
        if (impl_->ev_kernel_start[i]) cudaEventDestroy(impl_->ev_kernel_start[i]);
        if (impl_->ev_kernel_stop[i]) cudaEventDestroy(impl_->ev_kernel_stop[i]);
        if (impl_->h_pinned_bgr[i]) cudaFreeHost(impl_->h_pinned_bgr[i]);
        if (impl_->h_pinned_sobel[i]) cudaFreeHost(impl_->h_pinned_sobel[i]);
        if (impl_->d_bgr[i]) cudaFree(impl_->d_bgr[i]);
        if (impl_->d_gray[i]) cudaFree(impl_->d_gray[i]);
        if (impl_->d_sobel[i]) cudaFree(impl_->d_sobel[i]);
        if (impl_->d_tensor[i]) cudaFree(impl_->d_tensor[i]);
    }
    delete impl_;
}
cv::Mat CudaVisionEngine::get_pinned_input_mat(int slot) { return cv::Mat(height_, width_, CV_8UC3, impl_->h_pinned_bgr[slot]); }
cv::Mat CudaVisionEngine::get_pinned_output_mat(int slot) { return cv::Mat(height_, width_, CV_8UC1, impl_->h_pinned_sobel[slot]); }
float* CudaVisionEngine::get_device_tensor_ptr(int slot) { return impl_->d_tensor[slot]; }
void* CudaVisionEngine::get_cuda_stream(int slot) { return impl_->streams[slot]; }
void CudaVisionEngine::enqueue_frame(int slot) {
    cudaStream_t st = impl_->streams[slot];
    dim3 blk(TILE_W, TILE_H), grd((width_+15)/16, (height_+15)/16), t_grd((MODEL_W+15)/16, (MODEL_H+15)/16);
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_bgr[slot], impl_->h_pinned_bgr[slot], bgr_bytes_, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaEventRecord(impl_->ev_kernel_start[slot], st));
    bgr_to_gray_kernel<<<grd, blk, 0, st>>>(impl_->d_bgr[slot], impl_->d_gray[slot], width_, height_);
    sobel_tiled_kernel<<<grd, blk, 0, st>>>(impl_->d_gray[slot], impl_->d_sobel[slot], width_, height_);
    bgr_resize_norm_nchw_kernel<<<t_grd, blk, 0, st>>>(impl_->d_bgr[slot], impl_->d_tensor[slot], width_, height_, MODEL_W, MODEL_H);
    CUDA_CHECK(cudaEventRecord(impl_->ev_kernel_stop[slot], st));
    CUDA_CHECK(cudaMemcpyAsync(impl_->h_pinned_sobel[slot], impl_->d_sobel[slot], gray_bytes_, cudaMemcpyDeviceToHost, st));
}
float CudaVisionEngine::sync_and_get_kernel_ms(int slot) {
    CUDA_CHECK(cudaStreamSynchronize(impl_->streams[slot]));
    float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, impl_->ev_kernel_start[slot], impl_->ev_kernel_stop[slot]));
    return ms;
}
