#include "vision_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>

#define TILE_W 16
#define TILE_H 16
#define SMEM_W (TILE_W + 2)
#define SMEM_H (TILE_H + 2)

CudaVisionEngine::CudaVisionEngine(int width, int height)
    : width_(width), height_(height),
      bgr_bytes_(static_cast<size_t>(width) * height * 3),
      gray_bytes_(static_cast<size_t>(width) * height) {
    cudaMallocHost(&h_pinned_bgr_, bgr_bytes_);
    cudaMallocHost(&h_pinned_edges_, gray_bytes_);

    cudaMalloc(&d_bgr_, bgr_bytes_);
    cudaMalloc(&d_gray_, gray_bytes_);
    cudaMalloc(&d_edges_, gray_bytes_);
}

CudaVisionEngine::~CudaVisionEngine() {
    if (h_pinned_bgr_) cudaFreeHost(h_pinned_bgr_);
    if (h_pinned_edges_) cudaFreeHost(h_pinned_edges_);
    if (d_bgr_) cudaFree(d_bgr_);
    if (d_gray_) cudaFree(d_gray_);
    if (d_edges_) cudaFree(d_edges_);
}

__global__ void bgr_to_gray_kernel(const uint8_t* __restrict__ bgr,
                                   uint8_t* __restrict__ gray,
                                   int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int bgr_idx = idx * 3;
    gray[idx] = static_cast<uint8_t>(
        0.114f * bgr[bgr_idx] +
        0.587f * bgr[bgr_idx + 1] +
        0.299f * bgr[bgr_idx + 2]);
}

__global__ void sobel_tiled_kernel(const uint8_t* __restrict__ gray,
                                   uint8_t* __restrict__ edges,
                                   int width, int height) {
    __shared__ uint8_t smem[SMEM_H * SMEM_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_W + tx;
    int y = blockIdx.y * TILE_H + ty;

    int cx = min(max(x, 0), width - 1);
    int cy = min(max(y, 0), height - 1);
    smem[(ty + 1) * SMEM_W + (tx + 1)] = gray[cy * width + cx];

    if (tx == 0)
        smem[(ty + 1) * SMEM_W] = gray[cy * width + max(x - 1, 0)];
    if (tx == TILE_W - 1 || x == width - 1)
        smem[(ty + 1) * SMEM_W + (tx + 2)] = gray[cy * width + min(x + 1, width - 1)];
    if (ty == 0)
        smem[tx + 1] = gray[max(y - 1, 0) * width + cx];
    if (ty == TILE_H - 1 || y == height - 1)
        smem[(ty + 2) * SMEM_W + (tx + 1)] = gray[min(y + 1, height - 1) * width + cx];

    if (tx == 0 && ty == 0)
        smem[0] = gray[max(y - 1, 0) * width + max(x - 1, 0)];
    if ((tx == TILE_W - 1 || x == width - 1) && ty == 0)
        smem[tx + 2] = gray[max(y - 1, 0) * width + min(x + 1, width - 1)];
    if (tx == 0 && (ty == TILE_H - 1 || y == height - 1))
        smem[(ty + 2) * SMEM_W] = gray[min(y + 1, height - 1) * width + max(x - 1, 0)];
    if ((tx == TILE_W - 1 || x == width - 1) && (ty == TILE_H - 1 || y == height - 1))
        smem[(ty + 2) * SMEM_W + (tx + 2)] = gray[min(y + 1, height - 1) * width + min(x + 1, width - 1)];

    __syncthreads();

    if (x >= width || y >= height) return;
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        edges[y * width + x] = 0;
        return;
    }

    int r0 = ty * SMEM_W;
    int r1 = (ty + 1) * SMEM_W;
    int r2 = (ty + 2) * SMEM_W;

    int gx = -smem[r0 + tx]     + smem[r0 + tx + 2]
             -2 * smem[r1 + tx] + 2 * smem[r1 + tx + 2]
             -smem[r2 + tx]     + smem[r2 + tx + 2];

    int gy = -smem[r0 + tx]     - 2 * smem[r0 + tx + 1] - smem[r0 + tx + 2]
             +smem[r2 + tx]     + 2 * smem[r2 + tx + 1] + smem[r2 + tx + 2];

    int mag = min(255, static_cast<int>(sqrtf(static_cast<float>(gx * gx + gy * gy))));
    edges[y * width + x] = static_cast<uint8_t>(mag);
}

float CudaVisionEngine::process_frame(const uint8_t* src_bgr, uint8_t* dst_edges) {
    std::memcpy(h_pinned_bgr_, src_bgr, bgr_bytes_);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(d_bgr_, h_pinned_bgr_, bgr_bytes_, cudaMemcpyHostToDevice);

    dim3 block(TILE_W, TILE_H);
    dim3 grid((width_ + TILE_W - 1) / TILE_W, (height_ + TILE_H - 1) / TILE_H);

    cudaEventRecord(start);
    bgr_to_gray_kernel<<<grid, block>>>(d_bgr_, d_gray_, width_, height_);
    sobel_tiled_kernel<<<grid, block>>>(d_gray_, d_edges_, width_, height_);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    cudaMemcpy(h_pinned_edges_, d_edges_, gray_bytes_, cudaMemcpyDeviceToHost);
    std::memcpy(dst_edges, h_pinned_edges_, gray_bytes_);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return kernel_ms;
}
