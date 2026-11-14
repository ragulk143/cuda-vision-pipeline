# RTX 3050 (6GB VRAM, SM_86) Benchmark Log — `cuda-vision-pipeline`

## Hardware & Environment
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (6144 MiB VRAM, Ampere SM_86)
- **OS / Toolchain:** WSL2 Ubuntu 22.04 | CUDA 12.1 (`nvcc 12.1.105`) | `g++ 11.4.0` (`-O3`) | OpenCV 4
- **Workload:** 1920x1080 (1080p) BGR -> Grayscale + 3x3 Sobel Edge Detection (200 frames)

## Measured Results (v1 vs v2)

| Pipeline Version | Memory Strategy | Frames Processed | Dropped Frames | CPU OpenCV Avg | GPU Kernel p50 | Kernel Speedup | E2E Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **v1 (Naive Baseline)** | Per-frame `cudaMalloc` / `cudaFree` | 80 / 200 | 120 | 8.02 ms | 0.217 ms | 37.0x | 22.01 FPS |
| **v2 (RAII + Shared Mem)** | Pre-allocated Pinned Host + Persistent VRAM + 18x18 Shared-Mem Tile | **200 / 200** | **0** | 7.37 ms | **0.476 ms** | **15.5x** | **46.41 FPS** |

## Engineering Observations
1. **Eliminating Allocation Overhead:** Moving `cudaMalloc` and `cudaFree` out of the per-frame worker loop into an RAII constructor/destructor eliminated GPU synchronization stalls, dropping frame loss from **60% (120/200)** to **0% (0/200)** and more than doubling E2E throughput (**22.01 -> 46.41 FPS**).
2. **Next Bottleneck (v3 Target):** Profiling shows GPU compute takes `<0.5 ms`, while host-side `cv::Mat` allocation + `std::memcpy` + synchronous `cudaMemcpy` accounts for the remaining ~21 ms. v3 will use **asynchronous `cudaMemcpyAsync` with CUDA Streams** and direct `cv::Mat` pointer wrapping over pinned memory.
