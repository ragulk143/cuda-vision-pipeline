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

# Hardware Benchmark & Profiling Logs — NVIDIA GeForce RTX 3050 (`SM_86`)

- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (6144 MiB GDDR6, Ampere `SM_86`)
- **Environment:** WSL2 Ubuntu 22.04 | CUDA Toolkit `12.1` | `g++ 11.4.0 -O3` | OpenCV `4.5.4`
- **Workload:** 200 frames @ `1920×1080` (`CV_8UC3` BGR → `CV_8UC1` Grayscale → `18×18` Shared-Memory Tiled Sobel)

---

## Summary Comparison (v1 → v3)

| Metric | v1 (Naive Baseline) | v2 (RAII + Shared Mem) | v3 (3-Slot Async Streams + Zero-Copy) |
| :--- | :---: | :---: | :---: |
| **Frames Processed** | 80 / 200 | 200 / 200 | **200 / 200** |
| **Dropped Frames** | 120 (60.0%) | 0 (0.0%) | **0 (0.0%)** |
| **GPU Kernel p50 (`cudaEvent`)** | 0.217 ms | 0.476 ms | **0.223 ms** |
| **E2E Latency p50** | — | ~21.50 ms | **2.093 ms** |
| **E2E Latency p95** | — | — | **2.340 ms** |
| **E2E Throughput** | 22.01 FPS | 46.41 FPS | **1,046.18 FPS** |

---

## v3 Nsight Systems (`nsys 2025.3.2`) CUDA API Trace

```text
Time (%)  Total Time (ns)  Num Calls   Avg (ns)   Med (ns)   Min (ns)  Max (ns)   StdDev (ns)            Name
--------  ---------------  ---------  ----------  ---------  --------  ---------  -----------  -------------------------
    50.5        185786762        204    910719.4   920487.5      4534    1040423     127217.5  cudaStreamSynchronize
    37.8        138877132          3  46292377.3     8818.0      8079  138860235   80166116.3  cudaStreamCreateWithFlags
     4.8         17478055          6   2913009.2  2882464.5   1644803    4268008    1264156.1  cudaHostAlloc
     1.7          6155583        402     15312.4     9123.0      6072     880526      48638.3  cudaMemcpyAsync
     1.6          5806849        402     14444.9     9416.5      3796    1531394      76578.0  cudaLaunchKernel
     1.4          5092375          6    848729.2   804562.5    649949    1128392     203762.7  cudaFreeHost
     1.0          3637345          9    404149.4   350966.0    310643     797775     152770.3  cudaMalloc
     0.7          2652700          9    294744.4   291787.0    215774     410968      67884.3  cudaFree
     0.6          2109875        402      5248.4     3513.5      1964     339090      18128.4  cudaEventRecord
```
