
# `cuda-vision-pipeline` — Real-Time Multithreaded C++17 & CUDA Vision Engine

A low-latency, multithreaded computer vision and GPU inference pipeline engineered from scratch in **Modern C++17**, **NVIDIA CUDA C++ (`SM_86`)**, and **OpenCV 4**, built and benchmarked on an **NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)**.

Designed around real-time robotics and edge visual perception constraints: zero frame drops, deterministic RAII memory management, shared-memory GPU tiling, and pipelined CPU–GPU execution.

---

## 1. System Architecture

```text
+-----------------------+      Zero-Copy Move      +----------------------------------------+

|  Thread 1: Producer   | -----------------------> |  Thread 2: GPU Vision Worker (RTX 3050)|
|  1080p Frame Capture  |   RingBuffer<FramePacket>|  - 3-Slot Pinned Host DMA (Zero-Copy)  |
|  (Zero-Copy cv::Mat   |   (std::mutex + CV,      |  - Non-Blocking cudaStream_t Queues    |
|   Pinned Wrapper)     |    Drops Oldest on Lag)  |  - BGR->Gray + 18x18 Shared-Mem Sobel  |
+-----------------------+                          +----------------------------------------+
                                                                       |
                                                                Zero-Copy Move
                                                                       v
                                                   +----------------------------------------+

                                                   |  Thread 3: Consumer & Telemetry        |
                                                   |  - Per-Slot Stream Sync & CUDA Events  |
                                                   |  - E2E Latency (p50/p95 ms) & FPS Audit|
                                                   +----------------------------------------+
```

---

## 2. Milestones Achieved (Completed)

- [x] **Hardware & Toolchain Verification:** Configured native `nvcc 12.1`, `g++ 11.4` (`-O3`), `CMake 3.22`, and `OpenCV 4` C++ inside WSL2 Ubuntu 22.04 targeting Ampere `SM_86`.
- [x] **Thread-Safe Zero-Copy `RingBuffer` (`include/ring_buffer.hpp`):** Decoupled capture, GPU processing, and telemetry threads using `std::mutex`, `std::condition_variable`, and `std::move` semantics (preventing deep pixel copies and bounding queue latency by dropping stale frames under load).
- [x] **Custom CUDA C++ Vision Kernels (`src/vision_kernels.cu`):**
  - `bgr_to_gray_kernel`: Coalesced global memory BGR-to-luminance conversion.
  - `sobel_tiled_kernel`: 2D spatial gradient edge detector utilizing an **`18×18` `__shared__` SRAM halo tile** to eliminate redundant global memory reads across neighboring threads.
- [x] **RAII Resource Management (`CudaVisionEngine`):** Encapsulated page-locked host buffers (`cudaMallocHost`) and persistent device VRAM (`cudaMalloc`) inside a non-copyable C++ RAII class (PIMPL-isolated from host headers), eliminating per-frame driver synchronization stalls.
- [x] **v3 — Asynchronous CUDA Streams & Zero-Copy Host Buffers:**
  - Replaced synchronous `cudaMemcpy` with `cudaMemcpyAsync` across **3 `cudaStreamNonBlocking` queues** (`NUM_SLOTS = 3`) with `SlotPool` synchronization to interleave CPU frame preparation, PCIe H2D/D2H DMA, and GPU kernel execution.
  - Wrapped `h_pinned_bgr_` directly inside `cv::Mat` headers to eliminate intermediate host `std::memcpy` calls, boosting E2E throughput to **1,046.18 FPS** (**22.5×** over v2).

---

## 3. Optimization Strategies & Measured RTX 3050 Benchmarks

**Test Configuration:** `1920×1080` (1080p) BGR frames (`200 frames`) | NVIDIA RTX 3050 (`6144 MiB VRAM, SM_86`) | CUDA `12.1` | `g++ 11.4 -O3`

| Pipeline Version | Memory & Kernel Strategy | Frames Processed | Dropped Frames | GPU Kernel p50 | E2E Latency p50 / p95 | E2E Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **v1 (Naive Baseline)** | Per-frame `cudaMalloc` / `cudaFree` + Global Memory Sobel | 80 / 200 | 120 (60%) | 0.217 ms | — | 22.01 FPS |
| **v2 (RAII + Shared Mem)** | Pre-allocated Pinned Host (`cudaMallocHost`) + Persistent VRAM + `18×18` Shared-Memory Halo Tile | 200 / 200 | 0 (0%) | 0.476 ms | ~21.5 ms / — | 46.41 FPS |
| **v3 (Async Streams + Zero-Copy)** | **3-Slot `cudaStreamNonBlocking` + `cudaMemcpyAsync` + Zero-Copy `cv::Mat` Pinned Host** | **200 / 200** | **0 (0%)** | **0.223 ms** | **2.093 ms / 2.340 ms** | **1,046.18 FPS** |

### Nsight Systems (`nsys 2025.3.2`) API & DMA Audit (v3)
Profiled across 200 1080p frames (plus 1 warmup frame):
- **`cudaLaunchKernel` (`402` calls = `201 × 2` kernels):** `9.4 µs` median dispatch overhead (`5.81 ms` total CPU launch time).
- **`cudaMemcpyAsync` (`402` calls = `201` H2D + `201` D2H):** `9.1 µs` median enqueue overhead (`6.16 ms` total CPU API time for `1.66 GB` PCIe traffic).
- **`cudaStreamSynchronize` (`204` calls):** `0.920 ms` median per frame (`185.79 ms` total wall time → **~1,046–1,061 FPS** steady-state throughput).

### Key Engineering Takeaways
1. **Eliminating Per-Frame Allocation Overhead (v1 → v2):** Moving `cudaMalloc`/`cudaFree` out of the worker loop into the `CudaVisionEngine` RAII constructor/destructor removed implicit GPU pipeline synchronization barriers, cutting dropped frames from **60% to 0%** and increasing throughput by **2.11×** (`22.01 -> 46.41 FPS`).
2. **Overlapping DMA & Eliminating Host Copies (v2 → v3):** Wrapping `cudaMallocHost` buffers directly in `cv::Mat` headers and pipelining 3 non-blocking CUDA streams cut E2E p50 latency from **~21.5 ms to 2.093 ms** and boosted throughput by **22.5×** (`46.41 -> 1,046.18 FPS`).

---

## 4. Remaining Roadmap (Next Milestones)

- [ ] **v4 — Live Camera / Video Stream & Visual Output (`cv::VideoCapture` + `cv::imshow`):**
  - Support `.mp4` video file ingestion and live webcam passthrough in WSL2 via `usbipd-win` (`/dev/video0`) with zero-copy `cap.read(pinned_mat)` and real-time side-by-side raw vs. GPU Sobel edge visualization + live HUD.
- [ ] **v5 — Deep Learning Inference Integration (ONNX Runtime GPU `< 80 MB`):**
  - Integrate **ONNX Runtime (`CUDAExecutionProvider`)** into Thread 2 with a custom CUDA `bgr_to_nchw_norm_kernel` for lightweight models under 80 MB (e.g., **MobileNetV3-Large ~21 MB** / **DeepLabV3-MobileNetV3 ~42 MB**), benchmarking **Batch 1 / 8 / 32** latency and VRAM usage.
- [ ] **v6 — Native TensorRT C++ Engine (`nvinfer1` FP16):**
  - Compile the `.onnx` model into a serialized TensorRT `.engine` plan targeting RTX 3050 `SM_86` Tensor Cores (`FP16`) and execute `enqueueV3` directly on the existing `cudaStream_t` queues.

---

## 5. Build & Run Instructions

### Prerequisites
- Linux / WSL2 Ubuntu 22.04+
- NVIDIA CUDA Toolkit (`12.1+`)
- OpenCV 4 (`libopencv-dev`)
- CMake (`3.18+`) & C++17 compiler (`g++ 11+`)

### Compile & Execute
```bash
git clone https://github.com
cd cuda-vision-pipeline
mkdir -p build && cd build
cmake ..
make -j$(nproc)
./vision_pipeline
```

### Enabling Live Laptop Webcam in WSL2 (Optional)
Because Windows owns integrated USB webcams by default, attach the camera to WSL2 from an **Administrator Windows PowerShell**:
```powershell
usbipd list
usbipd attach --wsl --busid <CAMERA-BUS-ID>
```
Verify `/dev/video0` appears inside Ubuntu (`ls /dev/video*`) before running live camera capture.

---

## 6. Repository Structure

```text
cuda-vision-pipeline/
├── CMakeLists.txt            # C++17 + CUDA (SM_86) build configuration
├── include/
│   ├── ring_buffer.hpp       # Thread-safe zero-copy circular queue
│   └── vision_kernels.cuh    # Pure C++ PIMPL interface for CudaVisionEngine
├── src/
│   ├── main.cpp              # 3-thread producer/worker/telemetry pipeline
│   └── vision_kernels.cu     # Async CUDA streams, Grayscale & Shared-Memory Sobel kernels
├── scripts/
│   └── export_onnx.py        # PyTorch ONNX export utility
├── models/
│   └── resnet50.onnx         # Pre-exported ONNX model (ignored in git if >50MB)
└── bench/
    └── results.md            # Detailed hardware & Nsight Systems benchmark logs
```
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
