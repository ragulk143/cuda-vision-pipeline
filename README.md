# `cuda-vision-pipeline` — Real-Time Multithreaded C++17 & CUDA Vision Engine

A low-latency, multithreaded computer vision and GPU inference pipeline engineered from scratch in **Modern C++17**, **NVIDIA CUDA C++ (`SM_86`)**, and **OpenCV 4**, built and benchmarked on an **NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)**.

Designed around real-time robotics and edge visual perception constraints: zero frame drops, deterministic RAII memory management, shared-memory GPU tiling, and pipelined CPU–GPU execution.

---

## 1. System Architecture

```text
+-----------------------+      Zero-Copy Move      +----------------------------------------+

|  Thread 1: Producer   | -----------------------> |  Thread 2: GPU Vision Worker (RTX 3050)|
|  1080p Frame Capture  |   RingBuffer<FramePacket>|  - Pinned Host DMA (cudaMallocHost)    |
|  (Camera / Video /    |   (std::mutex + CV,      |  - Custom CUDA BGR -> Grayscale Kernel |
|   Synthetic Stream)   |    Drops Oldest on Lag)  |  - 18x18 Shared-Memory Tiled Sobel     |
+-----------------------+                          +----------------------------------------+
                                                                       |
                                                                Zero-Copy Move
                                                                       v
                                                   +----------------------------------------+

                                                   |  Thread 3: Consumer & Telemetry        |
                                                   |  - Live E2E Latency (p50/p95 ms)       |
                                                   |  - Throughput (FPS) & Frame Drop Audit |
                                                   +----------------------------------------+
```

---

## 2. Milestones Achieved (Completed)

- [x] **Hardware & Toolchain Verification:** Configured native `nvcc 12.1`, `g++ 11.4` (`-O3`), `CMake 3.22`, and `OpenCV 4` C++ inside WSL2 Ubuntu 22.04 targeting Ampere `SM_86`.
- [x] **Thread-Safe Zero-Copy `RingBuffer` (`include/ring_buffer.hpp`):** Decoupled capture, GPU processing, and telemetry threads using `std::mutex`, `std::condition_variable`, and `std::move` semantics (preventing deep pixel copies and bounding queue latency by dropping stale frames under load).
- [x] **Custom CUDA C++ Vision Kernels (`src/vision_kernels.cu`):**
  - `bgr_to_gray_kernel`: Coalesced global memory BGR-to-luminance conversion.
  - `sobel_tiled_kernel`: 2D spatial gradient edge detector utilizing an **`18×18` `__shared__` SRAM halo tile** to eliminate redundant global memory reads across neighboring threads.
- [x] **RAII Resource Management (`CudaVisionEngine`):** Encapsulated page-locked host buffers (`cudaMallocHost`) and persistent device VRAM (`cudaMalloc`) inside a non-copyable C++ RAII class, eliminating per-frame driver synchronization stalls.

---

## 3. Optimization Strategies & Measured RTX 3050 Benchmarks

**Test Configuration:** `1920×1080` (1080p) BGR frames (`200 frames`) | NVIDIA RTX 3050 (`6144 MiB VRAM, SM_86`) | CUDA `12.1` | `g++ 11.4 -O3`

| Pipeline Version | Memory & Kernel Strategy | Frames Processed | Dropped Frames | CPU OpenCV Avg | GPU Kernel p50 | Kernel Speedup vs. CPU | E2E Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **v1 (Naive Baseline)** | Per-frame `cudaMalloc` / `cudaFree` + Global Memory Sobel | 80 / 200 | 120 (60%) | 8.02 ms | 0.217 ms | 37.0x | 22.01 FPS |
| **v2 (RAII + Shared Mem)** | Pre-allocated Pinned Host (`cudaMallocHost`) + Persistent VRAM + `18×18` Shared-Memory Halo Tile | **200 / 200** | **0 (0%)** | 7.37 ms | **0.476 ms** | **15.5x** | **46.41 FPS** |

### Key Engineering Takeaways
1. **Eliminating Per-Frame Allocation Overhead:** Moving `cudaMalloc`/`cudaFree` out of the worker loop into the `CudaVisionEngine` RAII constructor/destructor removed implicit GPU pipeline synchronization barriers, cutting dropped frames from **60% to 0%** and increasing end-to-end throughput by **2.11×** (`22.01 -> 46.41 FPS`).
2. **Bottleneck Analysis for v3:** While GPU kernel execution completes in `<0.5 ms`, synchronous `cudaMemcpy` calls and host-side `cv::Mat` buffer copies account for ~21 ms per frame. Overlapping host-to-device DMA with kernel execution via **CUDA Streams** is the primary target for v3.

---

## 4. Remaining Roadmap (Next Milestones)

- [ ] **v3 — Asynchronous CUDA Streams & Zero-Copy Host Buffers:**
  - Replace synchronous `cudaMemcpy` with `cudaMemcpyAsync` across multiple `cudaStream_t` queues to interleave CPU frame preparation with GPU kernel execution (Target: **150+ FPS E2E**).
  - Wrap `h_pinned_bgr_` directly inside `cv::Mat` headers to eliminate intermediate `std::memcpy` calls.
- [ ] **v4 — Live Camera / Video Stream & Visual Output (`cv::VideoCapture` + `cv::imshow`):**
  - Support `.mp4` video file ingestion and live webcam passthrough in WSL2 via `usbipd-win` (`/dev/video0`) with real-time side-by-side raw vs. GPU Sobel edge visualization.
- [ ] **v5 — Deep Learning Inference Integration (`ResNet-50`):**
  - Integrate **LibTorch C++ (`CUDA 12.1`)** or **ONNX Runtime (`v1.19.2`)** into Thread 2 to run real-time `ResNet-50` classification alongside custom CUDA preprocessing kernels, benchmarking **Batch 1 / 8 / 32** latency and VRAM usage.

---

## 5. Build & Run Instructions

### Prerequisites
- Linux / WSL2 Ubuntu 22.04+
- NVIDIA CUDA Toolkit (`12.1+`)
- OpenCV 4 (`libopencv-dev`)
- CMake (`3.18+`) & C++17 compiler (`g++ 11+`)

### Compile & Execute
```bash
git clone https://github.com<your-username>/cuda-vision-pipeline.git
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
│   └── vision_kernels.cuh    # RAII CudaVisionEngine interface
├── src/
│   ├── main.cpp              # 3-thread producer/worker/telemetry pipeline
│   └── vision_kernels.cu     # Custom CUDA Grayscale & Shared-Memory Sobel kernels
├── scripts/
│   └── export_onnx.py        # PyTorch ResNet-50 ONNX export utility
├── models/
│   └── resnet50.onnx         # Pre-exported ResNet-50 model (ignored in git if >50MB)
└── bench/
    └── results.md            # Detailed hardware benchmark logs
```
