import time
import numpy as np
import openvino as ov

core = ov.Core()
print("CPU Device:", core.get_property("CPU", "FULL_DEVICE_NAME"))

models = [
    ("MobileNetV3-Large", "models/mobilenet_v3_large.onnx"),
    ("ResNet-50", "models/resnet50.onnx"),
]

dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

for name, path in models:
    ov_model = core.read_model(path)
    # Compile with throughput/latency hints for 12th Gen Intel CPU
    compiled_model = core.compile_model(ov_model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
    infer_request = compiled_model.create_infer_request()

    # Warmup
    for _ in range(20):
        infer_request.infer({0: dummy_input})

    # Benchmark 300 iterations
    runs = 300
    start = time.perf_counter()
    for _ in range(runs):
        infer_request.infer({0: dummy_input})
    elapsed = time.perf_counter() - start

    fps = runs / elapsed
    latency_ms = (elapsed / runs) * 1000
    print(f"{name:18s} | {fps:7.2f} FPS | {latency_ms:6.2f} ms/frame")
