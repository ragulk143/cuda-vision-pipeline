import time
import cv2
import numpy as np
import openvino as ov

# 1. Load & Compile MobileNetV3-Large on 12th Gen Intel CPU
core = ov.Core()
ov_model = core.read_model("models/mobilenet_v3_large.onnx")
compiled_model = core.compile_model(ov_model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
infer_request = compiled_model.create_infer_request()

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 2. Open V4L2 Webcam with MJPG lock
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: Camera /dev/video0 not found! Attach via usbipd-win and run: sudo chmod 666 /dev/video*")
    exit(1)

print("Running live camera + Sobel + OpenVINO HUD... Press 'q' or Ctrl+C to stop and print audit.")

sobel_times = []
inf_times = []
e2e_times = []
wall_start = time.perf_counter()

try:
    while True:
        t_e2e_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        # --- A. CPU SIMD Sobel Edge Kernel ---
        t_sobel_start = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        sobel_mag = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
        sobel_bgr = cv2.cvtColor(sobel_mag, cv2.COLOR_GRAY2BGR)
        sobel_ms = (time.perf_counter() - t_sobel_start) * 1000.0
        sobel_fps = 1000.0 / sobel_ms

        # --- B. OpenVINO MobileNetV3-Large Inference ---
        rgb = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
        tensor = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None, ...]

        t_inf_start = time.perf_counter()
        result = infer_request.infer({0: tensor})
        inf_ms = (time.perf_counter() - t_inf_start) * 1000.0
        inf_fps = 1000.0 / inf_ms

        logits = next(iter(result.values())).squeeze()
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        top_id = int(np.argmax(probs))
        conf = float(probs[top_id]) * 100.0

        # --- C. Split-Screen Stitch & Downscale ---
        combined = np.hstack((frame, sobel_bgr))
        display = cv2.pyrDown(combined)

        e2e_ms = (time.perf_counter() - t_e2e_start) * 1000.0
        e2e_fps = 1000.0 / e2e_ms

        # Record metrics (skip first 5 warmup frames)
        if len(e2e_times) >= 5 or time.perf_counter() - wall_start > 0.5:
            sobel_times.append(sobel_ms)
            inf_times.append(inf_ms)
            e2e_times.append(e2e_ms)

        # Telemetry HUD
        cv2.rectangle(display, (10, 10), (540, 155), (0, 0, 0), -1)
        cv2.putText(display, f"CPU SIMD Sobel Kernel:  {sobel_fps:6.1f} FPS ({sobel_ms:.2f} ms)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
        cv2.putText(display, f"OpenVINO MobileNetV3:   {inf_fps:6.1f} FPS ({inf_ms:.2f} ms)", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(display, f"Live Camera Split HUD:  {e2e_fps:6.1f} FPS (Sensor-Bound)", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        cv2.putText(display, f"Top-1 Class #{top_id} ({conf:.1f}%) | Left: Raw | Right: Sobel", (20, 142),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("Intel OpenVINO + CPU Sobel Split-Screen HUD", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user (Ctrl+C).")

finally:
    cap.release()
    cv2.destroyAllWindows()

    if len(e2e_times) > 0:
        s_arr = np.array(sobel_times)
        i_arr = np.array(inf_times)
        e_arr = np.array(e2e_times)

        print("\n" + "=" * 62)
        print(f"  INTEL OPENVINO + CPU SOBEL LIVE AUDIT ({len(e_arr)} frames)")
        print("=" * 62)
        print(f"  CPU SIMD Sobel Kernel : {1000.0/np.mean(s_arr):7.2f} Avg FPS | p50: {np.percentile(s_arr, 50):5.2f} ms | p95: {np.percentile(s_arr, 95):5.2f} ms")
        print(f"  OpenVINO MobileNetV3 : {1000.0/np.mean(i_arr):7.2f} Avg FPS | p50: {np.percentile(i_arr, 50):5.2f} ms | p95: {np.percentile(i_arr, 95):5.2f} ms")
        print(f"  Live Camera Split HUD : {1000.0/np.mean(e_arr):7.2f} Avg FPS | p50: {np.percentile(e_arr, 50):5.2f} ms | p95: {np.percentile(e_arr, 95):5.2f} ms")
        print("=" * 62)
