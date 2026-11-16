#pragma once

#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

struct InferenceResult {
    int class_id = -1;
    float confidence = 0.0f;
    float infer_ms = 0.0f;
};

class OnnxGpuEngine {
public:
    OnnxGpuEngine(const std::string& model_path, int batch_size = 1)
        : env_(ORT_LOGGING_LEVEL_WARNING, "cuda_vision_v5"),
          batch_size_(batch_size),
          cuda_mem_info_("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault),
          cpu_mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        opts.AppendExecutionProvider_CUDA(cuda_opts);

        session_ = Ort::Session(env_, model_path.c_str(), opts);
        Ort::AllocatorWithDefaultOptions alloc;
        in_name_ = session_.GetInputNameAllocated(0, alloc).get();
        out_name_ = session_.GetOutputNameAllocated(0, alloc).get();
        auto out_shape = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        num_classes_ = (out_shape.size() >= 2 && out_shape[1] > 0) ? static_cast<int>(out_shape[1]) : 1000;
        logits_.resize(static_cast<size_t>(batch_size_) * num_classes_, 0.0f);
    }
    InferenceResult infer_from_gpu_ptr(float* d_input_nchw) {
        std::array<int64_t, 4> in_shape{batch_size_, 3, 224, 224};
        std::array<int64_t, 2> out_shape{batch_size_, num_classes_};
        size_t in_elems = static_cast<size_t>(batch_size_) * 3 * 224 * 224;

        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            cuda_mem_info_, d_input_nchw, in_elems, in_shape.data(), in_shape.size());
        Ort::Value out_tensor = Ort::Value::CreateTensor<float>(
            cpu_mem_info_, logits_.data(), logits_.size(), out_shape.data(), out_shape.size());

        Ort::IoBinding binding(session_);
        binding.BindInput(in_name_.c_str(), in_tensor);
        binding.BindOutput(out_name_.c_str(), out_tensor);
        session_.Run(Ort::RunOptions{nullptr}, binding);

        // Softmax Top-1 over batch item 0
        float max_logit = *std::max_element(logits_.begin(), logits_.begin() + num_classes_);
        double sum_exp = 0.0;
        int best_id = 0;
        float best_val = logits_[0];
        for (int i = 0; i < num_classes_; ++i) {
            sum_exp += std::exp(static_cast<double>(logits_[i] - max_logit));
            if (logits_[i] > best_val) { best_val = logits_[i]; best_id = i; }
        }
        float conf = static_cast<float>(std::exp(static_cast<double>(best_val - max_logit)) / sum_exp);
        return InferenceResult{best_id, conf, 0.0f};
    }

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    int batch_size_;
    int num_classes_ = 1000;
    Ort::MemoryInfo cuda_mem_info_;
    Ort::MemoryInfo cpu_mem_info_;
    std::string in_name_;
    std::string out_name_;
    std::vector<float> logits_;
};
