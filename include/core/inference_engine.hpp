#pragma once
// ============================================================
// ArchNeuronX v2 - InferenceEngine
// CUDA Streams + Mixed Precision (FP16 Tensor Cores)
// Optimized for low-latency real-time inference
// ============================================================
#include <torch/torch.h>
#include <memory>
#include <chrono>
#include <atomic>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace archneuronx {
namespace core {

struct InferenceResult {
    float buy_probability;
    float sell_probability;
    float hold_probability;
    float confidence;          // max(probabilities)
    std::chrono::microseconds latency_us;
    bool gpu_used;
};

class InferenceEngine {
public:
    explicit InferenceEngine(const std::string& model_path,
                              bool use_fp16 = true,
                              int device_id = 0);
    ~InferenceEngine();

    // Non-copyable, movable
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&) = default;

    // Primary inference method - thread-safe
    [[nodiscard]] InferenceResult predict(const torch::Tensor& input);

    // Batch inference for higher throughput
    [[nodiscard]] std::vector<InferenceResult> predict_batch(
        const std::vector<torch::Tensor>& inputs);

    // Warm up CUDA kernels (call once before first real inference)
    void warmup(int num_iterations = 10);

    // Model info
    [[nodiscard]] std::string device_info() const;
    [[nodiscard]] size_t model_param_count() const;
    [[nodiscard]] double avg_latency_us() const;

    // Hot reload - swap model without downtime
    void reload_model(const std::string& new_model_path);

private:
    torch::jit::script::Module model_;
    torch::Device device_;
    bool use_fp16_;

#ifdef USE_CUDA
    cudaStream_t stream_inference_;
    cudaStream_t stream_preprocess_;
#endif

    // Latency tracking
    std::atomic<double> total_latency_us_{0.0};
    std::atomic<uint64_t> inference_count_{0};

    // Mutex for hot reload
    mutable std::shared_mutex model_mutex_;

    torch::Tensor preprocess_input(const torch::Tensor& raw);
    InferenceResult decode_output(const torch::Tensor& output,
                                   std::chrono::microseconds latency);
};

} // namespace core
} // namespace archneuronx
