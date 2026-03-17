#pragma once
// ============================================================
// ArchNeuronX v3 - GPU Optimization System
// Mixed Precision (AMP), TensorRT, Memory Management
// 2-5x inference speed improvement for regime-aware ensemble
// ============================================================

#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <vector>
#include <memory>
#include <chrono>
#include <mutex>

namespace archneuronx {
namespace gpu {

/**
 * @brief GPU optimization configuration
 */
struct GPUOptimizerConfig {
    // Mixed precision settings
    bool enable_amp = true;                    // Automatic Mixed Precision
    torch::ScalarType amp_dtype = torch::kHalf; // FP16 for inference
    bool keep_batchnorm_fp32 = true;          // Keep BatchNorm in FP32
    
    // Memory management
    size_t memory_pool_size = 1024 * 1024 * 1024; // 1GB pool
    bool enable_memory_pool = true;
    bool enable_caching_allocator = true;
    double memory_threshold = 0.8;             // Alert at 80% usage
    
    // TensorRT settings
    bool enable_tensorrt = false;             // TensorRT for production
    int tensorrt_max_batch_size = 32;
    int tensorrt_max_workspace_size = 1 << 30; // 1GB
    bool tensorrt_fp16 = true;                 // FP16 in TensorRT
    
    // Performance optimization
    bool enable_streaming = true;             // CUDA streams for parallelism
    int num_streams = 4;                      // Number of CUDA streams
    bool enable_graph_capture = true;          // CUDA graph capture
    int warmup_iterations = 3;                 // Warmup iterations
    
    // Monitoring
    bool enable_profiling = false;             // CUDA profiling
    int profiling_interval_ms = 1000;          // Profiling interval
    bool enable_memory_monitoring = true;      // Memory usage tracking
};

/**
 * @brief GPU memory statistics
 */
struct GPUMemoryStats {
    size_t total_memory;           // Total GPU memory
    size_t allocated_memory;       // Currently allocated
    size_t cached_memory;          // Cached in allocator
    size_t max_allocated_memory;   // Peak allocation
    double utilization;            // Memory utilization (0-1)
    std::chrono::system_clock::time_point timestamp;
};

/**
 * @brief Performance metrics for GPU operations
 */
struct GPUPerformanceMetrics {
    double inference_time_ms;     // Average inference time
    double throughput;             // Predictions per second
    double gpu_utilization;        // GPU utilization percentage
    double power_usage_watts;      // Power consumption
    double temperature_celsius;    // GPU temperature
    int cuda_kernel_launches;      // Number of kernel launches
    std::chrono::system_clock::time_point last_update;
};

/**
 * @brief Advanced GPU optimizer for neural trading
 * 
 * Provides mixed precision inference, memory management, and performance
 * optimization specifically designed for high-frequency trading scenarios.
 */
class GPUOptimizer {
public:
    explicit GPUOptimizer(const GPUOptimizerConfig& config = GPUOptimizerConfig{});
    ~GPUOptimizer();

    // Initialize GPU resources and optimization
    bool initialize();
    void shutdown();

    // Mixed precision inference optimization
    torch::Tensor optimize_tensor(const torch::Tensor& input);
    torch::Tensor amp_inference(torch::jit::script::Module& model,
                                const std::vector<torch::Tensor>& inputs);
    
    // Batch processing with streams
    std::vector<torch::Tensor> batch_inference(
        torch::jit::script::Module& model,
        const std::vector<std::vector<torch::Tensor>>& batch_inputs,
        bool use_amp = true
    );

    // Memory management
    void enable_memory_pooling();
    void disable_memory_pooling();
    void clear_gpu_cache();
    GPUMemoryStats get_memory_stats() const;
    bool is_memory_available(size_t required_bytes) const;
    
    // Performance monitoring
    GPUPerformanceMetrics get_performance_metrics() const;
    void start_profiling();
    void stop_profiling();
    void reset_metrics();

    // TensorRT integration
    bool enable_tensorrt();
    torch::jit::script::Module optimize_model_tensorrt(torch::jit::script::Module& model);
    bool is_tensorrt_available() const;

    // CUDA stream management
    cudaStream_t get_stream(int stream_id = 0);
    void synchronize_stream(int stream_id = 0);
    void synchronize_all_streams();

    // Graph capture for repeated operations
    bool capture_inference_graph(torch::jit::script::Module& model,
                                 const std::vector<torch::Tensor>& example_inputs);
    torch::Tensor run_captured_graph(const std::vector<torch::Tensor>& inputs);

    // Device management
    int get_device_count() const;
    int get_current_device() const;
    bool set_device(int device_id);
    std::string get_device_info(int device_id = -1) const;

    // Warmup and benchmarking
    void warmup_models(std::vector<torch::jit::script::Module>& models,
                      const std::vector<torch::Tensor>& example_inputs);
    double benchmark_model(torch::jit::script::Module& model,
                           const std::vector<torch::Tensor>& inputs,
                           int iterations = 100);

    // Error handling and recovery
    bool check_gpu_errors();
    void reset_gpu_state();
    std::string get_last_error() const;

private:
    GPUOptimizerConfig config_;
    
    // CUDA resources
    std::vector<cudaStream_t> cuda_streams_;
    cudaGraph_t inference_graph_;
    cudaGraphExec_t graph_exec_;
    bool graph_captured_;
    
    // Monitoring
    std::unique_ptr<GPUMemoryStats> memory_stats_;
    std::unique_ptr<GPUPerformanceMetrics> performance_metrics_;
    std::chrono::high_resolution_clock::time_point profiling_start_;
    bool profiling_active_;
    
    // TensorRT
    bool tensorrt_available_;
    void* tensorrt_engine_;  // Simplified TRT handle
    
    // Memory management
    void* memory_pool_;
    bool memory_pool_enabled_;
    
    // Thread safety
    mutable std::mutex optimizer_mutex_;
    
    // Internal methods
    bool initialize_cuda();
    bool initialize_memory_pool();
    bool initialize_tensorrt();
    void update_memory_stats();
    void update_performance_metrics();
    
    // CUDA utility methods
    bool check_cuda_error(cudaError_t error, const std::string& operation);
    std::string cuda_error_string(cudaError_t error) const;
    
    // Mixed precision utilities
    torch::Tensor convert_to_amp_dtype(const torch::Tensor& tensor);
    torch::Tensor convert_from_amp_dtype(const torch::Tensor& tensor);
    
    // Benchmarking utilities
    std::vector<double> measure_inference_times(
        torch::jit::script::Module& model,
        const std::vector<torch::Tensor>& inputs,
        int iterations
    );
    
    // Memory utilities
    size_t get_tensor_memory_size(const torch::Tensor& tensor) const;
    void optimize_memory_layout(torch::Tensor& tensor);
};

/**
 * @brief RAII GPU context manager
 */
class GPUContext {
public:
    explicit GPUContext(int device_id = 0);
    ~GPUContext();
    
    bool is_valid() const;
    int get_device_id() const;

private:
    int device_id_;
    bool context_valid_;
};

/**
 * @brief Automatic mixed precision context
 */
class AMPScope {
public:
    explicit AMPScope(GPUOptimizer& optimizer);
    ~AMPScope();
    
    bool is_enabled() const;

private:
    GPUOptimizer& optimizer_;
    bool amp_enabled_;
    torch::ScalarType original_dtype_;
};

} // namespace gpu
} // namespace archneuronx
