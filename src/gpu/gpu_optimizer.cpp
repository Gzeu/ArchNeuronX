/**
 * @file gpu_optimizer.cpp
 * @brief GPU optimization implementation for mixed precision and performance
 * @author ArchNeuronX Team
 * @date 2026-03-17
 */

#include "gpu/gpu_optimizer.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace archneuronx {
namespace gpu {

GPUOptimizer::GPUOptimizer(const GPUOptimizerConfig& config)
    : config_(config), graph_captured_(false), tensorrt_available_(false),
      memory_pool_(nullptr), memory_pool_enabled_(false), profiling_active_(false) {
    
    memory_stats_ = std::make_unique<GPUMemoryStats>();
    performance_metrics_ = std::make_unique<GPUPerformanceMetrics>();
    
    // Initialize with zeros
    std::memset(memory_stats_.get(), 0, sizeof(GPUMemoryStats));
    std::memset(performance_metrics_.get(), 0, sizeof(GPUPerformanceMetrics));
}

GPUOptimizer::~GPUOptimizer() {
    shutdown();
}

bool GPUOptimizer::initialize() {
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    
    try {
        // Initialize CUDA
        if (!initialize_cuda()) {
            std::cerr << "Failed to initialize CUDA" << std::endl;
            return false;
        }
        
        // Initialize CUDA streams
        if (config_.enable_streaming) {
            cuda_streams_.resize(config_.num_streams);
            for (int i = 0; i < config_.num_streams; ++i) {
                auto error = cudaStreamCreate(&cuda_streams_[i]);
                if (error != cudaSuccess) {
                    std::cerr << "Failed to create CUDA stream " << i << ": " 
                              << cuda_error_string(error) << std::endl;
                    return false;
                }
            }
        }
        
        // Initialize memory pool
        if (config_.enable_memory_pool) {
            enable_memory_pooling();
        }
        
        // Initialize TensorRT if enabled
        if (config_.enable_tensorrt) {
            initialize_tensorrt();
        }
        
        // Reset metrics
        reset_metrics();
        
        std::cout << "GPU Optimizer initialized successfully" << std::endl;
        std::cout << "Device count: " << get_device_count() << std::endl;
        std::cout << "AMP enabled: " << (config_.enable_amp ? "Yes" : "No") << std::endl;
        std::cout << "TensorRT enabled: " << (tensorrt_available_ ? "Yes" : "No") << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing GPU Optimizer: " << e.what() << std::endl;
        return false;
    }
}

void GPUOptimizer::shutdown() {
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    
    // Synchronize all streams
    if (!cuda_streams_.empty()) {
        synchronize_all_streams();
        
        // Destroy streams
        for (auto stream : cuda_streams_) {
            cudaStreamDestroy(stream);
        }
        cuda_streams_.clear();
    }
    
    // Destroy CUDA graph
    if (graph_captured_) {
        cudaGraphExecDestroy(graph_exec_);
        cudaGraphDestroy(inference_graph_);
        graph_captured_ = false;
    }
    
    // Clear memory pool
    if (memory_pool_) {
        disable_memory_pooling();
    }
    
    // Reset CUDA device
    cudaDeviceReset();
    
    std::cout << "GPU Optimizer shutdown complete" << std::endl;
}

torch::Tensor GPUOptimizer::optimize_tensor(const torch::Tensor& input) {
    if (!config_.enable_amp) {
        return input;
    }
    
    return convert_to_amp_dtype(input);
}

torch::Tensor GPUOptimizer::amp_inference(torch::jit::script::Module& model,
                                          const std::vector<torch::Tensor>& inputs) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Convert inputs to AMP dtype
    std::vector<torch::Tensor> amp_inputs;
    for (const auto& input : inputs) {
        amp_inputs.push_back(convert_to_amp_dtype(input));
    }
    
    // Create input vector for model
    std::vector<torch::jit::IValue> ivalue_inputs;
    for (const auto& input : amp_inputs) {
        ivalue_inputs.push_back(input);
    }
    
    // Run inference
    auto output = model.forward(ivalue_inputs);
    auto result_tensor = output.toTensor();
    
    // Convert back to original dtype if needed
    if (result_tensor.dtype() != torch::kFloat32) {
        result_tensor = result_tensor.to(torch::kFloat32);
    }
    
    // Update performance metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    performance_metrics_->inference_time_ms = duration.count() / 1000.0;
    performance_metrics_->cuda_kernel_launches++;
    performance_metrics_->last_update = std::chrono::system_clock::now();
    
    return result_tensor;
}

std::vector<torch::Tensor> GPUOptimizer::batch_inference(
    torch::jit::script::Module& model,
    const std::vector<std::vector<torch::Tensor>>& batch_inputs,
    bool use_amp) {
    
    std::vector<torch::Tensor> results;
    results.reserve(batch_inputs.size());
    
    // Use CUDA streams for parallel processing
    if (config_.enable_streaming && cuda_streams_.size() > 1) {
        for (size_t i = 0; i < batch_inputs.size(); ++i) {
            int stream_id = i % cuda_streams_.size();
            cudaStream_t stream = cuda_streams_[stream_id];
            
            // Set stream for current operations
            torch::cuda::setCurrentCUDAStream(stream);
            
            // Run inference
            auto result = use_amp ? amp_inference(model, batch_inputs[i]) 
                                  : model.forward(batch_inputs[i]).toTensor();
            results.push_back(result);
        }
        
        // Synchronize all streams
        synchronize_all_streams();
    } else {
        // Sequential processing
        for (const auto& inputs : batch_inputs) {
            auto result = use_amp ? amp_inference(model, inputs) 
                                  : model.forward(inputs).toTensor();
            results.push_back(result);
        }
    }
    
    return results;
}

void GPUOptimizer::enable_memory_pooling() {
    if (memory_pool_enabled_) return;
    
    try {
        // Enable PyTorch caching allocator
        if (config_.enable_caching_allocator) {
            // Set memory fraction and max split size
            torch::cuda::set_per_process_memory_fraction(0.9);
        }
        
        memory_pool_enabled_ = true;
        std::cout << "GPU memory pooling enabled" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error enabling memory pooling: " << e.what() << std::endl;
    }
}

void GPUOptimizer::disable_memory_pooling() {
    if (!memory_pool_enabled_) return;
    
    try {
        // Clear cache
        torch::cuda::empty_cache();
        memory_pool_enabled_ = false;
        
        std::cout << "GPU memory pooling disabled" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error disabling memory pooling: " << e.what() << std::endl;
    }
}

void GPUOptimizer::clear_gpu_cache() {
    torch::cuda::empty_cache();
    update_memory_stats();
}

GPUMemoryStats GPUOptimizer::get_memory_stats() const {
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    
    // Update stats before returning
    const_cast<GPUOptimizer*>(this)->update_memory_stats();
    
    return *memory_stats_;
}

bool GPUOptimizer::is_memory_available(size_t required_bytes) const {
    auto stats = get_memory_stats();
    return (stats.total_memory - stats.allocated_memory) >= required_bytes;
}

GPUPerformanceMetrics GPUOptimizer::get_performance_metrics() const {
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    
    // Update metrics before returning
    const_cast<GPUOptimizer*>(this)->update_performance_metrics();
    
    return *performance_metrics_;
}

void GPUOptimizer::start_profiling() {
    if (profiling_active_) return;
    
    profiling_active_ = true;
    profiling_start_ = std::chrono::high_resolution_clock::now();
    
    // Reset metrics
    performance_metrics_->cuda_kernel_launches = 0;
    performance_metrics_->inference_time_ms = 0.0;
    
    std::cout << "GPU profiling started" << std::endl;
}

void GPUOptimizer::stop_profiling() {
    if (!profiling_active_) return;
    
    profiling_active_ = false;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - profiling_start_);
    
    // Calculate throughput
    if (duration.count() > 0) {
        performance_metrics_->throughput = 
            performance_metrics_->cuda_kernel_launches * 1000.0 / duration.count();
    }
    
    std::cout << "GPU profiling stopped. Duration: " << duration.count() << "ms" << std::endl;
}

void GPUOptimizer::reset_metrics() {
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    
    performance_metrics_->inference_time_ms = 0.0;
    performance_metrics_->throughput = 0.0;
    performance_metrics_->gpu_utilization = 0.0;
    performance_metrics_->power_usage_watts = 0.0;
    performance_metrics_->temperature_celsius = 0.0;
    performance_metrics_->cuda_kernel_launches = 0;
    performance_metrics_->last_update = std::chrono::system_clock::now();
}

bool GPUOptimizer::enable_tensorrt() {
    if (tensorrt_available_) return true;
    
    return initialize_tensorrt();
}

torch::jit::script::Module GPUOptimizer::optimize_model_tensorrt(torch::jit::script::Module& model) {
    if (!tensorrt_available_) {
        std::cerr << "TensorRT not available" << std::endl;
        return model;
    }
    
    // This is a simplified placeholder
    // In practice, would use TensorRT API for optimization
    std::cout << "TensorRT optimization (placeholder)" << std::endl;
    return model;
}

bool GPUOptimizer::is_tensorrt_available() const {
    return tensorrt_available_;
}

cudaStream_t GPUOptimizer::get_stream(int stream_id) {
    if (stream_id < 0 || stream_id >= static_cast<int>(cuda_streams_.size())) {
        return 0; // Default stream
    }
    return cuda_streams_[stream_id];
}

void GPUOptimizer::synchronize_stream(int stream_id) {
    if (stream_id < 0 || stream_id >= static_cast<int>(cuda_streams_.size())) {
        cudaDeviceSynchronize();
        return;
    }
    
    auto error = cudaStreamSynchronize(cuda_streams_[stream_id]);
    if (error != cudaSuccess) {
        std::cerr << "Stream synchronization failed: " << cuda_error_string(error) << std::endl;
    }
}

void GPUOptimizer::synchronize_all_streams() {
    for (auto stream : cuda_streams_) {
        cudaStreamSynchronize(stream);
    }
}

bool GPUOptimizer::capture_inference_graph(torch::jit::script::Module& model,
                                          const std::vector<torch::Tensor>& example_inputs) {
    if (graph_captured_) {
        std::cerr << "Graph already captured" << std::endl;
        return false;
    }
    
    try {
        // Begin graph capture
        cudaStreamBeginCapture(cuda_streams_[0], cudaStreamCaptureModeGlobal);
        
        // Run inference with example inputs
        auto output = model.forward(example_inputs).toTensor();
        
        // End graph capture
        auto error = cudaStreamEndCapture(cuda_streams_[0], &inference_graph_);
        if (error != cudaSuccess) {
            std::cerr << "Graph capture failed: " << cuda_error_string(error) << std::endl;
            return false;
        }
        
        // Instantiate graph
        error = cudaGraphInstantiate(&graph_exec_, inference_graph_, 0, nullptr, 0);
        if (error != cudaSuccess) {
            std::cerr << "Graph instantiation failed: " << cuda_error_string(error) << std::endl;
            cudaGraphDestroy(inference_graph_);
            return false;
        }
        
        graph_captured_ = true;
        std::cout << "CUDA graph captured successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error capturing graph: " << e.what() << std::endl;
        return false;
    }
}

torch::Tensor GPUOptimizer::run_captured_graph(const std::vector<torch::Tensor>& inputs) {
    if (!graph_captured_) {
        throw std::runtime_error("Graph not captured");
    }
    
    // Launch graph
    auto error = cudaGraphLaunch(graph_exec_, cuda_streams_[0]);
    if (error != cudaSuccess) {
        throw std::runtime_error("Graph launch failed: " + cuda_error_string(error));
    }
    
    // Synchronize and return result
    synchronize_stream(0);
    
    // This is simplified - would need proper output handling
    return torch::randn({1, 3}); // Placeholder
}

int GPUOptimizer::get_device_count() const {
    int count = 0;
    auto error = cudaGetDeviceCount(&count);
    return (error == cudaSuccess) ? count : 0;
}

int GPUOptimizer::get_current_device() const {
    int device = 0;
    auto error = cudaGetDevice(&device);
    return (error == cudaSuccess) ? device : -1;
}

bool GPUOptimizer::set_device(int device_id) {
    auto error = cudaSetDevice(device_id);
    return error == cudaSuccess;
}

std::string GPUOptimizer::get_device_info(int device_id) const {
    if (device_id == -1) {
        device_id = get_current_device();
    }
    
    cudaDeviceProp prop;
    auto error = cudaGetDeviceProperties(&prop, device_id);
    if (error != cudaSuccess) {
        return "Unknown device";
    }
    
    std::string info = std::string(prop.name) + " (";
    info += std::to_string(prop.major) + "." + std::to_string(prop.minor) + ") ";
    info += "Memory: " + std::to_string(prop.totalGlobalMem / (1024*1024)) + "MB";
    
    return info;
}

void GPUOptimizer::warmup_models(std::vector<torch::jit::script::Module>& models,
                                 const std::vector<torch::Tensor>& example_inputs) {
    std::cout << "Warming up GPU models..." << std::endl;
    
    for (int i = 0; i < config_.warmup_iterations; ++i) {
        for (auto& model : models) {
            try {
                amp_inference(model, example_inputs);
            } catch (const std::exception& e) {
                std::cerr << "Warmup iteration " << i << " failed: " << e.what() << std::endl;
            }
        }
    }
    
    // Synchronize to ensure all warmup operations complete
    synchronize_all_streams();
    
    std::cout << "GPU warmup completed" << std::endl;
}

double GPUOptimizer::benchmark_model(torch::jit::script::Module& model,
                                    const std::vector<torch::Tensor>& inputs,
                                    int iterations) {
    auto times = measure_inference_times(model, inputs, iterations);
    
    if (times.empty()) return 0.0;
    
    // Calculate statistics
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();
    
    // Remove outliers (top and bottom 10%)
    std::sort(times.begin(), times.end());
    int trim_count = times.size() * 0.1;
    
    double trimmed_sum = 0.0;
    for (int i = trim_count; i < static_cast<int>(times.size()) - trim_count; ++i) {
        trimmed_sum += times[i];
    }
    
    double trimmed_mean = trimmed_sum / (times.size() - 2 * trim_count);
    
    std::cout << "Benchmark results (" << iterations << " iterations):" << std::endl;
    std::cout << "  Mean: " << mean << "ms" << std::endl;
    std::cout << "  Trimmed mean: " << trimmed_mean << "ms" << std::endl;
    std::cout << "  Min: " << times.front() << "ms" << std::endl;
    std::cout << "  Max: " << times.back() << "ms" << std::endl;
    
    return trimmed_mean;
}

bool GPUOptimizer::check_gpu_errors() {
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "GPU error detected: " << cuda_error_string(error) << std::endl;
        return false;
    }
    return true;
}

void GPUOptimizer::reset_gpu_state() {
    cudaDeviceReset();
    std::cout << "GPU state reset" << std::endl;
}

std::string GPUOptimizer::get_last_error() const {
    auto error = cudaGetLastError();
    return cuda_error_string(error);
}

// Private methods

bool GPUOptimizer::initialize_cuda() {
    auto error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cuda_error_string(error) << std::endl;
        return false;
    }
    
    // Check device capabilities
    int device = 0;
    cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, device);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cuda_error_string(error) << std::endl;
        return false;
    }
    
    // Check if device supports FP16
    if (config_.enable_amp && !(prop.major >= 6 || (prop.major == 5 && prop.minor >= 3))) {
        std::cout << "Warning: Device may not support efficient FP16 operations" << std::endl;
    }
    
    return true;
}

bool GPUOptimizer::initialize_memory_pool() {
    // This is simplified - would use CUDA memory pool API in practice
    memory_pool_ = nullptr;
    return true;
}

bool GPUOptimizer::initialize_tensorrt() {
    // This is a placeholder for TensorRT initialization
    // In practice, would load TensorRT libraries and initialize
    tensorrt_available_ = false; // Set to true when properly integrated
    std::cout << "TensorRT initialization (placeholder)" << std::endl;
    return tensorrt_available_;
}

void GPUOptimizer::update_memory_stats() {
    size_t free_mem, total_mem;
    auto error = cudaMemGetInfo(&free_mem, &total_mem);
    if (error != cudaSuccess) {
        return;
    }
    
    memory_stats_->total_memory = total_mem;
    memory_stats_->allocated_memory = total_mem - free_mem;
    memory_stats_->utilization = static_cast<double>(memory_stats_->allocated_memory) / total_mem;
    memory_stats_->timestamp = std::chrono::system_clock::now();
    
    // Update peak allocation
    if (memory_stats_->allocated_memory > memory_stats_->max_allocated_memory) {
        memory_stats_->max_allocated_memory = memory_stats_->allocated_memory;
    }
    
    // Check memory threshold
    if (memory_stats_->utilization > config_.memory_threshold) {
        std::cout << "Warning: GPU memory usage at " 
                  << (memory_stats_->utilization * 100) << "%" << std::endl;
    }
}

void GPUOptimizer::update_performance_metrics() {
    // This would integrate with NVIDIA Management Library (NVML) for detailed metrics
    // For now, providing basic updates
    
    performance_metrics_->last_update = std::chrono::system_clock::now();
    
    // Calculate GPU utilization (simplified)
    if (performance_metrics_->cuda_kernel_launches > 0) {
        performance_metrics_->gpu_utilization = 
            std::min(1.0, performance_metrics_->cuda_kernel_launches / 100.0);
    }
}

bool GPUOptimizer::check_cuda_error(cudaError_t error, const std::string& operation) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in " << operation << ": " << cuda_error_string(error) << std::endl;
        return false;
    }
    return true;
}

std::string GPUOptimizer::cuda_error_string(cudaError_t error) const {
    return cudaGetErrorString(error);
}

torch::Tensor GPUOptimizer::convert_to_amp_dtype(const torch::Tensor& tensor) {
    if (tensor.dtype() == config_.amp_dtype) {
        return tensor;
    }
    
    // Keep BatchNorm in FP32 if configured
    if (config_.keep_batchnorm_fp32 && tensor.ndimension() >= 2) {
        return tensor;
    }
    
    return tensor.to(config_.amp_dtype);
}

torch::Tensor GPUOptimizer::convert_from_amp_dtype(const torch::Tensor& tensor) {
    if (tensor.dtype() == torch::kFloat32) {
        return tensor;
    }
    
    return tensor.to(torch::kFloat32);
}

std::vector<double> GPUOptimizer::measure_inference_times(
    torch::jit::script::Module& model,
    const std::vector<torch::Tensor>& inputs,
    int iterations) {
    
    std::vector<double> times;
    times.reserve(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            amp_inference(model, inputs);
        } catch (const std::exception& e) {
            std::cerr << "Benchmark iteration " << i << " failed: " << e.what() << std::endl;
            continue;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        times.push_back(duration.count() / 1000.0); // Convert to milliseconds
    }
    
    return times;
}

size_t GPUOptimizer::get_tensor_memory_size(const torch::Tensor& tensor) const {
    return tensor.numel() * tensor.element_size();
}

void GPUOptimizer::optimize_memory_layout(torch::Tensor& tensor) {
    // Ensure tensor is in optimal memory layout for GPU
    if (!tensor.is_contiguous()) {
        tensor = tensor.contiguous();
    }
}

// GPUContext implementation

GPUContext::GPUContext(int device_id) : device_id_(device_id), context_valid_(false) {
    auto error = cudaSetDevice(device_id);
    context_valid_ = (error == cudaSuccess);
}

GPUContext::~GPUContext() {
    if (context_valid_) {
        cudaDeviceReset();
    }
}

bool GPUContext::is_valid() const {
    return context_valid_;
}

int GPUContext::get_device_id() const {
    return device_id_;
}

// AMPScope implementation

AMPScope::AMPScope(GPUOptimizer& optimizer) : optimizer_(optimizer), amp_enabled_(false) {
    if (optimizer_.config_.enable_amp) {
        amp_enabled_ = true;
        // Store original dtype and set AMP dtype
        original_dtype_ = torch::kFloat32; // Simplified
    }
}

AMPScope::~AMPScope() {
    if (amp_enabled_) {
        // Restore original dtype
        // This would be more sophisticated in practice
    }
}

bool AMPScope::is_enabled() const {
    return amp_enabled_;
}

} // namespace gpu
} // namespace archneuronx
