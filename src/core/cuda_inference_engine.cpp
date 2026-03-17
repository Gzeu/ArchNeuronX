/**
 * @file cuda_inference_engine.cpp
 * @brief CUDA inference engine implementation
 * @author George Pricop
 * @date 2025-10-02
 */

#include "core/cuda_inference_engine.hpp"
#include <chrono>
#include <iostream>
#include <fstream>

namespace ArchNeuronX {
namespace Core {

// CUDA kernel implementations
__global__ void matMulKernel(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void reluKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void tanhKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = tanhf(data[idx]);
    }
}

__global__ void sigmoidKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
    }
}

__global__ void softmaxKernel(float* data, int size, int dim) {
    extern __shared__ float sdata[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Find max value for numerical stability
    float max_val = -INFINITY;
    for (int i = tid; i < dim; i += blockDim.x) {
        max_val = fmaxf(max_val, data[idx * dim + i]);
    }
    
    // Reduce to find global max
    sdata[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    max_val = sdata[0];
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float exp_val = expf(data[idx * dim + i] - max_val);
        data[idx * dim + i] = exp_val;
        sum += exp_val;
    }
    
    // Reduce sum
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    sum = sdata[0];
    
    // Normalize
    for (int i = tid; i < dim; i += blockDim.x) {
        data[idx * dim + i] /= sum;
    }
}

CUDAInferenceEngine::CUDAInferenceEngine(const Config& config)
    : config_(config), initialized_(false), memory_pool_(nullptr), memory_pool_size_(0) {
}

CUDAInferenceEngine::~CUDAInferenceEngine() {
    // Clean up CUDA resources
    for (auto& stream : streams_) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
    
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
    
    if (memory_pool_) {
        cudaFree(memory_pool_);
    }
    
    // Clean up models
    for (auto& [name, model_info] : models_) {
        if (model_info.graph) {
            cudaGraphDestroy(model_info.graph);
        }
        if (model_info.graph_exec) {
            cudaGraphExecDestroy(model_info.graph_exec);
        }
    }
}

bool CUDAInferenceEngine::initialize() {
    if (initialized_) {
        return true;
    }
    
    try {
        if (!setupCUDA()) {
            return false;
        }
        
        if (!setupMemoryPool()) {
            return false;
        }
        
        if (!setupStreams()) {
            return false;
        }
        
        if (!setupCUBLAS()) {
            return false;
        }
        
        initialized_ = true;
        std::cout << "CUDA Inference Engine initialized on device " 
                  << config_.device_id << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize CUDA Inference Engine: " 
                  << e.what() << std::endl;
        return false;
    }
}

bool CUDAInferenceEngine::setupCUDA() {
    // Check CUDA device
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return false;
    }
    
    if (config_.device_id >= device_count) {
        std::cerr << "Invalid device ID: " << config_.device_id << std::endl;
        return false;
    }
    
    // Set device
    error = cudaSetDevice(config_.device_id);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Get device properties
    error = cudaGetDeviceProperties(&device_prop_, config_.device_id);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    std::cout << "Using CUDA device: " << device_prop_.name << std::endl;
    return true;
}

bool CUDAInferenceEngine::setupMemoryPool() {
    memory_pool_size_ = config_.memory_pool_size;
    
    cudaError_t error = cudaMalloc(&memory_pool_, memory_pool_size_);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate memory pool: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    allocated_blocks_.clear();
    return true;
}

bool CUDAInferenceEngine::setupStreams() {
    streams_.resize(config_.stream_count);
    
    for (int i = 0; i < config_.stream_count; ++i) {
        cudaError_t error = cudaStreamCreate(&streams_[i]);
        if (error != cudaSuccess) {
            std::cerr << "Failed to create CUDA stream " << i << ": " 
                      << cudaGetErrorString(error) << std::endl;
            return false;
        }
    }
    
    return true;
}

bool CUDAInferenceEngine::setupCUBLAS() {
    cublasStatus_t status = cublasCreate(&cublas_handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create CUBLAS handle" << std::endl;
        return false;
    }
    
    // Set stream for CUBLAS
    cublasSetStream(cublas_handle_, streams_[0]);
    
    return true;
}

bool CUDAInferenceEngine::loadModel(std::shared_ptr<torch::nn::Module> model,
                                  const std::string& model_name) {
    if (!initialized_) {
        std::cerr << "CUDA Inference Engine not initialized" << std::endl;
        return false;
    }
    
    try {
        // Move model to GPU
        model->to(torch::kCUDA);
        model->eval();
        
        ModelInfo model_info;
        model_info.model = model;
        model_info.is_optimized = false;
        model_info.is_captured = false;
        model_info.graph = nullptr;
        model_info.graph_exec = nullptr;
        
        // Create dummy input for graph capture
        // This should be customized based on expected input shape
        model_info.dummy_input = torch::randn({1, 50, config_.max_batch_size})
                                   .to(torch::kCUDA);
        
        models_[model_name] = model_info;
        
        // Optimize model
        optimizeModel(model_name);
        
        std::cout << "Model '" << model_name << "' loaded successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model '" << model_name << "': " 
                  << e.what() << std::endl;
        return false;
    }
}

torch::Tensor CUDAInferenceEngine::inference(const std::string& model_name,
                                           torch::Tensor input) {
    auto it = models_.find(model_name);
    if (it == models_.end()) {
        throw std::runtime_error("Model not found: " + model_name);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Ensure input is on GPU
    if (!input.device().is_cuda()) {
        input = input.to(torch::kCUDA);
    }
    
    torch::Tensor output;
    
    if (it->second.is_captured && config_.enable_graph_capture) {
        // Use captured graph for maximum performance
        cudaGraphExec_t graph_exec = it->second.graph_exec;
        cudaGraphLaunch(graph_exec, streams_[0]);
        cudaStreamSynchronize(streams_[0]);
        
        // Note: In a real implementation, you'd need to handle output tensor management
        // This is a simplified version
        output = it->second.model->forward(input);
    } else {
        // Standard inference
        output = runInference(it->second, input, streams_[0]);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    updateStats(model_name, duration);
    
    return output;
}

torch::Tensor CUDAInferenceEngine::runInference(ModelInfo& model_info,
                                              torch::Tensor input,
                                              cudaStream_t stream) {
    // Set stream if provided
    if (stream) {
        torch::cuda::set_current_stream(torch::cuda::CUDAStream(stream));
    }
    
    // Run inference with no gradient
    torch::NoGradGuard no_grad;
    return model_info.model->forward(input);
}

bool CUDAInferenceEngine::optimizeModel(const std::string& model_name) {
    auto it = models_.find(model_name);
    if (it == models_.end()) {
        return false;
    }
    
    try {
        // Enable Tensor Core usage if available
        if (config_.enable_tensor_cores && device_prop_.major >= 7) {
            // Enable mixed precision if supported
            if (config_.enable_fp16) {
                it->second.model->to(torch::kHalf);
            }
        }
        
        // Optimize for inference
        it->second.model->eval();
        
        // Capture CUDA graph if enabled
        if (config_.enable_graph_capture) {
            captureModelGraph(model_name);
        }
        
        it->second.is_optimized = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to optimize model '" << model_name << "': " 
                  << e.what() << std::endl;
        return false;
    }
}

bool CUDAInferenceEngine::captureModelGraph(const std::string& model_name) {
    auto it = models_.find(model_name);
    if (it == models_.end()) {
        return false;
    }
    
    try {
        // Begin graph capture
        cudaGraph_t graph;
        cudaStreamBeginCapture(streams_[0], cudaStreamCaptureModeGlobal);
        
        // Run inference with dummy input
        auto dummy_output = it->second.model->forward(it->second.dummy_input);
        
        // End graph capture
        cudaStreamEndCapture(streams_[0], &graph);
        
        // Instantiate graph
        cudaGraphExec_t graph_exec;
        cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
        
        it->second.graph = graph;
        it->second.graph_exec = graph_exec;
        it->second.is_captured = true;
        
        std::cout << "CUDA graph captured for model '" << model_name << "'" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to capture graph for model '" << model_name << "': " 
                  << e.what() << std::endl;
        return false;
    }
}

void CUDAInferenceEngine::warmUp(const std::string& model_name, int warmup_iterations) {
    auto it = models_.find(model_name);
    if (it == models_.end()) {
        return;
    }
    
    std::cout << "Warming up model '" << model_name << "' with " 
              << warmup_iterations << " iterations..." << std::endl;
    
    for (int i = 0; i < warmup_iterations; ++i) {
        auto dummy_input = torch::randn({1, 50, config_.max_batch_size})
                              .to(torch::kCUDA);
        auto output = inference(model_name, dummy_input);
        
        // Synchronize to ensure computation completes
        cudaDeviceSynchronize();
    }
    
    std::cout << "Warm up completed for model '" << model_name << "'" << std::endl;
}

void CUDAInferenceEngine::updateStats(const std::string& model_name, double inference_time) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    inference_times_[model_name] += inference_time;
    inference_counts_[model_name]++;
}

std::map<std::string, double> CUDAInferenceEngine::getInferenceStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    std::map<std::string, double> stats;
    for (const auto& [name, total_time] : inference_times_) {
        int count = inference_counts_.at(name);
        stats[name + "_avg_time_us"] = total_time / count;
        stats[name + "_total_inferences"] = count;
    }
    
    return stats;
}

std::pair<size_t, size_t> CUDAInferenceEngine::getMemoryUsage() const {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    
    cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error == cudaSuccess) {
        return {free_bytes, total_bytes};
    }
    
    return {0, 0};
}

void CUDAInferenceEngine::customMatMul(const float* A, const float* B, float* C,
                                    int M, int N, int K, cudaStream_t stream) {
    // Launch custom matrix multiplication kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);
    
    matMulKernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
}

void CUDAInferenceEngine::customActivation(float* data, int size, 
                                        const std::string& activation,
                                        cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    if (activation == "relu") {
        reluKernel<<<gridSize, blockSize, 0, stream>>>(data, size);
    } else if (activation == "tanh") {
        tanhKernel<<<gridSize, blockSize, 0, stream>>>(data, size);
    } else if (activation == "sigmoid") {
        sigmoidKernel<<<gridSize, blockSize, 0, stream>>>(data, size);
    }
}

} // namespace Core
} // namespace ArchNeuronX
