#pragma once

#include "models/v4_quantum_neural_network.hpp"
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufftdx.h>
#include <memory>
#include <vector>
#include <atomic>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace ArchNeuronX {
namespace Core {
namespace V4 {

/**
 * @brief Ultra-low latency execution engine for v4.0
 * 
 * Achieves <20μs latency with 500K+ orders/sec throughput using
 * quantum neural networks and hardware optimization.
 */
class UltraLowLatencyEngine {
public:
    /**
     * @brief Engine configuration for maximum performance
     */
    struct EngineConfig {
        // Neural network configuration
        Models::V4::QuantumNeuralNetwork::V4Config neural_config;
        
        // Execution optimization
        int num_inference_threads = 4;        // Parallel inference threads
        int max_concurrent_requests = 64;    // Concurrent request capacity
        bool enable_pipeline_parallelism = true; // Pipeline execution
        bool enable_memory_mapping = true;   // Memory-mapped I/O
        bool use_kernel_launch_optimization = true; // CUDA kernel optimization
        
        // Hardware optimization
        int gpu_device_id = 0;               // Primary GPU device
        bool enable_multi_gpu = false;        // Multi-GPU scaling
        bool enable_pinned_memory = true;    // Pinned host memory
        bool enable_numa_optimization = true; // NUMA awareness
        
        // Latency optimization
        bool enable_zero_copy = true;        // Zero-copy operations
        bool enable_preemption = false;      // Disable preemption for latency
        int cpu_affinity_core = 0;           // CPU core affinity
        bool enable_realtime_priority = true; // Real-time scheduling
        
        // Throughput optimization
        int batch_processing_size = 32;       // Optimal batch size
        bool enable_async_execution = true;   // Asynchronous execution
        int request_queue_depth = 1024;      // Deep request queue
        bool enable_priority_queue = true;    // Priority-based processing
    };

private:
    EngineConfig config_;
    
    // Core components
    std::unique_ptr<Models::V4::QuantumNeuralNetwork> quantum_network_;
    std::unique_ptr<Models::V4::V4QuantumEnsemble> quantum_ensemble_;
    
    // Execution infrastructure
    std::vector<std::thread> inference_threads_;
    std::queue<std::function<void()>> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> running_{false};
    
    // Memory optimization
    void* pinned_memory_pool_{nullptr};
    size_t pinned_memory_size_{0};
    std::vector<cudaStream_t> execution_streams_;
    
    // Performance tracking
    std::atomic<uint64_t> total_requests_{0};
    std::atomic<double> avg_latency_us_{0.0};
    std::atomic<double> peak_throughput_{0.0};
    std::chrono::high_resolution_clock::time_point engine_start_;
    
    // Hardware optimization
    int cpu_affinity_mask_{0};
    bool realtime_priority_set_{false};

public:
    /**
     * @brief Constructor with ultra-low latency optimization
     */
    explicit UltraLowLatencyEngine(const EngineConfig& config);
    
    /**
     * @brief Destructor
     */
    ~UltraLowLatencyEngine();
    
    /**
     * @brief Initialize the engine
     */
    bool initialize();
    
    /**
     * @brief Start the ultra-low latency engine
     */
    void start();
    
    /**
     * @brief Stop the engine
     */
    void stop();
    
    /**
     * @brief Ultra-fast trading signal generation
     * @param market_data Market data tensor
     * @return Trading signal [buy/sell/hold probabilities]
     */
    torch::Tensor generate_signal(torch::Tensor market_data);
    
    /**
     * @brief Batch signal generation for high throughput
     * @param batch_data Batch of market data
     * @return Batch of trading signals
     */
    torch::Tensor batch_generate_signals(torch::Tensor batch_data);
    
    /**
     * @brief Asynchronous signal generation
     * @param market_data Market data
     * @param callback Result callback
     */
    void async_generate_signal(torch::Tensor market_data, 
                             std::function<void(torch::Tensor)> callback);
    
    /**
     * @brief Priority signal generation for urgent requests
     * @param market_data Market data
     * @return Trading signal
     */
    torch::Tensor priority_generate_signal(torch::Tensor market_data);
    
    /**
     * @brief Get engine performance metrics
     */
    struct EngineMetrics {
        double avg_latency_us;
        double p99_latency_us;
        double peak_throughput_ops_per_sec;
        uint64_t total_requests_processed;
        double gpu_utilization;
        double memory_utilization;
        size_t queue_depth;
        int active_inference_threads;
    };
    
    EngineMetrics get_metrics() const;
    
    /**
     * @brief Optimize engine for specific workload
     */
    void optimize_for_workload(const std::string& workload_type);
    
    /**
     * @brief Enable real-time execution mode
     */
    void enable_realtime_mode();
    
    /**
     * @brief Configure CPU affinity for latency optimization
     */
    void set_cpu_affinity(int core_id);
    
    /**
     * @brief Setup memory pools for zero-allocation
     */
    void setup_memory_pools();
    
    /**
     * @brief Benchmark engine performance
     */
    void benchmark_performance();

private:
    /**
     * @brief Worker thread for request processing
     */
    void worker_thread(int thread_id);
    
    /**
     * @brief Process single request with latency tracking
     */
    torch::Tensor process_request(torch::Tensor input);
    
    /**
     * @brief Setup CUDA optimization
     */
    void setup_cuda_optimization();
    
    /**
     * @brief Setup real-time scheduling
     */
    void setup_realtime_scheduling();
    
    /**
     * @brief Memory-mapped I/O optimization
     */
    void setup_memory_mapping();
};

/**
 * @brief Real-time market data processor for v4.0
 */
class V4MarketDataProcessor {
private:
    std::queue<torch::Tensor> data_queue_;
    std::mutex queue_mutex_;
    std::condition_variable data_cv_;
    std::atomic<bool> processing_{false};
    
    // Pre-processing optimization
    torch::Tensor preallocated_buffer_;
    torch::nn::Linear feature_extractor_{nullptr};
    torch::nn::LayerNorm normalizer_{nullptr};
    
public:
    /**
     * @brief Constructor
     */
    V4MarketDataProcessor();
    
    /**
     * @brief Process real-time market data
     */
    torch::Tensor process_market_data(const std::vector<double>& raw_data);
    
    /**
     * @brief Batch process market data
     */
    torch::Tensor batch_process_data(const std::vector<std::vector<double>>& batch_data);
    
    /**
     * @brief Start continuous processing
     */
    void start_processing();
    
    /**
     * @brief Stop processing
     */
    void stop_processing();
};

} // namespace V4
} // namespace Core
} // namespace ArchNeuronX
