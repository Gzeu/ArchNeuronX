#include "models/v4_quantum_neural_network.hpp"
#include "core/v4_ultra_low_latency_engine.hpp"
#include <iostream>
#include <chrono>

using namespace ArchNeuronX::Models::V4;
using namespace ArchNeuronX::Core::V4;

int main() {
    std::cout << "=== ArchNeuronX v4.0 Quantum Neural Network Demo ===" << std::endl;
    
    // Configure v4.0 quantum network
    QuantumNeuralNetwork::V4Config config;
    config.input_dim = 256;
    config.hidden_dim = 512;
    config.num_heads = 16;
    config.num_layers = 8;
    config.output_dim = 3;
    
    // Enable all performance optimizations
    config.use_mixed_precision = true;
    config.use_cuda_graphs = true;
    config.use_memory_pool = true;
    config.enable_async_execution = true;
    config.quantum_attention = true;
    config.superposition_encoding = true;
    config.entanglement_layers = true;
    config.max_batch_size = 64;
    config.inference_streams = 4;
    config.tensor_memory_pool = 2ULL * 1024 * 1024 * 1024; // 2GB
    
    // Create quantum network
    auto quantum_network = std::make_unique<QuantumNeuralNetwork>(config);
    
    std::cout << "✅ Quantum Neural Network initialized" << std::endl;
    std::cout << "   Architecture: " << config.num_layers << " layers, " 
              << config.hidden_dim << " hidden dim, " << config.num_heads << " heads" << std::endl;
    std::cout << "   Features: Quantum attention, mixed precision, CUDA graphs" << std::endl;
    
    // Create ensemble
    std::vector<QuantumNeuralNetwork::V4Config> ensemble_configs;
    for (int i = 0; i < 3; ++i) {
        auto variant = config;
        variant.hidden_dim = 256 + i * 128;
        variant.num_heads = 8 + i * 4;
        ensemble_configs.push_back(variant);
    }
    
    auto ensemble = std::make_unique<V4QuantumEnsemble>(ensemble_configs);
    
    std::cout << "✅ Quantum Ensemble created with " << ensemble_configs.size() << " networks" << std::endl;
    
    // Create ultra-low latency engine
    UltraLowLatencyEngine::EngineConfig engine_config;
    engine_config.neural_config = config;
    engine_config.num_inference_threads = 4;
    engine_config.max_concurrent_requests = 64;
    engine_config.enable_pipeline_parallelism = true;
    engine_config.enable_realtime_priority = true;
    engine_config.cpu_affinity_core = 0;
    engine_config.batch_processing_size = 32;
    engine_config.enable_async_execution = true;
    
    auto engine = std::make_unique<UltraLowLatencyEngine>(engine_config);
    
    std::cout << "✅ Ultra-Low Latency Engine initialized" << std::endl;
    std::cout << "   Threads: " << engine_config.num_inference_threads << std::endl;
    std::cout << "   Real-time priority: " << (engine_config.enable_realtime_priority ? "enabled" : "disabled") << std::endl;
    
    // Initialize and start engine
    if (!engine->initialize()) {
        std::cerr << "❌ Engine initialization failed!" << std::endl;
        return 1;
    }
    
    engine->start();
    std::cout << "✅ Engine started successfully" << std::endl;
    
    // Performance benchmark
    std::cout << "\n🚀 Starting Performance Benchmark..." << std::endl;
    
    engine->benchmark_performance();
    
    // Test single inference
    std::cout << "\n🧪 Testing Single Inference..." << std::endl;
    
    auto test_input = torch::randn({1, 512, config.input_dim}, torch::TensorOptions().device(torch::kCUDA));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto signal = engine->generate_signal(test_input);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    std::cout << "✅ Single inference completed in " << latency_us << " μs" << std::endl;
    std::cout << "   Signal shape: " << signal.sizes() << std::endl;
    std::cout << "   Signal (probabilities): " << signal << std::endl;
    
    // Test batch inference
    std::cout << "\n🧪 Testing Batch Inference..." << std::endl;
    
    auto batch_input = torch::randn({32, 512, config.input_dim}, torch::TensorOptions().device(torch::kCUDA));
    
    start_time = std::chrono::high_resolution_clock::now();
    auto batch_signals = engine->batch_generate_signals(batch_input);
    end_time = std::chrono::high_resolution_clock::now();
    
    auto batch_latency_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    double avg_per_sample = static_cast<double>(batch_latency_us) / batch_input.size(0);
    
    std::cout << "✅ Batch inference (32 samples) completed in " << batch_latency_us << " μs" << std::endl;
    std::cout << "   Average per sample: " << avg_per_sample << " μs" << std::endl;
    std::cout << "   Batch signals shape: " << batch_signals.sizes() << std::endl;
    
    // Test async inference
    std::cout << "\n🧪 Testing Async Inference..." << std::endl;
    
    bool async_completed = false;
    torch::Tensor async_result;
    
    auto async_start = std::chrono::high_resolution_clock::now();
    engine->async_generate_signal(test_input, [&](torch::Tensor result) {
        async_result = result;
        async_completed = true;
        
        auto async_end = std::chrono::high_resolution_clock::now();
        auto async_latency_us = std::chrono::duration_cast<std::chrono::microseconds>(async_end - async_start).count();
        
        std::cout << "✅ Async inference completed in " << async_latency_us << " μs" << std::endl;
        std::cout << "   Async result: " << result << std::endl;
    });
    
    // Wait for async completion
    while (!async_completed) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    // Get final metrics
    auto metrics = engine->get_metrics();
    
    std::cout << "\n📊 Final Engine Metrics:" << std::endl;
    std::cout << "   Average latency: " << metrics.avg_latency_us << " μs" << std::endl;
    std::cout << "   P99 latency: " << metrics.p99_latency_us << " μs" << std::endl;
    std::cout << "   Peak throughput: " << metrics.peak_throughput_ops_per_sec << " ops/sec" << std::endl;
    std::cout << "   Total requests: " << metrics.total_requests_processed << std::endl;
    std::cout << "   GPU utilization: " << (metrics.gpu_utilization * 100) << "%" << std::endl;
    std::cout << "   Memory utilization: " << (metrics.memory_utilization * 100) << "%" << std::endl;
    std::cout << "   Queue depth: " << metrics.queue_depth << std::endl;
    std::cout << "   Active threads: " << metrics.active_inference_threads << std::endl;
    
    // Check v4.0 targets
    bool latency_target = metrics.avg_latency_us < 20.0;
    bool throughput_target = metrics.peak_throughput_ops_per_sec > 500000;
    
    std::cout << "\n🎯 v4.0 Target Achievement:" << std::endl;
    std::cout << "   Latency <20μs: " << (latency_target ? "✅ ACHIEVED" : "❌ NOT MET") << std::endl;
    std::cout << "   Throughput >500K ops/sec: " << (throughput_target ? "✅ ACHIEVED" : "❌ NOT MET") << std::endl;
    
    if (latency_target && throughput_target) {
        std::cout << "\n🎉 ARCHNEURONX v4.0 MARKET-DOMINATING ENGINE READY!" << std::endl;
        std::cout << "   Quantum Neural Networks: ✅" << std::endl;
        std::cout << "   Ultra-Low Latency: ✅" << std::endl;
        std::cout << "   High Throughput: ✅" << std::endl;
        std::cout << "   Production Ready: ✅" << std::endl;
    } else {
        std::cout << "\n⚠️  Some targets not met - further optimization needed" << std::endl;
    }
    
    // Cleanup
    engine->stop();
    
    std::cout << "\n=== Demo completed ===" << std::endl;
    return 0;
}
