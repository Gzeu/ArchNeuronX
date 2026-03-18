#include "core/v4_ultra_low_latency_engine.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <random>

using namespace ArchNeuronX::Core::V4;

class V4PerformanceTest {
private:
    std::unique_ptr<UltraLowLatencyEngine> engine_;
    UltraLowLatencyEngine::EngineConfig config_;
    
    // Test parameters
    static constexpr int WARMUP_ITERATIONS = 100;
    static constexpr int BENCHMARK_ITERATIONS = 10000;
    static constexpr int BATCH_SIZE = 32;
    static constexpr int INPUT_DIM = 256;
    static constexpr int SEQ_LEN = 512;

public:
    V4PerformanceTest() {
        // Configure v4.0 engine for maximum performance
        config_.neural_config.input_dim = INPUT_DIM;
        config_.neural_config.hidden_dim = 512;
        config_.neural_config.num_heads = 16;
        config_.neural_config.num_layers = 8;
        config_.neural_config.output_dim = 3;
        
        // Enable all optimizations
        config_.neural_config.use_mixed_precision = true;
        config_.neural_config.use_cuda_graphs = true;
        config_.neural_config.quantum_attention = true;
        config_.neural_config.superposition_encoding = true;
        config_.neural_config.entanglement_layers = true;
        
        config_.num_inference_threads = 4;
        config_.max_concurrent_requests = 64;
        config_.enable_pipeline_parallelism = true;
        config_.enable_realtime_priority = true;
        config_.enable_zero_copy = true;
        config_.batch_processing_size = BATCH_SIZE;
        config_.enable_async_execution = true;
        
        engine_ = std::make_unique<UltraLowLatencyEngine>(config_);
    }
    
    bool initialize() {
        if (!engine_->initialize()) {
            std::cerr << "❌ Failed to initialize v4.0 engine" << std::endl;
            return false;
        }
        
        engine_->start();
        std::cout << "✅ v4.0 engine initialized and started" << std::endl;
        return true;
    }
    
    void run_all_tests() {
        std::cout << "🚀 ArchNeuronX v4.0 Performance Test Suite" << std::endl;
        std::cout << "===========================================" << std::endl;
        
        // Warmup
        std::cout << "\n🔥 Warming up engine..." << std::endl;
        warmup_engine();
        
        // Latency tests
        std::cout << "\n⚡ Latency Tests:" << std::endl;
        test_single_inference_latency();
        test_batch_inference_latency();
        test_async_inference_latency();
        
        // Throughput tests
        std::cout << "\n🌊 Throughput Tests:" << std::endl;
        test_single_thread_throughput();
        test_multi_thread_throughput();
        test_concurrent_requests();
        
        // Stress tests
        std::cout << "\n💪 Stress Tests:" << std::endl;
        test_memory_pressure();
        test_gpu_utilization();
        
        // v4.0 target validation
        std::cout << "\n🎯 v4.0 Target Validation:" << std::endl;
        validate_v4_targets();
        
        std::cout << "\n✅ All tests completed!" << std::endl;
    }
    
private:
    void warmup_engine() {
        auto warmup_input = torch::randn({1, SEQ_LEN, INPUT_DIM}, torch::TensorOptions().device(torch::kCUDA));
        
        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            engine_->generate_signal(warmup_input);
        }
        
        std::cout << "   Warmup completed: " << WARMUP_ITERATIONS << " iterations" << std::endl;
    }
    
    void test_single_inference_latency() {
        std::cout << "   Testing single inference latency..." << std::endl;
        
        auto test_input = torch::randn({1, SEQ_LEN, INPUT_DIM}, torch::TensorOptions().device(torch::kCUDA));
        
        std::vector<double> latencies;
        latencies.reserve(BENCHMARK_ITERATIONS);
        
        for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = engine_->generate_signal(test_input);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            latencies.push_back(latency_us);
        }
        
        // Calculate statistics
        std::sort(latencies.begin(), latencies.end());
        double avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double p50 = latencies[latencies.size() * 0.5];
        double p95 = latencies[latencies.size() * 0.95];
        double p99 = latencies[latencies.size() * 0.99];
        double min_latency = latencies[0];
        double max_latency = latencies.back();
        
        std::cout << "     Average: " << avg_latency << " μs" << std::endl;
        std::cout << "     P50: " << p50 << " μs" << std::endl;
        std::cout << "     P95: " << p95 << " μs" << std::endl;
        std::cout << "     P99: " << p99 << " μs" << std::endl;
        std::cout << "     Min: " << min_latency << " μs" << std::endl;
        std::cout << "     Max: " << max_latency << " μs" << std::endl;
        
        // Check v4.0 latency target
        bool latency_target = avg_latency < 20.0;
        std::cout << "     v4.0 Target (<20μs): " << (latency_target ? "✅ PASS" : "❌ FAIL") << std::endl;
    }
    
    void test_batch_inference_latency() {
        std::cout << "   Testing batch inference latency..." << std::endl;
        
        auto batch_input = torch::randn({BATCH_SIZE, SEQ_LEN, INPUT_DIM}, torch::TensorOptions().device(torch::kCUDA));
        
        std::vector<double> batch_latencies;
        batch_latencies.reserve(BENCHMARK_ITERATIONS / 10); // Fewer iterations for batch
        
        for (int i = 0; i < BENCHMARK_ITERATIONS / 10; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = engine_->batch_generate_signals(batch_input);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            batch_latencies.push_back(latency_us);
        }
        
        double avg_batch_latency = std::accumulate(batch_latencies.begin(), batch_latencies.end(), 0.0) / batch_latencies.size();
        double avg_per_sample = avg_batch_latency / BATCH_SIZE;
        
        std::cout << "     Batch (" << BATCH_SIZE << " samples): " << avg_batch_latency << " μs" << std::endl;
        std::cout << "     Average per sample: " << avg_per_sample << " μs" << std::endl;
    }
    
    void test_async_inference_latency() {
        std::cout << "   Testing async inference latency..." << std::endl;
        
        auto test_input = torch::randn({1, SEQ_LEN, INPUT_DIM}, torch::TensorOptions().device(torch::kCUDA));
        
        std::vector<double> async_latencies;
        async_latencies.reserve(BENCHMARK_ITERATIONS / 5);
        
        for (int i = 0; i < BENCHMARK_ITERATIONS / 5; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            
            bool completed = false;
            torch::Tensor result;
            
            engine_->async_generate_signal(test_input, [&](torch::Tensor async_result) {
                auto end = std::chrono::high_resolution_clock::now();
                auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                async_latencies.push_back(latency_us);
                completed = true;
            });
            
            // Wait for completion
            while (!completed) {
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        }
        
        double avg_async_latency = std::accumulate(async_latencies.begin(), async_latencies.end(), 0.0) / async_latencies.size();
        std::cout << "     Average async latency: " << avg_async_latency << " μs" << std::endl;
    }
    
    void test_single_thread_throughput() {
        std::cout << "   Testing single-thread throughput..." << std::endl;
        
        auto test_input = torch::randn({1, SEQ_LEN, INPUT_DIM}, torch::TensorOptions().device(torch::kCUDA));
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
            engine_->generate_signal(test_input);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_sec = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        
        double throughput_ops_per_sec = BENCHMARK_ITERATIONS / std::max(1.0, duration_sec);
        double throughput_orders_per_sec = throughput_ops_per_sec / 3.0; // Assuming 3 ops per order
        
        std::cout << "     Operations/sec: " << throughput_ops_per_sec << std::endl;
        std::cout << "     Orders/sec: " << throughput_orders_per_sec << std::endl;
        
        // Check v4.0 throughput target
        bool throughput_target = throughput_ops_per_sec > 500000;
        std::cout << "     v4.0 Target (>500K ops/sec): " << (throughput_target ? "✅ PASS" : "❌ FAIL") << std::endl;
    }
    
    void test_multi_thread_throughput() {
        std::cout << "   Testing multi-thread throughput..." << std::endl;
        
        const int num_threads = 4;
        const int iterations_per_thread = BENCHMARK_ITERATIONS / num_threads;
        
        auto test_input = torch::randn({1, SEQ_LEN, INPUT_DIM}, torch::TensorOptions().device(torch::kCUDA));
        
        std::vector<std::thread> threads;
        std::atomic<int> completed_threads{0};
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                for (int i = 0; i < iterations_per_thread; ++i) {
                    engine_->generate_signal(test_input);
                }
                completed_threads++;
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_sec = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        
        double total_ops = num_threads * iterations_per_thread;
        double throughput_ops_per_sec = total_ops / std::max(1.0, duration_sec);
        
        std::cout << "     Multi-thread ops/sec: " << throughput_ops_per_sec << std::endl;
        std::cout << "     Threads used: " << num_threads << std::endl;
    }
    
    void test_concurrent_requests() {
        std::cout << "   Testing concurrent request handling..." << std::endl;
        
        const int concurrent_requests = 32;
        const int iterations_per_request = 100;
        
        std::vector<std::thread> threads;
        std::atomic<int> completed_requests{0};
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int r = 0; r < concurrent_requests; ++r) {
            threads.emplace_back([&, r]() {
                auto test_input = torch::randn({1, SEQ_LEN, INPUT_DIM}, torch::TensorOptions().device(torch::kCUDA));
                
                for (int i = 0; i < iterations_per_request; ++i) {
                    engine_->generate_signal(test_input);
                }
                
                completed_requests++;
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_sec = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        
        double total_ops = concurrent_requests * iterations_per_request;
        double throughput_ops_per_sec = total_ops / std::max(1.0, duration_sec);
        
        std::cout << "     Concurrent ops/sec: " << throughput_ops_per_sec << std::endl;
        std::cout << "     Concurrent requests: " << concurrent_requests << std::endl;
    }
    
    void test_memory_pressure() {
        std::cout << "   Testing memory pressure..." << std::endl;
        
        // Create large batch to test memory handling
        auto large_batch = torch::randn({64, SEQ_LEN, INPUT_DIM}, torch::TensorOptions().device(torch::kCUDA));
        
        std::vector<double> memory_latencies;
        
        for (int i = 0; i < 1000; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = engine_->batch_generate_signals(large_batch);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            memory_latencies.push_back(latency_us);
        }
        
        double avg_memory_latency = std::accumulate(memory_latencies.begin(), memory_latencies.end(), 0.0) / memory_latencies.size();
        std::cout << "     Large batch (64 samples) avg latency: " << avg_memory_latency << " μs" << std::endl;
    }
    
    void test_gpu_utilization() {
        std::cout << "   Testing GPU utilization..." << std::endl;
        
        auto test_input = torch::randn({BATCH_SIZE, SEQ_LEN, INPUT_DIM}, torch::TensorOptions().device(torch::kCUDA));
        
        // Sustained load test
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < 5000; ++i) {
            engine_->batch_generate_signals(test_input);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_sec = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        
        double sustained_throughput = (5000.0 * BATCH_SIZE) / std::max(1.0, duration_sec);
        std::cout << "     Sustained throughput: " << sustained_throughput << " ops/sec" << std::endl;
    }
    
    void validate_v4_targets() {
        auto metrics = engine_->get_metrics();
        
        std::cout << "   Current metrics:" << std::endl;
        std::cout << "     Average latency: " << metrics.avg_latency_us << " μs" << std::endl;
        std::cout << "     P99 latency: " << metrics.p99_latency_us << " μs" << std::endl;
        std::cout << "     Peak throughput: " << metrics.peak_throughput_ops_per_sec << " ops/sec" << std::endl;
        std::cout << "     GPU utilization: " << (metrics.gpu_utilization * 100) << "%" << std::endl;
        std::cout << "     Memory utilization: " << (metrics.memory_utilization * 100) << "%" << std::endl;
        
        // Validate v4.0 targets
        bool latency_target = metrics.avg_latency_us < 20.0;
        bool throughput_target = metrics.peak_throughput_ops_per_sec > 500000;
        bool gpu_target = metrics.gpu_utilization < 0.95;
        bool memory_target = metrics.memory_utilization < 0.90;
        
        std::cout << "\n   v4.0 Target Achievement:" << std::endl;
        std::cout << "     Latency <20μs: " << (latency_target ? "✅ ACHIEVED" : "❌ NOT MET") << std::endl;
        std::cout << "     Throughput >500K ops/sec: " << (throughput_target ? "✅ ACHIEVED" : "❌ NOT MET") << std::endl;
        std::cout << "     GPU <95%: " << (gpu_target ? "✅ ACHIEVED" : "❌ NOT MET") << std::endl;
        std::cout << "     Memory <90%: " << (memory_target ? "✅ ACHIEVED" : "❌ NOT MET") << std::endl;
        
        bool all_targets_met = latency_target && throughput_target && gpu_target && memory_target;
        std::cout << "\n   Overall v4.0 Status: " << (all_targets_met ? "🎉 MARKET-DOMINATING!" : "⚠️  OPTIMIZATION NEEDED") << std::endl;
    }
};

int main() {
    std::cout << "🧪 ArchNeuronX v4.0 Performance Test Suite" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    try {
        V4PerformanceTest test;
        
        if (!test.initialize()) {
            std::cerr << "❌ Failed to initialize test suite" << std::endl;
            return 1;
        }
        
        test.run_all_tests();
        
        std::cout << "\n✅ Performance test completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
