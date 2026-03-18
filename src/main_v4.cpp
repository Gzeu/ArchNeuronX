#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <signal.h>
#include <atomic>

// v4.0 headers
#include "core/v4_ultra_low_latency_engine.hpp"
#include "api/v4_rest_server.hpp"
#include "monitoring/logger.hpp"

using namespace ArchNeuronX::Core::V4;
using namespace ArchNeuronX::API;

// Global shutdown handler
std::atomic<bool> g_shutdown{false};

void signal_handler(int signal) {
    std::cout << "\n🛑 Received signal " << signal << ", shutting down gracefully..." << std::endl;
    g_shutdown = true;
}

int main(int argc, char* argv[]) {
    std::cout << "🚀 ArchNeuronX v4.0 - Market-Dominating Execution Engine" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    try {
        // Parse command line arguments
        std::string config_path = "config/v4_production.json";
        int port = 8080;
        
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--config" && i + 1 < argc) {
                config_path = argv[++i];
            } else if (arg == "--port" && i + 1 < argc) {
                port = std::stoi(argv[++i]);
            } else if (arg == "--help") {
                std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  --config <path>    Configuration file path" << std::endl;
                std::cout << "  --port <port>      REST API port (default: 8080)" << std::endl;
                std::cout << "  --help             Show this help" << std::endl;
                return 0;
            }
        }
        
        std::cout << "📋 Configuration:" << std::endl;
        std::cout << "   Config file: " << config_path << std::endl;
        std::cout << "   API port: " << port << std::endl;
        
        // Initialize v4.0 ultra-low latency engine
        std::cout << "\n🧠 Initializing v4.0 Quantum Neural Engine..." << std::endl;
        
        UltraLowLatencyEngine::EngineConfig engine_config;
        
        // Configure for ultra-low latency
        engine_config.neural_config.input_dim = 256;
        engine_config.neural_config.hidden_dim = 512;
        engine_config.neural_config.num_heads = 16;
        engine_config.neural_config.num_layers = 8;
        engine_config.neural_config.output_dim = 3;
        
        // Enable all v4.0 optimizations
        engine_config.neural_config.use_mixed_precision = true;
        engine_config.neural_config.use_cuda_graphs = true;
        engine_config.neural_config.quantum_attention = true;
        engine_config.neural_config.superposition_encoding = true;
        engine_config.neural_config.entanglement_layers = true;
        
        engine_config.num_inference_threads = 4;
        engine_config.max_concurrent_requests = 64;
        engine_config.enable_pipeline_parallelism = true;
        engine_config.enable_realtime_priority = true;
        engine_config.enable_zero_copy = true;
        engine_config.batch_processing_size = 32;
        engine_config.enable_async_execution = true;
        
        auto engine = std::make_unique<UltraLowLatencyEngine>(engine_config);
        
        if (!engine->initialize()) {
            std::cerr << "❌ Failed to initialize v4.0 engine!" << std::endl;
            return 1;
        }
        
        std::cout << "✅ v4.0 Quantum Neural Engine initialized" << std::endl;
        std::cout << "   Architecture: " << engine_config.neural_config.num_layers 
                  << " layers, " << engine_config.neural_config.hidden_dim << " hidden dim" << std::endl;
        std::cout << "   Optimizations: Mixed precision, CUDA graphs, Quantum attention" << std::endl;
        
        // Start the engine
        engine->start();
        std::cout << "✅ v4.0 Engine started successfully" << std::endl;
        
        // Initialize v4.0 REST API server
        std::cout << "\n🌐 Initializing v4.0 REST API Server..." << std::endl;
        
        V4RestServer::Config server_config;
        server_config.port = port;
        server_config.num_threads = 8;
        server_config.enable_ssl = false;
        server_config.enable_compression = true;
        server_config.max_request_size = 1024 * 1024; // 1MB
        server_config.timeout_ms = 5000;
        
        auto server = std::make_unique<V4RestServer>(server_config, engine.get());
        
        if (!server->initialize()) {
            std::cerr << "❌ Failed to initialize v4.0 REST server!" << std::endl;
            engine->stop();
            return 1;
        }
        
        std::cout << "✅ v4.0 REST API Server initialized" << std::endl;
        std::cout << "   Port: " << port << std::endl;
        std::cout << "   Threads: " << server_config.num_threads << std::endl;
        
        // Start the server
        server->start();
        std::cout << "✅ v4.0 REST API Server started" << std::endl;
        
        // Performance benchmark
        std::cout << "\n🚀 Running v4.0 Performance Benchmark..." << std::endl;
        engine->benchmark_performance();
        
        // Display v4.0 status
        std::cout << "\n📊 ArchNeuronX v4.0 Status:" << std::endl;
        std::cout << "   🧠 Quantum Neural Engine: ✅ Running" << std::endl;
        std::cout << "   🌐 REST API Server: ✅ Running on http://localhost:" << port << std::endl;
        std::cout << "   🎯 Target Latency: <20μs" << std::endl;
        std::cout << "   🎯 Target Throughput: 500K+ ops/sec" << std::endl;
        std::cout << "   🔧 Optimizations: Mixed precision, CUDA graphs, Quantum attention" << std::endl;
        
        std::cout << "\n🌟 v4.0 API Endpoints:" << std::endl;
        std::cout << "   GET  /api/v4/status          - Engine status and metrics" << std::endl;
        std::cout << "   GET  /api/v4/health          - Health check" << std::endl;
        std::cout << "   POST /api/v4/signal          - Generate trading signal" << std::endl;
        std::cout << "   POST /api/v4/batch-signal    - Batch signal generation" << std::endl;
        std::cout << "   GET  /api/v4/performance     - Performance metrics" << std::endl;
        std::cout << "   GET  /api/v4/models          - Model information" << std::endl;
        std::cout << "   GET  /                      - Welcome page" << std::endl;
        
        std::cout << "\n⚡ v4.0 is ready for market domination!" << std::endl;
        std::cout << "   Press Ctrl+C to shutdown gracefully..." << std::endl;
        
        // Main loop
        while (!g_shutdown) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Periodic status update
            static auto last_status = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_status).count() >= 30) {
                auto metrics = engine->get_metrics();
                std::cout << "📈 Status: " << metrics.total_requests_processed 
                          << " requests, " << metrics.avg_latency_us << "μs avg latency" << std::endl;
                last_status = now;
            }
        }
        
        // Graceful shutdown
        std::cout << "\n🛑 Shutting down v4.0..." << std::endl;
        
        // Stop server first
        server->stop();
        std::cout << "✅ REST API Server stopped" << std::endl;
        
        // Stop engine
        engine->stop();
        std::cout << "✅ Quantum Neural Engine stopped" << std::endl;
        
        // Final metrics
        auto final_metrics = engine->get_metrics();
        std::cout << "\n📊 Final v4.0 Metrics:" << std::endl;
        std::cout << "   Total requests: " << final_metrics.total_requests_processed << std::endl;
        std::cout << "   Average latency: " << final_metrics.avg_latency_us << "μs" << std::endl;
        std::cout << "   Peak throughput: " << final_metrics.peak_throughput_ops_per_sec << " ops/sec" << std::endl;
        std::cout << "   P99 latency: " << final_metrics.p99_latency_us << "μs" << std::endl;
        
        // Check v4.0 targets
        bool latency_target = final_metrics.avg_latency_us < 20.0;
        bool throughput_target = final_metrics.peak_throughput_ops_per_sec > 500000;
        
        std::cout << "\n🎯 v4.0 Target Achievement:" << std::endl;
        std::cout << "   Latency <20μs: " << (latency_target ? "✅ ACHIEVED" : "❌ NOT MET") << std::endl;
        std::cout << "   Throughput >500K ops/sec: " << (throughput_target ? "✅ ACHIEVED" : "❌ NOT MET") << std::endl;
        
        if (latency_target && throughput_target) {
            std::cout << "\n🎉 ARCHNEURONX v4.0 MARKET-DOMINATING ENGINE SUCCESS!" << std::endl;
        } else {
            std::cout << "\n⚠️  Some v4.0 targets not met - further optimization needed" << std::endl;
        }
        
        std::cout << "\n✅ v4.0 shutdown complete" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
