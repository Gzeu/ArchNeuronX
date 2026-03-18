#pragma once

#include "core/v4_ultra_low_latency_engine.hpp"
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <functional>
#include <map>
#include <vector>

namespace ArchNeuronX {
namespace API {
namespace V4 {

/**
 * @brief v4.0 Ultra-fast REST API server
 * 
 * High-performance HTTP server optimized for <20μs response times
 * with quantum neural network integration.
 */
class V4RestServer {
public:
    /**
     * @brief Server configuration
     */
    struct Config {
        int port = 8080;
        int num_threads = 8;
        bool enable_ssl = false;
        bool enable_compression = true;
        size_t max_request_size = 1024 * 1024; // 1MB
        int timeout_ms = 5000;
        std::string ssl_cert_path = "";
        std::string ssl_key_path = "";
    };

private:
    Config config_;
    Core::V4::UltraLowLatencyEngine* engine_;
    
    // Server infrastructure
    std::atomic<bool> running_{false};
    std::vector<std::thread> worker_threads_;
    int server_fd_{-1};
    
    // Performance tracking
    std::atomic<uint64_t> total_requests_{0};
    std::atomic<double> avg_response_time_ms_{0.0};
    
    // v4.0 optimizations
    std::map<std::string, std::function<std::string(const std::string&)>> handlers_;
    std::vector<char> response_buffer_; // Pre-allocated response buffer

public:
    /**
     * @brief Constructor
     */
    V4RestServer(const Config& config, Core::V4::UltraLowLatencyEngine* engine);
    
    /**
     * @brief Destructor
     */
    ~V4RestServer();
    
    /**
     * @brief Initialize the server
     */
    bool initialize();
    
    /**
     * @brief Start the server
     */
    void start();
    
    /**
     * @brief Stop the server
     */
    void stop();
    
    /**
     * @brief Get server metrics
     */
    struct ServerMetrics {
        uint64_t total_requests;
        double avg_response_time_ms;
        int active_connections;
        double requests_per_second;
    };
    
    ServerMetrics get_metrics() const;

private:
    /**
     * @brief Setup v4.0 API endpoints
     */
    setup_v4_endpoints();
    
    /**
     * @brief Worker thread for request processing
     */
    void worker_thread(int thread_id);
    
    /**
     * @brief Handle HTTP request
     */
    std::string handle_request(const std::string& method, const std::string& path, const std::string& body);
    
    /**
     * @brief v4.0 API endpoint handlers
     */
    std::string handle_status(const std::string& body);
    std::string handle_health(const std::string& body);
    std::string handle_signal(const std::string& body);
    std::string handle_batch_signal(const std::string& body);
    std::string handle_performance(const std::string& body);
    std::string handle_models(const std::string& body);
    std::string handle_welcome(const std::string& body);
    
    /**
     * @brief Parse JSON request body
     */
    std::map<std::string, std::string> parse_json(const std::string& json);
    
    /**
     * @brief Generate JSON response
     */
    std::string generate_json_response(const std::map<std::string, std::string>& data);
    
    /**
     * @brief Setup server socket with v4.0 optimizations
     */
    bool setup_server_socket();
};

} // namespace V4
} // namespace API
} // namespace ArchNeuronX
