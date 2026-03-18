#pragma once

#include "../agents/quantum_trading_agent.hpp"
#include "../models/quantum_trading_signals.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>

namespace archneuronx {
namespace web {

/**
 * Quantum Agent Web Integration
 * 
 * This class provides the bridge between quantum trading agents
 * and the web interface, enabling real-time monitoring and control.
 */
class QuantumAgentWebIntegration {
public:
    struct WebIntegrationConfig {
        int port = 8080;
        std::string host = "localhost";
        int websocket_port = 3001;
        int update_interval_ms = 1000;
        bool enable_real_time_updates = true;
        bool enable_agent_control = true;
        int max_concurrent_connections = 100;
    };

    struct AgentWebStatus {
        std::string agent_id;
        std::string status;  // "active", "training", "idle"
        double performance_metric;
        double quantum_coherence;
        int total_actions;
        double win_rate;
        std::string current_action;
        double confidence;
        std::chrono::system_clock::time_point last_update;
    };

    struct SystemWebStatus {
        std::vector<AgentWebStatus> agents;
        double system_performance;
        double quantum_coordination;
        int total_agents;
        int active_agents;
        std::chrono::system_clock::time_point last_update;
    };

public:
    explicit QuantumAgentWebIntegration(const WebIntegrationConfig& config);
    ~QuantumAgentWebIntegration();

    // Integration lifecycle
    void initialize();
    void start_web_server();
    void stop_web_server();
    
    // Agent registration
    void register_agent(std::shared_ptr<agents::QuantumTradingAgent> agent, const std::string& agent_id);
    void unregister_agent(const std::string& agent_id);
    
    // Web interface integration
    void integrate_with_web_interface();
    void setup_api_endpoints();
    void setup_websocket_handlers();
    void start_real_time_updates();
    
    // Data access
    SystemWebStatus get_system_status() const;
    AgentWebStatus get_agent_status(const std::string& agent_id) const;
    std::vector<AgentWebStatus> get_all_agent_status() const;
    
    // Agent control
    void start_agent_training(const std::string& agent_id);
    void stop_agent_training(const std::string& agent_id);
    void reset_agent(const std::string& agent_id);
    void coordinate_agents();
    
    // WebSocket communication
    void broadcast_agent_update(const AgentWebStatus& status);
    void broadcast_system_update(const SystemWebStatus& status);
    void send_agent_command(const std::string& agent_id, const std::string& command);
    
    // Performance monitoring
    void update_agent_performance(const std::string& agent_id);
    void update_system_metrics();
    void generate_performance_report();

private:
    WebIntegrationConfig config_;
    
    // Agent management
    std::map<std::string, std::shared_ptr<agents::QuantumTradingAgent>> agents_;
    std::map<std::string, AgentWebStatus> agent_status_;
    SystemWebStatus system_status_;
    
    // Web server components
    std::unique_ptr<class HttpServer> http_server_;
    std::unique_ptr<class WebSocketServer> websocket_server_;
    
    // Threading
    std::atomic<bool> running_;
    std::thread update_thread_;
    std::mutex agents_mutex_;
    
    // Performance tracking
    std::vector<double> performance_history_;
    std::vector<double> coherence_history_;
    std::chrono::system_clock::time_point last_performance_update_;
    
    // Private methods
    void setup_http_routes();
    void setup_websocket_routes();
    void update_agent_status_internal(const std::string& agent_id);
    void update_system_status_internal();
    void start_background_updates();
    void stop_background_updates();
    
    // JSON serialization
    std::string agent_status_to_json(const AgentWebStatus& status) const;
    std::string system_status_to_json(const SystemWebStatus& status) const;
    std::string performance_report_to_json() const;
    
    // API handlers
    std::string handle_get_agents(const std::string& path);
    std::string handle_get_agent_status(const std::string& path);
    std::string handle_get_system_status(const std::string& path);
    std::string handle_post_agent_command(const std::string& path, const std::string& body);
    std::string handle_get_performance_report(const std::string& path);
};

/**
 * HTTP Server for Quantum Agent Web Interface
 */
class HttpServer {
public:
    explicit HttpServer(int port);
    
    void start();
    void stop();
    void add_route(const std::string& path, std::function<std::string(const std::string&)> handler);
    
private:
    int port_;
    std::atomic<bool> running_;
    std::thread server_thread_;
    std::map<std::string, std::function<std::string(const std::string&)>> routes_;
    
    void server_loop();
    void handle_request(const std::string& request);
};

/**
 * WebSocket Server for Real-time Updates
 */
class WebSocketServer {
public:
    explicit WebSocketServer(int port);
    
    void start();
    void stop();
    void broadcast_message(const std::string& message);
    void send_message(const std::string& client_id, const std::string& message);
    
    void set_message_handler(std::function<void(const std::string&, const std::string&)> handler);
    
private:
    int port_;
    std::atomic<bool> running_;
    std::thread server_thread_;
    std::map<std::string, std::string> clients_;
    std::function<void(const std::string&, const std::string&)> message_handler_;
    
    void server_loop();
    void handle_websocket_connection(const std::string& client_id);
    void handle_websocket_message(const std::string& client_id, const std::string& message);
};

/**
 * Quantum Agent Web Controller
 * 
 * Provides REST API endpoints for controlling quantum agents
 * through the web interface.
 */
class QuantumAgentWebController {
public:
    explicit QuantumAgentWebController(std::shared_ptr<QuantumAgentWebIntegration> integration);
    
    // API endpoints
    std::string get_agents();
    std::string get_agent_status(const std::string& agent_id);
    std::string get_system_status();
    std::string get_performance_report();
    
    // Agent control
    std::string start_agent_training(const std::string& agent_id);
    std::string stop_agent_training(const std::string& agent_id);
    std::string reset_agent(const std::string& agent_id);
    std::string coordinate_all_agents();
    
    // Real-time data
    std::string get_real_time_updates();
    std::string get_quantum_metrics();
    std::string get_trading_signals();

private:
    std::shared_ptr<QuantumAgentWebIntegration> integration_;
    
    std::string create_json_response(const std::string& status, const std::string& data);
    std::string create_error_response(const std::string& error);
};

/**
 * WebSocket Message Handler
 * 
 * Handles WebSocket messages for real-time communication
 * between the web interface and quantum agents.
 */
class QuantumAgentWebSocketHandler {
public:
    explicit QuantumAgentWebSocketHandler(std::shared_ptr<QuantumAgentWebIntegration> integration);
    
    void handle_message(const std::string& client_id, const std::string& message);
    void handle_connection(const std::string& client_id);
    void handle_disconnection(const std::string& client_id);
    
private:
    std::shared_ptr<QuantumAgentWebIntegration> integration_;
    
    void handle_agent_command(const std::string& client_id, const std::string& command);
    void handle_subscription_request(const std::string& client_id, const std::string& subscription);
    void handle_unsubscription_request(const std::string& client_id, const std::string& subscription);
    
    std::string create_websocket_response(const std::string& type, const std::string& data);
};

} // namespace web
} // namespace archneuronx
