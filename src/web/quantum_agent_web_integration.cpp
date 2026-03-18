#include "quantum_agent_web_integration.hpp"
#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>
#include <algorithm>

namespace archneuronx {
namespace web {

// ============================================================================
// Quantum Agent Web Integration Implementation
// ============================================================================

QuantumAgentWebIntegration::QuantumAgentWebIntegration(const WebIntegrationConfig& config)
    : config_(config), running_(false) {
    
    // Initialize HTTP server
    http_server_ = std::make_unique<HttpServer>(config_.port);
    
    // Initialize WebSocket server
    websocket_server_ = std::make_unique<WebSocketServer>(config_.websocket_port);
    
    // Initialize system status
    system_status_.total_agents = 0;
    system_status_.active_agents = 0;
    system_status_.system_performance = 0.0;
    system_status_.quantum_coordination = 0.0;
    system_status_.last_update = std::chrono::system_clock::now();
    
    std::cout << "🌐 Quantum Agent Web Integration initialized" << std::endl;
    std::cout << "   HTTP Port: " << config_.port << std::endl;
    std::cout << "   WebSocket Port: " << config_.websocket_port << std::endl;
    std::cout << "   Update Interval: " << config_.update_interval_ms << "ms" << std::endl;
}

QuantumAgentWebIntegration::~QuantumAgentWebIntegration() {
    stop_web_server();
}

void QuantumAgentWebIntegration::initialize() {
    std::cout << "🔧 Initializing Web Integration..." << std::endl;
    
    // Setup API endpoints
    setup_api_endpoints();
    
    // Setup WebSocket handlers
    setup_websocket_handlers();
    
    std::cout << "✅ Web Integration initialized successfully!" << std::endl;
}

void QuantumAgentWebIntegration::start_web_server() {
    std::cout << "🚀 Starting Web Servers..." << std::endl;
    
    // Start HTTP server
    http_server_->start();
    
    // Start WebSocket server
    websocket_server_->start();
    
    // Start real-time updates
    if (config_.enable_real_time_updates) {
        start_real_time_updates();
    }
    
    running_ = true;
    
    std::cout << "✅ Web Servers started successfully!" << std::endl;
    std::cout << "   HTTP Server: http://localhost:" << config_.port << std::endl;
    std::cout << "   WebSocket Server: ws://localhost:" << config_.websocket_port << std::endl;
}

void QuantumAgentWebIntegration::stop_web_server() {
    if (running_) {
        std::cout << "🛑 Stopping Web Servers..." << std::endl;
        
        running_ = false;
        
        // Stop background updates
        stop_background_updates();
        
        // Stop servers
        http_server_->stop();
        websocket_server_->stop();
        
        std::cout << "✅ Web Servers stopped successfully!" << std::endl;
    }
}

void QuantumAgentWebIntegration::register_agent(
    std::shared_ptr<agents::QuantumTradingAgent> agent, 
    const std::string& agent_id) {
    
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    agents_[agent_id] = agent;
    
    // Initialize agent status
    AgentWebStatus status;
    status.agent_id = agent_id;
    status.status = "idle";
    status.performance_metric = 0.0;
    status.quantum_coherence = 1.0;
    status.total_actions = 0;
    status.win_rate = 0.0;
    status.current_action = "HOLD";
    status.confidence = 0.0;
    status.last_update = std::chrono::system_clock::now();
    
    agent_status_[agent_id] = status;
    
    // Update system status
    system_status_.total_agents = agents_.size();
    system_status_.last_update = std::chrono::system_clock::now();
    
    std::cout << "🤖 Agent registered: " << agent_id << std::endl;
    std::cout << "   Total agents: " << system_status_.total_agents << std::endl;
}

void QuantumAgentWebIntegration::unregister_agent(const std::string& agent_id) {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    agents_.erase(agent_id);
    agent_status_.erase(agent_id);
    
    // Update system status
    system_status_.total_agents = agents_.size();
    system_status_.last_update = std::chrono::system_clock::now();
    
    std::cout << "🤖 Agent unregistered: " << agent_id << std::endl;
}

void QuantumAgentWebIntegration::integrate_with_web_interface() {
    std::cout << "🔗 Integrating with Web Interface..." << std::endl;
    
    // Start web servers
    start_web_server();
    
    // Setup real-time updates
    if (config_.enable_real_time_updates) {
        start_background_updates();
    }
    
    std::cout << "✅ Web Interface Integration completed!" << std::endl;
    std::cout << "   API Endpoints: http://localhost:" << config_.port << "/api/v4/quantum/" << std::endl;
    std::cout << "   WebSocket: ws://localhost:" << config_.websocket_port << "/quantum/" << std::endl;
}

void QuantumAgentWebIntegration::setup_api_endpoints() {
    std::cout << "📡 Setting up API Endpoints..." << std::endl;
    
    // Create web controller
    auto controller = std::make_shared<QuantumAgentWebController>(
        std::shared_ptr<QuantumAgentWebIntegration>(this)
    );
    
    // Setup HTTP routes
    http_server_->add_route("/api/v4/quantum/agents", 
        [controller](const std::string& path) { return controller->get_agents(); });
    
    http_server_->add_route("/api/v4/quantum/agents/status", 
        [controller](const std::string& path) { return controller->get_system_status(); });
    
    http_server_->add_route("/api/v4/quantum/agents/performance", 
        [controller](const std::string& path) { return controller->get_performance_report(); });
    
    // Dynamic agent routes
    http_server_->add_route("/api/v4/quantum/agents/", 
        [controller](const std::string& path) { 
            // Extract agent ID from path
            if (path.find("/status") != std::string::npos) {
                size_t start = path.find("/agents/") + 9;
                size_t end = path.find("/status");
                std::string agent_id = path.substr(start, end - start);
                return controller->get_agent_status(agent_id);
            }
            return "{}";
        });
    
    std::cout << "   GET /api/v4/quantum/agents - List all agents" << std::endl;
    std::cout << "   GET /api/v4/quantum/agents/status - System status" << std::endl;
    std::cout << "   GET /api/v4/quantum/agents/{id}/status - Agent status" << std::endl;
    std::cout << "   GET /api/v4/quantum/agents/performance - Performance report" << std::endl;
}

void QuantumAgentWebIntegration::setup_websocket_handlers() {
    std::cout << "🔌 Setting up WebSocket Handlers..." << std::endl;
    
    // Create WebSocket handler
    auto handler = std::make_shared<QuantumAgentWebSocketHandler>(
        std::shared_ptr<QuantumAgentWebIntegration>(this)
    );
    
    // Set message handler
    websocket_server_->set_message_handler(
        [handler](const std::string& client_id, const std::string& message) {
            handler->handle_message(client_id, message);
        }
    );
    
    std::cout << "   WebSocket handlers configured" << std::endl;
    std::cout << "   Real-time updates enabled" << std::endl;
}

void QuantumAgentWebIntegration::start_real_time_updates() {
    std::cout << "⚡ Starting Real-time Updates..." << std::endl;
    
    start_background_updates();
    
    std::cout << "   Update interval: " << config_.update_interval_ms << "ms" << std::endl;
}

QuantumAgentWebIntegration::SystemWebStatus QuantumAgentWebIntegration::get_system_status() const {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    return system_status_;
}

QuantumAgentWebIntegration::AgentWebStatus QuantumAgentWebIntegration::get_agent_status(
    const std::string& agent_id) const {
    
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    auto it = agent_status_.find(agent_id);
    if (it != agent_status_.end()) {
        return it->second;
    }
    
    // Return empty status if agent not found
    AgentWebStatus empty_status;
    empty_status.agent_id = agent_id;
    empty_status.status = "not_found";
    return empty_status;
}

std::vector<QuantumAgentWebIntegration::AgentWebStatus> 
QuantumAgentWebIntegration::get_all_agent_status() const {
    
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    std::vector<AgentWebStatus> all_status;
    for (const auto& pair : agent_status_) {
        all_status.push_back(pair.second);
    }
    
    return all_status;
}

void QuantumAgentWebIntegration::start_agent_training(const std::string& agent_id) {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    auto it = agents_.find(agent_id);
    if (it != agents_.end()) {
        // Start training (simplified)
        update_agent_status_internal(agent_id);
        
        std::cout << "🎓 Started training for agent: " << agent_id << std::endl;
    }
}

void QuantumAgentWebIntegration::stop_agent_training(const std::string& agent_id) {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    auto it = agents_.find(agent_id);
    if (it != agents_.end()) {
        // Stop training (simplified)
        update_agent_status_internal(agent_id);
        
        std::cout << "🛑 Stopped training for agent: " << agent_id << std::endl;
    }
}

void QuantumAgentWebIntegration::reset_agent(const std::string& agent_id) {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    auto it = agents_.find(agent_id);
    if (it != agents_.end()) {
        it->second->reset();
        update_agent_status_internal(agent_id);
        
        std::cout << "🔄 Reset agent: " << agent_id << std::endl;
    }
}

void QuantumAgentWebIntegration::coordinate_agents() {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    // Coordinate all agents (simplified)
    update_system_status_internal();
    
    std::cout << "🤝 Coordinated " << agents_.size() << " agents" << std::endl;
}

void QuantumAgentWebIntegration::broadcast_agent_update(const AgentWebStatus& status) {
    std::string message = agent_status_to_json(status);
    websocket_server_->broadcast_message(message);
}

void QuantumAgentWebIntegration::broadcast_system_update(const SystemWebStatus& status) {
    std::string message = system_status_to_json(status);
    websocket_server_->broadcast_message(message);
}

void QuantumAgentWebIntegration::send_agent_command(
    const std::string& agent_id, 
    const std::string& command) {
    
    std::string message = R"({"type": "command", "agent_id": ")" + agent_id + 
                      R"(", "command": ")" + command + R"("})";
    
    websocket_server_->broadcast_message(message);
    
    std::cout << "📤 Sent command to agent " << agent_id << ": " << command << std::endl;
}

void QuantumAgentWebIntegration::update_agent_performance(const std::string& agent_id) {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    auto it = agents_.find(agent_id);
    if (it != agents_.end()) {
        update_agent_status_internal(agent_id);
        
        // Broadcast update
        auto status = get_agent_status(agent_id);
        broadcast_agent_update(status);
    }
}

void QuantumAgentWebIntegration::update_system_metrics() {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    update_system_status_internal();
    
    // Broadcast update
    broadcast_system_update(system_status_);
}

void QuantumAgentWebIntegration::generate_performance_report() {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    
    std::string report = performance_report_to_json();
    websocket_server_->broadcast_message(report);
    
    std::cout << "📊 Performance report generated and broadcasted" << std::endl;
}

void QuantumAgentWebIntegration::update_agent_status_internal(const std::string& agent_id) {
    auto it = agents_.find(agent_id);
    if (it != agents_.end()) {
        auto& agent = it->second;
        auto& status = agent_status_[agent_id];
        
        // Update status from agent
        status.performance_metric = agent->get_performance_metric();
        status.quantum_coherence = agent->get_quantum_coherence();
        status.total_actions = agent->get_total_actions();
        status.win_rate = agent->get_win_rate();
        status.last_update = std::chrono::system_clock::now();
        
        // Update status based on agent state
        if (status.performance_metric > 0.1) {
            status.status = "active";
        } else if (status.total_actions > 0) {
            status.status = "training";
        } else {
            status.status = "idle";
        }
    }
}

void QuantumAgentWebIntegration::update_system_status_internal() {
    // Count active agents
    system_status_.active_agents = 0;
    double total_performance = 0.0;
    double total_coherence = 0.0;
    
    for (const auto& pair : agent_status_) {
        const auto& status = pair.second;
        if (status.status == "active") {
            system_status_.active_agents++;
        }
        total_performance += status.performance_metric;
        total_coherence += status.quantum_coherence;
    }
    
    // Calculate averages
    if (agent_status_.size() > 0) {
        system_status_.system_performance = total_performance / agent_status_.size();
        system_status_.quantum_coordination = total_coherence / agent_status_.size();
    }
    
    system_status_.last_update = std::chrono::system_clock::now();
}

void QuantumAgentWebIntegration::start_background_updates() {
    running_ = true;
    
    update_thread_ = std::thread([this]() {
        while (running_) {
            // Update all agent statuses
            for (const auto& pair : agents_) {
                update_agent_performance(pair.first);
            }
            
            // Update system metrics
            update_system_metrics();
            
            // Sleep for update interval
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.update_interval_ms));
        }
    });
    
    std::cout << "⚡ Background updates started" << std::endl;
}

void QuantumAgentWebIntegration::stop_background_updates() {
    if (update_thread_.joinable()) {
        running_ = false;
        update_thread_.join();
        std::cout << "⚡ Background updates stopped" << std::endl;
    }
}

std::string QuantumAgentWebIntegration::agent_status_to_json(const AgentWebStatus& status) const {
    std::ostringstream json;
    
    json << R"({"type": "agent_update", "data": {)";
    json << R"("agent_id": ")" << status.agent_id << R"(",)";
    json << R"("status": ")" << status.status << R"(",)";
    json << R"("performance_metric": )" << status.performance_metric << R"(,)";
    json << R"("quantum_coherence": )" << status.quantum_coherence << R"(,)";
    json << R"("total_actions": )" << status.total_actions << R"(,)";
    json << R"("win_rate": )" << status.win_rate << R"(,)";
    json << R"("current_action": ")" << status.current_action << R"(",)";
    json << R"("confidence": )" << status.confidence << R"(,)";
    
    // Add timestamp
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        status.last_update.time_since_epoch()).count();
    json << R"("timestamp": )" << timestamp;
    
    json << R"(}})";
    
    return json.str();
}

std::string QuantumAgentWebIntegration::system_status_to_json(const SystemWebStatus& status) const {
    std::ostringstream json;
    
    json << R"({"type": "system_update", "data": {)";
    json << R"("total_agents": )" << status.total_agents << R"(,)";
    json << R"("active_agents": )" << status.active_agents << R"(,)";
    json << R"("system_performance": )" << status.system_performance << R"(,)";
    json << R"("quantum_coordination": )" << status.quantum_coordination << R"(,)";
    
    // Add agents array
    json << R"("agents": [)";
    bool first = true;
    for (const auto& agent_status : status.agents) {
        if (!first) json << ",";
        json << R"({"agent_id": ")" << agent_status.agent_id << R"(",)";
        json << R"("status": ")" << agent_status.status << R"(",)";
        json << R"("performance": )" << agent_status.performance_metric;
        json << R"(})";
        first = false;
    }
    json << R"(],)";
    
    // Add timestamp
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        status.last_update.time_since_epoch()).count();
    json << R"("timestamp": )" << timestamp;
    
    json << R"(}})";
    
    return json.str();
}

std::string QuantumAgentWebIntegration::performance_report_to_json() const {
    std::ostringstream json;
    
    json << R"({"type": "performance_report", "data": {)";
    json << R"("total_agents": )" << system_status_.total_agents << R"(,)";
    json << R"("active_agents": )" << system_status_.active_agents << R"(,)";
    json << R"("system_performance": )" << system_status_.system_performance << R"(,)";
    json << R"("quantum_coordination": )" << system_status_.quantum_coordination << R"(,)";
    
    // Add performance history
    json << R"("performance_history": [)";
    bool first = true;
    for (double perf : performance_history_) {
        if (!first) json << ",";
        json << perf;
        first = false;
    }
    json << R"(],)";
    
    // Add coherence history
    json << R"("coherence_history": [)";
    first = true;
    for (double coherence : coherence_history_) {
        if (!first) json << ",";
        json << coherence;
        first = false;
    }
    json << R"(],)";
    
    // Add timestamp
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    json << R"("timestamp": )" << timestamp;
    
    json << R"(}})";
    
    return json.str();
}

// ============================================================================
// HTTP Server Implementation
// ============================================================================

HttpServer::HttpServer(int port) : port_(port), running_(false) {
}

void HttpServer::start() {
    running_ = true;
    server_thread_ = std::thread(&HttpServer::server_loop, this);
    
    std::cout << "🌐 HTTP Server started on port " << port_ << std::endl;
}

void HttpServer::stop() {
    running_ = false;
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    
    std::cout << "🛑 HTTP Server stopped" << std::endl;
}

void HttpServer::add_route(const std::string& path, 
    std::function<std::string(const std::string&)> handler) {
    
    routes_[path] = handler;
}

void HttpServer::server_loop() {
    // Simplified HTTP server implementation
    // In a real implementation, this would handle actual HTTP requests
    
    while (running_) {
        // Simulate handling requests
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// ============================================================================
// WebSocket Server Implementation
// ============================================================================

WebSocketServer::WebSocketServer(int port) : port_(port), running_(false) {
}

void WebSocketServer::start() {
    running_ = true;
    server_thread_ = std::thread(&WebSocketServer::server_loop, this);
    
    std::cout << "🔌 WebSocket Server started on port " << port_ << std::endl;
}

void WebSocketServer::stop() {
    running_ = false;
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    
    std::cout << "🛑 WebSocket Server stopped" << std::endl;
}

void WebSocketServer::broadcast_message(const std::string& message) {
    // Broadcast message to all connected clients
    for (const auto& client : clients_) {
        send_message(client.first, message);
    }
}

void WebSocketServer::send_message(const std::string& client_id, const std::string& message) {
    // Send message to specific client
    std::cout << "📤 Sending message to client " << client_id << ": " << message << std::endl;
}

void WebSocketServer::set_message_handler(
    std::function<void(const std::string&, const std::string&)> handler) {
    
    message_handler_ = handler;
}

void WebSocketServer::server_loop() {
    // Simplified WebSocket server implementation
    // In a real implementation, this would handle actual WebSocket connections
    
    while (running_) {
        // Simulate handling WebSocket messages
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// ============================================================================
// Quantum Agent Web Controller Implementation
// ============================================================================

QuantumAgentWebController::QuantumAgentWebController(
    std::shared_ptr<QuantumAgentWebIntegration> integration)
    : integration_(integration) {
}

std::string QuantumAgentWebController::get_agents() {
    auto all_status = integration_->get_all_agent_status();
    
    std::ostringstream json;
    json << R"({"status": "success", "data": {)";
    json << R"("agents": [)";
    
    bool first = true;
    for (const auto& status : all_status) {
        if (!first) json << ",";
        json << R"({"id": ")" << status.agent_id << R"(",)";
        json << R"("status": ")" << status.status << R"(",)";
        json << R"("performance": )" << status.performance_metric << R"(,)";
        json << R"("coherence": )" << status.quantum_coherence;
        json << R"(})";
        first = false;
    }
    
    json << R"(]}})";
    
    return json.str();
}

std::string QuantumAgentWebController::get_agent_status(const std::string& agent_id) {
    auto status = integration_->get_agent_status(agent_id);
    
    if (status.status == "not_found") {
        return create_error_response("Agent not found: " + agent_id);
    }
    
    std::ostringstream json;
    json << R"({"status": "success", "data": {)";
    json << R"("agent_id": ")" << status.agent_id << R"(",)";
    json << R"("status": ")" << status.status << R"(",)";
    json << R"("performance": )" << status.performance_metric << R"(,)";
    json << R"("coherence": )" << status.quantum_coherence << R"(,)";
    json << R"("total_actions": )" << status.total_actions << R"(,)";
    json << R"("win_rate": )" << status.win_rate << R"(,)";
    json << R"("current_action": ")" << status.current_action << R"(",)";
    json << R"("confidence": )" << status.confidence;
    json << R"(}})";
    
    return json.str();
}

std::string QuantumAgentWebController::get_system_status() {
    auto status = integration_->get_system_status();
    
    std::ostringstream json;
    json << R"({"status": "success", "data": {)";
    json << R"("total_agents": )" << status.total_agents << R"(,)";
    json << R"("active_agents": )" << status.active_agents << R"(,)";
    json << R"("system_performance": )" << status.system_performance << R"(,)";
    json << R"("quantum_coordination": )" << status.quantum_coordination;
    json << R"(}})";
    
    return json.str();
}

std::string QuantumAgentWebController::get_performance_report() {
    // Generate performance report
    std::ostringstream json;
    json << R"({"status": "success", "data": {)";
    json << R"("report_type": "performance",)";
    json << R"("generated_at": ")" << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() << R"(",)";
    json << R"("message": "Performance report generated successfully")";
    json << R"(}})";
    
    return json.str();
}

std::string QuantumAgentWebController::start_agent_training(const std::string& agent_id) {
    integration_->start_agent_training(agent_id);
    
    return create_json_response("success", 
        R"({"message": "Training started for agent: )" + agent_id + R"("})");
}

std::string QuantumAgentWebController::stop_agent_training(const std::string& agent_id) {
    integration_->stop_agent_training(agent_id);
    
    return create_json_response("success", 
        R"({"message": "Training stopped for agent: )" + agent_id + R"("})");
}

std::string QuantumAgentWebController::reset_agent(const std::string& agent_id) {
    integration_->reset_agent(agent_id);
    
    return create_json_response("success", 
        R"({"message": "Agent reset: )" + agent_id + R"("})");
}

std::string QuantumAgentWebController::coordinate_all_agents() {
    integration_->coordinate_agents();
    
    return create_json_response("success", 
        R"({"message": "All agents coordinated successfully"})");
}

std::string QuantumAgentWebController::create_json_response(
    const std::string& status, 
    const std::string& data) {
    
    std::ostringstream json;
    json << R"({"status": ")" << status << R"(", "data": )" << data << R"(})";
    return json.str();
}

std::string QuantumAgentWebController::create_error_response(const std::string& error) {
    return create_json_response("error", R"({"error": ")" + error + R"("})");
}

// ============================================================================
// WebSocket Message Handler Implementation
// ============================================================================

QuantumAgentWebSocketHandler::QuantumAgentWebSocketHandler(
    std::shared_ptr<QuantumAgentWebIntegration> integration)
    : integration_(integration) {
}

void QuantumAgentWebSocketHandler::handle_message(const std::string& client_id, const std::string& message) {
    // Parse message (simplified)
    if (message.find("command") != std::string::npos) {
        handle_agent_command(client_id, message);
    } else if (message.find("subscribe") != std::string::npos) {
        handle_subscription_request(client_id, message);
    } else if (message.find("unsubscribe") != std::string::npos) {
        handle_unsubscription_request(client_id, message);
    }
}

void QuantumAgentWebSocketHandler::handle_connection(const std::string& client_id) {
    std::cout << "🔌 WebSocket client connected: " << client_id << std::endl;
    
    // Send initial status
    auto status = integration_->get_system_status();
    auto response = create_websocket_response("system_status", 
        integration_->system_status_to_json(status));
    
    // Send response (simplified)
    std::cout << "📤 Sent initial status to client: " << client_id << std::endl;
}

void QuantumAgentWebSocketHandler::handle_disconnection(const std::string& client_id) {
    std::cout << "🔌 WebSocket client disconnected: " << client_id << std::endl;
}

void QuantumAgentWebSocketHandler::handle_agent_command(const std::string& client_id, const std::string& command) {
    std::cout << "📤 Received command from " << client_id << ": " << command << std::endl;
    
    // Parse and execute command (simplified)
    if (command.find("start_training") != std::string::npos) {
        // Extract agent ID and start training
        // This is simplified - in practice would parse JSON
        integration_->coordinate_all_agents();
    }
}

void QuantumAgentWebSocketHandler::handle_subscription_request(const std::string& client_id, const std::string& subscription) {
    std::cout << "📥 Client " << client_id << " subscribed to: " << subscription << std::endl;
}

void QuantumAgentWebSocketHandler::handle_unsubscription_request(const std::string& client_id, const std::string& subscription) {
    std::cout << "📤 Client " << client_id << " unsubscribed from: " << subscription << std::endl;
}

std::string QuantumAgentWebSocketHandler::create_websocket_response(
    const std::string& type, 
    const std::string& data) {
    
    std::ostringstream json;
    json << R"({"type": ")" << type << R"(", "data": )" << data << R"(})";
    return json.str();
}

} // namespace web
} // namespace archneuronx
