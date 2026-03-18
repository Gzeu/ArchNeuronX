#include "complete_trading_system.hpp"
#include <iostream>
#include <memory>
#include <signal.h>
#include <thread>
#include <chrono>

using namespace archneuronx::core;

// Global system instance for signal handling
std::unique_ptr<CompleteTradingSystem> g_system;
std::atomic<bool> g_shutdown_requested(false);

// Signal handler for graceful shutdown
void signal_handler(int signal) {
    std::cout << "\n🛑 Received signal " << signal << ", initiating graceful shutdown..." << std::endl;
    g_shutdown_requested = true;
    
    if (g_system) {
        g_system->stop();
    }
}

// Print system banner
void print_system_banner() {
    std::cout << R"(
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║           🚀 ARCHNEURONX v4.0 - COMPLETE TRADING SYSTEM      ║
    ║                                                          ║
    ║  🧠 Quantum Neural Networks    🤖 Quantum Trading Agents   ║
    ║  🤖 HuggingFace Integration    🌐 Web Interface          ║
    ║  🤝 Multi-Agent Coordination  📊 Real-time Monitoring     ║
    ║  ⚡ Ultra-Low Latency          🛡️ Risk Management         ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════════╝
)" << std::endl;
}

// Print system status
void print_system_status(const CompleteTradingSystem::SystemStatus& status) {
    std::cout << "\n📊 SYSTEM STATUS:" << std::endl;
    std::cout << "   Status: " << status.status << std::endl;
    std::cout << "   Performance: " << std::fixed << std::setprecision(2) << (status.performance_metric * 100) << "%" << std::endl;
    std::cout << "   Quantum Coherence: " << std::setprecision(3) << status.quantum_coherence << std::endl;
    std::cout << "   Active Agents: " << status.active_agents << std::endl;
    std::cout << "   Total Trades: " << status.total_trades << std::endl;
    std::cout << "   Total P&L: $" << std::setprecision(2) << status.total_pnl << std::endl;
    std::cout << "   Win Rate: " << std::setprecision(1) << (status.win_rate * 100) << "%" << std::endl;
    
    std::cout << "\n🔧 COMPONENTS STATUS:" << std::endl;
    std::cout << "   Quantum Neural Networks: " << (status.quantum_neural_networks_active ? "✅ Active" : "❌ Inactive") << std::endl;
    std::cout << "   Quantum Agents: " << (status.quantum_agents_active ? "✅ Active" : "❌ Inactive") << std::endl;
    std::cout << "   LLM Integration: " << (status.llm_integration_active ? "✅ Active" : "❌ Inactive") << std::endl;
    std::cout << "   Web Interface: " << (status.web_interface_active ? "✅ Active" : "❌ Inactive") << std::endl;
    std::cout << "   Multi-Agent Coordination: " << (status.multi_agent_coordination_active ? "✅ Active" : "❌ Inactive") << std::endl;
}

// Print performance metrics
void print_performance_metrics(const std::map<std::string, double>& metrics) {
    std::cout << "\n📈 PERFORMANCE METRICS:" << std::endl;
    
    for (const auto& [name, value] : metrics) {
        std::cout << "   " << name << ": " << std::fixed << std::setprecision(3) << value << std::endl;
    }
}

// Interactive command handler
void handle_interactive_commands(CompleteTradingSystem& system) {
    std::string command;
    
    while (!g_shutdown_requested) {
        std::cout << "\n🎯 Enter command (help for available commands): ";
        std::getline(std::cin, command);
        
        if (command == "help") {
            std::cout << "\n📋 AVAILABLE COMMANDS:" << std::endl;
            std::cout << "   status          - Show system status" << std::endl;
            std::cout << "   metrics         - Show performance metrics" << std::endl;
            std::cout << "   trade           - Execute single trading cycle" << std::endl;
            std::cout << "   continuous      - Start continuous trading" << std::endl;
            std::cout << "   stop            - Stop trading" << std::endl;
            std::cout << "   add_agent       - Add new agent" << std::endl;
            std::cout << "   remove_agent    - Remove agent" << std::endl;
            std::cout << "   coordinate      - Coordinate all agents" << std::endl;
            std::cout << "   switch_llm      - Switch LLM model" << std::endl;
            std::cout << "   optimize        - Optimize performance" << std::endl;
            std::cout << "   emergency       - Emergency stop" << std::endl;
            std::cout << "   exit            - Exit system" << std::endl;
            
        } else if (command == "status") {
            auto status = system.get_system_status();
            print_system_status(status);
            
        } else if (command == "metrics") {
            auto metrics = system.get_performance_metrics();
            print_performance_metrics(metrics);
            
        } else if (command == "trade") {
            std::cout << "🔄 Executing trading cycle..." << std::endl;
            system.run_trading_session();
            
        } else if (command == "continuous") {
            std::cout << "🔄 Starting continuous trading..." << std::endl;
            system.run_continuous_trading();
            
        } else if (command == "stop") {
            std::cout << "🛑 Stopping trading..." << std::endl;
            system.stop();
            
        } else if (command == "add_agent") {
            std::string agent_id;
            std::cout << "Enter agent ID: ";
            std::getline(std::cin, agent_id);
            
            if (!agent_id.empty()) {
                system.add_agent(agent_id);
                std::cout << "✅ Agent added: " << agent_id << std::endl;
            }
            
        } else if (command == "remove_agent") {
            std::string agent_id;
            std::cout << "Enter agent ID: ";
            std::getline(std::cin, agent_id);
            
            if (!agent_id.empty()) {
                system.remove_agent(agent_id);
                std::cout << "✅ Agent removed: " << agent_id << std::endl;
            }
            
        } else if (command == "coordinate") {
            std::cout << "🤝 Coordinating all agents..." << std::endl;
            system.coordinate_all_agents();
            
        } else if (command == "switch_llm") {
            std::string model_name;
            std::cout << "Enter model name: ";
            std::getline(std::cin, model_name);
            
            if (!model_name.empty()) {
                system.switch_llm_model(model_name);
                std::cout << "✅ LLM model switched to: " << model_name << std::endl;
            }
            
        } else if (command == "optimize") {
            std::cout << "⚡ Optimizing system performance..." << std::endl;
            system.optimize_system_performance();
            
        } else if (command == "emergency") {
            std::cout << "🚨 Emergency stop initiated..." << std::endl;
            system.emergency_stop();
            break;
            
        } else if (command == "exit") {
            std::cout << "👋 Exiting system..." << std::endl;
            g_shutdown_requested = true;
            break;
            
        } else if (!command.empty()) {
            std::cout << "❌ Unknown command: " << command << std::endl;
            std::cout << "Type 'help' for available commands" << std::endl;
        }
    }
}

// Main application
int main(int argc, char* argv[]) {
    try {
        // Print system banner
        print_system_banner();
        
        // Setup signal handlers
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        
        // Configure system
        CompleteTradingSystem::SystemConfig config;
        config.system_name = "ArchNeuronX v4.0";
        config.version = "4.0.0";
        config.enable_quantum_neural_networks = true;
        config.enable_quantum_agents = true;
        config.enable_llm_integration = true;
        config.enable_web_interface = true;
        config.enable_multi_agent_coordination = true;
        
        // Quantum configuration
        config.quantum_heads = 16;
        config.quantum_layers = 6;
        config.quantum_states = 8;
        config.quantum_coherence_threshold = 0.8;
        
        // Agent configuration
        config.num_agents = 5;
        config.agent_learning_rate = 0.001;
        config.agent_exploration_rate = 0.1;
        config.agent_memory_size = 10000;
        
        // LLM configuration
        config.llm_provider = "huggingface";
        config.llm_model = "mistralai/Mistral-7B-v0.1";
        config.llm_confidence_threshold = 0.8;
        config.enable_llm_enhancement = true;
        
        // Web interface configuration
        config.http_port = 8080;
        config.websocket_port = 3001;
        config.update_interval_ms = 1000;
        
        // Trading configuration
        config.num_assets = 50;
        config.max_position_size = 0.1;
        config.risk_tolerance = 0.05;
        config.portfolio_rebalance_threshold = 0.05;
        
        // Performance configuration
        config.enable_gpu_acceleration = true;
        config.enable_flash_attention = true;
        config.enable_model_caching = true;
        config.max_concurrent_requests = 100;
        
        // Create system
        std::cout << "🔧 Creating Complete Trading System..." << std::endl;
        g_system = std::make_unique<CompleteTradingSystem>(config);
        
        // Initialize system
        std::cout << "🚀 Initializing system..." << std::endl;
        if (!g_system->initialize()) {
            std::cerr << "❌ Failed to initialize system" << std::endl;
            return 1;
        }
        
        // Start system
        std::cout << "🚀 Starting system..." << std::endl;
        g_system->start();
        
        // Show initial status
        auto status = g_system->get_system_status();
        print_system_status(status);
        
        // Show web interface information
        if (config.enable_web_interface) {
            std::cout << "\n🌐 WEB INTERFACE:" << std::endl;
            std::cout << "   HTTP Server: http://localhost:" << config.http_port << std::endl;
            std::cout << "   WebSocket: ws://localhost:" << config.websocket_port << std::endl;
            std::cout << "   API Endpoints: http://localhost:" << config.http_port << "/api/v4/" << std::endl;
            std::cout << "   Dashboard: http://localhost:" << config.http_port << "/dashboard" << std::endl;
        }
        
        // Show system information
        std::cout << "\n📋 SYSTEM INFORMATION:" << std::endl;
        std::cout << g_system->get_system_info() << std::endl;
        
        // Interactive mode or continuous mode
        if (argc > 1 && std::string(argv[1]) == "--continuous") {
            std::cout << "\n🔄 Running in continuous mode..." << std::endl;
            std::cout << "Press Ctrl+C to stop" << std::endl;
            
            // Run continuous trading
            g_system->run_continuous_trading();
            
        } else {
            std::cout << "\n🎯 Running in interactive mode..." << std::endl;
            std::cout << "Type 'help' for available commands" << std::endl;
            
            // Interactive mode
            handle_interactive_commands(*g_system);
        }
        
        // Shutdown
        std::cout << "\n🛑 Shutting down system..." << std::endl;
        g_system->shutdown();
        
        // Final status
        auto final_status = g_system->get_system_status();
        print_system_status(final_status);
        
        std::cout << "\n✨ ArchNeuronX v4.0 Complete Trading System shutdown complete!" << std::endl;
        std::cout << "🚀 Thank you for using ArchNeuronX!" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Fatal error: " << e.what() << std::endl;
        
        if (g_system) {
            g_system->emergency_stop();
            g_system->shutdown();
        }
        
        return 1;
    }
}
