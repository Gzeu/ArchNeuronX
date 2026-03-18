#include "../agents/quantum_trading_agent.hpp"
#include "../models/quantum_trading_signals.hpp"
#include "../web/quantum_agent_web_integration.hpp"
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <thread>

using namespace archneuronx;

/**
 * Quantum Agent Integration Demo
 * 
 * This demonstrates the complete integration of quantum neural networks
 * with autonomous trading agents for real-time decision making.
 */

class QuantumTradingSystemIntegration {
public:
    struct IntegrationConfig {
        // Agent configuration
        int num_agents = 3;
        int input_features = 128;
        int hidden_dim = 256;
        int num_heads = 16;
        
        // Training configuration
        int training_episodes = 100;
        int steps_per_episode = 50;
        double learning_rate = 0.001;
        
        // Quantum configuration
        int quantum_states = 8;
        double quantum_coherence_threshold = 0.8;
        bool use_quantum_coordination = true;
        
        // Environment configuration
        int num_assets = 10;
        double market_volatility = 0.02;
        bool use_quantum_market = true;
    };

public:
    explicit QuantumTradingSystemIntegration(const IntegrationConfig& config);
    
    // System lifecycle
    void initialize();
    void run_training();
    void run_live_trading();
    void evaluate_performance();
    
    // Integration methods
    void integrate_agents_with_models();
    void setup_quantum_coordination();
    void configure_market_environment();
    
    // Performance monitoring
    void print_system_status();
    void print_quantum_metrics();
    void print_trading_performance();

private:
    IntegrationConfig config_;
    
    // System components
    std::vector<std::unique_ptr<agents::QuantumTradingAgent>> agents_;
    std::unique_ptr<agents::QuantumMultiAgentSystem> multi_agent_system_;
    std::unique_ptr<agents::QuantumTradingEnvironment> environment_;
    
    // Web integration
    std::unique_ptr<web::QuantumAgentWebIntegration> web_integration_;
    
    // Quantum models
    std::vector<std::unique_ptr<models::QuantumTradingSignals>> quantum_models_;
    
    // System state
    bool initialized_ = false;
    int current_episode_ = 0;
    double total_reward_ = 0.0;
    int total_steps_ = 0;
    
    // Performance metrics
    std::vector<double> episode_rewards_;
    std::vector<double> quantum_coherence_history_;
    std::vector<double> win_rates_;
    
    void initialize_agents();
    void initialize_environment();
    void initialize_web_integration();
    void train_single_episode();
    void execute_trading_step();
    void update_performance_metrics();
};

QuantumTradingSystemIntegration::QuantumTradingSystemIntegration(const IntegrationConfig& config)
    : config_(config) {
    
    std::cout << "🚀 Initializing Quantum Trading System Integration..." << std::endl;
    std::cout << "🤖 Number of Agents: " << config_.num_agents << std::endl;
    std::cout << "🧠 Quantum Heads: " << config_.num_heads << std::endl;
    std::cout << "   Quantum States: " << config_.quantum_states << std::endl;
    std::cout << "📊 Number of Assets: " << config_.num_assets << std::endl;
    std::cout << std::endl;
}

void QuantumTradingSystemIntegration::initialize() {
    if (!initialized_) {
        std::cout << "🔧 Setting up Quantum Trading System Integration..." << std::endl;
        
        // Initialize components
        initialize_agents();
        initialize_environment();
        initialize_web_integration();
        
        // Setup integration
        integrate_agents_with_models();
        setup_quantum_coordination();
        configure_market_environment();
        
        initialized_ = true;
        std::cout << "✅ Quantum Trading System Integration completed!" << std::endl;
        std::cout << std::endl;
    }
}

void QuantumTradingSystemIntegration::initialize_agents() {
    std::cout << "🤖 Initializing Quantum Trading Agents..." << std::endl;
    
    // Create individual agents
    for (int i = 0; i < config_.num_agents; ++i) {
        agents::QuantumTradingAgent::AgentConfig agent_config;
        agent_config.input_features = config_.input_features;
        agent_config.hidden_dim = config_.hidden_dim;
        agent_config.num_heads = config_.num_heads;
        agent_config.quantum_states = config_.quantum_states;
        agent_config.learning_rate = config_.learning_rate;
        agent_config.quantum_coherence_threshold = config_.quantum_coherence_threshold;
        
        auto agent = std::make_unique<agents::QuantumTradingAgent>(agent_config);
        agent->initialize();
        agents_.push_back(std::move(agent));
        
        std::cout << "   Agent " << (i + 1) << " initialized" << std::endl;
    }
    
    // Create multi-agent system
    agents::QuantumMultiAgentSystem::MultiAgentConfig multi_config;
    multi_config.num_agents = config_.num_agents;
    multi_config.use_quantum_coordination = config_.use_quantum_coordination;
    multi_config.quantum_communication_states = config_.quantum_states;
    
    multi_agent_system_ = std::make_unique<agents::QuantumMultiAgentSystem>(multi_config);
    multi_agent_system_->initialize();
    
    std::cout << "✅ Quantum Trading Agents initialized" << std::endl;
}

void QuantumTradingSystemIntegration::initialize_environment() {
    std::cout << "🌐 Initializing Quantum Trading Environment..." << std::endl;
    
    agents::QuantumTradingEnvironment::EnvironmentConfig env_config;
    env_config.num_assets = config_.num_assets;
    env_config.market_volatility = config_.market_volatility;
    env_config.use_quantum_market = config_.use_quantum_market;
    env_config.quantum_market_states = config_.quantum_states;
    
    environment_ = std::make_unique<agents::QuantumTradingEnvironment>(env_config);
    
    std::cout << "✅ Quantum Trading Environment initialized" << std::endl;
}

void QuantumTradingSystemIntegration::initialize_web_integration() {
    std::cout << "🌐 Initializing Web Integration..." << std::endl;
    
    // Create web integration
    web::QuantumAgentWebIntegration::WebIntegrationConfig web_config;
    web_config.port = 8080;
    web_config.websocket_port = 3001;
    web_config.update_interval_ms = 1000;
    web_config.enable_real_time_updates = true;
    web_config.enable_agent_control = true;
    
    web_integration_ = std::make_unique<web::QuantumAgentWebIntegration>(web_config);
    web_integration_->initialize();
    
    // Register all agents with web integration
    for (size_t i = 0; i < agents_.size(); ++i) {
        std::string agent_id = "agent_" + std::to_string(i + 1);
        web_integration_->register_agent(agents_[i], agent_id);
        std::cout << "   Agent " << agent_id << " registered with web interface" << std::endl;
    }
    
    std::cout << "✅ Web Integration initialized" << std::endl;
}

void QuantumTradingSystemIntegration::integrate_agents_with_models() {
    std::cout << "🔗 Integrating Agents with Quantum Models..." << std::endl;
    
    // Create quantum models for each agent
    for (int i = 0; i < config_.num_agents; ++i) {
        models::QuantumTradingSignals::QuantumSignalConfig model_config;
        model_config.input_features = config_.input_features;
        model_config.hidden_dim = config_.hidden_dim;
        model_config.num_heads = config_.num_heads;
        model_config.quantum_states = config_.quantum_states;
        
        auto quantum_model = std::make_unique<models::QuantumTradingSignals>(model_config);
        quantum_models_.push_back(std::move(quantum_model));
        
        std::cout << "   Agent " << (i + 1) << " integrated with quantum model" << std::endl;
    }
    
    std::cout << "✅ Agent-Model Integration completed" << std::endl;
}

void QuantumTradingSystemIntegration::setup_quantum_coordination() {
    std::cout << "🤝 Setting up Quantum Coordination..." << std::endl;
    
    if (config_.use_quantum_coordination) {
        // Enable quantum coordination between agents
        multi_agent_system_->coordinate_agents();
        
        std::cout << "   Quantum coordination enabled" << std::endl;
        std::cout << "   Communication channels: " << config_.quantum_states << " quantum states" << std::endl;
    }
    
    std::cout << "✅ Quantum Coordination setup completed" << std::endl;
}

void QuantumTradingSystemIntegration::configure_market_environment() {
    std::cout << "📊 Configuring Market Environment..." << std::endl;
    
    // Reset environment to initial state
    auto initial_state = environment_->reset();
    
    std::cout << "   Market initialized with " << config_.num_assets << " assets" << std::endl;
    std::cout << "   Market volatility: " << config_.market_volatility << std::endl;
    std::cout << "   Quantum market: " << (config_.use_quantum_market ? "enabled" : "disabled") << std::endl;
    
    std::cout << "✅ Market Environment configured" << std::endl;
}

void QuantumTradingSystemIntegration::run_training() {
    std::cout << "🎓 Starting Quantum Agent Training..." << std::endl;
    std::cout << "   Training Episodes: " << config_.training_episodes << std::endl;
    std::cout << "   Steps per Episode: " << config_.steps_per_episode << std::endl;
    std::cout << std::endl;
    
    for (int episode = 0; episode < config_.training_episodes; ++episode) {
        current_episode_ = episode;
        
        std::cout << "📊 Episode " << (episode + 1) << "/" << config_.training_episodes << std::endl;
        
        train_single_episode();
        
        // Print progress every 10 episodes
        if ((episode + 1) % 10 == 0) {
            print_system_status();
            print_quantum_metrics();
        }
        
        // Small delay for readability
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << std::endl;
    std::cout << "✅ Quantum Agent Training completed!" << std::endl;
    std::cout << "📈 Total Episodes: " << config_.training_episodes << std::endl;
    std::cout << "🎯 Average Reward: " << (total_reward_ / config_.training_episodes) << std::endl;
    std::cout << std::endl;
}

void QuantumTradingSystemIntegration::train_single_episode() {
    // Reset environment
    auto market_state = environment_->reset();
    
    // Reset agents
    for (auto& agent : agents_) {
        agent->reset();
    }
    
    double episode_reward = 0.0;
    
    // Run episode
    for (int step = 0; step < config_.steps_per_episode; ++step) {
        // Get market data
        auto market_data = market_state.prices;
        
        // Generate actions from all agents
        std::vector<agents::AgentAction> actions;
        for (auto& agent : agents_) {
            agent->step(market_data);
            auto agent_state = agent->get_current_state();
            auto action = agent->select_action(agent_state);
            actions.push_back(action);
        }
        
        // Execute actions in environment
        market_state = environment_->step(actions);
        
        // Update episode reward
        for (const auto& action : actions) {
            episode_reward += action.confidence * 0.01;  // Simplified reward
        }
        
        // Coordinate agents
        multi_agent_system_->coordinate_agents();
        
        total_steps_++;
    }
    
    // Store episode reward
    episode_rewards_.push_back(episode_reward);
    total_reward_ += episode_reward;
    
    // Update quantum coherence history
    double avg_coherence = 0.0;
    for (const auto& agent : agents_) {
        avg_coherence += agent->get_quantum_coherence();
    }
    avg_coherence /= agents_.size();
    quantum_coherence_history_.push_back(avg_coherence);
}

void QuantumTradingSystemIntegration::run_live_trading() {
    std::cout << "🔄 Starting Live Trading Simulation..." << std::endl;
    
    // Start web interface
    if (web_integration_) {
        web_integration_->integrate_with_web_interface();
    }
    
    // Initialize for live trading
    auto market_state = environment_->reset();
    
    for (int step = 0; step < 100; ++step) {
        std::cout << "📊 Trading Step " << (step + 1) << "/100" << std::endl;
        
        execute_trading_step();
        
        // Print status every 10 steps
        if ((step + 1) % 10 == 0) {
            print_system_status();
            print_trading_performance();
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    std::cout << std::endl;
    std::cout << "✅ Live Trading Simulation completed!" << std::endl;
}

void QuantumTradingSystemIntegration::integrate_with_web_interface() {
    std::cout << "🌐 Integrating with Web Interface..." << std::endl;
    
    if (web_integration_) {
        web_integration_->integrate_with_web_interface();
        
        std::cout << "   Web Interface: http://localhost:8080" << std::endl;
        std::cout << "   WebSocket: ws://localhost:3001" << std::endl;
        std::cout << "   API Endpoints: /api/v4/quantum/" << std::endl;
        std::cout << "   Real-time Updates: " << (config_.update_interval_ms) << "ms" << std::endl;
    }
    
    std::cout << "✅ Web Interface Integration completed!" << std::endl;
}

void QuantumTradingSystemIntegration::execute_trading_step() {
    // Get current market state
    auto market_state = environment_->get_current_state();
    auto market_data = market_state.prices;
    
    // Generate actions from all agents
    std::vector<agents::AgentAction> actions;
    for (auto& agent : agents_) {
        agent->step(market_data);
        auto agent_state = agent->get_current_state();
        auto action = agent->select_action(agent_state);
        actions.push_back(action);
        
        std::cout << "   Agent Action: " << action.symbol << " " << action.action 
                 << " (confidence: " << action.confidence << ")" << std::endl;
    }
    
    // Execute actions
    auto new_state = environment_->step(actions);
    
    // Coordinate agents
    multi_agent_system_->coordinate_agents();
    
    // Update performance
    update_performance_metrics();
}

void QuantumTradingSystemIntegration::evaluate_performance() {
    std::cout << "📊 Evaluating System Performance..." << std::endl;
    
    print_system_status();
    print_quantum_metrics();
    print_trading_performance();
    
    std::cout << std::endl;
    std::cout << "✅ Performance Evaluation completed!" << std::endl;
}

void QuantumTradingSystemIntegration::print_system_status() {
    std::cout << "\n🔍 System Status:" << std::endl;
    std::cout << "   Current Episode: " << (current_episode_ + 1) << std::endl;
    std::cout << "   Total Steps: " << total_steps_ << std::endl;
    std::cout << "   Total Reward: " << total_reward_ << std::endl;
    
    if (!episode_rewards_.empty()) {
        double avg_reward = 0.0;
        for (double reward : episode_rewards_) {
            avg_reward += reward;
        }
        avg_reward /= episode_rewards_.size();
        std::cout << "   Average Episode Reward: " << avg_reward << std::endl;
    }
}

void QuantumTradingSystemIntegration::print_quantum_metrics() {
    std::cout << "\n🧠 Quantum Metrics:" << std::endl;
    
    // Agent quantum coherence
    std::cout << "   Agent Quantum Coherence:" << std::endl;
    for (size_t i = 0; i < agents_.size(); ++i) {
        double coherence = agents_[i]->get_quantum_coherence();
        std::cout << "     Agent " << (i + 1) << ": " << coherence << std::endl;
    }
    
    // System quantum coordination
    if (multi_agent_system_) {
        double coordination = multi_agent_system_->get_quantum_coordination();
        std::cout << "   System Quantum Coordination: " << coordination << std::endl;
    }
    
    // Average quantum coherence
    if (!quantum_coherence_history_.empty()) {
        double avg_coherence = 0.0;
        for (double coherence : quantum_coherence_history_) {
            avg_coherence += coherence;
        }
        avg_coherence /= quantum_coherence_history_.size();
        std::cout << "   Average Quantum Coherence: " << avg_coherence << std::endl;
    }
}

void QuantumTradingSystemIntegration::print_trading_performance() {
    std::cout << "\n📈 Trading Performance:" << std::endl;
    
    // Agent performance
    std::cout << "   Agent Performance:" << std::endl;
    for (size_t i = 0; i < agents_.size(); ++i) {
        double performance = agents_[i]->get_performance_metric();
        double win_rate = agents_[i]->get_win_rate();
        int total_actions = agents_[i]->get_total_actions();
        
        std::cout << "     Agent " << (i + 1) << ":" << std::endl;
        std::cout << "       Performance: " << performance << std::endl;
        std::cout << "       Win Rate: " << win_rate * 100 << "%" << std::endl;
        std::cout << "       Total Actions: " << total_actions << std::endl;
    }
    
    // System performance
    if (multi_agent_system_) {
        double system_performance = multi_agent_system_->get_system_performance();
        std::cout << "   System Performance: " << system_performance << std::endl;
    }
    
    // Environment performance
    if (environment_) {
        double total_return = environment_->get_total_return();
        std::cout << "   Environment Total Return: " << total_return << std::endl;
    }
}

void QuantumTradingSystemIntegration::update_performance_metrics() {
    // Update win rates
    win_rates_.clear();
    for (const auto& agent : agents_) {
        win_rates_.push_back(agent->get_win_rate());
    }
}

// ============================================================================
// Main Integration Demo
// ============================================================================

int main() {
    std::cout << "🚀 ArchNeuronX v4.0 - Quantum Agent Integration Demo" << std::endl;
    std::cout << "🤖 Multi-Agent Quantum Trading System" << std::endl;
    std::cout << "🧠 16-head Quantum Neural Networks" << std::endl;
    std::cout << "🤝 Quantum Agent Coordination" << std::endl;
    std::cout << "🌐 Quantum Market Environment" << std::endl;
    std::cout << std::endl;
    
    try {
        // Configure integration
        QuantumTradingSystemIntegration::IntegrationConfig config;
        config.num_agents = 3;
        config.num_heads = 16;
        config.quantum_states = 8;
        config.training_episodes = 50;
        config.steps_per_episode = 25;
        config.use_quantum_coordination = true;
        config.use_quantum_market = true;
        
        // Create and initialize system
        auto integration_system = std::make_unique<QuantumTradingSystemIntegration>(config);
        integration_system->initialize();
        
        // Run training
        integration_system->run_training();
        
        // Run live trading simulation
        integration_system->run_live_trading();
        
        // Evaluate performance
        integration_system->evaluate_performance();
        
        std::cout << "\n🎉 Quantum Agent Integration Demo Completed Successfully!" << std::endl;
        std::cout << "🚀 ArchNeuronX v4.0 is ready for quantum multi-agent trading!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
