#pragma once

#include "../models/quantum_trading_signals.hpp"
#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <functional>

namespace archneuronx {
namespace agents {

/**
 * Quantum Trading Agent
 * 
 * This agent combines quantum neural networks with reinforcement learning
 * for autonomous trading decisions and portfolio management.
 */
class QuantumTradingAgent {
public:
    struct AgentConfig {
        // Quantum model configuration
        int input_features = 128;
        int hidden_dim = 256;
        int num_heads = 16;
        int num_layers = 6;
        
        // Agent configuration
        double learning_rate = 0.001;
        double discount_factor = 0.99;
        double exploration_rate = 0.1;
        int memory_size = 10000;
        int batch_size = 32;
        
        // Trading configuration
        double max_position_size = 0.1;
        double risk_tolerance = 0.05;
        int max_positions = 10;
        
        // Quantum configuration
        int quantum_states = 8;
        double quantum_coherence_threshold = 0.8;
        bool use_quantum_exploration = true;
    };

    struct AgentState {
        torch::Tensor market_state;
        torch::Tensor portfolio_state;
        torch::Tensor quantum_state;
        double current_value;
        double total_pnl;
        int active_positions;
        std::chrono::system_clock::time_point last_action;
    };

    struct AgentAction {
        std::string symbol;
        std::string action;  // "BUY", "SELL", "HOLD"
        double quantity;
        double confidence;
        torch::Tensor action_q_values;
        torch::Tensor quantum_attention;
    };

    struct Experience {
        AgentState state;
        AgentAction action;
        double reward;
        AgentState next_state;
        bool done;
    };

public:
    explicit QuantumTradingAgent(const AgentConfig& config);
    ~QuantumTradingAgent() = default;

    // Agent lifecycle
    void initialize();
    void reset();
    void step(const torch::Tensor& market_data);
    
    // Action selection
    AgentAction select_action(const AgentState& state);
    AgentAction select_quantum_action(const AgentState& state);
    AgentAction explore_action(const AgentState& state);
    AgentAction exploit_action(const AgentState& state);
    
    // Learning and training
    void learn();
    void train_quantum_network();
    void update_quantum_policy();
    void optimize_quantum_values();
    
    // Experience replay
    void store_experience(const Experience& experience);
    std::vector<Experience> sample_experience(int batch_size);
    void update_priorities();
    
    // Quantum-specific methods
    void update_quantum_state();
    double calculate_quantum_advantage();
    torch::Tensor compute_quantum_targets();
    
    // Performance monitoring
    double get_performance_metric() const;
    double get_quantum_coherence() const;
    int get_total_actions() const;
    double get_win_rate() const;
    
    // Model management
    void save_agent(const std::string& path);
    void load_agent(const std::string& path);
    
    // Agent state
    AgentState get_current_state() const { return current_state_; }
    bool is_initialized() const { return initialized_; }

private:
    AgentConfig config_;
    
    // Core components
    std::unique_ptr<models::QuantumTradingSignals> quantum_model_;
    std::unique_ptr<torch::nn::Module> q_network_;
    std::unique_ptr<torch::nn::Module> target_network_;
    std::unique_ptr<torch::optim::Optimizer> optimizer_;
    
    // Agent state
    AgentState current_state_;
    std::vector<Experience> experience_buffer_;
    
    // Training state
    int training_step_ = 0;
    double total_reward_ = 0.0;
    int total_actions_ = 0;
    int successful_actions_ = 0;
    
    // Quantum state
    torch::Tensor quantum_policy_;
    torch::Tensor quantum_values_;
    double quantum_coherence_ = 1.0;
    
    // Exploration
    double exploration_rate_;
    std::mt19937 random_generator_;
    
    // Performance metrics
    std::vector<double> performance_history_;
    double current_performance_ = 0.0;
    
    // Initialization
    bool initialized_ = false;
    
    // Private methods
    void initialize_networks();
    void initialize_quantum_components();
    void update_target_network();
    
    // Action utilities
    AgentAction create_action_from_q_values(const torch::Tensor& q_values, const std::string& symbol);
    double calculate_reward(const AgentState& state, const AgentAction& action, const AgentState& next_state);
    
    // Learning utilities
    torch::Tensor compute_loss(const std::vector<Experience>& batch);
    void update_network_parameters();
    
    // State utilities
    AgentState create_state_from_market_data(const torch::Tensor& market_data);
    void update_portfolio_state(const AgentAction& action);
};

/**
 * Quantum Multi-Agent System
 * 
 * Manages multiple quantum trading agents for different assets
 * or trading strategies with quantum coordination.
 */
class QuantumMultiAgentSystem {
public:
    struct MultiAgentConfig {
        int num_agents = 5;
        AgentConfig agent_config;
        bool use_quantum_coordination = true;
        double coordination_strength = 0.1;
        int quantum_communication_states = 4;
    };

public:
    explicit QuantumMultiAgentSystem(const MultiAgentConfig& config);
    
    // System management
    void initialize();
    void reset_all_agents();
    void step_all_agents(const torch::Tensor& market_data);
    
    // Agent coordination
    void coordinate_agents();
    void share_quantum_information();
    void resolve_conflicts();
    
    // Collective learning
    void collective_learn();
    void update_quantum_coordination();
    
    // System monitoring
    double get_system_performance() const;
    double get_quantum_coordination() const;
    std::vector<double> get_agent_performances() const;
    
    // System state
    std::vector<AgentState> get_all_agent_states() const;
    std::vector<AgentAction> get_all_agent_actions() const;

private:
    MultiAgentConfig config_;
    std::vector<std::unique_ptr<QuantumTradingAgent>> agents_;
    
    // Quantum coordination
    torch::Tensor coordination_matrix_;
    torch::Tensor quantum_communication_channel_;
    
    // System metrics
    double system_performance_ = 0.0;
    double quantum_coordination_ = 0.0;
    
    void initialize_coordination_matrix();
    void update_quantum_communication();
};

/**
 * Quantum Environment Interface
 * 
 * Provides the environment interface for quantum trading agents
 * with realistic market simulation and quantum state management.
 */
class QuantumTradingEnvironment {
public:
    struct EnvironmentConfig {
        int num_assets = 10;
        int lookback_window = 100;
        double transaction_cost = 0.001;
        double slippage = 0.0005;
        bool use_quantum_market = true;
        int quantum_market_states = 16;
    };

    struct MarketState {
        torch::Tensor prices;
        torch::Tensor volumes;
        torch::Tensor returns;
        torch::Tensor quantum_market_state;
        double market_volatility;
        double market_trend;
        std::chrono::system_clock::time_point timestamp;
    };

public:
    explicit QuantumTradingEnvironment(const EnvironmentConfig& config);
    
    // Environment interface
    MarketState reset();
    MarketState step(const std::vector<AgentAction>& actions);
    
    // Quantum market simulation
    MarketState generate_quantum_market_state();
    void update_quantum_market_dynamics();
    
    // Reward calculation
    double calculate_reward(const AgentAction& action, const MarketState& market_state);
    double calculate_portfolio_return(const std::vector<AgentAction>& actions);
    
    // Environment state
    bool is_done() const;
    MarketState get_current_state() const;
    double get_total_return() const;

private:
    EnvironmentConfig config_;
    MarketState current_state_;
    std::vector<MarketState> history_;
    
    // Quantum market dynamics
    torch::Tensor quantum_market_matrix_;
    torch::Tensor quantum_transition_probabilities_;
    
    // Performance tracking
    double total_return_ = 0.0;
    int current_step_ = 0;
    
    void initialize_quantum_market();
    MarketState simulate_market_step(const std::vector<AgentAction>& actions);
};

} // namespace agents
} // namespace archneuronx
