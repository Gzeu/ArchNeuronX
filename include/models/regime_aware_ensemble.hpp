#pragma once
// ============================================================
// ArchNeuronX v3 - Regime-Aware Ensemble System
// Adapts model weights and predictions based on market regime
// Prevents overfitting by specializing models per regime
// ============================================================

#include "models/ensemble.hpp"
#include "regime/regime_detector.hpp"
#include <torch/torch.h>
#include <unordered_map>
#include <mutex>

namespace archneuronx {
namespace models {

/**
 * @brief Regime-specific model configuration
 */
struct RegimeModelConfig {
    std::string model_name;
    double base_weight;           // Base weight for this regime
    double performance_multiplier; // Performance boost in this regime
    bool is_active;              // Whether model is active in current regime
    int regime_specific_accuracy; // Historical accuracy in this regime
};

/**
 * @brief Regime-aware ensemble configuration
 */
struct RegimeEnsembleConfig {
    // Weight adaptation settings
    double adaptation_rate = 0.1;          // How fast weights adapt to regime
    double min_weight_threshold = 0.05;    // Minimum weight for any model
    double regime_boost_factor = 1.5;      // Boost factor for regime-specialized models
    
    // Performance tracking
    int performance_window = 50;           // Window for performance tracking
    double regime_switch_penalty = 0.8;    // Penalty when regime changes
    
    // Risk management
    double max_regime_concentration = 0.6; // Max weight concentration in one regime
    bool enable_regime_diversification = true; // Force diversification across regimes
    
    // Model selection
    int max_models_per_regime = 5;         // Maximum active models per regime
    bool auto_disable_underperforming = true; // Auto-disable poor performing models
};

/**
 * @brief Performance metrics for regime-aware ensemble
 */
struct RegimeEnsembleMetrics {
    double overall_accuracy;
    double regime_specific_accuracy[8];    // Accuracy per regime
    double weight_entropy;                 // Diversity of weight distribution
    double regime_stability_score;        // How stable regime detection is
    int regime_switches;                   // Number of regime switches
    double adaptation_speed;              // How fast ensemble adapts
    std::chrono::system_clock::time_point last_update;
};

/**
 * @brief Regime-aware ensemble model
 * 
 * Extends the base ensemble system with regime-specific adaptation:
 * - Different model weights per market regime
 * - Dynamic model activation/deactivation based on regime
 * - Performance tracking per regime
 * - Anti-overfitting through regime diversification
 */
class RegimeAwareEnsemble : public EnsembleModel {
public:
    explicit RegimeAwareEnsemble(
        const RegimeEnsembleConfig& config = RegimeEnsembleConfig{},
        const regime::RegimeConfig& regime_config = regime::RegimeConfig{}
    );
    
    ~RegimeAwareEnsemble() = default;

    // Initialize regime detector and ensemble
    bool initialize();

    // Add model with regime-specific configuration
    void add_model_with_regime_config(
        const std::string& name,
        torch::jit::script::Module model,
        const std::unordered_map<regime::MarketRegime, RegimeModelConfig>& regime_configs
    );

    // Regime-aware prediction (main interface)
    torch::Tensor predict_regime_aware(
        const torch::Tensor& temporal_input,
        const torch::Tensor& static_input,
        const torch::Device& device,
        const std::vector<double>& prices,
        const std::vector<double>& volumes
    );

    // Update with market data and regime
    void update_with_market_data(
        const std::vector<double>& prices,
        const std::vector<double>& volumes,
        const std::chrono::system_clock::time_point& timestamp = std::chrono::system_clock::now()
    );

    // Update model performance with regime context
    void update_performance_regime_aware(
        const std::string& model_name,
        bool correct,
        regime::MarketRegime regime
    );

    // Get current regime and ensemble state
    regime::RegimeResult get_current_regime() const;
    RegimeEnsembleMetrics get_metrics() const;

    // Regime-specific weight management
    void adapt_weights_to_regime(regime::MarketRegime regime);
    void rebalance_regime_weights();
    std::vector<RegimeModelConfig> get_regime_model_configs(regime::MarketRegime regime) const;

    // Performance analysis
    std::unordered_map<regime::MarketRegime, double> get_regime_performance() const;
    std::vector<std::string> get_best_models_for_regime(regime::MarketRegime regime) const;

    // Export/import regime configurations
    void export_regime_config(const std::string& filepath) const;
    bool import_regime_config(const std::string& filepath);

    // Training and optimization
    void optimize_for_regime(regime::MarketRegime target_regime, int optimization_epochs = 100);
    void train_regime_classifier(const std::string& historical_data_path);

    // Risk management
    bool is_overfitting_detected() const;
    double calculate_overfitting_risk() const;
    void apply_overfitting_mitigation();

private:
    RegimeEnsembleConfig config_;
    std::unique_ptr<regime::RegimeDetector> regime_detector_;
    
    // Regime-specific model configurations
    std::unordered_map<regime::MarketRegime, std::vector<RegimeModelConfig>> regime_model_configs_;
    
    // Performance tracking per regime
    std::unordered_map<regime::MarketRegime, std::unordered_map<std::string, std::deque<bool>>> regime_performance_history_;
    
    // Current state
    regime::RegimeResult current_regime_;
    RegimeEnsembleMetrics metrics_;
    mutable std::mutex ensemble_mutex_;
    
    // Internal methods
    void initialize_regime_configs();
    void update_metrics();
    void calculate_regime_specific_weights(regime::MarketRegime regime);
    void apply_regime_diversification();
    std::vector<double> get_regime_weight_vector(regime::MarketRegime regime) const;
    
    // Overfitting detection
    double calculate_weight_entropy() const;
    double calculate_regime_correlation() const;
    bool detect_performance_degradation() const;
    
    // Utility methods
    std::string regime_to_string(regime::MarketRegime regime) const;
    double calculate_model_regime_score(const std::string& model_name, regime::MarketRegime regime) const;
};

} // namespace models
} // namespace archneuronx
