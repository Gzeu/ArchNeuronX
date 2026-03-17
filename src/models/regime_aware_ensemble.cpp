/**
 * @file regime_aware_ensemble.cpp
 * @brief Regime-aware ensemble implementation
 * @author ArchNeuronX Team
 * @date 2026-03-17
 */

#include "models/regime_aware_ensemble.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

namespace archneuronx {
namespace models {

RegimeAwareEnsemble::RegimeAwareEnsemble(
    const RegimeEnsembleConfig& config,
    const regime::RegimeConfig& regime_config)
    : EnsembleModel(100), config_(config) {
    
    // Initialize regime detector
    regime_detector_ = std::make_unique<regime::RegimeDetector>(regime_config);
    
    // Initialize metrics
    metrics_.overall_accuracy = 0.0;
    std::fill(std::begin(metrics_.regime_specific_accuracy), std::end(metrics_.regime_specific_accuracy), 0.0);
    metrics_.weight_entropy = 0.0;
    metrics_.regime_stability_score = 0.0;
    metrics_.regime_switches = 0;
    metrics_.adaptation_speed = 0.0;
    metrics_.last_update = std::chrono::system_clock::now();
    
    // Initialize regime configurations
    initialize_regime_configs();
}

bool RegimeAwareEnsemble::initialize() {
    std::lock_guard<std::mutex> lock(ensemble_mutex_);
    
    try {
        // Initialize regime detector
        if (!regime_detector_->initialize()) {
            std::cerr << "Failed to initialize regime detector" << std::endl;
            return false;
        }
        
        // Get initial regime
        current_regime_ = regime_detector_->get_current_regime();
        
        std::cout << "RegimeAwareEnsemble initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing RegimeAwareEnsemble: " << e.what() << std::endl;
        return false;
    }
}

void RegimeAwareEnsemble::add_model_with_regime_config(
    const std::string& name,
    torch::jit::script::Module model,
    const std::unordered_map<regime::MarketRegime, RegimeModelConfig>& regime_configs) {
    
    std::lock_guard<std::mutex> lock(ensemble_mutex_);
    
    // Add model to base ensemble
    add_model(name, model, 1.0);
    
    // Store regime-specific configurations
    for (const auto& [regime, config] : regime_configs) {
        regime_model_configs_[regime].push_back(config);
    }
    
    // Initialize performance tracking for this model
    for (int i = 0; i < 8; ++i) {
        regime::MarketRegime regime = static_cast<regime::MarketRegime>(i);
        regime_performance_history_[regime][name] = std::deque<bool>();
    }
    
    std::cout << "Added model '" << name << "' with regime-specific configurations" << std::endl;
}

torch::Tensor RegimeAwareEnsemble::predict_regime_aware(
    const torch::Tensor& temporal_input,
    const torch::Tensor& static_input,
    const torch::Device& device,
    const std::vector<double>& prices,
    const std::vector<double>& volumes) {
    
    std::lock_guard<std::mutex> lock(ensemble_mutex_);
    
    // Update regime detection
    regime_detector_->update_tick(prices.back(), volumes.back(), std::chrono::system_clock::now());
    current_regime_ = regime_detector_->get_current_regime();
    
    // Adapt weights to current regime
    adapt_weights_to_regime(current_regime_.regime);
    
    // Get regime-aware prediction
    torch::Tensor prediction = predict(temporal_input, static_input, device);
    
    // Apply regime-specific adjustments
    if (current_regime_.is_transition) {
        // Reduce confidence during transitions
        prediction = prediction * config_.regime_switch_penalty;
        prediction = torch::softmax(prediction, -1);
    }
    
    // Update metrics
    update_metrics();
    
    return prediction;
}

void RegimeAwareEnsemble::update_with_market_data(
    const std::vector<double>& prices,
    const std::vector<double>& volumes,
    const std::chrono::system_clock::time_point& timestamp) {
    
    std::lock_guard<std::mutex> lock(ensemble_mutex_);
    
    // Update regime detector
    regime_detector_->update_tick(prices.back(), volumes.back(), timestamp);
    regime::RegimeResult new_regime = regime_detector_->get_current_regime();
    
    // Check for regime switch
    if (new_regime.regime != current_regime_.regime) {
        metrics_.regime_switches++;
        // Rebalance weights for new regime
        adapt_weights_to_regime(new_regime.regime);
    }
    
    current_regime_ = new_regime;
}

void RegimeAwareEnsemble::update_performance_regime_aware(
    const std::string& model_name,
    bool correct,
    regime::MarketRegime regime) {
    
    std::lock_guard<std::mutex> lock(ensemble_mutex_);
    
    // Update base ensemble
    update_weights(model_name, correct);
    
    // Update regime-specific performance
    auto& history = regime_performance_history_[regime][model_name];
    history.push_back(correct);
    
    // Keep window size manageable
    while (history.size() > config_.performance_window) {
        history.pop_front();
    }
    
    // Update regime-specific accuracy
    if (!history.empty()) {
        int correct_count = std::count(history.begin(), history.end(), true);
        metrics_.regime_specific_accuracy[static_cast<int>(regime)] = 
            static_cast<double>(correct_count) / history.size();
    }
    
    // Check if model needs to be disabled for this regime
    if (config_.auto_disable_underperforming) {
        double accuracy = metrics_.regime_specific_accuracy[static_cast<int>(regime)];
        if (accuracy < 0.3 && history.size() >= 20) { // Poor performance
            auto it = std::find_if(regime_model_configs_[regime].begin(),
                                  regime_model_configs_[regime].end(),
                                  [&model_name](const RegimeModelConfig& config) {
                                      return config.model_name == model_name;
                                  });
            if (it != regime_model_configs_[regime].end()) {
                it->is_active = false;
            }
        }
    }
}

regime::RegimeResult RegimeAwareEnsemble::get_current_regime() const {
    std::lock_guard<std::mutex> lock(ensemble_mutex_);
    return current_regime_;
}

RegimeEnsembleMetrics RegimeAwareEnsemble::get_metrics() const {
    std::lock_guard<std::mutex> lock(ensemble_mutex_);
    return metrics_;
}

void RegimeAwareEnsemble::adapt_weights_to_regime(regime::MarketRegime regime) {
    // Calculate regime-specific weights
    calculate_regime_specific_weights(regime);
    
    // Apply diversification if enabled
    if (config_.enable_regime_diversification) {
        apply_regime_diversification();
    }
    
    // Normalize weights
    rebalance_weights();
}

void RegimeAwareEnsemble::calculate_regime_specific_weights(regime::MarketRegime regime) {
    auto current_weights = get_weights();
    
    for (auto& weight_info : current_weights) {
        double regime_multiplier = 1.0;
        
        // Find regime-specific configuration for this model
        auto regime_configs = regime_model_configs_[regime];
        auto it = std::find_if(regime_configs.begin(), regime_configs.end(),
                              [&weight_info](const RegimeModelConfig& config) {
                                  return config.model_name == weight_info.name;
                              });
        
        if (it != regime_configs.end() && it->is_active) {
            // Apply regime-specific boost
            regime_multiplier = it->performance_multiplier * it->base_weight;
            
            // Consider regime-specific performance
            double regime_accuracy = metrics_.regime_specific_accuracy[static_cast<int>(regime)];
            if (regime_accuracy > 0.5) {
                regime_multiplier *= (1.0 + regime_accuracy);
            }
        } else {
            // Penalize models not configured for this regime
            regime_multiplier = 0.5;
        }
        
        // Apply adaptation rate
        double target_weight = weight_info.weight * regime_multiplier;
        weight_info.weight = weight_info.weight * (1.0 - config_.adaptation_rate) + 
                           target_weight * config_.adaptation_rate;
    }
    
    // Update weights in base ensemble
    // Note: This would require extending the base EnsembleModel to allow direct weight setting
    // For now, we'll use the existing rebalance_weights() mechanism
}

void RegimeAwareEnsemble::apply_regime_diversification() {
    auto current_weights = get_weights();
    
    // Check for over-concentration
    double max_weight = 0.0;
    for (const auto& weight_info : current_weights) {
        max_weight = std::max(max_weight, weight_info.weight);
    }
    
    if (max_weight > config_.max_regime_concentration) {
        // Redistribute excess weight
        double excess = max_weight - config_.max_regime_concentration;
        double redistribution = excess / (current_weights.size() - 1);
        
        for (auto& weight_info : current_weights) {
            if (weight_info.weight == max_weight) {
                weight_info.weight = config_.max_regime_concentration;
            } else {
                weight_info.weight += redistribution;
            }
        }
    }
}

std::vector<RegimeModelConfig> RegimeAwareEnsemble::get_regime_model_configs(regime::MarketRegime regime) const {
    std::lock_guard<std::mutex> lock(ensemble_mutex_);
    return regime_model_configs_.at(regime);
}

std::unordered_map<regime::MarketRegime, double> RegimeAwareEnsemble::get_regime_performance() const {
    std::lock_guard<std::mutex> lock(ensemble_mutex_);
    
    std::unordered_map<regime::MarketRegime, double> performance;
    for (int i = 0; i < 8; ++i) {
        regime::MarketRegime regime = static_cast<regime::MarketRegime>(i);
        performance[regime] = metrics_.regime_specific_accuracy[i];
    }
    
    return performance;
}

std::vector<std::string> RegimeAwareEnsemble::get_best_models_for_regime(regime::MarketRegime regime) const {
    std::lock_guard<std::mutex> lock(ensemble_mutex_);
    
    std::vector<std::pair<std::string, double>> model_scores;
    
    // Calculate scores for all models in this regime
    auto regime_configs = regime_model_configs_.at(regime);
    for (const auto& config : regime_configs) {
        if (config.is_active) {
            double score = calculate_model_regime_score(config.model_name, regime);
            model_scores.emplace_back(config.model_name, score);
        }
    }
    
    // Sort by score (descending)
    std::sort(model_scores.begin(), model_scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Extract model names (top N)
    std::vector<std::string> best_models;
    int max_models = std::min(config_.max_models_per_regime, static_cast<int>(model_scores.size()));
    
    for (int i = 0; i < max_models; ++i) {
        best_models.push_back(model_scores[i].first);
    }
    
    return best_models;
}

bool RegimeAwareEnsemble::is_overfitting_detected() const {
    return calculate_overfitting_risk() > 0.7;
}

double RegimeAwareEnsemble::calculate_overfitting_risk() const {
    double risk = 0.0;
    
    // Weight entropy (low entropy = overfitting)
    double entropy = calculate_weight_entropy();
    risk += (1.0 - entropy) * 0.3;
    
    // Regime correlation (high correlation = overfitting)
    double correlation = calculate_regime_correlation();
    risk += correlation * 0.4;
    
    // Performance degradation
    if (detect_performance_degradation()) {
        risk += 0.3;
    }
    
    return std::min(1.0, risk);
}

void RegimeAwareEnsemble::apply_overfitting_mitigation() {
    // Increase diversification
    config_.max_regime_concentration = std::max(0.3, config_.max_regime_concentration - 0.1);
    
    // Increase adaptation rate
    config_.adaptation_rate = std::min(0.3, config_.adaptation_rate + 0.05);
    
    // Enable regularization (reduce weights of overperforming models)
    auto current_weights = get_weights();
    for (auto& weight_info : current_weights) {
        if (weight_info.weight > 0.5) {
            weight_info.weight *= 0.9;
        }
    }
    
    std::cout << "Applied overfitting mitigation measures" << std::endl;
}

// Private methods

void RegimeAwareEnsemble::initialize_regime_configs() {
    // Initialize default configurations for all regimes
    for (int i = 0; i < 8; ++i) {
        regime::MarketRegime regime = static_cast<regime::MarketRegime>(i);
        regime_model_configs_[regime] = std::vector<RegimeModelConfig>();
    }
}

void RegimeAwareEnsemble::update_metrics() {
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - metrics_.last_update);
    
    metrics_.last_update = now;
    metrics_.weight_entropy = calculate_weight_entropy();
    metrics_.regime_stability_score = regime_detector_->get_regime_stability();
    
    // Calculate adaptation speed (simplified)
    if (duration.count() > 0) {
        metrics_.adaptation_speed = metrics_.weight_entropy / duration.count();
    }
    
    // Calculate overall accuracy
    auto current_weights = get_weights();
    double weighted_accuracy = 0.0;
    for (const auto& weight_info : current_weights) {
        weighted_accuracy += weight_info.weight * weight_info.rolling_accuracy;
    }
    metrics_.overall_accuracy = weighted_accuracy;
}

double RegimeAwareEnsemble::calculate_weight_entropy() const {
    auto current_weights = get_weights();
    
    double entropy = 0.0;
    for (const auto& weight_info : current_weights) {
        if (weight_info.weight > 0.0) {
            entropy -= weight_info.weight * std::log2(weight_info.weight);
        }
    }
    
    return entropy;
}

double RegimeAwareEnsemble::calculate_regime_correlation() const {
    // Simplified correlation calculation
    // In practice, this would analyze correlation between model predictions and regime changes
    
    double correlation = 0.0;
    
    // Check if models are too sensitive to regime changes
    if (metrics_.regime_switches > 10) {
        correlation = std::min(1.0, metrics_.regime_switches / 50.0);
    }
    
    return correlation;
}

bool RegimeAwareEnsemble::detect_performance_degradation() const {
    // Check if recent performance is significantly worse than historical
    double recent_accuracy = metrics_.overall_accuracy;
    
    // Simple threshold check (in practice, would use statistical tests)
    return recent_accuracy < 0.4;
}

double RegimeAwareEnsemble::calculate_model_regime_score(const std::string& model_name, regime::MarketRegime regime) const {
    double score = 0.0;
    
    // Base weight
    auto regime_configs = regime_model_configs_.at(regime);
    auto it = std::find_if(regime_configs.begin(), regime_configs.end(),
                          [&model_name](const RegimeModelConfig& config) {
                              return config.model_name == model_name;
                          });
    
    if (it != regime_configs.end()) {
        score += it->base_weight * it->performance_multiplier;
    }
    
    // Regime-specific performance
    double regime_accuracy = metrics_.regime_specific_accuracy[static_cast<int>(regime)];
    score += regime_accuracy * 2.0;
    
    // Overall performance
    auto current_weights = get_weights();
    auto weight_it = std::find_if(current_weights.begin(), current_weights.end(),
                                 [&model_name](const ModelWeight& weight) {
                                     return weight.name == model_name;
                                 });
    
    if (weight_it != current_weights.end()) {
        score += weight_it->rolling_accuracy;
    }
    
    return score;
}

std::string RegimeAwareEnsemble::regime_to_string(regime::MarketRegime regime) const {
    switch (regime) {
        case regime::MarketRegime::BULL_LOW_VOL: return "BULL_LOW_VOL";
        case regime::MarketRegime::BULL_HIGH_VOL: return "BULL_HIGH_VOL";
        case regime::MarketRegime::BEAR_LOW_VOL: return "BEAR_LOW_VOL";
        case regime::MarketRegime::BEAR_HIGH_VOL: return "BEAR_HIGH_VOL";
        case regime::MarketRegime::SIDEWAYS_LOW_VOL: return "SIDEWAYS_LOW_VOL";
        case regime::MarketRegime::SIDEWAYS_HIGH_VOL: return "SIDEWAYS_HIGH_VOL";
        case regime::MarketRegime::TRANSITION: return "TRANSITION";
        case regime::MarketRegime::UNKNOWN: return "UNKNOWN";
        default: return "INVALID";
    }
}

} // namespace models
} // namespace archneuronx
