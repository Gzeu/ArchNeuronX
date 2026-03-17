/**
 * @file regime_detector.cpp
 * @brief Market regime detection implementation
 * @author ArchNeuronX Team
 * @date 2026-03-17
 */

#include "regime/regime_detector.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace archneuronx {
namespace regime {

RegimeDetector::RegimeDetector(const RegimeConfig& config) 
    : config_(config), classifier_loaded_(false), initialized_(false) {
    
    // Initialize current regime as unknown
    current_regime_.regime = MarketRegime::UNKNOWN;
    current_regime_.confidence = 0.0;
    current_regime_.is_transition = false;
    current_regime_.timestamp = std::chrono::system_clock::now();
}

bool RegimeDetector::initialize() {
    try {
        // Reserve space for data structures
        price_history_.reserve(config_.price_window * 2);
        volume_history_.reserve(config_.volume_window * 2);
        timestamp_history_.reserve(config_.price_window * 2);
        regime_history_.reserve(1000);
        
        // Load ML classifier if enabled
        if (config_.use_ml_classifier && !config_.model_path.empty()) {
            classifier_loaded_ = load_classifier(config_.model_path);
            if (!classifier_loaded_) {
                std::cerr << "Warning: Failed to load regime classifier, using statistical only" << std::endl;
            }
        }
        
        initialized_ = true;
        std::cout << "RegimeDetector initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing RegimeDetector: " << e.what() << std::endl;
        return false;
    }
}

RegimeResult RegimeDetector::detect_regime(
    const std::vector<double>& prices,
    const std::vector<double>& volumes,
    const std::chrono::system_clock::time_point& timestamp) {
    
    if (!initialized_) {
        throw std::runtime_error("RegimeDetector not initialized");
    }
    
    if (prices.size() < config_.price_window || volumes.size() < config_.volume_window) {
        throw std::invalid_argument("Insufficient data for regime detection");
    }
    
    // Extract features
    MarketFeatures features = extract_features(prices, volumes, timestamp);
    
    // Statistical classification
    MarketRegime statistical_regime = classify_statistical(features);
    
    // ML classification (if available)
    std::vector<double> ml_probabilities;
    if (classifier_loaded_) {
        ml_probabilities = classify_ml(features);
    }
    
    // Combine results
    RegimeResult result = combine_results(statistical_regime, ml_probabilities, features, timestamp);
    
    // Update current state
    current_regime_ = result;
    regime_history_.push_back(result);
    
    // Keep history size manageable
    if (regime_history_.size() > 1000) {
        regime_history_.erase(regime_history_.begin());
    }
    
    return result;
}

void RegimeDetector::update_tick(double price, double volume, const std::chrono::system_clock::time_point& timestamp) {
    // Add new data
    price_history_.push_back(price);
    volume_history_.push_back(volume);
    timestamp_history_.push_back(timestamp);
    
    // Maintain window sizes
    while (price_history_.size() > config_.price_window * 2) {
        price_history_.pop_front();
        volume_history_.pop_front();
        timestamp_history_.pop_front();
    }
    
    // Update regime if we have enough data
    if (price_history_.size() >= config_.price_window) {
        std::vector<double> prices(price_history_.begin(), price_history_.end());
        std::vector<double> volumes(volume_history_.begin(), volume_history_.end());
        detect_regime(prices, volumes, timestamp);
    }
}

RegimeResult RegimeDetector::get_current_regime() const {
    return current_regime_;
}

std::vector<RegimeResult> RegimeDetector::get_regime_history(int max_items) const {
    if (max_items >= static_cast<int>(regime_history_.size())) {
        return regime_history_;
    }
    
    return std::vector<RegimeResult>(
        regime_history_.end() - max_items,
        regime_history_.end()
    );
}

bool RegimeDetector::is_in_transition() const {
    return current_regime_.is_transition;
}

double RegimeDetector::get_regime_stability() const {
    if (regime_history_.size() < 10) {
        return 0.0; // Not enough data
    }
    
    // Calculate entropy of recent regimes
    double entropy = calculate_regime_entropy(regime_history_);
    
    // Convert entropy to stability (inverse relationship)
    return std::max(0.0, 1.0 - entropy);
}

MarketFeatures RegimeDetector::extract_features(
    const std::vector<double>& prices,
    const std::vector<double>& volumes,
    const std::chrono::system_clock::time_point& timestamp) const {
    
    MarketFeatures features;
    features.timestamp = timestamp;
    
    // Price-based features
    features.price_trend = calculate_trend(prices);
    features.volatility = calculate_volatility(prices);
    features.momentum = calculate_momentum(prices);
    features.mean_reversion = calculate_mean_reversion(prices);
    features.price_acceleration = calculate_acceleration(prices);
    
    // Volume-based features
    double avg_volume = std::accumulate(volumes.begin(), volumes.end(), 0.0) / volumes.size();
    features.volume_ratio = volumes.back() / avg_volume;
    features.volume_trend = calculate_trend(volumes);
    
    // Volatility ratio
    double recent_vol = calculate_volatility(
        std::vector<double>(prices.end() - config_.volatility_window, prices.end())
    );
    double historical_vol = calculate_volatility(prices);
    features.volatility_ratio = historical_vol > 0 ? recent_vol / historical_vol : 1.0;
    
    return features;
}

MarketRegime RegimeDetector::classify_statistical(const MarketFeatures& features) const {
    // Determine trend direction
    bool is_bull = features.price_trend > config_.trend_threshold;
    bool is_bear = features.price_trend < -config_.trend_threshold;
    bool is_sideways = !is_bull && !is_bear;
    
    // Determine volatility level
    bool is_high_vol = features.volatility > config_.volatility_threshold;
    
    // Combine to determine regime
    if (is_bull && !is_high_vol) return MarketRegime::BULL_LOW_VOL;
    if (is_bull && is_high_vol) return MarketRegime::BULL_HIGH_VOL;
    if (is_bear && !is_high_vol) return MarketRegime::BEAR_LOW_VOL;
    if (is_bear && is_high_vol) return MarketRegime::BEAR_HIGH_VOL;
    if (is_sideways && !is_high_vol) return MarketRegime::SIDEWAYS_LOW_VOL;
    if (is_sideways && is_high_vol) return MarketRegime::SIDEWAYS_HIGH_VOL;
    
    return MarketRegime::UNKNOWN;
}

std::vector<double> RegimeDetector::classify_ml(const MarketFeatures& features) const {
    if (!classifier_loaded_) {
        return std::vector<double>(8, 0.125); // Uniform distribution
    }
    
    try {
        // Create feature tensor
        std::vector<float> feature_vec = {
            static_cast<float>(features.price_trend),
            static_cast<float>(features.volatility),
            static_cast<float>(features.volume_ratio),
            static_cast<float>(features.momentum),
            static_cast<float>(features.mean_reversion),
            static_cast<float>(features.volatility_ratio),
            static_cast<float>(features.price_acceleration),
            static_cast<float>(features.volume_trend)
        };
        
        auto features_tensor = torch::from_blob(feature_vec.data(), {1, 8}, torch::kFloat32);
        
        // Get ML prediction
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(features_tensor);
        auto output = regime_classifier_.forward(inputs);
        auto probabilities = output.toTensor();
        
        // Convert to vector
        std::vector<double> result;
        auto probs_accessor = probabilities.accessor<float, 2>();
        for (int i = 0; i < 8; ++i) {
            result.push_back(probs_accessor[0][i]);
        }
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in ML classification: " << e.what() << std::endl;
        return std::vector<double>(8, 0.125); // Uniform distribution
    }
}

RegimeResult RegimeDetector::combine_results(
    MarketRegime statistical_regime,
    const std::vector<double>& ml_probabilities,
    const MarketFeatures& features,
    const std::chrono::system_clock::time_point& timestamp) const {
    
    RegimeResult result;
    result.timestamp = timestamp;
    result.features = features;
    result.probabilities = ml_probabilities;
    
    if (classifier_loaded_ && !ml_probabilities.empty()) {
        // Find highest probability regime
        auto max_it = std::max_element(ml_probabilities.begin(), ml_probabilities.end());
        int max_index = std::distance(ml_probabilities.begin(), max_it);
        
        result.confidence = *max_it;
        
        // Convert index to enum
        result.regime = static_cast<MarketRegime>(max_index);
        
        // Use statistical as fallback if confidence is low
        if (result.confidence < config_.confidence_threshold) {
            result.regime = statistical_regime;
            result.confidence = 0.5; // Moderate confidence for statistical
        }
    } else {
        // Use statistical only
        result.regime = statistical_regime;
        result.confidence = 0.7; // Moderate confidence for statistical
        result.probabilities = std::vector<double>(8, 0.125);
        result.probabilities[static_cast<int>(statistical_regime)] = 0.7;
    }
    
    // Check for transition
    result.is_transition = detect_transition(regime_history_);
    
    return result;
}

double RegimeDetector::calculate_trend(const std::vector<double>& prices) const {
    if (prices.size() < 2) return 0.0;
    
    // Linear regression to calculate trend
    double n = prices.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    
    for (size_t i = 0; i < prices.size(); ++i) {
        sum_x += i;
        sum_y += prices[i];
        sum_xy += i * prices[i];
        sum_x2 += i * i;
    }
    
    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    
    // Normalize by price level
    double avg_price = sum_y / n;
    return avg_price > 0 ? slope / avg_price : 0.0;
}

double RegimeDetector::calculate_volatility(const std::vector<double>& prices) const {
    if (prices.size() < 2) return 0.0;
    
    // Calculate returns
    std::vector<double> returns;
    for (size_t i = 1; i < prices.size(); ++i) {
        if (prices[i-1] > 0) {
            returns.push_back((prices[i] - prices[i-1]) / prices[i-1]);
        }
    }
    
    if (returns.empty()) return 0.0;
    
    // Calculate standard deviation of returns
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double variance = 0.0;
    
    for (double ret : returns) {
        variance += (ret - mean) * (ret - mean);
    }
    
    return std::sqrt(variance / returns.size());
}

double RegimeDetector::calculate_momentum(const std::vector<double>& prices) const {
    if (prices.size() < 10) return 0.0;
    
    // Short-term momentum: price change over last 10 periods
    double recent_change = (prices.back() - prices[prices.size() - 10]) / prices[prices.size() - 10];
    return recent_change;
}

double RegimeDetector::calculate_mean_reversion(const std::vector<double>& prices) const {
    if (prices.size() < 20) return 0.0;
    
    // Calculate mean reversion strength using Hurst exponent approximation
    double mean_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
    double deviation_sum = 0.0;
    
    for (double price : prices) {
        deviation_sum += std::abs(price - mean_price);
    }
    
    double avg_deviation = deviation_sum / prices.size();
    return mean_price > 0 ? avg_deviation / mean_price : 0.0;
}

double RegimeDetector::calculate_acceleration(const std::vector<double>& prices) const {
    if (prices.size() < 3) return 0.0;
    
    // Second derivative approximation
    size_t n = prices.size();
    double first_deriv1 = (prices[n-1] - prices[n-2]) / prices[n-2];
    double first_deriv2 = (prices[n-2] - prices[n-3]) / prices[n-3];
    
    return first_deriv1 - first_deriv2;
}

bool RegimeDetector::detect_transition(const std::vector<RegimeResult>& history) const {
    if (history.size() < config_.transition_window) return false;
    
    // Check if recent regimes have high entropy (indicating transition)
    double entropy = calculate_regime_entropy(history);
    return entropy > config_.transition_threshold;
}

double RegimeDetector::calculate_regime_entropy(const std::vector<RegimeResult>& history) const {
    if (history.size() < 2) return 0.0;
    
    // Count regime frequencies in recent history
    std::vector<int> regime_counts(8, 0);
    int window_size = std::min(config_.transition_window, static_cast<int>(history.size()));
    
    for (int i = history.size() - window_size; i < static_cast<int>(history.size()); ++i) {
        regime_counts[static_cast<int>(history[i].regime)]++;
    }
    
    // Calculate Shannon entropy
    double entropy = 0.0;
    for (int count : regime_counts) {
        if (count > 0) {
            double probability = static_cast<double>(count) / window_size;
            entropy -= probability * std::log2(probability);
        }
    }
    
    return entropy;
}

bool RegimeDetector::load_classifier(const std::string& model_path) {
    try {
        regime_classifier_ = torch::jit::load(model_path);
        std::cout << "Regime classifier loaded from: " << model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading regime classifier: " << e.what() << std::endl;
        return false;
    }
}

std::string RegimeDetector::regime_to_string(MarketRegime regime) const {
    switch (regime) {
        case MarketRegime::BULL_LOW_VOL: return "BULL_LOW_VOL";
        case MarketRegime::BULL_HIGH_VOL: return "BULL_HIGH_VOL";
        case MarketRegime::BEAR_LOW_VOL: return "BEAR_LOW_VOL";
        case MarketRegime::BEAR_HIGH_VOL: return "BEAR_HIGH_VOL";
        case MarketRegime::SIDEWAYS_LOW_VOL: return "SIDEWAYS_LOW_VOL";
        case MarketRegime::SIDEWAYS_HIGH_VOL: return "SIDEWAYS_HIGH_VOL";
        case MarketRegime::TRANSITION: return "TRANSITION";
        case MarketRegime::UNKNOWN: return "UNKNOWN";
        default: return "INVALID";
    }
}

} // namespace regime
} // namespace archneuronx
