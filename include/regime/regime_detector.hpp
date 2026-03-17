#pragma once
// ============================================================
// ArchNeuronX v3 - Market Regime Detection System
// Real-time regime classification: Bull/Bear/Sideways + Volatility
// Uses statistical features + ML classifier for adaptive trading
// ============================================================

#include <torch/torch.h>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <deque>

namespace archneuronx {
namespace regime {

/**
 * @brief Market regime enumeration
 */
enum class MarketRegime {
    BULL_LOW_VOL,     // Bull market with low volatility
    BULL_HIGH_VOL,    // Bull market with high volatility  
    BEAR_LOW_VOL,     // Bear market with low volatility
    BEAR_HIGH_VOL,    // Bear market with high volatility
    SIDEWAYS_LOW_VOL, // Sideways market with low volatility
    SIDEWAYS_HIGH_VOL, // Sideways market with high volatility
    TRANSITION,       // Regime transition period
    UNKNOWN           // Unable to classify
};

/**
 * @brief Regime detection configuration
 */
struct RegimeConfig {
    // Feature extraction window
    int price_window = 60;           // Price history window (minutes)
    int volume_window = 30;          // Volume history window
    int volatility_window = 20;       // Volatility calculation window
    
    // Classification thresholds
    double trend_threshold = 0.02;    // Trend strength threshold (2%)
    double volatility_threshold = 0.015; // Volatility threshold (1.5%)
    double volume_spike_threshold = 2.0; // Volume spike multiplier
    
    // ML classifier settings
    bool use_ml_classifier = true;    // Use neural classifier vs statistical
    std::string model_path = "models/regime_classifier.pt";
    double confidence_threshold = 0.6; // Minimum confidence for classification
    
    // Transition detection
    int transition_window = 10;       // Window to detect transitions
    double transition_threshold = 0.3; // Entropy threshold for transition
};

/**
 * @brief Market features for regime detection
 */
struct MarketFeatures {
    double price_trend;              // Normalized price trend (-1 to 1)
    double volatility;               // Normalized volatility (0 to 1)
    double volume_ratio;             // Current volume / average volume
    double momentum;                 // Short-term momentum
    double mean_reversion;           // Mean reversion strength
    double volatility_ratio;         // Current vol / average vol
    double price_acceleration;       // Second derivative of price
    double volume_trend;             // Volume trend direction
    std::chrono::system_clock::time_point timestamp;
};

/**
 * @brief Regime detection result with confidence
 */
struct RegimeResult {
    MarketRegime regime;
    double confidence;               // Classification confidence (0-1)
    MarketFeatures features;         // Raw features used
    std::vector<double> probabilities; // Full probability distribution
    bool is_transition;             // Currently in transition
    std::chrono::system_clock::time_point timestamp;
};

/**
 * @brief Advanced market regime detector
 * 
 * Combines statistical analysis with ML classification for robust
 * regime detection in real-time trading scenarios.
 */
class RegimeDetector {
public:
    explicit RegimeDetector(const RegimeConfig& config = RegimeConfig{});
    ~RegimeDetector() = default;

    // Initialize ML models and data structures
    bool initialize();

    // Detect current market regime from price/volume data
    RegimeResult detect_regime(
        const std::vector<double>& prices,
        const std::vector<double>& volumes,
        const std::chrono::system_clock::time_point& timestamp = std::chrono::system_clock::now()
    );

    // Update detector with new tick data
    void update_tick(double price, double volume, const std::chrono::system_clock::time_point& timestamp);

    // Get current regime without new data
    RegimeResult get_current_regime() const;

    // Get regime history for analysis
    std::vector<RegimeResult> get_regime_history(int max_items = 100) const;

    // Check if we're in a regime transition
    bool is_in_transition() const;

    // Get regime stability score (0-1, higher = more stable)
    double get_regime_stability() const;

    // Export regime data for training/improvement
    void export_regime_data(const std::string& filepath) const;

    // Load pre-trained regime classifier
    bool load_classifier(const std::string& model_path);

    // Train new regime classifier from historical data
    bool train_classifier(const std::vector<MarketFeatures>& features, 
                         const std::vector<MarketRegime>& labels);

private:
    RegimeConfig config_;
    
    // Data storage
    std::deque<double> price_history_;
    std::deque<double> volume_history_;
    std::deque<std::chrono::system_clock::time_point> timestamp_history_;
    std::vector<RegimeResult> regime_history_;
    
    // ML classifier
    torch::jit::script::Module regime_classifier_;
    bool classifier_loaded_;
    
    // Current state
    RegimeResult current_regime_;
    bool initialized_;

    // Feature extraction methods
    MarketFeatures extract_features(
        const std::vector<double>& prices,
        const std::vector<double>& volumes,
        const std::chrono::system_clock::time_point& timestamp
    ) const;

    // Statistical regime classification
    MarketRegime classify_statistical(const MarketFeatures& features) const;

    // ML regime classification
    std::vector<double> classify_ml(const MarketFeatures& features) const;

    // Combine statistical and ML results
    RegimeResult combine_results(
        MarketRegime statistical_regime,
        const std::vector<double>& ml_probabilities,
        const MarketFeatures& features,
        const std::chrono::system_clock::time_point& timestamp
    ) const;

    // Utility methods
    double calculate_trend(const std::vector<double>& prices) const;
    double calculate_volatility(const std::vector<double>& prices) const;
    double calculate_momentum(const std::vector<double>& prices) const;
    double calculate_mean_reversion(const std::vector<double>& prices) const;
    double calculate_acceleration(const std::vector<double>& prices) const;
    
    // Regime transition detection
    bool detect_transition(const std::vector<RegimeResult>& history) const;
    double calculate_regime_entropy(const std::vector<RegimeResult>& history) const;
    
    // String conversions
    std::string regime_to_string(MarketRegime regime) const;
    MarketRegime string_to_regime(const std::string& regime_str) const;
};

} // namespace regime
} // namespace archneuronx
