#pragma once
// ============================================================
// ArchNeuronX v2 - Market Data Preprocessor
// Computes: RSI, MACD, Bollinger Bands, ATR, OBV, Stoch, ADX
// Input: OHLCV vectors → Output: Feature tensor
// ============================================================
#include <torch/torch.h>
#include <vector>
#include <string>
#include <unordered_map>

namespace archneuronx {
namespace data {

struct OHLCVBar {
    double open, high, low, close, volume;
    int64_t timestamp_ms;
};

struct TechnicalFeatures {
    // Trend
    double rsi_14;
    double rsi_7;
    double macd_line;
    double macd_signal;
    double macd_histogram;
    double adx_14;
    double di_plus_14;
    double di_minus_14;
    // Volatility
    double bb_upper;     // Bollinger Band upper
    double bb_middle;
    double bb_lower;
    double bb_pct_b;     // %B position within bands
    double bb_bandwidth;
    double atr_14;
    // Volume
    double obv;          // On-Balance Volume
    double volume_sma_20;
    double volume_ratio; // current / sma
    // Momentum
    double stoch_k_14;
    double stoch_d_3;
    double roc_10;       // Rate of Change
    // Price
    double sma_20;
    double sma_50;
    double ema_12;
    double ema_26;
    double price_sma20_ratio;
    double price_normalized;  // (close - min) / (max - min)
};

class DataPreprocessor {
public:
    explicit DataPreprocessor(size_t sequence_length = 168,  // 1 week hourly
                               bool normalize = true);

    // Process raw OHLCV bars → feature tensor
    // Output shape: [sequence_length, num_features]
    [[nodiscard]] torch::Tensor process(
        const std::vector<OHLCVBar>& bars) const;

    // Process batch of sequences
    // Output shape: [batch, sequence_length, num_features]
    [[nodiscard]] torch::Tensor process_batch(
        const std::vector<std::vector<OHLCVBar>>& batch) const;

    // Extract only technical features (no normalization)
    [[nodiscard]] TechnicalFeatures compute_features(
        const std::vector<OHLCVBar>& bars) const;

    // Feature count (for model input_size config)
    [[nodiscard]] static constexpr size_t feature_count() { return 25; }

    // Feature names for interpretability
    [[nodiscard]] static std::vector<std::string> feature_names();

private:
    size_t sequence_length_;
    bool normalize_;

    // --- Technical Indicator Implementations ---
    [[nodiscard]] std::vector<double> rsi(
        const std::vector<double>& closes, int period) const;

    [[nodiscard]] std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    macd(const std::vector<double>& closes,
          int fast = 12, int slow = 26, int signal = 9) const;

    [[nodiscard]] std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    bollinger_bands(const std::vector<double>& closes,
                     int period = 20, double num_std = 2.0) const;

    [[nodiscard]] std::vector<double> atr(
        const std::vector<double>& highs,
        const std::vector<double>& lows,
        const std::vector<double>& closes,
        int period = 14) const;

    [[nodiscard]] std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    adx(const std::vector<double>& highs,
         const std::vector<double>& lows,
         const std::vector<double>& closes,
         int period = 14) const;

    [[nodiscard]] std::vector<double> obv(
        const std::vector<double>& closes,
        const std::vector<double>& volumes) const;

    [[nodiscard]] std::tuple<std::vector<double>, std::vector<double>>
    stochastic(const std::vector<double>& highs,
                const std::vector<double>& lows,
                const std::vector<double>& closes,
                int k_period = 14, int d_period = 3) const;

    [[nodiscard]] std::vector<double> ema(
        const std::vector<double>& data, int period) const;

    [[nodiscard]] std::vector<double> sma(
        const std::vector<double>& data, int period) const;

    // Min-max normalization per feature
    [[nodiscard]] torch::Tensor normalize_features(
        const torch::Tensor& features) const;
};

} // namespace data
} // namespace archneuronx
