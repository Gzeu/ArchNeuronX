/**
 * @file technical_indicators.hpp
 * @brief Technical Indicators v2 - LibTorch GPU-Accelerated
 * @version 2.0.0
 * @date 2026-03-16
 *
 * RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, Stochastic,
 * Williams %R, CCI, EMA, SMA - all computed on LibTorch tensors
 * for seamless GPU acceleration.
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include <optional>
#include <cmath>
#include <stdexcept>

namespace ArchNeuronX {
namespace Data {
namespace Indicators {

// ============================================================
// Return types
// ============================================================
struct MACDResult {
    torch::Tensor macd_line;
    torch::Tensor signal_line;
    torch::Tensor histogram;
};

struct BollingerBands {
    torch::Tensor upper;
    torch::Tensor middle;
    torch::Tensor lower;
    torch::Tensor bandwidth;   // (upper - lower) / middle
    torch::Tensor percent_b;   // (price - lower) / (upper - lower)
};

struct Stochastic {
    torch::Tensor k;  // %K fast
    torch::Tensor d;  // %D smoothed
};

struct FeatureMatrix {
    torch::Tensor features;    // [T, F] tensor for model input
    std::vector<std::string> feature_names;
};

// ============================================================
// TechnicalAnalysis class - all methods static, GPU-friendly
// ============================================================
class TechnicalAnalysis {
public:
    // ------- Moving Averages --------------------------------

    /** Simple Moving Average */
    static torch::Tensor sma(const torch::Tensor& prices, int period);

    /** Exponential Moving Average (Wilder's smoothing) */
    static torch::Tensor ema(const torch::Tensor& prices, int period);

    /** Weighted Moving Average */
    static torch::Tensor wma(const torch::Tensor& prices, int period);

    // ------- Momentum Indicators ---------------------------

    /**
     * @brief Relative Strength Index
     * @param prices  Close prices tensor [T]
     * @param period  Look-back (default 14)
     * @return RSI values in [0, 100]
     */
    static torch::Tensor rsi(const torch::Tensor& prices, int period = 14);

    /**
     * @brief MACD - Moving Average Convergence/Divergence
     * @param prices    Close prices [T]
     * @param fast      Fast EMA period (default 12)
     * @param slow      Slow EMA period (default 26)
     * @param signal_p  Signal line period (default 9)
     */
    static MACDResult macd(const torch::Tensor& prices,
                           int fast = 12, int slow = 26, int signal_p = 9);

    /**
     * @brief Stochastic Oscillator %K / %D
     * @param high      High prices [T]
     * @param low       Low prices [T]
     * @param close     Close prices [T]
     * @param k_period  %K look-back (default 14)
     * @param d_period  %D smoothing (default 3)
     */
    static Stochastic stochastic(const torch::Tensor& high,
                                 const torch::Tensor& low,
                                 const torch::Tensor& close,
                                 int k_period = 14, int d_period = 3);

    /** Williams %R - overbought/oversold */
    static torch::Tensor williams_r(const torch::Tensor& high,
                                    const torch::Tensor& low,
                                    const torch::Tensor& close,
                                    int period = 14);

    /** Rate of Change (ROC) momentum */
    static torch::Tensor roc(const torch::Tensor& prices, int period = 10);

    // ------- Volatility Indicators -------------------------

    /**
     * @brief Bollinger Bands
     * @param prices   Close prices [T]
     * @param period   MA period (default 20)
     * @param num_std  Standard deviation multiplier (default 2.0)
     */
    static BollingerBands bollinger_bands(const torch::Tensor& prices,
                                          int period = 20,
                                          double num_std = 2.0);

    /**
     * @brief Average True Range - measures volatility
     * @param high    High prices [T]
     * @param low     Low prices [T]
     * @param close   Close prices [T]
     * @param period  ATR period (default 14)
     */
    static torch::Tensor atr(const torch::Tensor& high,
                             const torch::Tensor& low,
                             const torch::Tensor& close,
                             int period = 14);

    /** Historical Volatility (annualized) */
    static torch::Tensor historical_volatility(const torch::Tensor& prices,
                                               int period = 20);

    // ------- Volume Indicators -----------------------------

    /** On-Balance Volume */
    static torch::Tensor obv(const torch::Tensor& close,
                             const torch::Tensor& volume);

    /**
     * @brief Volume-Weighted Average Price
     * @param period  0 = session cumulative VWAP
     */
    static torch::Tensor vwap(const torch::Tensor& high,
                              const torch::Tensor& low,
                              const torch::Tensor& close,
                              const torch::Tensor& volume,
                              int period = 0);

    /** Accumulation/Distribution Line */
    static torch::Tensor adl(const torch::Tensor& high,
                             const torch::Tensor& low,
                             const torch::Tensor& close,
                             const torch::Tensor& volume);

    /** Chaikin Money Flow */
    static torch::Tensor cmf(const torch::Tensor& high,
                             const torch::Tensor& low,
                             const torch::Tensor& close,
                             const torch::Tensor& volume,
                             int period = 20);

    // ------- Trend Indicators ------------------------------

    /** Commodity Channel Index */
    static torch::Tensor cci(const torch::Tensor& high,
                             const torch::Tensor& low,
                             const torch::Tensor& close,
                             int period = 20);

    /** Average Directional Index (ADX) */
    static torch::Tensor adx(const torch::Tensor& high,
                             const torch::Tensor& low,
                             const torch::Tensor& close,
                             int period = 14);

    // ------- Feature Engineering ---------------------------

    /**
     * @brief Compute complete feature matrix for model input
     *
     * Produces a [T, 30+] tensor with all technical features
     * normalized and ready for Transformer/CNN/MLP models.
     *
     * Features: OHLCV + SMA(5,10,20,50) + EMA(12,26) +
     *           RSI(14) + MACD + BB(20) + ATR(14) +
     *           OBV + VWAP + Stochastic + Williams%R + CCI
     *
     * @param open    Open prices [T]
     * @param high    High prices [T]
     * @param low     Low prices [T]
     * @param close   Close prices [T]
     * @param volume  Volume [T]
     * @param normalize  Apply z-score normalization (default true)
     * @return FeatureMatrix with [T, F] tensor + feature names
     */
    static FeatureMatrix compute_feature_matrix(
        const torch::Tensor& open,
        const torch::Tensor& high,
        const torch::Tensor& low,
        const torch::Tensor& close,
        const torch::Tensor& volume,
        bool normalize = true);

private:
    /** Wilder's smoothing (used in RSI, ATR) */
    static torch::Tensor wilder_smooth(const torch::Tensor& data, int period);

    /** Rolling max over window */
    static torch::Tensor rolling_max(const torch::Tensor& data, int window);

    /** Rolling min over window */
    static torch::Tensor rolling_min(const torch::Tensor& data, int window);

    /** Rolling mean over window */
    static torch::Tensor rolling_mean(const torch::Tensor& data, int window);

    /** Rolling std over window */
    static torch::Tensor rolling_std(const torch::Tensor& data, int window);
};

} // namespace Indicators
} // namespace Data
} // namespace ArchNeuronX
