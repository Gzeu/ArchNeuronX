/**
 * @file technical_indicators.hpp
 * @brief Technical analysis indicators library for ArchNeuronX
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include <vector>
#include <optional>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "market_data.hpp"

namespace ArchNeuronX {
namespace Data {
namespace Indicators {

/**
 * @class TechnicalAnalysis
 * @brief Comprehensive technical analysis indicators calculator
 */
class TechnicalAnalysis {
public:
    /**
     * @brief Calculate Simple Moving Average (SMA)
     * @param prices Vector of prices
     * @param period Moving average period
     * @return Vector of SMA values
     */
    static std::vector<double> calculateSMA(const std::vector<double>& prices, int period);
    
    /**
     * @brief Calculate Exponential Moving Average (EMA)
     * @param prices Vector of prices
     * @param period Moving average period
     * @param smoothing_factor Smoothing factor (default: 2.0)
     * @return Vector of EMA values
     */
    static std::vector<double> calculateEMA(const std::vector<double>& prices, 
                                           int period, 
                                           double smoothing_factor = 2.0);
    
    /**
     * @brief Calculate Weighted Moving Average (WMA)
     * @param prices Vector of prices
     * @param period Moving average period
     * @return Vector of WMA values
     */
    static std::vector<double> calculateWMA(const std::vector<double>& prices, int period);
    
    /**
     * @brief Calculate Relative Strength Index (RSI)
     * @param prices Vector of prices
     * @param period RSI period (default: 14)
     * @return Vector of RSI values (0-100)
     */
    static std::vector<double> calculateRSI(const std::vector<double>& prices, int period = 14);
    
    /**
     * @brief Calculate MACD (Moving Average Convergence Divergence)
     * @param prices Vector of prices
     * @param fast_period Fast EMA period (default: 12)
     * @param slow_period Slow EMA period (default: 26)
     * @param signal_period Signal line period (default: 9)
     * @return Tuple of (MACD line, Signal line, Histogram)
     */
    static std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> 
        calculateMACD(const std::vector<double>& prices, 
                     int fast_period = 12, 
                     int slow_period = 26, 
                     int signal_period = 9);
    
    /**
     * @brief Calculate Bollinger Bands
     * @param prices Vector of prices
     * @param period Moving average period (default: 20)
     * @param std_dev Standard deviations (default: 2.0)
     * @return Tuple of (Upper band, Middle band, Lower band)
     */
    static std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
        calculateBollingerBands(const std::vector<double>& prices,
                               int period = 20,
                               double std_dev = 2.0);
    
    /**
     * @brief Calculate Stochastic Oscillator
     * @param highs Vector of high prices
     * @param lows Vector of low prices
     * @param closes Vector of close prices
     * @param k_period %K period (default: 14)
     * @param d_period %D period (default: 3)
     * @return Tuple of (%K values, %D values)
     */
    static std::tuple<std::vector<double>, std::vector<double>>
        calculateStochastic(const std::vector<double>& highs,
                           const std::vector<double>& lows,
                           const std::vector<double>& closes,
                           int k_period = 14,
                           int d_period = 3);
    
    /**
     * @brief Calculate Average True Range (ATR)
     * @param highs Vector of high prices
     * @param lows Vector of low prices
     * @param closes Vector of close prices
     * @param period ATR period (default: 14)
     * @return Vector of ATR values
     */
    static std::vector<double> calculateATR(const std::vector<double>& highs,
                                           const std::vector<double>& lows,
                                           const std::vector<double>& closes,
                                           int period = 14);
    
    /**
     * @brief Calculate Commodity Channel Index (CCI)
     * @param highs Vector of high prices
     * @param lows Vector of low prices
     * @param closes Vector of close prices
     * @param period CCI period (default: 20)
     * @return Vector of CCI values
     */
    static std::vector<double> calculateCCI(const std::vector<double>& highs,
                                           const std::vector<double>& lows,
                                           const std::vector<double>& closes,
                                           int period = 20);
    
    /**
     * @brief Calculate Williams %R
     * @param highs Vector of high prices
     * @param lows Vector of low prices
     * @param closes Vector of close prices
     * @param period Williams %R period (default: 14)
     * @return Vector of Williams %R values
     */
    static std::vector<double> calculateWilliamsR(const std::vector<double>& highs,
                                                  const std::vector<double>& lows,
                                                  const std::vector<double>& closes,
                                                  int period = 14);
    
    /**
     * @brief Calculate Volume Weighted Average Price (VWAP)
     * @param prices Vector of typical prices (HLC/3)
     * @param volumes Vector of volumes
     * @return Vector of VWAP values
     */
    static std::vector<double> calculateVWAP(const std::vector<double>& prices,
                                            const std::vector<double>& volumes);
    
    /**
     * @brief Calculate Money Flow Index (MFI)
     * @param highs Vector of high prices
     * @param lows Vector of low prices
     * @param closes Vector of close prices
     * @param volumes Vector of volumes
     * @param period MFI period (default: 14)
     * @return Vector of MFI values
     */
    static std::vector<double> calculateMFI(const std::vector<double>& highs,
                                           const std::vector<double>& lows,
                                           const std::vector<double>& closes,
                                           const std::vector<double>& volumes,
                                           int period = 14);
    
    /**
     * @brief Calculate Parabolic SAR
     * @param highs Vector of high prices
     * @param lows Vector of low prices
     * @param initial_af Initial acceleration factor (default: 0.02)
     * @param max_af Maximum acceleration factor (default: 0.20)
     * @return Vector of Parabolic SAR values
     */
    static std::vector<double> calculateParabolicSAR(const std::vector<double>& highs,
                                                     const std::vector<double>& lows,
                                                     double initial_af = 0.02,
                                                     double max_af = 0.20);
    
    /**
     * @brief Calculate Ichimoku Cloud components
     * @param highs Vector of high prices
     * @param lows Vector of low prices
     * @param closes Vector of close prices
     * @param tenkan_period Tenkan-sen period (default: 9)
     * @param kijun_period Kijun-sen period (default: 26)
     * @param senkou_period Senkou span B period (default: 52)
     * @return Tuple of (Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, Chikou Span)
     */
    static std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, 
                      std::vector<double>, std::vector<double>>
        calculateIchimoku(const std::vector<double>& highs,
                         const std::vector<double>& lows,
                         const std::vector<double>& closes,
                         int tenkan_period = 9,
                         int kijun_period = 26,
                         int senkou_period = 52);
    
    /**
     * @brief Calculate all technical indicators for OHLCV data
     * @param ohlcv_data Vector of OHLCV candlestick data
     * @return Vector of TechnicalIndicators structures
     */
    static std::vector<TechnicalIndicators> calculateAllIndicators(
        const std::vector<OHLCV>& ohlcv_data);
    
    /**
     * @brief Calculate volatility (standard deviation of returns)
     * @param prices Vector of prices
     * @param period Volatility calculation period
     * @return Vector of volatility values
     */
    static std::vector<double> calculateVolatility(const std::vector<double>& prices,
                                                  int period = 20);
    
    /**
     * @brief Calculate price returns
     * @param prices Vector of prices
     * @param logarithmic Use logarithmic returns (default: true)
     * @return Vector of returns
     */
    static std::vector<double> calculateReturns(const std::vector<double>& prices,
                                               bool logarithmic = true);
    
    /**
     * @brief Calculate correlation between two price series
     * @param prices1 First price series
     * @param prices2 Second price series
     * @param period Correlation calculation period
     * @return Vector of correlation values
     */
    static std::vector<double> calculateCorrelation(const std::vector<double>& prices1,
                                                   const std::vector<double>& prices2,
                                                   int period = 20);
    
    /**
     * @brief Detect price patterns
     * @param ohlcv_data Vector of OHLCV data
     * @return Vector of pattern names detected
     */
    static std::vector<std::string> detectPatterns(const std::vector<OHLCV>& ohlcv_data);
    
private:
    /**
     * @brief Calculate standard deviation
     * @param values Vector of values
     * @param mean Mean of the values
     * @return Standard deviation
     */
    static double calculateStdDev(const std::vector<double>& values, double mean);
    
    /**
     * @brief Calculate mean of a vector
     * @param values Vector of values
     * @return Mean value
     */
    static double calculateMean(const std::vector<double>& values);
    
    /**
     * @brief Find maximum value in range
     * @param values Vector of values
     * @param start Start index
     * @param end End index
     * @return Maximum value
     */
    static double findMax(const std::vector<double>& values, size_t start, size_t end);
    
    /**
     * @brief Find minimum value in range
     * @param values Vector of values
     * @param start Start index
     * @param end End index
     * @return Minimum value
     */
    static double findMin(const std::vector<double>& values, size_t start, size_t end);
    
    /**
     * @brief Check if a candlestick is bullish
     * @param candle OHLCV candlestick
     * @return True if bullish (close > open)
     */
    static bool isBullish(const OHLCV& candle);
    
    /**
     * @brief Check if a candlestick is bearish
     * @param candle OHLCV candlestick
     * @return True if bearish (close < open)
     */
    static bool isBearish(const OHLCV& candle);
    
    /**
     * @brief Calculate candlestick body size
     * @param candle OHLCV candlestick
     * @return Body size as percentage of total range
     */
    static double getBodySize(const OHLCV& candle);
    
    /**
     * @brief Check for doji pattern
     * @param candle OHLCV candlestick
     * @param threshold Doji threshold (default: 0.1%)
     * @return True if doji pattern
     */
    static bool isDoji(const OHLCV& candle, double threshold = 0.001);
};

/**
 * @brief Helper function to extract price series from OHLCV data
 * @param ohlcv_data Vector of OHLCV data
 * @param price_type Price type ("open", "high", "low", "close")
 * @return Vector of prices
 */
std::vector<double> extractPrices(const std::vector<OHLCV>& ohlcv_data,
                                 const std::string& price_type);

/**
 * @brief Extract volumes from OHLCV data
 * @param ohlcv_data Vector of OHLCV data
 * @return Vector of volumes
 */
std::vector<double> extractVolumes(const std::vector<OHLCV>& ohlcv_data);

/**
 * @brief Calculate typical prices (HLC/3) from OHLCV data
 * @param ohlcv_data Vector of OHLCV data
 * @return Vector of typical prices
 */
std::vector<double> calculateTypicalPrices(const std::vector<OHLCV>& ohlcv_data);

} // namespace Indicators
} // namespace Data
} // namespace ArchNeuronX