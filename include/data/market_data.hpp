/**
 * @file market_data.hpp
 * @brief Market data structures and interfaces for ArchNeuronX
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <map>
#include <optional>
#include <memory>
#include "core/logger.hpp"

namespace ArchNeuronX {
namespace Data {

/**
 * @enum TimeFrame
 * @brief Supported timeframes for market data
 */
enum class TimeFrame {
    MINUTE_1 = 1,      ///< 1 minute
    MINUTE_5 = 5,      ///< 5 minutes
    MINUTE_15 = 15,    ///< 15 minutes
    MINUTE_30 = 30,    ///< 30 minutes
    HOUR_1 = 60,       ///< 1 hour
    HOUR_4 = 240,      ///< 4 hours
    DAY_1 = 1440,      ///< 1 day
    WEEK_1 = 10080     ///< 1 week
};

/**
 * @enum MarketDataType
 * @brief Types of market data
 */
enum class MarketDataType {
    OHLCV,        ///< Open, High, Low, Close, Volume
    TICK,         ///< Tick data
    ORDER_BOOK,   ///< Order book data
    TRADES        ///< Individual trades
};

/**
 * @struct OHLCV
 * @brief Open, High, Low, Close, Volume candlestick data
 */
struct OHLCV {
    std::chrono::system_clock::time_point timestamp;
    double open = 0.0;
    double high = 0.0;
    double low = 0.0;
    double close = 0.0;
    double volume = 0.0;
    int64_t trades_count = 0;
    
    /**
     * @brief Calculate typical price (HLC/3)
     * @return Typical price
     */
    double getTypicalPrice() const {
        return (high + low + close) / 3.0;
    }
    
    /**
     * @brief Calculate weighted close price (HLCC/4)
     * @return Weighted close price
     */
    double getWeightedClose() const {
        return (high + low + close + close) / 4.0;
    }
    
    /**
     * @brief Calculate true range
     * @param prev_close Previous candle's close price
     * @return True range value
     */
    double getTrueRange(double prev_close) const {
        double hl = high - low;
        double hc = std::abs(high - prev_close);
        double lc = std::abs(low - prev_close);
        return std::max({hl, hc, lc});
    }
    
    /**
     * @brief Convert to JSON string
     * @return JSON representation
     */
    std::string toJson() const;
    
    /**
     * @brief Create from JSON string
     * @param json_str JSON string
     * @return OHLCV object
     */
    static OHLCV fromJson(const std::string& json_str);
    
    /**
     * @brief Check if candle is valid
     * @return True if all values are valid
     */
    bool isValid() const {
        return open > 0 && high > 0 && low > 0 && close > 0 && 
               high >= std::max({open, close, low}) &&
               low <= std::min({open, close, high});
    }
};

/**
 * @struct TickData
 * @brief Individual tick/trade data
 */
struct TickData {
    std::chrono::system_clock::time_point timestamp;
    std::string symbol;
    double price = 0.0;
    double quantity = 0.0;
    bool is_buyer_maker = false;
    std::string trade_id;
    
    /**
     * @brief Convert to JSON string
     * @return JSON representation
     */
    std::string toJson() const;
};

/**
 * @struct OrderBookEntry
 * @brief Order book bid/ask entry
 */
struct OrderBookEntry {
    double price = 0.0;
    double quantity = 0.0;
    
    OrderBookEntry() = default;
    OrderBookEntry(double p, double q) : price(p), quantity(q) {}
};

/**
 * @struct OrderBook
 * @brief Order book snapshot
 */
struct OrderBook {
    std::chrono::system_clock::time_point timestamp;
    std::string symbol;
    std::vector<OrderBookEntry> bids;  ///< Sorted by price descending
    std::vector<OrderBookEntry> asks;  ///< Sorted by price ascending
    
    /**
     * @brief Get best bid price
     * @return Best bid price or 0 if no bids
     */
    double getBestBid() const {
        return bids.empty() ? 0.0 : bids.front().price;
    }
    
    /**
     * @brief Get best ask price
     * @return Best ask price or 0 if no asks
     */
    double getBestAsk() const {
        return asks.empty() ? 0.0 : asks.front().price;
    }
    
    /**
     * @brief Calculate bid-ask spread
     * @return Spread in price units
     */
    double getSpread() const {
        double bid = getBestBid();
        double ask = getBestAsk();
        return (bid > 0 && ask > 0) ? ask - bid : 0.0;
    }
    
    /**
     * @brief Calculate mid price
     * @return Mid price between best bid and ask
     */
    double getMidPrice() const {
        double bid = getBestBid();
        double ask = getBestAsk();
        return (bid > 0 && ask > 0) ? (bid + ask) / 2.0 : 0.0;
    }
};

/**
 * @struct MarketInfo
 * @brief Trading pair market information
 */
struct MarketInfo {
    std::string symbol;
    std::string base_asset;
    std::string quote_asset;
    double tick_size = 0.0;
    double lot_size = 0.0;
    double min_quantity = 0.0;
    double max_quantity = 0.0;
    double min_notional = 0.0;
    bool is_trading = true;
    
    /**
     * @brief Round price to valid tick size
     * @param price Input price
     * @return Rounded price
     */
    double roundPrice(double price) const;
    
    /**
     * @brief Round quantity to valid lot size
     * @param quantity Input quantity
     * @return Rounded quantity
     */
    double roundQuantity(double quantity) const;
};

/**
 * @struct TechnicalIndicators
 * @brief Container for technical analysis indicators
 */
struct TechnicalIndicators {
    // Moving Averages
    std::optional<double> sma_20;      ///< Simple Moving Average 20
    std::optional<double> sma_50;      ///< Simple Moving Average 50
    std::optional<double> ema_12;      ///< Exponential Moving Average 12
    std::optional<double> ema_26;      ///< Exponential Moving Average 26
    
    // Oscillators
    std::optional<double> rsi_14;      ///< Relative Strength Index 14
    std::optional<double> stoch_k;     ///< Stochastic %K
    std::optional<double> stoch_d;     ///< Stochastic %D
    
    // MACD
    std::optional<double> macd_line;   ///< MACD line
    std::optional<double> macd_signal; ///< MACD signal line
    std::optional<double> macd_hist;   ///< MACD histogram
    
    // Bollinger Bands
    std::optional<double> bb_upper;    ///< Bollinger Band upper
    std::optional<double> bb_middle;   ///< Bollinger Band middle (SMA)
    std::optional<double> bb_lower;    ///< Bollinger Band lower
    
    // Volatility
    std::optional<double> atr_14;      ///< Average True Range 14
    std::optional<double> volatility;  ///< Price volatility
    
    // Volume
    std::optional<double> volume_sma;  ///< Volume Simple Moving Average
    std::optional<double> vwap;        ///< Volume Weighted Average Price
    
    /**
     * @brief Convert to feature vector for ML
     * @return Vector of indicator values (NaN for missing values)
     */
    std::vector<double> toFeatureVector() const;
    
    /**
     * @brief Check if all indicators are calculated
     * @return True if no missing values
     */
    bool isComplete() const;
};

/**
 * @struct MarketDataPoint
 * @brief Complete market data point with indicators
 */
struct MarketDataPoint {
    std::string symbol;
    TimeFrame timeframe;
    OHLCV ohlcv;
    TechnicalIndicators indicators;
    std::optional<OrderBook> order_book;
    std::vector<TickData> recent_trades;
    
    /**
     * @brief Convert to feature vector for ML training
     * @return Complete feature vector
     */
    std::vector<double> toMLFeatureVector() const;
    
    /**
     * @brief Get timestamp
     * @return Data point timestamp
     */
    std::chrono::system_clock::time_point getTimestamp() const {
        return ohlcv.timestamp;
    }
};

/**
 * @brief Convert TimeFrame to string
 * @param tf TimeFrame value
 * @return String representation
 */
std::string timeFrameToString(TimeFrame tf);

/**
 * @brief Convert string to TimeFrame
 * @param tf_str String representation
 * @return TimeFrame value
 */
TimeFrame stringToTimeFrame(const std::string& tf_str);

/**
 * @brief Convert TimeFrame to minutes
 * @param tf TimeFrame value
 * @return Number of minutes
 */
int timeFrameToMinutes(TimeFrame tf);

/**
 * @brief Calculate price change percentage
 * @param old_price Previous price
 * @param new_price Current price
 * @return Percentage change
 */
double calculatePriceChange(double old_price, double new_price);

/**
 * @brief Validate OHLCV data consistency
 * @param data Vector of OHLCV data
 * @return Number of invalid entries found
 */
size_t validateOHLCVData(const std::vector<OHLCV>& data);

} // namespace Data
} // namespace ArchNeuronX