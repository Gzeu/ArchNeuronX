/**
 * @file market_data.cpp
 * @brief Implementation of market data structures and utilities
 * @author George Pricop
 * @date 2025-10-02
 */

#include "data/market_data.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace ArchNeuronX {
namespace Data {

std::string OHLCV::toJson() const {
    std::ostringstream oss;
    auto time_t = std::chrono::system_clock::to_time_t(timestamp);
    
    oss << "{";
    oss << "\"timestamp\":" << std::chrono::duration_cast<std::chrono::seconds>(timestamp.time_since_epoch()).count() << ",";
    oss << "\"open\":" << std::fixed << std::setprecision(8) << open << ",";
    oss << "\"high\":" << std::fixed << std::setprecision(8) << high << ",";
    oss << "\"low\":" << std::fixed << std::setprecision(8) << low << ",";
    oss << "\"close\":" << std::fixed << std::setprecision(8) << close << ",";
    oss << "\"volume\":" << std::fixed << std::setprecision(2) << volume << ",";
    oss << "\"trades_count\":" << trades_count;
    oss << "}";
    
    return oss.str();
}

OHLCV OHLCV::fromJson(const std::string& json_str) {
    // Simple JSON parsing - in production, use proper JSON library
    OHLCV ohlcv;
    // Implementation would parse JSON string
    // For now, return default constructed object
    return ohlcv;
}

std::string TickData::toJson() const {
    std::ostringstream oss;
    auto time_t = std::chrono::system_clock::to_time_t(timestamp);
    
    oss << "{";
    oss << "\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(timestamp.time_since_epoch()).count() << ",";
    oss << "\"symbol\":\"" << symbol << "\",";
    oss << "\"price\":" << std::fixed << std::setprecision(8) << price << ",";
    oss << "\"quantity\":" << std::fixed << std::setprecision(8) << quantity << ",";
    oss << "\"is_buyer_maker\":" << (is_buyer_maker ? "true" : "false") << ",";
    oss << "\"trade_id\":\"" << trade_id << "\"";
    oss << "}";
    
    return oss.str();
}

double MarketInfo::roundPrice(double price) const {
    if (tick_size <= 0) return price;
    return std::round(price / tick_size) * tick_size;
}

double MarketInfo::roundQuantity(double quantity) const {
    if (lot_size <= 0) return quantity;
    return std::round(quantity / lot_size) * lot_size;
}

double Position::calculatePnL(double current_market_price) const {
    if (size == 0) return 0.0;
    return size * (current_market_price - entry_price);
}

std::vector<double> TechnicalIndicators::toFeatureVector() const {
    std::vector<double> features;
    
    // Moving Averages
    features.push_back(sma_20.value_or(std::numeric_limits<double>::quiet_NaN()));
    features.push_back(sma_50.value_or(std::numeric_limits<double>::quiet_NaN()));
    features.push_back(ema_12.value_or(std::numeric_limits<double>::quiet_NaN()));
    features.push_back(ema_26.value_or(std::numeric_limits<double>::quiet_NaN()));
    
    // Oscillators
    features.push_back(rsi_14.value_or(std::numeric_limits<double>::quiet_NaN()));
    features.push_back(stoch_k.value_or(std::numeric_limits<double>::quiet_NaN()));
    features.push_back(stoch_d.value_or(std::numeric_limits<double>::quiet_NaN()));
    
    // MACD
    features.push_back(macd_line.value_or(std::numeric_limits<double>::quiet_NaN()));
    features.push_back(macd_signal.value_or(std::numeric_limits<double>::quiet_NaN()));
    features.push_back(macd_hist.value_or(std::numeric_limits<double>::quiet_NaN()));
    
    // Bollinger Bands
    features.push_back(bb_upper.value_or(std::numeric_limits<double>::quiet_NaN()));
    features.push_back(bb_middle.value_or(std::numeric_limits<double>::quiet_NaN()));
    features.push_back(bb_lower.value_or(std::numeric_limits<double>::quiet_NaN()));
    
    // Volatility
    features.push_back(atr_14.value_or(std::numeric_limits<double>::quiet_NaN()));
    features.push_back(volatility.value_or(std::numeric_limits<double>::quiet_NaN()));
    
    // Volume
    features.push_back(volume_sma.value_or(std::numeric_limits<double>::quiet_NaN()));
    features.push_back(vwap.value_or(std::numeric_limits<double>::quiet_NaN()));
    
    return features;
}

bool TechnicalIndicators::isComplete() const {
    return sma_20.has_value() && sma_50.has_value() && ema_12.has_value() && ema_26.has_value() &&
           rsi_14.has_value() && stoch_k.has_value() && stoch_d.has_value() &&
           macd_line.has_value() && macd_signal.has_value() && macd_hist.has_value() &&
           bb_upper.has_value() && bb_middle.has_value() && bb_lower.has_value() &&
           atr_14.has_value() && volatility.has_value() &&
           volume_sma.has_value() && vwap.has_value();
}

std::vector<double> MarketDataPoint::toMLFeatureVector() const {
    std::vector<double> features;
    
    // OHLCV data
    features.push_back(ohlcv.open);
    features.push_back(ohlcv.high);
    features.push_back(ohlcv.low);
    features.push_back(ohlcv.close);
    features.push_back(ohlcv.volume);
    
    // Technical indicators
    auto indicator_features = indicators.toFeatureVector();
    features.insert(features.end(), indicator_features.begin(), indicator_features.end());
    
    // Order book features (if available)
    if (order_book.has_value()) {
        features.push_back(order_book->getBestBid());
        features.push_back(order_book->getBestAsk());
        features.push_back(order_book->getSpread());
        features.push_back(order_book->getMidPrice());
    } else {
        // Add NaN values for missing order book data
        for (int i = 0; i < 4; ++i) {
            features.push_back(std::numeric_limits<double>::quiet_NaN());
        }
    }
    
    return features;
}

std::string timeFrameToString(TimeFrame tf) {
    switch (tf) {
        case TimeFrame::MINUTE_1: return "1m";
        case TimeFrame::MINUTE_5: return "5m";
        case TimeFrame::MINUTE_15: return "15m";
        case TimeFrame::MINUTE_30: return "30m";
        case TimeFrame::HOUR_1: return "1h";
        case TimeFrame::HOUR_4: return "4h";
        case TimeFrame::DAY_1: return "1d";
        case TimeFrame::WEEK_1: return "1w";
        default: return "unknown";
    }
}

TimeFrame stringToTimeFrame(const std::string& tf_str) {
    if (tf_str == "1m") return TimeFrame::MINUTE_1;
    if (tf_str == "5m") return TimeFrame::MINUTE_5;
    if (tf_str == "15m") return TimeFrame::MINUTE_15;
    if (tf_str == "30m") return TimeFrame::MINUTE_30;
    if (tf_str == "1h") return TimeFrame::HOUR_1;
    if (tf_str == "4h") return TimeFrame::HOUR_4;
    if (tf_str == "1d") return TimeFrame::DAY_1;
    if (tf_str == "1w") return TimeFrame::WEEK_1;
    return TimeFrame::HOUR_1; // Default
}

int timeFrameToMinutes(TimeFrame tf) {
    return static_cast<int>(tf);
}

double calculatePriceChange(double old_price, double new_price) {
    if (old_price == 0.0) return 0.0;
    return ((new_price - old_price) / old_price) * 100.0;
}

size_t validateOHLCVData(const std::vector<OHLCV>& data) {
    size_t invalid_count = 0;
    
    for (const auto& candle : data) {
        if (!candle.isValid()) {
            invalid_count++;
        }
    }
    
    return invalid_count;
}

} // namespace Data
} // namespace ArchNeuronX