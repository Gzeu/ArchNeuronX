/**
 * @file openclaw_core.cpp
 * @brief Core OpenCLaw functionality implementation
 * @author OpenCLaw Team
 * @date 2025-10-02
 */

#include "openclaw/openclaw_core.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <thread>
#include <future>

namespace OpenCLawCore {

OpenCLawEngine::OpenCLawEngine(const Config& config) 
    : config_(config), initialized_(false) {
    
    // Initialize default venues
    if (config_.supported_venues.empty()) {
        config_.supported_venues = {"binance", "coinbase", "kraken", "bybit", "okx", "huobi"};
    }
    
    // Initialize venue metrics with default values
    for (const auto& venue : config_.supported_venues) {
        std::map<std::string, double> metrics;
        metrics["fill_rate"] = 0.98;
        metrics["slippage_bps"] = 2.0;
        metrics["latency_ms"] = 50.0;
        metrics["liquidity_score"] = 0.8;
        venue_metrics_[venue] = metrics;
    }
}

bool OpenCLawEngine::initialize() {
    std::cout << "🚀 Initializing OpenCLaw Core Engine..." << std::endl;
    
    try {
        // Validate configuration
        if (config_.max_slippage_bps <= 0 || config_.min_fill_rate <= 0) {
            std::cerr << "❌ Invalid configuration parameters" << std::endl;
            return false;
        }
        
        initialized_ = true;
        std::cout << "✅ OpenCLaw Core Engine initialized" << std::endl;
        std::cout << "📍 Supported venues: ";
        for (const auto& venue : config_.supported_venues) {
            std::cout << venue << " ";
        }
        std::cout << std::endl;
        std::cout << "📊 Smart routing: " << (config_.enable_smart_routing ? "enabled" : "disabled") << std::endl;
        std::cout << "🔬 Microstructure analysis: " << (config_.enable_microstructure_analysis ? "enabled" : "disabled") << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to initialize OpenCLaw Engine: " << e.what() << std::endl;
        return false;
    }
}

std::vector<TradingSignal> OpenCLawEngine::generate_signals(const std::vector<MarketData>& market_data) {
    if (!initialized_) {
        std::cerr << "❌ Engine not initialized" << std::endl;
        return {};
    }
    
    std::vector<TradingSignal> signals;
    
    if (market_data.empty()) {
        return signals;
    }
    
    std::cout << "📈 Generating signals from " << market_data.size() << " data points" << std::endl;
    
    // Analyze market data
    double rsi = calculate_rsi(market_data);
    double macd = calculate_macd(market_data);
    auto [upper_band, lower_band] = calculate_bollinger_bands(market_data);
    double regime = detect_market_regime(market_data);
    
    const auto& latest = market_data.back();
    
    // Generate primary signal
    TradingSignal signal;
    signal.symbol = latest.symbol;
    signal.timestamp = latest.timestamp;
    signal.indicators["RSI"] = rsi;
    signal.indicators["MACD"] = macd;
    signal.indicators["Upper_BB"] = upper_band;
    signal.indicators["Lower_BB"] = lower_band;
    signal.indicators["Regime"] = regime;
    
    // Signal analysis logic
    Signal price_action = analyze_price_action(market_data);
    double signal_strength = calculate_signal_strength(market_data);
    
    // Convert to trading signal
    if (price_action == Signal::BUY || price_action == Signal::STRONG_BUY) {
        signal.signal = price_action;
        signal.confidence = std::abs(signal_strength);
        signal.price_target = latest.price * 1.02;  // 2% target
        signal.stop_loss = latest.price * 0.98;      // 2% stop loss
        signal.take_profit = latest.price * 1.04;     // 4% take profit
        signal.reasoning = "RSI oversold + MACD bullish + price action";
    } else if (price_action == Signal::SELL || price_action == Signal::STRONG_SELL) {
        signal.signal = price_action;
        signal.confidence = std::abs(signal_strength);
        signal.price_target = latest.price * 0.98;  // 2% target
        signal.stop_loss = latest.price * 1.02;      // 2% stop loss
        signal.take_profit = latest.price * 0.96;     // 4% take profit
        signal.reasoning = "RSI overbought + MACD bearish + price action";
    } else {
        signal.signal = Signal::NEUTRAL;
        signal.confidence = 0.3;
        signal.price_target = latest.price;
        signal.stop_loss = latest.price * 0.99;
        signal.take_profit = latest.price * 1.01;
        signal.reasoning = "Neutral market conditions";
    }
    
    signals.push_back(signal);
    
    std::cout << "📊 Generated signal: " << static_cast<int>(signal.signal) 
              << " with confidence: " << signal.confidence << std::endl;
    
    return signals;
}

ExecutionResult OpenCLawEngine::execute_order(const TradingSignal& signal, double quantity) {
    if (!initialized_) {
        std::cerr << "❌ Engine not initialized" << std::endl;
        return {};
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "🔄 Executing order: " << signal.symbol 
              << " " << static_cast<int>(signal.signal) << std::endl;
    
    ExecutionResult result;
    result.timestamp = std::chrono::system_clock::now();
    
    try {
        // Select best venue
        std::string best_venue = select_best_venue(signal.symbol, "MARKET");
        result.venue = best_venue;
        
        // Analyze market microstructure for execution strategy
        MarketData market_data;
        market_data.symbol = signal.symbol;
        market_data.price = signal.price_target;
        market_data.bid = signal.price_target * 0.999;  // Simulated bid
        market_data.ask = signal.price_target * 1.001;  // Simulated ask
        market_data.spread = market_data.ask - market_data.bid;
        market_data.timestamp = result.timestamp;
        
        std::string execution_strategy = determine_optimal_algorithm(best_venue, market_data);
        result.execution_algorithm = execution_strategy;
        
        // Simulate order execution
        std::this_thread::sleep_for(std::chrono::milliseconds(20 + (rand() % 30)));
        
        // Calculate execution metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        result.latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        // Simulate fill result (95% success rate)
        bool success = (rand() % 100) < 95;
        result.success = success;
        
        if (success) {
            result.fill_price = market_data.bid + (rand() % 100) / 10000.0;  // Random slippage
            result.filled_quantity = quantity;
            result.slippage_bps = (result.fill_price - market_data.bid) / market_data.bid * 10000;
        } else {
            result.fill_price = 0.0;
            result.filled_quantity = 0.0;
            result.slippage_bps = 0.0;
        }
        
        // Update venue metrics
        double fill_rate = success ? 1.0 : 0.0;
        update_venue_metrics(best_venue, fill_rate, result.slippage_bps, result.latency_ms);
        
        std::cout << "✅ Order executed on " << best_venue << std::endl;
        std::cout << "📊 Fill price: $" << result.fill_price << std::endl;
        std::cout << "📊 Slippage: " << result.slippage_bps << " bps" << std::endl;
        std::cout << "📊 Latency: " << result.latency_ms << " ms" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Order execution failed: " << e.what() << std::endl;
        result.success = false;
    }
    
    return result;
}

std::map<std::string, std::map<std::string, double>> OpenCLawEngine::get_venue_metrics() const {
    return venue_metrics_;
}

void OpenCLawEngine::update_venue_metrics(const std::string& venue, 
                                         double fill_rate,
                                         double slippage_bps,
                                         double latency_ms) {
    if (!venue_metrics_.count(venue)) {
        return;
    }
    
    auto& metrics = venue_metrics_[venue];
    
    // Exponential moving average for metrics
    const double alpha = 0.1;  // Smoothing factor
    
    metrics["fill_rate"] = alpha * fill_rate + (1 - alpha) * metrics["fill_rate"];
    metrics["slippage_bps"] = alpha * slippage_bps + (1 - alpha) * metrics["slippage_bps"];
    metrics["latency_ms"] = alpha * latency_ms + (1 - alpha) * metrics["latency_ms"];
    
    std::cout << "📊 Updated " << venue << " metrics" << std::endl;
}

std::string OpenCLawEngine::analyze_microstructure(const MarketData& data) {
    if (!config_.enable_microstructure_analysis) {
        return "simple";
    }
    
    // Calculate spread metrics
    double spread_pct = (data.spread / data.price) * 100;
    
    // Determine market conditions based on spread
    std::string strategy;
    if (spread_pct < 0.01) {  // Tight spread
        strategy = "passive_liquidity_provision";
    } else if (spread_pct < 0.05) {  // Normal spread
        strategy = "balanced_execution";
    } else if (spread_pct < 0.1) {  // Wide spread
        strategy = "aggressive_execution";
    } else {  // Very wide spread
        strategy = "opportunistic_execution";
    }
    
    std::cout << "🔬 Microstructure analysis: " << strategy 
              << " (spread: " << spread_pct << "%)" << std::endl;
    
    return strategy;
}

std::string OpenCLawEngine::select_best_venue(const std::string& symbol, 
                                          const std::string& order_type) {
    if (!config_.enable_smart_routing) {
        return config_.supported_venues[0];  // Return first venue
    }
    
    // Rank venues by performance metrics
    std::vector<std::pair<std::string, double>> venue_scores;
    
    for (const auto& [venue, metrics] : venue_metrics_) {
        double fill_rate = metrics.at("fill_rate");
        double slippage = metrics.at("slippage_bps");
        double latency = metrics.at("latency_ms");
        
        // Calculate venue score (weighted combination)
        double score = (fill_rate * 0.5) + 
                      ((10.0 - slippage) / 10.0) * 0.3 +  // Normalize slippage
                      ((100.0 - latency) / 100.0) * 0.2;  // Normalize latency
        
        venue_scores.push_back({venue, score});
    }
    
    // Sort venues by score
    std::sort(venue_scores.begin(), venue_scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Return best venue
    return venue_scores.empty() ? config_.supported_venues[0] : venue_scores[0].first;
}

std::string OpenCLawEngine::determine_optimal_algorithm(const std::string& venue, 
                                                   const MarketData& market_data) {
    // Simplified algorithm selection based on venue and market conditions
    if (market_data.spread < 0.01) {
        return "twap";  // Use TWAP for tight spreads
    } else if (venue == "binance") {
        return "limit";  // Use limit orders on Binance
    } else if (venue == "coinbase") {
        return "market";  // Use market orders on Coinbase
    } else {
        return "adaptive";  // Default to adaptive
    }
}

// Technical analysis implementations

double OpenCLawEngine::calculate_rsi(const std::vector<MarketData>& data, int period) {
    if (data.size() < period + 1) {
        return 50.0;  // Default RSI
    }
    
    // Calculate RSI (simplified)
    double gains = 0.0, losses = 0.0;
    
    for (size_t i = 1; i < data.size(); ++i) {
        double change = data[i].price - data[i-1].price;
        if (change > 0) {
            gains += change;
        } else {
            losses -= change;
        }
    }
    
    double avg_gain = gains / period;
    double avg_loss = std::abs(losses) / period;
    double rs = avg_gain / avg_loss;
    
    return 100.0 - (100.0 / (1.0 + rs));
}

double OpenCLawEngine::calculate_macd(const std::vector<MarketData>& data) {
    if (data.size() < 26) {
        return 0.0;  // Default MACD
    }
    
    // Simplified MACD calculation
    double ema_12 = data[0].price;
    double ema_26 = data[0].price;
    
    for (size_t i = 1; i < data.size(); ++i) {
        ema_12 = ema_12 * 0.9231 + data[i].price * 0.0769;  // 12-period EMA
        ema_26 = ema_26 * 0.9615 + data[i].price * 0.0385;  // 26-period EMA
    }
    
    return ema_12 - ema_26;  // MACD line
}

std::pair<double, double> OpenCLawEngine::calculate_bollinger_bands(const std::vector<MarketData>& data) {
    if (data.size() < 20) {
        return {data.back().price * 1.02, data.back().price * 0.98};  // Default bands
    }
    
    // Calculate 20-period SMA and standard deviation
    double sum = 0.0;
    for (const auto& point : data) {
        sum += point.price;
    }
    double sma = sum / data.size();
    
    double variance = 0.0;
    for (const auto& point : data) {
        variance += std::pow(point.price - sma, 2);
    }
    variance /= data.size();
    double std_dev = std::sqrt(variance);
    
    double upper_band = sma + (2.0 * std_dev);
    double lower_band = sma - (2.0 * std_dev);
    
    return {upper_band, lower_band};
}

double OpenCLawEngine::detect_market_regime(const std::vector<MarketData>& data) {
    if (data.size() < 50) {
        return 0.0;  // Default regime
    }
    
    // Calculate volatility over last 50 periods
    double returns_variance = 0.0;
    for (size_t i = 1; i < std::min(size_t(50), data.size()); ++i) {
        double ret = (data[i].price - data[i-1].price) / data[i-1].price;
        returns_variance += std::pow(ret - 0.001, 2);  // Assume 0.1% daily return
    }
    returns_variance /= std::min(size_t(50), data.size());
    
    double volatility = std::sqrt(returns_variance);
    
    // Classify regime based on volatility
    if (volatility > 0.02) {
        return 4.0;  // High volatility
    } else if (volatility > 0.01) {
        return 2.0;  // Normal volatility
    } else {
        return 1.0;  // Low volatility
    }
}

Signal OpenCLawEngine::analyze_price_action(const std::vector<MarketData>& data) {
    if (data.size() < 10) {
        return Signal::NEUTRAL;
    }
    
    // Simple price action analysis (simplified)
    double current_price = data.back().price;
    double sma_5 = 0.0, sma_10 = 0.0;
    
    for (size_t i = std::max(size_t(1), data.size() - 9); i < data.size(); ++i) {
        if (i >= data.size() - 4) {
            sma_5 += data[i].price;
        }
        if (i >= data.size() - 9) {
            sma_10 += data[i].price;
        }
    }
    
    sma_5 /= std::min(size_t(5), data.size() - 4);
    sma_10 /= std::min(size_t(10), data.size() - 9);
    
    // Price action logic
    if (current_price > sma_5 && current_price > sma_10) {
        return Signal::BUY;  // Uptrend
    } else if (current_price < sma_5 && current_price < sma_10) {
        return Signal::SELL;  // Downtrend
    } else if (current_price > sma_5 && current_price < sma_10) {
        return Signal::STRONG_BUY;  // Strong uptrend
    } else if (current_price < sma_5 && current_price > sma_10) {
        return Signal::STRONG_SELL;  // Strong downtrend
    }
    
    return Signal::NEUTRAL;
}

double OpenCLawEngine::calculate_signal_strength(const std::vector<MarketData>& data) {
    if (data.size() < 20) {
        return 0.5;  // Default strength
    }
    
    // Calculate signal strength based on multiple factors
    double rsi = calculate_rsi(data);
    double macd = calculate_macd(data);
    auto [upper_bb, lower_bb] = calculate_bollinger_bands(data);
    const auto& current = data.back();
    
    double strength = 0.0;
    
    // RSI contribution (30%)
    if (rsi < 30) {
        strength += 0.3;  // Oversold
    } else if (rsi > 70) {
        strength -= 0.3;  // Overbought
    }
    
    // MACD contribution (25%)
    if (macd > 0) {
        strength += 0.25;  // Bullish
    } else {
        strength -= 0.25;  // Bearish
    }
    
    // Bollinger Bands contribution (25%)
    if (current.price < lower_bb) {
        strength += 0.25;  // Below lower band
    } else if (current.price > upper_bb) {
        strength -= 0.25;  // Above upper band
    }
    
    // Price momentum contribution (20%)
    double momentum = (current.price - data[data.size()-5].price) / data[data.size()-5].price;
    strength += std::max(-0.2, std::min(0.2, momentum * 10));
    
    return std::max(-1.0, std::min(1.0, strength));
}

// Factory implementation

std::unique_ptr<OpenCLawEngine> OpenCLawFactory::create_engine(
    const OpenCLawEngine::Config& config) {
    return std::make_unique<OpenCLawEngine>(config);
}

std::unique_ptr<OpenCLawEngine> OpenCLawFactory::create_default_engine() {
    OpenCLawEngine::Config default_config;
    return std::make_unique<OpenCLawEngine>(default_config);
}

} // namespace OpenCLawCore
