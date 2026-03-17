/**
 * @file openclaw_core.hpp
 * @brief Core OpenCLaw functionality from official repository
 * @author OpenCLaw Team
 * @date 2025-10-02
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <chrono>

namespace OpenCLawCore {

/**
 * @brief Core OpenCLaw signal types
 */
enum class Signal {
    STRONG_BUY = 2,
    BUY = 1,
    NEUTRAL = 0,
    SELL = -1,
    STRONG_SELL = -2
};

/**
 * @brief Market data structure
 */
struct MarketData {
    std::string symbol;
    double price;
    double volume;
    double bid;
    double ask;
    double spread;
    std::chrono::system_clock::time_point timestamp;
    
    MarketData() : price(0.0), volume(0.0), bid(0.0), ask(0.0), spread(0.0) {}
};

/**
 * @brief Trading signal with metadata
 */
struct TradingSignal {
    Signal signal;
    double confidence;
    std::string symbol;
    double price_target;
    double stop_loss;
    double take_profit;
    std::string reasoning;
    std::chrono::system_clock::time_point timestamp;
    std::map<std::string, double> indicators;
    
    TradingSignal() : signal(Signal::NEUTRAL), confidence(0.0), 
                   price_target(0.0), stop_loss(0.0), take_profit(0.0) {}
};

/**
 * @brief Order execution result
 */
struct ExecutionResult {
    bool success;
    std::string venue;
    double fill_price;
    double filled_quantity;
    double slippage_bps;
    double latency_ms;
    std::string execution_algorithm;
    std::chrono::system_clock::time_point timestamp;
    
    ExecutionResult() : success(false), fill_price(0.0), filled_quantity(0.0), 
                     slippage_bps(0.0), latency_ms(0.0) {}
};

/**
 * @brief Core OpenCLaw engine
 */
class OpenCLawEngine {
public:
    /**
     * @brief Configuration
     */
    struct Config {
        std::vector<std::string> supported_venues;
        double max_slippage_bps = 5.0;
        double min_fill_rate = 0.95;
        int execution_timeout_ms = 5000;
        bool enable_smart_routing = true;
        bool enable_microstructure_analysis = true;
        bool enable_adaptive_execution = true;
    };
    
    /**
     * @brief Constructor
     */
    explicit OpenCLawEngine(const Config& config = Config{});
    
    /**
     * @brief Initialize the OpenCLaw engine
     */
    bool initialize();
    
    /**
     * @brief Process market data and generate signals
     * @param market_data Current market data
     * @return Trading signals
     */
    std::vector<TradingSignal> generate_signals(const std::vector<MarketData>& market_data);
    
    /**
     * @brief Execute order with smart routing
     * @param signal Trading signal
     * @param quantity Order quantity
     * @return Execution result
     */
    ExecutionResult execute_order(const TradingSignal& signal, double quantity);
    
    /**
     * @brief Get venue performance metrics
     */
    std::map<std::string, std::map<std::string, double>> get_venue_metrics() const;
    
    /**
     * @brief Update venue performance
     */
    void update_venue_metrics(const std::string& venue, 
                          double fill_rate,
                          double slippage_bps,
                          double latency_ms);
    
    /**
     * @brief Analyze market microstructure
     */
    std::string analyze_microstructure(const MarketData& data);
    
    /**
     * @brief Select best execution venue
     */
    std::string select_best_venue(const std::string& symbol, 
                               const std::string& order_type);

private:
    Config config_;
    bool initialized_;
    
    // Venue performance tracking
    std::map<std::string, std::map<std::string, double>> venue_metrics_;
    
    // Technical analysis methods
    double calculate_rsi(const std::vector<MarketData>& data, int period = 14);
    double calculate_macd(const std::vector<MarketData>& data);
    double calculate_bollinger_bands(const std::vector<MarketData>& data);
    double detect_market_regime(const std::vector<MarketData>& data);
    
    // Signal generation
    Signal analyze_price_action(const std::vector<MarketData>& data);
    double calculate_signal_strength(const std::vector<MarketData>& data);
    
    // Smart routing
    std::vector<std::string> rank_venues_by_liquidity(const std::string& symbol);
    std::string determine_optimal_algorithm(const std::string& venue, 
                                      const MarketData& market_data);
};

/**
 * @brief Factory for creating OpenCLaw instances
 */
class OpenCLawFactory {
public:
    static std::unique_ptr<OpenCLawEngine> create_engine(
        const OpenCLawEngine::Config& config = OpenCLawEngine::Config{});
    
    static std::unique_ptr<OpenCLawEngine> create_default_engine();
};

} // namespace OpenCLawCore
