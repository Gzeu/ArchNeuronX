/**
 * @file openclaw_integration.hpp
 * @brief OpenCLaw integration module for advanced trading automation
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>
#include <map>

namespace ArchNeuronX {
namespace Trading {

/**
 * @brief Trading signal types from OpenCLaw
 */
enum class SignalType {
    STRONG_BUY = 1,
    BUY = 2,
    HOLD = 3,
    SELL = 4,
    STRONG_SELL = 5
};

/**
 * @brief Market regime detection
 */
enum class MarketRegime {
    BULL_MARKET = 1,
    BEAR_MARKET = 2,
    SIDEWAYS_MARKET = 3,
    HIGH_VOLATILITY = 4,
    LOW_VOLATILITY = 5
};

/**
 * @brief Advanced order types
 */
enum class OrderType {
    MARKET = 1,
    LIMIT = 2,
    STOP_LOSS = 3,
    TAKE_PROFIT = 4,
    TRAILING_STOP = 5,
    ICEBERG = 6,
    TWAP = 7,
    VWAP = 8
};

/**
 * @brief Position sizing strategies
 */
enum class PositionSizing {
    FIXED = 1,
    PERCENTAGE = 2,
    KELLY = 3,
    VOLATILITY_ADJUSTED = 4,
    RISK_PARITY = 5
};

/**
 * @brief OpenCLaw signal data structure
 */
struct OpenCLawSignal {
    SignalType signal_type;
    double confidence;
    double price_target;
    double stop_loss;
    double take_profit;
    MarketRegime regime;
    std::string reasoning;
    std::chrono::system_clock::time_point timestamp;
    std::map<std::string, double> technical_indicators;
};

/**
 * @brief Advanced order execution
 */
struct AdvancedOrder {
    std::string symbol;
    OrderType order_type;
    double quantity;
    double price;
    double stop_loss;
    double take_profit;
    int time_in_force_seconds;
    bool iceberg_hidden;
    double trail_amount;
    std::string execution_algorithm;
};

/**
 * @brief Portfolio allocation
 */
struct PortfolioAllocation {
    std::string symbol;
    double allocation_percent;
    double current_position;
    double target_position;
    double unrealized_pnl;
    double daily_pnl;
};

/**
 * @brief Risk metrics
 */
struct RiskMetrics {
    double var_95;              // Value at Risk 95%
    double expected_shortfall;   // Expected Shortfall
    double max_drawdown;        // Maximum Drawdown
    double sharpe_ratio;        // Sharpe Ratio
    double sortino_ratio;       // Sortino Ratio
    double calmar_ratio;        // Calmar Ratio
    double beta;               // Beta relative to market
    double alpha;              // Alpha relative to market
    double information_ratio;   // Information Ratio
};

/**
 * @brief OpenCLaw integration class
 * 
 * Integrates OpenCLaw advanced trading capabilities with ArchNeuronX
 * including market microstructure analysis, smart order routing,
 * and sophisticated risk management.
 */
class OpenCLawIntegration {
public:
    /**
     * @brief Configuration for OpenCLaw integration
     */
    struct Config {
        bool enable_smart_order_routing = true;
        bool enable_market_microstructure = true;
        bool enable_adaptive_execution = true;
        bool enable_ml_signal_filtering = true;
        bool enable_regime_detection = true;
        double max_slippage_bps = 5.0;
        double min_fill_rate = 0.95;
        int execution_timeout_ms = 5000;
        std::vector<std::string> supported_venues = {
            "binance", "coinbase", "kraken", "bybit", "okx"
        };
    };

    /**
     * @brief Constructor
     * @param config Integration configuration
     */
    explicit OpenCLawIntegration(const Config& config);

    /**
     * @brief Initialize OpenCLaw integration
     * @return True if initialization successful
     */
    bool initialize();

    /**
     * @brief Process trading signals from OpenCLaw
     * @param signals Vector of OpenCLaw signals
     * @return Processed signals with ArchNeuronX enhancements
     */
    std::vector<OpenCLawSignal> process_signals(
        const std::vector<OpenCLawSignal>& signals);

    /**
     * @brief Execute advanced order with OpenCLaw routing
     * @param order Advanced order to execute
     * @return Execution result
     */
    bool execute_advanced_order(const AdvancedOrder& order);

    /**
     * @brief Optimize portfolio allocation using OpenCLaw algorithms
     * @param current_allocations Current portfolio state
     * @param signals Current trading signals
     * @return Optimized allocations
     */
    std::vector<PortfolioAllocation> optimize_portfolio(
        const std::vector<PortfolioAllocation>& current_allocations,
        const std::vector<OpenCLawSignal>& signals);

    /**
     * @brief Calculate comprehensive risk metrics
     * @param portfolio Current portfolio state
     * @param market_data Market data for risk calculation
     * @return Risk metrics
     */
    RiskMetrics calculate_risk_metrics(
        const std::vector<PortfolioAllocation>& portfolio,
        const std::map<std::string, double>& market_data);

    /**
     * @brief Detect current market regime
     * @param market_data Recent market data
     * @return Detected market regime
     */
    MarketRegime detect_market_regime(
        const std::map<std::string, double>& market_data);

    /**
     * @brief Apply smart order routing
     * @param order Order to route
     * @return Best venue and execution strategy
     */
    std::pair<std::string, std::string> route_smart_order(
        const AdvancedOrder& order);

    /**
     * @brief Apply market microstructure analysis
     * @param order_book Current order book
     * @return Optimal execution strategy
     */
    std::string analyze_market_microstructure(
        const std::map<std::string, std::vector<double>>& order_book);

    /**
     * @brief Get integration status
     * @return Status information
     */
    std::map<std::string, std::string> get_status() const;

private:
    Config config_;
    bool initialized_;
    
    // Market microstructure analysis
    struct MicrostructureData {
        double bid_ask_spread;
        double order_flow_imbalance;
        double volume_weighted_price;
        double liquidity_score;
        std::chrono::system_clock::time_point last_update;
    };
    
    std::map<std::string, MicrostructureData> microstructure_data_;
    
    // Smart order routing cache
    struct VenueMetrics {
        double avg_fill_rate;
        double avg_slippage_bps;
        double latency_ms;
        double liquidity_score;
        std::string venue_name;
    };
    
    std::map<std::string, VenueMetrics> venue_metrics_;
    
    // Private methods
    void update_venue_metrics(const std::string& venue, 
                           double fill_rate, 
                           double slippage_bps,
                           double latency_ms);
    
    double calculate_optimal_size(const std::string& symbol,
                               double target_size,
                               MarketRegime regime);
    
    std::vector<std::string> select_best_venues(const std::string& symbol,
                                              OrderType order_type);
    
    bool validate_order_risk(const AdvancedOrder& order,
                          const RiskMetrics& risk_metrics);
    
    void update_microstructure_data(const std::string& symbol,
                               const std::map<std::string, double>& market_data);
};

/**
 * @brief OpenCLaw signal processor
 */
class OpenCLawSignalProcessor {
public:
    /**
     * @brief Process raw OpenCLaw signals
     * @param raw_signals Raw signals from OpenCLaw
     * @return Processed signals with confidence scoring
     */
    static std::vector<OpenCLawSignal> process_raw_signals(
        const std::vector<std::map<std::string, double>>& raw_signals);
    
    /**
     * @brief Apply ML filtering to signals
     * @param signals Input signals
     * @return Filtered high-confidence signals
     */
    static std::vector<OpenCLawSignal> apply_ml_filtering(
        const std::vector<OpenCLawSignal>& signals);
    
    /**
     * @brief Ensemble multiple signal sources
     * @param signals Vector of signal arrays
     * @return Ensemble signal with weighted confidence
     */
    static OpenCLawSignal ensemble_signals(
        const std::vector<std::vector<OpenCLawSignal>>& signals);
};

} // namespace Trading
} // namespace ArchNeuronX
