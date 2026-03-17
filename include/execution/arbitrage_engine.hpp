#pragma once
// ============================================================
// ArchNeuronX v3 - Statistical Arbitrage Engine
// Cross-exchange arbitrage opportunities detection
// Pairs trading, triangular arbitrage, latency arbitrage
// Real-time alpha generation for market domination
// ============================================================

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <atomic>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <torch/torch.h>
#include "execution/smart_order_router.hpp"
#include "data/realtime_feed.hpp"

namespace archneuronx {
namespace execution {

/**
 * @brief Arbitrage opportunity types
 */
enum class ArbitrageType {
    CROSS_EXCHANGE,      // Price differences between exchanges
    PAIRS_TRADING,       // Statistical arbitrage between correlated assets
    TRIANGULAR,          // Three-currency arbitrage (forex)
    LATENCY,             // Speed-based arbitrage
    MARKET_MAKING,       // Bid-ask spread capture
    MERGER_ARB,          // Merger arbitrage
    CONVERTIBLE_BOND,     // Convertible bond arbitrage
    STATISTICAL_MOMENTUM  // Momentum-based statistical arbitrage
};

/**
 * @brief Arbitrage opportunity structure
 */
struct ArbitrageOpportunity {
    std::string opportunity_id;
    ArbitrageType type;
    std::vector<std::string> symbols;
    std::vector<std::string> exchanges;
    
    // Pricing information
    std::vector<double> entry_prices;
    std::vector<double> exit_prices;
    std::vector<double> quantities;
    double expected_profit_bps;
    double risk_adjusted_return;
    double max_drawdown_risk;
    
    // Execution parameters
    std::chrono::milliseconds max_execution_time;
    double min_profit_threshold_bps;
    double max_position_size;
    int confidence_score;
    
    // Market conditions
    double volatility_ratio;
    double liquidity_score;
    double correlation_strength;
    
    // Timing
    std::chrono::system_clock::time_point discovery_time;
    std::chrono::system_clock::time_point expiry_time;
    bool is_active;
    
    // Execution plan
    std::vector<std::string> execution_steps;
    std::unordered_map<std::string, double> hedge_ratios;
};

/**
 * @brief Pairs trading statistics
 */
struct PairsTradingStats {
    std::string symbol1;
    std::string symbol2;
    double correlation;
    double cointegration_score;
    double half_life_days;
    double spread_mean;
    double spread_std;
    double z_score;
    double signal_strength;
    std::chrono::system_clock::time_point last_update;
};

/**
 * @brief Triangular arbitrage data
 */
struct TriangularArbitrageData {
    std::string currency1;
    std::string currency2;
    std::string currency3;
    
    // Exchange rates
    double rate12;  // Currency1 -> Currency2
    double rate23;  // Currency2 -> Currency3
    double rate31;  // Currency3 -> Currency1
    
    // Implied rates
    double implied_rate13;  // Rate12 * Rate23
    double arbitrage_spread_bps;
    double execution_cost_bps;
    double net_profit_bps;
    
    std::chrono::system_clock::time_point timestamp;
};

/**
 * @brief Cross-exchange price discrepancy
 */
struct CrossExchangeArbitrage {
    std::string symbol;
    std::string exchange_buy;
    std::string exchange_sell;
    double price_buy;
    double price_sell;
    double spread_bps;
    double liquidity_available;
    double execution_cost_bps;
    double net_profit_bps;
    std::chrono::system_clock::time_point timestamp;
};

/**
 * @brief Arbitrage engine configuration
 */
struct ArbitrageEngineConfig {
    // Detection settings
    bool enable_cross_exchange = true;
    bool enable_pairs_trading = true;
    bool enable_triangular = true;
    bool enable_latency_arb = true;
    bool enable_market_making = true;
    
    // Performance thresholds
    double min_profit_threshold_bps = 5.0;
    double max_execution_time_ms = 100.0;
    double max_position_size_usd = 100000.0;
    double min_liquidity_requirement = 50000.0;
    double max_volatility_ratio = 2.0;
    
    // Risk management
    double max_leverage = 3.0;
    double max_drawdown_limit = 0.02;
    double correlation_threshold = 0.7;
    double cointegration_threshold = 0.8;
    
    // Statistical parameters
    int correlation_lookback_days = 30;
    int volatility_lookback_days = 14;
    double z_score_threshold = 2.0;
    double half_life_max_days = 30;
    
    // Execution settings
    bool enable_parallel_execution = true;
    int max_concurrent_arbitrages = 10;
    double slippage_tolerance_bps = 2.0;
    int execution_timeout_seconds = 30;
    
    // ML settings
    bool enable_ml_prediction = true;
    std::string prediction_model_path = "models/arbitrage_predictor.pt";
    double prediction_confidence_threshold = 0.7;
    
    // Update intervals
    int opportunity_scan_interval_ms = 100;
    int correlation_update_interval_ms = 5000;
    int market_data_update_interval_ms = 50;
};

/**
 * @brief Advanced Statistical Arbitrage Engine
 * 
 * Detects and executes various types of arbitrage opportunities
 * across multiple exchanges and asset classes with real-time
 * market data analysis and ML-enhanced prediction.
 */
class ArbitrageEngine {
public:
    explicit ArbitrageEngine(const ArbitrageEngineConfig& config = ArbitrageEngineConfig{});
    ~ArbitrageEngine();

    // Initialization
    bool initialize(SmartOrderRouter& router, RealtimeFeed& market_data);
    void shutdown();
    bool is_initialized() const;

    // Opportunity detection
    std::vector<ArbitrageOpportunity> scan_opportunities();
    std::vector<CrossExchangeArbitrage> detect_cross_exchange_arbitrage();
    std::vector<PairsTradingStats> detect_pairs_trading_opportunities();
    std::vector<TriangularArbitrageData> detect_triangular_arbitrage();
    
    // Opportunity execution
    bool execute_arbitrage(const ArbitrageOpportunity& opportunity);
    bool execute_cross_exchange_arb(const CrossExchangeArbitrage& arb);
    bool execute_pairs_trade(const PairsTradingStats& pairs);
    bool execute_triangular_arb(const TriangularArbitrageData& triangular);
    
    // Performance monitoring
    struct ArbitragePerformance {
        uint64_t total_opportunities_detected;
        uint64_t successful_executions;
        uint64_t failed_executions;
        double total_profit_bps;
        double avg_execution_time_ms;
        double avg_profit_per_trade_bps;
        double max_profit_bps;
        double max_loss_bps;
        double sharpe_ratio;
        std::unordered_map<ArbitrageType, uint64_t> type_counts;
        std::chrono::system_clock::time_point last_update;
    };
    
    ArbitragePerformance get_performance_metrics() const;
    void reset_performance_metrics();

    // Statistical analysis
    void update_correlation_matrix();
    void update_volatility_estimates();
    void update_cointegration_relationships();
    
    // Risk management
    bool validate_opportunity_risk(const ArbitrageOpportunity& opportunity);
    double calculate_position_risk(const ArbitrageOpportunity& opportunity);
    void apply_risk_limits(ArbitrageOpportunity& opportunity);

    // ML prediction
    bool load_prediction_model(const std::string& model_path);
    double predict_success_probability(const ArbitrageOpportunity& opportunity);
    torch::Tensor extract_arbitrage_features(const ArbitrageOpportunity& opportunity);

    // Market data integration
    void on_market_data_update(const MarketTick& tick);
    void on_orderbook_update(const OrderBook& book);
    void update_market_prices();

private:
    ArbitrageEngineConfig config_;
    SmartOrderRouter* router_;
    RealtimeFeed* market_data_;
    
    // Market data storage
    std::unordered_map<std::string, double> current_prices_;
    std::unordered_map<std::string, OrderBook> current_orderbooks_;
    std::unordered_map<std::string, std::vector<double>> price_history_;
    mutable std::shared_mutex market_data_mutex_;
    
    // Statistical models
    std::unordered_map<std::string, std::unordered_map<std::string, double>> correlation_matrix_;
    std::unordered_map<std::string, double> volatility_estimates_;
    std::unordered_map<std::string, PairsTradingStats> pairs_relationships_;
    mutable std::shared_mutex stats_mutex_;
    
    // Active opportunities
    std::queue<ArbitrageOpportunity> opportunity_queue_;
    std::unordered_map<std::string, ArbitrageOpportunity> active_arbitrages_;
    std::mutex opportunity_mutex_;
    
    // ML model
    torch::jit::script::Module prediction_model_;
    bool model_loaded_;
    
    // Threading
    std::atomic<bool> running_;
    std::vector<std::thread> worker_threads_;
    std::thread scanner_thread_;
    
    // Performance tracking
    mutable std::mutex performance_mutex_;
    ArbitragePerformance performance_;
    
    // Internal methods
    void initialize_background_threads();
    void shutdown_background_threads();
    void opportunity_scanner_thread();
    void worker_thread();
    
    // Detection algorithms
    std::vector<CrossExchangeArbitrage> scan_cross_exchange_prices();
    std::vector<PairsTradingStats> scan_pairs_relationships();
    std::vector<TriangularArbitrageData> scan_triangular_opportunities();
    
    // Statistical calculations
    double calculate_correlation(const std::vector<double>& series1, const std::vector<double>& series2);
    double calculate_cointegration(const std::vector<double>& series1, const std::vector<double>& series2);
    double calculate_half_life(const std::vector<double>& spread_series);
    double calculate_z_score(double spread, double mean, double std_dev);
    
    // Risk calculations
    double calculate_var(const std::vector<double>& returns, double confidence = 0.95);
    double calculate_sharpe_ratio(const std::vector<double>& returns, double risk_free_rate = 0.02);
    double calculate_max_drawdown(const std::vector<double>& equity_curve);
    
    // Execution helpers
    bool execute_multi_leg_strategy(const ArbitrageOpportunity& opportunity);
    bool hedge_residual_risk(const ArbitrageOpportunity& opportunity);
    void monitor_arbitrage_execution(const std::string& opportunity_id);
    
    // ML utilities
    std::vector<float> build_feature_vector(const ArbitrageOpportunity& opportunity);
    void update_ml_model(const ArbitrageOpportunity& opportunity, bool success);
};

/**
 * @brief Pairs Trading Strategy
 */
class PairsTradingStrategy {
public:
    explicit PairsTradingStrategy(ArbitrageEngine& engine);
    
    struct PairsSignal {
        std::string symbol1;
        std::string symbol2;
        double current_spread;
        double z_score;
        std::string signal; // "LONG", "SHORT", "NEUTRAL"
        double confidence;
        std::chrono::system_clock::time_point timestamp;
    };
    
    PairsSignal generate_signal(const PairsTradingStats& stats);
    bool execute_signal(const PairsSignal& signal);
    void update_pairs_stats(const std::string& symbol1, const std::string& symbol2);

private:
    ArbitrageEngine& engine_;
    std::unordered_map<std::string, PairsTradingStats> pairs_stats_;
    
    double calculate_optimal_hedge_ratio(const std::vector<double>& prices1, const std::vector<double>& prices2);
    std::vector<double> calculate_spread_series(const std::vector<double>& prices1, 
                                               const std::vector<double>& prices2, 
                                               double hedge_ratio);
};

/**
 * @brief Triangular Arbitrage Strategy
 */
class TriangularArbitrageStrategy {
public:
    explicit TriangularArbitrageStrategy(ArbitrageEngine& engine);
    
    std::vector<TriangularArbitrageData> find_opportunities();
    bool execute_triangular_arb(const TriangularArbitrageData& data);
    
    struct CurrencyTriangle {
        std::string currency1;
        std::string currency2;
        std::string currency3;
        std::vector<std::string> trading_pairs;
    };
    
    std::vector<CurrencyTriangle> get_currency_triangles();

private:
    ArbitrageEngine& engine_;
    std::vector<CurrencyTriangle> currency_triangles_;
    
    void initialize_currency_triangles();
    double calculate_implied_rate(double rate12, double rate23);
    bool validate_triangle_execution(const TriangularArbitrageData& data);
};

/**
 * @brief Latency Arbitrage Strategy
 */
class LatencyArbitrageStrategy {
public:
    explicit LatencyArbitrageStrategy(ArbitrageEngine& engine);
    
    struct LatencyOpportunity {
        std::string symbol;
        std::string fast_exchange;
        std::string slow_exchange;
        double price_difference_bps;
        std::chrono::microseconds time_advantage;
        double profit_potential_bps;
        std::chrono::system_clock::time_point timestamp;
    };
    
    std::vector<LatencyOpportunity> detect_latency_opportunities();
    bool execute_latency_arb(const LatencyOpportunity& opportunity);
    
    void update_latency_measurements(const std::string& exchange, std::chrono::microseconds latency);
    std::chrono::microseconds get_exchange_latency(const std::string& exchange);

private:
    ArbitrageEngine& engine_;
    std::unordered_map<std::string, std::chrono::microseconds> exchange_latencies_;
    std::mutex latency_mutex_;
    
    void measure_exchange_latencies();
    double calculate_speed_advantage_value(std::chrono::microseconds time_advantage);
};

/**
 * @brief Risk Manager for Arbitrage
 */
class ArbitrageRiskManager {
public:
    explicit ArbitrageRiskManager(const ArbitrageEngineConfig& config);
    
    struct RiskAssessment {
        bool is_acceptable;
        double risk_score;
        std::vector<std::string> risk_factors;
        double recommended_position_size;
        double suggested_stop_loss_bps;
    };
    
    RiskAssessment assess_opportunity_risk(const ArbitrageOpportunity& opportunity);
    bool validate_execution_risk(const ArbitrageOpportunity& opportunity);
    void update_risk_parameters(const ArbitrageOpportunity& opportunity, bool success);

private:
    ArbitrageEngineConfig config_;
    std::unordered_map<ArbitrageType, double> type_risk_factors_;
    std::unordered_map<std::string, double> exchange_risk_factors_;
    
    double calculate_liquidity_risk(const ArbitrageOpportunity& opportunity);
    double calculate_execution_risk(const ArbitrageOpportunity& opportunity);
    double calculate_market_risk(const ArbitrageOpportunity& opportunity);
    double calculate_operational_risk(const ArbitrageOpportunity& opportunity);
};

/**
 * @brief Performance Analyzer for Arbitrage
 */
class ArbitragePerformanceAnalyzer {
public:
    explicit ArbitragePerformanceAnalyzer(ArbitrageEngine& engine);
    
    struct PerformanceReport {
        double total_return_bps;
        double risk_adjusted_return;
        double win_rate;
        double profit_factor;
        double avg_trade_duration_minutes;
        double sharpe_ratio;
        double max_drawdown_bps;
        std::unordered_map<ArbitrageType, double> type_performance;
        std::unordered_map<std::string, double> exchange_performance;
    };
    
    PerformanceReport generate_report(const std::chrono::hours period = std::chrono::hours(24));
    void record_trade_result(const ArbitrageOpportunity& opportunity, bool success, double profit_bps);
    
    std::vector<std::string> get_insights();
    std::vector<std::string> get_recommendations();

private:
    ArbitrageEngine& engine_;
    std::vector<std::tuple<ArbitrageOpportunity, bool, double>> trade_history_;
    std::mutex history_mutex_;
    
    double calculate_type_performance(ArbitrageType type);
    double calculate_exchange_performance(const std::string& exchange);
    std::vector<std::string> analyze_performance_patterns();
};

/**
 * @brief RAII Arbitrage Engine Context
 */
class ArbitrageEngineContext {
public:
    explicit ArbitrageEngineContext(const ArbitrageEngineConfig& config = ArbitrageEngineConfig{});
    ~ArbitrageEngineContext();
    
    ArbitrageEngine& get_engine();
    bool is_valid() const;

private:
    std::unique_ptr<ArbitrageEngine> engine_;
    bool valid_;
};

} // namespace execution
} // namespace archneuronx
