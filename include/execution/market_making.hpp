#pragma once
// ============================================================
// ArchNeuronX v3 - Advanced Market Making Algorithm
// Regime-aware liquidity provision and spread capture
// Inventory management and adverse selection protection
// Institutional-grade market making for profit generation
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
#include "regime/regime_detector.hpp"

namespace archneuronx {
namespace execution {

/**
 * @brief Market making strategy types
 */
enum class MarketMakingStrategy {
    STATIC_SPREAD,           // Fixed bid-ask spread
    DYNAMIC_SPREAD,          // Variable spread based on volatility
    INVENTORY_AWARE,         // Spread based on inventory risk
    ADVERSE_SELECTION_PROTECTED, // Protection against toxic flow
    REGIME_AWARE,            // Adapt spreads based on market regime
    PREDICTIVE,              // ML-based spread prediction
    HYBRID                   // Combination of multiple strategies
};

/**
 * @brief Quote structure
 */
struct Quote {
    std::string quote_id;
    std::string symbol;
    std::string exchange;
    
    // Quote details
    double bid_price;
    double ask_price;
    double bid_size;
    double ask_size;
    double spread_bps;
    
    // Quote metadata
    std::chrono::system_clock::time_point timestamp;
    std::chrono::system_clock::time_point expiry;
    bool is_active;
    
    // Risk metrics
    double inventory_risk_contribution;
    double adverse_selection_score;
    double expected_profit_bps;
    double probability_of_fill;
};

/**
 * @brief Inventory position
 */
struct InventoryPosition {
    std::string symbol;
    double quantity;              // Positive = long, Negative = short
    double average_price;
    double unrealized_pnl;
    double realized_pnl;
    double inventory_cost;
    std::chrono::system_clock::time_point last_update;
    
    // Risk metrics
    double inventory_risk;
    double concentration_ratio;
    double leverage_utilization;
};

/**
 * @brief Market making parameters
 */
struct MarketMakingParams {
    // Spread parameters
    double base_spread_bps;
    double max_spread_bps;
    double min_spread_bps;
    double spread_volatility_multiplier;
    
    // Size parameters
    double base_quote_size;
    double max_quote_size;
    double min_quote_size;
    double size_inventory_multiplier;
    
    // Risk parameters
    double max_inventory_usd;
    double max_position_ratio;
    double inventory_target_ratio;
    double stop_loss_threshold_bps;
    
    // Timing parameters
    int quote_duration_seconds;
    int refresh_interval_ms;
    int inventory_rebalance_threshold;
    
    // Protection parameters
    double adverse_selection_threshold;
    double toxic_flow_detection_threshold;
    double latency_protection_ms;
    double inventory_skew_tolerance;
    
    // Regime-specific parameters
    std::unordered_map<int, double> regime_spread_multipliers;
    std::unordered_map<int, double> regime_size_multipliers;
    std::unordered_map<int, double> regime_inventory_limits;
};

/**
 * @brief Market making performance metrics
 */
struct MarketMakingPerformance {
    // Trading metrics
    uint64_t total_quotes_posted;
    uint64_t quotes_filled;
    uint64_t quotes_cancelled;
    double fill_rate;
    double avg_fill_size;
    
    // Profit metrics
    double total_profit_bps;
    double spread_capture_bps;
    double inventory_pnl_bps;
    double adverse_selection_cost_bps;
    
    // Risk metrics
    double max_inventory_utilization;
    double avg_inventory_utilization;
    double inventory_turnover_rate;
    double sharpe_ratio;
    
    // Timing metrics
    double avg_quote_duration_ms;
    double avg_time_to_fill_ms;
    double quote_cancellation_rate;
    
    std::chrono::system_clock::time_point last_update;
};

/**
 * @brief Market Making Configuration
 */
struct MarketMakerConfig {
    // Strategy configuration
    MarketMakingStrategy strategy = MarketMakingStrategy::HYBRID;
    bool enable_regime_awareness = true;
    bool enable_adverse_selection_protection = true;
    bool enable_inventory_management = true;
    bool enable_ml_prediction = false;
    
    // Risk management
    bool enable_position_limits = true;
    bool enable_stop_loss = true;
    bool enable_inventory_rebalancing = true;
    double max_leverage = 2.0;
    double max_drawdown_limit = 0.05;
    
    // Execution settings
    bool enable_parallel_quoting = true;
    int max_concurrent_quotes = 50;
    bool enable_quote_caching = true;
    int cache_ttl_seconds = 10;
    
    // ML settings
    std::string prediction_model_path = "models/market_maker.pt";
    double prediction_confidence_threshold = 0.6;
    bool enable_online_learning = true;
    
    // Monitoring settings
    bool enable_performance_tracking = true;
    int performance_update_interval_ms = 1000;
    bool enable_real_time_alerts = true;
    
    // Update intervals
    int quote_update_interval_ms = 100;
    int inventory_update_interval_ms = 500;
    int performance_update_interval_sec = 5;
};

/**
 * @brief Advanced Market Making Algorithm
 * 
 * Implements sophisticated market making strategies with regime awareness,
 * inventory management, and protection against adverse selection.
 * Optimized for high-frequency trading with sub-millisecond execution.
 */
class MarketMaker {
public:
    explicit MarketMaker(const MarketMakerConfig& config = MarketMakerConfig{});
    ~MarketMaker();

    // Initialization
    bool initialize(SmartOrderRouter& router, regime::RegimeDetector& regime_detector);
    void shutdown();
    bool is_initialized() const;

    // Symbol management
    bool add_symbol(const std::string& symbol, const MarketMakingParams& params);
    bool remove_symbol(const std::string& symbol);
    std::vector<std::string> get_active_symbols() const;
    MarketMakingParams get_symbol_params(const std::string& symbol) const;

    // Quote management
    Quote generate_quote(const std::string& symbol);
    std::vector<Quote> generate_quotes(const std::vector<std::string>& symbols);
    bool post_quote(const Quote& quote);
    bool cancel_quote(const std::string& quote_id);
    void cancel_all_quotes(const std::string& symbol);

    // Inventory management
    void update_inventory(const std::string& symbol, double quantity_change, double price);
    InventoryPosition get_inventory_position(const std::string& symbol) const;
    std::unordered_map<std::string, InventoryPosition> get_all_inventory() const;
    double get_total_inventory_value() const;

    // Quote execution
    bool on_quote_fill(const std::string& quote_id, double fill_quantity, double fill_price);
    bool on_quote_cancel(const std::string& quote_id);
    void on_market_data_update(const std::string& symbol, double bid_price, double ask_price);

    // Strategy management
    void set_strategy(MarketMakingStrategy strategy);
    MarketMakingStrategy get_strategy() const;
    void update_strategy_parameters(const std::string& symbol, const MarketMakingParams& params);

    // Performance monitoring
    MarketMakingPerformance get_performance_metrics() const;
    void reset_performance_metrics();
    std::vector<std::string> get_performance_insights() const;

    // Risk management
    bool check_position_limits(const std::string& symbol, double quantity) const;
    bool check_inventory_risk() const;
    void apply_risk_controls(Quote& quote);
    void emergency_stop_all_quoting();

    // ML prediction
    bool load_prediction_model(const std::string& model_path);
    double predict_fill_probability(const Quote& quote);
    double predict_adverse_selection_risk(const Quote& quote);
    torch::Tensor extract_quote_features(const Quote& quote);

    // Regime awareness
    void on_regime_change(int new_regime);
    int get_current_regime() const;
    void update_regime_parameters(int regime_id);

private:
    MarketMakerConfig config_;
    SmartOrderRouter* router_;
    regime::RegimeDetector* regime_detector_;
    
    // Symbol configurations
    std::unordered_map<std::string, MarketMakingParams> symbol_params_;
    std::unordered_map<std::string, bool> active_symbols_;
    mutable std::shared_mutex symbols_mutex_;
    
    // Quote management
    std::unordered_map<std::string, Quote> active_quotes_;
    std::queue<Quote> quote_queue_;
    std::mutex quotes_mutex_;
    
    // Inventory tracking
    std::unordered_map<std::string, InventoryPosition> inventory_positions_;
    mutable std::shared_mutex inventory_mutex_;
    
    // Market data
    std::unordered_map<std::string, std::pair<double, double>> current_quotes_; // symbol -> (bid, ask)
    std::unordered_map<std::string, std::chrono::system_clock::time_point> last_market_update_;
    mutable std::shared_mutex market_data_mutex_;
    
    // ML model
    torch::jit::script::Module prediction_model_;
    bool model_loaded_;
    
    // Current state
    std::atomic<int> current_regime_;
    std::atomic<MarketMakingStrategy> current_strategy_;
    
    // Threading
    std::atomic<bool> running_;
    std::vector<std::thread> worker_threads_;
    std::thread quote_generator_thread_;
    std::thread inventory_manager_thread_;
    
    // Performance tracking
    mutable std::mutex performance_mutex_;
    MarketMakingPerformance performance_;
    
    // Internal methods
    void initialize_background_threads();
    void shutdown_background_threads();
    void quote_generator_thread_func();
    void inventory_manager_thread_func();
    void worker_thread_func();
    
    // Quote generation algorithms
    Quote generate_static_spread_quote(const std::string& symbol);
    Quote generate_dynamic_spread_quote(const std::string& symbol);
    Quote generate_inventory_aware_quote(const std::string& symbol);
    Quote generate_adverse_selection_protected_quote(const std::string& symbol);
    Quote generate_regime_aware_quote(const std::string& symbol);
    Quote generate_hybrid_quote(const std::string& symbol);
    
    // Spread calculation
    double calculate_optimal_spread(const std::string& symbol);
    double calculate_inventory_skew(const std::string& symbol);
    double calculate_adverse_selection_adjustment(const std::string& symbol);
    double calculate_regime_adjustment(const std::string& symbol);
    
    // Quote sizing
    double calculate_optimal_size(const std::string& symbol, double base_size);
    double calculate_inventory_adjusted_size(const std::string& symbol, double base_size);
    double calculate_volatility_adjusted_size(const std::string& symbol, double base_size);
    
    // Risk management
    bool check_position_limit(const std::string& symbol, double quantity) const;
    bool check_leverage_limit() const;
    bool check_drawdown_limit() const;
    void apply_position_limits(Quote& quote);
    void apply_stop_loss(Quote& quote);
    
    // Inventory management
    void rebalance_inventory();
    double calculate_inventory_risk(const std::string& symbol) const;
    double calculate_total_inventory_risk() const;
    void update_inventory_metrics();
    
    // Adverse selection detection
    bool detect_toxic_order_flow(const std::string& symbol);
    double calculate_toxicity_score(const std::string& symbol);
    void protect_against_adverse_selection(Quote& quote);
    
    // Performance analysis
    void update_performance_metrics();
    double calculate_sharpe_ratio() const;
    double calculate_inventory_turnover() const;
    std::vector<std::string> generate_performance_insights() const;
    
    // ML utilities
    std::vector<float> build_feature_vector(const Quote& quote);
    void update_ml_model(const Quote& quote, bool filled, double profit_bps);
};

/**
 * @brief Adverse Selection Detector
 */
class AdverseSelectionDetector {
public:
    explicit AdverseSelectionDetector(const MarketMakerConfig& config);
    
    struct ToxicityMetrics {
        double toxicity_score;
        double order_flow_toxicity;
        double price_impact_score;
        double timing_pattern_score;
        bool is_toxic;
        std::chrono::system_clock::time_point timestamp;
    };
    
    ToxicityMetrics analyze_order_flow(const std::string& symbol, 
                                     const std::vector<MarketTick>& recent_trades);
    bool is_toxic_flow_detected(const std::string& symbol);
    void update_detection_model(const std::string& symbol, bool was_toxic, double outcome);

private:
    MarketMakerConfig config_;
    std::unordered_map<std::string, std::vector<MarketTick>> recent_trades_;
    std::unordered_map<std::string, double> toxicity_scores_;
    std::mutex detection_mutex_;
    
    double calculate_order_flow_toxicity(const std::vector<MarketTick>& trades);
    double calculate_price_impact(const std::vector<MarketTick>& trades);
    double calculate_timing_pattern(const std::vector<MarketTick>& trades);
};

/**
 * @brief Inventory Manager
 */
class InventoryManager {
public:
    explicit InventoryManager(const MarketMakerConfig& config);
    
    struct InventoryRisk {
        double total_risk;
        double concentration_risk;
        double leverage_risk;
        double market_risk;
        bool exceeds_limits;
        std::vector<std::string> risk_factors;
    };
    
    InventoryRisk assess_inventory_risk(const std::unordered_map<std::string, InventoryPosition>& positions);
    std::vector<std::string> get_rebalancing_recommendations();
    double calculate_optimal_hedge_ratio(const std::string& symbol1, const std::string& symbol2);

private:
    MarketMakerConfig config_;
    std::unordered_map<std::string, double> hedge_ratios_;
    std::mutex inventory_mutex_;
    
    double calculate_concentration_risk(const std::unordered_map<std::string, InventoryPosition>& positions);
    double calculate_leverage_risk(const std::unordered_map<std::string, InventoryPosition>& positions);
    void update_hedge_ratios();
};

/**
 * @brief Quote Optimizer
 */
class QuoteOptimizer {
public:
    explicit QuoteOptimizer(const MarketMakerConfig& config);
    
    struct OptimizationResult {
        Quote optimized_quote;
        double expected_profit_bps;
        double fill_probability;
        double risk_adjusted_return;
        std::vector<std::string> optimization_steps;
    };
    
    OptimizationResult optimize_quote(const Quote& base_quote, 
                                   const InventoryPosition& inventory,
                                   const MarketMakingParams& params);
    std::vector<Quote> optimize_quote_ladder(const std::string& symbol, int num_levels);

private:
    MarketMakerConfig config_;
    
    double optimize_spread_for_inventory(const Quote& quote, const InventoryPosition& inventory);
    double optimize_size_for_risk(const Quote& quote, double inventory_risk);
    std::vector<std::string> generate_optimization_steps(const Quote& original, const Quote& optimized);
};

/**
 * @brief Performance Analyzer for Market Making
 */
class MarketMakingPerformanceAnalyzer {
public:
    explicit MarketMakingPerformanceAnalyzer(MarketMaker& market_maker);
    
    struct PerformanceReport {
        double total_return_bps;
        double spread_capture_efficiency;
        double inventory_efficiency;
        double adverse_selection_cost;
        double risk_adjusted_return;
        std::unordered_map<std::string, double> symbol_performance;
        std::vector<std::string> key_insights;
        std::vector<std::string> recommendations;
    };
    
    PerformanceReport generate_report(const std::chrono::hours period = std::chrono::hours(24));
    void record_quote_execution(const Quote& quote, bool filled, double profit_bps);
    std::vector<std::string> get_real_time_alerts();

private:
    MarketMaker& market_maker_;
    std::vector<std::tuple<Quote, bool, double>> execution_history_;
    std::mutex history_mutex_;
    
    double calculate_spread_capture_efficiency();
    double calculate_inventory_efficiency();
    std::vector<std::string> analyze_performance_patterns();
};

/**
 * @brief RAII Market Maker Context
 */
class MarketMakerContext {
public:
    explicit MarketMakerContext(const MarketMakerConfig& config = MarketMakerConfig{});
    ~MarketMakerContext();
    
    MarketMaker& get_market_maker();
    bool is_valid() const;

private:
    std::unique_ptr<MarketMaker> market_maker_;
    bool valid_;
};

} // namespace execution
} // namespace archneuronx
