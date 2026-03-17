#pragma once
// ============================================================
// ArchNeuronX v3 - Smart Order Routing System
// AI-Optimized venue selection for sub-millisecond execution
// Cross-exchange liquidity aggregation and execution optimization
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
#include "regime/regime_detector.hpp"

namespace archneuronx {
namespace execution {

/**
 * @brief Venue performance metrics
 */
struct VenueMetrics {
    std::string venue_id;
    double avg_fill_rate;           // Average order fill rate
    double avg_execution_time_ms;   // Average execution time
    double price_improvement_bps;   // Price improvement in basis points
    double liquidity_score;         // Current liquidity availability
    double cost_per_trade;          // All-in cost per trade
    double reliability_score;       // Connection and execution reliability
    uint64_t total_orders;          // Total orders processed
    uint64_t successful_fills;      // Successfully filled orders
    std::chrono::system_clock::time_point last_update;
    
    // Regime-specific performance
    std::unordered_map<int, double> regime_performance; // Performance per regime
};

/**
 * @brief Order request structure
 */
struct OrderRequest {
    std::string order_id;
    std::string symbol;
    std::string side;               // "BUY" or "SELL"
    double quantity;
    double price;                   // Limit price (0 for market orders)
    OrderType type;                // MARKET, LIMIT, IOC, FOK
    std::chrono::system_clock::time_point timestamp;
    std::chrono::system_clock::time_point expiry;
    
    // Routing preferences
    double max_slippage_bps;        // Maximum acceptable slippage
    bool require_liquidity_check;   // Require pre-trade liquidity check
    bool allow_partial_fills;       // Allow partial order fills
    int min_fill_percentage;        // Minimum fill percentage
    
    // AI routing features
    std::vector<float> routing_features; // Feature vector for ML model
    int preferred_venue_priority;   // User-specified venue preference
};

/**
 * @brief Execution venue information
 */
struct ExecutionVenue {
    std::string venue_id;
    std::string venue_name;
    std::vector<std::string> supported_symbols;
    bool supports_market_orders;
    bool supports_limit_orders;
    bool supports_ioc_fok;
    double min_order_size;
    double max_order_size;
    double maker_fee_bps;
    double taker_fee_bps;
    
    // Connectivity
    std::string endpoint_url;
    std::string api_key;
    std::chrono::milliseconds latency_estimate;
    bool is_connected;
    
    // Current state
    VenueMetrics metrics;
    std::unordered_map<std::string, double> current_liquidity; // Symbol -> liquidity
    std::unordered_map<std::string, double> current_spreads;   // Symbol -> spread
};

/**
 * @brief Routing decision
 */
struct RoutingDecision {
    std::string venue_id;
    double confidence_score;
    std::vector<std::string> routing_reasons;
    double expected_fill_rate;
    double expected_execution_time_ms;
    double expected_cost_bps;
    double expected_slippage_bps;
    std::chrono::system_clock::time_point decision_time;
};

/**
 * @brief Liquidity analysis result
 */
struct LiquidityAnalysis {
    std::string symbol;
    double total_liquidity;         // Total available liquidity
    double best_bid_price;
    double best_ask_price;
    double bid_size;
    double ask_size;
    double spread_bps;
    double market_depth;            // Depth at various price levels
    std::vector<std::pair<double, double>> orderbook; // Price -> Size
    std::chrono::system_clock::time_point timestamp;
};

/**
 * @brief Smart Order Router Configuration
 */
struct SmartRouterConfig {
    // Routing algorithm settings
    bool enable_ml_routing = true;
    bool enable_regime_aware_routing = true;
    bool enable_liquidity_aggregation = true;
    bool enable_cost_optimization = true;
    
    // Performance weights
    double speed_weight = 0.3;          // Weight for execution speed
    double cost_weight = 0.3;           // Weight for trading costs
    double fill_rate_weight = 0.2;       // Weight for fill probability
    double liquidity_weight = 0.2;       // Weight for liquidity availability
    
    // Risk constraints
    double max_venue_concentration = 0.4; // Max concentration in single venue
    double max_slippage_bps = 10.0;      // Maximum acceptable slippage
    int min_liquidity_score = 0.7;       // Minimum venue liquidity score
    
    // ML model settings
    std::string routing_model_path = "models/smart_router.pt";
    double routing_confidence_threshold = 0.6;
    int feature_vector_size = 50;
    
    // Update intervals
    int metrics_update_interval_ms = 1000;
    int liquidity_update_interval_ms = 500;
    int venue_health_check_interval_ms = 2000;
    
    // Performance optimization
    bool enable_parallel_routing = true;
    int max_routing_threads = 4;
    bool enable_venue_caching = true;
    int cache_ttl_seconds = 30;
};

/**
 * @brief AI-Optimized Smart Order Router
 * 
 * Implements intelligent venue selection using machine learning,
 * real-time liquidity analysis, and regime-aware routing strategies
 * for optimal trade execution across multiple exchanges.
 */
class SmartOrderRouter {
public:
    explicit SmartOrderRouter(const SmartRouterConfig& config = SmartRouterConfig{});
    ~SmartOrderRouter();

    // Initialization
    bool initialize();
    void shutdown();
    bool is_initialized() const;

    // Venue management
    bool add_venue(const ExecutionVenue& venue);
    bool remove_venue(const std::string& venue_id);
    std::vector<std::string> get_available_venues() const;
    std::vector<ExecutionVenue> get_venue_details() const;

    // Order routing
    RoutingDecision route_order(const OrderRequest& order);
    std::vector<RoutingDecision> route_order_multi_venue(const OrderRequest& order, int max_venues = 3);
    bool execute_order(const OrderRequest& order, const RoutingDecision& routing);

    // Liquidity analysis
    LiquidityAnalysis analyze_liquidity(const std::string& symbol);
    std::unordered_map<std::string, LiquidityAnalysis> analyze_multi_symbol_liquidity(
        const std::vector<std::string>& symbols);

    // Performance monitoring
    void update_venue_metrics(const std::string& venue_id, const VenueMetrics& metrics);
    std::unordered_map<std::string, VenueMetrics> get_all_venue_metrics() const;
    VenueMetrics get_venue_metrics(const std::string& venue_id) const;

    // Regime-aware routing
    void set_current_regime(int regime_id);
    int get_current_regime() const;
    void update_regime_performance(const std::string& venue_id, int regime_id, double performance);

    // ML routing model
    bool load_routing_model(const std::string& model_path);
    bool train_routing_model(const std::vector<OrderRequest>& training_data,
                            const std::vector<RoutingDecision>& training_labels);
    torch::Tensor extract_routing_features(const OrderRequest& order, const LiquidityAnalysis& liquidity);

    // Performance optimization
    void enable_parallel_routing(bool enable);
    void set_routing_threads(int num_threads);
    void clear_venue_cache();

    // Analytics and reporting
    struct RoutingAnalytics {
        uint64_t total_orders_routed;
        uint64_t successful_executions;
        double avg_execution_time_ms;
        double avg_cost_bps;
        double avg_slippage_bps;
        std::unordered_map<std::string, uint64_t> venue_usage_counts;
        std::unordered_map<int, double> regime_performance;
    };
    
    RoutingAnalytics get_routing_analytics() const;
    void reset_analytics();

private:
    SmartRouterConfig config_;
    
    // Venues and metrics
    std::unordered_map<std::string, ExecutionVenue> venues_;
    std::unordered_map<std::string, VenueMetrics> venue_metrics_;
    mutable std::shared_mutex venues_mutex_;
    
    // Current market state
    std::atomic<int> current_regime_;
    std::unordered_map<std::string, LiquidityAnalysis> liquidity_cache_;
    mutable std::shared_mutex liquidity_mutex_;
    
    // ML routing model
    torch::jit::script::Module routing_model_;
    bool model_loaded_;
    
    // Threading and performance
    std::vector<std::thread> routing_threads_;
    std::atomic<bool> running_;
    std::queue<OrderRequest> routing_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Analytics
    mutable std::mutex analytics_mutex_;
    RoutingAnalytics analytics_;
    
    // Internal methods
    void initialize_background_threads();
    void shutdown_background_threads();
    void routing_worker_thread();
    
    // Core routing algorithms
    RoutingDecision route_with_ml(const OrderRequest& order);
    RoutingDecision route_with_rules(const OrderRequest& order);
    RoutingDecision route_regime_aware(const OrderRequest& order);
    
    // Venue selection algorithms
    std::string select_best_venue_speed(const OrderRequest& order);
    std::string select_best_venue_cost(const OrderRequest& order);
    std::string select_best_venue_liquidity(const OrderRequest& order);
    std::string select_best_venue_balanced(const OrderRequest& order);
    
    // Liquidity and market analysis
    void update_liquidity_cache();
    void fetch_venue_liquidity(const std::string& venue_id);
    double calculate_venue_score(const std::string& venue_id, const OrderRequest& order);
    
    // Performance monitoring
    void update_venue_health();
    void cache_venue_performance();
    void cleanup_expired_cache();
    
    // ML model utilities
    std::vector<float> build_feature_vector(const OrderRequest& order, 
                                           const LiquidityAnalysis& liquidity,
                                           const VenueMetrics& metrics);
    double predict_venue_performance(const std::vector<float>& features);
    
    // Utility methods
    bool is_venue_suitable(const std::string& venue_id, const OrderRequest& order);
    double calculate_expected_cost(const std::string& venue_id, const OrderRequest& order);
    double calculate_expected_slippage(const std::string& venue_id, const OrderRequest& order);
    std::vector<std::string> generate_routing_reasons(const std::string& venue_id, 
                                                     const OrderRequest& order);
};

/**
 * @brief Liquidity Aggregator
 * 
 * Aggregates liquidity across multiple venues for optimal execution
 */
class LiquidityAggregator {
public:
    explicit LiquidityAggregator(SmartOrderRouter& router);
    
    struct AggregatedLiquidity {
        std::string symbol;
        double total_bid_size;
        double total_ask_size;
        double weighted_bid_price;
        double weighted_ask_price;
        std::vector<std::pair<std::string, double>> venue_contributions;
        std::chrono::system_clock::time_point timestamp;
    };
    
    AggregatedLiquidity aggregate_liquidity(const std::string& symbol);
    std::unordered_map<std::string, AggregatedLiquidity> aggregate_multi_symbol(
        const std::vector<std::string>& symbols);
    
    double calculate_optimal_execution_size(const std::string& symbol, double target_quantity);
    std::vector<std::pair<std::string, double>> get_venue_allocation(
        const std::string& symbol, double quantity);

private:
    SmartOrderRouter& router_;
    
    void aggregate_venue_liquidity(const std::string& symbol,
                                   std::unordered_map<std::string, LiquidityAnalysis>& venue_liquidity);
    double calculate_weighted_price(const std::vector<std::pair<double, double>>& liquidity_side);
};

/**
 * @brief Execution Cost Analyzer
 * 
 * Analyzes and optimizes execution costs across venues
 */
class ExecutionCostAnalyzer {
public:
    explicit ExecutionCostAnalyzer(SmartOrderRouter& router);
    
    struct CostAnalysis {
        double explicit_cost_bps;        // Fees and commissions
        double implicit_cost_bps;        // Slippage and market impact
        double opportunity_cost_bps;     // Missed opportunities
        double total_cost_bps;
        std::vector<std::string> cost_breakdown;
    };
    
    CostAnalysis analyze_execution_cost(const OrderRequest& order, 
                                       const std::string& venue_id);
    std::unordered_map<std::string, CostAnalysis> compare_venue_costs(
        const OrderRequest& order);
    
    double optimize_execution_strategy(const OrderRequest& order,
                                     std::vector<std::string>& optimal_venues);

private:
    SmartOrderRouter& router_;
    
    double calculate_explicit_cost(const std::string& venue_id, const OrderRequest& order);
    double calculate_implicit_cost(const std::string& venue_id, const OrderRequest& order);
    double calculate_market_impact(const std::string& venue_id, const OrderRequest& order);
};

/**
 * @brief RAII Smart Router Context
 */
class SmartRouterContext {
public:
    explicit SmartRouterContext(const SmartRouterConfig& config = SmartRouterConfig{});
    ~SmartRouterContext();
    
    SmartOrderRouter& get_router();
    bool is_valid() const;

private:
    std::unique_ptr<SmartOrderRouter> router_;
    bool valid_;
};

} // namespace execution
} // namespace archneuronx
