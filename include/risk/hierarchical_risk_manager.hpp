#pragma once
// ============================================================
// ArchNeuronX v3 - Hierarchical Risk Management System
// Institutional-grade risk controls for portfolio management
// Multi-level risk assessment and automated mitigation
// Real-time VaR, stress testing, and circuit breakers
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
#include "execution/arbitrage_engine.hpp"
#include "execution/market_making.hpp"
#include "regime/regime_detector.hpp"

namespace archneuronx {
namespace risk {

/**
 * @brief Risk level enumeration
 */
enum class RiskLevel {
    NO_RISK,        // 0-10% VaR
    LOW_RISK,        // 10-20% VaR
    MEDIUM_RISK,     // 20-30% VaR
    HIGH_RISK,        // 30-40% VaR
    EXTREME_RISK,    // 40-50% VaR
    CRITICAL_RISK    // >50% VaR
};

/**
 * @brief Risk category enumeration
 */
enum class RiskCategory {
    MARKET_RISK,      // Market price movements
    CREDIT_RISK,      // Counterparty default
    LIQUIDITY_RISK,   // Market liquidity
    OPERATIONAL_RISK, // System failures
    MODEL_RISK,       // Model errors
    CONCENTRATION_RISK, // Position concentration
    LEVERAGE_RISK,   // Excessive leverage
    CORRELATION_RISK, // Asset correlation
    REGIME_RISK       // Market regime changes
};

/**
 * @brief Risk metric structure
 */
struct RiskMetric {
    std::string metric_id;
    std::string name;
    RiskCategory category;
    double current_value;
    double threshold_warning;
    double threshold_critical;
    RiskLevel current_level;
    std::chrono::system_clock::time_point last_updated;
    std::string description;
    std::vector<std::string> mitigation_actions;
};

/**
 * @brief Position risk assessment
 */
struct PositionRisk {
    std::string symbol;
    std::string strategy_id;
    double quantity;
    double average_price;
    double current_price;
    double unrealized_pnl;
    double realized_pnl;
    
    // Risk metrics
    double position_var;
    double position_beta;
    double position_volatility;
    double liquidity_score;
    double concentration_ratio;
    
    // Risk contributions
    double market_risk_contribution;
    double credit_risk_contribution;
    double liquidity_risk_contribution;
    double model_risk_contribution;
    
    std::chrono::system_clock::time_point last_updated;
};

/**
 * @brief Portfolio risk assessment
 */
struct PortfolioRisk {
    double total_value;
    double total_pnl;
    double total_var;
    double total_cvar;        // Conditional VaR
    double total_es;          // Expected Shortfall
    double sharpe_ratio;
    double sortino_ratio;
    double max_drawdown;
    double current_drawdown;
    
    // Risk breakdown
    std::unordered_map<RiskCategory, double> risk_contributions;
    std::unordered_map<std::string, double> asset_contributions;
    std::unordered_map<std::string, double> strategy_contributions;
    
    // Risk metrics
    double leverage_ratio;
    double concentration_ratio;
    double correlation_risk;
    double beta_exposure;
    
    std::chrono::system_clock::time_point last_updated;
};

/**
 * @brief Stress test scenario
 */
struct StressTestScenario {
    std::string scenario_id;
    std::string name;
    std::string description;
    
    // Scenario parameters
    std::unordered_map<std::string, double> price_shocks;    // Symbol -> shock percentage
    std::unordered_map<std::string, double> volatility_shocks;
    std::unordered_map<std::string, double> correlation_shocks;
    
    // Scenario results
    double portfolio_pnl;
    double portfolio_var;
    double max_drawdown;
    double worst_loss;
    double recovery_time_days;
    
    std::chrono::system_clock::time_point last_run;
    bool is_active;
};

/**
 * @brief Circuit breaker configuration
 */
struct CircuitBreakerConfig {
    std::string breaker_id;
    std::string name;
    RiskCategory category;
    
    // Trigger conditions
    double trigger_threshold;
    double reset_threshold;
    std::chrono::milliseconds trigger_duration;
    
    // Actions
    bool stop_new_orders;
    bool cancel_existing_orders;
    bool reduce_position_sizes;
    bool liquidate_positions;
    double position_reduction_factor;
    
    // Status
    bool is_active;
    bool is_triggered;
    std::chrono::system_clock::time_point last_triggered;
    std::chrono::system_clock::time_point last_reset;
    
    int trigger_count;
    int max_triggers_per_day;
};

/**
 * @brief Hierarchical Risk Management Configuration
 */
struct HierarchicalRiskConfig {
    // Portfolio-level settings
    double max_portfolio_var = 0.02;              // 2% daily VaR
    double max_portfolio_leverage = 3.0;
    double max_concentration_ratio = 0.3;           // 30% in single asset
    double max_correlation_threshold = 0.8;
    double min_sharpe_ratio = 0.5;
    
    // Strategy-level settings
    double max_strategy_var = 0.01;               // 1% daily VaR per strategy
    double max_strategy_leverage = 2.0;
    double max_strategy_drawdown = 0.15;             // 15% max drawdown
    
    // Position-level settings
    double max_position_size = 1000000.0;           // $1M max position
    double max_position_var = 0.005;               // 0.5% daily VaR per position
    double min_liquidity_score = 0.3;
    
    // Stress testing
    int stress_test_interval_hours = 6;
    int max_stress_scenarios = 50;
    double stress_test_confidence = 0.99;
    
    // Circuit breakers
    bool enable_circuit_breakers = true;
    int max_circuit_breakers = 10;
    std::chrono::milliseconds breaker_check_interval = 100;
    
    // Monitoring
    bool enable_real_time_monitoring = true;
    int monitoring_update_interval_ms = 1000;
    bool enable_risk_alerts = true;
    double alert_threshold_multiplier = 0.8;
    
    // ML risk prediction
    bool enable_ml_risk_prediction = true;
    std::string risk_model_path = "models/risk_predictor.pt";
    double prediction_confidence_threshold = 0.7;
    int prediction_update_interval_hours = 24;
};

/**
 * @brief Hierarchical Risk Manager
 * 
 * Implements multi-level risk management for institutional trading:
 * - Portfolio-level risk controls
 * - Strategy-level risk monitoring
 * - Position-level risk assessment
 * - Real-time stress testing
 * - Circuit breaker protection
 * - ML-enhanced risk prediction
 */
class HierarchicalRiskManager {
public:
    explicit HierarchicalRiskManager(const HierarchicalRiskConfig& config = HierarchicalRiskConfig{});
    ~HierarchicalRiskManager();

    // Initialization
    bool initialize();
    void shutdown();
    bool is_initialized() const;

    // Position management
    void add_position(const std::string& symbol, const std::string& strategy_id, 
                     double quantity, double price);
    void remove_position(const std::string& symbol);
    void update_position_price(const std::string& symbol, double price);
    void update_position_pnl(const std::string& symbol, double realized_pnl);
    
    // Risk assessment
    PositionRisk assess_position_risk(const std::string& symbol);
    PortfolioRisk assess_portfolio_risk();
    std::unordered_map<std::string, PositionRisk> assess_all_positions();
    
    // VaR calculation
    double calculate_var(const std::vector<double>& returns, double confidence = 0.95) const;
    double calculate_cvar(const std::vector<double>& returns, double confidence = 0.95) const;
    double calculate_expected_shortfall(const std::vector<double>& returns) const;
    
    // Stress testing
    void run_stress_tests();
    void add_stress_scenario(const StressTestScenario& scenario);
    std::vector<StressTestScenario> get_stress_results() const;
    StressTestScenario run_custom_stress_test(const std::unordered_map<std::string, double>& shocks);
    
    // Circuit breakers
    void add_circuit_breaker(const CircuitBreakerConfig& config);
    void remove_circuit_breaker(const std::string& breaker_id);
    std::vector<CircuitBreakerConfig> get_active_breakers() const;
    bool check_circuit_breakers();
    
    // Risk monitoring
    struct RiskAlert {
        std::string alert_id;
        RiskLevel level;
        RiskCategory category;
        std::string message;
        std::string recommendation;
        std::chrono::system_clock::time_point timestamp;
        bool is_acknowledged;
    };
    
    std::vector<RiskAlert> get_risk_alerts() const;
    void acknowledge_alert(const std::string& alert_id);
    void clear_alerts();
    
    // ML risk prediction
    bool load_risk_model(const std::string& model_path);
    double predict_portfolio_risk(const PortfolioRisk& portfolio);
    double predict_position_risk(const PositionRisk& position);
    void update_risk_model(const PortfolioRisk& portfolio, bool was_risk_event);

    // Regulatory compliance
    struct ComplianceReport {
        bool var_compliance;
        bool leverage_compliance;
        bool concentration_compliance;
        bool reporting_compliance;
        std::vector<std::string> compliance_issues;
        std::chrono::system_clock::time_point report_time;
    };
    
    ComplianceReport generate_compliance_report();
    bool check_regulatory_compliance();

    // Performance monitoring
    struct RiskPerformanceMetrics {
        double risk_adjusted_return;
        double risk_efficiency;
        double risk_coverage;
        double false_positive_rate;
        double detection_latency_ms;
        std::chrono::system_clock::time_point last_update;
    };
    
    RiskPerformanceMetrics get_performance_metrics() const;
    void reset_performance_metrics();

    // Configuration management
    void update_config(const HierarchicalRiskConfig& config);
    HierarchicalRiskConfig get_config() const;

private:
    HierarchicalRiskConfig config_;
    
    // Position tracking
    std::unordered_map<std::string, PositionRisk> positions_;
    mutable std::shared_mutex positions_mutex_;
    
    // Portfolio state
    PortfolioRisk portfolio_risk_;
    mutable std::shared_mutex portfolio_mutex_;
    
    // Stress testing
    std::vector<StressTestScenario> stress_scenarios_;
    std::vector<StressTestScenario> stress_results_;
    mutable std::mutex stress_mutex_;
    
    // Circuit breakers
    std::unordered_map<std::string, CircuitBreakerConfig> circuit_breakers_;
    mutable std::mutex breakers_mutex_;
    
    // Risk alerts
    std::queue<RiskAlert> risk_alerts_;
    mutable std::mutex alerts_mutex_;
    
    // ML model
    torch::jit::script::Module risk_model_;
    bool model_loaded_;
    
    // Threading
    std::atomic<bool> running_;
    std::thread monitoring_thread_;
    std::thread stress_test_thread_;
    std::thread circuit_breaker_thread_;
    
    // Performance tracking
    mutable std::mutex performance_mutex_;
    RiskPerformanceMetrics performance_metrics_;
    
    // Internal methods
    void initialize_background_threads();
    void shutdown_background_threads();
    void monitoring_thread_func();
    void stress_test_thread_func();
    void circuit_breaker_thread_func();
    
    // Risk calculation methods
    double calculate_position_var(const PositionRisk& position);
    double calculate_portfolio_var(const std::vector<PositionRisk>& positions);
    double calculate_correlation_risk(const std::vector<PositionRisk>& positions);
    double calculate_concentration_risk(const std::vector<PositionRisk>& positions);
    double calculate_leverage_risk(const std::vector<PositionRisk>& positions);
    
    // VaR calculation methods
    double calculate_historical_var(const std::vector<double>& returns, double confidence) const;
    double calculate_monte_carlo_var(const std::vector<PositionRisk>& positions, double confidence) const;
    double calculate_parametric_var(const std::vector<PositionRisk>& positions, double confidence) const;
    
    // Stress testing methods
    void run_market_crash_scenario();
    void run_volatility_spike_scenario();
    void run_liquidity_crisis_scenario();
    void run_correlation_breakdown_scenario();
    void run_custom_scenario(const StressTestScenario& scenario);
    
    // Circuit breaker methods
    bool check_portfolio_breaker(const CircuitBreakerConfig& breaker);
    bool check_strategy_breaker(const CircuitBreakerConfig& breaker);
    bool check_position_breaker(const CircuitBreakerConfig& breaker);
    void trigger_breaker_actions(const CircuitBreakerConfig& breaker);
    void reset_breaker(const std::string& breaker_id);
    
    // Alert management
    void generate_alert(RiskLevel level, RiskCategory category, const std::string& message);
    void check_alert_thresholds();
    
    // ML methods
    torch::Tensor extract_portfolio_features(const PortfolioRisk& portfolio);
    torch::Tensor extract_position_features(const PositionRisk& position);
    void update_ml_training_data(const PortfolioRisk& portfolio, bool was_risk_event);
    
    // Utility methods
    std::vector<double> calculate_returns(const std::vector<PositionRisk>& positions);
    double calculate_sharpe_ratio(const std::vector<double>& returns, double risk_free_rate = 0.02) const;
    double calculate_sortino_ratio(const std::vector<double>& returns) const;
    double calculate_max_drawdown(const std::vector<double>& equity_curve) const;
    
    // Compliance checking
    bool check_var_limits();
    bool check_leverage_limits();
    bool check_concentration_limits();
    bool check_reporting_requirements();
};

/**
 * @brief Real-time Risk Monitor
 */
class RealTimeRiskMonitor {
public:
    explicit RealTimeRiskMonitor(HierarchicalRiskManager& risk_manager);
    
    struct MonitoringConfig {
        int update_interval_ms = 100;
        double risk_threshold_multiplier = 0.8;
        bool enable_predictive_alerts = true;
        int max_alerts_per_minute = 10;
    };
    
    void start_monitoring(const MonitoringConfig& config = MonitoringConfig{});
    void stop_monitoring();
    
    struct RealTimeRiskMetrics {
        double current_portfolio_var;
        double current_leverage;
        double current_concentration;
        double current_correlation_risk;
        RiskLevel overall_risk_level;
        std::vector<RiskMetric> current_metrics;
        std::chrono::system_clock::time_point timestamp;
    };
    
    RealTimeRiskMetrics get_current_metrics() const;
    std::vector<std::string> get_risk_warnings() const;

private:
    HierarchicalRiskManager& risk_manager_;
    MonitoringConfig config_;
    std::atomic<bool> monitoring_active_;
    std::thread monitoring_thread_;
    
    void monitoring_thread_func();
    void update_risk_metrics();
    void check_threshold_violations();
};

/**
 * @brief Stress Test Engine
 */
class StressTestEngine {
public:
    explicit StressTestEngine(HierarchicalRiskManager& risk_manager);
    
    struct StressTestConfig {
        int num_monte_carlo_simulations = 10000;
        double confidence_levels[3] = {0.95, 0.99, 0.999};
        int time_horizons_days[3] = {1, 5, 22};
        bool enable_extreme_scenarios = true;
        double extreme_scenario_probability = 0.001;
    };
    
    void run_stress_test_suite(const StressTestConfig& config = StressTestConfig{});
    std::vector<StressTestScenario> get_results() const;
    
    StressTestScenario create_scenario(const std::string& name,
                                   const std::string& description,
                                   const std::unordered_map<std::string, double>& price_shocks);

private:
    HierarchicalRiskManager& risk_manager_;
    std::vector<StressTestScenario> results_;
    
    void run_historical_scenarios();
    void run_monte_carlo_scenarios(const StressTestConfig& config);
    void run_extreme_scenarios();
    void analyze_scenario_results(StressTestScenario& scenario);
};

/**
 * @brief Risk Analytics Engine
 */
class RiskAnalyticsEngine {
public:
    explicit RiskAnalyticsEngine(HierarchicalRiskManager& risk_manager);
    
    struct AnalyticsReport {
        double risk_efficiency_score;
        double risk_coverage_score;
        double prediction_accuracy;
        double alert_effectiveness;
        std::vector<std::string> key_insights;
        std::vector<std::string> recommendations;
        std::chrono::system_clock::time_point report_time;
    };
    
    AnalyticsReport generate_report(const std::chrono::hours period = std::chrono::hours(24));
    void record_risk_event(const std::string& event_type, double severity);
    std::vector<std::string> get_risk_insights();
    std::vector<std::string> get_optimization_recommendations();

private:
    HierarchicalRiskManager& risk_manager_;
    std::vector<std::tuple<std::string, double, std::chrono::system_clock::time_point>> risk_events_;
    std::mutex events_mutex_;
    
    void analyze_risk_patterns();
    void calculate_efficiency_metrics();
    void generate_insights();
};

/**
 * @brief RAII Hierarchical Risk Manager Context
 */
class HierarchicalRiskManagerContext {
public:
    explicit HierarchicalRiskManagerContext(const HierarchicalRiskConfig& config = HierarchicalRiskConfig{});
    ~HierarchicalRiskManagerContext();
    
    HierarchicalRiskManager& get_risk_manager();
    bool is_valid() const;

private:
    std::unique_ptr<HierarchicalRiskManager> risk_manager_;
    bool valid_;
};

} // namespace risk
} // namespace archneuronx
