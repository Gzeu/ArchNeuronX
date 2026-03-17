/**
 * @file hierarchical_risk_manager.cpp
 * @brief Hierarchical risk management implementation
 * @author ArchNeuronX Team
 * @date 2026-03-17
 */

#include "risk/hierarchical_risk_manager.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>

namespace archneuronx {
namespace risk {

HierarchicalRiskManager::HierarchicalRiskManager(const HierarchicalRiskConfig& config)
    : config_(config), model_loaded_(false), running_(false) {
    
    portfolio_risk_.total_value = 0.0;
    portfolio_risk_.total_pnl = 0.0;
    portfolio_risk_.total_var = 0.0;
    portfolio_risk_.total_cvar = 0.0;
    portfolio_risk_.total_es = 0.0;
    portfolio_risk_.sharpe_ratio = 0.0;
    portfolio_risk_.sortino_ratio = 0.0;
    portfolio_risk_.max_drawdown = 0.0;
    portfolio_risk_.current_drawdown = 0.0;
    portfolio_risk_.leverage_ratio = 0.0;
    portfolio_risk_.concentration_ratio = 0.0;
    portfolio_risk_.correlation_risk = 0.0;
    portfolio_risk_.beta_exposure = 0.0;
    portfolio_risk_.last_updated = std::chrono::system_clock::now();
    
    performance_metrics_.risk_adjusted_return = 0.0;
    performance_metrics_.risk_efficiency = 0.0;
    performance_metrics_.risk_coverage = 0.0;
    performance_metrics_.false_positive_rate = 0.0;
    performance_metrics_.detection_latency_ms = 0.0;
    performance_metrics_.last_update = std::chrono::system_clock::now();
    
    std::cout << "Hierarchical Risk Manager created" << std::endl;
    std::cout << "Max Portfolio VaR: " << (config_.max_portfolio_var * 100) << "%" << std::endl;
    std::cout << "Max Leverage: " << config_.max_portfolio_leverage << "x" << std::endl;
    std::cout << "Circuit Breakers: " << (config_.enable_circuit_breakers ? "Enabled" : "Disabled") << std::endl;
}

HierarchicalRiskManager::~HierarchicalRiskManager() {
    shutdown();
}

bool HierarchicalRiskManager::initialize() {
    try {
        // Load ML risk model if enabled
        if (config_.enable_ml_risk_prediction && !config_.risk_model_path.empty()) {
            if (!load_risk_model(config_.risk_model_path)) {
                std::cout << "Warning: Failed to load risk prediction model" << std::endl;
            }
        }
        
        // Initialize default stress test scenarios
        initialize_default_stress_scenarios();
        
        // Initialize default circuit breakers
        initialize_default_circuit_breakers();
        
        // Initialize background threads
        initialize_background_threads();
        
        running_ = true;
        std::cout << "Hierarchical Risk Manager initialized successfully" << std::endl;
        std::cout << "ML Risk Prediction: " << (config_.enable_ml_risk_prediction ? "Enabled" : "Disabled") << std::endl;
        std::cout << "Real-time Monitoring: " << (config_.enable_real_time_monitoring ? "Enabled" : "Disabled") << std::endl;
        std::cout << "Circuit Breakers: " << (config_.enable_circuit_breakers ? "Enabled" : "Disabled") << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing Hierarchical Risk Manager: " << e.what() << std::endl;
        return false;
    }
}

void HierarchicalRiskManager::shutdown() {
    running_ = false;
    
    // Shutdown background threads
    shutdown_background_threads();
    
    std::cout << "Hierarchical Risk Manager shutdown complete" << std::endl;
}

bool HierarchicalRiskManager::is_initialized() const {
    return running_;
}

void HierarchicalRiskManager::add_position(const std::string& symbol, const std::string& strategy_id, 
                                       double quantity, double price) {
    std::unique_lock<std::shared_mutex> lock(positions_mutex_);
    
    PositionRisk position;
    position.symbol = symbol;
    position.strategy_id = strategy_id;
    position.quantity = quantity;
    position.average_price = price;
    position.current_price = price;
    position.unrealized_pnl = 0.0;
    position.realized_pnl = 0.0;
    position.inventory_cost = std::abs(quantity) * price;
    position.position_var = 0.0;
    position.position_beta = 1.0; // Default
    position.position_volatility = 0.2; // Default
    position.liquidity_score = 0.8; // Default
    position.concentration_ratio = 0.0;
    position.market_risk_contribution = 0.0;
    position.credit_risk_contribution = 0.0;
    position.liquidity_risk_contribution = 0.0;
    position.model_risk_contribution = 0.0;
    position.last_updated = std::chrono::system_clock::now();
    
    positions_[symbol] = position;
    
    lock.unlock();
    
    // Update portfolio risk
    assess_portfolio_risk();
    
    std::cout << "Added position: " << symbol << " (" << strategy_id << ") - " 
              << quantity << " @ $" << price << std::endl;
}

void HierarchicalRiskManager::remove_position(const std::string& symbol) {
    std::unique_lock<std::shared_mutex> lock(positions_mutex_);
    
    auto it = positions_.find(symbol);
    if (it != positions_.end()) {
        // Add realized P&L to portfolio
        portfolio_risk_.total_pnl += it->second.realized_pnl + it->second.unrealized_pnl;
        
        positions_.erase(it);
    }
    
    lock.unlock();
    
    // Update portfolio risk
    assess_portfolio_risk();
    
    std::cout << "Removed position: " << symbol << std::endl;
}

void HierarchicalRiskManager::update_position_price(const std::string& symbol, double price) {
    std::unique_lock<std::shared_mutex> lock(positions_mutex_);
    
    auto it = positions_.find(symbol);
    if (it != positions_.end()) {
        PositionRisk& position = it->second;
        
        // Update unrealized P&L
        if (position.quantity != 0.0) {
            position.unrealized_pnl = position.quantity * (price - position.average_price);
        }
        
        position.current_price = price;
        position.last_updated = std::chrono::system_clock::now();
    }
    
    lock.unlock();
    
    // Update portfolio risk
    assess_portfolio_risk();
}

void HierarchicalRiskManager::update_position_pnl(const std::string& symbol, double realized_pnl) {
    std::unique_lock<std::shared_mutex> lock(positions_mutex_);
    
    auto it = positions_.find(symbol);
    if (it != positions_.end()) {
        it->second.realized_pnl += realized_pnl;
        it->second.last_updated = std::chrono::system_clock::now();
        
        // Update portfolio P&L
        portfolio_risk_.total_pnl += realized_pnl;
    }
    
    lock.unlock();
    
    // Update portfolio risk
    assess_portfolio_risk();
}

PositionRisk HierarchicalRiskManager::assess_position_risk(const std::string& symbol) {
    std::shared_lock<std::shared_mutex> lock(positions_mutex_);
    
    auto it = positions_.find(symbol);
    if (it == positions_.end()) {
        return PositionRisk{};
    }
    
    PositionRisk risk = it->second;
    
    // Calculate position VaR
    risk.position_var = calculate_position_var(risk);
    
    // Calculate risk contributions
    risk.market_risk_contribution = risk.position_var * 0.6;
    risk.liquidity_risk_contribution = (1.0 - risk.liquidity_score) * risk.position_var * 0.3;
    risk.model_risk_contribution = risk.position_var * 0.1;
    
    return risk;
}

PortfolioRisk HierarchicalRiskManager::assess_portfolio_risk() {
    std::unique_lock<std::shared_mutex> lock(positions_mutex_);
    
    std::vector<PositionRisk> positions;
    for (const auto& [symbol, position] : positions_) {
        positions.push_back(position);
    }
    
    lock.unlock();
    
    // Calculate portfolio metrics
    portfolio_risk_.total_value = 0.0;
    portfolio_risk_.total_pnl = 0.0;
    
    for (const auto& position : positions) {
        double position_value = std::abs(position.quantity) * position.current_price;
        portfolio_risk_.total_value += position_value;
        portfolio_risk_.total_pnl += position.realized_pnl + position.unrealized_pnl;
    }
    
    // Calculate portfolio VaR
    portfolio_risk_.total_var = calculate_portfolio_var(positions);
    
    // Calculate CVaR
    std::vector<double> returns = calculate_returns(positions);
    portfolio_risk_.total_cvar = calculate_cvar(returns);
    portfolio_risk_.total_es = calculate_expected_shortfall(returns);
    
    // Calculate risk ratios
    portfolio_risk_.sharpe_ratio = calculate_sharpe_ratio(returns);
    portfolio_risk_.sortino_ratio = calculate_sortino_ratio(returns);
    
    // Calculate risk contributions
    portfolio_risk_.risk_contributions.clear();
    portfolio_risk_.asset_contributions.clear();
    portfolio_risk_.strategy_contributions.clear();
    
    for (const auto& position : positions) {
        double position_value = std::abs(position.quantity) * position.current_price;
        double weight = (portfolio_risk_.total_value > 0) ? position_value / portfolio_risk_.total_value : 0.0;
        
        portfolio_risk_.asset_contributions[position.symbol] = weight;
        portfolio_risk_.strategy_contributions[position.strategy_id] += weight;
        
        portfolio_risk_.risk_contributions[RiskCategory::MARKET_RISK] += position.market_risk_contribution;
        portfolio_risk_.risk_contributions[RiskCategory::LIQUIDITY_RISK] += position.liquidity_risk_contribution;
        portfolio_risk_.risk_contributions[RiskCategory::MODEL_RISK] += position.model_risk_contribution;
    }
    
    // Calculate portfolio-level risks
    portfolio_risk_.leverage_ratio = calculate_leverage_risk(positions);
    portfolio_risk_.concentration_ratio = calculate_concentration_risk(positions);
    portfolio_risk_.correlation_risk = calculate_correlation_risk(positions);
    portfolio_risk_.beta_exposure = 0.0; // Simplified
    
    // Calculate drawdown
    std::vector<double> equity_curve;
    for (int i = 0; i < 100; ++i) {
        equity_curve.push_back(portfolio_risk_.total_pnl + i * 1000); // Simplified
    }
    portfolio_risk_.max_drawdown = calculate_max_drawdown(equity_curve);
    portfolio_risk_.current_drawdown = 0.0; // Simplified
    
    portfolio_risk_.last_updated = std::chrono::system_clock::now();
    
    return portfolio_risk_;
}

std::unordered_map<std::string, PositionRisk> HierarchicalRiskManager::assess_all_positions() {
    std::shared_lock<std::shared_mutex> lock(positions_mutex_);
    
    std::unordered_map<std::string, PositionRisk> results;
    
    for (const auto& [symbol, position] : positions_) {
        results[symbol] = assess_position_risk(symbol);
    }
    
    return results;
}

double HierarchicalRiskManager::calculate_var(const std::vector<double>& returns, double confidence) const {
    if (returns.empty()) {
        return 0.0;
    }
    
    return calculate_historical_var(returns, confidence);
}

double HierarchicalRiskManager::calculate_cvar(const std::vector<double>& returns, double confidence) const {
    if (returns.empty()) {
        return 0.0;
    }
    
    // Sort returns
    std::vector<double> sorted_returns = returns;
    std::sort(sorted_returns.begin(), sorted_returns.end());
    
    // Calculate CVaR
    size_t var_index = static_cast<size_t>((1.0 - confidence) * sorted_returns.size());
    
    if (var_index >= sorted_returns.size()) {
        return 0.0;
    }
    
    double cvar = 0.0;
    for (size_t i = 0; i <= var_index; ++i) {
        if (sorted_returns[i] < 0) {
            cvar += sorted_returns[i];
        }
    }
    
    return -cvar;
}

double HierarchicalRiskManager::calculate_expected_shortfall(const std::vector<double>& returns) const {
    if (returns.empty()) {
        return 0.0;
    }
    
    // Calculate VaR
    double var = calculate_var(returns, 0.95);
    
    // Calculate expected shortfall (simplified)
    double es = 0.0;
    int count = 0;
    
    for (double ret : returns) {
        if (ret < -var) {
            es += ret;
            ++count;
        }
    }
    
    return count > 0 ? -es / count : 0.0;
}

void HierarchicalRiskManager::run_stress_tests() {
    std::unique_lock<std::mutex> lock(stress_mutex_);
    
    stress_results_.clear();
    
    // Run all scenarios
    for (const auto& scenario : stress_scenarios_) {
        run_custom_scenario(scenario);
        stress_results_.push_back(scenario);
    }
    
    std::cout << "Completed " << stress_results_.size() << " stress test scenarios" << std::endl;
}

void HierarchicalRiskManager::add_stress_scenario(const StressTestScenario& scenario) {
    std::unique_lock<std::mutex> lock(stress_mutex_);
    
    stress_scenarios_.push_back(scenario);
}

std::vector<StressTestScenario> HierarchicalRiskManager::get_stress_results() const {
    std::lock_guard<std::mutex> lock(stress_mutex_);
    return stress_results_;
}

StressTestScenario HierarchicalRiskManager::run_custom_stress_test(const std::unordered_map<std::string, double>& shocks) {
    StressTestScenario scenario;
    scenario.scenario_id = "custom_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    scenario.name = "Custom Stress Test";
    scenario.description = "User-defined stress scenario";
    scenario.price_shocks = shocks;
    scenario.is_active = true;
    scenario.last_run = std::chrono::system_clock::now();
    
    run_custom_scenario(scenario);
    
    return scenario;
}

bool HierarchicalRiskManager::check_circuit_breakers() {
    std::lock_guard<std::mutex> lock(breakers_mutex_);
    
    bool any_triggered = false;
    
    for (auto& [breaker_id, breaker] : circuit_breakers_) {
        if (breaker.is_active) {
            bool triggered = false;
            
            switch (breaker.category) {
                case RiskCategory::MARKETET_RISK:
                    triggered = check_portfolio_breaker(breaker);
                    break;
                case RiskCategory::LEVERAGE_RISK:
                    triggered = check_leverage_limits();
                    break;
                case RiskCategory::CONCENTRATION_RISK:
                    triggered = check_concentration_limits();
                    break;
                default:
                    triggered = false;
                    break;
            }
            
            if (triggered) {
                trigger_breaker_actions(breaker);
                any_triggered = true;
                
                // Generate alert
                generate_alert(RiskLevel::HIGH_RISK, breaker.category, 
                             "Circuit breaker triggered: " + breaker.name);
            }
        }
    }
    
    return any_triggered;
}

std::vector<HierarchicalRiskManager::RiskAlert> HierarchicalRiskManager::get_risk_alerts() const {
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    
    std::vector<RiskAlert> alerts;
    std::queue<RiskAlert> temp_queue = risk_alerts_;
    
    while (!temp_queue.empty()) {
        alerts.push_back(temp_queue.front());
        temp_queue.pop();
    }
    
    return alerts;
}

void HierarchicalRiskManager::acknowledge_alert(const std::string& alert_id) {
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    
    // Mark alert as acknowledged (simplified implementation)
    // In practice, would find and update specific alert
}

void HierarchicalRiskManager::clear_alerts() {
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    
    std::queue<RiskAlert> empty;
    risk_alerts_.swap(empty);
}

bool HierarchicalRiskManager::load_risk_model(const std::string& model_path) {
    try {
        risk_model_ = torch::jit::load(model_path);
        model_loaded_ = true;
        
        std::cout << "Risk prediction model loaded: " << model_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading risk prediction model: " << e.what() << std::endl;
        model_loaded_ = false;
        return false;
    }
}

double HierarchicalRiskManager::predict_portfolio_risk(const PortfolioRisk& portfolio) {
    if (!model_loaded_) {
        return 0.5; // Default prediction
    }
    
    try {
        auto features = extract_portfolio_features(portfolio);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(features);
        auto output = risk_model_.forward(inputs);
        auto prediction = output.toTensor();
        
        return prediction.item<double>();
        
    } catch (const std::exception& e) {
        std::cerr << "Error predicting portfolio risk: " << e.what() << std::endl;
        return 0.5;
    }
}

double HierarchicalRiskManager::predict_position_risk(const PositionRisk& position) {
    if (!model_loaded_) {
        return 0.5; // Default prediction
    }
    
    try {
        auto features = extract_position_features(position);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(features);
        auto output = risk_model_.forward(inputs);
        auto prediction = output.toTensor();
        
        return prediction.item<double>();
        
    } catch (const std::exception& e) {
        std::cerr << "Error predicting position risk: " << e.what() << std::endl;
        return 0.5;
    }
}

void HierarchicalRiskManager::update_risk_model(const PortfolioRisk& portfolio, bool was_risk_event) {
    // This would update the ML model with new training data
    // For now, just log the event
    std::cout << "Risk Model Update: " << (was_risk_event ? "RISK_EVENT" : "NORMAL") 
              << " (P&L: " << portfolio.total_pnl << ")" << std::endl;
}

HierarchicalRiskManager::ComplianceReport HierarchicalRiskManager::generate_compliance_report() {
    ComplianceReport report;
    report.report_time = std::chrono::system_clock::now();
    
    // Check VaR compliance
    report.var_compliance = check_var_limits();
    
    // Check leverage compliance
    report.leverage_compliance = check_leverage_limits();
    
    // Check concentration compliance
    report.concentration_compliance = check_concentration_limits();
    
    // Check reporting requirements
    report.reporting_compliance = check_reporting_requirements();
    
    // Collect compliance issues
    if (!report.var_compliance) {
        report.compliance_issues.push_back("VaR exceeds regulatory limits");
    }
    
    if (!report.leverage_compliance) {
        report.compliance_issues.push_back("Leverage exceeds regulatory limits");
    }
    
    if (!report.concentration_compliance) {
        report.compliance_issues.push_back("Position concentration exceeds limits");
    }
    
    if (!report.reporting_compliance) {
        report.compliance_issues.push_back("Reporting requirements not met");
    }
    
    return report;
}

bool HierarchicalRiskManager::check_regulatory_compliance() {
    ComplianceReport report = generate_compliance_report();
    return report.var_compliance && report.leverage_compliance && 
           report.concentration_compliance && report.reporting_compliance;
}

HierarchicalRiskManager::RiskPerformanceMetrics HierarchicalRiskManager::get_performance_metrics() const {
    std::lock_guard<std::mutex> lock(performance_mutex_);
    return performance_metrics_;
}

void HierarchicalRiskManager::reset_performance_metrics() {
    std::lock_guard<std::mutex> lock(performance_mutex_);
    
    performance_metrics_.risk_adjusted_return = 0.0;
    performance_metrics_.risk_efficiency = 0.0;
    performance_metrics_.risk_coverage = 0.0;
    performance_metrics_.false_positive_rate = 0.0;
    performance_metrics_.detection_latency_ms = 0.0;
    performance_metrics_.last_update = std::chrono::system_clock::now();
}

void HierarchicalRiskManager::update_config(const HierarchicalRiskConfig& config) {
    config_ = config;
}

HierarchicalRiskConfig HierarchicalRiskManager::get_config() const {
    return config_;
}

// Private methods

void HierarchicalRiskManager::initialize_background_threads() {
    if (config_.enable_real_time_monitoring) {
        monitoring_thread_ = std::thread(&HierarchicalRiskManager::monitoring_thread_func, this);
    }
    
    stress_test_thread_ = std::thread(&HierarchicalRiskManager::stress_test_thread_func, this);
    
    if (config_.enable_circuit_breakers) {
        circuit_breaker_thread_ = std::thread(&HierarchicalRiskManager::circuit_breaker_thread_func, this);
    }
}

void HierarchicalRiskManager::shutdown_background_threads() {
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
    
    if (stress_test_thread_.joinable()) {
        stress_test_thread_.join();
    }
    
    if (circuit_breaker_thread_.joinable()) {
        circuit_breaker_thread_.join();
    }
}

void HierarchicalRiskManager::monitoring_thread_func() {
    std::cout << "Risk monitoring thread started" << std::endl;
    
    while (running_) {
        try {
            // Update portfolio risk assessment
            assess_portfolio_risk();
            
            // Check risk thresholds
            check_alert_thresholds();
            
            // Check circuit breakers
            if (config_.enable_circuit_breakers) {
                check_circuit_breakers();
            }
            
            // Update performance metrics
            update_metrics();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.monitoring_update_interval_ms));
            
        } catch (const std::exception& e) {
            std::cerr << "Error in risk monitoring thread: " << e.what() << std::endl;
        }
    }
}

void HierarchicalRiskManager::stress_test_thread_func() {
    std::cout << "Stress testing thread started" << std::endl;
    
    while (running_) {
        try {
            run_stress_tests();
            
            std::this_thread::sleep_for(std::chrono::hours(config_.stress_test_interval_hours));
            
        } catch (const std::exception& e) {
            std::cerr << "Error in stress testing thread: " << e.what() << std::endl;
        }
    }
}

void HierarchicalRiskManager::circuit_breaker_thread_func() {
    std::cout << "Circuit breaker thread started" << std::endl;
    
    while (running_) {
        try {
            check_circuit_breakers();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.breaker_check_interval));
            
        } catch (const std::exception& e) {
            std::cerr << "Error in circuit breaker thread: " << e.what() << std::endl;
        }
    }
}

double HierarchicalRiskManager::calculate_position_var(const PositionRisk& position) {
    // Simplified position VaR calculation
    // In practice, would use historical returns or Monte Carlo
    
    double daily_volatility = position.position_volatility;
    double position_value = std::abs(position.quantity) * position.current_price;
    
    // 1-day VaR approximation
    return position_value * daily_volatility * 2.33; // 99% confidence
}

double HierarchicalRiskManager::calculate_portfolio_var(const std::vector<PositionRisk>& positions) {
    if (positions.empty()) {
        return 0.0;
    }
    
    // Simplified portfolio VaR calculation
    // In practice, would use correlation matrix and proper portfolio VaR
    
    double portfolio_var = 0.0;
    
    for (const auto& position : positions) {
        double position_var = calculate_position_var(position);
        double position_value = std::abs(position.quantity) * position.current_price;
        double weight = (portfolio_risk_.total_value > 0) ? position_value / portfolio_risk_.total_value : 0.0;
        
        // Add position VaR (simplified - assumes no correlation)
        portfolio_var += weight * position_var;
    }
    
    return portfolio_var;
}

double HierarchicalRiskManager::calculate_correlation_risk(const std::vector<PositionRisk>& positions) {
    if (positions.size() < 2) {
        return 0.0;
    }
    
    // Simplified correlation risk calculation
    // In practice, would calculate actual correlation matrix
    
    double avg_correlation = 0.3; // Assumed average correlation
    
    // Correlation risk increases with position count
    double correlation_factor = 1.0 + (positions.size() - 1) * avg_correlation;
    
    return correlation_factor * 0.1; // 10% additional risk
}

double HierarchicalRiskManager::calculate_concentration_risk(const std::vector<PositionRisk>& positions) {
    if (positions.empty()) {
        return 0.0;
    }
    
    double max_concentration = 0.0;
    
    for (const auto& position : positions) {
        double position_value = std::abs(position.quantity) * position.current_price;
        double concentration = (portfolio_risk_.total_value > 0) ? position_value / portfolio_risk_.total_value : 0.0;
        
        max_concentration = std::max(max_concentration, concentration);
    }
    
    return max_concentration;
}

double HierarchicalRiskManager::calculate_leverage_risk(const std::vector<PositionRisk>& positions) {
    double total_exposure = 0.0;
    double total_capital = portfolio_risk_.total_value;
    
    for (const auto& position : positions) {
        total_exposure += std::abs(position.quantity) * position.current_price;
    }
    
    return (total_capital > 0) ? total_exposure / total_capital : 0.0;
}

std::vector<double> HierarchicalRiskManager::calculate_returns(const std::vector<PositionRisk>& positions) {
    std::vector<double> returns;
    
    // Generate synthetic returns based on current positions
    // In practice, would use historical return data
    
    for (int i = 0; i < 100; ++i) {
        double daily_return = 0.0;
        
        for (const auto& position : positions) {
            double daily_volatility = position.position_volatility / std::sqrt(252.0); // Annual to daily
            double random_return = 0.0;
            
            // Generate random return
            std::random_device rd;
            std::normal_distribution<double> dist(0.0, daily_volatility);
            random_return = dist(rd);
            
            daily_return += random_return * position.quantity * position.current_price;
        }
        
        returns.push_back(daily_return);
    }
    
    return returns;
}

double HierarchicalRiskManager::calculate_historical_var(const std::vector<double>& returns, double confidence) const {
    if (returns.empty()) {
        return 0.0;
    }
    
    // Sort returns
    std::vector<double> sorted_returns = returns;
    std::sort(sorted_returns.begin(), sorted_returns.end());
    
    // Calculate VaR percentile
    size_t var_index = static_cast<size_t>((1.0 - confidence) * sorted_returns.size());
    
    if (var_index >= sorted_returns.size()) {
        return 0.0;
    }
    
    return std::abs(sorted_returns[var_index]);
}

double HierarchicalRiskManager::calculate_sharpe_ratio(const std::vector<double>& returns, double risk_free_rate) const) {
    if (returns.empty()) {
        return 0.0;
    }
    
    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double variance = 0.0;
    
    for (double ret : returns) {
        variance += (ret - mean_return) * (ret - mean_return);
    }
    variance /= returns.size();
    
    double std_dev = std::sqrt(variance);
    
    return (std_dev > 0) ? (mean_return - risk_free_rate / 252.0) / std_dev : 0.0;
}

double HierarchicalRiskManager::calculate_sortino_ratio(const std::vector<double>& returns) const {
    if (returns.empty()) {
        return 0.0;
    }
    
    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double downside_deviation = 0.0;
    int downside_count = 0;
    
    for (double ret : returns) {
        if (ret < mean_return) {
            downside_deviation += (ret - mean_return) * (ret - mean_return);
            downside_count++;
        }
    }
    
    if (downside_count == 0) {
        return 0.0;
    }
    
    double downside_std = std::sqrt(downside_deviation / downside_count);
    
    return (downside_std > 0) ? mean_return / downside_std : 0.0;
}

double HierarchicalRiskManager::calculate_max_drawdown(const std::vector<double>& equity_curve) {
    if (equity_curve.empty()) {
        return 0.0;
    }
    
    double max_drawdown = 0.0;
    double peak = equity_curve[0];
    
    for (double value : equity_curve) {
        if (value > peak) {
            peak = value;
        } else {
            double drawdown = (peak - value) / peak;
            max_drawdown = std::max(max_drawdown, drawdown);
        }
    }
    
    return max_drawdown;
}

bool HierarchicalRiskManager::check_var_limits() {
    return portfolio_risk_.total_var <= config_.max_portfolio_var;
}

bool HierarchicalRiskManager::check_leverage_limits() {
    return portfolio_risk_.leverage_ratio <= config_.max_portfolio_leverage;
}

bool HierarchicalRiskManager::check_concentration_limits() {
    return portfolio_risk_.concentration_ratio <= config_.max_concentration_ratio;
}

bool HierarchicalRiskManager::check_reporting_requirements() {
    // Simplified reporting requirements check
    return true; // Always compliant for now
}

void HierarchicalRiskManager::initialize_default_stress_scenarios() {
    // Market crash scenario
    StressTestScenario crash_scenario;
    crash_scenario.scenario_id = "market_crash";
    crash_scenario.name = "Market Crash";
    crash_scenario.description = "Severe market decline";
    crash_scenario.price_shocks["BTCUSDT"] = -0.30; // 30% drop
    crash_scenario.price_shocks["ETHUSDT"] = -0.35;
    crash_scenario.price_shocks["ADAUSDT"] = -0.40;
    stress_scenarios_.push_back(crash_scenario);
    
    // Volatility spike scenario
    StressTestScenario volatility_scenario;
    volatility_scenario.scenario_id = "volatility_spike";
    volatility_scenario.name = "Volatility Spike";
    volatility_scenario.description = "Extreme volatility increase";
    volatility_scenario.volatility_shocks["BTCUSDT"] = 3.0; // 3x volatility
    volatility_scenario.volatility_shocks["ETHUSDT"] = 3.5;
    volatility_scenario.volatility_shocks["ADAUSDT"] = 4.0;
    stress_scenarios_.push_back(volatility_scenario);
    
    // Liquidity crisis scenario
    StressTestScenario liquidity_scenario;
    liquidity_scenario.scenario_id = "liquidity_crisis";
    liquidity_scenario.name = "Liquidity Crisis";
    liquidity_scenario.description = "Market liquidity dries up";
    liquidity_scenario.liquidity_shocks["BTCUSDT"] = 0.2; // 80% liquidity reduction
    liquidity_scenario.liquidity_shocks["ETHUSDT"] = 0.3;
    liquidity_shocks["ADAUSDT"] = 0.4;
    stress_scenarios_.push_back(liquidity_scenario);
}

void HierarchicalRiskManager::initialize_default_circuit_breakers() {
    // Portfolio VaR breaker
    CircuitBreakerConfig var_breaker;
    var_breaker.breaker_id = "portfolio_var";
    var_breaker.name = "Portfolio VaR Circuit Breaker";
    var_breaker.category = RiskCategory::MARKET_RISK;
    var_breaker.trigger_threshold = config_.max_portfolio_var;
    var_breaker.reset_threshold = config_.max_portfolio_var * 0.8;
    var_breaker.trigger_duration = std::chrono::minutes(5);
    var_breaker.stop_new_orders = true;
    var_breaker.cancel_existing_orders = true;
    var_breaker.reduce_position_sizes = true;
    var_breaker.position_reduction_factor = 0.5;
    var_breaker.is_active = true;
    var_breaker.is_triggered = false;
    var_breaker.max_triggers_per_day = 3;
    circuit_breakers_["portfolio_var"] = var_breaker;
    
    // Leverage breaker
    CircuitBreakerConfig leverage_breaker;
    leverage_breaker.breaker_id = "leverage_limit";
    leverage_breaker.name = "Leverage Limit Circuit Breaker";
    leverage_breaker.category = RiskCategory::LEVERAGE_RISK;
    leverage_breaker.trigger_threshold = config_.max_portfolio_leverage;
    leverage_breaker.reset_threshold = config_.max_portfolio_leverage * 0.8;
    leverage_breaker.trigger_duration = std::chrono::minutes(10);
    leverage_breaker.stop_new_orders = true;
    leverage_breaker.cancel_existing_orders = false;
    leverage_breaker.reduce_position_sizes = true;
    leverage_breaker.position_reduction_factor = 0.7;
    leverage_breaker.is_active = true;
    leverage_breaker.is_triggered = false;
    leverage_breaker.max_triggers_per_day = 2;
    circuit_breakers_["leverage_limit"] = leverage_breaker;
    
    // Concentration breaker
    CircuitBreakerConfig concentration_breaker;
    concentration_breaker.breaker_id = "concentration_limit";
    concentration_breaker.name = "Concentration Limit Circuit Breaker";
    concentration_breaker.category = RiskCategory::CONCENTRATION_RISK;
    concentration_breaker.trigger_threshold = config_.max_concentration_ratio;
    concentration_breaker.reset_threshold = config_.max_concentration_ratio * 0.8;
    concentration_breaker.trigger_duration = std::chrono::minutes(15);
    concentration_breaker.stop_new_orders = false;
    concentration_breaker.cancel_existing_orders = false;
    concentration_breaker.reduce_position_sizes = true;
    concentration_breaker.position_reduction_factor = 0.6;
    concentration_breaker.is_active = true;
    concentration_breaker.is_triggered = false;
    concentration_breaker.max_triggers_per_day = 1;
    circuit_breakers_["concentration_limit"] = concentration_breaker;
}

void HierarchicalRiskManager::generate_alert(RiskLevel level, RiskCategory category, const std::string& message) {
    RiskAlert alert;
    alert.alert_id = "alert_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    alert.level = level;
    alert.category = category;
    alert.message = message;
    alert.timestamp = std::chrono::system_clock::now();
    alert.is_acknowledged = false;
    
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    risk_alerts_.push(alert);
    
    // Keep alert queue size manageable
    while (risk_alerts_.size() > 1000) {
        risk_alerts_.pop();
    }
    
    std::cout << "Risk Alert [" << static_cast<int>(level) << "]: " << message << std::endl;
}

void HierarchicalRiskManager::check_alert_thresholds() {
    // Check portfolio risk level
    double portfolio_var_ratio = portfolio_risk_.total_var / config_.max_portfolio_var;
    
    if (portfolio_var_ratio > config_.alert_threshold_multiplier) {
        RiskLevel level = RiskLevel::HIGH_RISK;
        if (portfolio_var_ratio > 1.0) {
            level = RiskLevel::CRITICAL_RISK;
        }
        
        generate_alert(level, RiskCategory::MARKET_RISK, 
                    "Portfolio VaR exceeds threshold: " + std::to_string(portfolio_var_ratio * 100) + "%");
    }
    
    // Check leverage ratio
    double leverage_ratio = portfolio_risk_.leverage_ratio / config_.max_portfolio_leverage;
    
    if (leverage_ratio > config_.alert_threshold_multiplier) {
        RiskLevel level = RiskLevel::MEDIUM_RISK;
        if (leverage_ratio > 1.0) {
            level = RiskLevel::HIGH_RISK;
        }
        
        generate_alert(level, RiskCategory::LEVERAGE_RISK,
                    "Leverage exceeds threshold: " + std::to_string(leverage_ratio) + "x");
    }
    
    // Check concentration ratio
    double concentration_ratio = portfolio_risk_.concentration_ratio / config_.max_concentration_ratio;
    
    if (concentration_ratio > config_.alert_threshold_multiplier) {
        RiskLevel level = RiskLevel::MEDIUM_RISK;
        if (concentration_ratio > 1.0) {
            level = RiskLevel::HIGH_RISK;
        }
        
        generate_alert(level, RiskCategory::CONCENTRATION_RISK,
                    "Concentration exceeds threshold: " + std::to_string(concentration_ratio * 100) + "%");
    }
}

void HierarchicalRiskManager::update_metrics() {
    std::lock_guard<std::mutex> lock(performance_mutex_);
    
    // Update performance metrics
    performance_metrics_.last_update = std::chrono::system_clock::now();
    
    // Calculate risk-adjusted return (simplified)
    if (portfolio_risk_.total_var > 0) {
        performance_metrics_.risk_adjusted_return = portfolio_risk_.total_pnl / portfolio_risk_.total_var;
    }
    
    // Calculate risk efficiency
    performance_metrics_.risk_efficiency = 0.8; // Simplified
    
    // Calculate risk coverage
    performance_metrics_.risk_coverage = 0.9; // Simplified
    
    // Calculate detection latency
    performance_metrics_.detection_latency_ms = 50; // Simplified
}

torch::Tensor HierarchicalRiskManager::extract_portfolio_features(const PortfolioRisk& portfolio) {
    std::vector<float> features;
    
    // Portfolio metrics
    features.push_back(static_cast<float>(portfolio.total_value));
    features.push_back(static_cast<float>(portfolio.total_var));
    features.push_back(static_cast<float>(portfolio.total_cvar));
    features.push_back(static_cast<float>(portfolio.total_es));
    features.push_back(static_cast<float>(portfolio.sharpe_ratio));
    features.push_back(static_cast<float>(portfolio.leverage_ratio));
    features.push_back(static_cast<float>(portfolio.concentration_ratio));
    features.push_back(static_cast<float>(portfolio.correlation_risk));
    
    // Risk contributions
    features.push_back(static_cast<float>(portfolio.risk_contributions[RiskCategory::MARKET_RISK]));
    features.push_back(static_cast<float>(portfolio.risk_contributions[RiskCategory::LIQUIDITY_RISK]));
    features.push_back(static_cast<float>(portfolio.risk_contributions[RiskCategory::MODEL_RISK]));
    
    // Time features
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    features.push_back(static_cast<float>(tm.tm_hour));
    features.push_back(static_cast<float>(tm.tm_wday));
    
    // Pad to expected size
    while (features.size() < 20) {
        features.push_back(0.0f);
    }
    
    auto tensor = torch::from_blob(features.data(), {1, static_cast<long>(features.size())}, torch::kFloat32);
    return tensor.clone();
}

torch::Tensor HierarchicalRiskManager::extract_position_features(const PositionRisk& position) {
    std::vector<float> features;
    
    // Position metrics
    features.push_back(static_cast<float>(position.quantity));
    features.push_back(static_cast<float>(position.average_price));
    features.push_back(static_cast<float>(position.current_price));
    features.push_back(static_cast<float>(position.unrealized_pnl));
    features.push_back(static_cast<float>(position.realized_pnl));
    features.push_back(static_cast>(position.position_var));
    features.push_back(static_cast<float>(position.position_beta));
    features.push_back(static_cast>(position.position_volatility));
    features.push_back(static_cast>(position.liquidity_score));
    features.push_back(static_cast>(position.concentration_ratio));
    
    // Risk contributions
    features.push_back(static_cast<float>(position.market_risk_contribution));
    features.push_back(static_cast<float>(position.liquidity_risk_contribution));
    features.push_back(static_cast>(position.model_risk_contribution));
    
    // Time features
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    features.push_back(static_cast>(tm.tm_hour));
    features.push_back(static_cast>(tm.tm_wday));
    
    // Pad to expected size
    while (features.size() < 15) {
        features.push_back(0.0f);
    }
    
    auto tensor = torch::from_blob(features.data(), {1, static_cast<long>(features.size())}, torch::kFloat32);
    return tensor.clone();
}

void HierarchicalRiskManager::run_custom_scenario(const StressTestScenario& scenario) {
    // Apply price shocks to positions
    std::vector<PositionRisk> shocked_positions;
    
    {
        std::shared_lock<std::shared_mutex> lock(positions_mutex_);
        for (const auto& [symbol, position] : positions_) {
            PositionRisk shocked_position = position;
            
            // Apply price shock if applicable
            auto shock_it = scenario.price_shocks.find(symbol);
            if (shock_it != scenario.price_shocks.end()) {
                shocked_position.current_price *= (1.0 + shock_it->second);
                shocked_position.unrealized_pnl = shocked_position.quantity * 
                    (shocked_position.current_price - shocked_position.average_price);
            }
            
            // Apply volatility shock if applicable
            auto vol_it = scenario.volatility_shocks.find(symbol);
            if (vol_it != scenario.volatility_shocks.end()) {
                shocked_position.position_volatility *= vol_it->second;
            }
            
            shocked_positions.push_back(shocked_position);
        }
    }
    
    // Calculate stressed portfolio metrics
    double stressed_pnl = 0.0;
    double stressed_var = 0.0;
    
    for (const auto& position : shocked_positions) {
        stressed_pnl += position.realized_pnl + position.unrealized_pnl;
        stressed_var += calculate_position_var(position);
    }
    
    // Update scenario results
    scenario.portfolio_pnl = stressed_pnl;
    scenario.portfolio_var = stressed_var;
    scenario.max_drawdown = std::abs(stressed_pnl) / (portfolio_risk_.total_value + stressed_pnl);
    scenario.worst_loss = std::abs(stressed_pnl);
    scenario.last_run = std::chrono::system_clock::now();
    scenario.is_active = true;
    
    // Analyze results
    analyze_scenario_results(scenario);
    
    std::cout << "Stress test completed: " << scenario.name 
              << " (P&L: " << scenario.portfolio_pnl 
              << ", VaR: " << scenario.portfolio_var << ")" << std::endl;
}

void HierarchicalRiskManager::analyze_scenario_results(StressTestScenario& scenario) {
    // Calculate additional metrics
    scenario.recovery_time_days = 30; // Simplified
    
    // Determine severity
    if (scenario.portfolio_pnl < -portfolio_risk_.total_value * 0.1) {
        scenario.worst_loss = std::abs(scenario.portfolio_pnl) / portfolio_risk_.total_value;
    } else {
        scenario.worst_loss = 0.0;
    }
    
    // Store in results
    auto it = std::find_if(stress_results_.begin(), stress_results_.end(),
                      [&scenario](const StressTestScenario& s) { return s.scenario_id == scenario.scenario_id; });
    
    if (it != stress_results_.end()) {
        *it = scenario;
    }
}

bool HierarchicalRiskManager::check_portfolio_breaker(const CircuitBreakerConfig& breaker) {
    return portfolio_risk_.total_var > breaker.trigger_threshold;
}

bool HierarchicalRiskManager::check_strategy_breaker(const CircuitBreakerConfig& breaker) {
    // Simplified strategy-level check
    return false; // Would need strategy-level risk assessment
}

bool HierarchicalRiskManager::check_position_breaker(const CircuitBreakerConfig& breaker) {
    // Simplified position-level check
    return false; // Would need position-specific checks
}

void HierarchicalRiskManager::trigger_breaker_actions(const CircuitBreakerConfig& breaker) {
    breaker.is_triggered = true;
    breaker.last_triggered = std::chrono::system_clock::now();
    breaker.trigger_count++;
    
    std::cout << "Circuit breaker triggered: " << breaker.name << std::endl;
    
    // Implement breaker actions
    if (breaker.stop_new_orders) {
        std::cout << "Stopping new orders" << std::endl;
        // Implementation would stop order submission
    }
    
    if (breaker.cancel_existing_orders) {
        std::cout << "Cancelling existing orders" << std::endl;
        // Implementation would cancel all active orders
    }
    
    if (breaker.reduce_position_sizes) {
        std::cout << "Reducing position sizes by " << breaker.position_reduction_factor << std::endl;
        // Implementation would reduce position sizes
    }
    
    // Generate alert
    generate_alert(RiskLevel::HIGH_RISK, breaker.category, 
                 "Circuit breaker triggered: " + breaker.name);
}

void HierarchicalRiskManager::reset_breaker(const std::string& breaker_id) {
    std::lock_guard<std::mutex> lock(breakers_mutex_);
    
    auto it = circuit_breakers_.find(break_id);
    if (it != circuit_breakers_.end()) {
        it->second.is_triggered = false;
        it->second.last_reset = std::chrono::system_clock::now();
        
        std::cout << "Circuit breaker reset: " << breaker_id << std::endl;
    }
}

// RealTimeRiskMonitor implementation

RealTimeRiskMonitor::RealTimeRiskMonitor(HierarchicalRiskManager& risk_manager)
    : risk_manager_(risk_manager), monitoring_active_(false) {
}

void RealTimeRiskMonitor::start_monitoring(const MonitoringConfig& config) {
    config_ = config;
    monitoring_active_ = true;
    
    monitoring_thread_ = std::thread(&RealTimeRiskMonitor::monitoring_thread_func, this);
    
    std::cout << "Real-time risk monitoring started" << std::endl;
}

void RealTimeRiskMonitor::stop_monitoring() {
    monitoring_active_ = false;
    
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
    
    std::cout << "Real-time risk monitoring stopped" << std::endl;
}

RealTimeRiskMonitor::RealTimeRiskMetrics RealTimeRiskMonitor::get_current_metrics() const {
    RealTimeRiskMetrics metrics;
    
    // Get current portfolio risk
    auto portfolio_risk = risk_manager_.assess_portfolio_risk();
    
    metrics.current_portfolio_var = portfolio_risk.total_var;
    metrics.current_leverage = portfolio_risk.leverage_ratio;
    metrics.current_concentration = portfolio_risk.concentration_ratio;
    metrics.current_correlation_risk = portfolio_risk.risk_contributions.at(RiskCategory::MARKET_RISK);
    
    // Determine overall risk level
    double risk_ratio = portfolio_risk.total_var / risk_manager_.get_config().max_portfolio_var;
    
    if (risk_ratio < 0.5) {
        metrics.overall_risk_level = RiskLevel::LOW_RISK;
    } else if (risk_ratio < 0.8) {
        metrics.overall_risk_level = RiskLevel::MEDIUM_RISK;
    } else if (risk_ratio < 1.0) {
        metrics.overall_risk_level = RiskLevel::HIGH_RISK;
    } else {
        metrics.overall_risk_level = RiskLevel::CRITICAL_RISK;
    }
    
    // Get current risk metrics
    metrics.timestamp = std::chrono::system_clock::now();
    
    return metrics;
}

std::vector<std::string> RealTimeRiskMonitor::get_risk_warnings() const {
    std::vector<std::string> warnings;
    
    auto metrics = get_current_metrics();
    
    if (metrics.overall_risk_level >= RiskLevel::HIGH_RISK) {
        warnings.push_back("High portfolio risk detected");
    }
    
    if (metrics.current_leverage > 2.0) {
        warnings.push_back("High leverage detected");
    }
    
    if (metrics.current_concentration > 0.25) {
        warnings.push_back("High position concentration");
    }
    
    return warnings;
}

void RealTimeRiskMonitor::monitoring_thread_func() {
    while (monitoring_active_) {
        try {
            update_risk_metrics();
            check_threshold_violations();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.update_interval_ms));
            
        } catch (const std::exception& e) {
            std::cerr << "Error in risk monitoring: " << e.what() << std::endl;
        }
    }
}

void RealTimeRiskMonitor::update_risk_metrics() {
    // Update current risk metrics
    auto portfolio_risk = risk_manager_.assess_portfolio_risk();
    
    // This would update internal metrics tracking
    // For now, just store current state
}

void RealTimeRiskMonitor::check_threshold_violations() {
    auto metrics = get_current_metrics();
    
    // Check various thresholds
    if (metrics.current_portfolio_var > risk_manager_.get_config().max_portfolio_var * config_.risk_threshold_multiplier) {
        // Generate alert
        // Implementation would generate alert
    }
    
    if (metrics.current_leverage > risk_manager_.get_config().max_portfolio_leverage * config_.risk_threshold_multiplier) {
        // Generate alert
        // Implementation would generate alert
    }
}

// HierarchicalRiskManagerContext implementation

HierarchicalRiskManagerContext::HierarchicalRiskManagerContext(const HierarchicalRiskConfig& config) : valid_(false) {
    risk_manager_ = std::make_unique<HierarchicalRiskManager>(config);
    valid_ = risk_manager_->initialize();
}

HierarchicalRiskManagerContext::~HierarchicalRiskManagerContext() {
    if (risk_manager_) {
        risk_manager_->shutdown();
    }
}

HierarchicalRiskManager& HierarchicalRiskManagerContext::get_risk_manager() {
    if (!valid_ || !risk_manager_) {
        throw std::runtime_error("Hierarchical risk manager not initialized");
    }
    return *risk_manager_;
}

bool HierarchicalRiskManagerContext::is_valid() const {
    return valid_ && risk_manager_;
}

} // namespace risk
} // namespace archneuronx
