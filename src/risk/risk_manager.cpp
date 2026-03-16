#include "risk/risk_manager.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <iostream>

namespace archneuronx {
namespace risk {

RiskManager::RiskManager(const RiskConfig& config)
    : config_(config), circuit_breaker_triggered_(false) {
    std::cout << "[RiskManager] Initialized with max_drawdown="
              << config.max_drawdown_pct << "%, VaR confidence="
              << config.var_confidence << std::endl;
}

double RiskManager::calculateVaR(const std::vector<double>& returns,
                                  double confidence_level) const {
    if (returns.empty()) throw std::invalid_argument("Returns vector is empty");

    std::vector<double> sorted_returns = returns;
    std::sort(sorted_returns.begin(), sorted_returns.end());

    double alpha = 1.0 - confidence_level;
    size_t index = static_cast<size_t>(alpha * sorted_returns.size());
    return -sorted_returns[std::min(index, sorted_returns.size() - 1)];
}

double RiskManager::calculateCVaR(const std::vector<double>& returns,
                                   double confidence_level) const {
    if (returns.empty()) throw std::invalid_argument("Returns vector is empty");

    std::vector<double> sorted_returns = returns;
    std::sort(sorted_returns.begin(), sorted_returns.end());

    double alpha = 1.0 - confidence_level;
    size_t cutoff = static_cast<size_t>(alpha * sorted_returns.size());

    double cvar_sum = 0.0;
    for (size_t i = 0; i < cutoff; ++i) {
        cvar_sum += sorted_returns[i];
    }
    return cutoff > 0 ? -cvar_sum / cutoff : 0.0;
}

double RiskManager::calculatePositionSize(double portfolio_value,
                                          double signal_confidence,
                                          double asset_volatility) const {
    // Kelly criterion adjusted position sizing
    double kelly_fraction = (signal_confidence - (1.0 - signal_confidence)) /
                            (asset_volatility + 1e-8);
    kelly_fraction = std::clamp(kelly_fraction, 0.0, config_.max_position_pct / 100.0);

    // Apply portfolio-level constraints
    double position_size = portfolio_value * kelly_fraction * config_.leverage_limit;
    return std::min(position_size, portfolio_value * config_.max_position_pct / 100.0);
}

bool RiskManager::checkCircuitBreaker(double current_drawdown_pct) {
    if (current_drawdown_pct >= config_.max_drawdown_pct) {
        if (!circuit_breaker_triggered_) {
            circuit_breaker_triggered_ = true;
            std::cerr << "[RiskManager] CIRCUIT BREAKER TRIGGERED! Drawdown: "
                      << current_drawdown_pct << "% >= limit "
                      << config_.max_drawdown_pct << "%" << std::endl;
        }
        return true;
    }
    return false;
}

void RiskManager::resetCircuitBreaker() {
    circuit_breaker_triggered_ = false;
    std::cout << "[RiskManager] Circuit breaker reset" << std::endl;
}

RiskMetrics RiskManager::calculateMetrics(const std::vector<double>& returns,
                                           double portfolio_value,
                                           double peak_value) const {
    RiskMetrics metrics;

    if (returns.empty()) return metrics;

    // Sharpe ratio (annualized, assuming daily returns)
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double variance = 0.0;
    for (double r : returns) variance += (r - mean) * (r - mean);
    variance /= returns.size();
    double std_dev = std::sqrt(variance);

    metrics.sharpe_ratio = std_dev > 0 ? (mean / std_dev) * std::sqrt(252.0) : 0.0;

    // Sortino ratio (downside deviation)
    double downside_var = 0.0;
    int neg_count = 0;
    for (double r : returns) {
        if (r < 0) { downside_var += r * r; ++neg_count; }
    }
    double downside_dev = neg_count > 0 ? std::sqrt(downside_var / neg_count) : 1e-8;
    metrics.sortino_ratio = (mean / downside_dev) * std::sqrt(252.0);

    // Max drawdown
    metrics.max_drawdown = peak_value > 0 ?
        (peak_value - portfolio_value) / peak_value * 100.0 : 0.0;

    // VaR and CVaR
    metrics.var_95 = calculateVaR(returns, 0.95);
    metrics.var_99 = calculateVaR(returns, 0.99);
    metrics.cvar_95 = calculateCVaR(returns, 0.95);

    // Win rate
    int wins = std::count_if(returns.begin(), returns.end(),
                             [](double r) { return r > 0; });
    metrics.win_rate = static_cast<double>(wins) / returns.size() * 100.0;

    return metrics;
}

} // namespace risk
} // namespace archneuronx
