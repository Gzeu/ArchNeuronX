/**
 * @file risk_manager.hpp
 * @brief Risk Management System for Algorithmic Trading
 * @version 2.0.0
 *
 * Implements:
 * - Position sizing (Kelly Criterion, Fixed Fractional, Volatility-adjusted)
 * - Stop-loss / Take-profit automation
 * - Value at Risk (VaR) calculation (Historical, Parametric, Monte Carlo)
 * - Portfolio-level risk metrics
 * - Maximum drawdown monitoring
 * - Exposure limits per asset/sector
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <functional>
#include <chrono>
#include <cmath>

namespace ArchNeuronX {
namespace Core {

// ============================================================
// Enums
// ============================================================
enum class PositionSizingMethod {
    FIXED_FRACTIONAL,      // Fixed % of portfolio
    KELLY_CRITERION,       // Optimal Kelly
    HALF_KELLY,            // Conservative Kelly
    VOLATILITY_ADJUSTED,   // 1/ATR sizing
    EQUAL_WEIGHT,          // Equal dollar per position
    RISK_PARITY,           // Equal risk contribution
};

enum class OrderSide { BUY, SELL };

enum class OrderType {
    MARKET,
    LIMIT,
    STOP,
    STOP_LIMIT,
    OCO,           // One-Cancels-Other
    TRAILING_STOP,
};

enum class MarketRegime {
    TRENDING_UP,
    TRENDING_DOWN,
    RANGING,
    HIGH_VOLATILITY,
    LOW_VOLATILITY,
    UNKNOWN,
};

// ============================================================
// Data structures
// ============================================================
struct Position {
    std::string symbol;
    double quantity = 0.0;
    double entry_price = 0.0;
    double current_price = 0.0;
    double stop_loss = 0.0;
    double take_profit = 0.0;
    double unrealized_pnl = 0.0;
    OrderSide side = OrderSide::BUY;
    std::chrono::system_clock::time_point entry_time;
};

struct RiskParameters {
    // Portfolio limits
    double max_portfolio_risk = 0.02;    // 2% max risk per trade
    double max_total_exposure = 0.95;    // 95% max invested
    double max_drawdown_limit = 0.15;    // 15% max drawdown before halt
    double max_position_size = 0.10;     // 10% per single position
    double max_sector_exposure = 0.30;   // 30% per sector
    double max_correlation = 0.70;       // Block highly correlated positions

    // Stop-loss
    double default_stop_loss_pct = 0.02; // 2% hard stop
    double trailing_stop_pct = 0.03;     // 3% trailing
    bool use_atr_stop = true;            // ATR-based dynamic stop
    double atr_multiplier = 2.0;

    // Take profit
    double risk_reward_ratio = 2.0;      // 2:1 R:R minimum
    bool use_partial_profit = true;      // Take 50% at 1:1

    // Position sizing
    PositionSizingMethod sizing_method = PositionSizingMethod::VOLATILITY_ADJUSTED;
    double kelly_fraction = 0.25;        // Quarter Kelly for safety
    double risk_per_trade = 0.01;        // 1% risk per trade

    // Regime adjustments
    double high_vol_scale = 0.5;         // Halve size in high volatility
    double trending_scale = 1.2;         // 20% more in strong trends
};

struct VaRResult {
    double var_95;          // 95% confidence VaR
    double var_99;          // 99% confidence VaR
    double cvar_95;         // Conditional VaR (Expected Shortfall)
    double cvar_99;
    std::string method;     // "historical", "parametric", "montecarlo"
};

struct RiskMetrics {
    double total_value;
    double cash;
    double invested;
    double unrealized_pnl;
    double realized_pnl;
    double sharpe_ratio;
    double sortino_ratio;
    double calmar_ratio;
    double max_drawdown;
    double current_drawdown;
    double win_rate;
    double profit_factor;
    VaRResult var;
    int num_open_positions;
    MarketRegime regime;
};

// ============================================================
// RiskManager class
// ============================================================
class RiskManager {
public:
    explicit RiskManager(const RiskParameters& params = {});

    // ---- Position Sizing ------------------------------------

    /**
     * @brief Calculate optimal position size
     * @param symbol       Asset symbol
     * @param portfolio_value  Current portfolio value
     * @param entry_price  Entry price
     * @param atr          Current ATR (for volatility sizing)
     * @param win_rate     Historical win rate [0,1]
     * @param avg_win      Average winning trade return
     * @param avg_loss     Average losing trade return (positive)
     * @return Quantity to buy/sell
     */
    double calculate_position_size(const std::string& symbol,
                                   double portfolio_value,
                                   double entry_price,
                                   double atr = 0.0,
                                   double win_rate = 0.5,
                                   double avg_win = 0.02,
                                   double avg_loss = 0.01);

    // ---- Stop-Loss / Take-Profit ----------------------------

    /**
     * @brief Calculate stop-loss and take-profit levels
     * @param entry_price  Trade entry price
     * @param side         BUY or SELL
     * @param atr          ATR for dynamic stop
     * @return {stop_loss, take_profit}
     */
    std::pair<double, double> calculate_stops(double entry_price,
                                               OrderSide side,
                                               double atr = 0.0);

    /** Update trailing stop based on current price */
    double update_trailing_stop(const Position& pos, double current_price);

    // ---- Trade Validation ----------------------------------

    /**
     * @brief Validate if a trade should be allowed
     * @return true if trade passes all risk checks
     */
    bool validate_trade(const std::string& symbol,
                        double quantity,
                        double entry_price,
                        OrderSide side,
                        const std::vector<Position>& open_positions,
                        double portfolio_value);

    // ---- Portfolio Risk Metrics ----------------------------

    /** Compute full risk metrics snapshot */
    RiskMetrics compute_metrics(
        const std::vector<Position>& positions,
        const std::vector<double>& returns_history,
        double portfolio_value,
        double cash);

    /** Value at Risk using historical simulation */
    VaRResult calculate_var_historical(
        const std::vector<double>& returns,
        double portfolio_value,
        int horizon_days = 1);

    /** Value at Risk using parametric (Gaussian) method */
    VaRResult calculate_var_parametric(
        double mean_return,
        double std_return,
        double portfolio_value,
        int horizon_days = 1);

    /** Maximum Drawdown from equity curve */
    double calculate_max_drawdown(
        const std::vector<double>& equity_curve);

    // ---- Market Regime Detection ---------------------------

    /**
     * @brief Detect current market regime
     * @param prices   Recent close prices
     * @param volumes  Recent volume data
     * @return Current market regime
     */
    MarketRegime detect_regime(
        const std::vector<double>& prices,
        const std::vector<double>& volumes = {});

    // ---- Emergency Controls --------------------------------

    /** Circuit breaker - halt trading if drawdown exceeded */
    bool is_circuit_breaker_active() const;

    /** Reset circuit breaker (manual) */
    void reset_circuit_breaker();

    /** Register callback for risk limit breaches */
    void on_risk_breach(std::function<void(const std::string&)> callback);

    // ---- Configuration ------------------------------------
    void update_params(const RiskParameters& params);
    const RiskParameters& params() const { return params_; }

private:
    RiskParameters params_;
    bool circuit_breaker_active_ = false;
    double peak_equity_ = 0.0;
    double current_drawdown_ = 0.0;

    std::function<void(const std::string&)> breach_callback_;

    double kelly_fraction(double win_rate, double avg_win, double avg_loss);
    double sharpe_ratio(const std::vector<double>& returns, double risk_free = 0.0);
    double sortino_ratio(const std::vector<double>& returns, double risk_free = 0.0);
};

} // namespace Core
} // namespace ArchNeuronX
