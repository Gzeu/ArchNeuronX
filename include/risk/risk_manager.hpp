#pragma once
// ============================================================
// ArchNeuronX v2 - Risk Manager
// Position sizing, VaR, drawdown control, regime detection
// ============================================================
#include <vector>
#include <string>
#include <optional>
#include <cmath>

namespace archneuronx {
namespace risk {

enum class MarketRegime {
    TRENDING_UP,
    TRENDING_DOWN,
    RANGING,
    HIGH_VOLATILITY,
    LOW_VOLATILITY
};

struct RiskParameters {
    double max_portfolio_risk_pct = 0.02;   // Max 2% portfolio at risk per trade
    double max_drawdown_pct       = 0.10;   // Stop trading if DD > 10%
    double kelly_fraction         = 0.25;   // Quarter-Kelly for safety
    double min_confidence         = 0.65;   // Min signal confidence to trade
    double max_position_size_pct  = 0.05;   // Max 5% portfolio in single position
    int    var_lookback_days      = 252;    // 1 year for VaR calculation
    double var_confidence         = 0.95;   // 95% VaR
};

struct PositionSize {
    double quantity;        // Number of units to buy/sell
    double notional_value;  // In account currency
    double risk_amount;     // Max loss amount
    double stop_loss_price;
    double take_profit_price;
    std::string sizing_method; // "kelly", "fixed_fractional", "volatility_based"
};

struct RiskMetrics {
    double current_drawdown_pct;
    double max_drawdown_pct;
    double var_95;
    double var_99;
    double sharpe_ratio;
    double sortino_ratio;
    double portfolio_value;
    MarketRegime current_regime;
    bool trading_halted;
};

class RiskManager {
public:
    explicit RiskManager(RiskParameters params = {});

    // ---- Position Sizing ----
    // Kelly Criterion (fractional)
    [[nodiscard]] PositionSize kelly_position(
        double portfolio_value,
        double win_rate,
        double avg_win_pct,
        double avg_loss_pct,
        double current_price,
        double atr) const;

    // Fixed Fractional
    [[nodiscard]] PositionSize fixed_fractional_position(
        double portfolio_value,
        double risk_pct,
        double entry_price,
        double stop_loss_price) const;

    // Volatility-Based (ATR)
    [[nodiscard]] PositionSize volatility_based_position(
        double portfolio_value,
        double atr,
        double entry_price,
        double atr_multiplier = 2.0) const;

    // ---- Value at Risk ----
    [[nodiscard]] double var_historical(
        const std::vector<double>& daily_returns,
        double confidence = 0.95,
        int horizon_days = 1) const;

    [[nodiscard]] double var_parametric(
        double portfolio_value,
        double daily_vol,
        double confidence = 0.95,
        int horizon_days = 1) const;

    // ---- Market Regime Detection ----
    // ADX-based trend detection + Bollinger Band squeeze
    [[nodiscard]] MarketRegime detect_regime(
        const std::vector<double>& closes,
        const std::vector<double>& highs,
        const std::vector<double>& lows,
        const std::vector<double>& volumes,
        int adx_period = 14) const;

    [[nodiscard]] std::string regime_to_string(MarketRegime r) const;

    // ---- Portfolio Risk Management ----
    void update_portfolio_state(double current_value,
                                 double peak_value,
                                 const std::vector<double>& recent_returns);

    // Returns false if trading should be halted
    [[nodiscard]] bool should_trade(float signal_confidence) const;

    [[nodiscard]] RiskMetrics get_metrics() const;

    // Dynamic stop-loss based on ATR
    [[nodiscard]] double atr_stop_loss(double entry_price,
                                        double atr,
                                        double multiplier,
                                        bool is_long) const;

    // Take-profit with reward/risk ratio
    [[nodiscard]] double calculate_take_profit(double entry_price,
                                                double stop_loss,
                                                double rr_ratio = 2.0,
                                                bool is_long = true) const;

private:
    RiskParameters params_;
    RiskMetrics metrics_{};

    // Technical indicators helpers
    [[nodiscard]] double calculate_adx(
        const std::vector<double>& highs,
        const std::vector<double>& lows,
        const std::vector<double>& closes,
        int period) const;

    [[nodiscard]] double calculate_atr(
        const std::vector<double>& highs,
        const std::vector<double>& lows,
        const std::vector<double>& closes,
        int period = 14) const;

    [[nodiscard]] double calculate_sharpe(
        const std::vector<double>& returns,
        double risk_free_rate = 0.05) const;
};

} // namespace risk
} // namespace archneuronx
