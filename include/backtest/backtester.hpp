#pragma once
// ============================================================
// ArchNeuronX v2 - Backtesting Engine
// Event-driven backtester with realistic simulation:
//   - Commission modeling
//   - Slippage simulation
//   - Market regime awareness
//   - Detailed trade journal
// ============================================================
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include "../data/preprocessor.hpp"

namespace archneuronx {
namespace backtest {

struct Trade {
    std::string symbol;
    std::string side;          // "BUY" or "SELL"
    double entry_price;
    double exit_price;
    double quantity;
    double pnl;
    double pnl_pct;
    double commission;
    int64_t entry_timestamp_ms;
    int64_t exit_timestamp_ms;
    int64_t hold_duration_ms;
    float entry_confidence;
    std::string exit_reason;   // "signal", "stop_loss", "take_profit", "end_of_data"
};

struct BacktestConfig {
    double initial_capital   = 10000.0;
    double commission_pct    = 0.001;   // 0.1% Binance spot
    double slippage_pct      = 0.0005;  // 0.05% slippage
    double max_position_pct  = 0.95;    // Max 95% of capital per trade
    float  min_confidence    = 0.65f;   // Min signal confidence to enter
    double stop_loss_atr_mult = 2.0;    // Stop = entry ± 2*ATR
    double take_profit_rr    = 2.0;     // TP = entry ± 2*stop_distance
    bool   allow_short       = false;   // Short selling enabled
    bool   compound_returns  = true;    // Reinvest profits
};

struct BacktestReport {
    // Returns
    double total_return_pct;
    double annualized_return_pct;
    double buy_hold_return_pct;     // Benchmark
    double alpha;                    // vs buy & hold
    // Risk-adjusted
    double sharpe_ratio;
    double sortino_ratio;
    double calmar_ratio;             // Return / Max Drawdown
    double max_drawdown_pct;
    double avg_drawdown_pct;
    // Trading stats
    int    total_trades;
    int    winning_trades;
    int    losing_trades;
    double win_rate;
    double avg_win_pct;
    double avg_loss_pct;
    double largest_win_pct;
    double largest_loss_pct;
    double profit_factor;            // Gross profit / Gross loss
    double expectancy;               // Expected return per trade
    double avg_hold_hours;
    // Capital
    double initial_capital;
    double final_equity;
    double peak_equity;
    // Monthly breakdown
    std::vector<std::pair<std::string, double>> monthly_returns;
    // Trade journal
    std::vector<Trade> trades;
    // Equity curve
    std::vector<std::pair<int64_t, double>> equity_curve;
};

using SignalFunction = std::function<std::string(
    const std::vector<archneuronx::data::OHLCVBar>&,  // bars
    float&  // output: confidence
)>;

class Backtester {
public:
    explicit Backtester(BacktestConfig config = {});

    // Run backtest with a signal-generating function
    [[nodiscard]] BacktestReport run(
        const std::vector<archneuronx::data::OHLCVBar>& bars,
        SignalFunction signal_fn);

    // Run with a TorchScript model directly
    [[nodiscard]] BacktestReport run_with_model(
        const std::vector<archneuronx::data::OHLCVBar>& bars,
        const std::string& model_path);

    // Walk-forward analysis (prevents overfitting)
    [[nodiscard]] std::vector<BacktestReport> walk_forward(
        const std::vector<archneuronx::data::OHLCVBar>& bars,
        SignalFunction signal_fn,
        int train_window = 1000,
        int test_window  = 200,
        int step_size    = 100);

    // Generate HTML report
    [[nodiscard]] std::string generate_html_report(
        const BacktestReport& report,
        const std::string& output_path = "") const;

private:
    BacktestConfig config_;

    double calculate_slippage(double price, const std::string& side) const;
    double apply_commission(double value) const;
};

} // namespace backtest
} // namespace archneuronx
