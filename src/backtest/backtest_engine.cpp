#include "backtest/backtest_engine.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace archneuronx {
namespace backtest {

BacktestEngine::BacktestEngine(std::shared_ptr<trading::SignalGenerator> signal_gen,
                                std::shared_ptr<risk::RiskManager> risk_manager,
                                const BacktestConfig& config)
    : signal_gen_(std::move(signal_gen)),
      risk_manager_(std::move(risk_manager)),
      config_(config) {
    std::cout << "[BacktestEngine] Initialized: "
              << config.start_date << " to " << config.end_date
              << ", initial capital=$" << config.initial_capital << std::endl;
}

BacktestResult BacktestEngine::run(const std::vector<OHLCV>& data) {
    BacktestResult result;
    result.initial_capital = config_.initial_capital;

    double portfolio_value = config_.initial_capital;
    double peak_value = config_.initial_capital;
    double cash = config_.initial_capital;
    double position = 0.0;
    double entry_price = 0.0;
    int trade_count = 0;
    std::vector<double> returns;
    double prev_portfolio = config_.initial_capital;

    for (size_t i = config_.lookback_period; i < data.size(); ++i) {
        const auto& bar = data[i];

        // Build market snapshot
        MarketSnapshot snapshot;
        snapshot.symbol = config_.symbol;
        snapshot.close_price = bar.close;
        snapshot.portfolio_value = portfolio_value;
        snapshot.volatility = calculateVolatility(data, i, 20);
        snapshot.atr = calculateATR(data, i, 14);
        snapshot.current_drawdown_pct = peak_value > 0 ?
            (peak_value - portfolio_value) / peak_value * 100.0 : 0.0;
        snapshot.features = buildFeatures(data, i);

        // Generate signal
        auto signal = signal_gen_->generate(snapshot);

        // Execute trade logic
        if (signal.action == SignalAction::BUY && cash > 0) {
            double buy_size = std::min(signal.position_size, cash);
            position = buy_size / bar.close;
            cash -= buy_size;
            entry_price = bar.close;
            ++trade_count;
            result.trades.push_back({bar.timestamp, "BUY", bar.close, position});
        } else if (signal.action == SignalAction::SELL && position > 0) {
            double sell_value = position * bar.close;
            double trade_return = (bar.close - entry_price) / entry_price;
            returns.push_back(trade_return);
            cash += sell_value;
            position = 0.0;
            ++trade_count;
            result.trades.push_back({bar.timestamp, "SELL", bar.close, sell_value});
        }

        // Update portfolio value
        portfolio_value = cash + (position * bar.close);
        peak_value = std::max(peak_value, portfolio_value);

        // Track daily returns
        double daily_return = (portfolio_value - prev_portfolio) / prev_portfolio;
        result.daily_returns.push_back(daily_return);
        prev_portfolio = portfolio_value;
    }

    // Calculate final metrics
    result.final_capital = portfolio_value;
    result.total_return_pct = (portfolio_value - config_.initial_capital) /
                               config_.initial_capital * 100.0;
    result.trade_count = trade_count;

    // Use risk manager for metrics
    auto metrics = risk_manager_->calculateMetrics(
        result.daily_returns, portfolio_value, peak_value);
    result.sharpe_ratio = metrics.sharpe_ratio;
    result.sortino_ratio = metrics.sortino_ratio;
    result.max_drawdown_pct = metrics.max_drawdown;
    result.win_rate = metrics.win_rate;
    result.var_95 = metrics.var_95;

    printSummary(result);
    return result;
}

void BacktestEngine::printSummary(const BacktestResult& result) const {
    std::cout << "\n=== BACKTEST RESULTS ==" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Initial Capital:  $" << result.initial_capital << std::endl;
    std::cout << "Final Capital:    $" << result.final_capital << std::endl;
    std::cout << "Total Return:      " << result.total_return_pct << "%" << std::endl;
    std::cout << "Sharpe Ratio:      " << result.sharpe_ratio << std::endl;
    std::cout << "Sortino Ratio:     " << result.sortino_ratio << std::endl;
    std::cout << "Max Drawdown:      " << result.max_drawdown_pct << "%" << std::endl;
    std::cout << "Win Rate:          " << result.win_rate << "%" << std::endl;
    std::cout << "VaR (95%):         " << result.var_95 * 100.0 << "%" << std::endl;
    std::cout << "Total Trades:      " << result.trade_count << std::endl;
    std::cout << "======================" << std::endl;
}

} // namespace backtest
} // namespace archneuronx
