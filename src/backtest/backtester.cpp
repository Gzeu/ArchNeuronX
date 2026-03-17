/**
 * @file backtester.cpp
 * @brief Comprehensive backtesting engine with walk-forward analysis
 * @author George Pricop
 * @date 2025-10-02
 */

#include "backtest/backtester.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace archneuronx {
namespace backtest {

Backtester::Backtester(BacktestConfig config) : config_(config) {
}

BacktestReport Backtester::run(const std::vector<data::OHLCVBar>& bars,
                               SignalFunction signal_fn) {
    BacktestReport report;
    report.initial_capital = config_.initial_capital;
    report.final_equity = config_.initial_capital;
    report.peak_equity = config_.initial_capital;
    
    double current_equity = config_.initial_capital;
    double current_position = 0.0;
    double entry_price = 0.0;
    int64_t entry_time = 0;
    float entry_confidence = 0.0f;
    std::string current_side = "NONE";
    
    // Calculate ATR for stop loss/take profit
    std::vector<double> atr_values = calculate_atr(bars, 14);
    
    for (size_t i = 50; i < bars.size(); ++i) { // Skip first 50 for indicators
        const auto& bar = bars[i];
        
        // Get signal
        float confidence = 0.0f;
        std::vector<data::OHLCVBar> recent_bars(bars.begin() + i - 50, bars.begin() + i);
        std::string signal = signal_fn(recent_bars, confidence);
        
        // Apply confidence filter
        if (confidence < config_.min_confidence) {
            signal = "HOLD";
        }
        
        // Position management
        double atr = (i < atr_values.size()) ? atr_values[i] : 0.0;
        double stop_loss = 0.0;
        double take_profit = 0.0;
        
        if (current_position != 0.0) {
            // Calculate stop loss and take profit
            double stop_distance = atr * config_.stop_loss_atr_mult;
            double profit_distance = stop_distance * config_.take_profit_rr;
            
            if (current_side == "LONG") {
                stop_loss = entry_price - stop_distance;
                take_profit = entry_price + profit_distance;
            } else if (current_side == "SHORT") {
                stop_loss = entry_price + stop_distance;
                take_profit = entry_price - profit_distance;
            }
        }
        
        // Check exit conditions
        bool should_exit = false;
        std::string exit_reason = "end_of_data";
        
        if (current_position != 0.0) {
            // Stop loss hit
            if ((current_side == "LONG" && bar.low <= stop_loss) ||
                (current_side == "SHORT" && bar.high >= stop_loss)) {
                should_exit = true;
                exit_reason = "stop_loss";
            }
            // Take profit hit
            else if ((current_side == "LONG" && bar.high >= take_profit) ||
                     (current_side == "SHORT" && bar.low <= take_profit)) {
                should_exit = true;
                exit_reason = "take_profit";
            }
            // Reversal signal
            else if ((current_side == "LONG" && signal == "SELL") ||
                     (current_side == "SHORT" && signal == "BUY")) {
                should_exit = true;
                exit_reason = "signal";
            }
        }
        
        // Execute exit
        if (should_exit) {
            Trade trade;
            trade.symbol = bars[0].symbol;
            trade.side = current_side;
            trade.entry_price = entry_price;
            trade.exit_price = (current_side == "LONG") ? bar.close : bar.close;
            trade.quantity = std::abs(current_position);
            trade.entry_timestamp_ms = entry_time;
            trade.exit_timestamp_ms = bar.timestamp;
            trade.hold_duration_ms = bar.timestamp - entry_time;
            trade.entry_confidence = entry_confidence;
            trade.exit_reason = exit_reason;
            
            // Apply slippage
            trade.exit_price = calculate_slippage(trade.exit_price, trade.side);
            
            // Calculate P&L
            if (current_side == "LONG") {
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity;
            } else {
                trade.pnl = (trade.entry_price - trade.exit_price) * trade.quantity;
            }
            
            // Apply commission
            trade.commission = apply_commission(trade.entry_price * trade.quantity) +
                            apply_commission(trade.exit_price * trade.quantity);
            trade.pnl -= trade.commission;
            
            trade.pnl_pct = (trade.pnl / (trade.entry_price * trade.quantity)) * 100.0;
            
            current_equity += trade.pnl;
            current_position = 0.0;
            current_side = "NONE";
            
            report.trades.push_back(trade);
        }
        
        // Execute entry
        if (current_position == 0.0 && (signal == "BUY" || signal == "SELL")) {
            double position_size = (current_equity * config_.max_position_pct) / bar.close;
            
            if (signal == "BUY" || (config_.allow_short && signal == "SELL")) {
                entry_price = calculate_slippage(bar.close, signal);
                entry_time = bar.timestamp;
                entry_confidence = confidence;
                current_position = (signal == "BUY") ? position_size : -position_size;
                current_side = signal;
            }
        }
        
        // Update equity curve
        double unrealized_pnl = 0.0;
        if (current_position != 0.0) {
            if (current_side == "LONG") {
                unrealized_pnl = (bar.close - entry_price) * std::abs(current_position);
            } else {
                unrealized_pnl = (entry_price - bar.close) * std::abs(current_position);
            }
        }
        
        double total_equity = current_equity + unrealized_pnl;
        report.equity_curve.push_back({bar.timestamp, total_equity});
        report.peak_equity = std::max(report.peak_equity, total_equity);
    }
    
    // Calculate final metrics
    report.final_equity = current_equity;
    report.total_return_pct = ((current_equity - config_.initial_capital) / config_.initial_capital) * 100.0;
    
    calculate_metrics(report, bars);
    
    return report;
}

std::vector<BacktestReport> Backtester::walk_forward(
    const std::vector<data::OHLCVBar>& bars,
    SignalFunction signal_fn,
    int train_window,
    int test_window,
    int step_size) {
    
    std::vector<BacktestReport> reports;
    
    int total_bars = static_cast<int>(bars.size());
    int current_start = train_window;
    
    while (current_start + test_window < total_bars) {
        std::cout << "Walk-forward window: " << current_start 
                  << " to " << current_start + test_window << std::endl;
        
        // Extract test window
        std::vector<data::OHLCVBar> test_bars(
            bars.begin() + current_start,
            bars.begin() + current_start + test_window
        );
        
        // Run backtest on this window
        BacktestReport window_report = run(test_bars, signal_fn);
        reports.push_back(window_report);
        
        // Move to next window
        current_start += step_size;
    }
    
    return reports;
}

std::vector<double> Backtester::calculate_atr(
    const std::vector<data::OHLCVBar>& bars, int period) {
    
    std::vector<double> atr_values(bars.size(), 0.0);
    
    if (bars.size() < period + 1) {
        return atr_values;
    }
    
    // Calculate True Range for each bar
    std::vector<double> true_ranges;
    
    for (size_t i = 1; i < bars.size(); ++i) {
        double high_low = bars[i].high - bars[i].low;
        double high_close_prev = std::abs(bars[i].high - bars[i-1].close);
        double low_close_prev = std::abs(bars[i].low - bars[i-1].close);
        
        double tr = std::max({high_low, high_close_prev, low_close_prev});
        true_ranges.push_back(tr);
    }
    
    // Calculate ATR using exponential moving average
    double ema = true_ranges[0];
    atr_values[period] = ema;
    
    for (size_t i = period + 1; i < bars.size(); ++i) {
        double alpha = 2.0 / (period + 1);
        ema = alpha * true_ranges[i-1] + (1 - alpha) * ema;
        atr_values[i] = ema;
    }
    
    return atr_values;
}

void Backtester::calculate_metrics(BacktestReport& report, 
                                  const std::vector<data::OHLCVBar>& bars) const {
    if (report.trades.empty()) {
        return;
    }
    
    // Basic trade statistics
    report.total_trades = static_cast<int>(report.trades.size());
    
    for (const auto& trade : report.trades) {
        if (trade.pnl > 0) {
            report.winning_trades++;
        } else {
            report.losing_trades++;
        }
    }
    
    report.win_rate = static_cast<double>(report.winning_trades) / report.total_trades;
    
    // Calculate returns statistics
    std::vector<double> wins, losses;
    double total_pnl = 0.0;
    double total_commission = 0.0;
    double total_hold_time = 0.0;
    
    for (const auto& trade : report.trades) {
        total_pnl += trade.pnl;
        total_commission += trade.commission;
        total_hold_time += trade.hold_duration_ms;
        
        if (trade.pnl > 0) {
            wins.push_back(trade.pnl_pct);
        } else {
            losses.push_back(trade.pnl_pct);
        }
    }
    
    if (!wins.empty()) {
        report.avg_win_pct = std::accumulate(wins.begin(), wins.end(), 0.0) / wins.size();
        report.largest_win_pct = *std::max_element(wins.begin(), wins.end());
    }
    
    if (!losses.empty()) {
        report.avg_loss_pct = std::accumulate(losses.begin(), losses.end(), 0.0) / losses.size();
        report.largest_loss_pct = *std::min_element(losses.begin(), losses.end());
    }
    
    // Profit factor and expectancy
    double gross_profit = std::accumulate(wins.begin(), wins.end(), 0.0);
    double gross_loss = std::accumulate(losses.begin(), losses.end(), 0.0);
    
    report.profit_factor = std::abs(gross_profit / gross_loss);
    report.expectancy = total_pnl / report.total_trades;
    report.avg_hold_hours = (total_hold_time / report.total_trades) / (1000.0 * 60.0 * 60.0);
    
    // Calculate drawdown
    double peak = report.initial_capital;
    double max_drawdown = 0.0;
    double current_drawdown = 0.0;
    
    for (const auto& [timestamp, equity] : report.equity_curve) {
        if (equity > peak) {
            peak = equity;
            current_drawdown = 0.0;
        } else {
            current_drawdown = (peak - equity) / peak * 100.0;
            max_drawdown = std::max(max_drawdown, current_drawdown);
        }
    }
    
    report.max_drawdown_pct = max_drawdown;
    
    // Calculate Sharpe ratio (simplified)
    if (report.equity_curve.size() > 1) {
        std::vector<double> returns;
        for (size_t i = 1; i < report.equity_curve.size(); ++i) {
            double ret = (report.equity_curve[i].second - report.equity_curve[i-1].second) / 
                        report.equity_curve[i-1].second;
            returns.push_back(ret);
        }
        
        double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
        double variance = 0.0;
        for (double ret : returns) {
            variance += (ret - mean_return) * (ret - mean_return);
        }
        variance /= returns.size();
        double std_dev = std::sqrt(variance);
        
        report.sharpe_ratio = (std_dev > 0) ? (mean_return / std_dev) * std::sqrt(252.0) : 0.0;
    }
    
    // Calculate Calmar ratio
    report.calmar_ratio = (report.max_drawdown_pct > 0) ? 
                       (report.total_return_pct / report.max_drawdown_pct) : 0.0;
    
    // Calculate buy & hold return (benchmark)
    if (!bars.empty()) {
        double start_price = bars[0].close;
        double end_price = bars.back().close;
        report.buy_hold_return_pct = ((end_price - start_price) / start_price) * 100.0;
        report.alpha = report.total_return_pct - report.buy_hold_return_pct;
    }
    
    // Generate monthly returns
    generate_monthly_returns(report, bars);
}

double Backtester::calculate_slippage(double price, const std::string& side) const {
    if (side == "BUY") {
        return price * (1.0 + config_.slippage_pct);
    } else {
        return price * (1.0 - config_.slippage_pct);
    }
}

double Backtester::apply_commission(double value) const {
    return value * config_.commission_pct;
}

void Backtester::generate_monthly_returns(BacktestReport& report, 
                                         const std::vector<data::OHLCVBar>& bars) const {
    if (report.equity_curve.empty()) {
        return;
    }
    
    // Group equity points by month
    std::map<std::string, double> month_end_equity;
    
    for (const auto& [timestamp, equity] : report.equity_curve) {
        std::time_t tt = timestamp / 1000;
        std::tm* ptm = std::localtime(&tt);
        
        std::ostringstream oss;
        oss << std::put_time(ptm, "%Y-%m");
        std::string month_key = oss.str();
        
        month_end_equity[month_key] = equity; // Last equity of each month
    }
    
    // Calculate monthly returns
    double prev_equity = report.initial_capital;
    for (const auto& [month, equity] : month_end_equity) {
        double monthly_return = ((equity - prev_equity) / prev_equity) * 100.0;
        report.monthly_returns.push_back({month, monthly_return});
        prev_equity = equity;
    }
}

std::string Backtester::generate_html_report(const BacktestReport& report,
                                          const std::string& output_path) const {
    std::ostringstream html;
    
    html << R"(
<!DOCTYPE html>
<html>
<head>
    <title>ArchNeuronX Backtest Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .metric-label { color: #7f8c8d; margin-top: 5px; }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>ArchNeuronX Backtest Report</h1>
        <p>Generated: )" << std::put_time(std::localtime(&std::time_t{}), "%Y-%m-%d %H:%M:%S") << R"(</p>
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value )" << report.total_return_pct << R"(%</div>
            <div class="metric-label">Total Return</div>
        </div>
        <div class="metric-card">
            <div class="metric-value )" << report.sharpe_ratio << R"(</div>
            <div class="metric-label">Sharpe Ratio</div>
        </div>
        <div class="metric-card">
            <div class="metric-value )" << report.max_drawdown_pct << R"(%</div>
            <div class="metric-label">Max Drawdown</div>
        </div>
        <div class="metric-card">
            <div class="metric-value )" << report.win_rate * 100 << R"(%</div>
            <div class="metric-label">Win Rate</div>
        </div>
    </div>
    
    <h2>Trading Statistics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Trades</td><td>)" << report.total_trades << R"(</td></tr>
        <tr><td>Winning Trades</td><td>)" << report.winning_trades << R"(</td></tr>
        <tr><td>Losing Trades</td><td>)" << report.losing_trades << R"(</td></tr>
        <tr><td>Win Rate</td><td>)" << report.win_rate * 100 << R"(%</td></tr>
        <tr><td>Average Win</td><td>)" << report.avg_win_pct << R"(%</td></tr>
        <tr><td>Average Loss</td><td>)" << report.avg_loss_pct << R"(%</td></tr>
        <tr><td>Profit Factor</td><td>)" << report.profit_factor << R"(</td></tr>
        <tr><td>Expectancy</td><td>)" << report.expectancy << R"(</td></tr>
    </table>
    
    <h2>Risk Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Max Drawdown</td><td>)" << report.max_drawdown_pct << R"(%</td></tr>
        <tr><td>Sharpe Ratio</td><td>)" << report.sharpe_ratio << R"(</td></tr>
        <tr><td>Calmar Ratio</td><td>)" << report.calmar_ratio << R"(</td></tr>
        <tr><td>Alpha vs Buy & Hold</td><td>)" << report.alpha << R"(%</td></tr>
    </table>
</body>
</html>)";
    
    std::string html_content = html.str();
    
    if (!output_path.empty()) {
        std::ofstream file(output_path);
        file << html_content;
        file.close();
        std::cout << "HTML report saved to: " << output_path << std::endl;
    }
    
    return html_content;
}

} // namespace backtest
} // namespace archneuronx
