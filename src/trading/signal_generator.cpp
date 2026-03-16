#include "trading/signal_generator.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>

namespace archneuronx {
namespace trading {

SignalGenerator::SignalGenerator(std::shared_ptr<models::IModel> model,
                                  std::shared_ptr<risk::RiskManager> risk_manager,
                                  const SignalConfig& config)
    : model_(std::move(model)),
      risk_manager_(std::move(risk_manager)),
      config_(config) {
    std::cout << "[SignalGenerator] Initialized with confidence threshold="
              << config.min_confidence << std::endl;
}

TradingSignal SignalGenerator::generate(const MarketSnapshot& snapshot) {
    auto raw_signal = model_->predict(snapshot.features);

    TradingSignal signal;
    signal.timestamp = std::chrono::system_clock::now();
    signal.symbol = snapshot.symbol;
    signal.price = snapshot.close_price;

    // Apply confidence threshold filter
    if (raw_signal.confidence < config_.min_confidence) {
        signal.action = SignalAction::HOLD;
        signal.confidence = raw_signal.confidence;
        signal.reason = "Confidence below threshold";
        return signal;
    }

    signal.action = raw_signal.action;
    signal.confidence = raw_signal.confidence;

    // Calculate VaR-adjusted position size
    signal.position_size = risk_manager_->calculatePositionSize(
        snapshot.portfolio_value,
        signal.confidence,
        snapshot.volatility
    );

    // Apply circuit breaker check
    if (risk_manager_->checkCircuitBreaker(snapshot.current_drawdown_pct)) {
        signal.action = SignalAction::HOLD;
        signal.reason = "Circuit breaker active";
        signal.position_size = 0.0;
        return signal;
    }

    // Generate stop loss and take profit levels
    double atr = snapshot.atr;  // Average True Range
    if (signal.action == SignalAction::BUY) {
        signal.stop_loss = signal.price - (config_.stop_loss_atr_multiplier * atr);
        signal.take_profit = signal.price + (config_.take_profit_atr_multiplier * atr);
    } else if (signal.action == SignalAction::SELL) {
        signal.stop_loss = signal.price + (config_.stop_loss_atr_multiplier * atr);
        signal.take_profit = signal.price - (config_.take_profit_atr_multiplier * atr);
    }

    signal.reason = "Model confidence: " + std::to_string(signal.confidence);
    return signal;
}

std::vector<TradingSignal> SignalGenerator::generateBatch(
    const std::vector<MarketSnapshot>& snapshots) {
    std::vector<TradingSignal> signals;
    signals.reserve(snapshots.size());

    for (const auto& snapshot : snapshots) {
        signals.push_back(generate(snapshot));
    }
    return signals;
}

void SignalGenerator::updateModel(std::shared_ptr<models::IModel> new_model) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    model_ = std::move(new_model);
    std::cout << "[SignalGenerator] Model updated" << std::endl;
}

} // namespace trading
} // namespace archneuronx
