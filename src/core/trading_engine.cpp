#include "core/trading_engine.hpp"
#include <spdlog/spdlog.h>
#include <chrono>
#include <thread>

namespace arch {

TradingEngine::TradingEngine(std::shared_ptr<DataManager> data_mgr,
                           std::shared_ptr<ModelManager> model_mgr,
                           std::shared_ptr<RiskManager> risk_mgr,
                           std::shared_ptr<OrderManager> order_mgr,
                           std::shared_ptr<PortfolioManager> portfolio_mgr)
    : data_manager_(data_mgr)
    , model_manager_(model_mgr)
    , risk_manager_(risk_mgr)
    , order_manager_(order_mgr)
    , portfolio_manager_(portfolio_mgr)
    , running_(false) {
    spdlog::info("TradingEngine initialized");
}

TradingEngine::~TradingEngine() {
    stop();
}

void TradingEngine::start() {
    if (running_) {
        spdlog::warn("TradingEngine already running");
        return;
    }

    running_ = true;
    spdlog::info("Starting TradingEngine");

    // Start the main trading loop in a separate thread
    trading_thread_ = std::thread(&TradingEngine::tradingLoop, this);
}

void TradingEngine::stop() {
    if (!running_) return;

    running_ = false;
    if (trading_thread_.joinable()) {
        trading_thread_.join();
    }
    spdlog::info("TradingEngine stopped");
}

void TradingEngine::tradingLoop() {
    spdlog::info("Trading loop started");

    while (running_) {
        try {
            // 1. Fetch latest market data
            auto market_data = data_manager_->getLatestData();

            // 2. Generate signals from models
            auto signals = model_manager_->generateSignals(market_data);

            // 3. Apply risk management
            auto filtered_signals = risk_manager_->filterSignals(signals);

            // 4. Execute orders
            for (const auto& signal : filtered_signals) {
                if (order_manager_->placeOrder(signal)) {
                    spdlog::info("Order placed for signal: {}", signal.symbol);
                }
            }

            // 5. Update portfolio
            portfolio_manager_->updatePortfolio();

            // Sleep for next iteration
            std::this_thread::sleep_for(std::chrono::seconds(1));

        } catch (const std::exception& e) {
            spdlog::error("Error in trading loop: {}", e.what());
            std::this_thread::sleep_for(std::chrono::seconds(5)); // Back off on error
        }
    }

    spdlog::info("Trading loop ended");
}

bool TradingEngine::isRunning() const {
    return running_;
}

} // namespace arch
