/**
 * @file trading_engine.hpp
 * @brief Core trading engine interface for ArchNeuronX
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "core/logger.hpp"

namespace ArchNeuronX {
namespace Trading {

/**
 * @enum SignalType
 * @brief Types of trading signals
 */
enum class SignalType {
    BUY,    ///< Buy signal
    SELL,   ///< Sell signal
    HOLD    ///< Hold signal
};

/**
 * @enum OrderStatus
 * @brief Status of trading orders
 */
enum class OrderStatus {
    PENDING,    ///< Order is pending execution
    FILLED,     ///< Order has been filled
    CANCELLED,  ///< Order was cancelled
    REJECTED    ///< Order was rejected
};

/**
 * @struct TradingSignal
 * @brief Container for trading signal information
 */
struct TradingSignal {
    std::string symbol;                                    ///< Trading symbol (e.g., "BTCUSDT")
    SignalType signal_type;                               ///< Type of signal
    double confidence;                                    ///< Confidence score (0.0 - 1.0)
    double price_target;                                  ///< Target price
    double stop_loss = 0.0;                              ///< Stop loss price
    double take_profit = 0.0;                            ///< Take profit price
    double position_size = 0.0;                          ///< Recommended position size
    std::string explanation;                              ///< Human-readable explanation
    std::chrono::system_clock::time_point timestamp;     ///< Signal generation time
    std::map<std::string, double> technical_indicators;  ///< Supporting technical analysis
    
    /**
     * @brief Convert signal to JSON string
     * @return JSON representation of signal
     */
    std::string toJson() const;
    
    /**
     * @brief Create signal from JSON string
     * @param json_str JSON string
     * @return TradingSignal object
     */
    static TradingSignal fromJson(const std::string& json_str);
};

/**
 * @struct Position
 * @brief Represents an open trading position
 */
struct Position {
    std::string symbol;
    double size;                                          ///< Position size (positive for long, negative for short)
    double entry_price;
    double current_price;
    double unrealized_pnl;
    double realized_pnl = 0.0;
    std::chrono::system_clock::time_point entry_time;
    std::chrono::system_clock::time_point last_update;
    
    /**
     * @brief Calculate current P&L
     * @param current_market_price Current market price
     * @return Unrealized P&L
     */
    double calculatePnL(double current_market_price) const;
    
    /**
     * @brief Check if position is long
     * @return True if long position
     */
    bool isLong() const { return size > 0; }
    
    /**
     * @brief Check if position is short
     * @return True if short position
     */
    bool isShort() const { return size < 0; }
};

/**
 * @struct RiskParameters
 * @brief Risk management parameters
 */
struct RiskParameters {
    double max_position_size = 0.1;          ///< Maximum position size as fraction of portfolio
    double max_daily_loss = 0.05;            ///< Maximum daily loss as fraction of portfolio
    double max_drawdown = 0.2;               ///< Maximum allowed drawdown
    double min_confidence_threshold = 0.7;   ///< Minimum confidence for signal execution
    bool use_stop_loss = true;               ///< Enable stop loss orders
    bool use_take_profit = true;             ///< Enable take profit orders
    double correlation_limit = 0.8;          ///< Maximum correlation between positions
    
    /**
     * @brief Validate risk parameters
     * @throws std::invalid_argument if parameters are invalid
     */
    void validate() const;
};

/**
 * @class TradingEngine
 * @brief Core trading engine for signal processing and execution
 */
class TradingEngine {
public:
    /**
     * @brief Constructor
     * @param name Engine name for identification
     */
    explicit TradingEngine(const std::string& name);
    
    /**
     * @brief Destructor
     */
    ~TradingEngine();
    
    /**
     * @brief Initialize trading engine
     * @param initial_capital Initial trading capital
     * @param risk_params Risk management parameters
     */
    void initialize(double initial_capital, const RiskParameters& risk_params);
    
    /**
     * @brief Start the trading engine
     */
    void start();
    
    /**
     * @brief Stop the trading engine
     */
    void stop();
    
    /**
     * @brief Process a trading signal
     * @param signal Trading signal to process
     * @return True if signal was accepted and processed
     */
    bool processSignal(const TradingSignal& signal);
    
    /**
     * @brief Update market price for a symbol
     * @param symbol Trading symbol
     * @param price Current market price
     */
    void updatePrice(const std::string& symbol, double price);
    
    /**
     * @brief Get current portfolio value
     * @return Total portfolio value
     */
    double getPortfolioValue() const;
    
    /**
     * @brief Get current positions
     * @return Map of symbol to position
     */
    std::map<std::string, Position> getPositions() const;
    
    /**
     * @brief Get trading statistics
     * @return Map of statistic name to value
     */
    std::map<std::string, double> getStatistics() const;
    
    /**
     * @brief Export trading history to CSV
     * @param filename Output filename
     */
    void exportHistory(const std::string& filename) const;
    
    /**
     * @brief Check if engine is running
     * @return True if engine is active
     */
    bool isRunning() const { return running_.load(); }
    
    // Getters
    const std::string& getName() const { return name_; }
    double getInitialCapital() const { return initial_capital_; }
    double getCurrentCapital() const { return current_capital_; }
    const RiskParameters& getRiskParameters() const { return risk_params_; }
    
private:
    /**
     * @brief Main engine loop
     */
    void engineLoop();
    
    /**
     * @brief Validate signal against risk parameters
     * @param signal Signal to validate
     * @return True if signal passes risk checks
     */
    bool validateSignal(const TradingSignal& signal) const;
    
    /**
     * @brief Calculate position size based on risk management
     * @param signal Trading signal
     * @return Recommended position size
     */
    double calculatePositionSize(const TradingSignal& signal) const;
    
    /**
     * @brief Execute trading signal
     * @param signal Validated trading signal
     * @return True if execution was successful
     */
    bool executeSignal(const TradingSignal& signal);
    
    /**
     * @brief Update all position P&L
     */
    void updatePositions();
    
    /**
     * @brief Check and execute stop loss/take profit orders
     */
    void checkStopOrders();
    
    /**
     * @brief Calculate portfolio statistics
     */
    void calculateStatistics();
    
    // Member variables
    std::string name_;
    bool initialized_ = false;
    std::atomic<bool> running_{false};
    
    double initial_capital_ = 0.0;
    double current_capital_ = 0.0;
    RiskParameters risk_params_;
    
    std::map<std::string, Position> positions_;
    std::map<std::string, double> current_prices_;
    std::vector<TradingSignal> signal_history_;
    
    // Statistics
    mutable std::map<std::string, double> statistics_;
    double total_pnl_ = 0.0;
    double max_drawdown_ = 0.0;
    double peak_portfolio_value_ = 0.0;
    int total_trades_ = 0;
    int winning_trades_ = 0;
    
    // Threading
    std::unique_ptr<std::thread> engine_thread_;
    mutable std::mutex engine_mutex_;
    std::condition_variable engine_cv_;
    std::vector<TradingSignal> pending_signals_;
};

/**
 * @brief Convert SignalType to string
 * @param type Signal type
 * @return String representation
 */
std::string signalTypeToString(SignalType type);

/**
 * @brief Convert string to SignalType
 * @param type_str String representation
 * @return Signal type
 */
SignalType stringToSignalType(const std::string& type_str);

} // namespace Trading
} // namespace ArchNeuronX