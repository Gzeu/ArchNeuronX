#pragma once

#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <functional>
#include <map>
#include <queue>

#include "core/inference_engine.hpp"
#include "models/quantum_trading_signals.hpp"
#include "agents/quantum_trading_agent.hpp"
#include "ml/huggingface_integration.hpp"

namespace trading {
namespace live {

// Market data structures
struct MarketData {
    std::string symbol;
    double price;
    double volume;
    std::chrono::system_clock::time_point timestamp;
    double bid;
    double ask;
    double spread;
    
    // Additional market metrics
    double volatility;
    double momentum;
    double rsi;
    double macd;
    double bollinger_upper;
    double bollinger_lower;
    
    MarketData() : price(0.0), volume(0.0), bid(0.0), ask(0.0), spread(0.0),
                   volatility(0.0), momentum(0.0), rsi(50.0), macd(0.0),
                   bollinger_upper(0.0), bollinger_lower(0.0) {}
};

struct Order {
    enum class Type { MARKET, LIMIT, STOP, STOP_LIMIT };
    enum class Side { BUY, SELL };
    enum class Status { PENDING, FILLED, CANCELLED, REJECTED };
    
    std::string id;
    std::string symbol;
    Type type;
    Side side;
    double quantity;
    double price;
    double stop_price;
    Status status;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point filled_at;
    double filled_quantity;
    double filled_price;
    double commission;
    
    Order() : type(Type::MARKET), side(Side::BUY), quantity(0.0), price(0.0),
              stop_price(0.0), status(Status::PENDING), filled_quantity(0.0),
              filled_price(0.0), commission(0.0) {}
};

struct Position {
    std::string symbol;
    double quantity;
    double avg_price;
    double unrealized_pnl;
    double realized_pnl;
    std::chrono::system_clock::time_point opened_at;
    std::vector<Order> orders;
    
    Position() : quantity(0.0), avg_price(0.0), unrealized_pnl(0.0),
                realized_pnl(0.0) {}
};

struct Portfolio {
    double total_value;
    double cash_balance;
    double margin_used;
    double margin_available;
    double total_pnl;
    double daily_pnl;
    std::map<std::string, Position> positions;
    
    Portfolio() : total_value(0.0), cash_balance(0.0), margin_used(0.0),
                  margin_available(0.0), total_pnl(0.0), daily_pnl(0.0) {}
};

struct RiskMetrics {
    double max_drawdown;
    double sharpe_ratio;
    double sortino_ratio;
    double var_95;
    double beta;
    double alpha;
    double win_rate;
    double profit_factor;
    double avg_win;
    double avg_loss;
    int total_trades;
    int winning_trades;
    int losing_trades;
    
    RiskMetrics() : max_drawdown(0.0), sharpe_ratio(0.0), sortino_ratio(0.0),
                    var_95(0.0), beta(0.0), alpha(0.0), win_rate(0.0),
                    profit_factor(0.0), avg_win(0.0), avg_loss(0.0),
                    total_trades(0), winning_trades(0), losing_trades(0) {}
};

// Exchange interface
class ExchangeInterface {
public:
    virtual ~ExchangeInterface() = default;
    
    // Market data
    virtual bool connect() = 0;
    virtual bool disconnect() = 0;
    virtual MarketData get_market_data(const std::string& symbol) = 0;
    virtual std::vector<std::string> get_available_symbols() = 0;
    
    // Trading operations
    virtual std::string place_order(const Order& order) = 0;
    virtual bool cancel_order(const std::string& order_id) = 0;
    virtual Order get_order_status(const std::string& order_id) = 0;
    virtual std::vector<Order> get_open_orders() = 0;
    
    // Portfolio management
    virtual Portfolio get_portfolio() = 0;
    virtual std::vector<Position> get_positions() = 0;
    
    // Account information
    virtual double get_account_balance() = 0;
    virtual double get_margin_used() = 0;
    virtual double get_margin_available() = 0;
};

// Binance exchange implementation
class BinanceExchange : public ExchangeInterface {
private:
    std::string api_key_;
    std::string api_secret_;
    std::string base_url_;
    bool connected_;
    std::map<std::string, MarketData> market_data_cache_;
    std::mutex cache_mutex_;
    
public:
    BinanceExchange(const std::string& api_key, const std::string& api_secret);
    
    bool connect() override;
    bool disconnect() override;
    MarketData get_market_data(const std::string& symbol) override;
    std::vector<std::string> get_available_symbols() override;
    
    std::string place_order(const Order& order) override;
    bool cancel_order(const std::string& order_id) override;
    Order get_order_status(const std::string& order_id) override;
    std::vector<Order> get_open_orders() override;
    
    Portfolio get_portfolio() override;
    std::vector<Position> get_positions() override;
    
    double get_account_balance() override;
    double get_margin_used() override;
    double get_margin_available() override;
};

// Risk management system
class RiskManager {
private:
    double max_position_size_;
    double max_daily_loss_;
    double max_drawdown_;
    double leverage_limit_;
    std::map<std::string, double> position_limits_;
    
public:
    RiskManager(double max_position_size = 10000.0,
                double max_daily_loss = 1000.0,
                double max_drawdown = 0.1,
                double leverage_limit = 2.0);
    
    bool validate_order(const Order& order, const Portfolio& portfolio);
    bool check_position_limits(const std::string& symbol, double quantity);
    bool check_risk_limits(const Portfolio& portfolio, const RiskMetrics& metrics);
    double calculate_position_size(const std::string& symbol, double risk_per_trade);
    void update_risk_parameters(double max_position_size, double max_daily_loss,
                               double max_drawdown, double leverage_limit);
};

// Portfolio manager
class PortfolioManager {
private:
    Portfolio portfolio_;
    RiskManager risk_manager_;
    std::mutex portfolio_mutex_;
    
public:
    PortfolioManager(const RiskManager& risk_manager);
    
    void update_portfolio(const MarketData& market_data);
    void add_position(const std::string& symbol, double quantity, double price);
    void close_position(const std::string& symbol, double quantity, double price);
    Portfolio get_portfolio() const;
    RiskMetrics calculate_risk_metrics() const;
    double calculate_unrealized_pnl(const std::string& symbol, double current_price);
    void update_pnl(const std::string& symbol, double current_price);
};

// Live trading engine
class LiveTradingEngine {
private:
    // Core components
    std::unique_ptr<ExchangeInterface> exchange_;
    std::unique_ptr<models::QuantumTradingSignals> quantum_signals_;
    std::unique_ptr<agents::QuantumTradingAgent> trading_agent_;
    std::unique_ptr<ml::HuggingFaceIntegration> llm_integration_;
    std::unique_ptr<PortfolioManager> portfolio_manager_;
    
    // Trading state
    std::atomic<bool> is_running_;
    std::atomic<bool> is_connected_;
    std::thread trading_thread_;
    std::thread market_data_thread_;
    std::mutex trading_mutex_;
    
    // Market data
    std::map<std::string, MarketData> market_data_;
    std::queue<MarketData> market_data_queue_;
    std::mutex market_data_mutex_;
    std::condition_variable market_data_cv_;
    
    // Orders and positions
    std::map<std::string, Order> orders_;
    std::map<std::string, Position> positions_;
    
    // Configuration
    std::vector<std::string> trading_symbols_;
    std::chrono::milliseconds trading_interval_;
    double risk_per_trade_;
    double max_position_size_;
    
    // Performance tracking
    std::atomic<int> total_trades_;
    std::atomic<int> winning_trades_;
    std::atomic<double> total_pnl_;
    std::atomic<double> daily_pnl_;
    
    // Callbacks
    std::function<void(const Order&)> on_order_filled_;
    std::function<void(const MarketData&)> on_market_data_;
    std::function<void(const Portfolio&)> on_portfolio_update_;
    std::function<void(const std::string&)> on_error_;
    
public:
    LiveTradingEngine(std::unique_ptr<ExchangeInterface> exchange,
                     std::unique_ptr<models::QuantumTradingSignals> quantum_signals,
                     std::unique_ptr<agents::QuantumTradingAgent> trading_agent,
                     std::unique_ptr<ml::HuggingFaceIntegration> llm_integration);
    
    ~LiveTradingEngine();
    
    // Engine control
    bool start();
    bool stop();
    bool is_running() const { return is_running_.load(); }
    bool is_connected() const { return is_connected_.load(); }
    
    // Configuration
    void set_trading_symbols(const std::vector<std::string>& symbols);
    void set_trading_interval(std::chrono::milliseconds interval);
    void set_risk_parameters(double risk_per_trade, double max_position_size);
    
    // Trading operations
    std::string place_order(const Order& order);
    bool cancel_order(const std::string& order_id);
    Order get_order_status(const std::string& order_id);
    std::vector<Order> get_open_orders();
    
    // Market data
    MarketData get_market_data(const std::string& symbol);
    std::vector<std::string> get_available_symbols();
    
    // Portfolio and performance
    Portfolio get_portfolio();
    RiskMetrics get_risk_metrics();
    double get_total_pnl() const { return total_pnl_.load(); }
    double get_daily_pnl() const { return daily_pnl_.load(); }
    int get_total_trades() const { return total_trades_.load(); }
    int get_winning_trades() const { return winning_trades_.load(); }
    
    // Callbacks
    void set_order_filled_callback(std::function<void(const Order&)> callback);
    void set_market_data_callback(std::function<void(const MarketData&)> callback);
    void set_portfolio_update_callback(std::function<void(const Portfolio&)> callback);
    void set_error_callback(std::function<void(const std::string&)> callback);
    
private:
    // Trading loop
    void trading_loop();
    void market_data_loop();
    void process_market_data();
    void execute_trading_logic();
    
    // Signal generation
    int generate_quantum_signal(const MarketData& market_data);
    int generate_agent_signal(const MarketData& market_data);
    int generate_llm_signal(const MarketData& market_data);
    int combine_signals(int quantum_signal, int agent_signal, int llm_signal);
    
    // Order management
    void place_buy_order(const std::string& symbol, double quantity, double price);
    void place_sell_order(const std::string& symbol, double quantity, double price);
    void manage_positions();
    void update_portfolio();
    
    // Risk management
    bool check_risk_limits();
    void apply_stop_loss();
    void apply_take_profit();
    
    // Performance tracking
    void update_performance_metrics();
    void calculate_daily_pnl();
    void log_trade(const Order& order);
};

// Alert system
class AlertSystem {
private:
    std::vector<std::function<void(const std::string&, const std::string&)>> alert_callbacks_;
    std::mutex alert_mutex_;
    
public:
    void add_alert_callback(std::function<void(const std::string&, const std::string&)> callback);
    void send_alert(const std::string& level, const std::string& message);
    void send_trade_alert(const Order& order);
    void send_risk_alert(const std::string& message);
    void send_performance_alert(const std::string& message);
};

} // namespace live
} // namespace trading
