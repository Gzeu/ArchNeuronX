#pragma once
// ============================================================
// ArchNeuronX v3 - Real-Time Market Data Feed System
// WebSocket integration with multiple exchanges
// Paper trading with real market data streams
// ============================================================

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <chrono>
#include <unordered_map>
#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/client.hpp>
#include <nlohmann/json.hpp>

namespace archneuronx {
namespace data {

/**
 * @brief Market tick data structure
 */
struct MarketTick {
    std::string symbol;              // Trading pair (e.g., "BTCUSDT")
    double price;                    // Current price
    double volume;                   // Trade volume
    double bid;                      // Best bid price
    double ask;                      // Best ask price
    double bid_size;                 // Best bid size
    double ask_size;                 // Best ask size
    std::chrono::system_clock::time_point timestamp; // Tick timestamp
    std::string exchange;            // Exchange identifier
    uint64_t trade_id;              // Unique trade ID
};

/**
 * @brief Order book level data
 */
struct OrderBookLevel {
    double price;
    double quantity;
    uint64_t orders_count;          // Number of orders at this level
};

/**
 * @brief Order book snapshot
 */
struct OrderBook {
    std::string symbol;
    std::vector<OrderBookLevel> bids; // Buy orders (sorted descending)
    std::vector<OrderBookLevel> asks; // Sell orders (sorted ascending)
    std::chrono::system_clock::time_point timestamp;
    std::string exchange;
    uint64_t sequence;              // Sequence number for ordering
};

/**
 * @brief Real-time feed configuration
 */
struct RealtimeFeedConfig {
    // Exchange settings
    std::vector<std::string> exchanges = {"binance", "coinbase", "kraken"};
    std::vector<std::string> symbols = {"BTCUSDT", "ETHUSDT", "ADAUSDT"};
    
    // Connection settings
    int reconnect_interval_ms = 5000;
    int max_reconnect_attempts = 10;
    int heartbeat_interval_ms = 30000;
    int connection_timeout_ms = 10000;
    
    // Data settings
    bool enable_trades = true;       // Trade data stream
    bool enable_orderbook = true;    // Order book updates
    bool enable_ticker = true;       // 24hr ticker data
    int orderbook_depth = 10;        // Number of levels to maintain
    int max_queue_size = 10000;      // Maximum queued messages
    
    // Paper trading settings
    bool enable_paper_trading = true;
    double paper_balance_usd = 10000.0;
    double paper_fee_rate = 0.001;   // 0.1% trading fee
    std::string paper_exchange = "binance";
    
    // Performance settings
    bool enable_compression = true;  // WebSocket compression
    int processing_threads = 2;      // Number of processing threads
    int batch_size = 100;            // Batch processing size
};

/**
 * @brief Paper trading position
 */
struct PaperPosition {
    std::string symbol;
    double quantity;                // Position size (positive = long, negative = short)
    double average_price;           // Average entry price
    double unrealized_pnl;         // Unrealized P&L
    double realized_pnl;            // Realized P&L
    std::chrono::system_clock::time_point entry_time;
    std::string exchange;
};

/**
 * @brief Paper trading order
 */
enum class OrderType {
    MARKET,
    LIMIT,
    STOP_LOSS,
    TAKE_PROFIT
};

enum class OrderSide {
    BUY,
    SELL
};

enum class OrderStatus {
    PENDING,
    FILLED,
    CANCELLED,
    REJECTED
};

struct PaperOrder {
    std::string order_id;
    std::string symbol;
    OrderType type;
    OrderSide side;
    double quantity;
    double price;                   // For limit orders
    OrderStatus status;
    std::chrono::system_clock::time_point created_time;
    std::chrono::system_clock::time_point filled_time;
    double filled_quantity;
    double average_fill_price;
};

/**
 * @brief Real-time market data feed client
 * 
 * Provides WebSocket connections to multiple exchanges for real-time
 * market data with paper trading capabilities.
 */
class RealtimeFeed {
public:
    using TickCallback = std::function<void(const MarketTick&)>;
    using OrderBookCallback = std::function<void(const OrderBook&)>;
    using ErrorCallback = std::function<void(const std::string&)>;

    explicit RealtimeFeed(const RealtimeFeedConfig& config = RealtimeFeedConfig{});
    ~RealtimeFeed();

    // Connection management
    bool connect();
    void disconnect();
    bool is_connected() const;
    
    // Subscription management
    bool subscribe_trades(const std::string& symbol, const std::string& exchange = "");
    bool subscribe_orderbook(const std::string& symbol, const std::string& exchange = "");
    bool subscribe_ticker(const std::string& symbol, const std::string& exchange = "");
    void unsubscribe_all();

    // Callback registration
    void set_tick_callback(TickCallback callback);
    void set_orderbook_callback(OrderBookCallback callback);
    void set_error_callback(ErrorCallback callback);

    // Data access
    std::vector<MarketTick> get_recent_ticks(const std::string& symbol, int max_count = 100) const;
    OrderBook get_current_orderbook(const std::string& symbol, const std::string& exchange = "") const;
    double get_current_price(const std::string& symbol, const std::string& exchange = "") const;

    // Paper trading functions
    bool enable_paper_trading(double initial_balance = 10000.0);
    void disable_paper_trading();
    bool is_paper_trading_enabled() const;
    
    std::string place_paper_order(const std::string& symbol, OrderSide side, 
                                  OrderType type, double quantity, double price = 0.0);
    bool cancel_paper_order(const std::string& order_id);
    std::vector<PaperOrder> get_paper_orders() const;
    std::vector<PaperPosition> get_paper_positions() const;
    
    double get_paper_balance() const;
    double get_paper_total_pnl() const;
    std::vector<std::string> get_paper_trade_history() const;

    // Performance monitoring
    struct FeedStats {
        uint64_t messages_received;
        uint64_t ticks_processed;
        uint64_t orderbook_updates;
        double messages_per_second;
        double latency_ms;
        std::chrono::system_clock::time_point last_update;
    };
    
    FeedStats get_feed_stats() const;
    void reset_stats();

    // Exchange-specific methods
    std::vector<std::string> get_supported_exchanges() const;
    std::vector<std::string> get_supported_symbols(const std::string& exchange = "") const;

private:
    RealtimeFeedConfig config_;
    
    // WebSocket clients for each exchange
    std::unordered_map<std::string, std::unique_ptr<websocketpp::client<websocketpp::config::asio_client>>> ws_clients_;
    
    // Data storage
    std::unordered_map<std::string, std::queue<MarketTick>> tick_queues_;
    std::unordered_map<std::string, OrderBook> orderbooks_;
    std::unordered_map<std::string, std::vector<MarketTick>> recent_ticks_;
    
    // Paper trading
    bool paper_trading_enabled_;
    double paper_balance_;
    std::unordered_map<std::string, PaperPosition> paper_positions_;
    std::unordered_map<std::string, PaperOrder> paper_orders_;
    std::vector<std::string> paper_trade_history_;
    uint64_t next_order_id_;
    
    // Threading
    std::vector<std::thread> processing_threads_;
    std::atomic<bool> running_;
    mutable std::mutex data_mutex_;
    mutable std::mutex paper_mutex_;
    
    // Callbacks
    TickCallback tick_callback_;
    OrderBookCallback orderbook_callback_;
    ErrorCallback error_callback_;
    
    // Statistics
    FeedStats stats_;
    
    // Internal methods
    bool initialize_exchange_client(const std::string& exchange);
    void connect_to_exchange(const std::string& exchange);
    void handle_websocket_message(const std::string& exchange, const std::string& message);
    void process_tick_data(const std::string& exchange, const nlohmann::json& data);
    void process_orderbook_data(const std::string& exchange, const nlohmann::json& data);
    void process_ticker_data(const std::string& exchange, const nlohmann::json& data);
    
    // Paper trading methods
    void process_paper_order(const PaperOrder& order, const MarketTick& tick);
    void update_paper_positions(const PaperOrder& order, const MarketTick& tick);
    double calculate_paper_fees(double amount) const;
    std::string generate_order_id() const;
    
    // Utility methods
    std::string get_exchange_websocket_url(const std::string& exchange) const;
    nlohmann::json create_subscription_message(const std::string& type, 
                                                const std::vector<std::string>& symbols) const;
    MarketTick parse_trade_data(const std::string& exchange, const nlohmann::json& data) const;
    OrderBook parse_orderbook_data(const std::string& exchange, const nlohmann::json& data) const;
    
    // Threading methods
    void processing_thread_func();
    void heartbeat_thread_func();
    void reconnect_thread_func();
    
    // Statistics
    void update_stats();
    double calculate_latency(const std::chrono::system_clock::time_point& timestamp) const;
};

/**
 * @brief RAII connection manager
 */
class FeedConnection {
public:
    explicit FeedConnection(RealtimeFeed& feed);
    ~FeedConnection();
    
    bool is_connected() const;

private:
    RealtimeFeed& feed_;
    bool connected_;
};

/**
 * @brief Paper trading manager
 */
class PaperTradingManager {
public:
    explicit PaperTradingManager(RealtimeFeed& feed);
    
    // Advanced paper trading features
    std::string place_bracket_order(const std::string& symbol, OrderSide side,
                                    double quantity, double entry_price,
                                    double stop_loss, double take_profit);
    
    std::string place_oco_order(const std::string& symbol, OrderSide side,
                                double quantity, double price1, double price2);
    
    void set_trailing_stop(const std::string& order_id, double trailing_percent);
    void set_position_size_limits(const std::string& symbol, double max_size);
    
    // Risk management
    void set_max_drawdown_limit(double max_drawdown_percent);
    void set_daily_loss_limit(double daily_loss_limit);
    bool check_risk_limits(const PaperOrder& order) const;
    
    // Performance analytics
    struct TradingStats {
        int total_trades;
        int winning_trades;
        int losing_trades;
        double win_rate;
        double total_pnl;
        double max_drawdown;
        double sharpe_ratio;
        double average_trade_duration;
        std::chrono::system_clock::time_point start_time;
    };
    
    TradingStats get_trading_stats() const;
    void reset_trading_stats();

private:
    RealtimeFeed& feed_;
    std::unordered_map<std::string, double> position_size_limits_;
    double max_drawdown_limit_;
    double daily_loss_limit_;
    double starting_balance_;
    double max_balance_;
    TradingStats stats_;
    
    void update_trading_stats(const PaperOrder& order);
    double calculate_drawdown() const;
};

} // namespace data
} // namespace archneuronx
