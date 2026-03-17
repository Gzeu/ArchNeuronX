/**
 * @file metatrader_provider.hpp
 * @brief MetaTrader 5 API data provider for forex and CFDs
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include "data_provider.hpp"
#include <curl/curl.h>
#include <json/json.h>
#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/client.hpp>
#include <openssl/hmac.h>

namespace ArchNeuronX {
namespace Data {

/**
 * @struct MetaTraderEndpoints
 * @brief MetaTrader 5 API endpoints
 */
struct MetaTraderEndpoints {
    static constexpr const char* REST_BASE = "https://mt5webapi.com/api";
    static constexpr const char* WS_BASE = "wss://mt5webapi.com/api";
    
    // REST endpoints
    static constexpr const char* AUTH = "/auth";
    static constexpr const char* ACCOUNTS = "/account";
    static constexpr const char* SYMBOLS = "/symbols";
    static constexpr const char* TICK = "/tick";
    static constexpr const char* CANDLES = "/candles";
    static constexpr const char* ORDER_SEND = "/order_send";
    static constexpr const char* ORDER_CLOSE = "/order_close";
    static constexpr const char|* TRADES = "/trades";
    static constexpr const char|* POSITIONS = "/positions";
    static constexpr const char|* HISTORY = "/history";
};

/**
 * @struct MT5Symbol
 * @brief MetaTrader 5 symbol information
 */
struct MT5Symbol {
    std::string symbol;
    std::string description;
    std::string group;
    std::string base_currency;
    std::string quote_currency;
    double tick_size;
    double contract_size;
    double point;
    int digits;
    double swap_long;
    double swap_short;
    double margin_initial;
    double margin_maintenance;
    bool visible;
    bool trade_mode;
    
    std::string to_json() const;
    static MT5Symbol from_json(const std::string& json_str);
};

/**
 * @struct MT5Tick
 * @brief MetaTrader 5 tick data
 */
struct MT5Tick {
    std::string symbol;
    double bid;
    double ask;
    double last;
    double volume;
    std::chrono::system_clock::time_point timestamp;
    
    std::string to_json() const;
    static MT5Tick from_json(const std::string& json_str);
};

/**
 * @class MetaTraderProvider
 * @brief MetaTrader 5 data provider for forex and CFDs
 */
class MetaTraderProvider : public DataProvider {
public:
    /**
     * @brief Constructor
     * @param config Provider configuration
     */
    explicit MetaTraderProvider(const DataProviderConfig& config);
    
    /**
     * @brief Destructor
     */
    ~MetaTraderProvider() override;
    
    // DataProvider interface implementation
    bool connect() override;
    void disconnect() override;
    bool is_connected() const override;
    ConnectionStatus get_status() const override;
    
    std::future<std::vector<OHLCV>> get_historical_data(
        const std::string& symbol,
        const std::string& timeframe,
        const std::chrono::system_clock::time_point& start,
        const std::chrono::system_clock::time_point& end
    ) override;
    
    std::future<std::vector<TickData>> get_real_time_ticks(
        const std::string& symbol,
        std::function<void(const TickData&)> callback
    ) override;
    
    std::future<double> get_current_price(const std::string& symbol) override;
    std::future<OrderBook> get_order_book(const std::string& symbol, int depth = 100) override;
    
    // MetaTrader specific methods
    std::future<std::vector<MT5Symbol>> get_symbols();
    std::future<std::vector<MT5Tick>> get_recent_ticks(const std::string& symbol, int count = 10);
    std::future<std::string> get_account_info();
    std::future<std::vector<std::string>> get_positions();
    std::future<std::vector<std::string>> get_trade_history(int days = 7);

private:
    CURL* curl_;
    websocketpp::client<websocketpp::config::asio_tls_client> ws_client_;
    std::string api_key_;
    std::string user_id_;
    int account_id_;
    
    std::atomic<bool> connected_;
    std::atomic<ConnectionStatus> status_;
    mutable std::mutex connection_mutex_;
    
    // WebSocket connections
    std::map<std::string, websocketpp::connection_hdl> ws_connections_;
    std::thread ws_thread_;
    
    // Authentication
    std::string auth_token_;
    std::chrono::steady_clock::time_point token_expiry_;
    
    // Rate limiting
    std::chrono::steady_clock::time_point last_request_time_;
    std::atomic<int> requests_per_second_;
    static constexpr int MAX_REQUESTS_PER_SECOND = 10;
    
    // Symbol mapping
    std::map<std::string, std::string> symbol_to_mt5_;
    std::map<std::string, std::string> mt5_to_symbol_;
    
    // Private methods
    bool authenticate();
    bool test_rest_connection();
    std::string make_rest_request(const std::string& endpoint, 
                               const std::string& method = "GET",
                               const std::string& body = "");
    void enforce_rate_limit();
    
    // WebSocket methods
    void setup_websocket_connection();
    void subscribe_to_symbol(const std::string& symbol);
    void subscribe_to_ticks(const std::string& symbol);
    void subscribe_to_candles(const std::string& symbol, const std::string& timeframe);
    
    // Data parsing methods
    std::vector<OHLCV> parse_candles_response(const std::string& response,
                                           const std::string& symbol);
    OrderBook parse_tick_response(const std::string& response,
                              const std::string& symbol);
    std::vector<MT5Symbol> parse_symbols_response(const std::string& response);
    
    // Utility methods
    std::string convert_symbol_to_mt5(const std::string& symbol);
    std::string convert_mt5_to_symbol(const std::string& mt5_symbol);
    std::string convert_timeframe_to_mt5(const std::string& timeframe);
    std::chrono::system_clock::time_point parse_mt5_timestamp(
        const std::string& timestamp);
    
    // Symbol mapping initialization
    void initialize_symbol_mapping();
    
    // Static callback for CURL
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp);
    
    // WebSocket message handlers
    void handle_tick_message(const std::string& message,
                           std::function<void(const TickData&)> callback);
    void handle_candle_message(const std::string& message);
    void handle_book_message(const std::string& message);
};

} // namespace Data
} // namespace ArchNeuronX
