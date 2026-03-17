/**
 * @file kraken_provider.hpp
 * @brief Kraken API data provider for cryptocurrencies
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
 * @struct KrakenEndpoints
 * @brief Kraken API endpoints
 */
struct KrakenEndpoints {
    static constexpr const char* REST_BASE = "https://api.kraken.com";
    static constexpr const char* WS_PUBLIC = "wss://ws.kraken.com";
    static constexpr const char* WS_PRIVATE = "wss://ws-auth.kraken.com";
    
    // REST endpoints
    static constexpr const char* SERVER_TIME = "/0/public/Time";
    static constexpr const char* ASSETS = "/0/public/Assets";
    static constexpr const char* ASSET_PAIRS = "/0/public/AssetPairs";
    static constexpr const char* TICKER = "/0/public/Ticker";
    static constexpr const char* OHLC = "/0/public/OHLC";
    static constexpr const char* DEPTH = "/0/public/Depth";
    static constexpr const char* TRADES = "/0/public/Trades";
    static constexpr const char* SPREAD = "/0/public/Spread";
    
    // Private endpoints
    static constexpr const char* BALANCE = "/0/private/Balance";
    static constexpr const char* TRADE_BALANCE = "/0/private/TradeBalance";
    static constexpr const char* OPEN_ORDERS = "/0/private/OpenOrders";
    static constexpr const char* CLOSED_ORDERS = "/0/private/ClosedOrders";
    static constexpr const char* ORDERS_INFO = "/0/private/OrdersInfo";
    static constexpr const char* TRADE_HISTORY = "/0/private/TradesHistory";
    static constexpr const char|* ADD_ORDER = "/0/private/AddOrder";
    static constexpr const char* CANCEL_ORDER = "/0/private/CancelOrder";
};

/**
 * @struct KrakenAssetPair
 * @brief Kraken asset pair information
 */
struct KrakenAssetPair {
    std::string name;
    std::string altname;
    std::string base;
    std::string quote;
    std::string wsname;
    bool tradable;
    bool marginable;
    
    std::string to_json() const;
    static KrakenAssetPair from_json(const std::string& json_str);
};

/**
 * @class KrakenProvider
 * @brief Kraken cryptocurrency exchange data provider
 */
class KrakenProvider : public DataProvider {
public:
    /**
     * @brief Constructor
     * @param config Provider configuration
     */
    explicit KrakenProvider(const DataProviderConfig& config);
    
    /**
     * @brief Destructor
     */
    ~KrakenProvider() override;
    
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
    
    // Kraken specific methods
    std::future<std::vector<KrakenAssetPair>> get_asset_pairs();
    std::future<std::map<std::string, double>> get_tickers(
        const std::vector<std::string>& pairs);
    std::future<std::chrono::system_clock::time_point> get_server_time();
    std::future<std::vector<std::string>> get_assets();

private:
    CURL* curl_;
    websocketpp::client<websocketpp::config::asio_client> ws_client_;
    std::string api_key_;
    std::string api_secret_;
    
    std::atomic<bool> connected_;
    std::atomic<ConnectionStatus> status_;
    mutable std::mutex connection_mutex_;
    
    // WebSocket connections
    std::map<std::string, websocketpp::connection_hdl> ws_connections_;
    std::thread ws_thread_;
    
    // Rate limiting
    std::chrono::steady_clock::time_point last_request_time_;
    std::atomic<int> requests_per_second_;
    static constexpr int MAX_REQUESTS_PER_SECOND = 15;
    
    // Symbol mapping
    std::map<std::string, std::string> symbol_to_kraken_;
    std::map<std::string, std::string> kraken_to_symbol_;
    
    // Private methods
    bool test_rest_connection();
    std::string make_rest_request(const std::string& endpoint, 
                               const std::string& method = "GET",
                               const std::map<std::string, std::string>& params = {});
    std::string generate_signature(const std::string& endpoint,
                               const std::map<std::string, std::string>& params);
    void enforce_rate_limit();
    
    // WebSocket methods
    void setup_websocket_connection();
    void subscribe_to_ticker(const std::string& pair);
    void subscribe_to_ohlc(const std::string& pair, const std::string& interval);
    void subscribe_to_trades(const std::string& pair);
    void subscribe_to_order_book(const std::string& pair, int depth);
    
    // Data parsing methods
    std::vector<OHLCV> parse_ohlc_response(const std::string& response,
                                          const std::string& pair);
    OrderBook parse_depth_response(const std::string& response,
                                const std::string& pair);
    double parse_ticker_response(const std::string& response,
                               const std::string& pair);
    
    // Utility methods
    std::string convert_symbol_to_kraken(const std::string& symbol);
    std::string convert_kraken_to_symbol(const std::string& kraken_symbol);
    std::string convert_timeframe_to_interval(const std::string& timeframe);
    std::chrono::system_clock::time_point parse_kraken_timestamp(
        const std::string& timestamp);
    
    // Symbol mapping initialization
    void initialize_symbol_mapping();
    
    // Static callback for CURL
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp);
    
    // WebSocket message handlers
    void handle_ticker_message(const std::string& message,
                             std::function<void(const TickData&)> callback);
    void handle_ohlc_message(const std::string& message);
    void handle_trade_message(const std::string& message,
                           std::function<void(const TickData&)> callback);
    void handle_depth_message(const std::string& message);
};

} // namespace Data
} // namespace ArchNeuronX
