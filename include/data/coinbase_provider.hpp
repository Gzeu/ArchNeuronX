/**
 * @file coinbase_provider.hpp
 * @brief Coinbase Pro API data provider for cryptocurrencies
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
 * @struct CoinbaseEndpoints
 * @brief Coinbase Pro API endpoints
 */
struct CoinbaseEndpoints {
    static constexpr const char* REST_BASE = "https://api.pro.coinbase.com";
    static constexpr const char* WS_BASE = "wss://ws-feed.pro.coinbase.com";
    
    // REST endpoints
    static constexpr const char* PRODUCTS = "/products";
    static constexpr const char* TICKER = "/products/{}/ticker";
    static constexpr const char* ORDER_BOOK = "/products/{}/book?level={}";
    static constexpr const char* TRADES = "/products/{}/trades";
    static constexpr const char* CANDLES = "/products/{}/candles";
    static constexpr const char* STATS = "/products/{}/stats";
    static constexpr const char* CURRENCIES = "/currencies";
    
    // WebSocket channels
    static constexpr const char* TICKER_CHANNEL = "ticker";
    static constexpr const char* LEVEL2_CHANNEL = "level2";
    static constexpr const char& MATCHES_CHANNEL = "matches";
    static constexpr const char* FULL_CHANNEL = "full";
};

/**
 * @struct CoinbaseProduct
 * @brief Coinbase product information
 */
struct CoinbaseProduct {
    std::string id;
    std::string base_currency;
    std::string quote_currency;
    std::string display_name;
    double base_min_size;
    double base_max_size;
    double quote_increment;
    bool status_online;
    
    std::string to_json() const;
    static CoinbaseProduct from_json(const std::string& json_str);
};

/**
 * @class CoinbaseProvider
 * @brief Coinbase Pro cryptocurrency exchange data provider
 */
class CoinbaseProvider : public DataProvider {
public:
    /**
     * @brief Constructor
     * @param config Provider configuration
     */
    explicit CoinbaseProvider(const DataProviderConfig& config);
    
    /**
     * @brief Destructor
     */
    ~CoinbaseProvider() override;
    
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
    
    // Coinbase specific methods
    std::future<std::vector<CoinbaseProduct>> get_products();
    std::future<std::map<std::string, double>> get_tickers(
        const std::vector<std::string>& symbols);
    std::future<std::vector<std::string>> get_available_currencies();

private:
    CURL* curl_;
    websocketpp::client<websocketpp::config::asio_tls_client> ws_client_;
    std::string api_key_;
    std::string api_secret_;
    std::string passphrase_;
    
    std::atomic<bool> connected_;
    std::atomic<ConnectionStatus> status_;
    mutable std::mutex connection_mutex_;
    
    // WebSocket connections
    std::map<std::string, websocketpp::connection_hdl> ws_connections_;
    std::thread ws_thread_;
    
    // Rate limiting
    std::chrono::steady_clock::time_point last_request_time_;
    std::atomic<int> requests_per_second_;
    static constexpr int MAX_REQUESTS_PER_SECOND = 10;
    
    // Private methods
    bool test_rest_connection();
    std::string make_rest_request(const std::string& endpoint, 
                               const std::string& method = "GET",
                               const std::string& body = "");
    std::string generate_signature(const std::string& timestamp,
                               const std::string& method,
                               const std::string& request_path,
                               const std::string& body);
    void enforce_rate_limit();
    
    // WebSocket methods
    void setup_websocket_connection();
    void subscribe_to_ticker(const std::string& symbol);
    void subscribe_to_order_book(const std::string& symbol, int depth);
    void subscribe_to_trades(const std::string& symbol);
    
    // Data parsing methods
    std::vector<OHLCV> parse_candles_response(const std::string& response,
                                            const std::string& symbol);
    OrderBook parse_order_book_response(const std::string& response,
                                     const std::string& symbol);
    
    // Utility methods
    std::string convert_timeframe_to_granularity(const std::string& timeframe);
    std::string format_coinbase_symbol(const std::string& symbol);
    std::chrono::system_clock::time_point parse_coinbase_timestamp(
        const std::string& timestamp);
    
    // Static callback for CURL
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp);
    
    // WebSocket message handlers
    void handle_ticker_message(const std::string& message,
                             std::function<void(const TickData&)> callback);
    void handle_order_book_message(const std::string& message);
    void handle_trade_message(const std::string& message,
                           std::function<void(const TickData&)> callback);
};

} // namespace Data
} // namespace ArchNeuronX
