/**
 * @file binance_provider.hpp
 * @brief Binance API data provider implementation for ArchNeuronX
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
 * @struct BinanceEndpoints
 * @brief Binance API endpoints
 */
struct BinanceEndpoints {
    static constexpr const char* MAINNET_BASE = "https://api.binance.com";
    static constexpr const char* TESTNET_BASE = "https://testnet.binance.vision";
    static constexpr const char* WS_MAINNET = "wss://stream.binance.com:9443/ws/";
    static constexpr const char* WS_TESTNET = "wss://testnet.binance.vision/ws/";
    
    // REST endpoints
    static constexpr const char* EXCHANGE_INFO = "/api/v3/exchangeInfo";
    static constexpr const char* KLINES = "/api/v3/klines";
    static constexpr const char* TICKER_PRICE = "/api/v3/ticker/price";
    static constexpr const char* ORDER_BOOK = "/api/v3/depth";
    static constexpr const char* RECENT_TRADES = "/api/v3/trades";
    static constexpr const char* TICKER_24HR = "/api/v3/ticker/24hr";
};

/**
 * @class BinanceProvider
 * @brief Binance cryptocurrency exchange data provider
 */
class BinanceProvider : public DataProvider {
public:
    /**
     * @brief Constructor
     * @param config Binance provider configuration
     */
    explicit BinanceProvider(const DataProviderConfig& config);
    
    /**
     * @brief Destructor
     */
    ~BinanceProvider() override;
    
    /**
     * @brief Initialize the Binance provider
     * @return True if initialization successful
     */
    bool initialize() override;
    
    /**
     * @brief Connect to Binance API
     * @return True if connection successful
     */
    bool connect() override;
    
    /**
     * @brief Disconnect from Binance API
     */
    void disconnect() override;
    
    /**
     * @brief Get historical OHLCV data from Binance
     * @param symbol Trading symbol (e.g., "BTCUSDT")
     * @param timeframe Data timeframe
     * @param start_time Start time for data
     * @param end_time End time for data
     * @param limit Maximum number of records
     * @return Future containing historical data
     */
    std::future<std::vector<OHLCV>> getHistoricalData(
        const std::string& symbol,
        TimeFrame timeframe,
        const std::chrono::system_clock::time_point& start_time,
        const std::chrono::system_clock::time_point& end_time,
        size_t limit = 1000
    ) override;
    
    /**
     * @brief Get current market price from Binance
     * @param symbol Trading symbol
     * @return Future containing current price
     */
    std::future<double> getCurrentPrice(const std::string& symbol) override;
    
    /**
     * @brief Get order book snapshot from Binance
     * @param symbol Trading symbol
     * @param depth Order book depth (5, 10, 20, 50, 100, 500, 1000, 5000)
     * @return Future containing order book
     */
    std::future<OrderBook> getOrderBook(
        const std::string& symbol, 
        int depth = 20
    ) override;
    
    /**
     * @brief Get recent trades from Binance
     * @param symbol Trading symbol
     * @param limit Number of recent trades (max 1000)
     * @return Future containing recent trades
     */
    std::future<std::vector<TickData>> getRecentTrades(
        const std::string& symbol,
        size_t limit = 100
    ) override;
    
    /**
     * @brief Subscribe to real-time data via WebSocket
     * @param symbol Trading symbol
     * @param timeframe Data timeframe
     * @param callback Callback function for data updates
     * @return True if subscription successful
     */
    bool subscribeToData(
        const std::string& symbol,
        TimeFrame timeframe,
        DataCallback callback
    ) override;
    
    /**
     * @brief Unsubscribe from real-time data
     * @param symbol Trading symbol
     * @param timeframe Data timeframe
     * @return True if unsubscription successful
     */
    bool unsubscribeFromData(
        const std::string& symbol,
        TimeFrame timeframe
    ) override;
    
    /**
     * @brief Get available trading symbols from Binance
     * @return Future containing list of symbols
     */
    std::future<std::vector<std::string>> getAvailableSymbols() override;
    
    /**
     * @brief Get market information for a symbol
     * @param symbol Trading symbol
     * @return Future containing market info
     */
    std::future<MarketInfo> getMarketInfo(const std::string& symbol) override;
    
    /**
     * @brief Get 24hr ticker statistics
     * @param symbol Trading symbol (empty for all symbols)
     * @return Future containing ticker data
     */
    std::future<Json::Value> get24hrTicker(const std::string& symbol = "");
    
private:
    using WebSocketClient = websocketpp::client<websocketpp::config::asio_tls_client>;
    using WebSocketMessage = websocketpp::config::asio_client::message_type::ptr;
    
    /**
     * @brief HTTP response structure
     */
    struct HttpResponse {
        std::string data;
        long response_code = 0;
        std::string error_message;
    };
    
    /**
     * @brief Make HTTP GET request
     * @param endpoint API endpoint
     * @param params Query parameters
     * @return HTTP response
     */
    HttpResponse makeHttpRequest(const std::string& endpoint, 
                                const std::map<std::string, std::string>& params = {});
    
    /**
     * @brief Generate HMAC-SHA256 signature for authenticated requests
     * @param query_string Query string to sign
     * @return Base64 encoded signature
     */
    std::string generateSignature(const std::string& query_string) const;
    
    /**
     * @brief Convert TimeFrame to Binance interval string
     * @param tf TimeFrame value
     * @return Binance interval string
     */
    std::string timeFrameToBinanceInterval(TimeFrame tf) const;
    
    /**
     * @brief Parse Binance kline data to OHLCV
     * @param kline_array JSON array of kline data
     * @return Vector of OHLCV data
     */
    std::vector<OHLCV> parseKlineData(const Json::Value& kline_array) const;
    
    /**
     * @brief Parse Binance order book data
     * @param json_data JSON order book data
     * @param symbol Trading symbol
     * @return OrderBook structure
     */
    OrderBook parseOrderBookData(const Json::Value& json_data, 
                                const std::string& symbol) const;
    
    /**
     * @brief Parse Binance trade data
     * @param json_data JSON trade data
     * @param symbol Trading symbol
     * @return Vector of TickData
     */
    std::vector<TickData> parseTradeData(const Json::Value& json_data, 
                                        const std::string& symbol) const;
    
    /**
     * @brief Initialize WebSocket connection
     * @return True if initialization successful
     */
    bool initializeWebSocket();
    
    /**
     * @brief WebSocket message handler
     * @param hdl Connection handle
     * @param msg Message
     */
    void onWebSocketMessage(websocketpp::connection_hdl hdl, WebSocketMessage msg);
    
    /**
     * @brief WebSocket connection opened handler
     * @param hdl Connection handle
     */
    void onWebSocketOpen(websocketpp::connection_hdl hdl);
    
    /**
     * @brief WebSocket connection closed handler
     * @param hdl Connection handle
     */
    void onWebSocketClose(websocketpp::connection_hdl hdl);
    
    /**
     * @brief WebSocket error handler
     * @param hdl Connection handle
     */
    void onWebSocketError(websocketpp::connection_hdl hdl);
    
    /**
     * @brief Process WebSocket kline message
     * @param json_data Kline data
     */
    void processKlineMessage(const Json::Value& json_data);
    
    /**
     * @brief WebSocket worker thread function
     */
    void webSocketWorker();
    
    /**
     * @brief Validate Binance symbol format
     * @param symbol Symbol to validate
     * @return True if symbol is valid
     */
    bool isValidSymbol(const std::string& symbol) const override;
    
    /**
     * @brief CURL write callback function
     * @param contents Response data
     * @param size Size multiplier
     * @param nmemb Number of elements
     * @param userp User pointer (HttpResponse)
     * @return Number of bytes processed
     */
    static size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp);
    
    // Member variables
    CURL* curl_handle_;
    std::string base_url_;
    std::string ws_url_;
    Json::CharReaderBuilder json_reader_builder_;
    
    // WebSocket client
    std::unique_ptr<WebSocketClient> ws_client_;
    websocketpp::connection_hdl ws_connection_hdl_;
    std::unique_ptr<std::thread> ws_thread_;
    std::atomic<bool> ws_connected_{false};
    
    // Exchange info cache
    mutable std::mutex exchange_info_mutex_;
    Json::Value exchange_info_;
    std::chrono::steady_clock::time_point exchange_info_update_time_;
    
    // Active subscriptions
    mutable std::mutex subscriptions_mutex_;
    std::set<std::string> active_subscriptions_;
};

} // namespace Data
} // namespace ArchNeuronX