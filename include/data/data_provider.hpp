/**
 * @file data_provider.hpp
 * @brief Abstract data provider interface for ArchNeuronX
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>
#include <future>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

#include "market_data.hpp"
#include "core/logger.hpp"

namespace ArchNeuronX {
namespace Data {

/**
 * @enum DataProviderType
 * @brief Types of data providers
 */
enum class DataProviderType {
    CRYPTO_EXCHANGE,  ///< Cryptocurrency exchange (Binance, Coinbase, etc.)
    FOREX_PROVIDER,   ///< Forex data provider (Alpha Vantage, etc.)
    STOCK_PROVIDER,   ///< Stock market data provider
    CUSTOM           ///< Custom data source
};

/**
 * @enum ConnectionStatus
 * @brief Connection status for data providers
 */
enum class ConnectionStatus {
    DISCONNECTED,     ///< Not connected
    CONNECTING,       ///< Connection in progress
    CONNECTED,        ///< Connected and operational
    RECONNECTING,     ///< Attempting to reconnect
    ERROR            ///< Connection error
};

/**
 * @struct DataProviderConfig
 * @brief Configuration for data providers
 */
struct DataProviderConfig {
    std::string name;                    ///< Provider name
    std::string base_url;                ///< API base URL
    std::string api_key;                 ///< API key
    std::string secret_key;              ///< Secret key (if required)
    bool use_testnet = false;            ///< Use testnet/sandbox
    int rate_limit_per_minute = 1200;    ///< Rate limit requests per minute
    int max_retries = 3;                 ///< Maximum retry attempts
    int timeout_ms = 5000;               ///< Request timeout in milliseconds
    bool enable_websocket = true;        ///< Enable WebSocket connections
    std::vector<std::string> symbols;    ///< Symbols to subscribe to
    
    /**
     * @brief Validate configuration
     * @throws std::invalid_argument if configuration is invalid
     */
    void validate() const;
};

/**
 * @struct RateLimiter
 * @brief Rate limiting for API requests
 */
class RateLimiter {
public:
    /**
     * @brief Constructor
     * @param requests_per_minute Maximum requests per minute
     */
    explicit RateLimiter(int requests_per_minute);
    
    /**
     * @brief Wait if necessary to respect rate limits
     * @return True if request can proceed, false if rate limited
     */
    bool waitForToken();
    
    /**
     * @brief Get current rate limit status
     * @return Remaining requests in current window
     */
    int getRemainingRequests() const;
    
private:
    const int max_requests_;
    mutable std::mutex mutex_;
    std::queue<std::chrono::steady_clock::time_point> request_times_;
};

/**
 * @brief Callback function type for real-time data
 */
using DataCallback = std::function<void(const MarketDataPoint&)>;

/**
 * @brief Callback function type for connection status changes
 */
using StatusCallback = std::function<void(ConnectionStatus, const std::string&)>;

/**
 * @class DataProvider
 * @brief Abstract base class for market data providers
 */
class DataProvider {
public:
    /**
     * @brief Constructor
     * @param type Provider type
     * @param config Provider configuration
     */
    DataProvider(DataProviderType type, const DataProviderConfig& config);
    
    /**
     * @brief Virtual destructor
     */
    virtual ~DataProvider();
    
    /**
     * @brief Initialize the data provider
     * @return True if initialization successful
     */
    virtual bool initialize() = 0;
    
    /**
     * @brief Connect to the data source
     * @return True if connection successful
     */
    virtual bool connect() = 0;
    
    /**
     * @brief Disconnect from the data source
     */
    virtual void disconnect() = 0;
    
    /**
     * @brief Get historical OHLCV data
     * @param symbol Trading symbol
     * @param timeframe Data timeframe
     * @param start_time Start time for data
     * @param end_time End time for data
     * @param limit Maximum number of records (0 = no limit)
     * @return Future containing historical data
     */
    virtual std::future<std::vector<OHLCV>> getHistoricalData(
        const std::string& symbol,
        TimeFrame timeframe,
        const std::chrono::system_clock::time_point& start_time,
        const std::chrono::system_clock::time_point& end_time,
        size_t limit = 0
    ) = 0;
    
    /**
     * @brief Get current market price
     * @param symbol Trading symbol
     * @return Future containing current price
     */
    virtual std::future<double> getCurrentPrice(const std::string& symbol) = 0;
    
    /**
     * @brief Get order book snapshot
     * @param symbol Trading symbol
     * @param depth Order book depth
     * @return Future containing order book
     */
    virtual std::future<OrderBook> getOrderBook(
        const std::string& symbol, 
        int depth = 20
    ) = 0;
    
    /**
     * @brief Get recent trades
     * @param symbol Trading symbol
     * @param limit Number of recent trades
     * @return Future containing recent trades
     */
    virtual std::future<std::vector<TickData>> getRecentTrades(
        const std::string& symbol,
        size_t limit = 100
    ) = 0;
    
    /**
     * @brief Subscribe to real-time data for a symbol
     * @param symbol Trading symbol
     * @param timeframe Data timeframe
     * @param callback Callback function for data updates
     * @return True if subscription successful
     */
    virtual bool subscribeToData(
        const std::string& symbol,
        TimeFrame timeframe,
        DataCallback callback
    ) = 0;
    
    /**
     * @brief Unsubscribe from real-time data
     * @param symbol Trading symbol
     * @param timeframe Data timeframe
     * @return True if unsubscription successful
     */
    virtual bool unsubscribeFromData(
        const std::string& symbol,
        TimeFrame timeframe
    ) = 0;
    
    /**
     * @brief Get available trading symbols
     * @return Future containing list of symbols
     */
    virtual std::future<std::vector<std::string>> getAvailableSymbols() = 0;
    
    /**
     * @brief Get market information for a symbol
     * @param symbol Trading symbol
     * @return Future containing market info
     */
    virtual std::future<MarketInfo> getMarketInfo(const std::string& symbol) = 0;
    
    /**
     * @brief Set status callback for connection events
     * @param callback Status callback function
     */
    void setStatusCallback(StatusCallback callback) {
        status_callback_ = callback;
    }
    
    // Getters
    DataProviderType getType() const { return type_; }
    const std::string& getName() const { return config_.name; }
    ConnectionStatus getStatus() const { return status_.load(); }
    const DataProviderConfig& getConfig() const { return config_; }
    bool isConnected() const { return status_.load() == ConnectionStatus::CONNECTED; }
    
    /**
     * @brief Get provider statistics
     * @return Map of statistic name to value
     */
    std::map<std::string, double> getStatistics() const;
    
protected:
    /**
     * @brief Update connection status
     * @param status New connection status
     * @param message Optional status message
     */
    void updateStatus(ConnectionStatus status, const std::string& message = "");
    
    /**
     * @brief Check if rate limit allows request
     * @return True if request can proceed
     */
    bool checkRateLimit();
    
    /**
     * @brief Record API request for statistics
     */
    void recordRequest();
    
    /**
     * @brief Record API error for statistics
     */
    void recordError();
    
    /**
     * @brief Validate symbol format
     * @param symbol Symbol to validate
     * @return True if symbol is valid
     */
    virtual bool isValidSymbol(const std::string& symbol) const;
    
    /**
     * @brief Parse error response from API
     * @param response API response
     * @return Error message
     */
    virtual std::string parseErrorMessage(const std::string& response) const;
    
    // Member variables
    DataProviderType type_;
    DataProviderConfig config_;
    std::atomic<ConnectionStatus> status_{ConnectionStatus::DISCONNECTED};
    
    std::unique_ptr<RateLimiter> rate_limiter_;
    StatusCallback status_callback_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    std::atomic<uint64_t> total_requests_{0};
    std::atomic<uint64_t> failed_requests_{0};
    std::chrono::steady_clock::time_point start_time_;
    
    // Threading for real-time data
    std::atomic<bool> running_{false};
    std::unique_ptr<std::thread> worker_thread_;
    mutable std::mutex callback_mutex_;
    std::map<std::pair<std::string, TimeFrame>, DataCallback> data_callbacks_;
};

/**
 * @brief Convert DataProviderType to string
 * @param type Provider type
 * @return String representation
 */
std::string dataProviderTypeToString(DataProviderType type);

/**
 * @brief Convert ConnectionStatus to string
 * @param status Connection status
 * @return String representation
 */
std::string connectionStatusToString(ConnectionStatus status);

/**
 * @brief Create data provider instance
 * @param type Provider type
 * @param config Provider configuration
 * @return Unique pointer to data provider
 */
std::unique_ptr<DataProvider> createDataProvider(
    DataProviderType type,
    const DataProviderConfig& config
);

} // namespace Data
} // namespace ArchNeuronX