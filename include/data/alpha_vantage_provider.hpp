/**
 * @file alpha_vantage_provider.hpp
 * @brief Alpha Vantage API data provider for stocks and forex
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include "data_provider.hpp"
#include <curl/curl.h>
#include <json/json.h>

namespace ArchNeuronX {
namespace Data {

/**
 * @struct AlphaVantageEndpoints
 * @brief Alpha Vantage API endpoints
 */
struct AlphaVantageEndpoints {
    static constexpr const char* BASE_URL = "https://www.alphavantage.co/query";
    
    // Stock endpoints
    static constexpr const char* TIME_SERIES_DAILY = "TIME_SERIES_DAILY";
    static constexpr const char* TIME_SERIES_INTRADAY = "TIME_SERIES_INTRADAY";
    static constexpr const char* TIME_SERIES_WEEKLY = "TIME_SERIES_WEEKLY";
    static constexpr const char* TIME_SERIES_MONTHLY = "TIME_SERIES_MONTHLY";
    
    // Forex endpoints
    static constexpr const char* FX_INTRADAY = "FX_INTRADAY";
    static constexpr const char* FX_DAILY = "FX_DAILY";
    static constexpr const char* FX_WEEKLY = "FX_WEEKLY";
    static constexpr const char* FX_MONTHLY = "FX_MONTHLY";
    
    // Market endpoints
    static constexpr const char* GLOBAL_QUOTE = "GLOBAL_QUOTE";
    static constexpr const char* SYMBOL_SEARCH = "SYMBOL_SEARCH";
    static constexpr const char* MARKET_STATUS = "MARKET_STATUS";
};

/**
 * @class AlphaVantageProvider
 * @brief Alpha Vantage financial data provider
 */
class AlphaVantageProvider : public DataProvider {
public:
    /**
     * @brief Constructor
     * @param config Provider configuration
     */
    explicit AlphaVantageProvider(const DataProviderConfig& config);
    
    /**
     * @brief Destructor
     */
    ~AlphaVantageProvider() override;
    
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
    
    // Alpha Vantage specific methods
    std::future<std::vector<std::string>> search_symbols(const std::string& keywords);
    std::future<MarketStatus> get_market_status(const std::string& market = "");
    std::future<std::map<std::string, double>> get_global_quotes(const std::vector<std::string>& symbols);

private:
    CURL* curl_;
    std::string api_key_;
    std::atomic<bool> connected_;
    std::atomic<ConnectionStatus> status_;
    mutable std::mutex connection_mutex_;
    
    // Rate limiting
    std::chrono::steady_clock::time_point last_request_time_;
    std::atomic<int> requests_per_minute_;
    static constexpr int MAX_REQUESTS_PER_MINUTE = 5;
    
    // Private methods
    bool test_api_connection();
    std::string make_api_request(const std::string& function, 
                               const std::map<std::string, std::string>& params);
    std::string build_query_string(const std::map<std::string, std::string>& params);
    void enforce_rate_limit();
    
    // Data parsing methods
    std::vector<OHLCV> parse_time_series_response(const std::string& response, 
                                                const std::string& symbol);
    std::vector<OHLCV> parse_forex_response(const std::string& response, 
                                           const std::string& from_symbol,
                                           const std::string& to_symbol);
    
    // Utility methods
    std::string convert_timeframe_to_function(const std::string& timeframe);
    std::chrono::system_clock::time_point parse_alpha_vantage_timestamp(const std::string& timestamp);
    
    // Static callback for CURL
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp);
};

} // namespace Data
} // namespace ArchNeuronX
