/**
 * @file yahoo_finance_provider.hpp
 * @brief Yahoo Finance API data provider for stocks and forex
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
 * @struct YahooFinanceEndpoints
 * @brief Yahoo Finance API endpoints
 */
struct YahooFinanceEndpoints {
    static constexpr const char* BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart";
    static constexpr const char* QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote";
    static constexpr const char* SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search";
    static constexpr const char* OPTIONS_URL = "https://query1.finance.yahoo.com/v7/finance/options";
    static constexpr const char* NEWS_URL = "https://query1.finance.yahoo.com/v1/finance/news";
    
    // Legacy API endpoints (still working)
    static constexpr const char* LEGACY_CHART = "https://query1.finance.yahoo.com/v7/finance/download";
    static constexpr const char* LEGACY_QUOTE = "https://query1.finance.yahoo.com/v7/finance/quoteSummary";
};

/**
 * @struct YahooFinanceSymbol
 * @brief Yahoo Finance symbol information
 */
struct YahooFinanceSymbol {
    std::string symbol;
    std::string name;
    std::string shortname;
    std::string longname;
    std::string exch;
    std::string type;
    std::string sector;
    std::string industry;
    double market_cap;
    double average_volume;
    
    std::string to_json() const;
    static YahooFinanceSymbol from_json(const std::string& json_str);
};

/**
 * @struct YahooFinanceQuote
 * @brief Yahoo Finance quote data
 */
struct YahooFinanceQuote {
    std::string symbol;
    std::string name;
    double current_price;
    double previous_close;
    double change;
    double change_percent;
    double day_high;
    double day_low;
    double volume;
    double avg_volume;
    double market_cap;
    double pe_ratio;
    double eps;
    std::string market_state;
    std::chrono::system_clock::time_point timestamp;
    
    std::string to_json() const;
    static YahooFinanceQuote from_json(const std::string& json_str);
};

/**
 * @class YahooFinanceProvider
 * @brief Yahoo Finance data provider for stocks and forex
 */
class YahooFinanceProvider : public DataProvider {
public:
    /**
     * @brief Constructor
     * @param config Provider configuration
     */
    explicit YahooFinanceProvider(const DataProviderConfig& config);
    
    /**
     * @brief Destructor
     */
    ~YahooFinanceProvider() override;
    
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
    
    // Yahoo Finance specific methods
    std::future<std::vector<YahooFinanceSymbol>> search_symbols(const std::string& keywords);
    std::future<std::vector<YahooFinanceQuote>> get_quotes(
        const std::vector<std::string>& symbols);
    std::future<std::vector<std::string>> get_market_movers(const std::string& market = "us_market");
    std::future<std::vector<std::string>> get_sector_performance();

private:
    CURL* curl_;
    std::string user_agent_;
    
    std::atomic<bool> connected_;
    std::atomic<ConnectionStatus> status_;
    mutable std::mutex connection_mutex_;
    
    // Rate limiting
    std::chrono::steady_clock::time_point last_request_time_;
    std::atomic<int> requests_per_minute_;
    static constexpr int MAX_REQUESTS_PER_MINUTE = 100;
    
    // Private methods
    bool test_api_connection();
    std::string make_api_request(const std::string& url);
    std::string build_chart_url(const std::string& symbol,
                               const std::string& interval,
                               const std::chrono::system_clock::time_point& start,
                               const std::chrono::system_clock::time_point& end);
    std::string build_quote_url(const std::vector<std::string>& symbols);
    void enforce_rate_limit();
    
    // Data parsing methods
    std::vector<OHLCV> parse_chart_response(const std::string& response,
                                           const std::string& symbol);
    std::vector<YahooFinanceQuote> parse_quote_response(const std::string& response);
    YahooFinanceQuote parse_single_quote(const json& quote_json);
    
    // Utility methods
    std::string convert_symbol_to_yahoo(const std::string& symbol);
    std::string convert_timeframe_to_interval(const std::string& timeframe);
    std::chrono::system_clock::time_point parse_yahoo_timestamp(
        const std::string& timestamp);
    std::string generate_crumb(); // Yahoo Finance requires crumb for some requests
    
    // Static callback for CURL
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp);
    
    // WebSocket simulation (Yahoo Finance doesn't provide official WebSocket)
    void simulate_real_time_data(const std::string& symbol,
                               std::function<void(const TickData&)> callback);
    std::thread polling_thread_;
    std::atomic<bool> polling_active_;
};

} // namespace Data
} // namespace ArchNeuronX
