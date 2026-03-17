/**
 * @file yahoo_finance_provider.cpp
 * @brief Yahoo Finance API data provider implementation
 * @author George Pricop
 * @date 2025-10-02
 */

#include "data/yahoo_finance_provider.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace ArchNeuronX {
namespace Data {

YahooFinanceProvider::YahooFinanceProvider(const DataProviderConfig& config) 
    : DataProvider(config), connected_(false), requests_per_minute_(0), polling_active_(false) {
    
    // Initialize CURL
    curl_ = curl_easy_init();
    if (!curl_) {
        throw std::runtime_error("Failed to initialize CURL");
    }
    
    // Set user agent to mimic browser
    user_agent_ = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36";
    
    last_request_time_ = std::chrono::steady_clock::now();
    status_ = ConnectionStatus::DISCONNECTED;
}

YahooFinanceProvider::~YahooFinanceProvider() {
    disconnect();
    if (curl_) {
        curl_easy_cleanup(curl_);
    }
}

bool YahooFinanceProvider::connect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    if (connected_) {
        return true;
    }
    
    try {
        // Test API connection with a simple quote request
        if (!test_api_connection()) {
            status_ = ConnectionStatus::ERROR;
            return false;
        }
        
        connected_ = true;
        status_ = ConnectionStatus::CONNECTED;
        requests_per_minute_ = 0;
        
        LOG_INFO("Yahoo Finance provider connected successfully");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Yahoo Finance connection failed: {}", e.what());
        status_ = ConnectionStatus::ERROR;
        return false;
    }
}

void YahooFinanceProvider::disconnect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    if (!connected_) {
        return;
    }
    
    // Stop polling thread
    polling_active_ = false;
    if (polling_thread_.joinable()) {
        polling_thread_.join();
    }
    
    connected_ = false;
    status_ = ConnectionStatus::DISCONNECTED;
    
    LOG_INFO("Yahoo Finance provider disconnected");
}

bool YahooFinanceProvider::is_connected() const {
    return connected_.load();
}

ConnectionStatus YahooFinanceProvider::get_status() const {
    return status_.load();
}

std::future<std::vector<OHLCV>> YahooFinanceProvider::get_historical_data(
    const std::string& symbol,
    const std::string& timeframe,
    const std::chrono::system_clock::time_point& start,
    const std::chrono::system_clock::time_point& end
) {
    return std::async(std::launch::async, [this, symbol, timeframe, start, end]() {
        std::vector<OHLCV> data;
        
        try {
            std::string yahoo_symbol = convert_symbol_to_yahoo(symbol);
            std::string interval = convert_timeframe_to_interval(timeframe);
            
            std::string url = build_chart_url(yahoo_symbol, interval, start, end);
            std::string response = make_api_request(url);
            
            data = parse_chart_response(response, symbol);
            
            // Filter by date range (Yahoo might return more data than requested)
            auto it = std::remove_if(data.begin(), data.end(), 
                [start, end](const OHLCV& ohlcv) {
                    return ohlcv.timestamp < start || ohlcv.timestamp > end;
                });
            data.erase(it, data.end());
            
            LOG_INFO("Retrieved {} OHLCV records for {} ({})", 
                     data.size(), symbol, timeframe);
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get historical data: {}", e.what());
        }
        
        return data;
    });
}

std::future<std::vector<TickData>> YahooFinanceProvider::get_real_time_ticks(
    const std::string& symbol,
    std::function<void(const TickData&)> callback
) {
    return std::async(std::launch::async, [this, symbol, callback]() {
        std::vector<TickData> ticks;
        
        try {
            // Yahoo Finance doesn't provide WebSocket, so we simulate with polling
            simulate_real_time_data(symbol, callback);
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to setup real-time ticks: {}", e.what());
        }
        
        return ticks;
    });
}

std::future<double> YahooFinanceProvider::get_current_price(const std::string& symbol) {
    return std::async(std::launch::async, [this, symbol]() {
        try {
            std::string yahoo_symbol = convert_symbol_to_yahoo(symbol);
            
            std::vector<std::string> symbols = {yahoo_symbol};
            std::string url = build_quote_url(symbols);
            std::string response = make_api_request(url);
            
            auto quotes = parse_quote_response(response);
            if (!quotes.empty()) {
                return quotes[0].current_price;
            }
            
            return 0.0;
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get current price: {}", e.what());
            return 0.0;
        }
    });
}

std::future<OrderBook> YahooFinanceProvider::get_order_book(const std::string& symbol, int depth) {
    return std::async(std::launch::async, [this, symbol, depth]() {
        OrderBook order_book;
        order_book.symbol = symbol;
        order_book.timestamp = std::chrono::system_clock::now();
        
        // Yahoo Finance doesn't provide order book data
        // Return empty order book
        LOG_WARN("Yahoo Finance does not provide order book data");
        
        return order_book;
    });
}

std::future<std::vector<YahooFinanceSymbol>> YahooFinanceProvider::search_symbols(const std::string& keywords) {
    return std::async(std::launch::async, [this, keywords]() {
        std::vector<YahooFinanceSymbol> symbols;
        
        try {
            std::string url = YahooFinanceEndpoints::SEARCH_URL + 
                             "?q=" + keywords + 
                             "&quotesCount=10" +
                             "&newsCount=0";
            
            std::string response = make_api_request(url);
            auto json_data = json::parse(response);
            
            if (json_data.contains("quotes")) {
                for (const auto& quote : json_data["quotes"]) {
                    YahooFinanceSymbol symbol_info;
                    symbol_info.symbol = quote["symbol"].get<std::string>();
                    symbol_info.name = quote["longname"].get<std::string>();
                    symbol_info.shortname = quote["shortname"].get<std::string>();
                    symbol_info.exch = quote["exchange"].get<std::string>();
                    symbol_info.type = quote["quoteType"].get<std::string>();
                    
                    if (quote.contains("marketCap")) {
                        symbol_info.market_cap = quote["marketCap"].get<double>();
                    }
                    
                    if (quote.contains("averageDailyVolume3Month")) {
                        symbol_info.average_volume = quote["averageDailyVolume3Month"].get<double>();
                    }
                    
                    symbols.push_back(symbol_info);
                }
            }
            
            LOG_INFO("Found {} symbols matching '{}'", symbols.size(), keywords);
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to search symbols: {}", e.what());
        }
        
        return symbols;
    });
}

std::future<std::vector<YahooFinanceQuote>> YahooFinanceProvider::get_quotes(
    const std::vector<std::string>& symbols
) {
    return std::async(std::launch::async, [this, symbols]() {
        std::vector<std::string> yahoo_symbols;
        
        // Convert symbols to Yahoo format
        for (const auto& symbol : symbols) {
            yahoo_symbols.push_back(convert_symbol_to_yahoo(symbol));
        }
        
        try {
            std::string url = build_quote_url(yahoo_symbols);
            std::string response = make_api_request(url);
            
            auto quotes = parse_quote_response(response);
            
            LOG_INFO("Retrieved quotes for {} symbols", quotes.size());
            
            return quotes;
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get quotes: {}", e.what());
            return std::vector<YahooFinanceQuote>();
        }
    });
}

std::future<std::vector<std::string>> YahooFinanceProvider::get_market_movers(const std::string& market) {
    return std::async(std::launch::async, [this, market]() {
        std::vector<std::string> movers;
        
        try {
            std::string url = YahooFinanceEndpoints::QUOTE_URL + 
                             "?ids=^GSPC,^DJI,^IXIC,^RUT" + // Major indices
                             "&fields=regularMarketPrice,regularMarketChangePercent" +
                             "&region=US" +
                             "&lang=en-US";
            
            std::string response = make_api_request(url);
            auto json_data = json::parse(response);
            
            if (json_data.contains("quoteResponse") && json_data["quoteResponse"].contains("result")) {
                for (const auto& quote : json_data["quoteResponse"]["result"]) {
                    std::string symbol = quote["symbol"].get<std::string>();
                    double change_percent = 0.0;
                    
                    if (quote.contains("regularMarketChangePercent")) {
                        change_percent = quote["regularMarketChangePercent"].get<double>();
                    }
                    
                    std::string mover_info = symbol + ": " + std::to_string(change_percent) + "%";
                    movers.push_back(mover_info);
                }
            }
            
            LOG_INFO("Retrieved {} market movers", movers.size());
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get market movers: {}", e.what());
        }
        
        return movers;
    });
}

std::future<std::vector<std::string>> YahooFinanceProvider::get_sector_performance() {
    return std::async(std::launch::async, [this]() {
        std::vector<std::string> sectors;
        
        try {
            // Yahoo Finance sector performance (using major sector ETFs)
            std::vector<std::string> sector_etfs = {
                "XLK", // Technology
                "XLF", // Financial
                "XLE", // Energy
                "XLV", // Health Care
                "XLI", // Industrial
                "XLU", // Utilities
                "XLP", // Consumer Staples
                "XLY", // Consumer Discretionary
                "XLB", // Materials
                "XLRE" // Real Estate
            };
            
            std::string url = build_quote_url(sector_etfs);
            std::string response = make_api_request(url);
            
            auto quotes = parse_quote_response(response);
            
            for (const auto& quote : quotes) {
                std::string sector_info = quote.symbol + ": " + 
                                       std::to_string(quote.change_percent) + "%";
                sectors.push_back(sector_info);
            }
            
            LOG_INFO("Retrieved {} sector performances", sectors.size());
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get sector performance: {}", e.what());
        }
        
        return sectors;
    });
}

// Private methods implementation

bool YahooFinanceProvider::test_api_connection() {
    try {
        std::string url = YahooFinanceEndpoints::QUOTE_URL + "?symbols=AAPL&fields=regularMarketPrice";
        std::string response = make_api_request(url);
        auto json_data = json::parse(response);
        return json_data.contains("quoteResponse");
        
    } catch (const std::exception& e) {
        LOG_ERROR("API connection test failed: {}", e.what());
        return false;
    }
}

std::string YahooFinanceProvider::make_api_request(const std::string& url) {
    enforce_rate_limit();
    
    std::string response;
    
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_, CURLOPT_USERAGENT, user_agent_.c_str());
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, 30L);
    
    CURLcode res = curl_easy_perform(curl_);
    if (res != CURLE_OK) {
        throw std::runtime_error("CURL request failed: " + std::string(curl_easy_strerror(res)));
    }
    
    long response_code;
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &response_code);
    if (response_code != 200) {
        throw std::runtime_error("HTTP error: " + std::to_string(response_code));
    }
    
    // Update request count
    requests_per_minute_++;
    
    return response;
}

std::string YahooFinanceProvider::build_chart_url(
    const std::string& symbol,
    const std::string& interval,
    const std::chrono::system_clock::time_point& start,
    const std::chrono::system_clock::time_point& end
) {
    auto start_ts = std::chrono::duration_cast<std::chrono::seconds>(
        start.time_since_epoch()).count();
    auto end_ts = std::chrono::duration_cast<std::chrono::seconds>(
        end.time_since_epoch()).count();
    
    std::string url = YahooFinanceEndpoints::BASE_URL +
                     "?symbol=" + symbol +
                     "&interval=" + interval +
                     "&period1=" + std::to_string(start_ts) +
                     "&period2=" + std::to_string(end_ts) +
                     "&includePrePost=true" +
                     "&events=div%7Csplit";
    
    return url;
}

std::string YahooFinanceProvider::build_quote_url(const std::vector<std::string>& symbols) {
    std::string url = YahooFinanceEndpoints::QUOTE_URL + "?symbols=";
    
    for (size_t i = 0; i < symbols.size(); ++i) {
        if (i > 0) url += ",";
        url += symbols[i];
    }
    
    url += "&fields=regularMarketPrice,regularMarketChange,regularMarketChangePercent," +
           "dayHigh,dayLow,regularMarketVolume,averageDailyVolume3Month," +
           "marketCap,forwardPE,forwardEPS,marketState";
    
    return url;
}

void YahooFinanceProvider::enforce_rate_limit() {
    auto now = std::chrono::steady_clock::now();
    auto time_since_last = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_request_time_).count();
    
    // Reset counter if more than a minute has passed
    if (time_since_last >= 60) {
        requests_per_minute_ = 0;
        last_request_time_ = now;
        return;
    }
    
    // If we've hit the limit, wait
    if (requests_per_minute_ >= MAX_REQUESTS_PER_MINUTE) {
        int wait_time = 60 - time_since_last;
        LOG_WARN("Rate limit reached, waiting {} seconds", wait_time);
        std::this_thread::sleep_for(std::chrono::seconds(wait_time));
        requests_per_minute_ = 0;
        last_request_time_ = std::chrono::steady_clock::now();
    }
    
    last_request_time_ = now;
}

std::vector<OHLCV> YahooFinanceProvider::parse_chart_response(
    const std::string& response,
    const std::string& symbol
) {
    std::vector<OHLCV> data;
    
    auto json_data = json::parse(response);
    
    if (json_data.contains("chart") && json_data["chart"].contains("result")) {
        auto result = json_data["chart"]["result"];
        
        if (!result.empty() && result[0].contains("timestamp")) {
            auto timestamps = result[0]["timestamp"];
            auto opens = result[0]["indicators"]["quote"][0]["open"];
            auto highs = result[0]["indicators"]["quote"][0]["high"];
            auto lows = result[0]["indicators"]["quote"][0]["low"];
            auto closes = result[0]["indicators"]["quote"][0]["close"];
            auto volumes = result[0]["indicators"]["quote"][0]["volume"];
            
            for (size_t i = 0; i < timestamps.size(); ++i) {
                OHLCV ohlcv;
                ohlcv.symbol = symbol;
                
                if (i < timestamps.size()) {
                    ohlcv.timestamp = std::chrono::system_clock::from_time_t(timestamps[i].get<long>());
                }
                
                if (i < opens.size() && !opens[i].is_null()) {
                    ohlcv.open = opens[i].get<double>();
                }
                
                if (i < highs.size() && !highs[i].is_null()) {
                    ohlcv.high = highs[i].get<double>();
                }
                
                if (i < lows.size() && !lows[i].is_null()) {
                    ohlcv.low = lows[i].get<double>();
                }
                
                if (i < closes.size() && !closes[i].is_null()) {
                    ohlcv.close = closes[i].get<double>();
                }
                
                if (i < volumes.size() && !volumes[i].is_null()) {
                    ohlcv.volume = volumes[i].get<double>();
                }
                
                data.push_back(ohlcv);
            }
        }
    }
    
    return data;
}

std::vector<YahooFinanceQuote> YahooFinanceProvider::parse_quote_response(const std::string& response) {
    std::vector<YahooFinanceQuote> quotes;
    
    auto json_data = json::parse(response);
    
    if (json_data.contains("quoteResponse") && json_data["quoteResponse"].contains("result")) {
        for (const auto& quote : json_data["quoteResponse"]["result"]) {
            quotes.push_back(parse_single_quote(quote));
        }
    }
    
    return quotes;
}

YahooFinanceQuote YahooFinanceProvider::parse_single_quote(const json& quote_json) {
    YahooFinanceQuote quote;
    
    quote.symbol = quote_json["symbol"].get<std::string>();
    quote.name = quote_json.value("longname", "");
    quote.current_price = quote_json.value("regularMarketPrice", 0.0);
    quote.previous_close = quote_json.value("regularMarketPreviousClose", 0.0);
    quote.change = quote_json.value("regularMarketChange", 0.0);
    quote.change_percent = quote_json.value("regularMarketChangePercent", 0.0);
    quote.day_high = quote_json.value("dayHigh", 0.0);
    quote.day_low = quote_json.value("dayLow", 0.0);
    quote.volume = quote_json.value("regularMarketVolume", 0.0);
    quote.avg_volume = quote_json.value("averageDailyVolume3Month", 0.0);
    quote.market_cap = quote_json.value("marketCap", 0.0);
    quote.pe_ratio = quote_json.value("forwardPE", 0.0);
    quote.eps = quote_json.value("forwardEPS", 0.0);
    quote.market_state = quote_json.value("marketState", "");
    quote.timestamp = std::chrono::system_clock::now();
    
    return quote;
}

std::string YahooFinanceProvider::convert_symbol_to_yahoo(const std::string& symbol) {
    // Convert common symbol formats to Yahoo Finance format
    std::string yahoo_symbol = symbol;
    
    // Replace / with - for crypto pairs
    std::replace(yahoo_symbol.begin(), yahoo_symbol.end(), '/', '-');
    
    // Add suffix for common stock exchanges if needed
    if (yahoo_symbol.find("-") == std::string::npos && 
        yahoo_symbol.length() <= 5 && 
        std::all_of(yahoo_symbol.begin(), yahoo_symbol.end(), ::isupper)) {
        yahoo_symbol += ".US"; // Assume US exchange
    }
    
    return yahoo_symbol;
}

std::string YahooFinanceProvider::convert_timeframe_to_interval(const std::string& timeframe) {
    // Map timeframes to Yahoo Finance intervals
    if (timeframe == "1m") return "1m";
    if (timeframe == "2m") return "2m";
    if (timeframe == "5m") return "5m";
    if (timeframe == "15m") return "15m";
    if (timeframe == "30m") return "30m";
    if (timeframe == "60m") return "1h";
    if (timeframe == "90m") return "90m";
    if (timeframe == "1h") return "1h";
    if (timeframe == "1d") return "1d";
    if (timeframe == "5d") return "5d";
    if (timeframe == "1wk") return "1wk";
    if (timeframe == "1mo") return "1mo";
    if (timeframe == "3mo") return "3mo";
    
    // Default to 1 hour
    return "1h";
}

void YahooFinanceProvider::simulate_real_time_data(
    const std::string& symbol,
    std::function<void(const TickData&)> callback
) {
    polling_active_ = true;
    
    polling_thread_ = std::thread([this, symbol, callback]() {
        while (polling_active_ && connected_) {
            try {
                auto price_future = get_current_price(symbol);
                double price = price_future.get();
                
                if (price > 0.0 && callback) {
                    TickData tick;
                    tick.symbol = symbol;
                    tick.price = price;
                    tick.quantity = 0.0; // Yahoo doesn't provide volume in quotes
                    tick.timestamp = std::chrono::system_clock::now();
                    
                    callback(tick);
                }
                
                // Poll every 5 seconds for "real-time" feel
                std::this_thread::sleep_for(std::chrono::seconds(5));
                
            } catch (const std::exception& e) {
                LOG_ERROR("Error in real-time polling: {}", e.what());
                std::this_thread::sleep_for(std::chrono::seconds(10));
            }
        }
    });
}

size_t YahooFinanceProvider::WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

} // namespace Data
} // namespace ArchNeuronX
