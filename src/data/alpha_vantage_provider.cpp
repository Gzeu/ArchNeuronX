/**
 * @file alpha_vantage_provider.cpp
 * @brief Alpha Vantage API data provider implementation
 * @author George Pricop
 * @date 2025-10-02
 */

#include "data/alpha_vantage_provider.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace ArchNeuronX {
namespace Data {

AlphaVantageProvider::AlphaVantageProvider(const DataProviderConfig& config) 
    : DataProvider(config), connected_(false), requests_per_minute_(0) {
    
    api_key_ = config.api_key;
    if (api_key_.empty()) {
        throw std::invalid_argument("Alpha Vantage API key is required");
    }
    
    // Initialize CURL
    curl_ = curl_easy_init();
    if (!curl_) {
        throw std::runtime_error("Failed to initialize CURL");
    }
    
    last_request_time_ = std::chrono::steady_clock::now();
    status_ = ConnectionStatus::DISCONNECTED;
}

AlphaVantageProvider::~AlphaVantageProvider() {
    disconnect();
    if (curl_) {
        curl_easy_cleanup(curl_);
    }
}

bool AlphaVantageProvider::connect() {
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
        
        LOG_INFO("Alpha Vantage provider connected successfully");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Alpha Vantage connection failed: {}", e.what());
        status_ = ConnectionStatus::ERROR;
        return false;
    }
}

void AlphaVantageProvider::disconnect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    if (!connected_) {
        return;
    }
    
    connected_ = false;
    status_ = ConnectionStatus::DISCONNECTED;
    
    LOG_INFO("Alpha Vantage provider disconnected");
}

bool AlphaVantageProvider::is_connected() const {
    return connected_.load();
}

ConnectionStatus AlphaVantageProvider::get_status() const {
    return status_.load();
}

std::future<std::vector<OHLCV>> AlphaVantageProvider::get_historical_data(
    const std::string& symbol,
    const std::string& timeframe,
    const std::chrono::system_clock::time_point& start,
    const std::chrono::system_clock::time_point& end
) {
    return std::async(std::launch::async, [this, symbol, timeframe, start, end]() {
        std::vector<OHLCV> data;
        
        try {
            std::string function = convert_timeframe_to_function(timeframe);
            
            std::map<std::string, std::string> params = {
                {"function", function},
                {"symbol", symbol},
                {"outputsize", "full"},
                {"apikey", api_key_}
            };
            
            // Add interval for intraday data
            if (function == AlphaVantageEndpoints::TIME_SERIES_INTRADAY) {
                params["interval"] = timeframe;
            }
            
            std::string response = make_api_request(function, params);
            data = parse_time_series_response(response, symbol);
            
            // Filter by date range
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

std::future<std::vector<TickData>> AlphaVantageProvider::get_real_time_ticks(
    const std::string& symbol,
    std::function<void(const TickData&)> callback
) {
    return std::async(std::launch::async, [this, symbol, callback]() {
        std::vector<TickData> ticks;
        
        try {
            // Alpha Vantage doesn't provide real-time tick data
            // We'll use GLOBAL_QUOTE endpoint for current price
            std::map<std::string, std::string> params = {
                {"function", AlphaVantageEndpoints::GLOBAL_QUOTE},
                {"symbol", symbol},
                {"apikey", api_key_}
            };
            
            // Poll every 5 seconds for "real-time" updates
            while (connected_) {
                try {
                    std::string response = make_api_request(
                        AlphaVantageEndpoints::GLOBAL_QUOTE, params);
                    
                    auto json_data = json::parse(response);
                    if (json_data.contains("Global Quote")) {
                        auto quote = json_data["Global Quote"];
                        
                        TickData tick;
                        tick.symbol = symbol;
                        tick.price = std::stod(quote["05. price"].get<std::string>());
                        tick.quantity = 0.0; // Alpha Vantage doesn't provide volume in quotes
                        tick.timestamp = std::chrono::system_clock::now();
                        
                        if (callback) {
                            callback(tick);
                        }
                        
                        ticks.push_back(tick);
                    }
                    
                    // Wait before next poll
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                    
                } catch (const std::exception& e) {
                    LOG_ERROR("Error in real-time tick polling: {}", e.what());
                    std::this_thread::sleep_for(std::chrono::seconds(10));
                }
            }
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to setup real-time ticks: {}", e.what());
        }
        
        return ticks;
    });
}

std::future<double> AlphaVantageProvider::get_current_price(const std::string& symbol) {
    return std::async(std::launch::async, [this, symbol]() {
        try {
            std::map<std::string, std::string> params = {
                {"function", AlphaVantageEndpoints::GLOBAL_QUOTE},
                {"symbol", symbol},
                {"apikey", api_key_}
            };
            
            std::string response = make_api_request(
                AlphaVantageEndpoints::GLOBAL_QUOTE, params);
            
            auto json_data = json::parse(response);
            if (json_data.contains("Global Quote")) {
                auto quote = json_data["Global Quote"];
                return std::stod(quote["05. price"].get<std::string>());
            }
            
            return 0.0;
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get current price: {}", e.what());
            return 0.0;
        }
    });
}

std::future<OrderBook> AlphaVantageProvider::get_order_book(const std::string& symbol, int depth) {
    return std::async(std::launch::async, [this, symbol, depth]() {
        OrderBook order_book;
        order_book.symbol = symbol;
        order_book.timestamp = std::chrono::system_clock::now();
        
        // Alpha Vantage doesn't provide order book data
        // Return empty order book with current price
        try {
            auto price_future = get_current_price(symbol);
            double current_price = price_future.get();
            
            // Create synthetic order book around current price
            double spread = current_price * 0.001; // 0.1% spread
            
            // Add bids
            for (int i = 0; i < depth / 2; ++i) {
                OrderBookEntry bid;
                bid.price = current_price - (spread * (i + 1));
                bid.quantity = 100.0 / (i + 1); // Decreasing size
                order_book.bids.push_back(bid);
            }
            
            // Add asks
            for (int i = 0; i < depth / 2; ++i) {
                OrderBookEntry ask;
                ask.price = current_price + (spread * (i + 1));
                ask.quantity = 100.0 / (i + 1); // Decreasing size
                order_book.asks.push_back(ask);
            }
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get order book: {}", e.what());
        }
        
        return order_book;
    });
}

std::future<std::vector<std::string>> AlphaVantageProvider::search_symbols(const std::string& keywords) {
    return std::async(std::launch::async, [this, keywords]() {
        std::vector<std::string> symbols;
        
        try {
            std::map<std::string, std::string> params = {
                {"function", AlphaVantageEndpoints::SYMBOL_SEARCH},
                {"keywords", keywords},
                {"apikey", api_key_}
            };
            
            std::string response = make_api_request(
                AlphaVantageEndpoints::SYMBOL_SEARCH, params);
            
            auto json_data = json::parse(response);
            if (json_data.contains("bestMatches")) {
                for (const auto& match : json_data["bestMatches"]) {
                    symbols.push_back(match["1. symbol"].get<std::string>());
                }
            }
            
            LOG_INFO("Found {} symbols matching '{}'", symbols.size(), keywords);
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to search symbols: {}", e.what());
        }
        
        return symbols;
    });
}

std::future<MarketStatus> AlphaVantageProvider::get_market_status(const std::string& market) {
    return std::async(std::launch::async, [this, market]() {
        MarketStatus status;
        status.market = market.empty() ? "US" : market;
        status.timestamp = std::chrono::system_clock::now();
        status.is_open = false; // Default to closed
        
        try {
            std::map<std::string, std::string> params = {
                {"function", AlphaVantageEndpoints::MARKET_STATUS},
                {"apikey", api_key_}
            };
            
            if (!market.empty()) {
                params["market"] = market;
            }
            
            std::string response = make_api_request(
                AlphaVantageEndpoints::MARKET_STATUS, params);
            
            auto json_data = json::parse(response);
            if (json_data.contains("market_status")) {
                auto market_data = json_data["market_status"];
                status.is_open = market_data["current_session"]["open"].get<bool>();
                status.next_open_time = market_data["current_session"]["open"].get<std::string>();
                status.next_close_time = market_data["current_session"]["close"].get<std::string>();
            }
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get market status: {}", e.what());
        }
        
        return status;
    });
}

std::future<std::map<std::string, double>> AlphaVantageProvider::get_global_quotes(
    const std::vector<std::string>& symbols
) {
    return std::async(std::launch::async, [this, symbols]() {
        std::map<std::string, double> quotes;
        
        try {
            for (const auto& symbol : symbols) {
                auto price_future = get_current_price(symbol);
                double price = price_future.get();
                
                if (price > 0.0) {
                    quotes[symbol] = price;
                }
                
                // Small delay to respect rate limits
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            
            LOG_INFO("Retrieved quotes for {} symbols", quotes.size());
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get global quotes: {}", e.what());
        }
        
        return quotes;
    });
}

// Private methods implementation

bool AlphaVantageProvider::test_api_connection() {
    try {
        std::map<std::string, std::string> params = {
            {"function", AlphaVantageEndpoints::GLOBAL_QUOTE},
            {"symbol", "AAPL"},
            {"apikey", api_key_}
        };
        
        std::string response = make_api_request(
            AlphaVantageEndpoints::GLOBAL_QUOTE, params);
        
        auto json_data = json::parse(response);
        return json_data.contains("Global Quote");
        
    } catch (const std::exception& e) {
        LOG_ERROR("API connection test failed: {}", e.what());
        return false;
    }
}

std::string AlphaVantageProvider::make_api_request(
    const std::string& function,
    const std::map<std::string, std::string>& params
) {
    enforce_rate_limit();
    
    std::string url = AlphaVantageEndpoints::BASE_URL + build_query_string(params);
    std::string response;
    
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
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

std::string AlphaVantageProvider::build_query_string(
    const std::map<std::string, std::string>& params
) {
    std::stringstream ss;
    bool first = true;
    
    for (const auto& [key, value] : params) {
        if (!first) {
            ss << "&";
        }
        ss << key << "=" << value;
        first = false;
    }
    
    return ss.str();
}

void AlphaVantageProvider::enforce_rate_limit() {
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

std::vector<OHLCV> AlphaVantageProvider::parse_time_series_response(
    const std::string& response,
    const std::string& symbol
) {
    std::vector<OHLCV> data;
    
    auto json_data = json::parse(response);
    
    // Find the time series key (varies by function)
    std::string time_series_key;
    for (const auto& [key, value] : json_data.items()) {
        if (key.find("Time Series") != std::string::npos) {
            time_series_key = key;
            break;
        }
    }
    
    if (time_series_key.empty()) {
        throw std::runtime_error("No time series data found in response");
    }
    
    auto time_series = json_data[time_series_key];
    
    for (const auto& [date_str, ohlcv_data] : time_series.items()) {
        OHLCV ohlcv;
        ohlcv.symbol = symbol;
        ohlcv.timestamp = parse_alpha_vantage_timestamp(date_str);
        ohlcv.open = std::stod(ohlcv_data["1. open"].get<std::string>());
        ohlcv.high = std::stod(ohlcv_data["2. high"].get<std::string>());
        ohlcv.low = std::stod(ohlcv_data["3. low"].get<std::string>());
        ohlcv.close = std::stod(ohlcv_data["4. close"].get<std::string>());
        ohlcv.volume = std::stod(ohlcv_data["5. volume"].get<std::string>());
        
        data.push_back(ohlcv);
    }
    
    // Sort by timestamp (newest first)
    std::sort(data.begin(), data.end(), 
        [](const OHLCV& a, const OHLCV& b) {
            return a.timestamp > b.timestamp;
        });
    
    return data;
}

std::string AlphaVantageProvider::convert_timeframe_to_function(const std::string& timeframe) {
    // Map timeframes to Alpha Vantage functions
    if (timeframe == "1m" || timeframe == "5m" || timeframe == "15m" || timeframe == "30m") {
        return AlphaVantageEndpoints::TIME_SERIES_INTRADAY;
    } else if (timeframe == "1h" || timeframe == "4h") {
        return AlphaVantageEndpoints::TIME_SERIES_INTRADAY;
    } else if (timeframe == "1d") {
        return AlphaVantageEndpoints::TIME_SERIES_DAILY;
    } else if (timeframe == "1w") {
        return AlphaVantageEndpoints::TIME_SERIES_WEEKLY;
    } else if (timeframe == "1M") {
        return AlphaVantageEndpoints::TIME_SERIES_MONTHLY;
    }
    
    // Default to daily
    return AlphaVantageEndpoints::TIME_SERIES_DAILY;
}

std::chrono::system_clock::time_point AlphaVantageProvider::parse_alpha_vantage_timestamp(
    const std::string& timestamp
) {
    std::tm tm = {};
    std::istringstream ss(timestamp);
    ss >> std::get_time(&tm, "%Y-%m-%d");
    
    return std::chrono::system_clock::from_time_t(std::mktime(&tm));
}

size_t AlphaVantageProvider::WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

} // namespace Data
} // namespace ArchNeuronX
