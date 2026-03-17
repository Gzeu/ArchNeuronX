/**
 * @file kraken_provider.cpp
 * @brief Kraken API data provider implementation
 * @author George Pricop
 * @date 2025-10-02
 */

#include "data/kraken_provider.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace ArchNeuronX {
namespace Data {

KrakenProvider::KrakenProvider(const DataProviderConfig& config) 
    : DataProvider(config), connected_(false), requests_per_second_(0) {
    
    api_key_ = config.api_key;
    api_secret_ = config.api_secret;
    
    // Initialize CURL
    curl_ = curl_easy_init();
    if (!curl_) {
        throw std::runtime_error("Failed to initialize CURL");
    }
    
    // Setup WebSocket client
    ws_client_.init_asio();
    
    last_request_time_ = std::chrono::steady_clock::now();
    status_ = ConnectionStatus::DISCONNECTED;
    
    // Initialize symbol mapping
    initialize_symbol_mapping();
}

KrakenProvider::~KrakenProvider() {
    disconnect();
    if (curl_) {
        curl_easy_cleanup(curl_);
    }
}

bool KrakenProvider::connect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    if (connected_) {
        return true;
    }
    
    try {
        // Test REST API connection
        if (!test_rest_connection()) {
            status_ = ConnectionStatus::ERROR;
            return false;
        }
        
        // Setup WebSocket connection
        setup_websocket_connection();
        
        connected_ = true;
        status_ = ConnectionStatus::CONNECTED;
        requests_per_second_ = 0;
        
        LOG_INFO("Kraken provider connected successfully");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Kraken connection failed: {}", e.what());
        status_ = ConnectionStatus::ERROR;
        return false;
    }
}

void KrakenProvider::disconnect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    if (!connected_) {
        return;
    }
    
    // Close WebSocket connections
    for (auto& [pair, hdl] : ws_connections_) {
        ws_client_.close(hdl, websocketpp::close::status::normal, "Disconnecting");
    }
    ws_connections_.clear();
    
    // Stop WebSocket client
    ws_client_.stop();
    
    if (ws_thread_.joinable()) {
        ws_thread_.join();
    }
    
    connected_ = false;
    status_ = ConnectionStatus::DISCONNECTED;
    
    LOG_INFO("Kraken provider disconnected");
}

bool KrakenProvider::is_connected() const {
    return connected_.load();
}

ConnectionStatus KrakenProvider::get_status() const {
    return status_.load();
}

std::future<std::vector<OHLCV>> KrakenProvider::get_historical_data(
    const std::string& symbol,
    const std::string& timeframe,
    const std::chrono::system_clock::time_point& start,
    const std::chrono::system_clock::time_point& end
) {
    return std::async(std::launch::async, [this, symbol, timeframe, start, end]() {
        std::vector<OHLCV> data;
        
        try {
            std::string kraken_pair = convert_symbol_to_kraken(symbol);
            std::string interval = convert_timeframe_to_interval(timeframe);
            
            // Kraken OHLC endpoint
            std::map<std::string, std::string> params = {
                {"pair", kraken_pair},
                {"interval", interval}
            };
            
            std::string response = make_rest_request(KrakenEndpoints::OHLC, "GET", params);
            data = parse_ohlc_response(response, symbol);
            
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

std::future<std::vector<TickData>> KrakenProvider::get_real_time_ticks(
    const std::string& symbol,
    std::function<void(const TickData&)> callback
) {
    return std::async(std::launch::async, [this, symbol, callback]() {
        std::vector<TickData> ticks;
        
        try {
            std::string kraken_pair = convert_symbol_to_kraken(symbol);
            
            // Subscribe to WebSocket ticker and trades
            subscribe_to_ticker(kraken_pair);
            subscribe_to_trades(kraken_pair);
            
            // Keep connection alive
            while (connected_) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to setup real-time ticks: {}", e.what());
        }
        
        return ticks;
    });
}

std::future<double> KrakenProvider::get_current_price(const std::string& symbol) {
    return std::async(std::launch::async, [this, symbol]() {
        try {
            std::string kraken_pair = convert_symbol_to_kraken(symbol);
            
            std::map<std::string, std::string> params = {
                {"pair", kraken_pair}
            };
            
            std::string response = make_rest_request(KrakenEndpoints::TICKER, "GET", params);
            double price = parse_ticker_response(response, kraken_pair);
            
            return price;
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get current price: {}", e.what());
            return 0.0;
        }
    });
}

std::future<OrderBook> KrakenProvider::get_order_book(const std::string& symbol, int depth) {
    return std::async(std::launch::async, [this, symbol, depth]() {
        OrderBook order_book;
        order_book.symbol = symbol;
        order_book.timestamp = std::chrono::system_clock::now();
        
        try {
            std::string kraken_pair = convert_symbol_to_kraken(symbol);
            
            std::map<std::string, std::string> params = {
                {"pair", kraken_pair},
                {"count", std::to_string(depth)}
            };
            
            std::string response = make_rest_request(KrakenEndpoints::DEPTH, "GET", params);
            order_book = parse_depth_response(response, symbol);
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get order book: {}", e.what());
        }
        
        return order_book;
    });
}

std::future<std::vector<KrakenAssetPair>> KrakenProvider::get_asset_pairs() {
    return std::async(std::launch::async, [this]() {
        std::vector<KrakenAssetPair> pairs;
        
        try {
            std::string response = make_rest_request(KrakenEndpoints::ASSET_PAIRS);
            auto json_data = json::parse(response);
            
            if (json_data.contains("result")) {
                auto result = json_data["result"];
                
                for (const auto& [pair_name, pair_info] : result.items()) {
                    KrakenAssetPair pair;
                    pair.name = pair_name;
                    pair.altname = pair_info["altname"].get<std::string>();
                    pair.base = pair_info["base"].get<std::string>();
                    pair.quote = pair_info["quote"].get<std::string>();
                    pair.wsname = pair_info["wsname"].get<std::string>();
                    pair.tradable = pair_info["tradable"].get<bool>();
                    pair.marginable = pair_info["marginable"].get<bool>();
                    
                    pairs.push_back(pair);
                    
                    // Update symbol mapping
                    std::string symbol = pair.base + "/" + pair.quote;
                    symbol_to_kraken_[symbol] = pair_name;
                    kraken_to_symbol_[pair_name] = symbol;
                }
            }
            
            LOG_INFO("Retrieved {} asset pairs from Kraken", pairs.size());
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get asset pairs: {}", e.what());
        }
        
        return pairs;
    });
}

std::future<std::map<std::string, double>> KrakenProvider::get_tickers(
    const std::vector<std::string>& pairs
) {
    return std::async(std::launch::async, [this, pairs]() {
        std::map<std::string, double> tickers;
        
        try {
            for (const auto& pair : pairs) {
                std::string kraken_pair = convert_symbol_to_kraken(pair);
                
                std::map<std::string, std::string> params = {
                    {"pair", kraken_pair}
                };
                
                std::string response = make_rest_request(KrakenEndpoints::TICKER, "GET", params);
                double price = parse_ticker_response(response, kraken_pair);
                
                if (price > 0.0) {
                    tickers[pair] = price;
                }
                
                // Small delay to respect rate limits
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            LOG_INFO("Retrieved tickers for {} pairs", tickers.size());
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get tickers: {}", e.what());
        }
        
        return tickers;
    });
}

std::future<std::chrono::system_clock::time_point> KrakenProvider::get_server_time() {
    return std::async(std::launch::async, [this]() {
        try {
            std::string response = make_rest_request(KrakenEndpoints::SERVER_TIME);
            auto json_data = json::parse(response);
            
            if (json_data.contains("result")) {
                auto result = json_data["result"];
                if (result.contains("unixtime")) {
                    auto unix_time = result["unixtime"].get<long>();
                    return std::chrono::system_clock::from_time_t(unix_time);
                }
            }
            
            return std::chrono::system_clock::now();
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get server time: {}", e.what());
            return std::chrono::system_clock::now();
        }
    });
}

std::future<std::vector<std::string>> KrakenProvider::get_assets() {
    return std::async(std::launch::async, [this]() {
        std::vector<std::string> assets;
        
        try {
            std::string response = make_rest_request(KrakenEndpoints::ASSETS);
            auto json_data = json::parse(response);
            
            if (json_data.contains("result")) {
                auto result = json_data["result"];
                
                for (const auto& [asset_name, asset_info] : result.items()) {
                    assets.push_back(asset_name);
                }
            }
            
            LOG_INFO("Retrieved {} assets from Kraken", assets.size());
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get assets: {}", e.what());
        }
        
        return assets;
    });
}

// Private methods implementation

bool KrakenProvider::test_rest_connection() {
    try {
        std::string response = make_rest_request(KrakenEndpoints::SERVER_TIME);
        auto json_data = json::parse(response);
        return json_data.contains("result");
        
    } catch (const std::exception& e) {
        LOG_ERROR("REST connection test failed: {}", e.what());
        return false;
    }
}

std::string KrakenProvider::make_rest_request(
    const std::string& endpoint,
    const std::string& method,
    const std::map<std::string, std::string>& params
) {
    enforce_rate_limit();
    
    std::string url = KrakenEndpoints::REST_BASE + endpoint;
    std::string response;
    
    if (method == "GET" && !params.empty()) {
        url += "?";
        bool first = true;
        for (const auto& [key, value] : params) {
            if (!first) url += "&";
            url += key + "=" + value;
            first = false;
        }
    }
    
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, 30L);
    
    // Add headers for private endpoints
    struct curl_slist* headers = nullptr;
    if (endpoint.find("/private/") != std::string::npos) {
        std::string signature = generate_signature(endpoint, params);
        headers = curl_slist_append(headers, ("API-Key: " + api_key_).c_str());
        headers = curl_slist_append(headers, ("API-Sign: " + signature).c_str());
        headers = curl_slist_append(headers, "Content-Type: application/x-www-form-urlencoded");
    }
    
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
    
    CURLcode res = curl_easy_perform(curl_);
    curl_slist_free_all(headers);
    
    if (res != CURLE_OK) {
        throw std::runtime_error("CURL request failed: " + std::string(curl_easy_strerror(res)));
    }
    
    long response_code;
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &response_code);
    if (response_code != 200) {
        throw std::runtime_error("HTTP error: " + std::to_string(response_code));
    }
    
    // Update request count
    requests_per_second_++;
    
    return response;
}

std::string KrakenProvider::generate_signature(
    const std::string& endpoint,
    std::map<std::string, std::string>& params
) {
    // Add nonce to parameters
    auto now = std::chrono::system_clock::now();
    auto nonce = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count());
    
    params["nonce"] = nonce;
    
    // Create post data
    std::string post_data;
    for (const auto& [key, value] : params) {
        if (!post_data.empty()) post_data += "&";
        post_data += key + "=" + value;
    }
    
    // Generate signature
    std::string message = post_data;
    unsigned char* digest = HMAC(EVP_sha512(), 
                               api_secret_.c_str(), api_secret_.length(),
                               (unsigned char*)message.c_str(), message.length(),
                               nullptr, nullptr);
    
    std::stringstream ss;
    for (int i = 0; i < 64; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)digest[i];
    }
    
    return ss.str();
}

void KrakenProvider::enforce_rate_limit() {
    auto now = std::chrono::steady_clock::now();
    auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_request_time_).count();
    
    // Reset counter if more than a second has passed
    if (time_since_last >= 1000) {
        requests_per_second_ = 0;
        last_request_time_ = now;
        return;
    }
    
    // If we've hit the limit, wait
    if (requests_per_second_ >= MAX_REQUESTS_PER_SECOND) {
        int wait_time = 1000 - time_since_last;
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));
        requests_per_second_ = 0;
        last_request_time_ = std::chrono::steady_clock::now();
    }
    
    last_request_time_ = now;
}

void KrakenProvider::setup_websocket_connection() {
    ws_thread_ = std::thread([this]() {
        auto con = ws_client_.get_connection(KrakenEndpoints::WS_PUBLIC);
        
        con->set_open_handler([this](websocketpp::connection_hdl hdl) {
            LOG_INFO("Kraken WebSocket connection opened");
        });
        
        con->set_message_handler([this](websocketpp::connection_hdl hdl, 
                                           websocketpp::client::message<websocketpp::config::asio_client>::type::ptr msg) {
            try {
                auto json_data = json::parse(msg->get_payload());
                
                if (json_data.contains("event")) {
                    std::string event = json_data["event"];
                    
                    if (event == "subscriptionStatus") {
                        LOG_INFO("Subscription status: {}", json_data["status"]);
                    } else if (event == "heartbeat") {
                        // Handle heartbeat
                    }
                } else if (json_data.contains("channelID")) {
                    // Handle data messages
                    std::string channel = json_data["channelID"];
                    
                    if (channel.find("ticker") != std::string::npos) {
                        // Handle ticker message
                    } else if (channel.find("ohlc") != std::string::npos) {
                        // Handle OHLC message
                    } else if (channel.find("trade") != std::string::npos) {
                        // Handle trade message
                    } else if (channel.find("book") != std::string::npos) {
                        // Handle order book message
                    }
                }
            } catch (const std::exception& e) {
                LOG_ERROR("Error parsing WebSocket message: {}", e.what());
            }
        });
        
        con->set_fail_handler([this](websocketpp::connection_hdl hdl) {
            LOG_ERROR("Kraken WebSocket connection failed");
            status_ = ConnectionStatus::ERROR;
        });
        
        ws_client_.connect(con);
        ws_client_.run();
    });
}

std::vector<OHLCV> KrakenProvider::parse_ohlc_response(
    const std::string& response,
    const std::string& pair
) {
    std::vector<OHLCV> data;
    
    auto json_data = json::parse(response);
    
    if (json_data.contains("result")) {
        auto result = json_data["result"];
        
        // Kraken returns data with pair name as key
        if (result.contains(pair)) {
            auto ohlc_data = result[pair];
            
            for (const auto& candle : ohlc_data) {
                OHLCV ohlcv;
                ohlcv.symbol = convert_kraken_to_symbol(pair);
                ohlcv.timestamp = std::chrono::system_clock::from_time_t(candle[0].get<long>());
                ohlcv.open = candle[1].get<double>();
                ohlcv.high = candle[2].get<double>();
                ohlcv.low = candle[3].get<double>();
                ohlcv.close = candle[4].get<double>();
                ohlcv.volume = candle[6].get<double>(); // Volume weighted average price
                ohlcv.trades_count = candle[7].get<int>(); // Number of trades
                
                data.push_back(ohlcv);
            }
        }
    }
    
    // Sort by timestamp (newest first)
    std::sort(data.begin(), data.end(), 
        [](const OHLCV& a, const OHLCV& b) {
            return a.timestamp > b.timestamp;
        });
    
    return data;
}

OrderBook KrakenProvider::parse_depth_response(
    const std::string& response,
    const std::string& pair
) {
    OrderBook order_book;
    order_book.symbol = convert_kraken_to_symbol(pair);
    order_book.timestamp = std::chrono::system_clock::now();
    
    auto json_data = json::parse(response);
    
    if (json_data.contains("result")) {
        auto result = json_data["result"];
        
        if (result.contains(pair)) {
            auto depth_data = result[pair];
            
            // Parse bids
            if (depth_data.contains("bids")) {
                for (const auto& bid : depth_data["bids"]) {
                    OrderBookEntry entry;
                    entry.price = bid[0].get<double>();
                    entry.quantity = bid[1].get<double>();
                    order_book.bids.push_back(entry);
                }
            }
            
            // Parse asks
            if (depth_data.contains("asks")) {
                for (const auto& ask : depth_data["asks"]) {
                    OrderBookEntry entry;
                    entry.price = ask[0].get<double>();
                    entry.quantity = ask[1].get<double>();
                    order_book.asks.push_back(entry);
                }
            }
        }
    }
    
    return order_book;
}

double KrakenProvider::parse_ticker_response(
    const std::string& response,
    const std::string& pair
) {
    auto json_data = json::parse(response);
    
    if (json_data.contains("result")) {
        auto result = json_data["result"];
        
        if (result.contains(pair)) {
            auto ticker_data = result[pair];
            
            // Try to get closing price (c[0]), fallback to last trade price (l[0])
            if (ticker_data.contains("c") && ticker_data["c"].is_array() && ticker_data["c"].size() > 0) {
                return ticker_data["c"][0].get<double>();
            } else if (ticker_data.contains("l") && ticker_data["l"].is_array() && ticker_data["l"].size() > 0) {
                return ticker_data["l"][0].get<double>();
            }
        }
    }
    
    return 0.0;
}

std::string KrakenProvider::convert_symbol_to_kraken(const std::string& symbol) {
    auto it = symbol_to_kraken_.find(symbol);
    if (it != symbol_to_kraken_.end()) {
        return it->second;
    }
    
    // Fallback: try to construct from symbol
    if (symbol.find("/") != std::string::npos) {
        return symbol; // Assume it's already in Kraken format
    }
    
    return symbol; // Return as-is if no mapping found
}

std::string KrakenProvider::convert_kraken_to_symbol(const std::string& kraken_symbol) {
    auto it = kraken_to_symbol_.find(kraken_symbol);
    if (it != kraken_to_symbol_.end()) {
        return it->second;
    }
    
    return kraken_symbol; // Return as-is if no mapping found
}

std::string KrakenProvider::convert_timeframe_to_interval(const std::string& timeframe) {
    // Map timeframes to Kraken intervals
    if (timeframe == "1m") return "1";
    if (timeframe == "5m") return "5";
    if (timeframe == "15m") return "15";
    if (timeframe == "30m") return "30";
    if (timeframe == "1h") return "60";
    if (timeframe == "4h") return "240";
    if (timeframe == "1d") return "1440";
    if (timeframe == "1w") return "10080";
    
    // Default to 1 hour
    return "60";
}

void KrakenProvider::initialize_symbol_mapping() {
    // Common Kraken symbol mappings
    symbol_to_kraken_["BTC/USD"] = "XBTUSD";
    kraken_to_symbol_["XBTUSD"] = "BTC/USD";
    
    symbol_to_kraken_["BTC/EUR"] = "XBTEUR";
    kraken_to_symbol_["XBTEUR"] = "BTC/EUR";
    
    symbol_to_kraken_["ETH/USD"] = "ETHUSD";
    kraken_to_symbol_["ETHUSD"] = "ETH/USD";
    
    symbol_to_kraken_["ETH/EUR"] = "ETHEUR";
    kraken_to_symbol_["ETHEUR"] = "ETH/EUR";
    
    symbol_to_kraken_["LTC/USD"] = "LTCUSD";
    kraken_to_symbol_["LTCUSD"] = "LTC/USD";
    
    symbol_to_kraken_["XRP/USD"] = "XRPUSD";
    kraken_to_symbol_["XRPUSD"] = "XRP/USD";
    
    // Add more mappings as needed
}

size_t KrakenProvider::WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

} // namespace Data
} // namespace ArchNeuronX
