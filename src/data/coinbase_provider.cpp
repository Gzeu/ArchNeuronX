/**
 * @file coinbase_provider.cpp
 * @brief Coinbase Pro API data provider implementation
 * @author George Pricop
 * @date 2025-10-02
 */

#include "data/coinbase_provider.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace ArchNeuronX {
namespace Data {

CoinbaseProvider::CoinbaseProvider(const DataProviderConfig& config) 
    : DataProvider(config), connected_(false), requests_per_second_(0) {
    
    api_key_ = config.api_key;
    api_secret_ = config.api_secret;
    passphrase_ = config.passphrase;
    
    if (api_key_.empty() || api_secret_.empty()) {
        throw std::invalid_argument("Coinbase API key and secret are required");
    }
    
    // Initialize CURL
    curl_ = curl_easy_init();
    if (!curl_) {
        throw std::runtime_error("Failed to initialize CURL");
    }
    
    // Setup WebSocket client
    ws_client_.init_asio();
    ws_client_.set_tls_init_handler([](websocketpp::lib::asio::ssl::context& ctx) {
        ctx.set_options(websocketpp::lib::asio::ssl::context::default_workarounds);
    });
    
    last_request_time_ = std::chrono::steady_clock::now();
    status_ = ConnectionStatus::DISCONNECTED;
}

CoinbaseProvider::~CoinbaseProvider() {
    disconnect();
    if (curl_) {
        curl_easy_cleanup(curl_);
    }
}

bool CoinbaseProvider::connect() {
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
        
        LOG_INFO("Coinbase provider connected successfully");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Coinbase connection failed: {}", e.what());
        status_ = ConnectionStatus::ERROR;
        return false;
    }
}

void CoinbaseProvider::disconnect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    if (!connected_) {
        return;
    }
    
    // Close WebSocket connections
    for (auto& [symbol, hdl] : ws_connections_) {
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
    
    LOG_INFO("Coinbase provider disconnected");
}

bool CoinbaseProvider::is_connected() const {
    return connected_.load();
}

ConnectionStatus CoinbaseProvider::get_status() const {
    return status_.load();
}

std::future<std::vector<OHLCV>> CoinbaseProvider::get_historical_data(
    const std::string& symbol,
    const std::string& timeframe,
    const std::chrono::system_clock::time_point& start,
    const std::chrono::system_clock::time_point& end
) {
    return std::async(std::launch::async, [this, symbol, timeframe, start, end]() {
        std::vector<OHLCV> data;
        
        try {
            std::string coinbase_symbol = format_coinbase_symbol(symbol);
            std::string granularity = convert_timeframe_to_granularity(timeframe);
            
            // Coinbase candles endpoint returns last 300 candles
            std::string endpoint = std::string(CoinbaseEndpoints::CANDLES)
                                    .replace(coinbase_symbol.find("{}"), 2, coinbase_symbol)
                                    .replace(granularity.find("{}"), 2, granularity);
            
            std::string response = make_rest_request(endpoint);
            data = parse_candles_response(response, symbol);
            
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

std::future<std::vector<TickData>> CoinbaseProvider::get_real_time_ticks(
    const std::string& symbol,
    std::function<void(const TickData&)> callback
) {
    return std::async(std::launch::async, [this, symbol, callback]() {
        std::vector<TickData> ticks;
        
        try {
            std::string coinbase_symbol = format_coinbase_symbol(symbol);
            
            // Subscribe to WebSocket ticker and trades
            subscribe_to_ticker(coinbase_symbol);
            subscribe_to_trades(coinbase_symbol);
            
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

std::future<double> CoinbaseProvider::get_current_price(const std::string& symbol) {
    return std::async(std::launch::async, [this, symbol]() {
        try {
            std::string coinbase_symbol = format_coinbase_symbol(symbol);
            std::string endpoint = std::string(CoinbaseEndpoints::TICKER)
                                    .replace(coinbase_symbol.find("{}"), 2, coinbase_symbol);
            
            std::string response = make_rest_request(endpoint);
            auto json_data = json::parse(response);
            
            return json_data["price"].get<double>();
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get current price: {}", e.what());
            return 0.0;
        }
    });
}

std::future<OrderBook> CoinbaseProvider::get_order_book(const std::string& symbol, int depth) {
    return std::async(std::launch::async, [this, symbol, depth]() {
        OrderBook order_book;
        order_book.symbol = symbol;
        order_book.timestamp = std::chrono::system_clock::now();
        
        try {
            std::string coinbase_symbol = format_coinbase_symbol(symbol);
            std::string level = depth == 1 ? "1" : "2";
            std::string endpoint = std::string(CoinbaseEndpoints::ORDER_BOOK)
                                    .replace(coinbase_symbol.find("{}"), 2, coinbase_symbol)
                                    .replace(level.find("{}"), 2, level);
            
            std::string response = make_rest_request(endpoint);
            order_book = parse_order_book_response(response, symbol);
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get order book: {}", e.what());
        }
        
        return order_book;
    });
}

std::future<std::vector<CoinbaseProduct>> CoinbaseProvider::get_products() {
    return std::async(std::launch::async, [this]() {
        std::vector<CoinbaseProduct> products;
        
        try {
            std::string response = make_rest_request(CoinbaseEndpoints::PRODUCTS);
            auto json_data = json::parse(response);
            
            for (const auto& product_json : json_data) {
                CoinbaseProduct product;
                product.id = product_json["id"].get<std::string>();
                product.base_currency = product_json["base_currency"].get<std::string>();
                product.quote_currency = product_json["quote_currency"].get<std::string>();
                product.display_name = product_json["display_name"].get<std::string>();
                product.base_min_size = product_json["base_min_size"].get<double>();
                product.base_max_size = product_json["base_max_size"].get<double>();
                product.quote_increment = product_json["quote_increment"].get<double>();
                product.status_online = product_json["status"].get<std::string>() == "online";
                
                products.push_back(product);
            }
            
            LOG_INFO("Retrieved {} products from Coinbase", products.size());
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get products: {}", e.what());
        }
        
        return products;
    });
}

std::future<std::map<std::string, double>> CoinbaseProvider::get_tickers(
    const std::vector<std::string>& symbols
) {
    return std::async(std::launch::async, [this, symbols]() {
        std::map<std::string, double> tickers;
        
        try {
            for (const auto& symbol : symbols) {
                std::string coinbase_symbol = format_coinbase_symbol(symbol);
                std::string endpoint = std::string(CoinbaseEndpoints::TICKER)
                                        .replace(coinbase_symbol.find("{}"), 2, coinbase_symbol);
                
                std::string response = make_rest_request(endpoint);
                auto json_data = json::parse(response);
                
                tickers[symbol] = json_data["price"].get<double>();
                
                // Small delay to respect rate limits
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            LOG_INFO("Retrieved tickers for {} symbols", tickers.size());
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get tickers: {}", e.what());
        }
        
        return tickers;
    });
}

std::future<std::vector<std::string>> CoinbaseProvider::get_available_currencies() {
    return std::async(std::launch::async, [this]() {
        std::vector<std::string> currencies;
        
        try {
            std::string response = make_rest_request(CoinbaseEndpoints::CURRENCIES);
            auto json_data = json::parse(response);
            
            for (const auto& currency : json_data) {
                if (currency["status"].get<std::string>() == "online") {
                    currencies.push_back(currency["id"].get<std::string>());
                }
            }
            
            LOG_INFO("Retrieved {} available currencies", currencies.size());
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get currencies: {}", e.what());
        }
        
        return currencies;
    });
}

// Private methods implementation

bool CoinbaseProvider::test_rest_connection() {
    try {
        std::string response = make_rest_request(CoinbaseEndpoints::PRODUCTS);
        auto json_data = json::parse(response);
        return json_data.is_array() && json_data.size() > 0;
        
    } catch (const std::exception& e) {
        LOG_ERROR("REST connection test failed: {}", e.what());
        return false;
    }
}

std::string CoinbaseProvider::make_rest_request(
    const std::string& endpoint,
    const std::string& method,
    const std::string& body
) {
    enforce_rate_limit();
    
    std::string url = CoinbaseEndpoints::REST_BASE + endpoint;
    std::string response;
    
    // Generate timestamp and signature
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count());
    
    std::string signature = generate_signature(timestamp, method, endpoint, body);
    
    // Set headers
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("CB-ACCESS-KEY: " + api_key_).c_str());
    headers = curl_slist_append(headers, ("CB-ACCESS-SIGN: " + signature).c_str());
    headers = curl_slist_append(headers, ("CB-ACCESS-TIMESTAMP: " + timestamp).c_str());
    headers = curl_slist_append(headers, ("CB-ACCESS-PASSPHRASE: " + passphrase_).c_str());
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, method.c_str());
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, 30L);
    
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

std::string CoinbaseProvider::generate_signature(
    const std::string& timestamp,
    const std::string& method,
    const std::string& request_path,
    const std::string& body
) {
    std::string message = timestamp + method + request_path + body;
    
    unsigned char* digest = HMAC(EVP_sha256(), 
                               api_secret_.c_str(), api_secret_.length(),
                               (unsigned char*)message.c_str(), message.length(),
                               nullptr, nullptr);
    
    std::stringstream ss;
    for (int i = 0; i < 32; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)digest[i];
    }
    
    return ss.str();
}

void CoinbaseProvider::enforce_rate_limit() {
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

void CoinbaseProvider::setup_websocket_connection() {
    ws_thread_ = std::thread([this]() {
        auto con = ws_client_.get_connection(CoinbaseEndpoints::WS_BASE);
        
        con->set_open_handler([this](websocketpp::connection_hdl hdl) {
            LOG_INFO("Coinbase WebSocket connection opened");
        });
        
        con->set_message_handler([this](websocketpp::connection_hdl hdl, 
                                           websocketpp::client::message<websocketpp::config::asio_client>::type::ptr msg) {
            try {
                auto json_data = json::parse(msg->get_payload());
                
                if (json_data.contains("type")) {
                    std::string type = json_data["type"];
                    
                    if (type == "ticker") {
                        // Handle ticker message
                    } else if (type == "l2update") {
                        // Handle order book update
                    } else if (type == "match") {
                        // Handle trade message
                    }
                }
            } catch (const std::exception& e) {
                LOG_ERROR("Error parsing WebSocket message: {}", e.what());
            }
        });
        
        con->set_fail_handler([this](websocketpp::connection_hdl hdl) {
            LOG_ERROR("Coinbase WebSocket connection failed");
            status_ = ConnectionStatus::ERROR;
        });
        
        ws_client_.connect(con);
        ws_client_.run();
    });
}

std::vector<OHLCV> CoinbaseProvider::parse_candles_response(
    const std::string& response,
    const std::string& symbol
) {
    std::vector<OHLCV> data;
    
    auto json_data = json::parse(response);
    
    for (const auto& candle : json_data) {
        OHLCV ohlcv;
        ohlcv.symbol = symbol;
        ohlcv.timestamp = std::chrono::system_clock::from_time_t(candle[0].get<long>());
        ohlcv.low = candle[1].get<double>();
        ohlcv.high = candle[2].get<double>();
        ohlcv.open = candle[3].get<double>();
        ohlcv.close = candle[4].get<double>();
        ohlcv.volume = candle[5].get<double>();
        
        data.push_back(ohlcv);
    }
    
    // Sort by timestamp (newest first)
    std::sort(data.begin(), data.end(), 
        [](const OHLCV& a, const OHLCV& b) {
            return a.timestamp > b.timestamp;
        });
    
    return data;
}

std::string CoinbaseProvider::convert_timeframe_to_granularity(const std::string& timeframe) {
    // Map timeframes to Coinbase granularities
    if (timeframe == "1m") return "60";
    if (timeframe == "5m") return "300";
    if (timeframe == "15m") return "900";
    if (timeframe == "1h") return "3600";
    if (timeframe == "6h") return "21600";
    if (timeframe == "1d") return "86400";
    
    // Default to 1 hour
    return "3600";
}

std::string CoinbaseProvider::format_coinbase_symbol(const std::string& symbol) {
    // Convert BTC/USD to BTC-USD format
    std::string formatted = symbol;
    std::replace(formatted.begin(), formatted.end(), '/', '-');
    return formatted;
}

size_t CoinbaseProvider::WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

} // namespace Data
} // namespace ArchNeuronX
