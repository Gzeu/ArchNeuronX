/**
 * @file binance_provider.cpp
 * @brief Binance API data provider implementation
 * @author George Pricop
 * @date 2025-10-02
 */

#include "data/binance_provider.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <nlohmann/json.hpp>
#include <openssl/hmac.h>
#include <openssl/sha.h>
#include <iomanip>

using json = nlohmann::json;

namespace ArchNeuronX {
namespace Data {

BinanceProvider::BinanceProvider(const DataProviderConfig& config) 
    : DataProvider(config), connected_(false), reconnect_attempts_(0) {
    
    api_base_ = config.use_testnet ? BinanceEndpoints::TESTNET_BASE : BinanceEndpoints::MAINNET;
    ws_base_ = config.use_testnet ? BinanceEndpoints::WS_TESTNET : BinanceEndpoints::WS_MAINNET;
    
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
}

BinanceProvider::~BinanceProvider() {
    disconnect();
    if (curl_) {
        curl_easy_cleanup(curl_);
    }
}

bool BinanceProvider::connect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    if (connected_) {
        return true;
    }
    
    try {
        // Test REST API connection
        if (!test_rest_connection()) {
            return false;
        }
        
        // Setup WebSocket connections
        setup_websocket_connections();
        
        connected_ = true;
        status_ = ConnectionStatus::CONNECTED;
        reconnect_attempts_ = 0;
        
        LOG_INFO("Binance provider connected successfully");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Binance connection failed: {}", e.what());
        status_ = ConnectionStatus::ERROR;
        return false;
    }
}

void BinanceProvider::disconnect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    if (!connected_) {
        return;
    }
    
    // Close WebSocket connections
    for (auto& [symbol, conn] : ws_connections_) {
        if (conn->get_state() == websocketpp::session::state::open) {
            conn->close(websocketpp::close::status::normal, "Disconnecting");
        }
    }
    ws_connections_.clear();
    
    // Stop asio
    ws_client_.stop();
    
    connected_ = false;
    status_ = ConnectionStatus::DISCONNECTED;
    
    LOG_INFO("Binance provider disconnected");
}

std::future<std::vector<OHLCV>> BinanceProvider::get_historical_data(
    const std::string& symbol,
    const std::string& timeframe,
    const std::chrono::system_clock::time_point& start,
    const std::chrono::system_clock::time_point& end
) {
    return std::async(std::launch::async, [this, symbol, timeframe, start, end]() {
        std::vector<OHLCV> data;
        
        try {
            // Convert timeframe to Binance interval
            std::string interval = convert_timeframe(timeframe);
            
            // Convert timestamps
            auto start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                start.time_since_epoch()).count();
            auto end_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                end.time_since_epoch()).count();
            
            // Build URL
            std::string url = api_base_ + std::string(BinanceEndpoints::KLINES) + 
                             "?symbol=" + symbol + 
                             "&interval=" + interval + 
                             "&startTime=" + std::to_string(start_ms) +
                             "&endTime=" + std::to_string(end_ms) +
                             "&limit=1000";
            
            // Make request
            std::string response = make_rest_request(url);
            auto json_data = json::parse(response);
            
            // Parse response
            for (const auto& kline : json_data) {
                OHLCV ohlcv;
                ohlcv.timestamp = std::chrono::system_clock::from_time_t(kline[0].get<long>() / 1000);
                ohlcv.open = kline[1].get<double>();
                ohlcv.high = kline[2].get<double>();
                ohlcv.low = kline[3].get<double>();
                ohlcv.close = kline[4].get<double>();
                ohlcv.volume = kline[5].get<double>();
                data.push_back(ohlcv);
            }
            
            LOG_INFO("Retrieved {} OHLCV records for {}", data.size(), symbol);
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get historical data: {}", e.what());
        }
        
        return data;
    });
}

std::future<std::vector<TickData>> BinanceProvider::get_real_time_ticks(
    const std::string& symbol,
    std::function<void(const TickData&)> callback
) {
    return std::async(std::launch::async, [this, symbol, callback]() {
        std::vector<TickData> ticks;
        
        try {
            // Subscribe to WebSocket stream
            std::string stream = symbol.lower() + "@trade";
            subscribe_websocket_stream(stream, [callback, symbol](const std::string& data) {
                try {
                    auto json_data = json::parse(data);
                    
                    if (json_data.contains("e") && json_data["e"] == "trade") {
                        TickData tick;
                        tick.symbol = symbol;
                        tick.price = json_data["p"].get<double>();
                        tick.quantity = json_data["q"].get<double>();
                        tick.timestamp = std::chrono::system_clock::from_time_t(
                            json_data["T"].get<long>() / 1000);
                        
                        if (callback) {
                            callback(tick);
                        }
                    }
                } catch (const std::exception& e) {
                    LOG_ERROR("Error parsing tick data: {}", e.what());
                }
            });
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to setup real-time ticks: {}", e.what());
        }
        
        return ticks;
    });
}

std::future<double> BinanceProvider::get_current_price(const std::string& symbol) {
    return std::async(std::launch::async, [this, symbol]() {
        try {
            std::string url = api_base_ + std::string(BinanceEndpoints::TICKER_PRICE) + 
                             "?symbol=" + symbol;
            
            std::string response = make_rest_request(url);
            auto json_data = json::parse(response);
            
            return json_data["price"].get<double>();
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get current price: {}", e.what());
            return 0.0;
        }
    });
}

std::future<OrderBook> BinanceProvider::get_order_book(const std::string& symbol, int depth) {
    return std::async(std::launch::async, [this, symbol, depth]() {
        OrderBook order_book;
        order_book.symbol = symbol;
        order_book.timestamp = std::chrono::system_clock::now();
        
        try {
            std::string url = api_base_ + std::string(BinanceEndpoints::ORDER_BOOK) + 
                             "?symbol=" + symbol + 
                             "&limit=" + std::to_string(depth);
            
            std::string response = make_rest_request(url);
            auto json_data = json::parse(response);
            
            // Parse bids
            for (const auto& bid : json_data["bids"]) {
                OrderBookEntry entry;
                entry.price = bid[0].get<double>();
                entry.quantity = bid[1].get<double>();
                order_book.bids.push_back(entry);
            }
            
            // Parse asks
            for (const auto& ask : json_data["asks"]) {
                OrderBookEntry entry;
                entry.price = ask[0].get<double>();
                entry.quantity = ask[1].get<double>();
                order_book.asks.push_back(entry);
            }
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get order book: {}", e.what());
        }
        
        return order_book;
    });
}

// Private methods implementation

bool BinanceProvider::test_rest_connection() {
    try {
        std::string url = api_base_ + std::string(BinanceEndpoints::EXCHANGE_INFO);
        std::string response = make_rest_request(url);
        
        auto json_data = json::parse(response);
        return json_data.contains("symbols");
        
    } catch (const std::exception& e) {
        LOG_ERROR("REST connection test failed: {}", e.what());
        return false;
    }
}

void BinanceProvider::setup_websocket_connections() {
    // Start WebSocket client in separate thread
    ws_thread_ = std::thread([this]() {
        ws_client_.run();
    });
}

std::string BinanceProvider::make_rest_request(const std::string& url) {
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
    
    return response;
}

std::string BinanceProvider::generate_signature(const std::string& query_string) {
    unsigned char* digest = HMAC(EVP_sha256(), 
                               config_.api_secret.c_str(), config_.api_secret.length(),
                               (unsigned char*)query_string.c_str(), query_string.length(),
                               nullptr, nullptr);
    
    std::stringstream ss;
    for (int i = 0; i < 32; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)digest[i];
    }
    
    return ss.str();
}

std::string BinanceProvider::convert_timeframe(const std::string& timeframe) {
    // Map common timeframes to Binance intervals
    if (timeframe == "1m") return "1m";
    if (timeframe == "5m") return "5m";
    if (timeframe == "15m") return "15m";
    if (timeframe == "1h") return "1h";
    if (timeframe == "4h") return "4h";
    if (timeframe == "1d") return "1d";
    
    // Default to 1h
    return "1h";
}

void BinanceProvider::subscribe_websocket_stream(
    const std::string& stream,
    std::function<void(const std::string&)> callback
) {
    std::string uri = ws_base_ + stream;
    
    auto con = ws_client_.get_connection(uri);
    
    con->set_open_handler([this](websocketpp::connection_hdl hdl) {
        LOG_INFO("WebSocket connection opened");
    });
    
    con->set_message_handler([callback](websocketpp::connection_hdl hdl, 
                                       websocketpp::client::message<websocketpp::config::asio_client>::type::ptr msg) {
        if (callback) {
            callback(msg->get_payload());
        }
    });
    
    con->set_fail_handler([this](websocketpp::connection_hdl hdl) {
        LOG_ERROR("WebSocket connection failed");
        status_ = ConnectionStatus::ERROR;
    });
    
    con->set_close_handler([this](websocketpp::connection_hdl hdl) {
        LOG_INFO("WebSocket connection closed");
        status_ = ConnectionStatus::DISCONNECTED;
    });
    
    ws_client_.connect(con);
}

size_t BinanceProvider::WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

void BinanceProvider::start_reconnect_timer() {
    if (reconnect_timer_ == nullptr) {
        reconnect_timer_ = std::make_unique<websocketpp::lib::asio::steady_timer>(
            ws_client_.get_io_service(), std::chrono::seconds(5));
    }
    
    reconnect_timer_->async_wait([this](const websocketpp::lib::error_code& ec) {
        if (!ec) {
            LOG_INFO("Attempting to reconnect...");
            if (connect()) {
                LOG_INFO("Reconnection successful");
            } else {
                reconnect_attempts_++;
                if (reconnect_attempts_ < max_reconnect_attempts_) {
                    start_reconnect_timer();
                }
            }
        }
    });
}

} // namespace Data
} // namespace ArchNeuronX
