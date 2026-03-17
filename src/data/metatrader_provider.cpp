/**
 * @file metatrader_provider.cpp
 * @brief MetaTrader 5 API data provider implementation
 * @author George Pricop
 * @date 2025-10-02
 */

#include "data/metatrader_provider.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace ArchNeuronX {
namespace Data {

MetaTraderProvider::MetaTraderProvider(const DataProviderConfig& config) 
    : DataProvider(config), connected_(false), requests_per_second_(0), account_id_(0) {
    
    api_key_ = config.api_key;
    user_id_ = config.user_id;
    account_id_ = config.account_id;
    
    if (api_key_.empty() || user_id_.empty()) {
        throw std::invalid_argument("MetaTrader API key and user ID are required");
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
    
    // Initialize symbol mapping
    initialize_symbol_mapping();
}

MetaTraderProvider::~MetaTraderProvider() {
    disconnect();
    if (curl_) {
        curl_easy_cleanup(curl_);
    }
}

bool MetaTraderProvider::connect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    if (connected_) {
        return true;
    }
    
    try {
        // Authenticate first
        if (!authenticate()) {
            status_ = ConnectionStatus::ERROR;
            return false;
        }
        
        // Test API connection
        if (!test_rest_connection()) {
            status_ = ConnectionStatus::ERROR;
            return false;
        }
        
        // Setup WebSocket connection
        setup_websocket_connection();
        
        connected_ = true;
        status_ = ConnectionStatus::CONNECTED;
        requests_per_second_ = 0;
        
        LOG_INFO("MetaTrader provider connected successfully");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("MetaTrader connection failed: {}", e.what());
        status_ = ConnectionStatus::ERROR;
        return false;
    }
}

void MetaTraderProvider::disconnect() {
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
    
    LOG_INFO("MetaTrader provider disconnected");
}

bool MetaTraderProvider::is_connected() const {
    return connected_.load();
}

ConnectionStatus MetaTraderProvider::get_status() const {
    return status_.load();
}

std::future<std::vector<OHLCV>> MetaTraderProvider::get_historical_data(
    const std::string& symbol,
    const std::string& timeframe,
    const std::chrono::system_clock::time_point& start,
    const std::chrono::system_clock::time_point& end
) {
    return std::async(std::launch::async, [this, symbol, timeframe, start, end]() {
        std::vector<OHLCV> data;
        
        try {
            std::string mt5_symbol = convert_symbol_to_mt5(symbol);
            std::string mt5_timeframe = convert_timeframe_to_mt5(timeframe);
            
            auto start_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                start.time_since_epoch()).count();
            auto end_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                end.time_since_epoch()).count();
            
            std::string body = json({
                {"symbol", mt5_symbol},
                {"timeframe", mt5_timeframe},
                {"from", start_ts},
                {"to", end_ts},
                {"count", 1000}
            }).dump();
            
            std::string response = make_rest_request(
                MetaTraderEndpoints::CANDLES, "POST", body);
            
            data = parse_candles_response(response, symbol);
            
            LOG_INFO("Retrieved {} OHLCV records for {} ({})", 
                     data.size(), symbol, timeframe);
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get historical data: {}", e.what());
        }
        
        return data;
    });
}

std::future<std::vector<TickData>> MetaTraderProvider::get_real_time_ticks(
    const std::string& symbol,
    std::function<void(const TickData&)> callback
) {
    return std::async(std::launch::async, [this, symbol, callback]() {
        std::vector<TickData> ticks;
        
        try {
            std::string mt5_symbol = convert_symbol_to_mt5(symbol);
            
            // Subscribe to WebSocket tick stream
            subscribe_to_ticks(mt5_symbol);
            
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

std::future<double> MetaTraderProvider::get_current_price(const std::string& symbol) {
    return std::async(std::launch::async, [this, symbol]() {
        try {
            std::string mt5_symbol = convert_symbol_to_mt5(symbol);
            
            std::string body = json({
                {"symbol", mt5_symbol}
            }).dump();
            
            std::string response = make_rest_request(
                MetaTraderEndpoints::TICK, "POST", body);
            
            auto order_book = parse_tick_response(response, symbol);
            
            // Return mid price from bid/ask
            if (!order_book.bids.empty() && !order_book.asks.empty()) {
                double bid = order_book.bids[0].price;
                double ask = order_book.asks[0].price;
                return (bid + ask) / 2.0;
            }
            
            return 0.0;
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get current price: {}", e.what());
            return 0.0;
        }
    });
}

std::future<OrderBook> MetaTraderProvider::get_order_book(const std::string& symbol, int depth) {
    return std::async(std::launch::async, [this, symbol, depth]() {
        OrderBook order_book;
        order_book.symbol = symbol;
        order_book.timestamp = std::chrono::system_clock::now();
        
        try {
            std::string mt5_symbol = convert_symbol_to_mt5(symbol);
            
            std::string body = json({
                {"symbol", mt5_symbol},
                {"depth", depth}
            }).dump();
            
            std::string response = make_rest_request(
                MetaTraderEndpoints::TICK, "POST", body);
            
            order_book = parse_tick_response(response, symbol);
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get order book: {}", e.what());
        }
        
        return order_book;
    });
}

std::future<std::vector<MT5Symbol>> MetaTraderProvider::get_symbols() {
    return std::async(std::launch::async, [this]() {
        std::vector<MT5Symbol> symbols;
        
        try {
            std::string response = make_rest_request(MetaTraderEndpoints::SYMBOLS);
            symbols = parse_symbols_response(response);
            
            LOG_INFO("Retrieved {} symbols from MetaTrader", symbols.size());
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get symbols: {}", e.what());
        }
        
        return symbols;
    });
}

std::future<std::vector<MT5Tick>> MetaTraderProvider::get_recent_ticks(
    const std::string& symbol, int count
) {
    return std::async(std::launch::async, [this, symbol, count]() {
        std::vector<MT5Tick> ticks;
        
        try {
            std::string mt5_symbol = convert_symbol_to_mt5(symbol);
            
            std::string body = json({
                {"symbol", mt5_symbol},
                {"count", count}
            }).dump();
            
            std::string response = make_rest_request(
                MetaTraderEndpoints::TICK, "POST", body);
            
            auto json_data = json::parse(response);
            
            if (json_data.contains("result") && json_data["result"].contains("tick")) {
                for (const auto& tick_json : json_data["result"]["tick"]) {
                    MT5Tick tick;
                    tick.symbol = symbol;
                    tick.bid = tick_json["bid"].get<double>();
                    tick.ask = tick_json["ask"].get<double>();
                    tick.last = tick_json["last"].get<double>();
                    tick.volume = tick_json["volume"].get<double>();
                    tick.timestamp = std::chrono::system_clock::from_time_t(
                        tick_json["time"].get<long>());
                    
                    ticks.push_back(tick);
                }
            }
            
            LOG_INFO("Retrieved {} recent ticks for {}", ticks.size(), symbol);
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get recent ticks: {}", e.what());
        }
        
        return ticks;
    });
}

std::future<std::string> MetaTraderProvider::get_account_info() {
    return std::async(std::launch::async, [this]() {
        try {
            std::string response = make_rest_request(MetaTraderEndpoints::ACCOUNTS);
            auto json_data = json::parse(response);
            
            if (json_data.contains("result")) {
                return json_data["result"].dump();
            }
            
            return "{}";
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get account info: {}", e.what());
            return "{}";
        }
    });
}

std::future<std::vector<std::string>> MetaTraderProvider::get_positions() {
    return std::async(std::launch::async, [this]() {
        std::vector<std::string> positions;
        
        try {
            std::string response = make_rest_request(MetaTraderEndpoints::POSITIONS);
            auto json_data = json::parse(response);
            
            if (json_data.contains("result")) {
                for (const auto& position : json_data["result"]) {
                    positions.push_back(position.dump());
                }
            }
            
            LOG_INFO("Retrieved {} positions", positions.size());
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get positions: {}", e.what());
        }
        
        return positions;
    });
}

std::future<std::vector<std::string>> MetaTraderProvider::get_trade_history(int days) {
    return std::async(std::launch::async, [this, days]() {
        std::vector<std::string> trades;
        
        try {
            auto now = std::chrono::system_clock::now();
            auto from_time = now - std::chrono::hours(24 * days);
            auto from_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                from_time.time_since_epoch()).count();
            
            std::string body = json({
                {"from", from_ts},
                {"to", std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()).count()}
            }).dump();
            
            std::string response = make_rest_request(
                MetaTraderEndpoints::HISTORY, "POST", body);
            
            auto json_data = json::parse(response);
            
            if (json_data.contains("result")) {
                for (const auto& trade : json_data["result"]) {
                    trades.push_back(trade.dump());
                }
            }
            
            LOG_INFO("Retrieved {} trades from last {} days", trades.size(), days);
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to get trade history: {}", e.what());
        }
        
        return trades;
    });
}

// Private methods implementation

bool MetaTraderProvider::authenticate() {
    try {
        std::string body = json({
            {"user_id", user_id_},
            {"api_key", api_key_}
        }).dump();
        
        std::string response = make_rest_request(
            MetaTraderEndpoints::AUTH, "POST", body);
        
        auto json_data = json::parse(response);
        
        if (json_data.contains("result")) {
            auth_token_ = json_data["result"]["token"].get<std::string>();
            auto expiry_seconds = json_data["result"]["expiry"].get<int>();
            token_expiry_ = std::chrono::steady_clock::now() + 
                               std::chrono::seconds(expiry_seconds);
            
            LOG_INFO("MetaTrader authentication successful");
            return true;
        }
        
        return false;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Authentication failed: {}", e.what());
        return false;
    }
}

bool MetaTraderProvider::test_rest_connection() {
    try {
        std::string response = make_rest_request(MetaTraderEndpoints::SYMBOLS);
        auto json_data = json::parse(response);
        return json_data.contains("result");
        
    } catch (const std::exception& e) {
        LOG_ERROR("REST connection test failed: {}", e.what());
        return false;
    }
}

std::string MetaTraderProvider::make_rest_request(
    const std::string& endpoint,
    const std::string& method,
    const std::string& body
) {
    enforce_rate_limit();
    
    std::string url = MetaTraderEndpoints::REST_BASE + endpoint;
    std::string response;
    
    // Set headers
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    if (!auth_token_.empty()) {
        headers = curl_slist_append(headers, ("Authorization: Bearer " + auth_token_).c_str());
    }
    
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

void MetaTraderProvider::enforce_rate_limit() {
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

void MetaTraderProvider::setup_websocket_connection() {
    ws_thread_ = std::thread([this]() {
        auto con = ws_client_.get_connection(MetaTraderEndpoints::WS_BASE);
        
        con->set_open_handler([this](websocketpp::connection_hdl hdl) {
            LOG_INFO("MetaTrader WebSocket connection opened");
        });
        
        con->set_message_handler([this](websocketpp::connection_hdl hdl, 
                                           websocketpp::client::message<websocketpp::config::asio_client>::type::ptr msg) {
            try {
                auto json_data = json::parse(msg->get_payload());
                
                // Handle different message types
                if (json_data.contains("type")) {
                    std::string type = json_data["type"];
                    
                    if (type == "tick") {
                        // Handle tick message
                    } else if (type == "candle") {
                        // Handle candle message
                    } else if (type == "book") {
                        // Handle order book message
                    }
                }
            } catch (const std::exception& e) {
                LOG_ERROR("Error parsing WebSocket message: {}", e.what());
            }
        });
        
        con->set_fail_handler([this](websocketpp::connection_hdl hdl) {
            LOG_ERROR("MetaTrader WebSocket connection failed");
            status_ = ConnectionStatus::ERROR;
        });
        
        ws_client_.connect(con);
        ws_client_.run();
    });
}

std::vector<OHLCV> MetaTraderProvider::parse_candles_response(
    const std::string& response,
    const std::string& symbol
) {
    std::vector<OHLCV> data;
    
    auto json_data = json::parse(response);
    
    if (json_data.contains("result")) {
        auto result = json_data["result"];
        
        if (result.contains("candle")) {
            for (const auto& candle : result["candle"]) {
                OHLCV ohlcv;
                ohlcv.symbol = symbol;
                ohlcv.timestamp = std::chrono::system_clock::from_time_t(
                    candle["time"].get<long>());
                ohlcv.open = candle["open"].get<double>();
                ohlcv.high = candle["high"].get<double>();
                ohlcv.low = candle["low"].get<double>();
                ohlcv.close = candle["close"].get<double>();
                ohlcv.volume = candle["volume"].get<double>();
                
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

OrderBook MetaTraderProvider::parse_tick_response(
    const std::string& response,
    const std::string& symbol
) {
    OrderBook order_book;
    order_book.symbol = symbol;
    order_book.timestamp = std::chrono::system_clock::now();
    
    auto json_data = json::parse(response);
    
    if (json_data.contains("result")) {
        auto result = json_data["result"];
        
        // Parse bids
        if (result.contains("bid")) {
            OrderBookEntry bid;
            bid.price = result["bid"].get<double>();
            bid.quantity = result["volume"].get<double>();
            order_book.bids.push_back(bid);
        }
        
        // Parse asks
        if (result.contains("ask")) {
            OrderBookEntry ask;
            ask.price = result["ask"].get<double>();
            ask.quantity = result["volume"].get<double>();
            order_book.asks.push_back(ask);
        }
    }
    
    return order_book;
}

std::vector<MT5Symbol> MetaTraderProvider::parse_symbols_response(const std::string& response) {
    std::vector<MT5Symbol> symbols;
    
    auto json_data = json::parse(response);
    
    if (json_data.contains("result")) {
        for (const auto& symbol_json : json_data["result"]) {
            MT5Symbol symbol;
            symbol.symbol = symbol_json["symbol"].get<std::string>();
            symbol.description = symbol_json.value("description", "");
            symbol.base_currency = symbol_json.value("base_currency", "");
            symbol.quote_currency = symbol_json.value("quote_currency", "");
            symbol.tick_size = symbol_json.value("tick_size", 0.0);
            symbol.contract_size = symbol_json.value("contract_size", 0.0);
            symbol.digits = symbol_json.value("digits", 5);
            symbol.visible = symbol_json.value("visible", true);
            symbol.trade_mode = symbol_json.value("trade_mode", true);
            
            symbols.push_back(symbol);
        }
    }
    
    return symbols;
}

std::string MetaTraderProvider::convert_symbol_to_mt5(const std::string& symbol) {
    auto it = symbol_to_mt5_.find(symbol);
    if (it != symbol_to_mt5_.end()) {
        return it->second;
    }
    
    // Fallback: convert common formats
    std::string mt5_symbol = symbol;
    std::replace(mt5_symbol.begin(), mt5_symbol.end(), '/', '');
    
    return mt5_symbol;
}

std::string MetaTraderProvider::convert_mt5_to_symbol(const std::string& mt5_symbol) {
    auto it = mt5_to_symbol_.find(mt5_symbol);
    if (it != mt5_to_symbol_.end()) {
        return it->second;
    }
    
    return mt5_symbol; // Return as-is if no mapping found
}

std::string MetaTraderProvider::convert_timeframe_to_mt5(const std::string& timeframe) {
    // Map timeframes to MetaTrader 5 intervals
    if (timeframe == "1m") return "M1";
    if (timeframe == "5m") return "M5";
    if (timeframe == "15m") return "M15";
    if (timeframe == "30m") return "M30";
    if (timeframe == "1h") return "H1";
    if (timeframe == "4h") return "H4";
    if (timeframe == "1d") return "D1";
    if (timeframe == "1w") return "W1";
    if (timeframe == "1M") return "MN1";
    
    // Default to 1 hour
    return "H1";
}

void MetaTraderProvider::initialize_symbol_mapping() {
    // Common forex symbol mappings
    symbol_to_mt5_["EUR/USD"] = "EURUSD";
    mt5_to_symbol_["EURUSD"] = "EUR/USD";
    
    symbol_to_mt5_["GBP/USD"] = "GBPUSD";
    mt5_to_symbol_["GBPUSD"] = "GBP/USD";
    
    symbol_to_mt5_["USD/JPY"] = "USDJPY";
    mt5_to_symbol_["USDJPY"] = "USD/JPY";
    
    symbol_to_mt5_["USD/CHF"] = "USDCHF";
    mt5_to_symbol_["USDCHF"] = "USD/CHF";
    
    // Add more mappings as needed
}

size_t MetaTraderProvider::WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

} // namespace Data
} // namespace ArchNeuronX
