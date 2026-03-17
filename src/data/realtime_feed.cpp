/**
 * @file realtime_feed.cpp
 * @brief Real-time market data feed implementation
 * @author ArchNeuronX Team
 * @date 2026-03-17
 */

#include "data/realtime_feed.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <sstream>
#include <iomanip>

namespace archneuronx {
namespace data {

RealtimeFeed::RealtimeFeed(const RealtimeFeedConfig& config)
    : config_(config), paper_trading_enabled_(false), paper_balance_(0.0),
      next_order_id_(1), running_(false) {
    
    stats_.messages_received = 0;
    stats_.ticks_processed = 0;
    stats_.orderbook_updates = 0;
    stats_.messages_per_second = 0.0;
    stats_.latency_ms = 0.0;
    stats_.last_update = std::chrono::system_clock::now();
}

RealtimeFeed::~RealtimeFeed() {
    disconnect();
}

bool RealtimeFeed::connect() {
    if (running_) {
        std::cerr << "Feed already connected" << std::endl;
        return false;
    }
    
    try {
        running_ = true;
        
        // Initialize exchange clients
        for (const auto& exchange : config_.exchanges) {
            if (!initialize_exchange_client(exchange)) {
                std::cerr << "Failed to initialize client for exchange: " << exchange << std::endl;
                continue;
            }
            
            connect_to_exchange(exchange);
        }
        
        // Start processing threads
        for (int i = 0; i < config_.processing_threads; ++i) {
            processing_threads_.emplace_back(&RealtimeFeed::processing_thread_func, this);
        }
        
        // Start heartbeat and reconnection threads
        processing_threads_.emplace_back(&RealtimeFeed::heartbeat_thread_func, this);
        processing_threads_.emplace_back(&RealtimeFeed::reconnect_thread_func, this);
        
        std::cout << "Real-time feed connected successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error connecting feed: " << e.what() << std::endl;
        running_ = false;
        return false;
    }
}

void RealtimeFeed::disconnect() {
    running_ = false;
    
    // Close WebSocket connections
    for (auto& [exchange, client] : ws_clients_) {
        if (client) {
            // Close connection
            client->close(websocketpp::close::status::normal, "Shutdown");
        }
    }
    
    // Wait for threads to finish
    for (auto& thread : processing_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    processing_threads_.clear();
    ws_clients_.clear();
    
    std::cout << "Real-time feed disconnected" << std::endl;
}

bool RealtimeFeed::is_connected() const {
    return running_ && !ws_clients_.empty();
}

bool RealtimeFeed::subscribe_trades(const std::string& symbol, const std::string& exchange) {
    std::string target_exchange = exchange.empty() ? config_.exchanges[0] : exchange;
    
    auto it = ws_clients_.find(target_exchange);
    if (it == ws_clients_.end() || !it->second) {
        std::cerr << "Exchange not connected: " << target_exchange << std::endl;
        return false;
    }
    
    // Create subscription message
    nlohmann::json msg = create_subscription_message("trades", {symbol});
    
    try {
        it->second->send(msg.dump());
        std::cout << "Subscribed to trades for " << symbol << " on " << target_exchange << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to subscribe to trades: " << e.what() << std::endl;
        return false;
    }
}

bool RealtimeFeed::subscribe_orderbook(const std::string& symbol, const std::string& exchange) {
    std::string target_exchange = exchange.empty() ? config_.exchanges[0] : exchange;
    
    auto it = ws_clients_.find(target_exchange);
    if (it == ws_clients_.end() || !it->second) {
        std::cerr << "Exchange not connected: " << target_exchange << std::endl;
        return false;
    }
    
    nlohmann::json msg = create_subscription_message("orderbook", {symbol});
    
    try {
        it->second->send(msg.dump());
        std::cout << "Subscribed to orderbook for " << symbol << " on " << target_exchange << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to subscribe to orderbook: " << e.what() << std::endl;
        return false;
    }
}

bool RealtimeFeed::subscribe_ticker(const std::string& symbol, const std::string& exchange) {
    std::string target_exchange = exchange.empty() ? config_.exchanges[0] : exchange;
    
    auto it = ws_clients_.find(target_exchange);
    if (it == ws_clients_.end() || !it->second) {
        std::cerr << "Exchange not connected: " << target_exchange << std::endl;
        return false;
    }
    
    nlohmann::json msg = create_subscription_message("ticker", {symbol});
    
    try {
        it->second->send(msg.dump());
        std::cout << "Subscribed to ticker for " << symbol << " on " << target_exchange << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to subscribe to ticker: " << e.what() << std::endl;
        return false;
    }
}

void RealtimeFeed::unsubscribe_all() {
    for (const auto& exchange : config_.exchanges) {
        auto it = ws_clients_.find(exchange);
        if (it != ws_clients_.end() && it->second) {
            nlohmann::json msg = create_subscription_message("unsubscribe", config_.symbols);
            it->second->send(msg.dump());
        }
    }
    
    std::cout << "Unsubscribed from all streams" << std::endl;
}

void RealtimeFeed::set_tick_callback(TickCallback callback) {
    tick_callback_ = callback;
}

void RealtimeFeed::set_orderbook_callback(OrderBookCallback callback) {
    orderbook_callback_ = callback;
}

void RealtimeFeed::set_error_callback(ErrorCallback callback) {
    error_callback_ = callback;
}

std::vector<MarketTick> RealtimeFeed::get_recent_ticks(const std::string& symbol, int max_count) const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = recent_ticks_.find(symbol);
    if (it == recent_ticks_.end()) {
        return {};
    }
    
    const auto& ticks = it->second;
    int start_idx = std::max(0, static_cast<int>(ticks.size()) - max_count);
    
    return std::vector<MarketTick>(ticks.begin() + start_idx, ticks.end());
}

OrderBook RealtimeFeed::get_current_orderbook(const std::string& symbol, const std::string& exchange) const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    std::string key = exchange.empty() ? symbol : symbol + "@" + exchange;
    auto it = orderbooks_.find(key);
    
    return (it != orderbooks_.end()) ? it->second : OrderBook{};
}

double RealtimeFeed::get_current_price(const std::string& symbol, const std::string& exchange) const {
    auto recent = get_recent_ticks(symbol, 1);
    if (!recent.empty()) {
        return recent.back().price;
    }
    
    // Fallback to orderbook mid-price
    auto book = get_current_orderbook(symbol, exchange);
    if (!book.bids.empty() && !book.asks.empty()) {
        return (book.bids[0].price + book.asks[0].price) / 2.0;
    }
    
    return 0.0;
}

bool RealtimeFeed::enable_paper_trading(double initial_balance) {
    std::lock_guard<std::mutex> lock(paper_mutex_);
    
    paper_trading_enabled_ = true;
    paper_balance_ = initial_balance;
    paper_positions_.clear();
    paper_orders_.clear();
    paper_trade_history_.clear();
    next_order_id_ = 1;
    
    std::cout << "Paper trading enabled with balance: $" << initial_balance << std::endl;
    return true;
}

void RealtimeFeed::disable_paper_trading() {
    std::lock_guard<std::mutex> lock(paper_mutex_);
    
    paper_trading_enabled_ = false;
    std::cout << "Paper trading disabled" << std::endl;
}

bool RealtimeFeed::is_paper_trading_enabled() const {
    std::lock_guard<std::mutex> lock(paper_mutex_);
    return paper_trading_enabled_;
}

std::string RealtimeFeed::place_paper_order(const std::string& symbol, OrderSide side, 
                                           OrderType type, double quantity, double price) {
    if (!paper_trading_enabled_) {
        return "";
    }
    
    std::lock_guard<std::mutex> lock(paper_mutex_);
    
    // Generate order ID
    std::string order_id = generate_order_id();
    
    // Create order
    PaperOrder order;
    order.order_id = order_id;
    order.symbol = symbol;
    order.type = type;
    order.side = side;
    order.quantity = quantity;
    order.price = price;
    order.status = OrderStatus::PENDING;
    order.created_time = std::chrono::system_clock::now();
    order.filled_quantity = 0.0;
    order.average_fill_price = 0.0;
    
    // Store order
    paper_orders_[order_id] = order;
    
    // For market orders, execute immediately
    if (type == OrderType::MARKET) {
        auto current_price = get_current_price(symbol);
        if (current_price > 0.0) {
            order.status = OrderStatus::FILLED;
            order.filled_time = std::chrono::system_clock::now();
            order.filled_quantity = quantity;
            order.average_fill_price = current_price;
            
            paper_orders_[order_id] = order;
            process_paper_order(order, {symbol, current_price, quantity, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                      std::chrono::system_clock::now(), config_.paper_exchange, 0});
        }
    }
    
    std::cout << "Placed paper order: " << order_id << " " << (side == OrderSide::BUY ? "BUY" : "SELL") 
              << " " << quantity << " " << symbol << std::endl;
    
    return order_id;
}

bool RealtimeFeed::cancel_paper_order(const std::string& order_id) {
    std::lock_guard<std::mutex> lock(paper_mutex_);
    
    auto it = paper_orders_.find(order_id);
    if (it == paper_orders_.end()) {
        return false;
    }
    
    if (it->second.status == OrderStatus::PENDING) {
        it->second.status = OrderStatus::CANCELLED;
        std::cout << "Cancelled paper order: " << order_id << std::endl;
        return true;
    }
    
    return false;
}

std::vector<PaperOrder> RealtimeFeed::get_paper_orders() const {
    std::lock_guard<std::mutex> lock(paper_mutex_);
    
    std::vector<PaperOrder> orders;
    for (const auto& [id, order] : paper_orders_) {
        orders.push_back(order);
    }
    
    return orders;
}

std::vector<PaperPosition> RealtimeFeed::get_paper_positions() const {
    std::lock_guard<std::mutex> lock(paper_mutex_);
    
    std::vector<PaperPosition> positions;
    for (const auto& [symbol, position] : paper_positions_) {
        if (std::abs(position.quantity) > 0.001) { // Filter tiny positions
            positions.push_back(position);
        }
    }
    
    return positions;
}

double RealtimeFeed::get_paper_balance() const {
    std::lock_guard<std::mutex> lock(paper_mutex_);
    return paper_balance_;
}

double RealtimeFeed::get_paper_total_pnl() const {
    std::lock_guard<std::mutex> lock(paper_mutex_);
    
    double total_pnl = paper_balance_ - config_.paper_balance_usd;
    
    // Add unrealized P&L from positions
    for (const auto& [symbol, position] : paper_positions_) {
        total_pnl += position.unrealized_pnl + position.realized_pnl;
    }
    
    return total_pnl;
}

std::vector<std::string> RealtimeFeed::get_paper_trade_history() const {
    std::lock_guard<std::mutex> lock(paper_mutex_);
    return paper_trade_history_;
}

RealtimeFeed::FeedStats RealtimeFeed::get_feed_stats() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return stats_;
}

void RealtimeFeed::reset_stats() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    stats_.messages_received = 0;
    stats_.ticks_processed = 0;
    stats_.orderbook_updates = 0;
    stats_.messages_per_second = 0.0;
    stats_.latency_ms = 0.0;
    stats_.last_update = std::chrono::system_clock::now();
}

std::vector<std::string> RealtimeFeed::get_supported_exchanges() const {
    return config_.exchanges;
}

std::vector<std::string> RealtimeFeed::get_supported_symbols(const std::string& exchange) const {
    if (exchange.empty()) {
        return config_.symbols;
    }
    
    // In practice, this would query each exchange for supported symbols
    return config_.symbols;
}

// Private methods

bool RealtimeFeed::initialize_exchange_client(const std::string& exchange) {
    try {
        auto client = std::make_unique<websocketpp::client<websocketpp::config::asio_client>>();
        
        // Set up handlers
        client->set_access_channels(websocketpp::log::alevel::all);
        client->clear_access_channels(websocketpp::log::alevel::frame_payload);
        
        client->set_message_handler([this, exchange](websocketpp::connection_hdl hdl, 
                                                       websocketpp::client<websocketpp::config::asio_client>::message_ptr msg) {
            handle_websocket_message(exchange, msg->get_payload());
        });
        
        client->set_fail_handler([this, exchange](websocketpp::connection_hdl hdl) {
            std::cerr << "Connection failed for exchange: " << exchange << std::endl;
            if (error_callback_) {
                error_callback_("Connection failed: " + exchange);
            }
        });
        
        client->set_open_handler([this, exchange](websocketpp::connection_hdl hdl) {
            std::cout << "Connected to exchange: " << exchange << std::endl;
        });
        
        // Initialize ASIO
        client->init_asio();
        
        ws_clients_[exchange] = std::move(client);
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing exchange client: " << e.what() << std::endl;
        return false;
    }
}

void RealtimeFeed::connect_to_exchange(const std::string& exchange) {
    auto it = ws_clients_.find(exchange);
    if (it == ws_clients_.end() || !it->second) {
        return;
    }
    
    try {
        std::string url = get_exchange_websocket_url(exchange);
        websocketpp::lib::error_code ec;
        auto con = it->second->get_connection(url, ec);
        
        if (ec) {
            std::cerr << "Could not create connection: " << ec.message() << std::endl;
            return;
        }
        
        it->second->connect(con);
        it->second->run();
        
    } catch (const std::exception& e) {
        std::cerr << "Error connecting to exchange: " << e.what() << std::endl;
    }
}

void RealtimeFeed::handle_websocket_message(const std::string& exchange, const std::string& message) {
    try {
        auto json_data = nlohmann::json::parse(message);
        
        // Update statistics
        stats_.messages_received++;
        update_stats();
        
        // Determine message type and process accordingly
        if (json_data.contains("stream")) {
            std::string stream = json_data["stream"];
            if (stream.find("trade") != std::string::npos) {
                process_tick_data(exchange, json_data["data"]);
            } else if (stream.find("depth") != std::string::npos) {
                process_orderbook_data(exchange, json_data["data"]);
            } else if (stream.find("ticker") != std::string::npos) {
                process_ticker_data(exchange, json_data["data"]);
            }
        } else if (json_data.contains("event")) {
            // Handle subscription acknowledgments
            std::cout << "Event from " << exchange << ": " << json_data["event"] << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing message from " << exchange << ": " << e.what() << std::endl;
        if (error_callback_) {
            error_callback_("Parse error: " + std::string(e.what()));
        }
    }
}

void RealtimeFeed::process_tick_data(const std::string& exchange, const nlohmann::json& data) {
    try {
        MarketTick tick = parse_trade_data(exchange, data);
        
        if (tick.price > 0.0) {
            std::lock_guard<std::mutex> lock(data_mutex_);
            
            // Store tick
            recent_ticks_[tick.symbol].push_back(tick);
            
            // Keep only recent ticks
            if (recent_ticks_[tick.symbol].size() > 1000) {
                recent_ticks_[tick.symbol].erase(recent_ticks_[tick.symbol].begin());
            }
            
            // Update statistics
            stats_.ticks_processed++;
            
            // Process paper trading orders
            if (paper_trading_enabled_) {
                std::lock_guard<std::mutex> paper_lock(paper_mutex_);
                for (auto& [order_id, order] : paper_orders_) {
                    if (order.status == OrderStatus::PENDING && order.symbol == tick.symbol) {
                        process_paper_order(order, tick);
                    }
                }
            }
            
            // Invoke callback
            if (tick_callback_) {
                tick_callback_(tick);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing tick data: " << e.what() << std::endl;
    }
}

void RealtimeFeed::process_orderbook_data(const std::string& exchange, const nlohmann::json& data) {
    try {
        OrderBook book = parse_orderbook_data(exchange, data);
        
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        // Store orderbook
        std::string key = book.symbol + "@" + exchange;
        orderbooks_[key] = book;
        
        // Update statistics
        stats_.orderbook_updates++;
        
        // Invoke callback
        if (orderbook_callback_) {
            orderbook_callback_(book);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing orderbook data: " << e.what() << std::endl;
    }
}

void RealtimeFeed::process_ticker_data(const std::string& exchange, const nlohmann::json& data) {
    // Process 24hr ticker statistics
    // This would update various market statistics
}

void RealtimeFeed::process_paper_order(const PaperOrder& order, const MarketTick& tick) {
    const_cast<PaperOrder&>(order).status = OrderStatus::FILLED;
    const_cast<PaperOrder&>(order).filled_time = std::chrono::system_clock::now();
    const_cast<PaperOrder&>(order).filled_quantity = order.quantity;
    const_cast<PaperOrder&>(order).average_fill_price = tick.price;
    
    // Calculate fees
    double trade_amount = order.quantity * tick.price;
    double fees = calculate_paper_fees(trade_amount);
    
    // Update balance and positions
    if (order.side == OrderSide::BUY) {
        paper_balance_ -= trade_amount + fees;
        
        // Update position
        auto& position = paper_positions_[order.symbol];
        if (position.quantity == 0.0) {
            position.symbol = order.symbol;
            position.quantity = order.quantity;
            position.average_price = tick.price;
            position.entry_time = std::chrono::system_clock::now();
            position.exchange = config_.paper_exchange;
        } else {
            // Average down/up
            double total_cost = (position.quantity * position.average_price) + trade_amount;
            position.quantity += order.quantity;
            position.average_price = total_cost / position.quantity;
        }
        
    } else { // SELL
        paper_balance_ += trade_amount - fees;
        
        // Update position
        auto& position = paper_positions_[order.symbol];
        if (position.quantity > 0.0) {
            // Realized P&L
            double realized_pnl = order.quantity * (tick.price - position.average_price);
            position.realized_pnl += realized_pnl;
            position.quantity -= order.quantity;
            
            if (position.quantity <= 0.001) {
                position.quantity = 0.0;
                position.unrealized_pnl = 0.0;
            }
        }
    }
    
    // Record trade
    std::stringstream trade_record;
    trade_record << std::fixed << std::setprecision(2)
                  << order.symbol << " " << (order.side == OrderSide::BUY ? "BUY" : "SELL")
                  << " " << order.quantity << " @ " << tick.price;
    paper_trade_history_.push_back(trade_record.str());
    
    // Keep history manageable
    if (paper_trade_history_.size() > 1000) {
        paper_trade_history_.erase(paper_trade_history_.begin());
    }
}

double RealtimeFeed::calculate_paper_fees(double amount) const {
    return amount * config_.paper_fee_rate;
}

std::string RealtimeFeed::generate_order_id() const {
    std::stringstream ss;
    ss << "PAPER_" << next_order_id_++ << "_" << 
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count();
    return ss.str();
}

std::string RealtimeFeed::get_exchange_websocket_url(const std::string& exchange) const {
    // Return WebSocket URLs for supported exchanges
    if (exchange == "binance") {
        return "wss://stream.binance.com:9443/ws";
    } else if (exchange == "coinbase") {
        return "wss://ws-feed.exchange.coinbase.com";
    } else if (exchange == "kraken") {
        return "wss://ws.kraken.com";
    }
    
    return "";
}

nlohmann::json RealtimeFeed::create_subscription_message(const std::string& type, 
                                                         const std::vector<std::string>& symbols) const {
    nlohmann::json msg;
    
    if (type == "trades") {
        msg["method"] = "SUBSCRIBE";
        for (const auto& symbol : symbols) {
            msg["params"].push_back(symbol.lower() + "@trade");
        }
    } else if (type == "orderbook") {
        msg["method"] = "SUBSCRIBE";
        for (const auto& symbol : symbols) {
            msg["params"].push_back(symbol.lower() + "@depth10");
        }
    } else if (type == "ticker") {
        msg["method"] = "SUBSCRIBE";
        for (const auto& symbol : symbols) {
            msg["params"].push_back(symbol.lower() + "@ticker");
        }
    } else if (type == "unsubscribe") {
        msg["method"] = "UNSUBSCRIBE";
        for (const auto& symbol : symbols) {
            msg["params"].push_back(symbol.lower());
        }
    }
    
    msg["id"] = 1;
    return msg;
}

MarketTick RealtimeFeed::parse_trade_data(const std::string& exchange, const nlohmann::json& data) const {
    MarketTick tick;
    
    // Parse based on exchange format
    if (exchange == "binance") {
        tick.symbol = data["s"];
        tick.price = std::stod(data["p"].get<std::string>());
        tick.volume = std::stod(data["q"].get<std::string>());
        tick.timestamp = std::chrono::system_clock::from_time_t(data["T"].get<uint64_t>() / 1000);
    } else {
        // Generic parsing
        tick.symbol = data.value("symbol", "");
        tick.price = data.value("price", 0.0);
        tick.volume = data.value("volume", 0.0);
        tick.timestamp = std::chrono::system_clock::now();
    }
    
    tick.exchange = exchange;
    tick.trade_id = data.value("id", 0);
    
    return tick;
}

OrderBook RealtimeFeed::parse_orderbook_data(const std::string& exchange, const nlohmann::json& data) const {
    OrderBook book;
    
    // Parse based on exchange format
    if (exchange == "binance") {
        book.symbol = data["s"];
        
        // Parse bids
        for (const auto& bid : data["bids"]) {
            OrderBookLevel level;
            level.price = std::stod(bid[0].get<std::string>());
            level.quantity = std::stod(bid[1].get<std::string>());
            book.bids.push_back(level);
        }
        
        // Parse asks
        for (const auto& ask : data["asks"]) {
            OrderBookLevel level;
            level.price = std::stod(ask[0].get<std::string>());
            level.quantity = std::stod(ask[1].get<std::string>());
            book.asks.push_back(level);
        }
    } else {
        // Generic parsing
        book.symbol = data.value("symbol", "");
        // ... generic parsing logic
    }
    
    book.exchange = exchange;
    book.timestamp = std::chrono::system_clock::now();
    book.sequence = data.value("lastUpdateId", 0);
    
    return book;
}

void RealtimeFeed::processing_thread_func() {
    while (running_) {
        try {
            // Process queued messages
            // This would handle batch processing of market data
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } catch (const std::exception& e) {
            std::cerr << "Error in processing thread: " << e.what() << std::endl;
        }
    }
}

void RealtimeFeed::heartbeat_thread_func() {
    while (running_) {
        try {
            // Send heartbeat messages to maintain connections
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.heartbeat_interval_ms));
        } catch (const std::exception& e) {
            std::cerr << "Error in heartbeat thread: " << e.what() << std::endl;
        }
    }
}

void RealtimeFeed::reconnect_thread_func() {
    while (running_) {
        try {
            // Check for disconnected exchanges and reconnect
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.reconnect_interval_ms));
        } catch (const std::exception& e) {
            std::cerr << "Error in reconnect thread: " << e.what() << std::endl;
        }
    }
}

void RealtimeFeed::update_stats() {
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - stats_.last_update);
    
    if (duration.count() > 0) {
        stats_.messages_per_second = static_cast<double>(stats_.messages_received) / duration.count();
        stats_.last_update = now;
    }
}

// FeedConnection implementation

FeedConnection::FeedConnection(RealtimeFeed& feed) : feed_(feed), connected_(false) {
    connected_ = feed_.connect();
}

FeedConnection::~FeedConnection() {
    if (connected_) {
        feed_.disconnect();
    }
}

bool FeedConnection::is_connected() const {
    return connected_ && feed_.is_connected();
}

// PaperTradingManager implementation

PaperTradingManager::PaperTradingManager(RealtimeFeed& feed) 
    : feed_(feed), max_drawdown_limit_(0.2), daily_loss_limit_(1000.0),
      starting_balance_(10000.0), max_balance_(10000.0) {
    
    stats_.total_trades = 0;
    stats_.winning_trades = 0;
    stats_.losing_trades = 0;
    stats_.win_rate = 0.0;
    stats_.total_pnl = 0.0;
    stats_.max_drawdown = 0.0;
    stats_.sharpe_ratio = 0.0;
    stats_.average_trade_duration = 0.0;
    stats_.start_time = std::chrono::system_clock::now();
}

std::string PaperTradingManager::place_bracket_order(const std::string& symbol, OrderSide side,
                                                    double quantity, double entry_price,
                                                    double stop_loss, double take_profit) {
    // Place main order
    std::string main_order = feed_.place_paper_order(symbol, side, OrderType::LIMIT, quantity, entry_price);
    
    if (main_order.empty()) {
        return "";
    }
    
    // Place stop loss order
    OrderSide stop_side = (side == OrderSide::BUY) ? OrderSide::SELL : OrderSide::BUY;
    std::string stop_order = feed_.place_paper_order(symbol, stop_side, OrderType::STOP_LOSS, quantity, stop_loss);
    
    // Place take profit order
    std::string profit_order = feed_.place_paper_order(symbol, stop_side, OrderType::TAKE_PROFIT, quantity, take_profit);
    
    return main_order; // Return main order ID
}

PaperTradingManager::TradingStats PaperTradingManager::get_trading_stats() const {
    return stats_;
}

void PaperTradingManager::update_trading_stats(const PaperOrder& order) {
    stats_.total_trades++;
    
    // Update other statistics based on order results
    // This would calculate win rate, P&L, etc.
}

} // namespace data
} // namespace archneuronx
