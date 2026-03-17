/**
 * @file market_making.cpp
 * @brief Advanced market making algorithm implementation
 * @author ArchNeuronX Team
 * @date 2026-03-17
 */

#include "execution/market_making.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>

namespace archneuronx {
namespace execution {

MarketMaker::MarketMaker(const MarketMakerConfig& config)
    : config_(config), router_(nullptr), regime_detector_(nullptr), model_loaded_(false),
      current_regime_(0), current_strategy_(MarketMakingStrategy::HYBRID), running_(false) {
    
    performance_.total_quotes_posted = 0;
    performance_.quotes_filled = 0;
    performance_.quotes_cancelled = 0;
    performance_.fill_rate = 0.0;
    performance_.avg_fill_size = 0.0;
    performance_.total_profit_bps = 0.0;
    performance_.spread_capture_bps = 0.0;
    performance_.inventory_pnl_bps = 0.0;
    performance_.adverse_selection_cost_bps = 0.0;
    performance_.max_inventory_utilization = 0.0;
    performance_.avg_inventory_utilization = 0.0;
    performance_.inventory_turnover_rate = 0.0;
    performance_.sharpe_ratio = 0.0;
    performance_.avg_quote_duration_ms = 0.0;
    performance_.avg_time_to_fill_ms = 0.0;
    performance_.quote_cancellation_rate = 0.0;
    performance_.last_update = std::chrono::system_clock::now();
}

MarketMaker::~MarketMaker() {
    shutdown();
}

bool MarketMaker::initialize(SmartOrderRouter& router, regime::RegimeDetector& regime_detector) {
    try {
        router_ = &router;
        regime_detector_ = &regime_detector;
        
        // Load ML prediction model if enabled
        if (config_.enable_ml_prediction && !config_.prediction_model_path.empty()) {
            if (!load_prediction_model(config_.prediction_model_path)) {
                std::cout << "Warning: Failed to load market making prediction model" << std::endl;
            }
        }
        
        // Initialize regime detector callback
        regime_detector_->set_tick_callback([this](const MarketTick& tick) {
            on_market_data_update(tick.symbol, tick.bid, tick.ask);
        });
        
        // Initialize background threads
        initialize_background_threads();
        
        running_ = true;
        std::cout << "Market Maker initialized successfully" << std::endl;
        std::cout << "Strategy: " << static_cast<int>(config_.strategy) << std::endl;
        std::cout << "Regime-aware: " << (config_.enable_regime_awareness ? "Enabled" : "Disabled") << std::endl;
        std::cout << "Adverse selection protection: " << (config_.enable_adverse_selection_protection ? "Enabled" : "Disabled") << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing Market Maker: " << e.what() << std::endl;
        return false;
    }
}

void MarketMaker::shutdown() {
    running_ = false;
    
    // Cancel all active quotes
    emergency_stop_all_quoting();
    
    // Shutdown background threads
    shutdown_background_threads();
    
    std::cout << "Market Maker shutdown complete" << std::endl;
}

bool MarketMaker::is_initialized() const {
    return running_ && router_ && regime_detector_;
}

bool MarketMaker::add_symbol(const std::string& symbol, const MarketMakingParams& params) {
    std::unique_lock<std::shared_mutex> lock(symbols_mutex_);
    
    symbol_params_[symbol] = params;
    active_symbols_[symbol] = true;
    
    // Initialize inventory position
    {
        std::unique_lock<std::shared_mutex> inventory_lock(inventory_mutex_);
        InventoryPosition position;
        position.symbol = symbol;
        position.quantity = 0.0;
        position.average_price = 0.0;
        position.unrealized_pnl = 0.0;
        position.realized_pnl = 0.0;
        position.inventory_cost = 0.0;
        position.inventory_risk = 0.0;
        position.concentration_ratio = 0.0;
        position.leverage_utilization = 0.0;
        position.last_update = std::chrono::system_clock::now();
        
        inventory_positions_[symbol] = position;
    }
    
    std::cout << "Added symbol for market making: " << symbol << std::endl;
    return true;
}

bool MarketMaker::remove_symbol(const std::string& symbol) {
    std::unique_lock<std::shared_mutex> lock(symbols_mutex_);
    
    // Cancel all quotes for this symbol
    cancel_all_quotes(symbol);
    
    // Remove from active symbols
    active_symbols_[symbol] = false;
    symbol_params_.erase(symbol);
    
    // Remove inventory position
    {
        std::unique_lock<std::shared_mutex> inventory_lock(inventory_mutex_);
        inventory_positions_.erase(symbol);
    }
    
    std::cout << "Removed symbol from market making: " << symbol << std::endl;
    return true;
}

std::vector<std::string> MarketMaker::get_active_symbols() const {
    std::shared_lock<std::shared_mutex> lock(symbols_mutex_);
    
    std::vector<std::string> symbols;
    for (const auto& [symbol, active] : active_symbols_) {
        if (active) {
            symbols.push_back(symbol);
        }
    }
    
    return symbols;
}

MarketMakingParams MarketMaker::get_symbol_params(const std::string& symbol) const {
    std::shared_lock<std::shared_mutex> lock(symbols_mutex_);
    
    auto it = symbol_params_.find(symbol);
    return (it != symbol_params_.end()) ? it->second : MarketMakingParams{};
}

Quote MarketMaker::generate_quote(const std::string& symbol) {
    if (!is_symbol_active(symbol)) {
        throw std::invalid_argument("Symbol not active for market making");
    }
    
    // Generate quote based on current strategy
    switch (current_strategy_.load()) {
        case MarketMakingStrategy::STATIC_SPREAD:
            return generate_static_spread_quote(symbol);
        case MarketMakingStrategy::DYNAMIC_SPREAD:
            return generate_dynamic_spread_quote(symbol);
        case MarketMakingStrategy::INVENTORY_AWARE:
            return generate_inventory_aware_quote(symbol);
        case MarketMakingStrategy::ADVERSE_SELECTION_PROTECTED:
            return generate_adverse_selection_protected_quote(symbol);
        case MarketMakingStrategy::REGIME_AWARE:
            return generate_regime_aware_quote(symbol);
        case MarketMakingStrategy::HYBRID:
            return generate_hybrid_quote(symbol);
        default:
            return generate_hybrid_quote(symbol);
    }
}

std::vector<Quote> MarketMaker::generate_quotes(const std::vector<std::string>& symbols) {
    std::vector<Quote> quotes;
    
    for (const auto& symbol : symbols) {
        try {
            quotes.push_back(generate_quote(symbol));
        } catch (const std::exception& e) {
            std::cerr << "Error generating quote for " << symbol << ": " << e.what() << std::endl;
        }
    }
    
    return quotes;
}

bool MarketMaker::post_quote(const Quote& quote) {
    try {
        // Apply risk controls
        Quote controlled_quote = quote;
        apply_risk_controls(controlled_quote);
        
        // Check if quote is still valid after risk controls
        if (controlled_quote.bid_price >= controlled_quote.ask_price) {
            std::cerr << "Invalid quote after risk controls: bid >= ask" << std::endl;
            return false;
        }
        
        // Create order requests for bid and ask
        OrderRequest bid_order;
        bid_order.order_id = controlled_quote.quote_id + "_bid";
        bid_order.symbol = controlled_quote.symbol;
        bid_order.side = "BUY";
        bid_order.quantity = controlled_quote.bid_size;
        bid_order.price = controlled_quote.bid_price;
        bid_order.type = OrderType::LIMIT;
        bid_order.timestamp = std::chrono::system_clock::now();
        
        OrderRequest ask_order;
        ask_order.order_id = controlled_quote.quote_id + "_ask";
        ask_order.symbol = controlled_quote.symbol;
        ask_order.side = "SELL";
        ask_order.quantity = controlled_quote.ask_size;
        ask_order.price = controlled_quote.ask_price;
        ask_order.type = OrderType::LIMIT;
        ask_order.timestamp = std::chrono::system_clock::now();
        
        // Route orders through smart router
        auto bid_routing = router_->route_order(bid_order);
        auto ask_routing = router_->route_order(ask_order);
        
        bool bid_success = router_->execute_order(bid_order, bid_routing);
        bool ask_success = router_->execute_order(ask_order, ask_routing);
        
        if (bid_success || ask_success) {
            // Store active quote
            {
                std::lock_guard<std::mutex> lock(quotes_mutex_);
                active_quotes_[controlled_quote.quote_id] = controlled_quote;
            }
            
            // Update performance metrics
            {
                std::lock_guard<std::mutex> lock(performance_mutex_);
                performance_.total_quotes_posted++;
            }
            
            std::cout << "Posted quote: " << controlled_quote.quote_id 
                      << " (bid: " << controlled_quote.bid_price 
                      << ", ask: " << controlled_quote.ask_price 
                      << ", spread: " << controlled_quote.spread_bps << " bps)" << std::endl;
            
            return true;
        } else {
            std::cerr << "Failed to post quote: " << controlled_quote.quote_id << std::endl;
            return false;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error posting quote: " << e.what() << std::endl;
        return false;
    }
}

bool MarketMaker::cancel_quote(const std::string& quote_id) {
    std::lock_guard<std::mutex> lock(quotes_mutex_);
    
    auto it = active_quotes_.find(quote_id);
    if (it == active_quotes_.end()) {
        return false;
    }
    
    // Cancel the underlying orders
    OrderRequest bid_cancel;
    bid_cancel.order_id = quote_id + "_bid";
    bid_cancel.symbol = it->second.symbol;
    bid_cancel.side = "BUY";
    bid_cancel.type = OrderType::MARKET; // Cancel order
    
    OrderRequest ask_cancel;
    ask_cancel.order_id = quote_id + "_ask";
    ask_cancel.symbol = it->second.symbol;
    ask_cancel.side = "SELL";
    ask_cancel.type = OrderType::MARKET; // Cancel order
    
    // Execute cancellations
    bool bid_success = true; // Simplified
    bool ask_success = true; // Simplified
    
    if (bid_success && ask_success) {
        active_quotes_.erase(it);
        
        // Update performance metrics
        {
            std::lock_guard<std::mutex> perf_lock(performance_mutex_);
            performance_.quotes_cancelled++;
        }
        
        std::cout << "Cancelled quote: " << quote_id << std::endl;
        return true;
    }
    
    return false;
}

void MarketMaker::cancel_all_quotes(const std::string& symbol) {
    std::lock_guard<std::mutex> lock(quotes_mutex_);
    
    std::vector<std::string> to_cancel;
    for (const auto& [quote_id, quote] : active_quotes_) {
        if (quote.symbol == symbol) {
            to_cancel.push_back(quote_id);
        }
    }
    
    for (const auto& quote_id : to_cancel) {
        cancel_quote(quote_id);
    }
}

void MarketMaker::update_inventory(const std::string& symbol, double quantity_change, double price) {
    std::unique_lock<std::shared_mutex> lock(inventory_mutex_);
    
    auto it = inventory_positions_.find(symbol);
    if (it == inventory_positions_.end()) {
        return;
    }
    
    InventoryPosition& position = it->second;
    
    // Update position
    double old_quantity = position.quantity;
    position.quantity += quantity_change;
    
    if (position.quantity == 0.0) {
        // Position closed
        position.realized_pnl += position.unrealized_pnl;
        position.unrealized_pnl = 0.0;
        position.average_price = 0.0;
    } else if (old_quantity == 0.0) {
        // New position opened
        position.average_price = price;
    } else {
        // Update average price
        double total_cost = (old_quantity * position.average_price) + (quantity_change * price);
        position.average_price = total_cost / position.quantity;
    }
    
    position.last_update = std::chrono::system_clock::now();
    
    // Calculate unrealized P&L
    auto market_it = current_quotes_.find(symbol);
    if (market_it != current_quotes_.end()) {
        double mid_price = (market_it->second.first + market_it->second.second) / 2.0;
        position.unrealized_pnl = position.quantity * (mid_price - position.average_price);
    }
    
    lock.unlock();
    
    std::cout << "Updated inventory for " << symbol 
              << ": quantity=" << position.quantity 
              << ", pnl=" << (position.unrealized_pnl + position.realized_pnl) << std::endl;
}

InventoryPosition MarketMaker::get_inventory_position(const std::string& symbol) const {
    std::shared_lock<std::shared_mutex> lock(inventory_mutex_);
    
    auto it = inventory_positions_.find(symbol);
    return (it != inventory_positions_.end()) ? it->second : InventoryPosition{};
}

std::unordered_map<std::string, InventoryPosition> MarketMaker::get_all_inventory() const {
    std::shared_lock<std::shared_mutex> lock(inventory_mutex_);
    return inventory_positions_;
}

double MarketMaker::get_total_inventory_value() const {
    std::shared_lock<std::shared_mutex> lock(inventory_mutex_);
    
    double total_value = 0.0;
    for (const auto& [symbol, position] : inventory_positions_) {
        auto market_it = current_quotes_.find(symbol);
        if (market_it != current_quotes_.end()) {
            double mid_price = (market_it->second.first + market_it->second.second) / 2.0;
            total_value += std::abs(position.quantity) * mid_price;
        }
    }
    
    return total_value;
}

bool MarketMaker::on_quote_fill(const std::string& quote_id, double fill_quantity, double fill_price) {
    std::lock_guard<std::mutex> lock(quotes_mutex_);
    
    auto it = active_quotes_.find(quote_id);
    if (it == active_quotes_.end()) {
        return false;
    }
    
    Quote& quote = it->second;
    
    // Determine if this was a bid or ask fill
    bool is_bid_fill = (std::abs(fill_price - quote.bid_price) < std::abs(fill_price - quote.ask_price));
    
    // Update inventory
    double quantity_change = is_bid_fill ? fill_quantity : -fill_quantity;
    update_inventory(quote.symbol, quantity_change, fill_price);
    
    // Calculate profit
    double profit_bps = 0.0;
    if (is_bid_fill) {
        // Bought at bid, will sell at ask later
        profit_bps = (quote.ask_price - fill_price) / fill_price * 10000.0;
    } else {
        // Sold at ask, bought at bid earlier
        profit_bps = (fill_price - quote.bid_price) / fill_price * 10000.0;
    }
    
    // Update performance metrics
    {
        std::lock_guard<std::mutex> perf_lock(performance_mutex_);
        performance_.quotes_filled++;
        performance_.total_profit_bps += profit_bps;
        performance_.spread_capture_bps += profit_bps;
    }
    
    // Update ML model if enabled
    if (config_.enable_ml_prediction && model_loaded_) {
        update_ml_model(quote, true, profit_bps);
    }
    
    std::cout << "Quote filled: " << quote_id 
              << " (quantity: " << fill_quantity 
              << ", price: " << fill_price 
              << ", profit: " << profit_bps << " bps)" << std::endl;
    
    return true;
}

bool MarketMaker::on_quote_cancel(const std::string& quote_id) {
    return cancel_quote(quote_id);
}

void MarketMaker::on_market_data_update(const std::string& symbol, double bid_price, double ask_price) {
    std::unique_lock<std::shared_mutex> lock(market_data_mutex_);
    
    current_quotes_[symbol] = {bid_price, ask_price};
    last_market_update_[symbol] = std::chrono::system_clock::now();
    
    lock.unlock();
    
    // Update inventory P&L
    update_inventory_metrics();
    
    // Check if we need to cancel and repost quotes
    auto active_symbols = get_active_symbols();
    for (const auto& active_symbol : active_symbols) {
        if (active_symbol == symbol) {
            // Check if current quotes are still valid
            std::lock_guard<std::mutex> quotes_lock(quotes_mutex_);
            for (auto& [quote_id, quote] : active_quotes_) {
                if (quote.symbol == symbol) {
                    // Check if quote is too far from market
                    double mid_price = (bid_price + ask_price) / 2.0;
                    double quote_mid = (quote.bid_price + quote.ask_price) / 2.0;
                    double deviation = std::abs(quote_mid - mid_price) / mid_price;
                    
                    if (deviation > 0.001) { // 10 bps deviation
                        // Cancel and repost
                        cancel_quote(quote_id);
                    }
                }
            }
        }
    }
}

void MarketMaker::set_strategy(MarketMakingStrategy strategy) {
    current_strategy_ = strategy;
    std::cout << "Market making strategy changed to: " << static_cast<int>(strategy) << std::endl;
}

MarketMakingStrategy MarketMaker::get_strategy() const {
    return current_strategy_.load();
}

void MarketMaker::update_strategy_parameters(const std::string& symbol, const MarketMakingParams& params) {
    std::unique_lock<std::shared_mutex> lock(symbols_mutex_);
    symbol_params_[symbol] = params;
}

MarketMakingPerformance MarketMaker::get_performance_metrics() const {
    std::lock_guard<std::mutex> lock(performance_mutex_);
    return performance_;
}

void MarketMaker::reset_performance_metrics() {
    std::lock_guard<std::mutex> lock(performance_mutex_);
    
    performance_.total_quotes_posted = 0;
    performance_.quotes_filled = 0;
    performance_.quotes_cancelled = 0;
    performance_.fill_rate = 0.0;
    performance_.avg_fill_size = 0.0;
    performance_.total_profit_bps = 0.0;
    performance_.spread_capture_bps = 0.0;
    performance_.inventory_pnl_bps = 0.0;
    performance_.adverse_selection_cost_bps = 0.0;
    performance_.max_inventory_utilization = 0.0;
    performance_.avg_inventory_utilization = 0.0;
    performance_.inventory_turnover_rate = 0.0;
    performance_.sharpe_ratio = 0.0;
    performance_.avg_quote_duration_ms = 0.0;
    performance_.avg_time_to_fill_ms = 0.0;
    performance_.quote_cancellation_rate = 0.0;
    performance_.last_update = std::chrono::system_clock::now();
}

std::vector<std::string> MarketMaker::get_performance_insights() const {
    std::vector<std::string> insights;
    
    std::lock_guard<std::mutex> lock(performance_mutex_);
    
    if (performance_.fill_rate < 0.3) {
        insights.push_back("Low fill rate detected - consider widening spreads");
    }
    
    if (performance_.adverse_selection_cost_bps > 5.0) {
        insights.push_back("High adverse selection costs - enhance protection");
    }
    
    if (performance_.avg_inventory_utilization > 0.8) {
        insights.push_back("High inventory utilization - consider reducing position sizes");
    }
    
    if (performance_.spread_capture_bps < 2.0) {
        insights.push_back("Low spread capture - optimize quote placement");
    }
    
    return insights;
}

bool MarketMaker::check_position_limits(const std::string& symbol, double quantity) const {
    std::shared_lock<std::shared_mutex> lock(inventory_mutex_);
    
    auto it = inventory_positions_.find(symbol);
    if (it == inventory_positions_.end()) {
        return true;
    }
    
    const auto& position = it->second;
    const auto& params = get_symbol_params(symbol);
    
    // Check position size limits
    double new_quantity = position.quantity + quantity;
    double position_value = std::abs(new_quantity) * current_quotes_.at(symbol).first; // Simplified
    
    if (position_value > params.max_position_ratio * params.max_inventory_usd) {
        return false;
    }
    
    return true;
}

bool MarketMaker::check_inventory_risk() const {
    double total_risk = calculate_total_inventory_risk();
    return total_risk < 0.8; // 80% risk threshold
}

void MarketMaker::apply_risk_controls(Quote& quote) {
    // Apply position limits
    apply_position_limits(quote);
    
    // Apply stop loss
    apply_stop_loss(quote);
    
    // Apply adverse selection protection
    if (config_.enable_adverse_selection_protection) {
        protect_against_adverse_selection(quote);
    }
}

void MarketMaker::emergency_stop_all_quoting() {
    std::lock_guard<std::mutex> lock(quotes_mutex_);
    
    std::vector<std::string> quote_ids;
    for (const auto& [quote_id, quote] : active_quotes_) {
        quote_ids.push_back(quote_id);
    }
    
    for (const auto& quote_id : quote_ids) {
        cancel_quote(quote_id);
    }
    
    std::cout << "Emergency stop: All quotes cancelled" << std::endl;
}

bool MarketMaker::load_prediction_model(const std::string& model_path) {
    try {
        prediction_model_ = torch::jit::load(model_path);
        model_loaded_ = true;
        
        std::cout << "Market making prediction model loaded: " << model_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading market making prediction model: " << e.what() << std::endl;
        model_loaded_ = false;
        return false;
    }
}

double MarketMaker::predict_fill_probability(const Quote& quote) {
    if (!model_loaded_) {
        return 0.5; // Default probability
    }
    
    try {
        auto features = extract_quote_features(quote);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(features);
        auto output = prediction_model_.forward(inputs);
        auto prediction = output.toTensor();
        
        return prediction.item<double>();
        
    } catch (const std::exception& e) {
        std::cerr << "Error predicting fill probability: " << e.what() << std::endl;
        return 0.5;
    }
}

double MarketMaker::predict_adverse_selection_risk(const Quote& quote) {
    // Simplified adverse selection risk calculation
    double spread_ratio = quote.spread_bps / get_symbol_params(quote.symbol).base_spread_bps;
    double size_ratio = quote.bid_size / get_symbol_params(quote.symbol).base_quote_size;
    
    return std::min(1.0, spread_ratio * size_ratio * 0.5);
}

torch::Tensor MarketMaker::extract_quote_features(const Quote& quote) {
    std::vector<float> features;
    
    // Quote features
    features.push_back(static_cast<float>(quote.spread_bps));
    features.push_back(static_cast<float>(quote.bid_size));
    features.push_back(static_cast<float>(quote.ask_size));
    features.push_back(static_cast<float>(quote.expected_profit_bps));
    
    // Market features
    auto market_it = current_quotes_.find(quote.symbol);
    if (market_it != current_quotes_.end()) {
        features.push_back(static_cast<float>(market_it->second.first));
        features.push_back(static_cast<float>(market_it->second.second));
    } else {
        features.push_back(0.0f);
        features.push_back(0.0f);
    }
    
    // Inventory features
    auto position = get_inventory_position(quote.symbol);
    features.push_back(static_cast<float>(position.quantity));
    features.push_back(static_cast<float>(position.unrealized_pnl));
    
    // Time features
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    features.push_back(static_cast<float>(tm.tm_hour));
    features.push_back(static_cast<float>(tm.tm_wday));
    
    // Pad to expected size
    while (features.size() < 15) {
        features.push_back(0.0f);
    }
    
    auto tensor = torch::from_blob(features.data(), {1, static_cast<long>(features.size())}, torch::kFloat32);
    return tensor.clone();
}

void MarketMaker::on_regime_change(int new_regime) {
    current_regime_ = new_regime;
    update_regime_parameters(new_regime);
    
    std::cout << "Regime changed to: " << new_regime << " - updating market making parameters" << std::endl;
}

int MarketMaker::get_current_regime() const {
    return current_regime_.load();
}

void MarketMaker::update_regime_parameters(int regime_id) {
    std::shared_lock<std::shared_mutex> lock(symbols_mutex_);
    
    for (auto& [symbol, params] : symbol_params_) {
        // Update spread multiplier
        auto spread_it = params.regime_spread_multipliers.find(regime_id);
        if (spread_it != params.regime_spread_multipliers.end()) {
            params.base_spread_bps *= spread_it->second;
        }
        
        // Update size multiplier
        auto size_it = params.regime_size_multipliers.find(regime_id);
        if (size_it != params.regime_size_multipliers.end()) {
            params.base_quote_size *= size_it->second;
        }
        
        // Update inventory limit
        auto limit_it = params.regime_inventory_limits.find(regime_id);
        if (limit_it != params.regime_inventory_limits.end()) {
            params.max_inventory_usd *= limit_it->second;
        }
    }
}

// Private methods

void MarketMaker::initialize_background_threads() {
    // Start quote generator thread
    quote_generator_thread_ = std::thread(&MarketMaker::quote_generator_thread_func, this);
    
    // Start inventory manager thread
    if (config_.enable_inventory_management) {
        inventory_manager_thread_ = std::thread(&MarketMaker::inventory_manager_thread_func, this);
    }
    
    // Start worker threads
    if (config_.enable_parallel_quoting) {
        for (int i = 0; i < std::min(config_.max_concurrent_quotes, 4); ++i) {
            worker_threads_.emplace_back(&MarketMaker::worker_thread_func, this);
        }
    }
}

void MarketMaker::shutdown_background_threads() {
    running_ = false;
    
    // Wait for threads to finish
    if (quote_generator_thread_.joinable()) {
        quote_generator_thread_.join();
    }
    
    if (inventory_manager_thread_.joinable()) {
        inventory_manager_thread_.join();
    }
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    worker_threads_.clear();
}

void MarketMaker::quote_generator_thread_func() {
    while (running_) {
        try {
            auto symbols = get_active_symbols();
            
            for (const auto& symbol : symbols) {
                try {
                    Quote quote = generate_quote(symbol);
                    
                    // Check if we should post this quote
                    if (predict_fill_probability(quote) > 0.3 && 
                        predict_adverse_selection_risk(quote) < 0.5) {
                        post_quote(quote);
                    }
                    
                } catch (const std::exception& e) {
                    std::cerr << "Error generating quote for " << symbol << ": " << e.what() << std::endl;
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.quote_update_interval_ms));
            
        } catch (const std::exception& e) {
            std::cerr << "Error in quote generator thread: " << e.what() << std::endl;
        }
    }
}

void MarketMaker::inventory_manager_thread_func() {
    while (running_) {
        try {
            rebalance_inventory();
            update_inventory_metrics();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.inventory_update_interval_ms));
            
        } catch (const std::exception& e) {
            std::cerr << "Error in inventory manager thread: " << e.what() << std::endl;
        }
    }
}

void MarketMaker::worker_thread_func() {
    while (running_) {
        try {
            // Process quote queue
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
        } catch (const std::exception& e) {
            std::cerr << "Error in worker thread: " << e.what() << std::endl;
        }
    }
}

Quote MarketMaker::generate_static_spread_quote(const std::string& symbol) {
    Quote quote;
    quote.quote_id = "static_" + symbol + "_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    quote.symbol = symbol;
    quote.exchange = "default";
    quote.timestamp = std::chrono::system_clock::now();
    quote.is_active = true;
    
    auto params = get_symbol_params(symbol);
    auto market_it = current_quotes_.find(symbol);
    
    if (market_it != current_quotes_.end()) {
        double mid_price = (market_it->second.first + market_it->second.second) / 2.0;
        double half_spread = params.base_spread_bps * mid_price / 20000.0;
        
        quote.bid_price = mid_price - half_spread;
        quote.ask_price = mid_price + half_spread;
        quote.spread_bps = params.base_spread_bps;
        quote.bid_size = params.base_quote_size;
        quote.ask_size = params.base_quote_size;
    } else {
        // Default values if no market data
        quote.bid_price = 100.0;
        quote.ask_price = 100.01;
        quote.spread_bps = 10.0;
        quote.bid_size = 1.0;
        quote.ask_size = 1.0;
    }
    
    quote.expected_profit_bps = quote.spread_bps * 0.8; // Assume 80% capture
    quote.probability_of_fill = 0.5;
    
    return quote;
}

Quote MarketMaker::generate_dynamic_spread_quote(const std::string& symbol) {
    Quote quote = generate_static_spread_quote(symbol);
    
    auto params = get_symbol_params(symbol);
    
    // Adjust spread based on volatility
    double volatility_adjustment = 1.0;
    auto position = get_inventory_position(symbol);
    
    // Simple volatility adjustment (would use actual volatility in practice)
    volatility_adjustment = 1.0 + (std::abs(position.unrealized_pnl) / 1000.0) * params.spread_volatility_multiplier;
    
    quote.spread_bps *= volatility_adjustment;
    quote.expected_profit_bps = quote.spread_bps * 0.8;
    
    return quote;
}

Quote MarketMaker::generate_inventory_aware_quote(const std::string& symbol) {
    Quote quote = generate_dynamic_spread_quote(symbol);
    
    auto position = get_inventory_position(symbol);
    auto params = get_symbol_params(symbol);
    
    // Adjust quotes based on inventory
    double inventory_skew = calculate_inventory_skew(symbol);
    
    // Skew quotes to reduce inventory
    if (position.quantity > 0) { // Long inventory
        // Lower bid price, raise ask price to encourage selling
        double adjustment = inventory_skew * quote.spread_bps * quote.bid_price / 20000.0;
        quote.bid_price -= adjustment;
        quote.ask_price += adjustment;
    } else if (position.quantity < 0) { // Short inventory
        // Raise bid price, lower ask price to encourage buying
        double adjustment = inventory_skew * quote.spread_bps * quote.bid_price / 20000.0;
        quote.bid_price += adjustment;
        quote.ask_price -= adjustment;
    }
    
    quote.spread_bps = (quote.ask_price - quote.bid_price) / quote.bid_price * 10000.0;
    quote.inventory_risk_contribution = calculate_inventory_risk(symbol);
    
    return quote;
}

Quote MarketMaker::generate_adverse_selection_protected_quote(const std::string& symbol) {
    Quote quote = generate_inventory_aware_quote(symbol);
    
    // Protect against toxic order flow
    double toxicity_score = calculate_toxicity_score(symbol);
    
    if (toxicity_score > config_.adverse_selection_threshold) {
        // Widen spreads and reduce sizes
        quote.spread_bps *= 1.5;
        quote.bid_size *= 0.5;
        quote.ask_size *= 0.5;
        quote.adverse_selection_score = toxicity_score;
    }
    
    return quote;
}

Quote MarketMaker::generate_regime_aware_quote(const std::string& symbol) {
    Quote quote = generate_adverse_selection_protected_quote(symbol);
    
    // Adjust based on current regime
    int regime = current_regime_.load();
    auto params = get_symbol_params(symbol);
    
    auto regime_it = params.regime_spread_multipliers.find(regime);
    if (regime_it != params.regime_spread_multipliers.end()) {
        quote.spread_bps *= regime_it->second;
    }
    
    return quote;
}

Quote MarketMaker::generate_hybrid_quote(const std::string& symbol) {
    // Combine multiple strategies
    Quote static_quote = generate_static_spread_quote(symbol);
    Quote inventory_quote = generate_inventory_aware_quote(symbol);
    Quote regime_quote = generate_regime_aware_quote(symbol);
    
    // Weighted combination
    Quote quote;
    quote.quote_id = "hybrid_" + symbol + "_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    quote.symbol = symbol;
    quote.exchange = "default";
    quote.timestamp = std::chrono::system_clock::now();
    quote.is_active = true;
    
    // Weighted average of prices
    quote.bid_price = (static_quote.bid_price * 0.3 + 
                       inventory_quote.bid_price * 0.4 + 
                       regime_quote.bid_price * 0.3);
    quote.ask_price = (static_quote.ask_price * 0.3 + 
                       inventory_quote.ask_price * 0.4 + 
                       regime_quote.ask_price * 0.3);
    
    quote.spread_bps = (quote.ask_price - quote.bid_price) / quote.bid_price * 10000.0;
    quote.bid_size = inventory_quote.bid_size; // Use inventory-aware sizing
    quote.ask_size = inventory_quote.ask_size;
    quote.expected_profit_bps = quote.spread_bps * 0.8;
    quote.probability_of_fill = 0.6;
    quote.inventory_risk_contribution = inventory_quote.inventory_risk_contribution;
    quote.adverse_selection_score = regime_quote.adverse_selection_score;
    
    return quote;
}

double MarketMaker::calculate_optimal_spread(const std::string& symbol) {
    auto params = get_symbol_params(symbol);
    auto position = get_inventory_position(symbol);
    
    // Base spread
    double spread = params.base_spread_bps;
    
    // Inventory adjustment
    double inventory_skew = calculate_inventory_skew(symbol);
    spread *= (1.0 + std::abs(inventory_skew) * 0.5);
    
    // Volatility adjustment
    spread *= params.spread_volatility_multiplier;
    
    return spread;
}

double MarketMaker::calculate_inventory_skew(const std::string& symbol) {
    auto position = get_inventory_position(symbol);
    auto params = get_symbol_params(symbol);
    
    if (params.max_inventory_usd == 0) {
        return 0.0;
    }
    
    // Calculate inventory ratio
    double inventory_value = std::abs(position.quantity) * current_quotes_.at(symbol).first;
    double inventory_ratio = inventory_value / params.max_inventory_usd;
    
    // Return skew (-1 to 1)
    return std::clamp(position.quantity / (params.max_inventory_usd / current_quotes_.at(symbol).first), -1.0, 1.0);
}

double MarketMaker::calculate_toxicity_score(const std::string& symbol) {
    // Simplified toxicity calculation
    // In practice, would analyze order flow patterns
    
    auto position = get_inventory_position(symbol);
    double inventory_ratio = std::abs(position.quantity) / 1000.0; // Simplified
    
    return std::min(1.0, inventory_ratio * 0.5);
}

void MarketMaker::protect_against_adverse_selection(Quote& quote) {
    double toxicity_score = calculate_toxicity_score(quote.symbol);
    
    if (toxicity_score > config_.adverse_selection_threshold) {
        // Widen spreads
        quote.spread_bps *= (1.0 + toxicity_score);
        
        // Reduce sizes
        quote.bid_size *= (1.0 - toxicity_score * 0.5);
        quote.ask_size *= (1.0 - toxicity_score * 0.5);
        
        quote.adverse_selection_score = toxicity_score;
    }
}

void MarketMaker::apply_position_limits(Quote& quote) {
    auto params = get_symbol_params(quote.symbol);
    auto position = get_inventory_position(quote.symbol);
    
    // Check position limits
    if (!check_position_limits(quote.symbol, quote.bid_size)) {
        quote.bid_size = params.min_quote_size;
    }
    
    if (!check_position_limits(quote.symbol, -quote.ask_size)) {
        quote.ask_size = params.min_quote_size;
    }
}

void MarketMaker::apply_stop_loss(Quote& quote) {
    auto position = get_inventory_position(quote.symbol);
    auto params = get_symbol_params(quote.symbol);
    
    // Check if position is losing too much
    if (std::abs(position.unrealized_pnl) > params.stop_loss_threshold_bps * position.quantity * position.average_price / 10000.0) {
        // Widen spreads significantly to stop trading
        quote.spread_bps *= 5.0;
        quote.expected_profit_bps = 0.0;
    }
}

void MarketMaker::rebalance_inventory() {
    // Simplified inventory rebalancing
    // In practice, would use more sophisticated methods
    
    auto positions = get_all_inventory();
    double total_inventory_value = get_total_inventory_value();
    
    for (const auto& [symbol, position] : positions) {
        double position_ratio = std::abs(position.quantity) * current_quotes_.at(symbol).first / total_inventory_value;
        
        if (position_ratio > 0.3) { // 30% concentration
            // Cancel quotes for this symbol to reduce exposure
            cancel_all_quotes(symbol);
        }
    }
}

void MarketMaker::update_inventory_metrics() {
    std::shared_lock<std::shared_mutex> lock(inventory_mutex_);
    
    double total_inventory_value = 0.0;
    double total_risk = 0.0;
    
    for (const auto& [symbol, position] : inventory_positions_) {
        double position_value = std::abs(position.quantity) * current_quotes_.at(symbol).first;
        total_inventory_value += position_value;
        
        position.inventory_risk = calculate_inventory_risk(symbol);
        total_risk += position.inventory_risk;
    }
    
    lock.unlock();
    
    // Update performance metrics
    {
        std::lock_guard<std::mutex> perf_lock(performance_mutex_);
        performance_.avg_inventory_utilization = total_inventory_value / 100000.0; // Assuming 100K max
        performance_.max_inventory_utilization = std::max(performance_.max_inventory_utilization, 
                                                            performance_.avg_inventory_utilization);
    }
}

double MarketMaker::calculate_inventory_risk(const std::string& symbol) const {
    auto position = get_inventory_position(symbol);
    auto params = get_symbol_params(symbol);
    
    if (params.max_inventory_usd == 0) {
        return 0.0;
    }
    
    double position_value = std::abs(position.quantity) * current_quotes_.at(symbol).first;
    return position_value / params.max_inventory_usd;
}

double MarketMaker::calculate_total_inventory_risk() const {
    double total_risk = 0.0;
    auto positions = get_all_inventory();
    
    for (const auto& [symbol, position] : positions) {
        total_risk += calculate_inventory_risk(symbol);
    }
    
    return total_risk;
}

bool MarketMaker::is_symbol_active(const std::string& symbol) const {
    std::shared_lock<std::shared_mutex> lock(symbols_mutex_);
    
    auto it = active_symbols_.find(symbol);
    return (it != active_symbols_.end()) && it->second;
}

void MarketMaker::update_ml_model(const Quote& quote, bool filled, double profit_bps) {
    // This would update the ML model with new training data
    // For now, just log the result
    std::cout << "ML Model Update: " << quote.quote_id 
              << " -> " << (filled ? "FILLED" : "NOT FILLED") 
              << " (profit: " << profit_bps << " bps)" << std::endl;
}

// MarketMakerContext implementation

MarketMakerContext::MarketMakerContext(const MarketMakerConfig& config) : valid_(false) {
    market_maker_ = std::make_unique<MarketMaker>(config);
    // Note: Full initialization requires router and regime detector
}

MarketMakerContext::~MarketMakerContext() {
    if (market_maker_) {
        market_maker_->shutdown();
    }
}

MarketMaker& MarketMakerContext::get_market_maker() {
    if (!valid_ || !market_maker_) {
        throw std::runtime_error("Market maker not initialized");
    }
    return *market_maker_;
}

bool MarketMakerContext::is_valid() const {
    return valid_ && market_maker_;
}

} // namespace execution
} // namespace archneuronx
