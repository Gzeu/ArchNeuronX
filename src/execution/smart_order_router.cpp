/**
 * @file smart_order_router.cpp
 * @brief AI-optimized smart order routing implementation
 * @author ArchNeuronX Team
 * @date 2026-03-17
 */

#include "execution/smart_order_router.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>

namespace archneuronx {
namespace execution {

SmartOrderRouter::SmartOrderRouter(const SmartRouterConfig& config)
    : config_(config), current_regime_(0), model_loaded_(false), running_(false) {
    
    analytics_.total_orders_routed = 0;
    analytics_.successful_executions = 0;
    analytics_.avg_execution_time_ms = 0.0;
    analytics_.avg_cost_bps = 0.0;
    analytics_.avg_slippage_bps = 0.0;
}

SmartOrderRouter::~SmartOrderRouter() {
    shutdown();
}

bool SmartOrderRouter::initialize() {
    try {
        // Load ML routing model if enabled
        if (config_.enable_ml_routing && !config_.routing_model_path.empty()) {
            if (!load_routing_model(config_.routing_model_path)) {
                std::cout << "Warning: Failed to load ML routing model, using rule-based routing" << std::endl;
            }
        }
        
        // Initialize background threads
        initialize_background_threads();
        
        running_ = true;
        std::cout << "Smart Order Router initialized successfully" << std::endl;
        std::cout << "Venues: " << venues_.size() << std::endl;
        std::cout << "ML Routing: " << (config_.enable_ml_routing ? "Enabled" : "Disabled") << std::endl;
        std::cout << "Regime-Aware: " << (config_.enable_regime_aware_routing ? "Enabled" : "Disabled") << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing Smart Order Router: " << e.what() << std::endl;
        return false;
    }
}

void SmartOrderRouter::shutdown() {
    running_ = false;
    
    // Shutdown background threads
    shutdown_background_threads();
    
    std::cout << "Smart Order Router shutdown complete" << std::endl;
}

bool SmartOrderRouter::is_initialized() const {
    return running_ && !venues_.empty();
}

bool SmartOrderRouter::add_venue(const ExecutionVenue& venue) {
    std::unique_lock<std::shared_mutex> lock(venues_mutex_);
    
    // Initialize venue metrics
    VenueMetrics metrics;
    metrics.venue_id = venue.venue_id;
    metrics.avg_fill_rate = 0.8; // Default values
    metrics.avg_execution_time_ms = 50.0;
    metrics.price_improvement_bps = 0.0;
    metrics.liquidity_score = 0.7;
    metrics.cost_per_trade = venue.taker_fee_bps;
    metrics.reliability_score = 0.9;
    metrics.total_orders = 0;
    metrics.successful_fills = 0;
    metrics.last_update = std::chrono::system_clock::now();
    
    // Initialize regime performance
    for (int i = 0; i < 8; ++i) {
        metrics.regime_performance[i] = 0.5; // Default performance
    }
    
    venues_[venue.venue_id] = venue;
    venue_metrics_[venue.venue_id] = metrics;
    
    std::cout << "Added venue: " << venue.venue_name << " (" << venue.venue_id << ")" << std::endl;
    return true;
}

bool SmartOrderRouter::remove_venue(const std::string& venue_id) {
    std::unique_lock<std::shared_mutex> lock(venues_mutex_);
    
    auto venue_it = venues_.find(venue_id);
    if (venue_it == venues_.end()) {
        return false;
    }
    
    venues_.erase(venue_it);
    venue_metrics_.erase(venue_id);
    
    std::cout << "Removed venue: " << venue_id << std::endl;
    return true;
}

std::vector<std::string> SmartOrderRouter::get_available_venues() const {
    std::shared_lock<std::shared_mutex> lock(venues_mutex_);
    
    std::vector<std::string> venue_ids;
    for (const auto& [venue_id, venue] : venues_) {
        if (venue.is_connected) {
            venue_ids.push_back(venue_id);
        }
    }
    
    return venue_ids;
}

std::vector<ExecutionVenue> SmartOrderRouter::get_venue_details() const {
    std::shared_lock<std::shared_mutex> lock(venues_mutex_);
    
    std::vector<ExecutionVenue> venues;
    for (const auto& [venue_id, venue] : venues_) {
        venues.push_back(venue);
    }
    
    return venues;
}

RoutingDecision SmartOrderRouter::route_order(const OrderRequest& order) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Update analytics
    {
        std::lock_guard<std::mutex> lock(analytics_mutex_);
        analytics_.total_orders_routed++;
    }
    
    RoutingDecision decision;
    
    try {
        // Check if order is valid
        if (order.quantity <= 0) {
            throw std::invalid_argument("Invalid order quantity");
        }
        
        // Get current liquidity
        LiquidityAnalysis liquidity = analyze_liquidity(order.symbol);
        
        // Select routing algorithm based on configuration
        if (config_.enable_ml_routing && model_loaded_) {
            decision = route_with_ml(order);
        } else if (config_.enable_regime_aware_routing) {
            decision = route_regime_aware(order);
        } else {
            decision = route_with_rules(order);
        }
        
        // Calculate expected metrics
        auto venue_it = venue_metrics_.find(decision.venue_id);
        if (venue_it != venue_metrics_.end()) {
            const auto& metrics = venue_it->second;
            decision.expected_fill_rate = metrics.avg_fill_rate;
            decision.expected_execution_time_ms = metrics.avg_execution_time_ms;
            decision.expected_cost_bps = calculate_expected_cost(decision.venue_id, order);
            decision.expected_slippage_bps = calculate_expected_slippage(decision.venue_id, order);
        }
        
        decision.decision_time = std::chrono::system_clock::now();
        
        // Update venue usage count
        {
            std::lock_guard<std::mutex> lock(analytics_mutex_);
            analytics_.venue_usage_counts[decision.venue_id]++;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "Order routed to " << decision.venue_id 
                  << " (confidence: " << decision.confidence_score 
                  << ", time: " << duration.count() << "μs)" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error routing order: " << e.what() << std::endl;
        
        // Return fallback decision
        decision.venue_id = "fallback";
        decision.confidence_score = 0.0;
        decision.expected_fill_rate = 0.0;
    }
    
    return decision;
}

std::vector<RoutingDecision> SmartOrderRouter::route_order_multi_venue(const OrderRequest& order, int max_venues) {
    std::vector<RoutingDecision> decisions;
    
    // Get all suitable venues
    std::vector<std::string> suitable_venues;
    {
        std::shared_lock<std::shared_mutex> lock(venues_mutex_);
        for (const auto& [venue_id, venue] : venues_) {
            if (is_venue_suitable(venue_id, order)) {
                suitable_venues.push_back(venue_id);
            }
        }
    }
    
    // Sort venues by score
    std::sort(suitable_venues.begin(), suitable_venues.end(),
              [this, &order](const std::string& a, const std::string& b) {
                  return calculate_venue_score(a, order) > calculate_venue_score(b, order);
              });
    
    // Generate routing decisions for top venues
    int num_venues = std::min(max_venues, static_cast<int>(suitable_venues.size()));
    for (int i = 0; i < num_venues; ++i) {
        RoutingDecision decision;
        decision.venue_id = suitable_venues[i];
        decision.confidence_score = calculate_venue_score(suitable_venues[i], order);
        decision.expected_fill_rate = 0.8; // Default
        decision.expected_execution_time_ms = 50.0; // Default
        decision.expected_cost_bps = calculate_expected_cost(suitable_venues[i], order);
        decision.expected_slippage_bps = calculate_expected_slippage(suitable_venues[i], order);
        decision.decision_time = std::chrono::system_clock::now();
        
        decisions.push_back(decision);
    }
    
    return decisions;
}

bool SmartOrderRouter::execute_order(const OrderRequest& order, const RoutingDecision& routing) {
    try {
        // This would interface with the actual exchange execution system
        // For now, we'll simulate execution
        
        std::cout << "Executing order " << order.order_id 
                  << " on venue " << routing.venue_id << std::endl;
        
        // Simulate execution time
        std::this_thread::sleep_for(std::chrono::milliseconds(
            static_cast<int>(routing.expected_execution_time_ms)));
        
        // Update analytics
        {
            std::lock_guard<std::mutex> lock(analytics_mutex_);
            analytics_.successful_executions++;
            
            // Update average execution time
            double total_time = analytics_.avg_execution_time_ms * (analytics_.successful_executions - 1);
            total_time += routing.expected_execution_time_ms;
            analytics_.avg_execution_time_ms = total_time / analytics_.successful_executions;
        }
        
        // Update venue metrics
        update_venue_metrics(routing.venue_id, venue_metrics_[routing.venue_id]);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error executing order: " << e.what() << std::endl;
        return false;
    }
}

LiquidityAnalysis SmartOrderRouter::analyze_liquidity(const std::string& symbol) {
    std::shared_lock<std::shared_mutex> lock(liquidity_mutex_);
    
    auto it = liquidity_cache_.find(symbol);
    if (it != liquidity_cache_.end()) {
        // Check if cache is still valid
        auto now = std::chrono::system_clock::now();
        auto age = std::chrono::duration_cast<std::chrono::seconds>(now - it->second.timestamp);
        
        if (age.count() < config_.cache_ttl_seconds) {
            return it->second;
        }
    }
    
    lock.unlock();
    
    // Cache miss or expired, fetch fresh data
    LiquidityAnalysis analysis;
    analysis.symbol = symbol;
    analysis.total_liquidity = 1000000.0; // Default: 1M
    analysis.best_bid_price = 100.0;
    analysis.best_ask_price = 100.01;
    analysis.bid_size = 500000.0;
    analysis.ask_size = 500000.0;
    analysis.spread_bps = 1.0;
    analysis.market_depth = 0.8;
    analysis.timestamp = std::chrono::system_clock::now();
    
    // Generate sample orderbook
    for (int i = 0; i < 10; ++i) {
        double bid_price = analysis.best_bid_price - i * 0.001;
        double ask_price = analysis.best_ask_price + i * 0.001;
        double size = 100000.0 / (i + 1);
        
        analysis.orderbook.emplace_back(bid_price, size);
        analysis.orderbook.emplace_back(ask_price, size);
    }
    
    // Update cache
    {
        std::unique_lock<std::shared_mutex> write_lock(liquidity_mutex_);
        liquidity_cache_[symbol] = analysis;
    }
    
    return analysis;
}

std::unordered_map<std::string, LiquidityAnalysis> SmartOrderRouter::analyze_multi_symbol_liquidity(
    const std::vector<std::string>& symbols) {
    
    std::unordered_map<std::string, LiquidityAnalysis> results;
    
    for (const auto& symbol : symbols) {
        results[symbol] = analyze_liquidity(symbol);
    }
    
    return results;
}

void SmartOrderRouter::update_venue_metrics(const std::string& venue_id, const VenueMetrics& metrics) {
    std::unique_lock<std::shared_mutex> lock(venues_mutex_);
    
    auto it = venue_metrics_.find(venue_id);
    if (it != venue_metrics_.end()) {
        it->second = metrics;
        it->second.last_update = std::chrono::system_clock::now();
    }
}

std::unordered_map<std::string, VenueMetrics> SmartOrderRouter::get_all_venue_metrics() const {
    std::shared_lock<std::shared_mutex> lock(venues_mutex_);
    return venue_metrics_;
}

VenueMetrics SmartOrderRouter::get_venue_metrics(const std::string& venue_id) const {
    std::shared_lock<std::shared_mutex> lock(venues_mutex_);
    
    auto it = venue_metrics_.find(venue_id);
    return (it != venue_metrics_.end()) ? it->second : VenueMetrics{};
}

void SmartOrderRouter::set_current_regime(int regime_id) {
    current_regime_ = regime_id;
    std::cout << "Current regime set to: " << regime_id << std::endl;
}

int SmartOrderRouter::get_current_regime() const {
    return current_regime_.load();
}

void SmartOrderRouter::update_regime_performance(const std::string& venue_id, int regime_id, double performance) {
    std::unique_lock<std::shared_mutex> lock(venues_mutex_);
    
    auto it = venue_metrics_.find(venue_id);
    if (it != venue_metrics_.end()) {
        it->second.regime_performance[regime_id] = performance;
    }
}

bool SmartOrderRouter::load_routing_model(const std::string& model_path) {
    try {
        routing_model_ = torch::jit::load(model_path);
        model_loaded_ = true;
        
        std::cout << "ML routing model loaded: " << model_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading routing model: " << e.what() << std::endl;
        model_loaded_ = false;
        return false;
    }
}

torch::Tensor SmartOrderRouter::extract_routing_features(const OrderRequest& order, const LiquidityAnalysis& liquidity) {
    std::vector<float> features;
    
    // Order features
    features.push_back(static_cast<float>(order.quantity));
    features.push_back(static_cast<float>(order.price));
    features.push_back(static_cast<float>(order.side == "BUY" ? 1.0 : 0.0));
    features.push_back(static_cast<float>(order.type));
    
    // Liquidity features
    features.push_back(static_cast<float>(liquidity.total_liquidity));
    features.push_back(static_cast<float>(liquidity.spread_bps));
    features.push_back(static_cast<float>(liquidity.market_depth));
    features.push_back(static_cast<float>(liquidity.bid_size));
    features.push_back(static_cast<float>(liquidity.ask_size));
    
    // Regime feature
    features.push_back(static_cast<float>(current_regime_.load()));
    
    // Time features
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    features.push_back(static_cast<float>(tm.tm_hour));
    features.push_back(static_cast<float>(tm.tm_wday));
    
    // Pad to expected size
    while (features.size() < static_cast<size_t>(config_.feature_vector_size)) {
        features.push_back(0.0f);
    }
    
    // Create tensor
    auto tensor = torch::from_blob(features.data(), {1, static_cast<long>(features.size())}, torch::kFloat32);
    return tensor.clone();
}

void SmartOrderRouter::enable_parallel_routing(bool enable) {
    config_.enable_parallel_routing = enable;
}

void SmartOrderRouter::set_routing_threads(int num_threads) {
    config_.max_routing_threads = num_threads;
}

void SmartOrderRouter::clear_venue_cache() {
    std::unique_lock<std::shared_mutex> lock(liquidity_mutex_);
    liquidity_cache_.clear();
}

SmartOrderRouter::RoutingAnalytics SmartOrderRouter::get_routing_analytics() const {
    std::lock_guard<std::mutex> lock(analytics_mutex_);
    return analytics_;
}

void SmartOrderRouter::reset_analytics() {
    std::lock_guard<std::mutex> lock(analytics_mutex_);
    
    analytics_.total_orders_routed = 0;
    analytics_.successful_executions = 0;
    analytics_.avg_execution_time_ms = 0.0;
    analytics_.avg_cost_bps = 0.0;
    analytics_.avg_slippage_bps = 0.0;
    analytics_.venue_usage_counts.clear();
    analytics_.regime_performance.clear();
}

// Private methods

void SmartOrderRouter::initialize_background_threads() {
    if (config_.enable_parallel_routing && config_.max_routing_threads > 0) {
        for (int i = 0; i < config_.max_routing_threads; ++i) {
            routing_threads_.emplace_back(&SmartOrderRouter::routing_worker_thread, this);
        }
    }
    
    // Start monitoring threads
    routing_threads_.emplace_back([this] {
        while (running_) {
            update_liquidity_cache();
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.liquidity_update_interval_ms));
        }
    });
    
    routing_threads_.emplace_back([this] {
        while (running_) {
            update_venue_health();
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.venue_health_check_interval_ms));
        }
    });
}

void SmartOrderRouter::shutdown_background_threads() {
    running_ = false;
    
    // Wake up any waiting threads
    queue_cv_.notify_all();
    
    // Wait for threads to finish
    for (auto& thread : routing_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    routing_threads_.clear();
}

void SmartOrderRouter::routing_worker_thread() {
    while (running_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this] { return !routing_queue_.empty() || !running_; });
        
        if (!running_) break;
        
        if (!routing_queue_.empty()) {
            OrderRequest order = routing_queue_.front();
            routing_queue_.pop();
            lock.unlock();
            
            // Process order
            route_order(order);
        }
    }
}

RoutingDecision SmartOrderRouter::route_with_ml(const OrderRequest& order) {
    RoutingDecision decision;
    
    try {
        // Get liquidity analysis
        LiquidityAnalysis liquidity = analyze_liquidity(order.symbol);
        
        // Extract features
        auto features = extract_routing_features(order, liquidity);
        
        // Get ML prediction
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(features);
        auto output = routing_model_.forward(inputs);
        auto predictions = output.toTensor();
        
        // Find best venue
        auto venues = get_available_venues();
        if (venues.empty()) {
            throw std::runtime_error("No available venues");
        }
        
        auto pred_accessor = predictions.accessor<float, 2>();
        int best_venue_idx = 0;
        float best_score = pred_accessor[0][0];
        
        for (size_t i = 1; i < venues.size() && i < static_cast<size_t>(predictions.size(1)); ++i) {
            if (pred_accessor[0][i] > best_score) {
                best_score = pred_accessor[0][i];
                best_venue_idx = i;
            }
        }
        
        decision.venue_id = venues[best_venue_idx];
        decision.confidence_score = best_score;
        decision.routing_reasons = generate_routing_reasons(decision.venue_id, order);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in ML routing: " << e.what() << std::endl;
        // Fallback to rule-based routing
        return route_with_rules(order);
    }
    
    return decision;
}

RoutingDecision SmartOrderRouter::route_with_rules(const OrderRequest& order) {
    RoutingDecision decision;
    
    // Multi-criteria decision making
    std::string best_venue = select_best_venue_balanced(order);
    
    decision.venue_id = best_venue;
    decision.confidence_score = calculate_venue_score(best_venue, order);
    decision.routing_reasons = generate_routing_reasons(best_venue, order);
    
    return decision;
}

RoutingDecision SmartOrderRouter::route_regime_aware(const OrderRequest& order) {
    RoutingDecision decision;
    
    // Get venue performance for current regime
    std::shared_lock<std::shared_mutex> lock(venues_mutex_);
    
    std::string best_venue;
    double best_score = 0.0;
    int current_regime = current_regime_.load();
    
    for (const auto& [venue_id, metrics] : venue_metrics_) {
        if (!is_venue_suitable(venue_id, order)) {
            continue;
        }
        
        // Weight by regime performance
        double regime_perf = 0.5; // Default
        auto regime_it = metrics.regime_performance.find(current_regime);
        if (regime_it != metrics.regime_performance.end()) {
            regime_perf = regime_it->second;
        }
        
        double score = calculate_venue_score(venue_id, order) * (1.0 + regime_perf);
        
        if (score > best_score) {
            best_score = score;
            best_venue = venue_id;
        }
    }
    
    decision.venue_id = best_venue;
    decision.confidence_score = best_score;
    decision.routing_reasons = generate_routing_reasons(best_venue, order);
    
    return decision;
}

std::string SmartOrderRouter::select_best_venue_speed(const OrderRequest& order) {
    std::shared_lock<std::shared_mutex> lock(venues_mutex_);
    
    std::string best_venue;
    double min_time = std::numeric_limits<double>::max();
    
    for (const auto& [venue_id, metrics] : venue_metrics_) {
        if (!is_venue_suitable(venue_id, order)) {
            continue;
        }
        
        if (metrics.avg_execution_time_ms < min_time) {
            min_time = metrics.avg_execution_time_ms;
            best_venue = venue_id;
        }
    }
    
    return best_venue;
}

std::string SmartOrderRouter::select_best_venue_cost(const OrderRequest& order) {
    std::shared_lock<std::shared_mutex> lock(venues_mutex_);
    
    std::string best_venue;
    double min_cost = std::numeric_limits<double>::max();
    
    for (const auto& [venue_id, metrics] : venue_metrics_) {
        if (!is_venue_suitable(venue_id, order)) {
            continue;
        }
        
        double cost = calculate_expected_cost(venue_id, order);
        if (cost < min_cost) {
            min_cost = cost;
            best_venue = venue_id;
        }
    }
    
    return best_venue;
}

std::string SmartOrderRouter::select_best_venue_liquidity(const OrderRequest& order) {
    std::shared_lock<std::shared_mutex> lock(venues_mutex_);
    
    std::string best_venue;
    double max_liquidity = 0.0;
    
    for (const auto& [venue_id, metrics] : venue_metrics_) {
        if (!is_venue_suitable(venue_id, order)) {
            continue;
        }
        
        if (metrics.liquidity_score > max_liquidity) {
            max_liquidity = metrics.liquidity_score;
            best_venue = venue_id;
        }
    }
    
    return best_venue;
}

std::string SmartOrderRouter::select_best_venue_balanced(const OrderRequest& order) {
    std::shared_lock<std::shared_mutex> lock(venues_mutex_);
    
    std::string best_venue;
    double best_score = 0.0;
    
    for (const auto& [venue_id, metrics] : venue_metrics_) {
        if (!is_venue_suitable(venue_id, order)) {
            continue;
        }
        
        double score = calculate_venue_score(venue_id, order);
        if (score > best_score) {
            best_score = score;
            best_venue = venue_id;
        }
    }
    
    return best_venue;
}

void SmartOrderRouter::update_liquidity_cache() {
    auto symbols = std::vector<std::string>{"BTCUSDT", "ETHUSDT", "ADAUSDT"};
    
    for (const auto& symbol : symbols) {
        analyze_liquidity(symbol);
    }
}

void SmartOrderRouter::update_venue_health() {
    std::shared_lock<std::shared_mutex> lock(venues_mutex_);
    
    for (auto& [venue_id, metrics] : venue_metrics_) {
        // Simulate health check
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.8, 1.0);
        
        metrics.reliability_score = dis(gen);
        metrics.last_update = std::chrono::system_clock::now();
    }
}

double SmartOrderRouter::calculate_venue_score(const std::string& venue_id, const OrderRequest& order) {
    auto it = venue_metrics_.find(venue_id);
    if (it == venue_metrics_.end()) {
        return 0.0;
    }
    
    const auto& metrics = it->second;
    
    // Multi-criteria scoring
    double speed_score = 1.0 / (1.0 + metrics.avg_execution_time_ms / 100.0);
    double cost_score = 1.0 / (1.0 + metrics.cost_per_trade / 10.0);
    double fill_score = metrics.avg_fill_rate;
    double liquidity_score = metrics.liquidity_score;
    double reliability_score = metrics.reliability_score;
    
    // Weighted combination
    double total_score = (config_.speed_weight * speed_score +
                         config_.cost_weight * cost_score +
                         config_.fill_rate_weight * fill_score +
                         config_.liquidity_weight * liquidity_score) * reliability_score;
    
    return total_score;
}

bool SmartOrderRouter::is_venue_suitable(const std::string& venue_id, const OrderRequest& order) {
    auto it = venues_.find(venue_id);
    if (it == venues_.end()) {
        return false;
    }
    
    const auto& venue = it->second;
    
    // Check if venue supports the symbol
    if (std::find(venue.supported_symbols.begin(), venue.supported_symbols.end(), order.symbol) 
        == venue.supported_symbols.end()) {
        return false;
    }
    
    // Check if venue supports the order type
    if (order.type == OrderType::MARKET && !venue.supports_market_orders) {
        return false;
    }
    
    if (order.type == OrderType::LIMIT && !venue.supports_limit_orders) {
        return false;
    }
    
    // Check order size limits
    if (order.quantity < venue.min_order_size || order.quantity > venue.max_order_size) {
        return false;
    }
    
    // Check if venue is connected and reliable
    if (!venue.is_connected || venue_metrics_[venue_id].reliability_score < 0.5) {
        return false;
    }
    
    return true;
}

double SmartOrderRouter::calculate_expected_cost(const std::string& venue_id, const OrderRequest& order) {
    auto it = venues_.find(venue_id);
    if (it == venues_.end()) {
        return 0.0;
    }
    
    const auto& venue = it->venue;
    
    // Base fee
    double fee_bps = (order.type == OrderType::LIMIT) ? venue.maker_fee_bps : venue.taker_fee_bps;
    
    // Expected slippage (simplified)
    double slippage_bps = 0.5; // Default estimate
    
    return fee_bps + slippage_bps;
}

double SmartOrderRouter::calculate_expected_slippage(const std::string& venue_id, const OrderRequest& order) {
    // Simplified slippage calculation
    // In practice, would use order book depth and market impact models
    
    auto liquidity = analyze_liquidity(order.symbol);
    double market_impact = (order.quantity / liquidity.total_liquidity) * 10.0; // Rough estimate
    
    return market_impact;
}

std::vector<std::string> SmartOrderRouter::generate_routing_reasons(const std::string& venue_id, 
                                                                  const OrderRequest& order) {
    std::vector<std::string> reasons;
    
    auto it = venue_metrics_.find(venue_id);
    if (it == venue_metrics_.end()) {
        reasons.push_back("Default venue");
        return reasons;
    }
    
    const auto& metrics = it->second;
    
    if (metrics.avg_execution_time_ms < 25.0) {
        reasons.push_back("Fast execution");
    }
    
    if (metrics.cost_per_trade < 2.0) {
        reasons.push_back("Low cost");
    }
    
    if (metrics.avg_fill_rate > 0.9) {
        reasons.push_back("High fill rate");
    }
    
    if (metrics.liquidity_score > 0.8) {
        reasons.push_back("Good liquidity");
    }
    
    if (reasons.empty()) {
        reasons.push_back("Balanced selection");
    }
    
    return reasons;
}

// SmartRouterContext implementation

SmartRouterContext::SmartRouterContext(const SmartRouterConfig& config) : valid_(false) {
    router_ = std::make_unique<SmartOrderRouter>(config);
    valid_ = router_->initialize();
}

SmartRouterContext::~SmartRouterContext() {
    if (router_) {
        router_->shutdown();
    }
}

SmartOrderRouter& SmartRouterContext::get_router() {
    if (!valid_ || !router_) {
        throw std::runtime_error("Smart router not initialized");
    }
    return *router_;
}

bool SmartRouterContext::is_valid() const {
    return valid_ && router_;
}

} // namespace execution
} // namespace archneuronx
