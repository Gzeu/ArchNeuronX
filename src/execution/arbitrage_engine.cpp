/**
 * @file arbitrage_engine.cpp
 * @brief Statistical arbitrage engine implementation
 * @author ArchNeuronX Team
 * @date 2026-03-17
 */

#include "execution/arbitrage_engine.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>

namespace archneuronx {
namespace execution {

ArbitrageEngine::ArbitrageEngine(const ArbitrageEngineConfig& config)
    : config_(config), router_(nullptr), market_data_(nullptr), model_loaded_(false), running_(false) {
    
    performance_.total_opportunities_detected = 0;
    performance_.successful_executions = 0;
    performance_.failed_executions = 0;
    performance_.total_profit_bps = 0.0;
    performance_.avg_execution_time_ms = 0.0;
    performance_.avg_profit_per_trade_bps = 0.0;
    performance_.max_profit_bps = 0.0;
    performance_.max_loss_bps = 0.0;
    performance_.sharpe_ratio = 0.0;
    performance_.last_update = std::chrono::system_clock::now();
}

ArbitrageEngine::~ArbitrageEngine() {
    shutdown();
}

bool ArbitrageEngine::initialize(SmartOrderRouter& router, RealtimeFeed& market_data) {
    try {
        router_ = &router;
        market_data_ = &market_data;
        
        // Load ML prediction model if enabled
        if (config_.enable_ml_prediction && !config_.prediction_model_path.empty()) {
            if (!load_prediction_model(config_.prediction_model_path)) {
                std::cout << "Warning: Failed to load arbitrage prediction model" << std::endl;
            }
        }
        
        // Initialize currency triangles for triangular arbitrage
        if (config_.enable_triangular) {
            // Initialize common currency triangles
            std::vector<std::tuple<std::string, std::string, std::string>> triangles = {
                {"BTC", "ETH", "USDT"},
                {"BTC", "LTC", "USDT"},
                {"ETH", "LTC", "USDT"}
            };
            
            for (const auto& [c1, c2, c3] : triangles) {
                // Store triangle information for later use
            }
        }
        
        // Set up market data callbacks
        market_data_->set_tick_callback([this](const MarketTick& tick) {
            on_market_data_update(tick);
        });
        
        market_data_->set_orderbook_callback([this](const OrderBook& book) {
            on_orderbook_update(book);
        });
        
        // Initialize background threads
        initialize_background_threads();
        
        running_ = true;
        std::cout << "Arbitrage Engine initialized successfully" << std::endl;
        std::cout << "Cross-exchange arb: " << (config_.enable_cross_exchange ? "Enabled" : "Disabled") << std::endl;
        std::cout << "Pairs trading: " << (config_.enable_pairs_trading ? "Enabled" : "Disabled") << std::endl;
        std::cout << "Triangular arb: " << (config_.enable_triangular ? "Enabled" : "Disabled") << std::endl;
        std::cout << "ML prediction: " << (config_.enable_ml_prediction ? "Enabled" : "Disabled") << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing Arbitrage Engine: " << e.what() << std::endl;
        return false;
    }
}

void ArbitrageEngine::shutdown() {
    running_ = false;
    
    // Shutdown background threads
    shutdown_background_threads();
    
    std::cout << "Arbitrage Engine shutdown complete" << std::endl;
}

bool ArbitrageEngine::is_initialized() const {
    return running_ && router_ && market_data_;
}

std::vector<ArbitrageOpportunity> ArbitrageEngine::scan_opportunities() {
    std::vector<ArbitrageOpportunity> opportunities;
    
    try {
        // Scan different types of arbitrage based on configuration
        if (config_.enable_cross_exchange) {
            auto cross_exchange_arbs = detect_cross_exchange_arbitrage();
            for (const auto& arb : cross_exchange_arbs) {
                ArbitrageOpportunity opp;
                opp.opportunity_id = "cross_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
                opp.type = ArbitrageType::CROSS_EXCHANGE;
                opp.symbols = {arb.symbol};
                opp.exchanges = {arb.exchange_buy, arb.exchange_sell};
                opp.entry_prices = {arb.price_buy};
                opp.exit_prices = {arb.price_sell};
                opp.expected_profit_bps = arb.net_profit_bps;
                opp.confidence_score = 85;
                opp.discovery_time = std::chrono::system_clock::now();
                opp.expiry_time = opp.discovery_time + std::chrono::seconds(30);
                opp.is_active = true;
                
                if (arb.net_profit_bps > config_.min_profit_threshold_bps) {
                    opportunities.push_back(opp);
                }
            }
        }
        
        if (config_.enable_pairs_trading) {
            auto pairs_arbs = detect_pairs_trading_opportunities();
            for (const auto& pairs : pairs_arbs) {
                if (std::abs(pairs.z_score) > config_.z_score_threshold) {
                    ArbitrageOpportunity opp;
                    opp.opportunity_id = "pairs_" + pairs.symbol1 + "_" + pairs.symbol2;
                    opp.type = ArbitrageType::PAIRS_TRADING;
                    opp.symbols = {pairs.symbol1, pairs.symbol2};
                    opp.expected_profit_bps = std::abs(pairs.z_score) * 10; // Rough estimate
                    opp.confidence_score = static_cast<int>(std::min(95.0, std::abs(pairs.z_score) * 20));
                    opp.correlation_strength = pairs.correlation;
                    opp.discovery_time = std::chrono::system_clock::now();
                    opp.expiry_time = opp.discovery_time + std::chrono::minutes(5);
                    opp.is_active = true;
                    
                    if (opp.expected_profit_bps > config_.min_profit_threshold_bps) {
                        opportunities.push_back(opp);
                    }
                }
            }
        }
        
        if (config_.enable_triangular) {
            auto triangular_arbs = detect_triangular_arbitrage();
            for (const auto& tri : triangular_arbs) {
                if (tri.net_profit_bps > config_.min_profit_threshold_bps) {
                    ArbitrageOpportunity opp;
                    opp.opportunity_id = "triangular_" + tri.currency1 + "_" + tri.currency2 + "_" + tri.currency3;
                    opp.type = ArbitrageType::TRIANGULAR;
                    opp.symbols = {tri.currency1 + tri.currency2, tri.currency2 + tri.currency3, tri.currency3 + tri.currency1};
                    opp.expected_profit_bps = tri.net_profit_bps;
                    opp.confidence_score = static_cast<int>(std::min(90.0, tri.net_profit_bps * 5));
                    opp.discovery_time = std::chrono::system_clock::now();
                    opp.expiry_time = opp.discovery_time + std::chrono::seconds(10);
                    opp.is_active = true;
                    
                    opportunities.push_back(opp);
                }
            }
        }
        
        // Sort opportunities by expected profit
        std::sort(opportunities.begin(), opportunities.end(),
                  [](const ArbitrageOpportunity& a, const ArbitrageOpportunity& b) {
                      return a.expected_profit_bps > b.expected_profit_bps;
                  });
        
        // Update performance metrics
        {
            std::lock_guard<std::mutex> lock(performance_mutex_);
            performance_.total_opportunities_detected += opportunities.size();
        }
        
        std::cout << "Scanned " << opportunities.size() << " arbitrage opportunities" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error scanning opportunities: " << e.what() << std::endl;
    }
    
    return opportunities;
}

std::vector<CrossExchangeArbitrage> ArbitrageEngine::detect_cross_exchange_arbitrage() {
    std::vector<CrossExchangeArbitrage> opportunities;
    
    std::shared_lock<std::shared_mutex> lock(market_data_mutex_);
    
    // Get symbols available on multiple exchanges
    std::vector<std::string> symbols = {"BTCUSDT", "ETHUSDT"};
    
    for (const auto& symbol : symbols) {
        std::vector<std::pair<std::string, double>> exchange_prices;
        
        // Collect prices from different exchanges
        for (const auto& [exchange, price] : current_prices_) {
            // Extract symbol from exchange data (simplified)
            if (exchange.find(symbol) != std::string::npos) {
                exchange_prices.emplace_back(exchange, price);
            }
        }
        
        // Check for arbitrage opportunities
        if (exchange_prices.size() >= 2) {
            // Sort by price
            std::sort(exchange_prices.begin(), exchange_prices.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });
            
            double lowest_price = exchange_prices.front().second;
            double highest_price = exchange_prices.back().second;
            std::string buy_exchange = exchange_prices.front().first;
            std::string sell_exchange = exchange_prices.back().first;
            
            double spread_bps = (highest_price - lowest_price) / lowest_price * 10000;
            
            if (spread_bps > config_.min_profit_threshold_bps) {
                CrossExchangeArbitrage arb;
                arb.symbol = symbol;
                arb.exchange_buy = buy_exchange;
                arb.exchange_sell = sell_exchange;
                arb.price_buy = lowest_price;
                arb.price_sell = highest_price;
                arb.spread_bps = spread_bps;
                arb.execution_cost_bps = 10.0; // Estimated
                arb.net_profit_bps = spread_bps - arb.execution_cost_bps;
                arb.timestamp = std::chrono::system_clock::now();
                
                opportunities.push_back(arb);
            }
        }
    }
    
    return opportunities;
}

std::vector<PairsTradingStats> ArbitrageEngine::detect_pairs_trading_opportunities() {
    std::vector<PairsTradingStats> opportunities;
    
    std::shared_lock<std::shared_mutex> lock(stats_mutex_);
    
    // Check existing pairs relationships
    for (auto& [pair_key, stats] : pairs_relationships_) {
        // Update current z-score
        if (price_history_.find(stats.symbol1) != price_history_.end() &&
            price_history_.find(stats.symbol2) != price_history_.end()) {
            
            const auto& prices1 = price_history_[stats.symbol1];
            const auto& prices2 = price_history_[stats.symbol2];
            
            if (prices1.size() >= 30 && prices2.size() >= 30) {
                // Calculate current spread
                double hedge_ratio = stats.correlation; // Simplified
                double current_spread = prices1.back() - hedge_ratio * prices2.back();
                
                stats.z_score = calculate_z_score(current_spread, stats.spread_mean, stats.spread_std);
                stats.last_update = std::chrono::system_clock::now();
                
                opportunities.push_back(stats);
            }
        }
    }
    
    return opportunities;
}

std::vector<TriangularArbitrageData> ArbitrageEngine::detect_triangular_arbitrage() {
    std::vector<TriangularArbitrageData> opportunities;
    
    // Common currency triangles
    std::vector<std::tuple<std::string, std::string, std::string>> triangles = {
        {"BTC", "ETH", "USDT"},
        {"BTC", "LTC", "USDT"},
        {"ETH", "LTC", "USDT"}
    };
    
    std::shared_lock<std::shared_mutex> lock(market_data_mutex_);
    
    for (const auto& [c1, c2, c3] : triangles) {
        TriangularArbitrageData data;
        data.currency1 = c1;
        data.currency2 = c2;
        data.currency3 = c3;
        
        // Get exchange rates (simplified - in practice would use actual forex pairs)
        std::string pair12 = c1 + c2;
        std::string pair23 = c2 + c3;
        std::string pair31 = c3 + c1;
        
        auto it12 = current_prices_.find(pair12);
        auto it23 = current_prices_.find(pair23);
        auto it31 = current_prices_.find(pair31);
        
        if (it12 != current_prices_.end() && 
            it23 != current_prices_.end() && 
            it31 != current_prices_.end()) {
            
            data.rate12 = it12->second;
            data.rate23 = it23->second;
            data.rate31 = it31->second;
            
            data.implied_rate13 = data.rate12 * data.rate23;
            data.arbitrage_spread_bps = std::abs(data.implied_rate13 - data.rate31) / data.rate31 * 10000;
            data.execution_cost_bps = 15.0; // Estimated for three legs
            data.net_profit_bps = data.arbitrage_spread_bps - data.execution_cost_bps;
            data.timestamp = std::chrono::system_clock::now();
            
            if (data.net_profit_bps > config_.min_profit_threshold_bps) {
                opportunities.push_back(data);
            }
        }
    }
    
    return opportunities;
}

bool ArbitrageEngine::execute_arbitrage(const ArbitrageOpportunity& opportunity) {
    try {
        std::cout << "Executing arbitrage: " << opportunity.opportunity_id 
                  << " (type: " << static_cast<int>(opportunity.type) 
                  << ", profit: " << opportunity.expected_profit_bps << " bps)" << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        bool success = false;
        
        switch (opportunity.type) {
            case ArbitrageType::CROSS_EXCHANGE:
                success = execute_multi_leg_strategy(opportunity);
                break;
            case ArbitrageType::PAIRS_TRADING:
                success = execute_pairs_trade(pairs_relationships_[opportunity.symbols[0] + "_" + opportunity.symbols[1]]);
                break;
            case ArbitrageType::TRIANGULAR:
                // Would implement triangular execution
                success = execute_multi_leg_strategy(opportunity);
                break;
            default:
                std::cerr << "Unsupported arbitrage type: " << static_cast<int>(opportunity.type) << std::endl;
                return false;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Update performance metrics
        {
            std::lock_guard<std::mutex> lock(performance_mutex_);
            if (success) {
                performance_.successful_executions++;
                performance_.total_profit_bps += opportunity.expected_profit_bps;
                performance_.max_profit_bps = std::max(performance_.max_profit_bps, opportunity.expected_profit_bps);
            } else {
                performance_.failed_executions++;
                performance_.max_loss_bps = std::min(performance_.max_loss_bps, opportunity.expected_profit_bps);
            }
            
            // Update average execution time
            double total_time = performance_.avg_execution_time_ms * (performance_.successful_executions + performance_.failed_executions - 1);
            total_time += duration.count();
            performance_.avg_execution_time_ms = total_time / (performance_.successful_executions + performance_.failed_executions);
            
            // Update type counts
            performance_.type_counts[opportunity.type]++;
        }
        
        // Update ML model if enabled
        if (config_.enable_ml_prediction && model_loaded_) {
            update_ml_model(opportunity, success);
        }
        
        std::cout << "Arbitrage execution " << (success ? "SUCCESS" : "FAILED") 
                  << " (time: " << duration.count() << "ms)" << std::endl;
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "Error executing arbitrage: " << e.what() << std::endl;
        return false;
    }
}

bool ArbitrageEngine::execute_multi_leg_strategy(const ArbitrageOpportunity& opportunity) {
    // Simplified multi-leg execution
    // In practice, would coordinate simultaneous execution across multiple venues
    
    for (size_t i = 0; i < opportunity.symbols.size(); ++i) {
        OrderRequest order;
        order.order_id = opportunity.opportunity_id + "_leg_" + std::to_string(i);
        order.symbol = opportunity.symbols[i];
        order.side = (i % 2 == 0) ? "BUY" : "SELL";
        order.quantity = opportunity.quantities.empty() ? 1.0 : opportunity.quantities[i];
        order.price = opportunity.entry_prices.empty() ? 0.0 : opportunity.entry_prices[i];
        order.type = OrderType::MARKET;
        order.timestamp = std::chrono::system_clock::now();
        
        // Route order through smart router
        auto routing = router_->route_order(order);
        bool execution_success = router_->execute_order(order, routing);
        
        if (!execution_success) {
            std::cerr << "Failed to execute leg " << i << " of arbitrage" << std::endl;
            return false;
        }
    }
    
    return true;
}

void ArbitrageEngine::on_market_data_update(const MarketTick& tick) {
    std::unique_lock<std::shared_mutex> lock(market_data_mutex_);
    
    // Update current prices
    std::string key = tick.exchange + "_" + tick.symbol;
    current_prices_[key] = tick.price;
    
    // Update price history
    if (price_history_[tick.symbol].size() > 1000) {
        price_history_[tick.symbol].erase(price_history_[tick.symbol].begin());
    }
    price_history_[tick.symbol].push_back(tick.price);
    
    lock.unlock();
    
    // Check for immediate arbitrage opportunities
    if (tick.price > 0) {
        // Quick cross-exchange check
        for (const auto& [exchange_key, price] : current_prices_) {
            if (exchange_key.find(tick.symbol) != std::string::npos && exchange_key != key) {
                double spread = std::abs(tick.price - price) / price * 10000;
                if (spread > config_.min_profit_threshold_bps) {
                    // Found potential arbitrage
                    ArbitrageOpportunity opp;
                    opp.opportunity_id = "instant_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count());
                    opp.type = ArbitrageType::CROSS_EXCHANGE;
                    opp.symbols = {tick.symbol};
                    opp.exchanges = {tick.exchange, exchange_key};
                    opp.expected_profit_bps = spread - 10.0; // Subtract estimated costs
                    opp.confidence_score = 90;
                    opp.discovery_time = std::chrono::system_clock::now();
                    opp.expiry_time = opp.discovery_time + std::chrono::seconds(5);
                    opp.is_active = true;
                    
                    // Add to opportunity queue
                    {
                        std::lock_guard<std::mutex> queue_lock(opportunity_mutex_);
                        opportunity_queue_.push(opp);
                    }
                }
            }
        }
    }
}

void ArbitrageEngine::on_orderbook_update(const OrderBook& book) {
    std::unique_lock<std::shared_mutex> lock(market_data_mutex_);
    current_orderbooks_[book.symbol + "@" + book.exchange] = book;
}

void ArbitrageEngine::update_market_prices() {
    // This would be called periodically to update market data
    update_correlation_matrix();
    update_volatility_estimates();
}

void ArbitrageEngine::update_correlation_matrix() {
    std::shared_lock<std::shared_mutex> lock(market_data_mutex_);
    
    // Calculate correlations between major pairs
    std::vector<std::string> symbols = {"BTCUSDT", "ETHUSDT", "ADAUSDT"};
    
    for (size_t i = 0; i < symbols.size(); ++i) {
        for (size_t j = i + 1; j < symbols.size(); ++j) {
            const auto& prices1 = price_history_[symbols[i]];
            const auto& prices2 = price_history_[symbols[j]];
            
            if (prices1.size() >= 30 && prices2.size() >= 30) {
                double correlation = calculate_correlation(prices1, prices2);
                correlation_matrix_[symbols[i]][symbols[j]] = correlation;
                correlation_matrix_[symbols[j]][symbols[i]] = correlation;
                
                // Update pairs trading stats if correlation is high enough
                if (std::abs(correlation) > config_.correlation_threshold) {
                    std::string pair_key = symbols[i] + "_" + symbols[j];
                    
                    PairsTradingStats& stats = pairs_relationships_[pair_key];
                    stats.symbol1 = symbols[i];
                    stats.symbol2 = symbols[j];
                    stats.correlation = correlation;
                    
                    // Calculate cointegration
                    double cointegration = calculate_cointegration(prices1, prices2);
                    stats.cointegration_score = cointegration;
                    
                    // Calculate spread statistics
                    std::vector<double> spread_series;
                    for (size_t k = 0; k < std::min(prices1.size(), prices2.size()); ++k) {
                        spread_series.push_back(prices1[k] - correlation * prices2[k]);
                    }
                    
                    double mean = std::accumulate(spread_series.begin(), spread_series.end(), 0.0) / spread_series.size();
                    double variance = 0.0;
                    for (double spread : spread_series) {
                        variance += (spread - mean) * (spread - mean);
                    }
                    variance /= spread_series.size();
                    double std_dev = std::sqrt(variance);
                    
                    stats.spread_mean = mean;
                    stats.spread_std = std_dev;
                    stats.half_life_days = calculate_half_life(spread_series);
                    stats.last_update = std::chrono::system_clock::now();
                }
            }
        }
    }
}

void ArbitrageEngine::update_volatility_estimates() {
    std::shared_lock<std::shared_mutex> lock(market_data_mutex_);
    
    for (const auto& [symbol, prices] : price_history_) {
        if (prices.size() >= 14) {
            // Calculate daily returns
            std::vector<double> returns;
            for (size_t i = 1; i < prices.size(); ++i) {
                if (prices[i-1] > 0) {
                    returns.push_back((prices[i] - prices[i-1]) / prices[i-1]);
                }
            }
            
            if (!returns.empty()) {
                double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
                double variance = 0.0;
                for (double ret : returns) {
                    variance += (ret - mean) * (ret - mean);
                }
                variance /= returns.size();
                
                // Annualized volatility
                volatility_estimates_[symbol] = std::sqrt(variance * 365.0);
            }
        }
    }
}

double ArbitrageEngine::calculate_correlation(const std::vector<double>& series1, const std::vector<double>& series2) {
    if (series1.size() != series2.size() || series1.empty()) {
        return 0.0;
    }
    
    size_t n = series1.size();
    double mean1 = std::accumulate(series1.begin(), series1.end(), 0.0) / n;
    double mean2 = std::accumulate(series2.begin(), series2.end(), 0.0) / n;
    
    double numerator = 0.0;
    double var1 = 0.0;
    double var2 = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        double diff1 = series1[i] - mean1;
        double diff2 = series2[i] - mean2;
        numerator += diff1 * diff2;
        var1 += diff1 * diff1;
        var2 += diff2 * diff2;
    }
    
    if (var1 == 0.0 || var2 == 0.0) {
        return 0.0;
    }
    
    return numerator / std::sqrt(var1 * var2);
}

double ArbitrageEngine::calculate_cointegration(const std::vector<double>& series1, const std::vector<double>& series2) {
    // Simplified cointegration test
    // In practice, would use proper statistical tests like Engle-Granger
    double correlation = calculate_correlation(series1, series2);
    
    // High correlation suggests possible cointegration
    return std::abs(correlation);
}

double ArbitrageEngine::calculate_half_life(const std::vector<double>& spread_series) {
    if (spread_series.size() < 2) {
        return 0.0;
    }
    
    // Calculate mean reversion speed
    double mean = std::accumulate(spread_series.begin(), spread_series.end(), 0.0) / spread_series.size();
    
    double numerator = 0.0;
    double denominator = 0.0;
    
    for (size_t i = 1; i < spread_series.size(); ++i) {
        double spread_change = spread_series[i] - spread_series[i-1];
        double spread_deviation = spread_series[i-1] - mean;
        
        numerator += spread_change * spread_deviation;
        denominator += spread_deviation * spread_deviation;
    }
    
    if (denominator == 0.0) {
        return 0.0;
    }
    
    double mean_reversion_speed = -numerator / denominator;
    if (mean_reversion_speed <= 0.0) {
        return 0.0;
    }
    
    // Convert to days (assuming daily data)
    return std::log(2.0) / mean_reversion_speed;
}

double ArbitrageEngine::calculate_z_score(double spread, double mean, double std_dev) {
    if (std_dev == 0.0) {
        return 0.0;
    }
    return (spread - mean) / std_dev;
}

bool ArbitrageEngine::load_prediction_model(const std::string& model_path) {
    try {
        prediction_model_ = torch::jit::load(model_path);
        model_loaded_ = true;
        
        std::cout << "Arbitrage prediction model loaded: " << model_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading arbitrage prediction model: " << e.what() << std::endl;
        model_loaded_ = false;
        return false;
    }
}

double ArbitrageEngine::predict_success_probability(const ArbitrageOpportunity& opportunity) {
    if (!model_loaded_) {
        return 0.5; // Default probability
    }
    
    try {
        auto features = extract_arbitrage_features(opportunity);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(features);
        auto output = prediction_model_.forward(inputs);
        auto prediction = output.toTensor();
        
        return prediction.item<double>();
        
    } catch (const std::exception& e) {
        std::cerr << "Error predicting success probability: " << e.what() << std::endl;
        return 0.5;
    }
}

torch::Tensor ArbitrageEngine::extract_arbitrage_features(const ArbitrageOpportunity& opportunity) {
    std::vector<float> features;
    
    // Basic features
    features.push_back(static_cast<float>(opportunity.expected_profit_bps));
    features.push_back(static_cast<float>(opportunity.confidence_score));
    features.push_back(static_cast<float>(opportunity.volatility_ratio));
    features.push_back(static_cast<float>(opportunity.liquidity_score));
    features.push_back(static_cast<float>(opportunity.correlation_strength));
    
    // Type features
    features.push_back(static_cast<float>(static_cast<int>(opportunity.type)));
    
    // Time features
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    features.push_back(static_cast<float>(tm.tm_hour));
    features.push_back(static_cast<float>(tm.tm_wday));
    
    // Market condition features
    features.push_back(static_cast<float>(current_prices_.size()));
    features.push_back(static_cast<float>(volatility_estimates_.size()));
    
    // Pad to expected size
    while (features.size() < 20) {
        features.push_back(0.0f);
    }
    
    auto tensor = torch::from_blob(features.data(), {1, static_cast<long>(features.size())}, torch::kFloat32);
    return tensor.clone();
}

ArbitrageEngine::ArbitragePerformance ArbitrageEngine::get_performance_metrics() const {
    std::lock_guard<std::mutex> lock(performance_mutex_);
    return performance_;
}

void ArbitrageEngine::reset_performance_metrics() {
    std::lock_guard<std::mutex> lock(performance_mutex_);
    
    performance_.total_opportunities_detected = 0;
    performance_.successful_executions = 0;
    performance_.failed_executions = 0;
    performance_.total_profit_bps = 0.0;
    performance_.avg_execution_time_ms = 0.0;
    performance_.avg_profit_per_trade_bps = 0.0;
    performance_.max_profit_bps = 0.0;
    performance_.max_loss_bps = 0.0;
    performance_.sharpe_ratio = 0.0;
    performance_.type_counts.clear();
    performance_.last_update = std::chrono::system_clock::now();
}

// Private methods

void ArbitrageEngine::initialize_background_threads() {
    // Start opportunity scanner thread
    scanner_thread_ = std::thread(&ArbitrageEngine::opportunity_scanner_thread, this);
    
    // Start worker threads for execution
    if (config_.enable_parallel_execution) {
        for (int i = 0; i < std::min(config_.max_concurrent_arbitrages, 4); ++i) {
            worker_threads_.emplace_back(&ArbitrageEngine::worker_thread, this);
        }
    }
}

void ArbitrageEngine::shutdown_background_threads() {
    running_ = false;
    
    // Wake up any waiting threads
    {
        std::lock_guard<std::mutex> lock(opportunity_mutex_);
        // Clear queue to unblock workers
        std::queue<ArbitrageOpportunity> empty;
        opportunity_queue_.swap(empty);
    }
    
    // Wait for threads to finish
    if (scanner_thread_.joinable()) {
        scanner_thread_.join();
    }
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    worker_threads_.clear();
}

void ArbitrageEngine::opportunity_scanner_thread() {
    while (running_) {
        try {
            // Scan for opportunities
            auto opportunities = scan_opportunities();
            
            // Add high-confidence opportunities to execution queue
            for (const auto& opp : opportunities) {
                if (opp.confidence_score >= 80 && opp.expected_profit_bps >= config_.min_profit_threshold_bps) {
                    std::lock_guard<std::mutex> lock(opportunity_mutex_);
                    opportunity_queue_.push(opp);
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.opportunity_scan_interval_ms));
            
        } catch (const std::exception& e) {
            std::cerr << "Error in opportunity scanner: " << e.what() << std::endl;
        }
    }
}

void ArbitrageEngine::worker_thread() {
    while (running_) {
        try {
            std::unique_lock<std::mutex> lock(opportunity_mutex_);
            
            // Wait for opportunity
            if (opportunity_queue_.empty()) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            
            ArbitrageOpportunity opportunity = opportunity_queue_.front();
            opportunity_queue_.pop();
            lock.unlock();
            
            // Validate risk
            if (!validate_opportunity_risk(opportunity)) {
                std::cout << "Opportunity rejected due to risk constraints: " << opportunity.opportunity_id << std::endl;
                continue;
            }
            
            // Execute arbitrage
            execute_arbitrage(opportunity);
            
        } catch (const std::exception& e) {
            std::cerr << "Error in worker thread: " << e.what() << std::endl;
        }
    }
}

bool ArbitrageEngine::validate_opportunity_risk(const ArbitrageOpportunity& opportunity) {
    // Check profit threshold
    if (opportunity.expected_profit_bps < config_.min_profit_threshold_bps) {
        return false;
    }
    
    // Check position size
    if (opportunity.max_position_size > config_.max_position_size_usd) {
        return false;
    }
    
    // Check volatility
    if (opportunity.volatility_ratio > config_.max_volatility_ratio) {
        return false;
    }
    
    // Check liquidity
    if (opportunity.liquidity_score < 0.5) {
        return false;
    }
    
    // Check expiry
    auto now = std::chrono::system_clock::now();
    if (opportunity.expiry_time < now) {
        return false;
    }
    
    return true;
}

void ArbitrageEngine::update_ml_model(const ArbitrageOpportunity& opportunity, bool success) {
    // This would update the ML model with new training data
    // For now, just log the result
    std::cout << "ML Model Update: " << opportunity.opportunity_id 
              << " -> " << (success ? "SUCCESS" : "FAILURE") << std::endl;
}

// ArbitrageEngineContext implementation

ArbitrageEngineContext::ArbitrageEngineContext(const ArbitrageEngineConfig& config) : valid_(false) {
    engine_ = std::make_unique<ArbitrageEngine>(config);
    // Note: Full initialization requires router and market data
}

ArbitrageEngineContext::~ArbitrageEngineContext() {
    if (engine_) {
        engine_->shutdown();
    }
}

ArbitrageEngine& ArbitrageEngineContext::get_engine() {
    if (!valid_ || !engine_) {
        throw std::runtime_error("Arbitrage engine not initialized");
    }
    return *engine_;
}

bool ArbitrageEngineContext::is_valid() const {
    return valid_ && engine_;
}

} // namespace execution
} // namespace archneuronx
