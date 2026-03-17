/**
 * @file openclaw_integration.cpp
 * @brief OpenCLaw integration implementation
 * @author George Pricop
 * @date 2025-10-02
 */

#include "trading/openclaw_integration.hpp"
#include "openclaw/openclaw_core.hpp"
#include "core/logger.hpp"
#include "data/data_aggregator.hpp"
#include "models/neural_networks.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>

namespace ArchNeuronX {
namespace Trading {

/**
 * @brief Enhanced OpenCLaw Integration with Official Core
 */
class OpenCLawIntegration {
public:
    /**
     * @brief Configuration for OpenCLaw integration
     */
    struct Config {
        bool enable_smart_order_routing = true;
        bool enable_market_microstructure = true;
        bool enable_adaptive_execution = true;
        bool enable_ml_signal_filtering = true;
        bool enable_regime_detection = true;
        double max_slippage_bps = 5.0;
        double min_fill_rate = 0.95;
        int execution_timeout_ms = 5000;
        std::vector<std::string> supported_venues = {
            "binance", "coinbase", "kraken", "bybit", "okx", "huobi"
        };
    };

    /**
     * @brief Constructor
     * @param config Integration configuration
     */
    explicit OpenCLawIntegration(const Config& config);

    /**
     * @brief Initialize OpenCLaw integration
     * @return True if initialization successful
     */
    bool initialize();

    /**
     * @brief Process trading signals from OpenCLaw
     * @param signals Vector of OpenCLaw signals
     * @return Processed signals with ArchNeuronX enhancements
     */
    std::vector<OpenCLawSignal> process_signals(
        const std::vector<OpenCLawSignal>& signals);

    /**
     * @brief Execute advanced order with OpenCLaw routing
     * @param order Advanced order to execute
     * @return Execution result
     */
    bool execute_advanced_order(const AdvancedOrder& order);

    /**
     * @brief Optimize portfolio allocation using OpenCLaw algorithms
     * @param current_allocations Current portfolio state
     * @param signals Current trading signals
     * @return Optimized allocations
     */
    std::vector<PortfolioAllocation> optimize_portfolio(
        const std::vector<PortfolioAllocation>& current_allocations,
        const std::vector<OpenCLawSignal>& signals);

    /**
     * @brief Calculate comprehensive risk metrics
     * @param portfolio Current portfolio state
     * @param market_data Market data for risk calculation
     * @return Risk metrics
     */
    RiskMetrics calculate_risk_metrics(
        const std::vector<PortfolioAllocation>& portfolio,
        const std::map<std::string, double>& market_data);

    /**
     * @brief Detect current market regime
     * @param market_data Recent market data
     * @return Detected market regime
     */
    MarketRegime detect_market_regime(
        const std::map<std::string, double>& market_data);

    /**
     * @brief Apply smart order routing
     * @param order Order to route
     * @return Best venue and execution strategy
     */
    std::pair<std::string, std::string> route_smart_order(
        const AdvancedOrder& order);

    /**
     * @brief Apply market microstructure analysis
     * @param order_book Current order book
     * @return Optimal execution strategy
     */
    std::string analyze_market_microstructure(
        const std::map<std::string, std::vector<double>>& order_book);

    /**
     * @brief Get integration status
     * @return Status information
     */
    std::map<std::string, std::string> get_status() const;

private:
    Config config_;
    bool initialized_;
    
    // OpenCLaw Core Engine
    std::unique_ptr<OpenCLawCore::OpenCLawEngine> openclaw_engine_;
    
    // Market microstructure analysis
    struct MicrostructureData {
        double bid_ask_spread;
        double order_flow_imbalance;
        double volume_weighted_price;
        double liquidity_score;
        std::chrono::system_clock::time_point last_update;
    };
    
    std::map<std::string, MicrostructureData> microstructure_data_;
    
    // Smart order routing cache
    struct VenueMetrics {
        double avg_fill_rate;
        double avg_slippage_bps;
        double latency_ms;
        double liquidity_score;
        std::string venue_name;
    };
    
    std::map<std::string, VenueMetrics> venue_metrics_;
    
    // Private methods
    void update_venue_metrics(const std::string& venue, 
                           double fill_rate, 
                           double slippage_bps,
                           double latency_ms);
    
    double calculate_optimal_size(const std::string& symbol,
                               double target_size,
                               MarketRegime regime);
    
    std::vector<std::string> select_best_venues(const std::string& symbol,
                                              OrderType order_type);
    
    bool validate_order_risk(const AdvancedOrder& order,
                          const RiskMetrics& risk_metrics);
    
    void update_microstructure_data(const std::string& symbol,
                               const std::map<std::string, double>& market_data);
};

OpenCLawIntegration::OpenCLawIntegration(const Config& config) 
    : config_(config), initialized_(false) {
}

bool OpenCLawIntegration::initialize() {
    std::cout << "🚀 Initializing Enhanced OpenCLaw Integration..." << std::endl;
    
    try {
        // Initialize OpenCLaw Core Engine
        openclaw_engine_ = OpenCLawCore::OpenCLawFactory::create_engine(config_);
        if (!openclaw_engine_->initialize()) {
            std::cerr << "❌ Failed to initialize OpenCLaw Core Engine" << std::endl;
            return false;
        }
        
        initialized_ = true;
        std::cout << "✅ Enhanced OpenCLaw Integration initialized successfully" << std::endl;
        std::cout << "📊 Supported venues: ";
        for (const auto& venue : config_.supported_venues) {
            std::cout << venue << " ";
        }
        std::cout << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to initialize OpenCLaw Integration: " << e.what() << std::endl;
        return false;
    }
}

std::vector<OpenCLawSignal> OpenCLawIntegration::process_signals(
    const std::vector<OpenCLawSignal>& signals) {
    
    std::vector<OpenCLawSignal> processed_signals;
    
    for (const auto& signal : signals) {
        OpenCLawSignal processed = signal;
        
        // Apply ML signal filtering if enabled
        if (config_.enable_ml_signal_filtering) {
            processed = OpenCLawSignalProcessor::apply_ml_filtering({signal})[0];
        }
        
        // Apply regime-based adjustments
        if (config_.enable_regime_detection) {
            // Adjust confidence based on regime
            if (processed.regime == MarketRegime::HIGH_VOLATILITY) {
                processed.confidence *= 0.8;  // Reduce confidence in high volatility
            } else if (processed.regime == MarketRegime::LOW_VOLATILITY) {
                processed.confidence *= 1.1;  // Increase confidence in low volatility
            }
        }
        
        processed_signals.push_back(processed);
    }
    
    std::cout << "📈 Processed " << processed_signals.size() 
              << " signals with OpenCLaw Core enhancements" << std::endl;
    
    return processed_signals;
}

bool OpenCLawIntegration::execute_advanced_order(const AdvancedOrder& order) {
    if (!initialized_) {
        std::cerr << "❌ OpenCLaw Integration not initialized" << std::endl;
        return false;
    }
    
    std::cout << "🔄 Executing advanced order with OpenCLaw Core: " << order.symbol 
              << " " << static_cast<int>(order.order_type) << std::endl;
    
    try {
        // Convert ArchNeuronX order to OpenCLaw signal
        OpenCLawCore::TradingSignal signal;
        signal.symbol = order.symbol;
        signal.timestamp = std::chrono::system_clock::now();
        auto [best_venue, execution_strategy] = route_smart_order(order);
        
        std::cout << "📍 Routing to venue: " << best_venue 
                  << " with strategy: " << execution_strategy << std::endl;
        
        // Simulate order execution (in real implementation, this would call venue APIs)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Update venue metrics
        double fill_rate = 0.95 + (rand() % 10) / 100.0;  // Simulate 95-100% fill rate
        double slippage = 1.0 + (rand() % 5);  // Simulate 1-5 bps slippage
        double latency = 30.0 + (rand() % 40);  // Simulate 30-70ms latency
        
        update_venue_metrics(best_venue, fill_rate, slippage, latency);
        
        std::cout << "✅ Order executed successfully" << std::endl;
        std::cout << "📊 Fill rate: " << fill_rate * 100 << "%" << std::endl;
        std::cout << "📊 Slippage: " << slippage << " bps" << std::endl;
        std::cout << "📊 Latency: " << latency << " ms" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Order execution failed: " << e.what() << std::endl;
        return false;
    }
}

std::vector<PortfolioAllocation> OpenCLawIntegration::optimize_portfolio(
    const std::vector<PortfolioAllocation>& current_allocations,
    const std::vector<OpenCLawSignal>& signals) {
    
    std::vector<PortfolioAllocation> optimized_allocations = current_allocations;
    
    std::cout << "🎯 Optimizing portfolio allocation..." << std::endl;
    
    // Calculate total portfolio value
    double total_value = 0.0;
    for (const auto& allocation : current_allocations) {
        total_value += std::abs(allocation.current_position) * allocation.current_position;
    }
    
    // Apply Kelly criterion for position sizing
    for (size_t i = 0; i < optimized_allocations.size(); ++i) {
        auto& allocation = optimized_allocations[i];
        
        // Find corresponding signal
        for (const auto& signal : signals) {
            if (signal.signal_type != SignalType::HOLD && 
                allocation.allocation_percent > 0.01) {  // Only active positions
                
                // Calculate Kelly fraction
                double win_rate = 0.55;  // Assume 55% win rate
                double avg_win = 0.02;   // 2% average win
                double avg_loss = 0.01;  // 1% average loss
                
                double kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win;
                kelly_fraction = std::max(0.0, std::min(0.25, kelly_fraction));  // Cap at 25%
                
                // Apply Kelly sizing
                double optimal_size = total_value * kelly_fraction * allocation.allocation_percent;
                allocation.target_position = optimal_size;
                
                std::cout << "📊 " << allocation.symbol 
                          << " Kelly fraction: " << kelly_fraction * 100 << "%" << std::endl;
                std::cout << "📊 " << allocation.symbol 
                          << " Target position: " << optimal_size << std::endl;
            }
        }
    }
    
    return optimized_allocations;
}

RiskMetrics OpenCLawIntegration::calculate_risk_metrics(
    const std::vector<PortfolioAllocation>& portfolio,
    const std::map<std::string, double>& market_data) {
    
    RiskMetrics metrics = {};
    
    std::cout << "🛡️ Calculating comprehensive risk metrics..." << std::endl;
    
    // Calculate portfolio returns
    std::vector<double> returns;
    for (const auto& allocation : portfolio) {
        if (allocation.daily_pnl != 0) {
            double portfolio_value = std::abs(allocation.current_position) * allocation.current_position;
            returns.push_back(allocation.daily_pnl / portfolio_value);
        }
    }
    
    if (returns.empty()) {
        return metrics;
    }
    
    // Calculate basic statistics
    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double variance = 0.0;
    for (double ret : returns) {
        variance += (ret - mean_return) * (ret - mean_return);
    }
    variance /= returns.size();
    double std_dev = std::sqrt(variance);
    
    // Calculate Sharpe ratio (assuming 252 trading days)
    metrics.sharpe_ratio = (std_dev > 0) ? (mean_return * 252) / (std_dev * std::sqrt(252)) : 0.0;
    
    // Calculate Sortino ratio (downside deviation only)
    double downside_variance = 0.0;
    for (double ret : returns) {
        if (ret < mean_return) {
            downside_variance += (ret - mean_return) * (ret - mean_return);
        }
    }
    downside_variance /= returns.size();
    double downside_std = std::sqrt(downside_variance);
    metrics.sortino_ratio = (downside_std > 0) ? (mean_return * 252) / (downside_std * std::sqrt(252)) : 0.0;
    
    // Calculate maximum drawdown
    double peak = 0.0;
    double max_drawdown = 0.0;
    for (const auto& allocation : portfolio) {
        double cumulative_pnl = allocation.daily_pnl;
        if (cumulative_pnl > peak) {
            peak = cumulative_pnl;
        } else {
            double drawdown = (peak - cumulative_pnl) / peak;
            max_drawdown = std::max(max_drawdown, drawdown);
        }
    }
    metrics.max_drawdown = max_drawdown;
    
    // Calculate Calmar ratio
    metrics.calmar_ratio = (max_drawdown > 0) ? (mean_return * 252) / max_drawdown : 0.0;
    
    // Calculate VaR 95% (simplified)
    std::sort(returns.begin(), returns.end());
    size_t var_index = static_cast<size_t>(returns.size() * 0.05);
    metrics.var_95 = returns.empty() ? 0.0 : -returns[var_index];
    
    // Calculate Expected Shortfall
    double expected_shortfall = 0.0;
    int shortfall_count = 0;
    for (size_t i = var_index; i < returns.size(); ++i) {
        expected_shortfall += returns[i];
        shortfall_count++;
    }
    metrics.expected_shortfall = shortfall_count > 0 ? expected_shortfall / shortfall_count : 0.0;
    
    std::cout << "📊 Risk Metrics Calculated:" << std::endl;
    std::cout << "   Sharpe Ratio: " << metrics.sharpe_ratio << std::endl;
    std::cout << "   Sortino Ratio: " << metrics.sortino_ratio << std::endl;
    std::cout << "   Max Drawdown: " << metrics.max_drawdown * 100 << "%" << std::endl;
    std::cout << "   VaR 95%: " << metrics.var_95 * 100 << "%" << std::endl;
    std::cout << "   Expected Shortfall: " << metrics.expected_shortfall * 100 << "%" << std::endl;
    
    return metrics;
}

MarketRegime OpenCLawIntegration::detect_market_regime(
    const std::map<std::string, double>& market_data) {
    
    std::cout << "🔍 Detecting market regime..." << std::endl;
    
    // Simplified regime detection based on volatility and trend
    double volatility = 0.0;
    double trend = 0.0;
    
    if (market_data.count("volatility") && market_data.count("trend")) {
        volatility = market_data.at("volatility");
        trend = market_data.at("trend");
    }
    
    MarketRegime regime;
    
    if (volatility > 0.03) {  // High volatility threshold
        regime = MarketRegime::HIGH_VOLATILITY;
    } else if (volatility < 0.01) {  // Low volatility threshold
        regime = MarketRegime::LOW_VOLATILITY;
    } else if (std::abs(trend) > 0.02) {  // Strong trend
        regime = (trend > 0) ? MarketRegime::BULL_MARKET : MarketRegime::BEAR_MARKET;
    } else {
        regime = MarketRegime::SIDEWAYS_MARKET;
    }
    
    std::string regime_name;
    switch (regime) {
        case MarketRegime::BULL_MARKET: regime_name = "BULL_MARKET"; break;
        case MarketRegime::BEAR_MARKET: regime_name = "BEAR_MARKET"; break;
        case MarketRegime::SIDEWAYS_MARKET: regime_name = "SIDEWAYS_MARKET"; break;
        case MarketRegime::HIGH_VOLATILITY: regime_name = "HIGH_VOLATILITY"; break;
        case MarketRegime::LOW_VOLATILITY: regime_name = "LOW_VOLATILITY"; break;
    }
    
    std::cout << "📈 Market regime detected: " << regime_name << std::endl;
    return regime;
}

std::pair<std::string, std::string> OpenCLawIntegration::route_smart_order(
    const AdvancedOrder& order) {
    
    if (!config_.enable_smart_order_routing) {
        return {"default", "simple"};
    }
    
    // Select best venues for this symbol
    auto best_venues = select_best_venues(order.symbol, order.order_type);
    
    if (best_venues.empty()) {
        return {"default", "simple"};
    }
    
    // Choose venue with best metrics
    std::string best_venue = best_venues[0];
    double best_score = 0.0;
    std::string best_strategy = "simple";
    
    for (const auto& venue : best_venues) {
        if (venue_metrics_.count(venue)) {
            const auto& metrics = venue_metrics_.at(venue);
            
            // Calculate venue score (weighted combination of metrics)
            double score = (metrics.avg_fill_rate * 0.4) + 
                          (10.0 - metrics.avg_slippage_bps) * 0.3) +  // Lower slippage is better
                          (100.0 - metrics.latency_ms) * 0.2) +  // Lower latency is better
                          (metrics.liquidity_score * 0.1);
            
            if (score > best_score) {
                best_score = score;
                best_venue = venue;
                
                // Choose strategy based on order type and venue characteristics
                if (order.order_type == OrderType::MARKET) {
                    best_strategy = (metrics.latency_ms < 50) ? "immediate" : "optimized";
                } else if (order.order_type == OrderType::LIMIT) {
                    best_strategy = (metrics.liquidity_score > 0.8) ? "passive" : "aggressive";
                } else if (order.order_type == OrderType::TWAP || order.order_type == OrderType::VWAP) {
                    best_strategy = "algorithmic";
                }
            }
        }
    }
    
    return {best_venue, best_strategy};
}

std::string OpenCLawIntegration::analyze_market_microstructure(
    const std::map<std::string, std::vector<double>>& order_book) {
    
    if (!config_.enable_market_microstructure) {
        return "simple";
    }
    
    std::cout << "🔬 Analyzing market microstructure..." << std::endl;
    
    // Calculate bid-ask spread
    double best_bid = 0.0, best_ask = 0.0;
    if (order_book.count("bids") && order_book.count("asks")) {
        const auto& bids = order_book.at("bids");
        const auto& asks = order_book.at("asks");
        
        if (!bids.empty()) best_bid = bids[0];
        if (!asks.empty()) best_ask = asks[0];
    }
    
    double spread = best_ask - best_bid;
    double spread_pct = (best_bid > 0) ? (spread / best_bid) * 100 : 0.0;
    
    // Determine execution strategy based on spread
    std::string strategy;
    if (spread_pct < 0.01) {  // Tight spread
        strategy = "passive_liquidity_provision";
    } else if (spread_pct < 0.05) {  // Normal spread
        strategy = "balanced_execution";
    } else {  // Wide spread
        strategy = "aggressive_execution";
    }
    
    std::cout << "📊 Bid-Ask spread: " << spread_pct << "%" << std::endl;
    std::cout << "📊 Execution strategy: " << strategy << std::endl;
    
    return strategy;
}

std::map<std::string, std::string> OpenCLawIntegration::get_status() const {
    std::map<std::string, std::string> status;
    
    status["integration_status"] = initialized_ ? "active" : "inactive";
    status["smart_routing"] = config_.enable_smart_order_routing ? "enabled" : "disabled";
    status["market_microstructure"] = config_.enable_market_microstructure ? "enabled" : "disabled";
    status["ml_filtering"] = config_.enable_ml_signal_filtering ? "enabled" : "disabled";
    status["regime_detection"] = config_.enable_regime_detection ? "enabled" : "disabled";
    status["supported_venues_count"] = std::to_string(config_.supported_venues.size());
    status["max_slippage_bps"] = std::to_string(config_.max_slippage_bps);
    
    return status;
}

// Private methods implementation

void OpenCLawIntegration::update_venue_metrics(const std::string& venue, 
                                               double fill_rate, 
                                               double slippage_bps,
                                               double latency_ms) {
    if (!venue_metrics_.count(venue)) {
        return;
    }
    
    auto& metrics = venue_metrics_[venue];
    
    // Exponential moving average for metrics
    const double alpha = 0.1;  // Smoothing factor
    
    metrics.avg_fill_rate = alpha * fill_rate + (1 - alpha) * metrics.avg_fill_rate;
    metrics.avg_slippage_bps = alpha * slippage_bps + (1 - alpha) * metrics.avg_slippage_bps;
    metrics.latency_ms = alpha * latency_ms + (1 - alpha) * metrics.latency_ms;
    
    std::cout << "📊 Updated " << venue << " metrics" << std::endl;
}

double OpenCLawIntegration::calculate_optimal_size(const std::string& symbol,
                                                double target_size,
                                                MarketRegime regime) {
    
    double adjusted_size = target_size;
    
    // Adjust size based on market regime
    switch (regime) {
        case MarketRegime::HIGH_VOLATILITY:
            adjusted_size *= 0.7;  // Reduce size in high volatility
            break;
        case MarketRegime::LOW_VOLATILITY:
            adjusted_size *= 1.2;  // Increase size in low volatility
            break;
        case MarketRegime::BULL_MARKET:
            adjusted_size *= 1.1;  // Slightly increase in bull market
            break;
        case MarketRegime::BEAR_MARKET:
            adjusted_size *= 0.8;  // Reduce size in bear market
            break;
        default:
            break;  // No adjustment for sideways market
    }
    
    return adjusted_size;
}

std::vector<std::string> OpenCLawIntegration::select_best_venues(const std::string& symbol,
                                                              OrderType order_type) {
    
    std::vector<std::pair<std::string, double>> venue_scores;
    
    for (const auto& [venue, metrics] : venue_metrics_) {
        // Calculate score based on venue metrics
        double score = (metrics.avg_fill_rate * 0.5) + 
                      ((10.0 - metrics.avg_slippage_bps) / 10.0) * 0.3 +  // Normalize slippage
                      ((100.0 - metrics.latency_ms) / 100.0) * 0.2;  // Normalize latency
        
        venue_scores.push_back({venue, score});
    }
    
    // Sort venues by score
    std::sort(venue_scores.begin(), venue_scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Return top 3 venues
    std::vector<std::string> best_venues;
    for (size_t i = 0; i < std::min(size_t(3), venue_scores.size()); ++i) {
        best_venues.push_back(venue_scores[i].first);
    }
    
    return best_venues;
}

// OpenCLawSignalProcessor implementation

std::vector<OpenCLawSignal> OpenCLawSignalProcessor::process_raw_signals(
    const std::vector<std::map<std::string, double>>& raw_signals) {
    
    std::vector<OpenCLawSignal> processed_signals;
    
    for (const auto& raw_signal : raw_signals) {
        OpenCLawSignal signal;
        
        // Convert raw signal to OpenCLaw format
        if (raw_signal.count("signal_strength")) {
            double strength = raw_signal.at("signal_strength");
            
            if (strength > 0.8) {
                signal.signal_type = SignalType::STRONG_BUY;
                signal.confidence = strength;
            } else if (strength > 0.6) {
                signal.signal_type = SignalType::BUY;
                signal.confidence = strength;
            } else if (strength < -0.8) {
                signal.signal_type = SignalType::STRONG_SELL;
                signal.confidence = std::abs(strength);
            } else if (strength < -0.6) {
                signal.signal_type = SignalType::SELL;
                signal.confidence = std::abs(strength);
            } else {
                signal.signal_type = SignalType::HOLD;
                signal.confidence = 0.5;
            }
        }
        
        signal.timestamp = std::chrono::system_clock::now();
        signal.technical_indicators = raw_signal;
        
        processed_signals.push_back(signal);
    }
    
    return processed_signals;
}

std::vector<OpenCLawSignal> OpenCLawSignalProcessor::apply_ml_filtering(
    const std::vector<OpenCLawSignal>& signals) {
    
    std::vector<OpenCLawSignal> filtered_signals;
    
    for (const auto& signal : signals) {
        OpenCLawSignal filtered = signal;
        
        // Apply confidence filtering
        if (signal.confidence < 0.3) {
            continue;  // Filter out low confidence signals
        }
        
        // Apply technical validation
        if (signal.technical_indicators.count("RSI")) {
            double rsi = signal.technical_indicators.at("RSI");
            
            // Filter signals in extreme overbought/oversold conditions
            if ((signal.signal_type == SignalType::BUY || signal.signal_type == SignalType::STRONG_BUY) && rsi > 75) {
                filtered.confidence *= 0.3;  // Heavily reduce confidence
            } else if ((signal.signal_type == SignalType::SELL || signal.signal_type == SignalType::STRONG_SELL) && rsi < 25) {
                filtered.confidence *= 0.3;  // Heavily reduce confidence
            }
        }
        
        filtered_signals.push_back(filtered);
    }
    
    return filtered_signals;
}

OpenCLawSignal OpenCLawSignalProcessor::ensemble_signals(
    const std::vector<std::vector<OpenCLawSignal>>& signals) {
    
    if (signals.empty()) {
        return OpenCLawSignal{};
    }
    
    OpenCLawSignal ensemble_signal;
    ensemble_signal.signal_type = SignalType::HOLD;
    ensemble_signal.confidence = 0.0;
    ensemble_signal.timestamp = std::chrono::system_clock::now();
    
    // Weighted voting based on confidence
    std::map<SignalType, double> vote_weights;
    std::map<SignalType, int> vote_counts;
    
    for (const auto& signal_group : signals) {
        for (const auto& signal : signal_group) {
            vote_weights[signal.signal_type] += signal.confidence;
            vote_counts[signal.signal_type]++;
        }
    }
    
    // Find the signal with highest weighted confidence
    auto max_weight_it = std::max_element(vote_weights.begin(), vote_weights.end(),
                                         [](const auto& a, const auto& b) {
                                             return a.second < b.second;
                                         });
    
    if (max_weight_it != vote_weights.end()) {
        ensemble_signal.signal_type = max_weight_it->first;
        ensemble_signal.confidence = max_weight_it->second / signals.size();
        
        // Add consensus information
        int total_votes = std::accumulate(vote_counts.begin(), vote_counts.end(), 0,
                                      [](int sum, const auto& pair) { return sum + pair.second; });
        
        ensemble_signal.reasoning = "Ensemble of " + std::to_string(signals.size()) + 
                                 " signal sources. Votes: " + std::to_string(vote_counts[ensemble_signal.signal_type]) + 
                                 "/" + std::to_string(total_votes);
    }
    
    return ensemble_signal;
}

} // namespace Trading
} // namespace ArchNeuronX
