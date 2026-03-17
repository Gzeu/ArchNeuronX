/**
 * @file data_aggregator.cpp
 * @brief Data aggregator implementation for combining multiple data sources
 * @author George Pricop
 * @date 2025-10-02
 */

#include "data/data_aggregator.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <thread>
#include <chrono>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace ArchNeuronX {
namespace Data {

DataAggregator::DataAggregator(const AggregationConfig& config) 
    : config_(config), real_time_active_(false) {
    
    // Initialize source metrics
    for (const auto& weight : config_.source_weights) {
        SourceMetrics metrics;
        metrics.accuracy_score = 1.0;
        metrics.latency_score = 1.0;
        metrics.completeness_score = 1.0;
        metrics.reliability_score = weight.reliability_score;
        metrics.last_update = std::chrono::steady_clock::now();
        metrics.successful_requests = 0;
        metrics.failed_requests = 0;
        
        source_metrics_[weight.provider_name] = metrics;
    }
}

DataAggregator::~DataAggregator() {
    stop_real_time_aggregation();
    
    // Join aggregation threads
    for (auto& thread : aggregation_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

std::future<AggregationResult> DataAggregator::aggregate_historical_data(
    const std::string& symbol,
    const std::string& timeframe,
    const std::chrono::system_clock::time_point& start,
    const std::chrono::system_clock::time_point& end,
    const std::vector<std::shared_ptr<DataProvider>>& providers
) {
    return std::async(std::launch::async, [this, symbol, timeframe, start, end, providers]() {
        AggregationResult result;
        result.symbol = symbol;
        result.timeframe = timeframe;
        result.start_time = start;
        result.end_time = end;
        result.aggregation_timestamp = std::chrono::system_clock::now();
        result.total_sources = providers.size();
        
        std::vector<std::future<std::vector<OHLCV>>> futures;
        std::map<std::string, std::shared_ptr<DataProvider>> provider_map;
        
        // Collect data from all providers
        for (const auto& provider : providers) {
            if (provider && provider->is_connected()) {
                auto future = provider->get_historical_data(symbol, timeframe, start, end);
                futures.push_back(std::move(future));
                provider_map[provider->get_provider_name()] = provider;
                LOG_INFO("Requesting historical data from: {}", provider->get_provider_name());
            }
        }
        
        // Wait for all requests to complete
        std::vector<std::vector<OHLCV>> source_data;
        std::vector<std::string> provider_names;
        
        for (size_t i = 0; i < futures.size(); ++i) {
            try {
                auto data = futures[i].get();
                source_data.push_back(data);
                
                // Find corresponding provider name
                for (const auto& [name, provider] : provider_map) {
                    if (provider && futures[i].valid()) {
                        provider_names.push_back(name);
                        break;
                    }
                }
                
                // Update source metrics
                update_source_metrics(provider_names.back(), true, 100.0, 1.0);
                
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to get data from provider: {}", e.what());
                result.failed_sources.push_back("provider_" + std::to_string(i));
                update_source_metrics("provider_" + std::to_string(i), false, 0.0, 0.0);
            }
        }
        
        result.successful_sources = source_data.size();
        
        if (source_data.empty()) {
            LOG_ERROR("No data received from any provider for {}", symbol);
            result.aggregation_method_used = "none";
            return result;
        }
        
        // Filter out providers that didn't return data
        std::vector<std::vector<OHLCV>> valid_data;
        std::vector<std::string> valid_provider_names;
        std::vector<double> valid_weights;
        
        for (size_t i = 0; i < source_data.size(); ++i) {
            if (!source_data[i].empty()) {
                valid_data.push_back(source_data[i]);
                valid_provider_names.push_back(provider_names[i]);
                valid_weights.push_back(calculate_source_weight(provider_names[i]));
            }
        }
        
        if (valid_data.size() < config_.min_sources) {
            LOG_WARN("Insufficient valid sources for aggregation: {}/{}", 
                     valid_data.size(), config_.min_sources);
            result.aggregation_method_used = "insufficient_sources";
            return result;
        }
        
        // Store source data in result
        for (size_t i = 0; i < valid_data.size(); ++i) {
            result.source_data[valid_provider_names[i]] = valid_data[i];
        }
        
        // Align time series (all sources may have different timestamps)
        result.aggregated_data = align_time_series(valid_data);
        
        // Remove outliers if enabled
        if (config_.enable_outlier_detection) {
            auto before_size = result.aggregated_data.size();
            result.aggregated_data = detect_and_remove_outliers(result.aggregated_data);
            auto after_size = result.aggregated_data.size();
            
            if (before_size != after_size) {
                result.outlier_removed.push_back(std::to_string(before_size - after_size) + " outliers removed");
                LOG_INFO("Removed {} outliers from aggregation", before_size - after_size);
            }
        }
        
        // Apply aggregation method
        std::vector<double> weights;
        
        switch (config_.method) {
            case AggregationMethod::FIRST_AVAILABLE:
                result.aggregated_data = aggregate_by_first_available(valid_data);
                result.aggregation_method_used = "first_available";
                break;
                
            case AggregationMethod::WEIGHTED_AVERAGE:
                weights = valid_weights;
                result.aggregated_data = aggregate_by_weighted_average(valid_data, weights);
                result.aggregation_method_used = "weighted_average";
                break;
                
            case AggregationMethod::MEDIAN:
                result.aggregated_data = aggregate_by_median(valid_data);
                result.aggregation_method_used = "median";
                break;
                
            case AggregationMethod::CONSENSUS:
                result.aggregated_data = aggregate_by_consensus(valid_data);
                result.aggregation_method_used = "consensus";
                break;
                
            case AggregationMethod::LATEST_TIMESTAMP:
                result.aggregated_data = aggregate_by_latest_timestamp(valid_data);
                result.aggregation_method_used = "latest_timestamp";
                break;
                
            case AggregationMethod::BEST_QUALITY:
                result.aggregated_data = aggregate_by_best_quality(valid_data);
                result.aggregation_method_used = "best_quality";
                break;
                
            case AggregationMethod::CUSTOM:
                // Custom aggregation would be provided by user
                result.aggregated_data = valid_data[0]; // Fallback
                result.aggregation_method_used = "custom_fallback";
                break;
                
            default:
                result.aggregated_data = aggregate_by_weighted_average(valid_data, valid_weights);
                result.aggregation_method_used = "weighted_average_fallback";
                break;
        }
        
        // Store weights used
        for (size_t i = 0; i < valid_provider_names.size(); ++i) {
            result.source_weights_used[valid_provider_names[i]] = weights[i];
        }
        
        // Calculate quality scores
        if (config_.enable_quality_scoring) {
            for (const auto& name : valid_provider_names) {
                const auto& metrics = source_metrics_[name];
                double quality_score = (metrics.accuracy_score * 0.4 + 
                                   metrics.reliability_score * 0.3 + 
                                   metrics.latency_score * 0.2 + 
                                   metrics.completeness_score * 0.1);
                
                result.quality_scores[name] = quality_score;
            }
        }
        
        LOG_INFO("Aggregated {} data points from {} sources using {} method", 
                 result.aggregated_data.size(), result.successful_sources, result.aggregation_method_used);
        
        return result;
    });
}

std::future<AggregationResult> DataAggregator::aggregate_current_prices(
    const std::string& symbol,
    const std::vector<std::shared_ptr<DataProvider>>& providers
) {
    return std::async(std::launch::async, [this, symbol, providers]() {
        AggregationResult result;
        result.symbol = symbol;
        result.aggregation_timestamp = std::chrono::system_clock::now();
        result.total_sources = providers.size();
        
        std::vector<std::future<double>> futures;
        std::map<std::string, std::shared_ptr<DataProvider>> provider_map;
        
        // Collect current prices from all providers
        for (const auto& provider : providers) {
            if (provider && provider->is_connected()) {
                auto future = provider->get_current_price(symbol);
                futures.push_back(std::move(future));
                provider_map[provider->get_provider_name()] = provider;
            }
        }
        
        // Wait for all requests to complete
        std::map<std::string, double> current_prices;
        std::vector<std::string> provider_names;
        std::vector<double> price_values;
        
        for (size_t i = 0; i < futures.size(); ++i) {
            try {
                double price = futures[i].get();
                if (price > 0.0) {
                    price_values.push_back(price);
                    
                    // Find corresponding provider name
                    for (const auto& [name, provider] : provider_map) {
                        if (provider && futures[i].valid()) {
                            provider_names.push_back(name);
                            current_prices[name] = price;
                            break;
                        }
                    }
                }
                
                // Update source metrics
                update_source_metrics(provider_names.back(), true, 50.0, 1.0);
                
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to get current price from provider: {}", e.what());
                result.failed_sources.push_back("provider_" + std::to_string(i));
                update_source_metrics("provider_" + std::to_string(i), false, 0.0, 0.0);
            }
        }
        
        result.successful_sources = current_prices.size();
        
        if (price_values.empty()) {
            LOG_ERROR("No prices received from any provider for {}", symbol);
            result.aggregation_method_used = "none";
            return result;
        }
        
        // Calculate aggregated price based on method
        double aggregated_price = 0.0;
        
        switch (config_.method) {
            case AggregationMethod::FIRST_AVAILABLE:
                if (!price_values.empty()) {
                    aggregated_price = price_values[0];
                }
                result.aggregation_method_used = "first_available";
                break;
                
            case AggregationMethod::WEIGHTED_AVERAGE:
                if (!price_values.empty()) {
                    std::vector<double> weights;
                    for (const auto& name : provider_names) {
                        weights.push_back(calculate_source_weight(name));
                    }
                    aggregated_price = calculate_weighted_average(price_values, weights);
                }
                result.aggregation_method_used = "weighted_average";
                break;
                
            case AggregationMethod::MEDIAN:
                if (!price_values.empty()) {
                    std::sort(price_values.begin(), price_values.end());
                    size_t n = price_values.size();
                    aggregated_price = (n % 2 == 0) ? 
                        price_values[n/2] : 
                        (price_values[n/2 - 1] + price_values[n/2]) / 2.0;
                }
                result.aggregation_method_used = "median";
                break;
                
            case AggregationMethod::CONSENSUS:
                // Use price that appears most frequently
                if (!price_values.empty()) {
                    std::map<double, int> frequency;
                    for (double price : price_values) {
                        frequency[price]++;
                    }
                    
                    auto max_it = std::max_element(frequency.begin(), frequency.end(),
                        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                            return a.second < b.second;
                        });
                    
                    if (max_it != frequency.end()) {
                        aggregated_price = max_it->first;
                    }
                }
                result.aggregation_method_used = "consensus";
                break;
                
            case AggregationMethod::BEST_QUALITY:
                if (!price_values.empty()) {
                    double best_score = 0.0;
                    for (const auto& name : provider_names) {
                        const auto& metrics = source_metrics_[name];
                        double quality_score = (metrics.accuracy_score * 0.4 + 
                                           metrics.reliability_score * 0.3 + 
                                           metrics.latency_score * 0.2 + 
                                           metrics.completeness_score * 0.1);
                        
                        if (quality_score > best_score) {
                            best_score = quality_score;
                            aggregated_price = current_prices[name];
                        }
                    }
                }
                result.aggregation_method_used = "best_quality";
                break;
                
            default:
                if (!price_values.empty()) {
                    aggregated_price = price_values[0];
                }
                result.aggregation_method_used = "weighted_average_fallback";
                break;
        }
        
        // Create OHLCV data point for the aggregated price
        if (aggregated_price > 0.0) {
            OHLCV ohlcv;
            ohlcv.symbol = symbol;
            ohlcv.timestamp = std::chrono::system_clock::now();
            ohlcv.open = aggregated_price;
            ohlcv.high = aggregated_price;
            ohlcv.low = aggregated_price;
            ohlcv.close = aggregated_price;
            ohlcv.volume = 0.0;
            
            result.aggregated_data.push_back(ohlcv);
        }
        
        // Store source data
        for (const auto& [name, price] : current_prices) {
            if (price > 0.0) {
                OHLCV ohlcv;
                ohlcv.symbol = symbol;
                ohlcv.timestamp = std::chrono::system_clock::now();
                ohlcv.open = price;
                ohlcv.high = price;
                ohlcv.low = price;
                ohlcv.close = price;
                ohlcv.volume = 0.0;
                
                result.source_data[name] = {ohlcv};
            }
        }
        
        // Store weights used
        for (const auto& name : provider_names) {
            result.source_weights_used[name] = calculate_source_weight(name);
        }
        
        // Calculate quality scores
        if (config_.enable_quality_scoring) {
            for (const auto& name : provider_names) {
                const auto& metrics = source_metrics_[name];
                double quality_score = (metrics.accuracy_score * 0.4 + 
                                   metrics.reliability_score * 0.3 + 
                                   metrics.latency_score * 0.2 + 
                                   metrics.completeness_score * 0.1);
                
                result.quality_scores[name] = quality_score;
            }
        }
        
        LOG_INFO("Aggregated price {} from {} sources using {} method", 
                 aggregated_price, result.successful_sources, result.aggregation_method_used);
        
        return result;
    });
}

std::future<OrderBook> DataAggregator::aggregate_order_books(
    const std::string& symbol,
    int depth,
    const std::vector<std::shared_ptr<DataProvider>>& providers
) {
    return std::async(std::launch::async, [this, symbol, depth, providers]() {
        std::vector<std::future<OrderBook>> futures;
        std::vector<OrderBook> order_books;
        
        // Collect order books from all providers
        for (const auto& provider : providers) {
            if (provider && provider->is_connected()) {
                auto future = provider->get_order_book(symbol, depth);
                futures.push_back(std::move(future));
            }
        }
        
        // Wait for all requests to complete
        for (auto& future : futures) {
            try {
                auto order_book = future.get();
                if (!order_book.bids.empty() || !order_book.asks.empty()) {
                    order_books.push_back(order_book);
                }
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to get order book: {}", e.what());
            }
        }
        
        if (order_books.empty()) {
            LOG_WARN("No order books received from any provider for {}", symbol);
            return OrderBook{};
        }
        
        // Aggregate order books
        OrderBook aggregated_book;
        aggregated_book.symbol = symbol;
        aggregated_book.timestamp = std::chrono::system_clock::now();
        
        if (config_.method == AggregationMethod::WEIGHTED_AVERAGE) {
            std::vector<double> weights;
            for (size_t i = 0; i < order_books.size(); ++i) {
                weights.push_back(1.0 / order_books.size()); // Equal weights
            }
            
            aggregated_book = aggregate_order_books_weighted(order_books, weights);
        } else {
            // Simple aggregation: use first available
            aggregated_book = order_books[0];
        }
        
        LOG_INFO("Aggregated order book for {} from {} sources", 
                 symbol, order_books.size());
        
        return aggregated_book;
    });
}

void DataAggregator::start_real_time_aggregation(
    const std::string& symbol,
    std::function<void(const TickData&)> callback,
    const std::vector<std::shared_ptr<DataProvider>>& providers
) {
    if (real_time_active_) {
        LOG_WARN("Real-time aggregation already active for {}", symbol);
        return;
    }
    
    real_time_active_ = true;
    
    // Start aggregation worker thread
    aggregation_threads_.push_back(std::thread([this, symbol, callback, providers]() {
        real_time_aggregation_worker(symbol, callback, providers);
    }));
    
    LOG_INFO("Started real-time aggregation for {} with {} providers", 
             symbol, providers.size());
}

void DataAggregator::stop_real_time_aggregation() {
    if (!real_time_active_) {
        return;
    }
    
    real_time_active_ = false;
    
    // Wait for threads to finish
    for (auto& thread : aggregation_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    aggregation_threads_.clear();
    
    LOG_INFO("Stopped real-time aggregation");
}

void DataAggregator::update_config(const AggregationConfig& config) {
    std::lock_guard<std::mutex> lock(aggregation_mutex_);
    config_ = config;
    
    LOG_INFO("Updated aggregation configuration: method={}", 
             static_cast<int>(config_.method));
}

std::map<std::string, double> DataAggregator::get_aggregation_stats() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    std::map<std::string, double> stats;
    
    for (const auto& [name, metrics] : source_metrics_) {
        double success_rate = metrics.successful_requests + metrics.failed_requests > 0 ? 
                              (double)metrics.successful_requests / 
                               (metrics.successful_requests + metrics.failed_requests) : 0.0;
        
        stats[name] = success_rate;
    }
    
    return stats;
}

// Private methods implementation

std::vector<OHLCV> DataAggregator::align_time_series(
    const std::vector<std::vector<OHLCV>>& source_data
) {
    if (source_data.empty()) {
        return {};
    }
    
    // Find the time range
    std::chrono::system_clock::time_point min_time = source_data[0][0].timestamp;
    std::chrono::system_clock::time_point max_time = source_data[0][0].timestamp;
    
    for (const auto& data : source_data) {
        for (const auto& point : data) {
            if (point.timestamp < min_time) {
                min_time = point.timestamp;
            }
            if (point.timestamp > max_time) {
                max_time = point.timestamp;
            }
        }
    }
    
    // Create aligned time series
    std::vector<OHLCV> aligned_data;
    
    for (const auto& source : source_data) {
        std::vector<OHLCV> aligned_source;
        
        for (const auto& point : source) {
            OHLCV aligned_point = point;
            
            // Find corresponding points in other sources
            std::vector<std::reference_wrapper<const OHLCV>> other_points;
            
            for (const auto& other_source : source_data) {
                if (&other_source != &source) {
                    for (const auto& other_point : other_source) {
                        if (std::abs(std::chrono::duration_cast<std::chrono::seconds>(
                            point.timestamp - other_point.timestamp).count()) <= 60) {
                            other_points.push_back(other_point);
                        }
                    }
                }
            }
            
            // Interpolate missing points
            std::vector<OHLCV> interpolated_points;
            
            for (const auto& other_ref : other_points) {
                auto other_point = other_ref.get();
                
                // Simple linear interpolation for missing points
                std::vector<OHLCV> segment = {point, other_point};
                
                for (size_t i = 1; i < 10; ++i) { // 10 interpolation points
                    double ratio = (double)i / 10.0;
                    OHLCV interpolated;
                    interpolated.timestamp = point.timestamp + 
                        std::chrono::duration_cast<std::chrono::seconds>(
                            (other_point.timestamp - point.timestamp) * ratio));
                    
                    // Interpolate OHLC values
                    interpolated.open = point.open + (other_point.open - point.open) * ratio;
                    interpolated.high = point.high + (other_point.high - point.high) * ratio;
                    interpolated.low = point.low + (other_point.low - point.low) * ratio;
                    interpolated.close = point.close + (other_point.close - point.close) * ratio;
                    interpolated.volume = point.volume + (other_point.volume - point.volume) * ratio;
                    
                    interpolated_points.push_back(interpolated);
                }
            }
            
            // Combine original and interpolated points
            aligned_source.insert(aligned_source.end(), 
                                 interpolated_points.begin(), interpolated_points.end());
        }
        
        aligned_data.insert(aligned_data.end(), aligned_source.begin(), aligned_source.end());
    }
    
    return aligned_data;
}

std::vector<OHLCV> DataAggregator::detect_and_remove_outliers(
    const std::vector<OHLCV>& data
) {
    if (data.size() < 10) { // Need sufficient data for outlier detection
        return data;
    }
    
    std::vector<OHLCV> filtered_data;
    
    // Detect outliers in closing prices using Z-score method
    std::vector<double> close_prices;
    for (const auto& point : data) {
        close_prices.push_back(point.close);
    }
    
    std::vector<double> z_scores = calculate_z_scores(close_prices);
    
    for (size_t i = 0; i < data.size(); ++i) {
        if (std::abs(z_scores[i]) > config_.outlier_threshold) {
            LOG_WARN("Outlier detected: {} at {} (z-score: {:.2f})", 
                     data[i].close, data[i].timestamp, z_scores[i]);
        } else {
            filtered_data.push_back(data[i]);
        }
    }
    
    return filtered_data;
}

std::vector<OHLCV> DataAggregator::aggregate_by_first_available(
    const std::vector<std::vector<OHLCV>>& source_data
) {
    if (!source_data.empty()) {
        return {};
    }
    
    // Return the first non-empty source
    for (const auto& data : source_data) {
        if (!data.empty()) {
            return data;
        }
    }
    
    return source_data[0];
}

std::vector<OHLCV> DataAggregator::aggregate_by_weighted_average(
    const std::vector<std::vector<OHLCV>>& source_data,
    const std::vector<double>& weights
) {
    if (source_data.empty() || weights.empty()) {
        return {};
    }
    
    std::vector<OHLCV> aggregated;
    size_t min_size = std::min_element(source_data.begin(), source_data.end(), 
                                      [](const std::vector<OHLCV>& a, const std::vector<OHLCV>& b) {
                                          return a.size() < b.size();
                                      });
    
    for (size_t i = 0; i < min_size; ++i) {
        double open = 0.0, high = 0.0, low = 0.0, close = 0.0, volume = 0.0;
        
        for (size_t j = 0; j < source_data.size(); ++j) {
            if (i < source_data[j].size()) {
                open += source_data[j][i].open * weights[j];
                high += source_data[j][i].high * weights[j];
                low += source_data[j][i].low * weights[j];
                close += source_data[j][i].close * weights[j];
                volume += source_data[j][i].volume * weights[j];
            }
        }
        
        OHLCV point;
        point.timestamp = source_data[0][i].timestamp;
        point.open = open / weights[i];
        point.high = high / weights[i];
        point.low = low / weights[i];
        point.close = close / weights[i];
        point.volume = volume / weights[i];
        
        aggregated.push_back(point);
    }
    
    return aggregated;
}

std::vector<OHLCV> DataAggregator::aggregate_by_median(
    const std::vector<std::vector<OHLCV>>& source_data
) {
    if (source_data.empty()) {
        return {};
    }
    
    std::vector<OHLCV> aggregated;
    size_t min_size = std::min_element(source_data.begin(), source_data.end(), 
                                      [](const std::vector<OHLCV>& a, const std::vector<OHLCV>& b) {
                                          return a.size() < b.size();
                                      });
    
    for (size_t i = 0; i < min_size; ++i) {
        std::vector<OHLCV> points_at_timestamp;
        
        // Collect all points at this timestamp from all sources
        for (const auto& source : source_data) {
            if (i < source.size()) {
                points_at_timestamp.push_back(source[i]);
            }
        }
        
        // Calculate median for each field
        std::vector<double> opens, highs, lows, closes, volumes;
        
        for (const auto& point : points_at_timestamp) {
            opens.push_back(point.open);
            highs.push_back(point.high);
            lows.push_back(point.low);
            closes.push_back(point.close);
            volumes.push_back(point.volume);
        }
        
        std::sort(opens.begin(), opens.end());
        std::sort(highs.begin(), highs.end());
        std::sort(lows.begin(), lows.end());
        std::sort(closes.begin(), closes.end());
        std::sort(volumes.begin(), volumes.end());
        
        OHLCV median_point;
        median_point.timestamp = points_at_timestamp[0].timestamp;
        median_point.open = opens[opens.size() / 2];
        median_point.high = highs[highs.size() / 2];
        median_point.low = lows[lows.size() / 2];
        median_point.close = closes[closes.size() / 2];
        median_point.volume = volumes[volumes.size() / 2];
        
        aggregated.push_back(median_point);
    }
    
    return aggregated;
}

std::vector<OHLCV> DataAggregator::aggregate_by_consensus(
    const std::values<std::vector<OHLCV>>& source_data
) {
    if (source_data.empty()) {
        return {};
    }
    
    std::vector<OHLCV> aggregated;
    size_t min_size = std::min_element(source_data.begin(), source_data.end(), 
                                      [](const std::vector<OHLCV>& a, const std::vector<OHLCV>& b) {
                                          return a.size() < b.size();
                                      });
    
    for (size_t i = 0; i < min_size; ++i) {
        std::map<std::string, int> close_values;
        
        // Count occurrences of each close price
        for (const auto& source : source_data) {
            if (i < source.size()) {
                int close_int = static_cast<int>(source[i].close * 100); // Convert to cents for integer comparison
                close_values[std::to_string(close_int)]++;
            }
        }
        
        // Find the most common close price (consensus)
        int max_count = 0;
        std::string consensus_close;
        
        for (const auto& [close_str, count] : close_values) {
            if (count > max_count) {
                max_count = count;
                consensus_close = close_str;
            }
        }
        
        double consensus_price = std::stod(consensus_close) / 100.0;
        
        // Create consensus point using median of other fields
        std::vector<OHLCV> points_at_timestamp;
        
        for (const auto& source : source_data) {
            if (i < source.size()) {
                points_at_timestamp.push_back(source[i]);
            }
        }
        
        std::vector<double> opens, highs, lows, volumes;
        
        for (const auto& point : points_at_timestamp) {
            opens.push_back(point.open);
            highs.push_back(point.high);
            lows.push_back(point.low);
            volumes.push_back(point.volume);
        }
        
        OHLCV consensus_point;
        consensus_point.timestamp = points_at_timestamp[0].timestamp;
        consensus_point.close = consensus_price;
        consensus_point.open = calculate_median(opens);
        consensus_point.high = calculate_median(highs);
        consensus_point.low = calculate_median(lows);
        consensus_point.volume = calculate_median(volumes);
        
        aggregated.push_back(consensus_point);
    }
    
    return aggregated;
}

std::vector<OHLCV> DataAggregator::aggregate_by_latest_timestamp(
    const std::vector<std::vector<OHLCV>>& source_data
) {
    if (source_data.empty()) {
        return {};
    }
    
    // Find the source with latest data
    size_t latest_source_idx = 0;
    std::chrono::system_clock::time_point latest_timestamp = source_data[0][0].timestamp;
    
    for (size_t i = 1; i < source_data.size(); ++i) {
        if (!source_data[i].empty() && 
            source_data[i][0].timestamp > latest_timestamp) {
            latest_timestamp = source_data[i][0].timestamp;
            latest_source_idx = i;
        }
    }
    
    return source_data[latest_source_idx];
}

std::vector<OHLCV> DataAggregator::aggregate_by_best_quality(
    const std::vector<std::vector<OHLCV>>& source_data
) {
    if (source_data.empty()) {
        return {};
    }
    
    std::vector<std::string> provider_names;
    std::vector<double> quality_scores;
    
    // Get provider names from source_data (assuming they're in order)
    if (!source_data.empty()) {
        provider_names.push_back("provider_0");
        provider_names.push_back("provider_1");
        provider_names.push_back("provider_2");
    }
    
    // Calculate quality scores for each provider
    for (const auto& name : provider_names) {
        const auto& metrics = source_metrics_[name];
        double quality_score = (metrics.accuracy_score * 0.4 + 
                           metrics.reliability_score * 0.3 + 
                           metrics.latency_score * 0.2 + 
                           metrics.completeness_score * 0.1);
        
        quality_scores.push_back(quality_score);
    }
    
    // Find provider with best quality score
    auto max_it = std::max_element(quality_scores.begin(), quality_scores.end());
    size_t best_idx = std::distance(quality_scores.begin(), max_it);
    
    if (best_idx < source_data.size()) {
        return source_data[best_idx];
    }
    
    return {};
}

double DataAggregator::calculate_source_weight(const std::string& provider_name) {
    auto it = std::find_if(config_.source_weights.begin(), config_.source_weights.end(),
        [&provider_name](const DataSourceWeight& weight) {
            return weight.provider_name == provider_name;
        });
    
    if (it != config_.source_weights.end()) {
        return it->weight;
    }
    
    // Default weight
    return 1.0;
}

void DataAggregator::update_source_metrics(
    const std::string& provider_name,
    bool success,
    double latency_ms,
    double completeness_score
) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    auto& metrics = source_metrics_[provider_name];
    
    if (success) {
        metrics.successful_requests++;
        
        // Update latency score (lower is better)
        if (metrics.latency_score == 0.0) {
            metrics.latency_score = 1.0;
        }
        metrics.latency_score = (metrics.latency_score * 0.9) + (latency_ms / 1000.0 * 0.1);
        
        // Update completeness score
        if (metrics.completeness_score == 0.0) {
            metrics.completeness_score = 1.0;
        }
        metrics.completeness_score = (metrics.completeness_score * 0.9) + (completeness_score * 0.1);
    } else {
        metrics.failed_requests++;
    }
    
    metrics.last_update = std::chrono::steady_clock::now();
}

std::vector<double> DataAggregator::calculate_dynamic_weights(
    const std::vector<std::string>& provider_names
) {
    std::vector<double> weights;
    
    for (const auto& name : provider_names) {
        const auto& metrics = source_metrics_[name];
        
        // Dynamic weight based on recent performance
        double success_rate = metrics.successful_requests + metrics.failed_requests > 0 ? 
                              (double)metrics.successful_requests / 
                               (metrics.successful_requests + metrics.failed_requests) : 0.0;
        
        // Combine multiple factors for dynamic weight
        double dynamic_weight = success_rate * 0.4 + 
                               metrics.reliability_score * 0.3 + 
                               metrics.latency_score * 0.2 + 
                               metrics.completeness_score * 0.1;
        
        // Apply staleness penalty
        auto time_since_update = std::chrono::duration_cast<std::chrono::minutes>(
            std::chrono::steady_clock::now() - metrics.last_update).count();
        
        if (time_since_update > config_.staleness_penalty_minutes) {
            dynamic_weight *= 0.5; // Apply penalty for stale data
        }
        
        weights.push_back(dynamic_weight);
    }
    
    return weights;
}

OrderBook DataAggregator::aggregate_order_books_simple(
    const std::vector<OrderBook>& order_books
) {
    if (order_books.empty()) {
        return OrderBook{};
    }
    
    // Simple aggregation: use first available order book
    return order_books[0];
}

OrderBook DataAggregator::aggregate_order_books_weighted(
    const std::vector<OrderBook>& order_books,
    const std::vector<double>& weights
) {
    OrderBook aggregated;
    
    if (order_books.empty() || weights.empty()) {
        return order_books.empty() ? OrderBook{} : order_books[0];
    }
    
    // Aggregate bids and asks separately
    std::vector<OrderBookEntry> aggregated_bids, aggregated_asks;
    
    // Weighted average for bids (best price = highest bid)
    for (const auto& order_book : order_books) {
        for (size_t i = 0; i < std::min(order_book.bids.size(), weights.size()); ++i) {
            if (i < weights.size()) {
                OrderBookEntry entry;
                entry.price = order_book.bids[i].price * weights[i];
                entry.quantity = order_book.bids[i].quantity;
                aggregated_bids.push_back(entry);
            }
        }
    }
    
    // Sort bids by price (highest first)
    std::sort(aggregated_bids.begin(), aggregated_bids.end(),
        [](const OrderBookEntry& a, const OrderBookEntry& b) {
            return a.price > b.price;
        });
    
    // Weighted average for asks (best price = lowest ask)
    for (size_t i = 0; i < std::min(order_book.asks.size(), weights.size()); ++i) {
        if (i < weights.size()) {
                OrderBookEntry entry;
                entry.price = order_book.asks[i].price * weights[i];
                entry.quantity = order_book.asks[i].quantity;
                aggregated_asks.push_back(entry);
            }
        }
    }
    
    // Sort asks by price (lowest first)
    std::sort(aggregated_asks.begin(), aggregated_asks.end(),
        [](const OrderBookEntry& a, const OrderBookEntry& b) {
            return a.price < b.price;
        });
    
    aggregated.symbol = order_books[0].symbol;
    aggregated.timestamp = std::chrono::system_clock::now();
    aggregated.bids = aggregated_bids;
    aggregated.asks = aggregated_asks;
    
    return aggregated;
}

void DataAggregator::real_time_aggregation_worker(
    const std::string& symbol,
    std::function<void(const TickData&)> callback,
    const std::vector<std::shared_ptr<DataProvider>>& providers
) {
    std::map<std::string, std::queue<TickData>> tick_queues;
    std::map<std::string, std::chrono::steady_clock::time_point> last_ticks;
    
    // Initialize tick queues for each provider
    for (const auto& provider : providers) {
        if (provider) {
            std::string name = provider->get_provider_name();
            tick_queues[name] = {};
            last_ticks[name] = std::chrono::steady_clock::now();
            
            // Start real-time data collection
            provider->get_real_time_ticks(symbol, [this, name, &tick_queues, &last_ticks, callback](const TickData& tick) {
                std::lock_guard<std::mutex> lock(aggregation_mutex_);
                tick_queues[name].push(tick);
                last_ticks[name] = tick.timestamp;
            });
        }
    }
    
    // Aggregation loop
    while (real_time_active_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 10 Hz aggregation
        
        std::map<std::string, std::vector<TickData>> current_ticks;
        std::vector<std::string> active_providers;
        
        // Collect latest ticks from all providers
        for (auto& [name, queue] : tick_queues) {
            std::lock_guard<std::mutex> lock(aggregation_mutex_);
            
            if (!queue.empty()) {
                current_ticks[name] = queue.back();
                active_providers.push_back(name);
                
                // Keep only recent ticks (last 100)
                while (queue.size() > 100) {
                    queue.pop();
                }
            }
        }
        
        // Aggregate ticks if we have data from multiple providers
        if (active_providers.size() >= config_.min_sources) {
            std::vector<TickData> aggregated_ticks;
            std::vector<std::string> provider_names;
            std::vector<double> weights;
            
            for (const auto& name : active_providers) {
                provider_names.push_back(name);
                weights.push_back(calculate_source_weight(name));
            }
            
            // Use median aggregation for real-time data
            for (const auto& name : active_providers) {
                if (!current_ticks[name].empty()) {
                    aggregated_ticks.push_back(current_ticks[name]);
                }
            }
            
            if (!aggregated_ticks.empty()) {
                // Simple median aggregation for real-time
                std::vector<double> prices;
                for (const auto& tick : aggregated_ticks) {
                    prices.push_back(tick.price);
                }
                
                std::sort(prices.begin(), prices.end());
                double median_price = prices[prices.size() / 2];
                
                TickData aggregated_tick;
                aggregated_tick.symbol = symbol;
                aggregated_tick.price = median_price;
                aggregated_tick.timestamp = std::chrono::system_clock::now();
                aggregated_tick.quantity = 0.0;
                
                if (callback) {
                    callback(aggregated_tick);
                }
            }
        }
    }
}

double DataAggregator::calculate_median(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    
    std::vector<double> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());
    
    size_t n = sorted_values.size();
    if (n % 2 == 0) {
        return sorted_values[n/2];
    } else {
        return (sorted_values[(n/2) - 1] + sorted_values[n/2]) / 2.0;
    }
}

double DataAggregator::calculate_weighted_average(
    const std::vector<double>& values, 
    const std::vector<double>& weights
) {
    if (values.empty() || weights.empty() || values.size() != weights.size()) {
        return 0.0;
    }
    
    double weighted_sum = 0.0;
    double weight_sum = 0.0;
    
    for (size_t i = 0; i < values.size(); ++i) {
        weighted_sum += values[i] * weights[i];
        weight_sum += weights[i];
    }
    
    return weight_sum > 0.0 ? weighted_sum / weight_sum : 0.0;
}

std::vector<double> DataAggregator::calculate_z_scores(const std::vector<double>& values) {
    if (values.size() < 2) {
        return {};
    }
    
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double sum_squares = 0.0;
    
    for (double value : values) {
        sum_squares += (value - mean) * (value - mean);
    }
    
    double std_dev = std::sqrt(sum_squares / values.size());
    
    std::vector<double> z_scores;
    for (double value : values) {
        double z_score = (value - mean) / std_dev;
        z_scores.push_back(z_score);
    }
    
    return z_scores;
}

bool DataAggregator::is_outlier(
    double value, 
    const std::vector<double>& values, 
    double threshold
) {
    if (values.size() < 2) {
        return false;
    }
    
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double sum_squares = 0.0;
    
    for (double val : values) {
        sum_squares += (val - mean) * (val - mean);
    }
    
    double std_dev = std::sqrt(sum_squares / values.size());
    
    double z_score = std::abs((value - mean) / std_dev);
    return z_score > threshold;
}

} // namespace Data
} // namespace ArchNeuronX
