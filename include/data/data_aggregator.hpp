/**
 * @file data_aggregator.hpp
 * @brief Data aggregator for combining multiple data sources
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <algorithm>
#include <chrono>
#include <functional>
#include <mutex>
#include <atomic>

#include "data_provider.hpp"
#include "market_data.hpp"
#include "technical_indicators.hpp"
#include "core/logger.hpp"

namespace ArchNeuronX {
namespace Data {

/**
 * @enum AggregationMethod
 * @brief Methods for aggregating multiple data sources
 */
enum class AggregationMethod {
    FIRST_AVAILABLE,     ///< Use first available source
    WEIGHTED_AVERAGE,   ///< Weighted average based on source reliability
    MEDIAN,            ///< Median of all sources
    CONSENSUS,          ///> Use values that agree between sources
    LATEST_TIMESTAMP,   ///< Use data with latest timestamp
    BEST_QUALITY,       ///< Prefer source with best data quality score
    CUSTOM              ///< Custom aggregation function
};

/**
 * @struct DataSourceWeight
 * @brief Weight configuration for data sources
 */
struct DataSourceWeight {
    std::string provider_name;
    double weight;
    double reliability_score;
    double latency_score;
    double completeness_score;
    
    DataSourceWeight(const std::string& name, double w, double reliability, 
                   double latency, double completeness)
        : provider_name(name), weight(w), reliability_score(reliability),
          latency_score(latency), completeness_score(completeness) {}
};

/**
 * @struct AggregationConfig
 * @brief Configuration for data aggregation
 */
struct AggregationConfig {
    AggregationMethod method = AggregationMethod::WEIGHTED_AVERAGE;
    std::vector<DataSourceWeight> source_weights;
    bool enable_quality_scoring = true;
    bool enable_outlier_detection = true;
    double outlier_threshold = 3.0;  // Standard deviations
    int min_sources = 2;
    int max_sources = 5;
    double staleness_penalty_minutes = 5.0;
    bool enable_cross_validation = true;
};

/**
 * @struct AggregationResult
 * @brief Result of data aggregation with metadata
 */
struct AggregationResult {
    std::vector<OHLCV> aggregated_data;
    std::map<std::string, std::vector<OHLCV>> source_data;
    std::map<std::string, double> source_weights_used;
    std::map<std::string, double> quality_scores;
    std::chrono::system_clock::time_point aggregation_timestamp;
    std::string aggregation_method_used;
    size_t total_sources;
    size_t successful_sources;
    std::vector<std::string> failed_sources;
    std::vector<std::string> outlier_removed;
    
    std::string to_json() const;
};

/**
 * @class DataAggregator
 * @brief Aggregates data from multiple providers for improved reliability
 */
class DataAggregator {
public:
    /**
     * @brief Constructor
     * @param config Aggregation configuration
     */
    explicit DataAggregator(const AggregationConfig& config);
    
    /**
     * @brief Destructor
     */
    ~DataAggregator();
    
    /**
     * @brief Aggregate historical data from multiple providers
     * @param symbol Trading symbol
     * @param timeframe Data timeframe
     * @param start Start time
     * @param end End time
     * @param providers Vector of provider instances
     * @return Aggregation result with metadata
     */
    std::future<AggregationResult> aggregate_historical_data(
        const std::string& symbol,
        const std::string& timeframe,
        const std::chrono::system_clock::time_point& start,
        const std::chrono::system_clock::time_point& end,
        const std::vector<std::shared_ptr<DataProvider>>& providers
    );
    
    /**
     * @brief Aggregate current price from multiple providers
     * @param symbol Trading symbol
     * @param providers Vector of provider instances
     * @return Aggregation result with current prices
     */
    std::future<AggregationResult> aggregate_current_prices(
        const std::string& symbol,
        const std::vector<std::shared_ptr<DataProvider>>& providers
    );
    
    /**
     * @brief Aggregate order books from multiple providers
     * @param symbol Trading symbol
     * @param depth Order book depth
     * @param providers Vector of provider instances
     * @return Aggregated order book
     */
    std::future<OrderBook> aggregate_order_books(
        const std::string& symbol,
        int depth,
        const std::vector<std::shared_ptr<DataProvider>>& providers
    );
    
    /**
     * @brief Real-time aggregation with continuous updates
     * @param symbol Trading symbol
     * @param callback Callback for aggregated ticks
     * @param providers Vector of provider instances
     */
    void start_real_time_aggregation(
        const std::string& symbol,
        std::function<void(const TickData&)> callback,
        const std::vector<std::shared_ptr<DataProvider>>& providers
    );
    
    /**
     * @brief Stop real-time aggregation
     */
    void stop_real_time_aggregation();
    
    /**
     * @brief Update aggregation configuration
     * @param config New configuration
     */
    void update_config(const AggregationConfig& config);
    
    /**
     * @brief Get aggregation statistics
     * @return Statistics about aggregation performance
     */
    std::map<std::string, double> get_aggregation_stats() const;

private:
    AggregationConfig config_;
    std::atomic<bool> real_time_active_;
    std::vector<std::thread> aggregation_threads_;
    mutable std::mutex aggregation_mutex_;
    
    // Quality scoring
    struct SourceMetrics {
        double accuracy_score;
        double latency_score;
        double completeness_score;
        double reliability_score;
        std::chrono::steady_clock::time_point last_update;
        int successful_requests;
        int failed_requests;
    };
    
    std::map<std::string, SourceMetrics> source_metrics_;
    mutable std::mutex metrics_mutex_;
    
    // Private methods
    std::vector<OHLCV> align_time_series(
        const std::vector<std::vector<OHLCV>>& source_data);
    
    std::vector<OHLCV> detect_and_remove_outliers(
        const std::vector<OHLCV>& data);
    
    std::vector<OHLCV> aggregate_by_first_available(
        const std::vector<std::vector<OHLCV>>& source_data);
    
    std::vector<OHLCV> aggregate_by_weighted_average(
        const std::vector<std::vector<OHLCV>>& source_data,
        const std::vector<double>& weights);
    
    std::vector<OHLCV> aggregate_by_median(
        const std::vector<std::vector<OHLCV>>& source_data);
    
    std::vector<OHLCV> aggregate_by_consensus(
        const std::vector<std::vector<OHLCV>>& source_data);
    
    std::vector<OHLCV> aggregate_by_latest_timestamp(
        const std::vector<std::vector<OHLCV>>& source_data);
    
    std::vector<OHLCV> aggregate_by_best_quality(
        const std::vector<std::vector<OHLCV>>& source_data);
    
    std::vector<OHLCV> custom_aggregation(
        const std::vector<std::vector<OHLCV>>& source_data,
        std::function<std::vector<OHLCV>(const std::vector<std::vector<OHLCV>&)>> custom_func);
    
    double calculate_source_weight(const std::string& provider_name);
    void update_source_metrics(const std::string& provider_name, 
                           bool success, 
                           double latency_ms,
                           double completeness_score);
    
    std::vector<double> calculate_dynamic_weights(
        const std::vector<std::string>& provider_names);
    
    OrderBook aggregate_order_books_simple(
        const std::vector<OrderBook>& order_books);
    
    OrderBook aggregate_order_books_weighted(
        const std::vector<OrderBook>& order_books,
        const std::vector<double>& weights);
    
    void real_time_aggregation_worker(
        const std::string& symbol,
        std::function<void(const TickData&)> callback,
        const std::vector<std::shared_ptr<DataProvider>>& providers);
    
    // Statistical utilities
    double calculate_median(const std::vector<double>& values);
    double calculate_weighted_average(const std::vector<double>& values, 
                                      const std::vector<double>& weights);
    std::vector<double> calculate_z_scores(const std::vector<double>& values);
    bool is_outlier(double value, const std::vector<double>& values, double threshold);
};

} // namespace Data
} // namespace ArchNeuronX
