/**
 * @file data_manager.hpp
 * @brief Data management orchestrator for ArchNeuronX
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <functional>

#include "data_provider.hpp"
#include "data_preprocessor.hpp"
#include "technical_indicators.hpp"
#include "core/logger.hpp"
#include "core/config.hpp"

namespace ArchNeuronX {
namespace Data {

/**
 * @enum DataManagerState
 * @brief Data manager operational states
 */
enum class DataManagerState {
    IDLE,           ///< Not running
    INITIALIZING,   ///< Initialization in progress
    RUNNING,        ///< Actively collecting data
    PAUSED,         ///< Temporarily paused
    ERROR,          ///< Error state
    STOPPING        ///< Shutdown in progress
};

/**
 * @struct DataSubscription
 * @brief Data subscription configuration
 */
struct DataSubscription {
    std::string symbol;
    TimeFrame timeframe;
    bool real_time = true;
    bool historical = false;
    std::chrono::system_clock::time_point historical_start;
    std::chrono::system_clock::time_point historical_end;
    DataCallback callback;
    
    std::string getId() const {
        return symbol + "_" + timeFrameToString(timeframe);
    }
};

/**
 * @struct DataManagerConfig
 * @brief Configuration for data manager
 */
struct DataManagerConfig {
    // Data providers
    std::vector<DataProviderConfig> provider_configs;
    
    // Data collection
    std::vector<std::string> symbols;
    std::vector<TimeFrame> timeframes;
    bool enable_real_time = true;
    bool enable_historical = true;
    
    // Historical data settings
    int historical_days = 365;          ///< Days of historical data to fetch
    bool backfill_missing_data = true;
    
    // Data processing
    PreprocessingConfig preprocessing_config;
    bool enable_technical_indicators = true;
    bool enable_data_validation = true;
    
    // Storage settings
    std::string data_directory = "data/";
    bool enable_data_caching = true;
    int cache_size_mb = 1024;           ///< Cache size in MB
    bool persist_to_disk = true;
    
    // Performance settings
    int max_concurrent_requests = 10;
    int data_buffer_size = 10000;
    int update_interval_ms = 1000;      ///< Real-time update interval
    
    /**
     * @brief Validate configuration
     * @throws std::invalid_argument if configuration is invalid
     */
    void validate() const;
};

/**
 * @struct DataStatistics
 * @brief Data collection and processing statistics
 */
struct DataStatistics {
    uint64_t total_data_points = 0;
    uint64_t processed_data_points = 0;
    uint64_t failed_requests = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_update;
    
    double data_points_per_second = 0.0;
    double cache_hit_ratio = 0.0;
    
    /**
     * @brief Convert to JSON string
     * @return JSON representation
     */
    std::string toJson() const;
};

/**
 * @class DataManager
 * @brief Orchestrates data collection, processing, and distribution
 */
class DataManager {
public:
    /**
     * @brief Constructor
     * @param config Data manager configuration
     */
    explicit DataManager(const DataManagerConfig& config);
    
    /**
     * @brief Destructor
     */
    ~DataManager();
    
    /**
     * @brief Initialize data manager and providers
     * @return True if initialization successful
     */
    bool initialize();
    
    /**
     * @brief Start data collection and processing
     * @return True if started successfully
     */
    bool start();
    
    /**
     * @brief Stop data collection and processing
     */
    void stop();
    
    /**
     * @brief Pause data collection temporarily
     */
    void pause();
    
    /**
     * @brief Resume data collection
     */
    void resume();
    
    /**
     * @brief Subscribe to real-time data updates
     * @param subscription Data subscription configuration
     * @return True if subscription successful
     */
    bool subscribeToData(const DataSubscription& subscription);
    
    /**
     * @brief Unsubscribe from data updates
     * @param symbol Trading symbol
     * @param timeframe Data timeframe
     * @return True if unsubscription successful
     */
    bool unsubscribeFromData(const std::string& symbol, TimeFrame timeframe);
    
    /**
     * @brief Get historical data for a symbol
     * @param symbol Trading symbol
     * @param timeframe Data timeframe
     * @param start_time Start time
     * @param end_time End time
     * @param include_indicators Whether to calculate technical indicators
     * @return Future containing historical data
     */
    std::future<std::vector<MarketDataPoint>> getHistoricalData(
        const std::string& symbol,
        TimeFrame timeframe,
        const std::chrono::system_clock::time_point& start_time,
        const std::chrono::system_clock::time_point& end_time,
        bool include_indicators = true
    );
    
    /**
     * @brief Get latest data point for a symbol
     * @param symbol Trading symbol
     * @param timeframe Data timeframe
     * @return Latest market data point
     */
    std::optional<MarketDataPoint> getLatestData(
        const std::string& symbol,
        TimeFrame timeframe
    );
    
    /**
     * @brief Get processed dataset for ML training
     * @param symbols Vector of symbols to include
     * @param timeframe Data timeframe
     * @param target_column Target column for prediction
     * @return Processed dataset ready for ML
     */
    std::future<ProcessedDataset> getMLDataset(
        const std::vector<std::string>& symbols,
        TimeFrame timeframe,
        const std::string& target_column = "close"
    );
    
    /**
     * @brief Validate data quality for a symbol
     * @param symbol Trading symbol
     * @param timeframe Data timeframe
     * @param start_time Start time for validation
     * @param end_time End time for validation
     * @return Data quality report
     */
    std::map<std::string, double> validateDataQuality(
        const std::string& symbol,
        TimeFrame timeframe,
        const std::chrono::system_clock::time_point& start_time,
        const std::chrono::system_clock::time_point& end_time
    );
    
    /**
     * @brief Get available symbols from all providers
     * @return Vector of available symbols
     */
    std::vector<std::string> getAvailableSymbols();
    
    /**
     * @brief Get market information for symbols
     * @param symbols Vector of symbols
     * @return Map of symbol to market info
     */
    std::map<std::string, MarketInfo> getMarketInfo(
        const std::vector<std::string>& symbols
    );
    
    /**
     * @brief Add custom data provider
     * @param provider Unique pointer to data provider
     * @return True if provider added successfully
     */
    bool addDataProvider(std::unique_ptr<DataProvider> provider);
    
    /**
     * @brief Remove data provider by name
     * @param provider_name Name of provider to remove
     * @return True if provider removed
     */
    bool removeDataProvider(const std::string& provider_name);
    
    /**
     * @brief Clear data cache
     * @param symbol Optional symbol to clear (empty = clear all)
     */
    void clearCache(const std::string& symbol = "");
    
    /**
     * @brief Export data to various formats
     * @param symbol Trading symbol
     * @param timeframe Data timeframe
     * @param start_time Start time
     * @param end_time End time
     * @param format Export format ("csv", "json", "parquet")
     * @param filepath Output file path
     * @return True if export successful
     */
    bool exportData(
        const std::string& symbol,
        TimeFrame timeframe,
        const std::chrono::system_clock::time_point& start_time,
        const std::chrono::system_clock::time_point& end_time,
        const std::string& format,
        const std::string& filepath
    );
    
    // Getters
    DataManagerState getState() const { return state_.load(); }
    const DataManagerConfig& getConfig() const { return config_; }
    DataStatistics getStatistics() const;
    size_t getActiveSubscriptions() const;
    std::vector<std::string> getConnectedProviders() const;
    
    /**
     * @brief Get data manager health status
     * @return Map of health metrics
     */
    std::map<std::string, std::string> getHealthStatus() const;
    
private:
    /**
     * @brief Initialize all configured data providers
     * @return True if all providers initialized successfully
     */
    bool initializeProviders();
    
    /**
     * @brief Data collection worker thread
     */
    void dataCollectionWorker();
    
    /**
     * @brief Data processing worker thread
     */
    void dataProcessingWorker();
    
    /**
     * @brief Handle real-time data callback
     * @param data_point Market data point
     */
    void handleRealtimeData(const MarketDataPoint& data_point);
    
    /**
     * @brief Process and enrich market data with indicators
     * @param raw_data Raw OHLCV data
     * @param symbol Trading symbol
     * @param timeframe Data timeframe
     * @return Enriched market data points
     */
    std::vector<MarketDataPoint> enrichMarketData(
        const std::vector<OHLCV>& raw_data,
        const std::string& symbol,
        TimeFrame timeframe
    );
    
    /**
     * @brief Cache market data
     * @param symbol Trading symbol
     * @param timeframe Data timeframe
     * @param data Market data points
     */
    void cacheData(
        const std::string& symbol,
        TimeFrame timeframe,
        const std::vector<MarketDataPoint>& data
    );
    
    /**
     * @brief Retrieve data from cache
     * @param symbol Trading symbol
     * @param timeframe Data timeframe
     * @param start_time Start time
     * @param end_time End time
     * @return Cached data points (if available)
     */
    std::vector<MarketDataPoint> getCachedData(
        const std::string& symbol,
        TimeFrame timeframe,
        const std::chrono::system_clock::time_point& start_time,
        const std::chrono::system_clock::time_point& end_time
    );
    
    /**
     * @brief Update statistics
     */
    void updateStatistics();
    
    /**
     * @brief Get cache key for symbol and timeframe
     * @param symbol Trading symbol
     * @param timeframe Data timeframe
     * @return Cache key string
     */
    std::string getCacheKey(const std::string& symbol, TimeFrame timeframe) const;
    
    DataManagerConfig config_;
    std::atomic<DataManagerState> state_{DataManagerState::IDLE};
    
    // Data providers
    std::vector<std::unique_ptr<DataProvider>> providers_;
    mutable std::mutex providers_mutex_;
    
    // Data preprocessing
    std::unique_ptr<DataPreprocessor> preprocessor_;
    
    // Threading
    std::unique_ptr<std::thread> collection_thread_;
    std::unique_ptr<std::thread> processing_thread_;
    std::atomic<bool> running_{false};
    
    // Data cache
    mutable std::mutex cache_mutex_;
    std::map<std::string, std::vector<MarketDataPoint>> data_cache_;
    
    // Subscriptions
    mutable std::mutex subscriptions_mutex_;
    std::map<std::string, DataSubscription> active_subscriptions_;
    
    // Data queue for processing
    std::queue<MarketDataPoint> data_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    DataStatistics statistics_;
};

/**
 * @brief Convert DataManagerState to string
 * @param state Data manager state
 * @return String representation
 */
std::string dataManagerStateToString(DataManagerState state);

} // namespace Data
} // namespace ArchNeuronX