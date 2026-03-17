#ifndef DATA_PROCESSOR_H
#define DATA_PROCESSOR_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <functional>

namespace ArchNeuronX {
namespace Data {

/**
 * @brief OHLCV data structure for candlestick information
 */
struct OHLCVData {
    std::chrono::system_clock::time_point timestamp;
    double open;
    double high;
    double low;
    double close;
    double volume;
    
    // Constructor
    OHLCVData(std::chrono::system_clock::time_point ts, double o, double h, double l, double c, double v)
        : timestamp(ts), open(o), high(h), low(l), close(c), volume(v) {}
};

/**
 * @brief Market data feed interface
 */
class IMarketDataFeed {
public:
    virtual ~IMarketDataFeed() = default;
    
    /**
     * @brief Get historical OHLCV data
     * @param symbol Trading symbol (e.g., "BTCUSDT", "EURUSD")
     * @param interval Time interval ("1m", "5m", "1h", "1d")
     * @param limit Number of candles to fetch
     * @return Vector of OHLCV data
     */
    virtual std::vector<OHLCVData> getHistoricalData(
        const std::string& symbol,
        const std::string& interval,
        int limit = 500
    ) = 0;
    
    /**
     * @brief Get real-time ticker data
     * @param symbol Trading symbol
     * @return Latest OHLCV data
     */
    virtual OHLCVData getRealtimeData(const std::string& symbol) = 0;
    
    /**
     * @brief Check if connection is active
     * @return True if connected
     */
    virtual bool isConnected() const = 0;
};

/**
 * @brief Binance API implementation for crypto data
 */
class BinanceDataFeed : public IMarketDataFeed {
public:
    explicit BinanceDataFeed(const std::string& api_key = "", const std::string& secret_key = "");
    
    std::vector<OHLCVData> getHistoricalData(
        const std::string& symbol,
        const std::string& interval,
        int limit = 500
    ) override;
    
    OHLCVData getRealtimeData(const std::string& symbol) override;
    bool isConnected() const override;
    
private:
    std::string api_key_;
    std::string secret_key_;
    std::string base_url_ = "https://api.binance.com";
    
    std::string makeRequest(const std::string& endpoint, const std::unordered_map<std::string, std::string>& params = {});
};

/**
 * @brief Alpha Vantage API implementation for forex data
 */
class AlphaVantageDataFeed : public IMarketDataFeed {
public:
    explicit AlphaVantageDataFeed(const std::string& api_key);
    
    std::vector<OHLCVData> getHistoricalData(
        const std::string& symbol,
        const std::string& interval,
        int limit = 500
    ) override;
    
    OHLCVData getRealtimeData(const std::string& symbol) override;
    bool isConnected() const override;
    
private:
    std::string api_key_;
    std::string base_url_ = "https://www.alphavantage.co";
    
    std::string makeRequest(const std::string& function, const std::unordered_map<std::string, std::string>& params);
};

/**
 * @brief Technical indicators calculator
 */
class TechnicalIndicators {
public:
    /**
     * @brief Calculate Simple Moving Average
     * @param prices Price vector
     * @param period Moving average period
     * @return SMA values
     */
    static std::vector<double> calculateSMA(const std::vector<double>& prices, int period);
    
    /**
     * @brief Calculate Exponential Moving Average
     * @param prices Price vector
     * @param period EMA period
     * @return EMA values
     */
    static std::vector<double> calculateEMA(const std::vector<double>& prices, int period);
    
    /**
     * @brief Calculate Relative Strength Index
     * @param prices Price vector
     * @param period RSI period (default 14)
     * @return RSI values
     */
    static std::vector<double> calculateRSI(const std::vector<double>& prices, int period = 14);
    
    /**
     * @brief Calculate MACD (Moving Average Convergence Divergence)
     * @param prices Price vector
     * @param fast_period Fast EMA period (default 12)
     * @param slow_period Slow EMA period (default 26)
     * @param signal_period Signal line period (default 9)
     * @return MACD line, signal line, and histogram
     */
    static std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> 
    calculateMACD(const std::vector<double>& prices, int fast_period = 12, int slow_period = 26, int signal_period = 9);
    
    /**
     * @brief Calculate Bollinger Bands
     * @param prices Price vector
     * @param period Moving average period (default 20)
     * @param std_dev Standard deviation multiplier (default 2.0)
     * @return Upper band, middle band (SMA), lower band
     */
    static std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    calculateBollingerBands(const std::vector<double>& prices, int period = 20, double std_dev = 2.0);
    
    /**
     * @brief Calculate Stochastic Oscillator
     * @param highs High prices
     * @param lows Low prices
     * @param closes Close prices
     * @param k_period %K period (default 14)
     * @param d_period %D period (default 3)
     * @return %K and %D values
     */
    static std::pair<std::vector<double>, std::vector<double>>
    calculateStochastic(const std::vector<double>& highs, const std::vector<double>& lows, 
                       const std::vector<double>& closes, int k_period = 14, int d_period = 3);
};

/**
 * @brief Data normalization and preprocessing utilities
 */
class DataNormalizer {
public:
    enum class NormalizationType {
        MINMAX,      // Min-Max scaling to [0, 1]
        ZSCORE,      // Z-score standardization
        ROBUST,      // Robust scaling using median and IQR
        TANH         // Tanh normalization
    };
    
    /**
     * @brief Normalize data using specified method
     * @param data Input data tensor
     * @param type Normalization type
     * @return Normalized tensor and scaling parameters
     */
    static std::pair<torch::Tensor, std::unordered_map<std::string, double>>
    normalize(const torch::Tensor& data, NormalizationType type = NormalizationType::MINMAX);
    
    /**
     * @brief Denormalize data using saved scaling parameters
     * @param normalized_data Normalized tensor
     * @param params Scaling parameters from normalization
     * @param type Normalization type used
     * @return Original scale tensor
     */
    static torch::Tensor denormalize(
        const torch::Tensor& normalized_data,
        const std::unordered_map<std::string, double>& params,
        NormalizationType type
    );
    
    /**
     * @brief Create sliding windows for time series
     * @param data Input data tensor [time_steps, features]
     * @param window_size Window size
     * @param step_size Step size for sliding
     * @return Windowed tensor [num_windows, window_size, features]
     */
    static torch::Tensor createSlidingWindows(
        const torch::Tensor& data,
        int window_size,
        int step_size = 1
    );
};

/**
 * @brief Main data processor orchestrating all data operations
 */
class DataProcessor {
public:
    struct Config {
        std::vector<std::string> symbols = {"BTCUSDT", "ETHUSDT", "EURUSD"};
        std::string interval = "1h";
        int history_limit = 1000;
        int window_size = 50;
        int step_size = 1;
        DataNormalizer::NormalizationType normalization = DataNormalizer::NormalizationType::MINMAX;
        bool include_technical_indicators = true;
        std::vector<std::string> indicators = {"SMA_20", "EMA_12", "RSI_14", "MACD", "BB_20"};
        int update_interval_ms = 60000; // 1 minute
    };
    
    explicit DataProcessor(const Config& config);
    ~DataProcessor();
    
    /**
     * @brief Initialize data feeds
     * @param crypto_feed Cryptocurrency data feed
     * @param forex_feed Forex data feed
     */
    void initializeFeeds(
        std::shared_ptr<IMarketDataFeed> crypto_feed,
        std::shared_ptr<IMarketDataFeed> forex_feed = nullptr
    );
    
    /**
     * @brief Start real-time data collection
     */
    void startRealtimeCollection();
    
    /**
     * @brief Stop real-time data collection
     */
    void stopRealtimeCollection();
    
    /**
     * @brief Get processed training data
     * @param symbol Trading symbol
     * @return Processed tensor ready for ML training
     */
    torch::Tensor getTrainingData(const std::string& symbol);
    
    /**
     * @brief Get latest processed data for inference
     * @param symbol Trading symbol
     * @return Latest processed tensor for prediction
     */
    torch::Tensor getLatestData(const std::string& symbol);
    
    /**
     * @brief Set data callback for real-time updates
     * @param callback Function to call when new data arrives
     */
    void setDataCallback(std::function<void(const std::string&, const torch::Tensor&)> callback);
    
    /**
     * @brief Save processed data to disk
     * @param path File path to save data
     * @param symbol Trading symbol
     */
    void saveProcessedData(const std::string& path, const std::string& symbol);
    
    /**
     * @brief Load processed data from disk
     * @param path File path to load data from
     * @param symbol Trading symbol
     */
    void loadProcessedData(const std::string& path, const std::string& symbol);
    
private:
    Config config_;
    std::shared_ptr<IMarketDataFeed> crypto_feed_;
    std::shared_ptr<IMarketDataFeed> forex_feed_;
    
    // Data storage
    std::unordered_map<std::string, std::vector<OHLCVData>> raw_data_;
    std::unordered_map<std::string, torch::Tensor> processed_data_;
    std::unordered_map<std::string, std::unordered_map<std::string, double>> normalization_params_;
    
    // Threading
    std::unique_ptr<std::thread> realtime_thread_;
    std::atomic<bool> stop_realtime_{false};
    std::mutex data_mutex_;
    
    // Callback
    std::function<void(const std::string&, const torch::Tensor&)> data_callback_;
    
    // Internal methods
    torch::Tensor processRawData(const std::vector<OHLCVData>& ohlcv_data, const std::string& symbol);
    torch::Tensor addTechnicalIndicators(const torch::Tensor& base_data, const std::vector<OHLCVData>& ohlcv_data);
    void realtimeCollectionLoop();
    std::vector<double> extractPrices(const std::vector<OHLCVData>& data, const std::string& price_type = "close");
};

} // namespace Data
} // namespace ArchNeuronX

#endif // DATA_PROCESSOR_H