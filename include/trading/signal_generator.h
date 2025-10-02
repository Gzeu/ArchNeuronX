#ifndef SIGNAL_GENERATOR_H
#define SIGNAL_GENERATOR_H

#include <torch/torch.h>
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>

namespace ArchNeuronX {
namespace Trading {

/**
 * @brief Trading signal enumeration
 */
enum class SignalType {
    BUY,
    SELL,
    HOLD
};

/**
 * @brief Trading signal with metadata
 */
struct TradingSignal {
    SignalType signal;
    double confidence;                           // Confidence score [0.0, 1.0]
    std::string symbol;
    std::chrono::system_clock::time_point timestamp;
    double predicted_price;                      // Optional price prediction
    double current_price;
    std::string explanation;                     // Human-readable explanation
    std::unordered_map<std::string, double> features; // Contributing factors
    
    // Risk management
    double stop_loss = 0.0;
    double take_profit = 0.0;
    double position_size = 0.0;
    
    // Constructor
    TradingSignal(SignalType sig, double conf, const std::string& sym)
        : signal(sig), confidence(conf), symbol(sym), 
          timestamp(std::chrono::system_clock::now()) {}
};

/**
 * @brief Signal generator interface
 */
class ISignalGenerator {
public:
    virtual ~ISignalGenerator() = default;
    
    /**
     * @brief Generate trading signal from input data
     * @param input_data Preprocessed market data
     * @param symbol Trading symbol
     * @return Generated trading signal
     */
    virtual TradingSignal generateSignal(
        const torch::Tensor& input_data,
        const std::string& symbol
    ) = 0;
    
    /**
     * @brief Batch signal generation
     * @param batch_data Batch of input data [batch_size, sequence_length, features]
     * @param symbols Corresponding symbols for each batch item
     * @return Vector of trading signals
     */
    virtual std::vector<TradingSignal> generateBatchSignals(
        const torch::Tensor& batch_data,
        const std::vector<std::string>& symbols
    ) = 0;
    
    /**
     * @brief Set confidence threshold for signal generation
     * @param threshold Minimum confidence threshold [0.0, 1.0]
     */
    virtual void setConfidenceThreshold(double threshold) = 0;
};

/**
 * @brief Neural network-based signal generator
 */
class NeuralSignalGenerator : public ISignalGenerator {
public:
    struct Config {
        std::string model_path;
        std::string model_type = "MLP";           // "MLP", "CNN", "HYBRID"
        double confidence_threshold = 0.6;        // Minimum confidence for signals
        bool use_ensemble = false;                // Use multiple models for voting
        std::vector<std::string> ensemble_paths;  // Paths to ensemble models
        bool enable_risk_management = true;       // Calculate stop-loss/take-profit
        double stop_loss_pct = 0.02;             // Stop-loss percentage (2%)
        double take_profit_pct = 0.04;           // Take-profit percentage (4%)
        int lookback_period = 50;                // Historical periods for analysis
    };
    
    explicit NeuralSignalGenerator(const Config& config);
    ~NeuralSignalGenerator();
    
    TradingSignal generateSignal(
        const torch::Tensor& input_data,
        const std::string& symbol
    ) override;
    
    std::vector<TradingSignal> generateBatchSignals(
        const torch::Tensor& batch_data,
        const std::vector<std::string>& symbols
    ) override;
    
    void setConfidenceThreshold(double threshold) override;
    
    /**
     * @brief Load trained model
     * @param model_path Path to saved model
     */
    void loadModel(const std::string& model_path);
    
    /**
     * @brief Enable/disable GPU acceleration
     * @param enable Enable GPU if available
     */
    void setGPUAcceleration(bool enable);
    
    /**
     * @brief Get model performance metrics
     * @return Performance statistics
     */
    std::unordered_map<std::string, double> getPerformanceMetrics() const;
    
private:
    Config config_;
    std::shared_ptr<torch::nn::Module> model_;
    std::vector<std::shared_ptr<torch::nn::Module>> ensemble_models_;
    torch::Device device_{torch::kCPU};
    
    // Performance tracking
    mutable std::unordered_map<std::string, double> performance_metrics_;
    
    // Internal methods
    torch::Tensor preprocessInput(const torch::Tensor& input_data);
    SignalType interpretPrediction(const torch::Tensor& prediction);
    double calculateConfidence(const torch::Tensor& prediction);
    std::string generateExplanation(
        const TradingSignal& signal,
        const torch::Tensor& input_data,
        const torch::Tensor& prediction
    );
    void calculateRiskLevels(TradingSignal& signal, const torch::Tensor& input_data);
    torch::Tensor ensemblePredict(const torch::Tensor& input_data);
};

/**
 * @brief Technical analysis-based signal generator
 */
class TechnicalSignalGenerator : public ISignalGenerator {
public:
    struct Config {
        double rsi_oversold = 30.0;
        double rsi_overbought = 70.0;
        double macd_signal_threshold = 0.0;
        double bb_squeeze_threshold = 0.02;
        double volume_spike_threshold = 2.0;
        std::vector<std::string> enabled_indicators = {"RSI", "MACD", "BB", "STOCH"};
        double confidence_threshold = 0.5;
    };
    
    explicit TechnicalSignalGenerator(const Config& config);
    
    TradingSignal generateSignal(
        const torch::Tensor& input_data,
        const std::string& symbol
    ) override;
    
    std::vector<TradingSignal> generateBatchSignals(
        const torch::Tensor& batch_data,
        const std::vector<std::string>& symbols
    ) override;
    
    void setConfidenceThreshold(double threshold) override;
    
private:
    Config config_;
    
    SignalType analyzeRSI(const torch::Tensor& rsi_values);
    SignalType analyzeMACD(const torch::Tensor& macd_line, const torch::Tensor& signal_line);
    SignalType analyzeBollingerBands(const torch::Tensor& price, const torch::Tensor& upper_band, const torch::Tensor& lower_band);
    SignalType analyzeStochastic(const torch::Tensor& k_values, const torch::Tensor& d_values);
    double calculateTechnicalConfidence(const std::vector<SignalType>& signals);
};

/**
 * @brief Hybrid signal generator combining neural and technical analysis
 */
class HybridSignalGenerator : public ISignalGenerator {
public:
    struct Config {
        NeuralSignalGenerator::Config neural_config;
        TechnicalSignalGenerator::Config technical_config;
        double neural_weight = 0.7;               // Weight for neural signals
        double technical_weight = 0.3;            // Weight for technical signals
        double agreement_bonus = 0.1;             // Bonus when both agree
        double disagreement_penalty = 0.2;        // Penalty when they disagree
    };
    
    explicit HybridSignalGenerator(const Config& config);
    
    TradingSignal generateSignal(
        const torch::Tensor& input_data,
        const std::string& symbol
    ) override;
    
    std::vector<TradingSignal> generateBatchSignals(
        const torch::Tensor& batch_data,
        const std::vector<std::string>& symbols
    ) override;
    
    void setConfidenceThreshold(double threshold) override;
    
private:
    Config config_;
    std::unique_ptr<NeuralSignalGenerator> neural_generator_;
    std::unique_ptr<TechnicalSignalGenerator> technical_generator_;
    
    TradingSignal combineSignals(
        const TradingSignal& neural_signal,
        const TradingSignal& technical_signal
    );
};

/**
 * @brief Signal history and performance tracking
 */
class SignalTracker {
public:
    struct SignalPerformance {
        int total_signals = 0;
        int correct_signals = 0;
        double accuracy = 0.0;
        double avg_confidence = 0.0;
        double total_return = 0.0;
        double sharpe_ratio = 0.0;
        std::unordered_map<std::string, int> signal_counts; // Count by type
    };
    
    /**
     * @brief Add a signal to tracking
     * @param signal Trading signal to track
     */
    void addSignal(const TradingSignal& signal);
    
    /**
     * @brief Update signal outcome
     * @param signal_id Signal identifier
     * @param outcome True if signal was correct
     * @param return_pct Return percentage achieved
     */
    void updateSignalOutcome(const std::string& signal_id, bool outcome, double return_pct);
    
    /**
     * @brief Get performance statistics
     * @param symbol Optional symbol filter
     * @return Performance metrics
     */
    SignalPerformance getPerformanceStats(const std::string& symbol = "") const;
    
    /**
     * @brief Get recent signals
     * @param count Number of recent signals to return
     * @param symbol Optional symbol filter
     * @return Vector of recent signals
     */
    std::vector<TradingSignal> getRecentSignals(int count = 10, const std::string& symbol = "") const;
    
    /**
     * @brief Save signal history to file
     * @param path File path to save to
     */
    void saveHistory(const std::string& path) const;
    
    /**
     * @brief Load signal history from file
     * @param path File path to load from
     */
    void loadHistory(const std::string& path);
    
private:
    std::vector<TradingSignal> signal_history_;
    std::unordered_map<std::string, std::pair<bool, double>> signal_outcomes_; // signal_id -> (outcome, return)
    mutable std::mutex history_mutex_;
    
    std::string generateSignalId(const TradingSignal& signal) const;
};

/**
 * @brief Signal broadcasting and notification system
 */
class SignalBroadcaster {
public:
    enum class NotificationType {
        CONSOLE,
        FILE,
        WEBHOOK,
        EMAIL
    };
    
    struct Config {
        std::vector<NotificationType> enabled_notifications = {NotificationType::CONSOLE};
        std::string log_file_path = "signals.log";
        std::string webhook_url;
        std::string email_smtp_server;
        std::string email_username;
        std::string email_password;
        std::vector<std::string> email_recipients;
    };
    
    explicit SignalBroadcaster(const Config& config);
    
    /**
     * @brief Broadcast a trading signal
     * @param signal Signal to broadcast
     */
    void broadcastSignal(const TradingSignal& signal);
    
    /**
     * @brief Set signal filter criteria
     * @param min_confidence Minimum confidence to broadcast
     * @param symbols Symbols to filter (empty for all)
     */
    void setFilter(double min_confidence, const std::vector<std::string>& symbols = {});
    
private:
    Config config_;
    double min_confidence_ = 0.0;
    std::vector<std::string> filtered_symbols_;
    
    void sendConsoleNotification(const TradingSignal& signal);
    void sendFileNotification(const TradingSignal& signal);
    void sendWebhookNotification(const TradingSignal& signal);
    void sendEmailNotification(const TradingSignal& signal);
    std::string formatSignalMessage(const TradingSignal& signal);
};

/**
 * @brief Signal generation factory
 */
class SignalGeneratorFactory {
public:
    enum class GeneratorType {
        NEURAL,
        TECHNICAL,
        HYBRID
    };
    
    /**
     * @brief Create signal generator
     * @param type Generator type
     * @param config_path Path to configuration file
     * @return Unique pointer to signal generator
     */
    static std::unique_ptr<ISignalGenerator> createGenerator(
        GeneratorType type,
        const std::string& config_path = ""
    );
};

} // namespace Trading
} // namespace ArchNeuronX

#endif // SIGNAL_GENERATOR_H