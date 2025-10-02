/**
 * @file training_manager.hpp
 * @brief Training manager with hyperparameter optimization for ArchNeuronX
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <functional>
#include <future>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>

#include <torch/torch.h>
#include "core/base_model.hpp"
#include "core/logger.hpp"
#include "core/config.hpp"

namespace ArchNeuronX {
namespace Models {

/**
 * @enum OptimizerType
 * @brief Supported optimizer types
 */
enum class OptimizerType {
    ADAM,       ///< Adam optimizer
    ADAMW,      ///< AdamW optimizer
    SGD,        ///< Stochastic Gradient Descent
    RMSPROP,    ///< RMSprop optimizer
    ADAGRAD     ///< Adagrad optimizer
};

/**
 * @enum SchedulerType
 * @brief Learning rate scheduler types
 */
enum class SchedulerType {
    NONE,           ///< No scheduler
    STEP_LR,        ///< Step decay
    EXPONENTIAL_LR, ///< Exponential decay
    COSINE_LR,      ///< Cosine annealing
    REDUCE_ON_PLATEAU, ///< Reduce on plateau
    CYCLIC_LR       ///< Cyclic learning rate
};

/**
 * @enum LossFunctionType
 * @brief Loss function types for trading
 */
enum class LossFunctionType {
    MSE,            ///< Mean Squared Error
    MAE,            ///< Mean Absolute Error
    HUBER,          ///< Huber Loss
    CROSS_ENTROPY,  ///< Cross Entropy (for classification)
    FOCAL,          ///< Focal Loss (for imbalanced data)
    TRADING_LOSS,   ///< Custom trading-specific loss
    SHARPE_LOSS     ///< Sharpe ratio-based loss
};

/**
 * @struct HyperparameterRange
 * @brief Range for hyperparameter optimization
 */
struct HyperparameterRange {
    std::string name;
    double min_value;
    double max_value;
    bool is_log_scale = false;  ///< Use log scale for search
    bool is_integer = false;    ///< Parameter is integer
    
    HyperparameterRange() = default;
    HyperparameterRange(const std::string& n, double min_val, double max_val, 
                       bool log_scale = false, bool integer = false)
        : name(n), min_value(min_val), max_value(max_val), 
          is_log_scale(log_scale), is_integer(integer) {}
};

/**
 * @struct OptimizationConfig
 * @brief Configuration for hyperparameter optimization
 */
struct OptimizationConfig {
    // Search strategy
    enum Strategy { GRID_SEARCH, RANDOM_SEARCH, BAYESIAN, OPTUNA } strategy = RANDOM_SEARCH;
    int max_trials = 100;
    int max_concurrent_trials = 4;
    double timeout_hours = 24.0;
    
    // Search space
    std::vector<HyperparameterRange> parameter_ranges;
    
    // Evaluation criteria
    std::string primary_metric = "val_accuracy";
    bool maximize_metric = true;
    double min_improvement = 0.001;
    int early_stopping_rounds = 20;
    
    // Cross-validation
    int cv_folds = 5;
    bool use_time_series_split = true;
    
    /**
     * @brief Add hyperparameter range
     * @param range Hyperparameter range to add
     */
    void addParameterRange(const HyperparameterRange& range) {
        parameter_ranges.push_back(range);
    }
    
    /**
     * @brief Validate optimization configuration
     * @throws std::invalid_argument if configuration is invalid
     */
    void validate() const;
};

/**
 * @struct TrainingSession
 * @brief Information about a training session
 */
struct TrainingSession {
    std::string session_id;
    std::string model_name;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    TrainingConfig config;
    std::vector<ModelMetrics> epoch_metrics;
    ModelMetrics final_metrics;
    bool completed = false;
    std::string error_message;
    
    /**
     * @brief Calculate total training time
     * @return Training duration
     */
    std::chrono::duration<double> getTrainingDuration() const {
        return end_time - start_time;
    }
    
    /**
     * @brief Convert to JSON string
     * @return JSON representation
     */
    std::string toJson() const;
};

/**
 * @class TrainingManager
 * @brief Manages model training, hyperparameter optimization, and experiments
 */
class TrainingManager {
public:
    /**
     * @brief Constructor
     * @param experiment_dir Directory for storing training experiments
     */
    explicit TrainingManager(const std::string& experiment_dir);
    
    /**
     * @brief Destructor
     */
    ~TrainingManager();
    
    /**
     * @brief Train a single model
     * @param model Model to train
     * @param train_data Training dataset
     * @param val_data Validation dataset (optional)
     * @param config Training configuration
     * @return Training session information
     */
    std::future<TrainingSession> trainModel(
        std::shared_ptr<BaseModel> model,
        const torch::data::DataLoader<>& train_data,
        const torch::data::DataLoader<>* val_data,
        const TrainingConfig& config
    );
    
    /**
     * @brief Optimize hyperparameters for a model
     * @param model_params Base model parameters
     * @param train_data Training dataset
     * @param val_data Validation dataset
     * @param opt_config Optimization configuration
     * @return Future containing best hyperparameters
     */
    std::future<std::map<std::string, double>> optimizeHyperparameters(
        const ModelCreationParams& model_params,
        const torch::data::DataLoader<>& train_data,
        const torch::data::DataLoader<>& val_data,
        const OptimizationConfig& opt_config
    );
    
    /**
     * @brief Run cross-validation experiment
     * @param model_params Model parameters
     * @param dataset Complete dataset
     * @param cv_folds Number of CV folds
     * @param use_time_series_split Use time series split
     * @return Vector of CV metrics
     */
    std::future<std::vector<ModelMetrics>> runCrossValidation(
        const ModelCreationParams& model_params,
        const torch::data::DataLoader<>& dataset,
        int cv_folds = 5,
        bool use_time_series_split = true
    );
    
    /**
     * @brief Compare multiple model architectures
     * @param model_configs Vector of model configurations
     * @param train_data Training dataset
     * @param val_data Validation dataset
     * @return Map of model name to performance metrics
     */
    std::future<std::map<std::string, ModelMetrics>> compareArchitectures(
        const std::vector<ModelCreationParams>& model_configs,
        const torch::data::DataLoader<>& train_data,
        const torch::data::DataLoader<>& val_data
    );
    
    /**
     * @brief Get training session by ID
     * @param session_id Session ID
     * @return Training session information
     */
    std::optional<TrainingSession> getTrainingSession(const std::string& session_id);
    
    /**
     * @brief List all training sessions
     * @param model_name Optional model name filter
     * @return Vector of training sessions
     */
    std::vector<TrainingSession> listTrainingSessions(
        const std::string& model_name = ""
    ) const;
    
    /**
     * @brief Cancel running training session
     * @param session_id Session ID to cancel
     * @return True if cancellation successful
     */
    bool cancelTraining(const std::string& session_id);
    
    /**
     * @brief Get training progress for active sessions
     * @return Map of session ID to progress percentage
     */
    std::map<std::string, double> getTrainingProgress() const;
    
    /**
     * @brief Export training results to CSV/JSON
     * @param filepath Output file path
     * @param format Export format ("csv" or "json")
     * @return True if export successful
     */
    bool exportResults(const std::string& filepath, const std::string& format);
    
    /**
     * @brief Set progress callback for training updates
     * @param callback Progress callback function
     */
    void setProgressCallback(std::function<void(const std::string&, double)> callback) {
        progress_callback_ = callback;
    }
    
private:
    /**
     * @brief Generate unique session ID
     * @return Session ID string
     */
    std::string generateSessionId();
    
    /**
     * @brief Save training session
     * @param session Training session to save
     */
    void saveTrainingSession(const TrainingSession& session);
    
    /**
     * @brief Load training sessions from disk
     */
    void loadTrainingSessions();
    
    /**
     * @brief Generate hyperparameter combinations
     * @param opt_config Optimization configuration
     * @return Vector of hyperparameter combinations
     */
    std::vector<std::map<std::string, double>> generateHyperparameterCombinations(
        const OptimizationConfig& opt_config
    );
    
    /**
     * @brief Evaluate single hyperparameter combination
     * @param model_params Base model parameters
     * @param hyperparams Hyperparameter values
     * @param train_data Training data
     * @param val_data Validation data
     * @return Evaluation metrics
     */
    ModelMetrics evaluateHyperparameters(
        const ModelCreationParams& model_params,
        const std::map<std::string, double>& hyperparams,
        const torch::data::DataLoader<>& train_data,
        const torch::data::DataLoader<>& val_data
    );
    
    std::string experiment_dir_;
    
    // Active training sessions
    mutable std::mutex sessions_mutex_;
    std::map<std::string, TrainingSession> training_sessions_;
    std::map<std::string, std::future<TrainingSession>> active_trainings_;
    
    // Progress tracking
    std::function<void(const std::string&, double)> progress_callback_;
    
    // Thread pool for concurrent training
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_{false};
};

/**
 * @brief Convert OptimizerType to string
 * @param type Optimizer type
 * @return String representation
 */
std::string optimizerTypeToString(OptimizerType type);

/**
 * @brief Convert SchedulerType to string
 * @param type Scheduler type
 * @return String representation
 */
std::string schedulerTypeToString(SchedulerType type);

/**
 * @brief Convert LossFunctionType to string
 * @param type Loss function type
 * @return String representation
 */
std::string lossFunctionTypeToString(LossFunctionType type);

/**
 * @brief Create optimizer from configuration
 * @param parameters Model parameters
 * @param type Optimizer type
 * @param learning_rate Learning rate
 * @param weight_decay Weight decay
 * @return Unique pointer to optimizer
 */
std::unique_ptr<torch::optim::Optimizer> createOptimizer(
    std::vector<torch::Tensor> parameters,
    OptimizerType type,
    double learning_rate,
    double weight_decay = 0.0
);

/**
 * @brief Create loss function from type
 * @param type Loss function type
 * @return Loss function module
 */
std::unique_ptr<torch::nn::Module> createLossFunction(LossFunctionType type);

} // namespace Models
} // namespace ArchNeuronX