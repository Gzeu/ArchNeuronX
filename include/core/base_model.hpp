/**
 * @file base_model.hpp
 * @brief Base neural network model interface for ArchNeuronX
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include <torch/torch.h>
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <chrono>
#include "core/logger.hpp"

namespace ArchNeuronX {
namespace Models {

/**
 * @enum ModelType
 * @brief Defines types of neural network models
 */
enum class ModelType {
    MLP,        ///< Multi-Layer Perceptron
    CNN,        ///< Convolutional Neural Network
    HYBRID      ///< Hybrid MLP-CNN model
};

/**
 * @enum TrainingPhase
 * @brief Defines current training phase
 */
enum class TrainingPhase {
    TRAIN,      ///< Training phase
    VALIDATION, ///< Validation phase
    TEST        ///< Testing phase
};

/**
 * @struct ModelMetrics
 * @brief Container for model performance metrics
 */
struct ModelMetrics {
    double loss = 0.0;
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1_score = 0.0;
    double sharpe_ratio = 0.0;
    double max_drawdown = 0.0;
    int64_t samples_processed = 0;
    std::chrono::milliseconds training_time{0};
    
    /**
     * @brief Convert metrics to string representation
     * @return Formatted string with all metrics
     */
    std::string toString() const;
};

/**
 * @struct TrainingConfig
 * @brief Configuration parameters for model training
 */
struct TrainingConfig {
    int epochs = 100;
    int batch_size = 32;
    double learning_rate = 0.001;
    double weight_decay = 0.0001;
    int patience = 10;              // Early stopping patience
    double min_delta = 0.001;       // Minimum improvement for early stopping
    bool use_gpu = true;
    int save_checkpoint_every = 10; // Save checkpoint every N epochs
    std::string optimizer = "adam"; // adam, sgd, rmsprop
    
    /**
     * @brief Validate training configuration
     * @throws std::invalid_argument if configuration is invalid
     */
    void validate() const;
};

/**
 * @class BaseModel
 * @brief Abstract base class for all neural network models
 */
class BaseModel : public torch::nn::Module {
public:
    /**
     * @brief Constructor
     * @param name Model name for identification
     * @param type Model type (MLP, CNN, HYBRID)
     */
    BaseModel(const std::string& name, ModelType type);
    
    /**
     * @brief Virtual destructor
     */
    virtual ~BaseModel() = default;
    
    /**
     * @brief Initialize model architecture
     * @param input_size Size of input features
     * @param output_size Size of output predictions
     */
    virtual void initialize(int64_t input_size, int64_t output_size) = 0;
    
    /**
     * @brief Forward pass through the network
     * @param input Input tensor
     * @return Output tensor
     */
    virtual torch::Tensor forward(torch::Tensor input) = 0;
    
    /**
     * @brief Train the model with given data
     * @param train_data Training dataset
     * @param val_data Validation dataset (optional)
     * @param config Training configuration
     * @return Training metrics
     */
    virtual ModelMetrics train(const torch::data::DataLoader<>& train_data,
                              const torch::data::DataLoader<>* val_data,
                              const TrainingConfig& config);
    
    /**
     * @brief Evaluate model on test data
     * @param test_data Test dataset
     * @return Evaluation metrics
     */
    virtual ModelMetrics evaluate(const torch::data::DataLoader<>& test_data);
    
    /**
     * @brief Make predictions on input data
     * @param input Input tensor
     * @return Prediction tensor
     */
    virtual torch::Tensor predict(const torch::Tensor& input);
    
    /**
     * @brief Save model to file
     * @param filepath Path to save model
     * @param save_optimizer Whether to save optimizer state
     */
    virtual void save(const std::string& filepath, bool save_optimizer = false);
    
    /**
     * @brief Load model from file
     * @param filepath Path to load model from
     * @param load_optimizer Whether to load optimizer state
     */
    virtual void load(const std::string& filepath, bool load_optimizer = false);
    
    /**
     * @brief Get model information as string
     * @return Formatted model information
     */
    virtual std::string getModelInfo() const;
    
    // Getters
    const std::string& getName() const { return name_; }
    ModelType getType() const { return type_; }
    bool isInitialized() const { return initialized_; }
    bool isTraining() const { return training_; }
    const ModelMetrics& getLastMetrics() const { return last_metrics_; }
    torch::Device getDevice() const { return device_; }
    
    // Setters
    void setDevice(const torch::Device& device);
    void setTrainingPhase(TrainingPhase phase);
    
protected:
    /**
     * @brief Calculate loss for given predictions and targets
     * @param predictions Model predictions
     * @param targets True targets
     * @return Loss tensor
     */
    virtual torch::Tensor calculateLoss(const torch::Tensor& predictions,
                                       const torch::Tensor& targets);
    
    /**
     * @brief Calculate accuracy for given predictions and targets
     * @param predictions Model predictions
     * @param targets True targets
     * @return Accuracy value
     */
    virtual double calculateAccuracy(const torch::Tensor& predictions,
                                   const torch::Tensor& targets);
    
    /**
     * @brief Calculate trading-specific metrics
     * @param predictions Model predictions
     * @param targets True targets
     * @param prices Price data for financial calculations
     * @return Updated metrics structure
     */
    virtual ModelMetrics calculateTradingMetrics(const torch::Tensor& predictions,
                                               const torch::Tensor& targets,
                                               const torch::Tensor& prices);
    
    /**
     * @brief Setup optimizer based on configuration
     * @param config Training configuration
     */
    virtual void setupOptimizer(const TrainingConfig& config);
    
    /**
     * @brief Setup loss function
     */
    virtual void setupLossFunction();
    
    /**
     * @brief Early stopping check
     * @param current_loss Current validation loss
     * @param patience Number of epochs to wait
     * @param min_delta Minimum improvement required
     * @return True if training should stop
     */
    bool shouldStopEarly(double current_loss, int patience, double min_delta);
    
    // Member variables
    std::string name_;
    ModelType type_;
    bool initialized_ = false;
    bool training_ = false;
    TrainingPhase current_phase_ = TrainingPhase::TRAIN;
    
    torch::Device device_ = torch::kCPU;
    std::unique_ptr<torch::optim::Optimizer> optimizer_;
    std::unique_ptr<torch::nn::Loss<torch::Tensor>> loss_function_;
    
    ModelMetrics last_metrics_;
    
    // Early stopping variables
    double best_val_loss_ = std::numeric_limits<double>::max();
    int epochs_without_improvement_ = 0;
    
    int64_t input_size_ = 0;
    int64_t output_size_ = 0;
};

/**
 * @brief Convert ModelType to string
 * @param type Model type
 * @return String representation
 */
std::string modelTypeToString(ModelType type);

/**
 * @brief Convert string to ModelType
 * @param type_str String representation
 * @return Model type
 */
ModelType stringToModelType(const std::string& type_str);

} // namespace Models
} // namespace ArchNeuronX