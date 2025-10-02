/**
 * @file mlp_model.hpp
 * @brief Multi-Layer Perceptron implementation for ArchNeuronX trading
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include <memory>
#include "core/base_model.hpp"

namespace ArchNeuronX {
namespace Models {

/**
 * @struct MLPConfig
 * @brief Configuration for MLP model
 */
struct MLPConfig {
    std::vector<int64_t> hidden_layers = {128, 64, 32};  ///< Hidden layer sizes
    double dropout_rate = 0.2;                          ///< Dropout probability
    bool use_batch_norm = true;                         ///< Use batch normalization
    std::string activation = "relu";                    ///< Activation function
    bool use_residual = false;                          ///< Use residual connections
    double weight_init_std = 0.1;                       ///< Weight initialization std
    bool use_layer_norm = false;                        ///< Use layer normalization
    
    /**
     * @brief Validate MLP configuration
     * @throws std::invalid_argument if configuration is invalid
     */
    void validate() const;
};

/**
 * @class MLPModel
 * @brief Multi-Layer Perceptron for financial time series prediction
 */
class MLPModel : public BaseModel {
public:
    /**
     * @brief Constructor
     * @param name Model name
     * @param config MLP configuration
     */
    MLPModel(const std::string& name, const MLPConfig& config);
    
    /**
     * @brief Initialize MLP architecture
     * @param input_size Size of input features
     * @param output_size Size of output predictions
     */
    void initialize(int64_t input_size, int64_t output_size) override;
    
    /**
     * @brief Forward pass through the MLP
     * @param input Input tensor [batch_size, input_size]
     * @return Output tensor [batch_size, output_size]
     */
    torch::Tensor forward(torch::Tensor input) override;
    
    /**
     * @brief Get model architecture as string
     * @return Formatted architecture description
     */
    std::string getModelInfo() const override;
    
    /**
     * @brief Get number of trainable parameters
     * @return Total number of parameters
     */
    int64_t getParameterCount() const;
    
    /**
     * @brief Set training/evaluation mode
     * @param training True for training mode, false for evaluation
     */
    void setTraining(bool training);
    
    /**
     * @brief Get layer activations for analysis
     * @param input Input tensor
     * @return Vector of activation tensors from each layer
     */
    std::vector<torch::Tensor> getLayerActivations(torch::Tensor input);
    
    /**
     * @brief Apply gradient clipping
     * @param max_norm Maximum gradient norm
     */
    void clipGradients(double max_norm = 1.0);
    
    // Getters
    const MLPConfig& getConfig() const { return config_; }
    
private:
    /**
     * @brief Create activation function
     * @param activation_name Name of activation function
     * @return Activation module
     */
    torch::nn::AnyModule createActivation(const std::string& activation_name);
    
    /**
     * @brief Initialize layer weights
     * @param layer Linear layer to initialize
     */
    void initializeWeights(torch::nn::Linear& layer);
    
    MLPConfig config_;
    
    // Network layers
    torch::nn::ModuleList layers_;
    torch::nn::ModuleList batch_norms_;
    torch::nn::ModuleList dropouts_;
    torch::nn::ModuleList layer_norms_;
    
    // Output layer
    torch::nn::Linear output_layer_{nullptr};
    torch::nn::Dropout output_dropout_{nullptr};
    
    // Activation function
    torch::nn::AnyModule activation_fn_;
    
    // Model state
    bool is_training_ = true;
    
    // Layer sizes for residual connections
    std::vector<int64_t> layer_sizes_;
};

/**
 * @class MLPEnsemble
 * @brief Ensemble of MLP models for improved prediction accuracy
 */
class MLPEnsemble : public BaseModel {
public:
    /**
     * @brief Constructor
     * @param name Ensemble name
     * @param num_models Number of models in ensemble
     * @param config Configuration for each MLP
     */
    MLPEnsemble(const std::string& name, int num_models, const MLPConfig& config);
    
    /**
     * @brief Initialize ensemble
     * @param input_size Size of input features
     * @param output_size Size of output predictions
     */
    void initialize(int64_t input_size, int64_t output_size) override;
    
    /**
     * @brief Forward pass through ensemble (average predictions)
     * @param input Input tensor
     * @return Ensemble prediction tensor
     */
    torch::Tensor forward(torch::Tensor input) override;
    
    /**
     * @brief Get individual model predictions
     * @param input Input tensor
     * @return Vector of predictions from each model
     */
    std::vector<torch::Tensor> getIndividualPredictions(torch::Tensor input);
    
    /**
     * @brief Get prediction variance (uncertainty)
     * @param input Input tensor
     * @return Prediction variance tensor
     */
    torch::Tensor getPredictionVariance(torch::Tensor input);
    
    /**
     * @brief Train ensemble with different data subsets
     * @param train_data Training dataset
     * @param val_data Validation dataset
     * @param config Training configuration
     * @return Ensemble training metrics
     */
    ModelMetrics train(const torch::data::DataLoader<>& train_data,
                      const torch::data::DataLoader<>* val_data,
                      const TrainingConfig& config) override;
    
    /**
     * @brief Get ensemble information
     * @return Formatted ensemble info
     */
    std::string getModelInfo() const override;
    
    // Getters
    int getNumModels() const { return static_cast<int>(models_.size()); }
    const std::vector<std::shared_ptr<MLPModel>>& getModels() const { return models_; }
    
private:
    /**
     * @brief Create bootstrap samples for ensemble training
     * @param train_data Original training data
     * @return Vector of bootstrap datasets
     */
    std::vector<torch::data::DataLoader<>> createBootstrapSamples(
        const torch::data::DataLoader<>& train_data
    );
    
    std::vector<std::shared_ptr<MLPModel>> models_;
    MLPConfig base_config_;
    int num_models_;
    
    // Ensemble aggregation method
    enum class AggregationMethod {
        MEAN,           ///< Simple average
        WEIGHTED_MEAN,  ///< Weighted by model performance
        MEDIAN,         ///< Median prediction
        VOTING          ///< Majority voting (for classification)
    } aggregation_method_ = AggregationMethod::MEAN;
    
    std::vector<double> model_weights_;  ///< Weights for weighted averaging
};

/**
 * @brief Create MLP model factory function
 * @param name Model name
 * @param config MLP configuration
 * @return Unique pointer to MLP model
 */
std::unique_ptr<MLPModel> createMLPModel(
    const std::string& name,
    const MLPConfig& config
);

/**
 * @brief Create MLP ensemble factory function
 * @param name Ensemble name
 * @param num_models Number of models
 * @param config Base MLP configuration
 * @return Unique pointer to MLP ensemble
 */
std::unique_ptr<MLPEnsemble> createMLPEnsemble(
    const std::string& name,
    int num_models,
    const MLPConfig& config
);

} // namespace Models
} // namespace ArchNeuronX