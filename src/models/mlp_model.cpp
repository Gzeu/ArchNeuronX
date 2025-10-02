/**
 * @file mlp_model.cpp
 * @brief Implementation of Multi-Layer Perceptron for trading
 * @author George Pricop
 * @date 2025-10-02
 */

#include "models/mlp_model.hpp"
#include <stdexcept>
#include <sstream>
#include <random>

namespace ArchNeuronX {
namespace Models {

void MLPConfig::validate() const {
    if (hidden_layers.empty()) {
        throw std::invalid_argument("MLP must have at least one hidden layer");
    }
    
    for (auto layer_size : hidden_layers) {
        if (layer_size <= 0) {
            throw std::invalid_argument("Hidden layer sizes must be positive");
        }
    }
    
    if (dropout_rate < 0.0 || dropout_rate >= 1.0) {
        throw std::invalid_argument("Dropout rate must be in range [0.0, 1.0)");
    }
    
    if (weight_init_std <= 0.0) {
        throw std::invalid_argument("Weight initialization std must be positive");
    }
    
    // Validate activation function
    std::vector<std::string> valid_activations = {
        "relu", "leaky_relu", "tanh", "sigmoid", "gelu", "swish"
    };
    
    if (std::find(valid_activations.begin(), valid_activations.end(), activation) == 
        valid_activations.end()) {
        throw std::invalid_argument("Invalid activation function: " + activation);
    }
}

MLPModel::MLPModel(const std::string& name, const MLPConfig& config)
    : BaseModel(name, ModelType::MLP), config_(config) {
    config_.validate();
    LOG_INFO("Created MLP model: " << name);
}

void MLPModel::initialize(int64_t input_size, int64_t output_size) {
    if (input_size <= 0 || output_size <= 0) {
        throw std::invalid_argument("Input and output sizes must be positive");
    }
    
    input_size_ = input_size;
    output_size_ = output_size;
    
    // Create activation function
    activation_fn_ = createActivation(config_.activation);
    
    // Build layer sizes vector
    layer_sizes_.clear();
    layer_sizes_.push_back(input_size);
    for (auto hidden_size : config_.hidden_layers) {
        layer_sizes_.push_back(hidden_size);
    }
    
    // Create hidden layers
    layers_ = torch::nn::ModuleList();
    batch_norms_ = torch::nn::ModuleList();
    dropouts_ = torch::nn::ModuleList();
    layer_norms_ = torch::nn::ModuleList();
    
    for (size_t i = 0; i < config_.hidden_layers.size(); ++i) {
        int64_t in_features = (i == 0) ? input_size : config_.hidden_layers[i-1];
        int64_t out_features = config_.hidden_layers[i];
        
        // Linear layer
        auto linear = torch::nn::Linear(in_features, out_features);
        initializeWeights(linear);
        layers_->push_back(linear);
        
        // Batch normalization
        if (config_.use_batch_norm) {
            batch_norms_->push_back(torch::nn::BatchNorm1d(out_features));
        }
        
        // Layer normalization
        if (config_.use_layer_norm) {
            layer_norms_->push_back(torch::nn::LayerNorm(
                torch::nn::LayerNormOptions({out_features})));
        }
        
        // Dropout
        if (config_.dropout_rate > 0.0) {
            dropouts_->push_back(torch::nn::Dropout(config_.dropout_rate));
        }
    }
    
    // Output layer
    int64_t last_hidden_size = config_.hidden_layers.back();
    output_layer_ = torch::nn::Linear(last_hidden_size, output_size);
    initializeWeights(output_layer_);
    
    // Output dropout
    if (config_.dropout_rate > 0.0) {
        output_dropout_ = torch::nn::Dropout(config_.dropout_rate * 0.5); // Lower dropout for output
    }
    
    // Register modules
    register_module("layers", layers_);
    register_module("batch_norms", batch_norms_);
    register_module("dropouts", dropouts_);
    register_module("layer_norms", layer_norms_);
    register_module("output_layer", output_layer_);
    if (output_dropout_) {
        register_module("output_dropout", output_dropout_);
    }
    
    initialized_ = true;
    LOG_INFO("MLP model initialized: " << getModelInfo());
}

torch::Tensor MLPModel::forward(torch::Tensor input) {
    if (!initialized_) {
        throw std::runtime_error("Model not initialized");
    }
    
    torch::Tensor x = input;
    
    // Forward through hidden layers
    for (size_t i = 0; i < layers_->size(); ++i) {
        // Linear transformation
        x = layers_[i]->as<torch::nn::Linear>()->forward(x);
        
        // Batch normalization
        if (config_.use_batch_norm && i < batch_norms_->size()) {
            x = batch_norms_[i]->as<torch::nn::BatchNorm1d>()->forward(x);
        }
        
        // Layer normalization
        if (config_.use_layer_norm && i < layer_norms_->size()) {
            x = layer_norms_[i]->as<torch::nn::LayerNorm>()->forward(x);
        }
        
        // Activation function
        x = activation_fn_.forward(x);
        
        // Dropout
        if (config_.dropout_rate > 0.0 && i < dropouts_->size()) {
            x = dropouts_[i]->as<torch::nn::Dropout>()->forward(x);
        }
        
        // Residual connection (if enabled and dimensions match)
        if (config_.use_residual && i > 0) {
            int64_t current_size = x.size(-1);
            int64_t prev_size = layer_sizes_[i];
            
            if (current_size == prev_size && i < layers_->size()) {
                // Add residual connection from previous layer
                // This is simplified - in practice, you'd need to store intermediate values
                // x = x + residual_connection;
            }
        }
    }
    
    // Output layer
    if (output_dropout_) {
        x = output_dropout_->forward(x);
    }
    
    x = output_layer_->forward(x);
    
    return x;
}

std::string MLPModel::getModelInfo() const {
    std::ostringstream oss;
    oss << "MLP Model: " << getName() << "\n";
    oss << "  Input Size: " << input_size_ << "\n";
    oss << "  Hidden Layers: [";
    for (size_t i = 0; i < config_.hidden_layers.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << config_.hidden_layers[i];
    }
    oss << "]\n";
    oss << "  Output Size: " << output_size_ << "\n";
    oss << "  Activation: " << config_.activation << "\n";
    oss << "  Dropout Rate: " << config_.dropout_rate << "\n";
    oss << "  Batch Norm: " << (config_.use_batch_norm ? "Yes" : "No") << "\n";
    oss << "  Residual: " << (config_.use_residual ? "Yes" : "No") << "\n";
    oss << "  Parameters: " << getParameterCount();
    
    return oss.str();
}

int64_t MLPModel::getParameterCount() const {
    int64_t total_params = 0;
    
    for (const auto& param : parameters()) {
        int64_t param_count = 1;
        for (int i = 0; i < param.dim(); ++i) {
            param_count *= param.size(i);
        }
        total_params += param_count;
    }
    
    return total_params;
}

void MLPModel::setTraining(bool training) {
    is_training_ = training;
    train(training);
}

std::vector<torch::Tensor> MLPModel::getLayerActivations(torch::Tensor input) {
    std::vector<torch::Tensor> activations;
    torch::Tensor x = input;
    
    // Store input
    activations.push_back(x.clone());
    
    // Forward through layers and store activations
    for (size_t i = 0; i < layers_->size(); ++i) {
        x = layers_[i]->as<torch::nn::Linear>()->forward(x);
        
        if (config_.use_batch_norm && i < batch_norms_->size()) {
            x = batch_norms_[i]->as<torch::nn::BatchNorm1d>()->forward(x);
        }
        
        x = activation_fn_.forward(x);
        activations.push_back(x.clone());
        
        if (config_.dropout_rate > 0.0 && is_training_ && i < dropouts_->size()) {
            x = dropouts_[i]->as<torch::nn::Dropout>()->forward(x);
        }
    }
    
    return activations;
}

void MLPModel::clipGradients(double max_norm) {
    torch::nn::utils::clip_grad_norm_(parameters(), max_norm);
}

torch::nn::AnyModule MLPModel::createActivation(const std::string& activation_name) {
    if (activation_name == "relu") {
        return torch::nn::AnyModule(torch::nn::ReLU());
    } else if (activation_name == "leaky_relu") {
        return torch::nn::AnyModule(torch::nn::LeakyReLU());
    } else if (activation_name == "tanh") {
        return torch::nn::AnyModule(torch::nn::Tanh());
    } else if (activation_name == "sigmoid") {
        return torch::nn::AnyModule(torch::nn::Sigmoid());
    } else if (activation_name == "gelu") {
        return torch::nn::AnyModule(torch::nn::GELU());
    } else if (activation_name == "swish") {
        // Swish activation: x * sigmoid(x)
        return torch::nn::AnyModule(torch::nn::SiLU());
    } else {
        LOG_WARN("Unknown activation function: " << activation_name << ", using ReLU");
        return torch::nn::AnyModule(torch::nn::ReLU());
    }
}

void MLPModel::initializeWeights(torch::nn::Linear& layer) {
    // Xavier/Glorot initialization
    torch::nn::init::xavier_uniform_(layer->weight, 1.0);
    
    // Zero bias initialization
    if (layer->options.bias()) {
        torch::nn::init::zeros_(layer->bias);
    }
    
    // Apply custom standard deviation if specified
    if (config_.weight_init_std != 0.1) {
        layer->weight.normal_(0.0, config_.weight_init_std);
    }
}

// MLPEnsemble Implementation
MLPEnsemble::MLPEnsemble(const std::string& name, int num_models, const MLPConfig& config)
    : BaseModel(name, ModelType::MLP), base_config_(config), num_models_(num_models) {
    
    if (num_models <= 0) {
        throw std::invalid_argument("Number of models must be positive");
    }
    
    LOG_INFO("Created MLP ensemble: " << name << " with " << num_models << " models");
}

void MLPEnsemble::initialize(int64_t input_size, int64_t output_size) {
    input_size_ = input_size;
    output_size_ = output_size;
    
    // Create individual models
    models_.clear();
    model_weights_.clear();
    
    for (int i = 0; i < num_models_; ++i) {
        std::string model_name = getName() + "_model_" + std::to_string(i);
        
        // Slightly vary configuration for diversity
        MLPConfig varied_config = base_config_;
        
        // Add small random variations to promote diversity
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dropout_dist(varied_config.dropout_rate * 0.8,
                                                     varied_config.dropout_rate * 1.2);
        varied_config.dropout_rate = std::min(0.5, dropout_dist(gen));
        
        auto model = std::make_shared<MLPModel>(model_name, varied_config);
        model->initialize(input_size, output_size);
        
        models_.push_back(model);
        model_weights_.push_back(1.0 / num_models_); // Equal weights initially
    }
    
    initialized_ = true;
    LOG_INFO("MLP ensemble initialized with " << num_models_ << " models");
}

torch::Tensor MLPEnsemble::forward(torch::Tensor input) {
    if (!initialized_) {
        throw std::runtime_error("Ensemble not initialized");
    }
    
    if (models_.empty()) {
        throw std::runtime_error("No models in ensemble");
    }
    
    // Get predictions from all models
    std::vector<torch::Tensor> predictions;
    predictions.reserve(models_.size());
    
    for (auto& model : models_) {
        predictions.push_back(model->forward(input));
    }
    
    // Apply aggregation method
    switch (aggregation_method_) {
        case AggregationMethod::MEAN: {
            torch::Tensor result = predictions[0];
            for (size_t i = 1; i < predictions.size(); ++i) {
                result = result + predictions[i];
            }
            return result / static_cast<double>(predictions.size());
        }
        
        case AggregationMethod::WEIGHTED_MEAN: {
            torch::Tensor result = predictions[0] * model_weights_[0];
            for (size_t i = 1; i < predictions.size(); ++i) {
                result = result + predictions[i] * model_weights_[i];
            }
            return result;
        }
        
        case AggregationMethod::MEDIAN: {
            // Stack predictions and compute median
            torch::Tensor stacked = torch::stack(predictions, 0);
            return std::get<0>(torch::median(stacked, 0));
        }
        
        case AggregationMethod::VOTING: {
            // For classification - majority voting
            torch::Tensor stacked = torch::stack(predictions, 0);
            return torch::mode(torch::argmax(stacked, -1), 0).values.to(torch::kFloat);
        }
        
        default:
            return predictions[0]; // Fallback to first model
    }
}

std::vector<torch::Tensor> MLPEnsemble::getIndividualPredictions(torch::Tensor input) {
    std::vector<torch::Tensor> predictions;
    predictions.reserve(models_.size());
    
    for (auto& model : models_) {
        predictions.push_back(model->forward(input));
    }
    
    return predictions;
}

torch::Tensor MLPEnsemble::getPredictionVariance(torch::Tensor input) {
    auto predictions = getIndividualPredictions(input);
    
    // Calculate mean
    torch::Tensor mean_pred = predictions[0];
    for (size_t i = 1; i < predictions.size(); ++i) {
        mean_pred = mean_pred + predictions[i];
    }
    mean_pred = mean_pred / static_cast<double>(predictions.size());
    
    // Calculate variance
    torch::Tensor variance = torch::zeros_like(mean_pred);
    for (const auto& pred : predictions) {
        torch::Tensor diff = pred - mean_pred;
        variance = variance + (diff * diff);
    }
    variance = variance / static_cast<double>(predictions.size());
    
    return variance;
}

std::string MLPEnsemble::getModelInfo() const {
    std::ostringstream oss;
    oss << "MLP Ensemble: " << getName() << "\n";
    oss << "  Number of Models: " << num_models_ << "\n";
    oss << "  Aggregation: ";
    
    switch (aggregation_method_) {
        case AggregationMethod::MEAN: oss << "Mean"; break;
        case AggregationMethod::WEIGHTED_MEAN: oss << "Weighted Mean"; break;
        case AggregationMethod::MEDIAN: oss << "Median"; break;
        case AggregationMethod::VOTING: oss << "Voting"; break;
    }
    
    oss << "\n";
    oss << "  Base Configuration: \n";
    oss << "    Hidden Layers: [";
    for (size_t i = 0; i < base_config_.hidden_layers.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << base_config_.hidden_layers[i];
    }
    oss << "]\n";
    oss << "    Activation: " << base_config_.activation << "\n";
    oss << "    Dropout: " << base_config_.dropout_rate;
    
    return oss.str();
}

// Factory functions
std::unique_ptr<MLPModel> createMLPModel(const std::string& name, const MLPConfig& config) {
    return std::make_unique<MLPModel>(name, config);
}

std::unique_ptr<MLPEnsemble> createMLPEnsemble(const std::string& name, 
                                               int num_models,
                                               const MLPConfig& config) {
    return std::make_unique<MLPEnsemble>(name, num_models, config);
}

} // namespace Models
} // namespace ArchNeuronX