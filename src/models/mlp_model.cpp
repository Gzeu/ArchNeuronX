// ============================================================
// ArchNeuronX v2 - Multi-Layer Perceptron Implementation
// Optimized for financial time series prediction
// ============================================================
#include "models/neural_networks.hpp"
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace ArchNeuronX {
namespace Models {

MLPNetwork::MLPNetwork(const Config& config) : config_(config) {
    buildNetwork();
}

torch::Tensor MLPNetwork::forward(torch::Tensor x) {
    // Flatten input if needed
    if (x.dim() == 3) {
        x = x.view({x.size(0), -1});
    }
    
    torch::Tensor prev_output;
    
    for (size_t i = 0; i < config_.hidden_sizes.size(); ++i) {
        auto layer = layers_[i]->as<torch::nn::Linear>();
        x = layer->forward(x);
        
        if (config_.use_batch_norm && i < batch_norms_.size()) {
            auto batch_norm = batch_norms_[i]->as<torch::nn::BatchNorm1d>();
            x = batch_norm->forward(x);
        }
        
        x = applyActivation(x, config_.activation);
        x = dropout_->forward(x);
        
        if (config_.use_residual && i > 0 && prev_output.defined()) {
            if (x.sizes() == prev_output.sizes()) {
                x = x + prev_output;
            }
        }
        
        prev_output = x;
    }
    
    // Final output layer
    auto final_layer = layers_[config_.hidden_sizes.size()]->as<torch::nn::Linear>();
    x = final_layer->forward(x);
    
    return x;
}

void MLPNetwork::buildNetwork() {
    // Create hidden layers
    int prev_size = config_.input_size;
    
    for (int hidden_size : config_.hidden_sizes) {
        auto linear = torch::nn::Linear(prev_size, hidden_size);
        layers_->push_back(linear);
        
        if (config_.use_batch_norm) {
            auto batch_norm = torch::nn::BatchNorm1d(hidden_size);
            batch_norms_->push_back(batch_norm);
        }
        
        prev_size = hidden_size;
    }
    
    // Create output layer
    auto output_layer = torch::nn::Linear(prev_size, config_.output_size);
    layers_->push_back(output_layer);
    
    // Create dropout layer
    dropout_ = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(config_.dropout_rate)));
    
    // Register all layers
    for (size_t i = 0; i < layers_->size(); ++i) {
        layers_[i] = register_module("layer_" + std::to_string(i), layers_[i]);
    }
    
    for (size_t i = 0; i < batch_norms_->size(); ++i) {
        batch_norms_[i] = register_module("batch_norm_" + std::to_string(i), batch_norms_[i]);
    }
}

torch::Tensor MLPNetwork::applyActivation(torch::Tensor x, const std::string& activation) {
    if (activation == "relu") {
        return torch::relu(x);
    } else if (activation == "tanh") {
        return torch::tanh(x);
    } else if (activation == "sigmoid") {
        return torch::sigmoid(x);
    } else if (activation == "gelu") {
        return torch::gelu(x);
    } else if (activation == "leaky_relu") {
        return torch::leaky_relu(x);
    } else {
        // Default to ReLU
        return torch::relu(x);
    }
}

void MLPNetwork::saveModel(const std::string& path) {
    try {
        // Save model state
        torch::save(*this, path + "_model.pt");
        
        // Save configuration
        json config_json = {
            {"input_size", config_.input_size},
            {"hidden_sizes", config_.hidden_sizes},
            {"output_size", config_.output_size},
            {"dropout_rate", config_.dropout_rate},
            {"use_batch_norm", config_.use_batch_norm},
            {"activation", config_.activation},
            {"use_residual", config_.use_residual}
        };
        
        std::ofstream config_file(path + "_config.json");
        config_file << config_json.dump(4);
        config_file.close();
        
        std::cout << "MLP model saved to: " << path << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving MLP model: " << e.what() << std::endl;
    }
}

void MLPNetwork::loadModel(const std::string& path) {
    try {
        // Load configuration
        std::ifstream config_file(path + "_config.json");
        if (config_file.is_open()) {
            json config_json;
            config_file >> config_json;
            config_file.close();
            
            config_.input_size = config_json.value("input_size", 50);
            config_.hidden_sizes = config_json.value("hidden_sizes", std::vector<int>{128, 64, 32});
            config_.output_size = config_json.value("output_size", 3);
            config_.dropout_rate = config_json.value("dropout_rate", 0.2);
            config_.use_batch_norm = config_json.value("use_batch_norm", true);
            config_.activation = config_json.value("activation", "relu");
            config_.use_residual = config_json.value("use_residual", false);
        }
        
        // Rebuild network with loaded configuration
        buildNetwork();
        
        // Load model state
        torch::load(*this, path + "_model.pt");
        
        std::cout << "MLP model loaded from: " << path << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading MLP model: " << e.what() << std::endl;
    }
}

// CNN Network Implementation
CNNNetwork::CNNNetwork(const Config& config) : config_(config) {
    buildNetwork();
}

torch::Tensor CNNNetwork::forward(torch::Tensor x) {
    // Ensure input is in correct format [batch, channels, sequence]
    if (x.dim() == 3 && x.size(2) == config_.sequence_length) {
        x = x.transpose(1, 2); // [batch, seq_len, features] -> [batch, features, seq_len]
    }
    
    // Convolutional layers
    for (size_t i = 0; i < config_.conv_channels.size(); ++i) {
        auto conv = conv_layers_[i]->as<torch::nn::Conv1d>();
        x = conv->forward(x);
        
        if (config_.use_batch_norm && i < batch_norms_->size()) {
            auto batch_norm = batch_norms_[i]->as<torch::nn::BatchNorm1d>();
            x = batch_norm->forward(x);
        }
        
        x = applyActivation(x, config_.activation);
        
        if (i < config_.conv_channels.size() - 1) {
            x = torch::max_pool1d(x, config_.pool_size);
        }
    }
    
    // Global average pooling
    x = global_pool_->forward(x);
    x = x.view({x.size(0), -1}); // Flatten
    
    // Fully connected layers
    for (size_t i = 0; i < config_.fc_sizes.size(); ++i) {
        auto fc = fc_layers_[i]->as<torch::nn::Linear>();
        x = fc->forward(x);
        x = applyActivation(x, config_.activation);
        x = dropout_->forward(x);
    }
    
    return x;
}

void CNNNetwork::buildNetwork() {
    int in_channels = config_.input_channels;
    int seq_len = config_.sequence_length;
    
    // Build convolutional layers
    for (size_t i = 0; i < config_.conv_channels.size(); ++i) {
        int out_channels = config_.conv_channels[i];
        int kernel_size = config_.kernel_sizes[i];
        int stride = config_.strides[i];
        
        auto conv = torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(kernel_size / 2));
        
        conv_layers_->push_back(conv);
        
        if (config_.use_batch_norm) {
            auto batch_norm = torch::nn::BatchNorm1d(out_channels);
            batch_norms_->push_back(batch_norm);
        }
        
        in_channels = out_channels;
        
        // Update sequence length after convolution and pooling
        seq_len = (seq_len + 2 * (kernel_size / 2) - kernel_size) / stride + 1;
        if (i < config_.conv_channels.size() - 1) {
            seq_len = (seq_len - config_.pool_size) / config_.pool_size + 1;
        }
    }
    
    // Global average pooling
    global_pool_ = register_module("global_pool", torch::nn::AdaptiveAvgPool1d(torch::nn::AdaptiveAvgPool1dOptions(1)));
    
    // Calculate input size for fully connected layers
    int fc_input_size = in_channels;
    
    // Build fully connected layers
    int prev_size = fc_input_size;
    for (int fc_size : config_.fc_sizes) {
        auto fc = torch::nn::Linear(prev_size, fc_size);
        fc_layers_->push_back(fc);
        prev_size = fc_size;
    }
    
    // Final output layer
    auto output_fc = torch::nn::Linear(prev_size, config_.output_size);
    fc_layers_->push_back(output_fc);
    
    // Create dropout
    dropout_ = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(config_.dropout_rate)));
    
    // Register all modules
    for (size_t i = 0; i < conv_layers_->size(); ++i) {
        conv_layers_[i] = register_module("conv_" + std::to_string(i), conv_layers_[i]);
    }
    
    for (size_t i = 0; i < batch_norms_->size(); ++i) {
        batch_norms_[i] = register_module("conv_batch_norm_" + std::to_string(i), batch_norms_[i]);
    }
    
    for (size_t i = 0; i < fc_layers_->size(); ++i) {
        fc_layers_[i] = register_module("fc_" + std::to_string(i), fc_layers_[i]);
    }
}

int CNNNetwork::calculateFcInputSize() {
    // This would calculate the flattened size after conv layers
    // For simplicity, returning a calculated value
    return config_.conv_channels.back() * (config_.sequence_length / config_.pool_size);
}

torch::Tensor CNNNetwork::applyActivation(torch::Tensor x, const std::string& activation) {
    if (activation == "relu") {
        return torch::relu(x);
    } else if (activation == "tanh") {
        return torch::tanh(x);
    } else if (activation == "sigmoid") {
        return torch::sigmoid(x);
    } else if (activation == "gelu") {
        return torch::gelu(x);
    } else {
        return torch::relu(x);
    }
}

void CNNNetwork::saveModel(const std::string& path) {
    try {
        torch::save(*this, path + "_model.pt");
        
        json config_json = {
            {"input_channels", config_.input_channels},
            {"sequence_length", config_.sequence_length},
            {"conv_channels", config_.conv_channels},
            {"kernel_sizes", config_.kernel_sizes},
            {"strides", config_.strides},
            {"fc_sizes", config_.fc_sizes},
            {"output_size", config_.output_size},
            {"dropout_rate", config_.dropout_rate},
            {"use_batch_norm", config_.use_batch_norm},
            {"activation", config_.activation},
            {"pool_size", config_.pool_size}
        };
        
        std::ofstream config_file(path + "_config.json");
        config_file << config_json.dump(4);
        config_file.close();
        
        std::cout << "CNN model saved to: " << path << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving CNN model: " << e.what() << std::endl;
    }
}

void CNNNetwork::loadModel(const std::string& path) {
    try {
        std::ifstream config_file(path + "_config.json");
        if (config_file.is_open()) {
            json config_json;
            config_file >> config_json;
            config_file.close();
            
            config_.input_channels = config_json.value("input_channels", 1);
            config_.sequence_length = config_json.value("sequence_length", 50);
            config_.conv_channels = config_json.value("conv_channels", std::vector<int>{32, 64, 128});
            config_.kernel_sizes = config_json.value("kernel_sizes", std::vector<int>{3, 3, 3});
            config_.strides = config_json.value("strides", std::vector<int>{1, 1, 1});
            config_.fc_sizes = config_json.value("fc_sizes", std::vector<int>{256, 128});
            config_.output_size = config_json.value("output_size", 3);
            config_.dropout_rate = config_json.value("dropout_rate", 0.3);
            config_.use_batch_norm = config_json.value("use_batch_norm", true);
            config_.activation = config_json.value("activation", "relu");
            config_.pool_size = config_json.value("pool_size", 2);
        }
        
        buildNetwork();
        torch::load(*this, path + "_model.pt");
        
        std::cout << "CNN model loaded from: " << path << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading CNN model: " << e.what() << std::endl;
    }
}

} // namespace Models
} // namespace ArchNeuronX
