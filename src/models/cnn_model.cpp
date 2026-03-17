// ============================================================
// ArchNeuronX v2 - Convolutional Neural Network Implementation
// Optimized for financial time series pattern recognition
// ============================================================
#include "models/neural_networks.hpp"
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace ArchNeuronX {
namespace Models {

// Note: CNN implementation is already included in mlp_model.cpp
// This file provides additional CNN-specific optimizations and utilities

class OptimizedCNNNetwork : public CNNNetwork {
public:
    struct OptimizedConfig : CNNNetwork::Config {
        bool use_residual_connections = false;
        bool use_attention = false;
        double attention_dropout = 0.1;
        std::vector<int> dilation_rates = {1, 2, 4}; // For dilated convolutions
    };

    explicit OptimizedCNNNetwork(const OptimizedConfig& config) : CNNNetwork(config), opt_config_(config) {
        if (opt_config_.use_attention) {
            buildAttentionModules();
        }
    }

    torch::Tensor forward(torch::Tensor x) override {
        torch::Tensor prev_features;
        
        for (size_t i = 0; i < opt_config_.conv_channels.size(); ++i) {
            auto conv = conv_layers_[i]->as<torch::nn::Conv1d>();
            x = conv->forward(x);
            
            if (opt_config_.use_batch_norm && i < batch_norms_->size()) {
                auto batch_norm = batch_norms_[i]->as<torch::nn::BatchNorm1d>();
                x = batch_norm->forward(x);
            }
            
            x = applyActivation(x, opt_config_.activation);
            
            // Residual connection
            if (opt_config_.use_residual_connections && prev_features.defined()) {
                if (x.sizes() == prev_features.sizes()) {
                    x = x + prev_features;
                }
            }
            
            prev_features = x;
            
            if (i < opt_config_.conv_channels.size() - 1) {
                x = torch::max_pool1d(x, opt_config_.pool_size);
            }
        }
        
        // Apply attention if enabled
        if (opt_config_.use_attention) {
            x = applyAttention(x);
        }
        
        // Continue with standard CNN forward pass
        x = global_pool_->forward(x);
        x = x.view({x.size(0), -1});
        
        for (size_t i = 0; i < opt_config_.fc_sizes.size(); ++i) {
            auto fc = fc_layers_[i]->as<torch::nn::Linear>();
            x = fc->forward(x);
            x = applyActivation(x, opt_config_.activation);
            x = dropout_->forward(x);
        }
        
        return x;
    }

private:
    OptimizedConfig opt_config_;
    torch::nn::MultiheadAttention attention_{nullptr};
    torch::nn::LayerNorm attention_norm_{nullptr};
    torch::nn::Dropout attention_dropout_{nullptr};

    void buildAttentionModules() {
        int feature_dim = opt_config_.conv_channels.back();
        attention_ = torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(feature_dim, 4)
                .dropout(opt_config_.attention_dropout)
                .batch_first(true)
        );
        attention_norm_ = torch::nn::LayerNorm(torch::nn::LayerNormOptions({feature_dim}));
        attention_dropout_ = torch::nn::Dropout(torch::nn::DropoutOptions(opt_config_.attention_dropout));
    }

    torch::Tensor applyAttention(torch::Tensor x) {
        // Reshape for attention: [batch, seq_len, features]
        auto seq_len = x.size(2);
        x = x.transpose(1, 2); // [batch, channels, seq_len] -> [batch, seq_len, channels]
        
        // Apply self-attention
        auto [attn_output, attn_weights] = attention_->forward(x, x, x);
        x = attention_dropout_->forward(attn_output);
        x = attention_norm_->forward(x + x); // Residual connection
        
        // Reshape back: [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.transpose(1, 2);
        
        return x;
    }
};

// Dilated CNN for capturing long-range dependencies
class DilatedCNNNetwork : public torch::nn::Module {
public:
    struct Config {
        int input_channels = 1;
        int sequence_length = 50;
        std::vector<int> channels = {32, 64, 128, 256};
        std::vector<int> kernel_sizes = {3, 3, 3, 3};
        std::vector<int> dilation_rates = {1, 2, 4, 8};
        int output_size = 3;
        double dropout_rate = 0.3;
        bool use_batch_norm = true;
        std::string activation = "relu";
    };

    explicit DilatedCNNNetwork(const Config& config) : config_(config) {
        buildNetwork();
    }

    torch::Tensor forward(torch::Tensor x) {
        if (x.dim() == 3 && x.size(2) == config_.sequence_length) {
            x = x.transpose(1, 2);
        }

        for (size_t i = 0; i < config_.channels.size(); ++i) {
            auto conv = conv_layers_[i]->as<torch::nn::Conv1d>();
            x = conv->forward(x);

            if (config_.use_batch_norm) {
                auto batch_norm = batch_norms_[i]->as<torch::nn::BatchNorm1d>();
                x = batch_norm->forward(x);
            }

            x = applyActivation(x, config_.activation);
            x = dropout_->forward(x);
        }

        x = torch::adaptive_avg_pool1d(x, 1);
        x = x.view({x.size(0), -1});
        x = output_layer_->forward(x);

        return x;
    }

private:
    Config config_;
    torch::nn::ModuleList conv_layers_{nullptr};
    torch::nn::ModuleList batch_norms_{nullptr};
    torch::nn::Linear output_layer_{nullptr};
    torch::nn::Dropout dropout_{nullptr};

    void buildNetwork() {
        int in_channels = config_.input_channels;

        for (size_t i = 0; i < config_.channels.size(); ++i) {
            int out_channels = config_.channels[i];
            int kernel_size = config_.kernel_sizes[i];
            int dilation = config_.dilation_rates[i];

            auto conv = torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                .dilation(dilation)
                .padding((kernel_size - 1) * dilation / 2));

            conv_layers_->push_back(conv);

            if (config_.use_batch_norm) {
                auto batch_norm = torch::nn::BatchNorm1d(out_channels);
                batch_norms_->push_back(batch_norm);
            }

            in_channels = out_channels;
        }

        output_layer_ = torch::nn::Linear(in_channels, config_.output_size);
        dropout_ = torch::nn::Dropout(torch::nn::DropoutOptions(config_.dropout_rate));

        // Register modules
        for (size_t i = 0; i < conv_layers_->size(); ++i) {
            conv_layers_[i] = register_module("dilated_conv_" + std::to_string(i), conv_layers_[i]);
        }

        for (size_t i = 0; i < batch_norms_->size(); ++i) {
            batch_norms_[i] = register_module("dilated_batch_norm_" + std::to_string(i), batch_norms_[i]);
        }

        output_layer_ = register_module("dilated_output", output_layer_);
        dropout_ = register_module("dilated_dropout", dropout_);
    }

    torch::Tensor applyActivation(torch::Tensor x, const std::string& activation) {
        if (activation == "relu") {
            return torch::relu(x);
        } else if (activation == "gelu") {
            return torch::gelu(x);
        } else if (activation == "leaky_relu") {
            return torch::leaky_relu(x);
        }
        return torch::relu(x);
    }
};

// Factory functions for creating CNN variants
std::shared_ptr<torch::nn::Module> createOptimizedCNN(const OptimizedCNNNetwork::OptimizedConfig& config) {
    return std::make_shared<OptimizedCNNNetwork>(config);
}

std::shared_ptr<torch::nn::Module> createDilatedCNN(const DilatedCNNNetwork::Config& config) {
    return std::make_shared<DilatedCNNNetwork>(config);
}

} // namespace Models
} // namespace ArchNeuronX
