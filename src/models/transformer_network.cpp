/**
 * @file transformer_network.cpp
 * @brief Transformer network implementation for time series prediction
 * @author George Pricop
 * @date 2025-10-02
 */

#include "models/neural_networks.hpp"
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <cmath>

namespace ArchNeuronX {
namespace Models {

TransformerNetwork::TransformerNetwork(const Config& config) : config_(config) {
    buildNetwork();
    createPositionalEncoding();
}

void TransformerNetwork::buildNetwork() {
    // Input projection layer
    input_projection_ = torch::nn::Linear(config_.input_size, config_.d_model);
    register_module("input_projection", input_projection_);
    
    // Transformer layers
    transformer_ = torch::nn::Transformer(
        torch::nn::TransformerOptions(
            config_.d_model,
            config_.nhead
        )
        .num_encoder_layers(config_.num_encoder_layers)
        .num_decoder_layers(config_.num_decoder_layers)
        .dim_feedforward(config_.dim_feedforward)
        .dropout(config_.dropout_rate)
        .activation(config_.activation)
    );
    register_module("transformer", transformer_);
    
    // Output projection layer
    output_projection_ = torch::nn::Linear(config_.d_model, config_.output_size);
    register_module("output_projection", output_projection_);
    
    // Dropout layer
    if (config_.dropout_rate > 0.0) {
        dropout_ = torch::nn::Dropout(config_.dropout_rate);
        register_module("dropout", dropout_);
    }
}

void TransformerNetwork::createPositionalEncoding() {
    if (!config_.use_positional_encoding) {
        return;
    }
    
    // Create positional encoding matrix
    positional_encoding_ = torch::zeros({config_.max_seq_length, config_.d_model});
    
    for (int pos = 0; pos < config_.max_seq_length; ++pos) {
        for (int i = 0; i < config_.d_model; ++i) {
            double value;
            if (i % 2 == 0) {
                value = std::sin(pos / std::pow(10000.0, (double)i / config_.d_model));
            } else {
                value = std::cos(pos / std::pow(10000.0, (double)(i - 1) / config_.d_model));
            }
            positional_encoding_[pos][i] = value;
        }
    }
    
    positional_encoding_ = positional_encoding_.unsqueeze(0); // [1, max_seq_length, d_model]
}

torch::Tensor TransformerNetwork::addPositionalEncoding(torch::Tensor x) {
    if (!config_.use_positional_encoding) {
        return x;
    }
    
    int64_t seq_len = x.size(1);
    if (seq_len > config_.max_seq_length) {
        // Truncate or handle longer sequences
        seq_len = config_.max_seq_length;
        x = x.slice(1, 0, seq_len);
    }
    
    auto pos_encoding = positional_encoding_.slice(1, 0, seq_len);
    return x + pos_encoding;
}

torch::Tensor TransformerNetwork::forward(torch::Tensor x, torch::Tensor tgt) {
    // Input: [batch_size, sequence_length, input_size]
    
    int64_t batch_size = x.size(0);
    int64_t seq_len = x.size(1);
    
    // Project input to model dimension
    x = input_projection_(x); // [batch_size, seq_len, d_model]
    
    // Add positional encoding
    x = addPositionalEncoding(x);
    
    // Apply dropout
    if (config_.dropout_rate > 0.0 && dropout_) {
        x = dropout_(x);
    }
    
    // Transformer expects [seq_len, batch_size, d_model]
    x = x.transpose(0, 1); // [seq_len, batch_size, d_model]
    
    torch::Tensor output;
    
    if (tgt.defined()) {
        // Encoder-decoder mode
        tgt = input_projection_(tgt);
        tgt = addPositionalEncoding(tgt);
        tgt = tgt.transpose(0, 1); // [tgt_len, batch_size, d_model]
        
        output = transformer_->forward(x, tgt); // [tgt_len, batch_size, d_model]
        
        // Use the last target token for classification
        output = output.slice(0, -1); // [1, batch_size, d_model]
    } else {
        // Encoder-only mode for classification
        // Create dummy target (same as input for autoencoding)
        output = transformer_->forward(x, x); // [seq_len, batch_size, d_model]
        
        // Use the last sequence token for classification
        output = output.slice(0, -1); // [1, batch_size, d_model]
    }
    
    // Transpose back: [batch_size, d_model]
    output = output.transpose(0, 1).squeeze(0);
    
    // Final output projection
    output = output_projection_(output); // [batch_size, output_size]
    
    return output;
}

void TransformerNetwork::saveModel(const std::string& path) {
    try {
        torch::save(*this, path);
        
        // Save configuration separately
        std::string config_path = path + ".config";
        std::ofstream config_file(config_path);
        if (config_file.is_open()) {
            config_file << "input_size=" << config_.input_size << "\n";
            config_file << "d_model=" << config_.d_model << "\n";
            config_file << "nhead=" << config_.nhead << "\n";
            config_file << "num_encoder_layers=" << config_.num_encoder_layers << "\n";
            config_file << "num_decoder_layers=" << config_.num_decoder_layers << "\n";
            config_file << "dim_feedforward=" << config_.dim_feedforward << "\n";
            config_file << "output_size=" << config_.output_size << "\n";
            config_file << "dropout_rate=" << config_.dropout_rate << "\n";
            config_file << "max_seq_length=" << config_.max_seq_length << "\n";
            config_file << "use_positional_encoding=" << config_.use_positional_encoding << "\n";
            config_file << "use_layer_norm=" << config_.use_layer_norm << "\n";
            config_file << "activation=" << config_.activation << "\n";
            config_file.close();
        }
        
        std::cout << "Transformer model saved to: " << path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error saving Transformer model: " << e.what() << std::endl;
        throw;
    }
}

void TransformerNetwork::loadModel(const std::string& path) {
    try {
        torch::load(*this, path);
        
        // Load configuration if available
        std::string config_path = path + ".config";
        std::ifstream config_file(config_path);
        if (config_file.is_open()) {
            std::string line;
            while (std::getline(config_file, line)) {
                size_t pos = line.find('=');
                if (pos != std::string::npos) {
                    std::string key = line.substr(0, pos);
                    std::string value = line.substr(pos + 1);
                    
                    if (key == "input_size") config_.input_size = std::stoi(value);
                    else if (key == "d_model") config_.d_model = std::stoi(value);
                    else if (key == "nhead") config_.nhead = std::stoi(value);
                    else if (key == "num_encoder_layers") config_.num_encoder_layers = std::stoi(value);
                    else if (key == "num_decoder_layers") config_.num_decoder_layers = std::stoi(value);
                    else if (key == "dim_feedforward") config_.dim_feedforward = std::stoi(value);
                    else if (key == "output_size") config_.output_size = std::stoi(value);
                    else if (key == "dropout_rate") config_.dropout_rate = std::stod(value);
                    else if (key == "max_seq_length") config_.max_seq_length = std::stoi(value);
                    else if (key == "use_positional_encoding") config_.use_positional_encoding = (value == "1");
                    else if (key == "use_layer_norm") config_.use_layer_norm = (value == "1");
                    else if (key == "activation") config_.activation = value;
                }
            }
            config_file.close();
            
            // Recreate positional encoding with loaded config
            createPositionalEncoding();
        }
        
        std::cout << "Transformer model loaded from: " << path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading Transformer model: " << e.what() << std::endl;
        throw;
    }
}

} // namespace Models
} // namespace ArchNeuronX
