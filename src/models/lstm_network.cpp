/**
 * @file lstm_network.cpp
 * @brief LSTM network implementation for time series prediction
 * @author George Pricop
 * @date 2025-10-02
 */

#include "models/neural_networks.hpp"
#include <torch/torch.h>
#include <iostream>
#include <fstream>

namespace ArchNeuronX {
namespace Models {

LSTMNetwork::LSTMNetwork(const Config& config) : config_(config) {
    buildNetwork();
}

void LSTMNetwork::buildNetwork() {
    // LSTM layer
    lstm_ = torch::nn::LSTM(torch::nn::LSTMOptions(
        config_.input_size, 
        config_.hidden_size)
        .num_layers(config_.num_layers)
        .dropout(config_.dropout_rate)
        .batch_first(true)
        .bidirectional(config_.bidirectional));
    
    register_module("lstm", lstm_);
    
    // Calculate input size for output layer
    int lstm_output_size = config_.hidden_size;
    if (config_.bidirectional) {
        lstm_output_size *= 2;
    }
    
    // Output layer
    output_layer_ = torch::nn::Linear(lstm_output_size, config_.output_size);
    register_module("output_layer", output_layer_);
    
    // Dropout layer
    if (config_.dropout_rate > 0.0) {
        dropout_ = torch::nn::Dropout(config_.dropout_rate);
        register_module("dropout", dropout_);
    }
    
    // Batch normalization
    if (config_.use_batch_norm) {
        batch_norm_ = torch::nn::BatchNorm1d(lstm_output_size);
        register_module("batch_norm", batch_norm_);
    }
    
    // Attention mechanism
    if (config_.use_attention) {
        int attention_dim = lstm_output_size;
        attention_query_ = torch::nn::Linear(attention_dim, attention_dim);
        attention_key_ = torch::nn::Linear(attention_dim, attention_dim);
        attention_value_ = torch::nn::Linear(attention_dim, attention_dim);
        
        register_module("attention_query", attention_query_);
        register_module("attention_key", attention_key_);
        register_module("attention_value", attention_value_);
    }
}

torch::Tensor LSTMNetwork::forward(torch::Tensor x) {
    // Input: [batch_size, sequence_length, input_size]
    
    // LSTM forward pass
    auto lstm_output = lstm_->forward(x);
    auto lstm_out = std::get<0>(lstm_output); // [batch_size, seq_len, hidden_size * num_directions]
    
    torch::Tensor final_output;
    
    if (config_.use_attention) {
        // Apply attention mechanism
        final_output = applyAttention(lstm_out);
    } else {
        // Use the last time step output
        final_output = lstm_out.slice(1, -1); // [batch_size, hidden_size * num_directions]
    }
    
    // Apply batch normalization
    if (config_.use_batch_norm && batch_norm_) {
        // Reshape for batch norm: [batch_size, features]
        final_output = batch_norm_(final_output);
    }
    
    // Apply dropout
    if (config_.dropout_rate > 0.0 && dropout_) {
        final_output = dropout_(final_output);
    }
    
    // Final output layer
    auto output = output_layer_(final_output); // [batch_size, output_size]
    
    return output;
}

torch::Tensor LSTMNetwork::applyAttention(torch::Tensor lstm_output) {
    // lstm_output: [batch_size, seq_len, hidden_size * num_directions]
    
    int64_t batch_size = lstm_output.size(0);
    int64_t seq_len = lstm_output.size(1);
    int64_t hidden_dim = lstm_output.size(2);
    
    // Compute queries, keys, values
    auto queries = attention_query_(lstm_output); // [batch_size, seq_len, hidden_dim]
    auto keys = attention_key_(lstm_output);     // [batch_size, seq_len, hidden_dim]
    auto values = attention_value_(lstm_output);   // [batch_size, seq_len, hidden_dim]
    
    // Compute attention scores
    // queries: [batch_size, seq_len, hidden_dim]
    // keys.transpose(-2, -1): [batch_size, hidden_dim, seq_len]
    auto scores = torch::matmul(queries, keys.transpose(-2, -1)); // [batch_size, seq_len, seq_len]
    scores = scores / std::sqrt(hidden_dim); // Scale by sqrt(hidden_dim)
    
    // Apply softmax to get attention weights
    auto attention_weights = torch::softmax(scores, -1); // [batch_size, seq_len, seq_len]
    
    // Apply attention weights to values
    auto context = torch::matmul(attention_weights, values); // [batch_size, seq_len, hidden_dim]
    
    // Global attention: average over sequence length
    auto attended_output = torch::mean(context, 1); // [batch_size, hidden_dim]
    
    return attended_output;
}

void LSTMNetwork::saveModel(const std::string& path) {
    try {
        torch::save(*this, path);
        
        // Save configuration separately
        std::string config_path = path + ".config";
        std::ofstream config_file(config_path);
        if (config_file.is_open()) {
            config_file << "input_size=" << config_.input_size << "\n";
            config_file << "hidden_size=" << config_.hidden_size << "\n";
            config_file << "num_layers=" << config_.num_layers << "\n";
            config_file << "output_size=" << config_.output_size << "\n";
            config_file << "dropout_rate=" << config_.dropout_rate << "\n";
            config_file << "use_batch_norm=" << config_.use_batch_norm << "\n";
            config_file << "use_attention=" << config_.use_attention << "\n";
            config_file << "bidirectional=" << config_.bidirectional << "\n";
            config_file << "activation=" << config_.activation << "\n";
            config_file.close();
        }
        
        std::cout << "LSTM model saved to: " << path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error saving LSTM model: " << e.what() << std::endl;
        throw;
    }
}

void LSTMNetwork::loadModel(const std::string& path) {
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
                    else if (key == "hidden_size") config_.hidden_size = std::stoi(value);
                    else if (key == "num_layers") config_.num_layers = std::stoi(value);
                    else if (key == "output_size") config_.output_size = std::stoi(value);
                    else if (key == "dropout_rate") config_.dropout_rate = std::stod(value);
                    else if (key == "use_batch_norm") config_.use_batch_norm = (value == "1");
                    else if (key == "use_attention") config_.use_attention = (value == "1");
                    else if (key == "bidirectional") config_.bidirectional = (value == "1");
                    else if (key == "activation") config_.activation = value;
                }
            }
            config_file.close();
        }
        
        std::cout << "LSTM model loaded from: " << path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading LSTM model: " << e.what() << std::endl;
        throw;
    }
}

} // namespace Models
} // namespace ArchNeuronX
