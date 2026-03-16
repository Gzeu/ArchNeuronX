#include "models/lstm_model.hpp"
#include <torch/torch.h>
#include <iostream>
#include <stdexcept>

namespace archneuronx {
namespace models {

LSTMModel::LSTMModel(const LSTMConfig& config)
    : config_(config) {
    // Initialize LSTM layers
    lstm_ = register_module("lstm",
        torch::nn::LSTM(torch::nn::LSTMOptions(config.input_size, config.hidden_size)
            .num_layers(config.num_layers)
            .batch_first(true)
            .dropout(config.dropout)
            .bidirectional(config.bidirectional)));

    int lstm_output_size = config.hidden_size * (config.bidirectional ? 2 : 1);

    // Fully connected output layer
    fc_ = register_module("fc",
        torch::nn::Linear(lstm_output_size, config.output_size));

    // Layer normalization
    layer_norm_ = register_module("layer_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({lstm_output_size})));

    // Move to device
    to(config.device);
}

torch::Tensor LSTMModel::forward(torch::Tensor input) {
    // input: [batch_size, seq_len, input_size]
    auto lstm_output = lstm_->forward(input);
    auto output = std::get<0>(lstm_output);  // [batch_size, seq_len, hidden_size]

    // Take last time step
    auto last_output = output.select(1, -1);  // [batch_size, hidden_size]

    // Apply layer normalization
    last_output = layer_norm_->forward(last_output);

    // Apply dropout during training
    if (is_training()) {
        last_output = torch::dropout(last_output, config_.dropout, true);
    }

    // Fully connected
    return fc_->forward(last_output);
}

TradingSignal LSTMModel::predict(const torch::Tensor& features) {
    this->eval();
    torch::NoGradGuard no_grad;

    auto output = forward(features.unsqueeze(0));
    auto probs = torch::softmax(output, 1);
    auto [max_prob, action] = probs.max(1);

    TradingSignal signal;
    signal.action = static_cast<SignalAction>(action.item<int>());
    signal.confidence = max_prob.item<float>();
    signal.timestamp = std::chrono::system_clock::now();
    signal.raw_output = output.squeeze(0);

    return signal;
}

void LSTMModel::save(const std::string& path) {
    torch::save(*this, path);
    std::cout << "[LSTMModel] Saved to: " << path << std::endl;
}

void LSTMModel::load(const std::string& path) {
    torch::load(*this, path);
    std::cout << "[LSTMModel] Loaded from: " << path << std::endl;
}

} // namespace models
} // namespace archneuronx
