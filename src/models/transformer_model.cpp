// ============================================================
// ArchNeuronX v2 - Temporal Fusion Transformer Implementation
// State-of-the-art for financial time series forecasting
// ============================================================
#include "models/transformer.hpp"
#include <torch/torch.h>
#include <iostream>
#include <stdexcept>

namespace archneuronx {
namespace models {

// Gated Residual Network Implementation
GatedResidualNetworkImpl::GatedResidualNetworkImpl(
    int64_t input_size,
    int64_t hidden_size,
    int64_t output_size,
    double dropout_rate,
    bool has_context
) {
    fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
    fc2 = register_module("fc2", torch::nn::Linear(hidden_size, output_size));
    
    if (has_context) {
        fc_skip = register_module("fc_skip", torch::nn::Linear(input_size, output_size));
    }
    
    gate = register_module("gate", torch::nn::Linear(hidden_size, output_size));
    layer_norm = register_module("layer_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({output_size})));
    dropout = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(dropout_rate)));
}

torch::Tensor GatedResidualNetworkImpl::forward(
    torch::Tensor x,
    std::optional<torch::Tensor> context
) {
    auto x_original = x;
    
    // Apply ELU activation
    x = torch::elu(fc1->forward(x));
    x = dropout->forward(x);
    x = fc2->forward(x);
    
    // Apply gate
    auto gate_values = torch::sigmoid(gate->forward(x));
    x = x * gate_values;
    
    // Add skip connection if available
    if (fc_skip) {
        x = x + fc_skip->forward(x_original);
    } else if (x.sizes() == x_original.sizes()) {
        x = x + x_original;
    }
    
    // Apply layer normalization
    x = layer_norm->forward(x);
    
    return x;
}

// Variable Selection Network Implementation
VariableSelectionNetworkImpl::VariableSelectionNetworkImpl(
    int64_t input_size,
    int64_t num_vars,
    int64_t hidden_size,
    double dropout_rate
) : num_vars_(num_vars) {
    // Create GRN for each variable
    for (int64_t i = 0; i < num_vars; ++i) {
        auto grn = GatedResidualNetwork(input_size / num_vars, hidden_size, hidden_size, dropout_rate);
        var_grns->push_back(grn);
    }
    
    // Softmax GRN for variable selection weights
    softmax_grn = GatedResidualNetwork(input_size, hidden_size, num_vars, dropout_rate);
}

std::pair<torch::Tensor, torch::Tensor> VariableSelectionNetworkImpl::forward(
    torch::Tensor x,
    std::optional<torch::Tensor> context
) {
    auto batch_size = x.size(0);
    auto feature_size = x.size(1);
    auto var_feature_size = feature_size / num_vars_;
    
    // Split input into variables
    std::vector<torch::Tensor> vars;
    for (int64_t i = 0; i < num_vars_; ++i) {
        auto start_idx = i * var_feature_size;
        auto end_idx = (i + 1) * var_feature_size;
        vars.push_back(x.slice(1, start_idx, end_idx));
    }
    
    // Process each variable through its GRN
    std::vector<torch::Tensor> processed_vars;
    for (int64_t i = 0; i < num_vars_; ++i) {
        auto processed = var_grns[i]->as<GatedResidualNetwork>()->forward(vars[i], context);
        processed_vars.push_back(processed);
    }
    
    // Concatenate processed variables
    auto processed_features = torch::cat(processed_vars, 1);
    
    // Calculate variable selection weights
    auto sparse_weights = torch::softmax(
        softmax_grn->forward(x, context), 
        -1
    );
    
    // Apply weights to processed features
    auto weighted_features = processed_features * sparse_weights;
    
    return {weighted_features, sparse_weights};
}

// Temporal Fusion Transformer Implementation
TemporalFusionTransformerImpl::TemporalFusionTransformerImpl(
    int64_t num_temporal_features,
    int64_t num_static_features,
    int64_t hidden_size,
    int64_t num_heads,
    int64_t num_encoder_steps,
    double dropout_rate
) : hidden_size_(hidden_size), num_encoder_steps_(num_encoder_steps) {
    
    // Variable Selection Networks
    static_vsn = VariableSelectionNetwork(
        num_static_features, 
        num_static_features, 
        hidden_size, 
        dropout_rate
    );
    temporal_vsn = VariableSelectionNetwork(
        num_temporal_features * num_encoder_steps,
        num_temporal_features,
        hidden_size,
        dropout_rate
    );
    
    // Static covariate encoders
    static_context_h = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout_rate);
    static_context_c = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout_rate);
    static_context_enrichment = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout_rate);
    
    // LSTM Encoder-Decoder
    encoder = torch::nn::LSTM(torch::nn::LSTMOptions(hidden_size, hidden_size)
        .num_layers(1)
        .dropout(dropout_rate)
        .batch_first(true));
    
    decoder = torch::nn::LSTM(torch::nn::LSTMOptions(hidden_size, hidden_size)
        .num_layers(1)
        .dropout(dropout_rate)
        .batch_first(true));
    
    // Temporal Self-Attention
    temporal_attn = torch::nn::MultiheadAttention(
        torch::nn::MultiheadAttentionOptions(hidden_size, num_heads)
            .dropout(dropout_rate)
            .batch_first(true)
    );
    
    attn_layer_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}));
    
    // Position-wise feed-forward
    positionwise_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout_rate);
    final_layer_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}));
    
    // Output projection
    output_proj = torch::nn::Linear(hidden_size, 3); // BUY/SELL/HOLD
    dropout = torch::nn::Dropout(torch::nn::DropoutOptions(dropout_rate));
}

torch::Tensor TemporalFusionTransformerImpl::forward(
    torch::Tensor temporal_inputs,
    torch::Tensor static_inputs
) {
    auto [processed_static, static_weights] = static_vsn->forward(static_inputs);
    auto [processed_temporal, temporal_weights] = temporal_vsn->forward(
        temporal_inputs.view({temporal_inputs.size(0), -1})
    );
    
    // Reshape temporal inputs
    processed_temporal = processed_temporal.view({
        temporal_inputs.size(0), 
        num_encoder_steps_, 
        -1
    });
    
    // Static context for encoder
    auto context_h = static_context_h->forward(processed_static);
    auto context_c = static_context_c->forward(processed_static);
    
    // LSTM encoding
    auto encoder_output = encoder->forward(processed_temporal, 
        torch::tanh(context_h).unsqueeze(0).repeat({1, processed_static.size(0), 1}),
        torch::tanh(context_c).unsqueeze(0).repeat({1, processed_static.size(0), 1))
    );
    
    auto encoded_sequence = std::get<0>(encoder_output);
    auto encoder_state = std::get<1>(encoder_output);
    
    // Self-attention
    auto [attn_output, attn_weights] = temporal_attn->forward(
        encoded_sequence, encoded_sequence, encoded_sequence
    );
    
    // Residual connection and layer norm
    auto attn_residual = attn_layer_norm->forward(encoded_sequence + attn_output);
    
    // Static enrichment
    auto enriched = static_context_enrichment->forward(processed_static);
    enriched = enriched.unsqueeze(1).expand({-1, num_encoder_steps_, -1});
    auto enriched_sequence = attn_residual + enriched;
    
    // Position-wise feed-forward
    auto positionwise_output = positionwise_grn->forward(enriched_sequence);
    auto final_output = final_layer_norm->forward(enriched_sequence + positionwise_output);
    
    // Take the last time step for prediction
    auto last_step = final_output.select(1, -1); // [batch, hidden_size]
    last_step = dropout->forward(last_step);
    
    // Output projection
    auto logits = output_proj->forward(last_step); // [batch, 3]
    
    return logits;
}

std::pair<torch::Tensor, torch::Tensor> TemporalFusionTransformerImpl::forward_with_attention(
    torch::Tensor temporal_inputs,
    torch::Tensor static_inputs
) {
    auto logits = forward(temporal_inputs, static_inputs);
    
    // For interpretability, we'll return variable selection weights
    auto [processed_static, static_weights] = static_vsn->forward(static_inputs);
    auto [processed_temporal, temporal_weights] = temporal_vsn->forward(
        temporal_inputs.view({temporal_inputs.size(0), -1})
    );
    
    return {logits, temporal_weights};
}

// Factory function for creating TFT model
TemporalFusionTransformer create_temporal_fusion_transformer(
    int64_t num_temporal_features,
    int64_t num_static_features,
    int64_t hidden_size,
    int64_t num_heads,
    int64_t num_encoder_steps,
    double dropout_rate
) {
    return TemporalFusionTransformer(
        num_temporal_features,
        num_static_features,
        hidden_size,
        num_heads,
        num_encoder_steps,
        dropout_rate
    );
}

} // namespace models
} // namespace archneuronx
