#pragma once
// ============================================================
// ArchNeuronX v2 - Temporal Fusion Transformer (TFT)
// State-of-the-art for financial time series forecasting
// Architecture: Variable Selection + LSTM Encoder + Multi-Head Attention
// Paper: https://arxiv.org/abs/1912.09363
// ============================================================
#include <torch/torch.h>
#include <vector>
#include <string>

namespace archneuronx {
namespace models {

// Gated Residual Network - core building block of TFT
struct GatedResidualNetworkImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc_skip{nullptr};
    torch::nn::Linear gate{nullptr};
    torch::nn::LayerNorm layer_norm{nullptr};
    torch::nn::Dropout dropout{nullptr};

    GatedResidualNetworkImpl(int64_t input_size,
                              int64_t hidden_size,
                              int64_t output_size,
                              double dropout_rate = 0.1,
                              bool has_context = false);

    torch::Tensor forward(torch::Tensor x,
                           std::optional<torch::Tensor> context = std::nullopt);
};
TORCH_MODULE(GatedResidualNetwork);

// Variable Selection Network
struct VariableSelectionNetworkImpl : torch::nn::Module {
    torch::nn::ModuleList var_grns;  // One GRN per variable
    GatedResidualNetwork softmax_grn{nullptr};
    int64_t num_vars_;

    VariableSelectionNetworkImpl(int64_t input_size,
                                   int64_t num_vars,
                                   int64_t hidden_size,
                                   double dropout_rate = 0.1);

    // Returns: (processed_features, variable_weights)
    std::pair<torch::Tensor, torch::Tensor>
    forward(torch::Tensor x,
             std::optional<torch::Tensor> context = std::nullopt);
};
TORCH_MODULE(VariableSelectionNetwork);

// Full Temporal Fusion Transformer
struct TemporalFusionTransformerImpl : torch::nn::Module {
    // Input processing
    VariableSelectionNetwork static_vsn{nullptr};
    VariableSelectionNetwork temporal_vsn{nullptr};

    // Static covariate encoders
    GatedResidualNetwork static_context_h{nullptr};
    GatedResidualNetwork static_context_c{nullptr};
    GatedResidualNetwork static_context_enrichment{nullptr};

    // LSTM Encoder-Decoder
    torch::nn::LSTM encoder{nullptr};
    torch::nn::LSTM decoder{nullptr};

    // Temporal Self-Attention
    torch::nn::MultiheadAttention temporal_attn{nullptr};
    torch::nn::LayerNorm attn_layer_norm{nullptr};

    // Position-wise feed-forward
    GatedResidualNetwork positionwise_grn{nullptr};
    torch::nn::LayerNorm final_layer_norm{nullptr};

    // Output projection: 3 classes (BUY/SELL/HOLD)
    torch::nn::Linear output_proj{nullptr};
    torch::nn::Dropout dropout{nullptr};

    // Config
    int64_t hidden_size_;
    int64_t num_encoder_steps_;

    TemporalFusionTransformerImpl(
        int64_t num_temporal_features,  // e.g., 20 (OHLCV + indicators)
        int64_t num_static_features,    // e.g., 5 (symbol embedding, etc.)
        int64_t hidden_size = 64,
        int64_t num_heads = 4,
        int64_t num_encoder_steps = 168,   // 1 week of hourly data
        double dropout_rate = 0.1
    );

    // Forward: returns logits [batch, 3] for BUY/SELL/HOLD
    torch::Tensor forward(
        torch::Tensor temporal_inputs,   // [batch, seq_len, num_temporal_features]
        torch::Tensor static_inputs      // [batch, num_static_features]
    );

    // Interpretability: return attention weights
    std::pair<torch::Tensor, torch::Tensor>
    forward_with_attention(
        torch::Tensor temporal_inputs,
        torch::Tensor static_inputs
    );
};
TORCH_MODULE(TemporalFusionTransformer);

} // namespace models
} // namespace archneuronx
