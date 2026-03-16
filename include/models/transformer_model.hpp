/**
 * @file transformer_model.hpp
 * @brief Financial Time-Series Transformer Model
 * @version 2.0.0
 *
 * Multi-head self-attention Transformer adapted for financial
 * time-series prediction. Supports both encoder-only (BERT-style)
 * and encoder-decoder architectures.
 *
 * Key features:
 * - Flash Attention (CUDA Tensor Core optimized)
 * - Rotary Position Embedding (RoPE)
 * - Pre-LayerNorm for training stability
 * - Mixed precision (fp16/bf16) inference
 */

#pragma once

#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>

namespace ArchNeuronX {
namespace Models {

// ============================================================
// Configuration
// ============================================================
struct TransformerConfig {
    // Model dimensions
    int d_model = 256;          // Embedding dimension
    int num_heads = 8;          // Attention heads
    int num_encoder_layers = 4; // Encoder depth
    int num_decoder_layers = 2; // Decoder depth (0 = encoder-only)
    int d_ff = 1024;            // Feed-forward hidden dim
    int max_seq_len = 512;      // Maximum sequence length
    int input_features = 30;    // Number of input features
    int output_classes = 3;     // Buy/Sell/Hold

    // Regularization
    double dropout = 0.1;
    double attention_dropout = 0.1;

    // Positional encoding type: "sinusoidal", "learnable", "rope"
    std::string pos_encoding = "rope";

    // Activation: "relu", "gelu", "swiglu"
    std::string activation = "gelu";

    // Precision for inference
    bool use_fp16 = false;
    bool use_bf16 = false;

    // Flash attention (requires CUDA + Ampere+)
    bool use_flash_attention = true;
};

// ============================================================
// Multi-Head Attention with optional Flash Attention
// ============================================================
struct MultiHeadAttentionImpl : torch::nn::Module {
    explicit MultiHeadAttentionImpl(const TransformerConfig& cfg);

    torch::Tensor forward(const torch::Tensor& query,
                         const torch::Tensor& key,
                         const torch::Tensor& value,
                         const torch::Tensor& mask = {});

private:
    int d_model_, num_heads_, d_k_;
    bool use_flash_;

    torch::nn::Linear q_proj_{nullptr}, k_proj_{nullptr},
                     v_proj_{nullptr}, out_proj_{nullptr};
    torch::nn::Dropout attn_dropout_{nullptr};

    // Standard scaled dot-product attention
    torch::Tensor scaled_dot_product_attention(
        const torch::Tensor& q, const torch::Tensor& k,
        const torch::Tensor& v, const torch::Tensor& mask);

    // Flash attention (memory-efficient)
    torch::Tensor flash_attention(
        const torch::Tensor& q, const torch::Tensor& k,
        const torch::Tensor& v, const torch::Tensor& mask);

    // Rotary Position Embedding
    torch::Tensor apply_rope(const torch::Tensor& x, int seq_len);
};
TORCH_MODULE(MultiHeadAttention);

// ============================================================
// Transformer Encoder Layer
// ============================================================
struct TransformerEncoderLayerImpl : torch::nn::Module {
    explicit TransformerEncoderLayerImpl(const TransformerConfig& cfg);

    torch::Tensor forward(const torch::Tensor& src,
                         const torch::Tensor& src_mask = {});

private:
    MultiHeadAttention self_attn_{nullptr};
    torch::nn::Linear ff1_{nullptr}, ff2_{nullptr};
    torch::nn::LayerNorm norm1_{nullptr}, norm2_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    std::string activation_;

    torch::Tensor activate(const torch::Tensor& x);
};
TORCH_MODULE(TransformerEncoderLayer);

// ============================================================
// Main Transformer Model
// ============================================================
struct FinancialTransformerImpl : torch::nn::Module {
    explicit FinancialTransformerImpl(const TransformerConfig& cfg);

    /**
     * @brief Forward pass
     * @param x  Input tensor [batch, seq_len, input_features]
     * @return Logits tensor [batch, output_classes]
     */
    torch::Tensor forward(const torch::Tensor& x);

    /**
     * @brief Forward with attention weights (for explainability)
     * @return {logits, attention_weights}
     */
    std::pair<torch::Tensor, torch::Tensor>
    forward_with_attention(const torch::Tensor& x);

    /**
     * @brief Get model config
     */
    const TransformerConfig& config() const { return config_; }

    /**
     * @brief Count total parameters
     */
    int64_t num_parameters() const;

    /**
     * @brief Convert to half precision for fast inference
     */
    void to_half_precision();

private:
    TransformerConfig config_;

    // Input projection: [features] -> [d_model]
    torch::nn::Linear input_proj_{nullptr};

    // Positional encoding
    torch::Tensor pos_encoding_;

    // Encoder stack
    torch::nn::ModuleList encoder_layers_;

    // Global average pooling + classification head
    torch::nn::LayerNorm output_norm_{nullptr};
    torch::nn::Linear classifier_{nullptr};
    torch::nn::Dropout dropout_{nullptr};

    void init_sinusoidal_encoding();
    void init_learnable_encoding();
};
TORCH_MODULE(FinancialTransformer);

// ============================================================
// Model factory helpers
// ============================================================
FinancialTransformer make_small_transformer(int input_features, int num_classes);
FinancialTransformer make_medium_transformer(int input_features, int num_classes);
FinancialTransformer make_large_transformer(int input_features, int num_classes);

} // namespace Models
} // namespace ArchNeuronX
