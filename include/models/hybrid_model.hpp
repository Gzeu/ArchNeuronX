/**
 * @file hybrid_model.hpp
 * @brief Hybrid MLP-CNN model for comprehensive financial analysis
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include <memory>
#include "core/base_model.hpp"
#include "mlp_model.hpp"
#include "cnn_model.hpp"

namespace ArchNeuronX {
namespace Models {

/**
 * @enum FusionStrategy
 * @brief Strategies for combining MLP and CNN outputs
 */
enum class FusionStrategy {
    CONCATENATION,    ///< Concatenate MLP and CNN features
    WEIGHTED_SUM,     ///< Weighted sum of predictions
    ATTENTION_FUSION, ///< Attention-based fusion
    GATING,          ///< Gated fusion mechanism
    ENSEMBLE_VOTE    ///< Ensemble voting
};

/**
 * @struct HybridConfig
 * @brief Configuration for hybrid MLP-CNN model
 */
struct HybridConfig {
    // Component configurations
    MLPConfig mlp_config;
    CNNConfig cnn_config;
    
    // Fusion settings
    FusionStrategy fusion_strategy = FusionStrategy::CONCATENATION;
    std::vector<double> component_weights = {0.5, 0.5};  ///< Weights for MLP, CNN
    
    // Fusion layer configuration
    std::vector<int64_t> fusion_layers = {64, 32};       ///< Dense layers after fusion
    double fusion_dropout = 0.2;
    bool use_fusion_batch_norm = true;
    
    // Attention fusion settings (if used)
    int64_t attention_dim = 64;
    int64_t attention_heads = 4;
    double attention_dropout = 0.1;
    
    // Gating mechanism settings (if used)
    bool use_learnable_gates = true;
    double gate_init_bias = 0.0;
    
    /**
     * @brief Validate hybrid configuration
     * @throws std::invalid_argument if configuration is invalid
     */
    void validate() const;
    
    /**
     * @brief Create default hybrid configuration
     * @param input_size Input feature size
     * @param sequence_length Sequence length
     * @return Default hybrid configuration
     */
    static HybridConfig createDefault(int64_t input_size, int64_t sequence_length);
};

/**
 * @class HybridModel
 * @brief Hybrid model combining MLP and CNN architectures
 */
class HybridModel : public BaseModel {
public:
    /**
     * @brief Constructor
     * @param name Model name
     * @param config Hybrid model configuration
     */
    HybridModel(const std::string& name, const HybridConfig& config);
    
    /**
     * @brief Initialize hybrid architecture
     * @param input_size Size of input features per timestep
     * @param output_size Size of output predictions
     */
    void initialize(int64_t input_size, int64_t output_size) override;
    
    /**
     * @brief Forward pass through hybrid model
     * @param input Input tensor [batch_size, sequence_length, features]
     * @return Output tensor [batch_size, output_size]
     */
    torch::Tensor forward(torch::Tensor input) override;
    
    /**
     * @brief Get model architecture as string
     * @return Formatted architecture description
     */
    std::string getModelInfo() const override;
    
    /**
     * @brief Get individual component predictions
     * @param input Input tensor
     * @return Tuple of (MLP prediction, CNN prediction, fusion gates)
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
        getComponentPredictions(torch::Tensor input);
    
    /**
     * @brief Get attention weights (if attention fusion is used)
     * @param input Input tensor
     * @return Attention weight tensor
     */
    torch::Tensor getAttentionWeights(torch::Tensor input);
    
    /**
     * @brief Get fusion gate values (if gating is used)
     * @param mlp_features MLP features
     * @param cnn_features CNN features
     * @return Gate values tensor
     */
    torch::Tensor getFusionGates(torch::Tensor mlp_features, torch::Tensor cnn_features);
    
    /**
     * @brief Set component training modes independently
     * @param mlp_training MLP training mode
     * @param cnn_training CNN training mode
     */
    void setComponentTraining(bool mlp_training, bool cnn_training);
    
    /**
     * @brief Freeze/unfreeze model components
     * @param freeze_mlp Freeze MLP parameters
     * @param freeze_cnn Freeze CNN parameters
     */
    void freezeComponents(bool freeze_mlp, bool freeze_cnn);
    
    /**
     * @brief Get number of trainable parameters
     * @return Total number of parameters
     */
    int64_t getParameterCount() const;
    
    // Getters
    const HybridConfig& getConfig() const { return config_; }
    const std::shared_ptr<MLPModel>& getMLPModel() const { return mlp_model_; }
    const std::shared_ptr<CNNModel>& getCNNModel() const { return cnn_model_; }
    
private:
    /**
     * @brief Apply concatenation fusion
     * @param mlp_features MLP output features
     * @param cnn_features CNN output features
     * @return Fused features
     */
    torch::Tensor applyConcatenationFusion(torch::Tensor mlp_features, 
                                          torch::Tensor cnn_features);
    
    /**
     * @brief Apply weighted sum fusion
     * @param mlp_output MLP predictions
     * @param cnn_output CNN predictions
     * @return Fused predictions
     */
    torch::Tensor applyWeightedSumFusion(torch::Tensor mlp_output, 
                                        torch::Tensor cnn_output);
    
    /**
     * @brief Apply attention-based fusion
     * @param mlp_features MLP features
     * @param cnn_features CNN features
     * @return Attention-fused features
     */
    torch::Tensor applyAttentionFusion(torch::Tensor mlp_features, 
                                      torch::Tensor cnn_features);
    
    /**
     * @brief Apply gated fusion mechanism
     * @param mlp_features MLP features
     * @param cnn_features CNN features
     * @return Gated fusion output
     */
    torch::Tensor applyGatedFusion(torch::Tensor mlp_features, 
                                  torch::Tensor cnn_features);
    
    HybridConfig config_;
    
    // Component models
    std::shared_ptr<MLPModel> mlp_model_;
    std::shared_ptr<CNNModel> cnn_model_;
    
    // Fusion layers
    torch::nn::ModuleList fusion_layers_;
    torch::nn::ModuleList fusion_batch_norms_;
    torch::nn::ModuleList fusion_dropouts_;
    
    // Attention mechanism (for attention fusion)
    torch::nn::MultiheadAttention attention_{nullptr};
    torch::nn::LayerNorm attention_norm_{nullptr};
    
    // Gating mechanism (for gated fusion)
    torch::nn::Linear gate_mlp_{nullptr};
    torch::nn::Linear gate_cnn_{nullptr};
    torch::nn::Sigmoid gate_activation_;
    
    // Component weights (for weighted fusion)
    torch::Tensor component_weights_;
};

/**
 * @brief Convert FusionStrategy to string
 * @param strategy Fusion strategy
 * @return String representation
 */
std::string fusionStrategyToString(FusionStrategy strategy);

/**
 * @brief Create hybrid model factory function
 * @param name Model name
 * @param config Hybrid configuration
 * @return Unique pointer to hybrid model
 */
std::unique_ptr<HybridModel> createHybridModel(
    const std::string& name,
    const HybridConfig& config
);

} // namespace Models
} // namespace ArchNeuronX