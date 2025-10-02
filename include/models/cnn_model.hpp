/**
 * @file cnn_model.hpp
 * @brief 1D Convolutional Neural Network for financial time series
 * @author George Pricop
 * @date 2025-10-02
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include <memory>
#include "core/base_model.hpp"

namespace ArchNeuronX {
namespace Models {

/**
 * @struct ConvLayer
 * @brief Configuration for a convolutional layer
 */
struct ConvLayer {
    int64_t out_channels = 32;     ///< Output channels
    int64_t kernel_size = 3;       ///< Convolution kernel size
    int64_t stride = 1;            ///< Stride
    int64_t padding = 1;           ///< Padding
    int64_t dilation = 1;          ///< Dilation
    bool use_batch_norm = true;    ///< Use batch normalization
    double dropout_rate = 0.1;     ///< Dropout rate after layer
    
    ConvLayer() = default;
    ConvLayer(int64_t out_ch, int64_t kernel, int64_t str = 1, int64_t pad = 1)
        : out_channels(out_ch), kernel_size(kernel), stride(str), padding(pad) {}
};

/**
 * @struct PoolingLayer
 * @brief Configuration for pooling layer
 */
struct PoolingLayer {
    enum Type { MAX, AVERAGE, ADAPTIVE_MAX, ADAPTIVE_AVG } type = MAX;
    int64_t kernel_size = 2;
    int64_t stride = 2;
    int64_t padding = 0;
    
    PoolingLayer() = default;
    PoolingLayer(Type t, int64_t kernel = 2, int64_t str = 2) 
        : type(t), kernel_size(kernel), stride(str) {}
};

/**
 * @struct CNNConfig
 * @brief Configuration for CNN model
 */
struct CNNConfig {
    std::vector<ConvLayer> conv_layers;
    std::vector<PoolingLayer> pool_layers;  ///< Optional pooling after each conv layer
    
    // Dense layers after convolution
    std::vector<int64_t> dense_layers = {64, 32};
    double dense_dropout = 0.3;
    
    // Architecture options
    std::string activation = "relu";        ///< Activation function
    bool use_global_pooling = true;         ///< Use global pooling before dense layers
    bool use_residual = false;              ///< Use residual connections
    double weight_init_std = 0.1;           ///< Weight initialization std
    
    // Attention mechanism
    bool use_attention = false;             ///< Use self-attention layers
    int64_t attention_heads = 8;            ///< Number of attention heads
    
    /**
     * @brief Create default CNN configuration for financial data
     * @param sequence_length Input sequence length
     * @return Default CNN configuration
     */
    static CNNConfig createDefault(int64_t sequence_length);
    
    /**
     * @brief Validate CNN configuration
     * @throws std::invalid_argument if configuration is invalid
     */
    void validate() const;
};

/**
 * @class CNNModel
 * @brief 1D Convolutional Neural Network for time series analysis
 */
class CNNModel : public BaseModel {
public:
    /**
     * @brief Constructor
     * @param name Model name
     * @param config CNN configuration
     */
    CNNModel(const std::string& name, const CNNConfig& config);
    
    /**
     * @brief Initialize CNN architecture
     * @param input_size Size of input features per timestep
     * @param output_size Size of output predictions
     */
    void initialize(int64_t input_size, int64_t output_size) override;
    
    /**
     * @brief Forward pass through the CNN
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
     * @brief Get number of trainable parameters
     * @return Total number of parameters
     */
    int64_t getParameterCount() const;
    
    /**
     * @brief Extract learned features from convolutional layers
     * @param input Input tensor
     * @return Feature tensor after convolution layers
     */
    torch::Tensor extractFeatures(torch::Tensor input);
    
    /**
     * @brief Get convolutional layer activations
     * @param input Input tensor
     * @return Vector of activation maps from each conv layer
     */
    std::vector<torch::Tensor> getConvActivations(torch::Tensor input);
    
    /**
     * @brief Apply gradient clipping
     * @param max_norm Maximum gradient norm
     */
    void clipGradients(double max_norm = 1.0);
    
    /**
     * @brief Visualize learned filters
     * @param layer_index Index of convolutional layer
     * @return Tensor containing filter weights
     */
    torch::Tensor getConvFilters(int layer_index) const;
    
    // Getters
    const CNNConfig& getConfig() const { return config_; }
    int64_t getSequenceLength() const { return sequence_length_; }
    
private:
    /**
     * @brief Calculate output size after convolution
     * @param input_size Input size
     * @param kernel_size Kernel size
     * @param stride Stride
     * @param padding Padding
     * @param dilation Dilation
     * @return Output size
     */
    int64_t calculateConvOutputSize(int64_t input_size, int64_t kernel_size, 
                                   int64_t stride, int64_t padding, 
                                   int64_t dilation = 1) const;
    
    /**
     * @brief Create pooling layer based on configuration
     * @param pool_config Pooling configuration
     * @return Pooling module
     */
    torch::nn::AnyModule createPoolingLayer(const PoolingLayer& pool_config);
    
    /**
     * @brief Initialize convolutional layer weights
     * @param layer Convolutional layer to initialize
     */
    void initializeConvWeights(torch::nn::Conv1d& layer);
    
    CNNConfig config_;
    
    // Network architecture
    torch::nn::ModuleList conv_layers_;
    torch::nn::ModuleList batch_norms_;
    torch::nn::ModuleList pool_layers_;
    torch::nn::ModuleList conv_dropouts_;
    
    // Self-attention layers (optional)
    torch::nn::ModuleList attention_layers_;
    
    // Dense layers
    torch::nn::ModuleList dense_layers_;
    torch::nn::ModuleList dense_batch_norms_;
    torch::nn::ModuleList dense_dropouts_;
    
    // Output layer
    torch::nn::Linear output_layer_{nullptr};
    
    // Activation functions
    torch::nn::AnyModule conv_activation_;
    torch::nn::AnyModule dense_activation_;
    
    // Global pooling
    torch::nn::AdaptiveAvgPool1d global_pool_{nullptr};
    
    // Model dimensions
    int64_t sequence_length_ = 0;
    int64_t input_features_ = 0;
    int64_t conv_output_size_ = 0;
};

/**
 * @class ResidualBlock
 * @brief Residual block for CNN with skip connections
 */
class ResidualBlock : public torch::nn::Module {
public:
    /**
     * @brief Constructor
     * @param in_channels Input channels
     * @param out_channels Output channels
     * @param kernel_size Convolution kernel size
     * @param stride Stride
     */
    ResidualBlock(int64_t in_channels, int64_t out_channels, 
                 int64_t kernel_size = 3, int64_t stride = 1);
    
    /**
     * @brief Forward pass through residual block
     * @param x Input tensor
     * @return Output tensor
     */
    torch::Tensor forward(torch::Tensor x);
    
private:
    torch::nn::Conv1d conv1_{nullptr};
    torch::nn::BatchNorm1d bn1_{nullptr};
    torch::nn::Conv1d conv2_{nullptr};
    torch::nn::BatchNorm1d bn2_{nullptr};
    torch::nn::Conv1d skip_connection_{nullptr};  ///< Skip connection (if needed)
    torch::nn::ReLU relu_;
    
    bool use_skip_connection_;
};

/**
 * @class CNNResNet
 * @brief ResNet-style CNN for financial time series
 */
class CNNResNet : public BaseModel {
public:
    /**
     * @brief Constructor
     * @param name Model name
     * @param num_blocks Number of residual blocks per stage
     * @param channels Channel progression
     */
    CNNResNet(const std::string& name, 
              const std::vector<int>& num_blocks,
              const std::vector<int64_t>& channels);
    
    /**
     * @brief Initialize ResNet architecture
     * @param input_size Size of input features per timestep
     * @param output_size Size of output predictions
     */
    void initialize(int64_t input_size, int64_t output_size) override;
    
    /**
     * @brief Forward pass through ResNet
     * @param input Input tensor [batch_size, sequence_length, features]
     * @return Output tensor [batch_size, output_size]
     */
    torch::Tensor forward(torch::Tensor input) override;
    
    /**
     * @brief Get model information
     * @return Formatted model info
     */
    std::string getModelInfo() const override;
    
private:
    /**
     * @brief Create residual stage
     * @param in_channels Input channels
     * @param out_channels Output channels
     * @param num_blocks Number of blocks in stage
     * @param stride Stride for first block
     * @return Sequential module containing blocks
     */
    torch::nn::Sequential createStage(int64_t in_channels, int64_t out_channels,
                                     int num_blocks, int64_t stride = 1);
    
    std::vector<int> num_blocks_;
    std::vector<int64_t> channels_;
    
    // Network layers
    torch::nn::Conv1d initial_conv_{nullptr};
    torch::nn::BatchNorm1d initial_bn_{nullptr};
    torch::nn::ModuleList stages_;
    torch::nn::AdaptiveAvgPool1d global_pool_{nullptr};
    torch::nn::Linear fc_{nullptr};
};

} // namespace Models
} // namespace ArchNeuronX