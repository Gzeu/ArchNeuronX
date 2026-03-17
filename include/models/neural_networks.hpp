#ifndef NEURAL_NETWORKS_H
#define NEURAL_NETWORKS_H

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <string>

namespace ArchNeuronX {
namespace Models {

/**
 * @brief Multi-Layer Perceptron for time series prediction
 * 
 * Implements a configurable MLP architecture optimized for financial time series
 * analysis with dropout, batch normalization, and various activation functions.
 */
class MLPNetwork : public torch::nn::Module {
public:
    /**
     * @brief Configuration structure for MLP parameters
     */
    struct Config {
        int input_size = 50;              // Number of input features
        std::vector<int> hidden_sizes = {128, 64, 32}; // Hidden layer dimensions
        int output_size = 3;              // Number of output classes (buy/sell/hold)
        double dropout_rate = 0.2;        // Dropout probability
        bool use_batch_norm = true;       // Enable batch normalization
        std::string activation = "relu";  // Activation function (relu, tanh, sigmoid)
        bool use_residual = false;        // Enable residual connections
    };

    /**
     * @brief Constructor
     * @param config Network configuration
     */
    explicit MLPNetwork(const Config& config);

    /**
     * @brief Forward pass
     * @param x Input tensor [batch_size, sequence_length, input_size]
     * @return Output tensor [batch_size, output_size]
     */
    torch::Tensor forward(torch::Tensor x);

    /**
     * @brief Get model configuration
     * @return Current configuration
     */
    const Config& getConfig() const { return config_; }

    /**
     * @brief Save model weights and configuration
     * @param path File path to save model
     */
    void saveModel(const std::string& path);

    /**
     * @brief Load model weights and configuration
     * @param path File path to load model from
     */
    void loadModel(const std::string& path);

private:
    Config config_;
    torch::nn::ModuleList layers_{nullptr};
    torch::nn::ModuleList batch_norms_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    
    void buildNetwork();
    torch::Tensor applyActivation(torch::Tensor x, const std::string& activation);
};

/**
 * @brief Convolutional Neural Network for time series pattern recognition
 * 
 * Implements 1D CNN architecture for detecting patterns in financial time series
 * with multiple convolutional layers, pooling, and fully connected layers.
 */
class CNNNetwork : public torch::nn::Module {
public:
    /**
     * @brief Configuration structure for CNN parameters
     */
    struct Config {
        int input_channels = 1;           // Number of input channels
        int sequence_length = 50;         // Input sequence length
        std::vector<int> conv_channels = {32, 64, 128}; // Convolutional layer channels
        std::vector<int> kernel_sizes = {3, 3, 3};      // Kernel sizes for each conv layer
        std::vector<int> strides = {1, 1, 1};           // Strides for each conv layer
        std::vector<int> fc_sizes = {256, 128};         // Fully connected layer sizes
        int output_size = 3;              // Number of output classes
        double dropout_rate = 0.3;        // Dropout probability
        bool use_batch_norm = true;       // Enable batch normalization
        std::string activation = "relu";  // Activation function
        int pool_size = 2;                // Max pooling size
    };

    /**
     * @brief Constructor
     * @param config Network configuration
     */
    explicit CNNNetwork(const Config& config);

    /**
     * @brief Forward pass
     * @param x Input tensor [batch_size, channels, sequence_length]
     * @return Output tensor [batch_size, output_size]
     */
    torch::Tensor forward(torch::Tensor x);

    /**
     * @brief Get model configuration
     * @return Current configuration
     */
    const Config& getConfig() const { return config_; }

    /**
     * @brief Save model weights and configuration
     * @param path File path to save model
     */
    void saveModel(const std::string& path);

    /**
     * @brief Load model weights and configuration
     * @param path File path to load model from
     */
    void loadModel(const std::string& path);

private:
    Config config_;
    torch::nn::ModuleList conv_layers_{nullptr};
    torch::nn::ModuleList batch_norms_{nullptr};
    torch::nn::ModuleList fc_layers_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    torch::nn::AdaptiveAvgPool1d global_pool_{nullptr};
    
    void buildNetwork();
    int calculateFcInputSize();
};

/**
 * @brief Long Short-Term Memory network for time series prediction
 * 
 * Implements LSTM architecture optimized for financial time series with
 * multi-layer stacking, dropout, and attention mechanisms for better
 * long-range dependency capture.
 */
class LSTMNetwork : public torch::nn::Module {
public:
    /**
     * @brief Configuration structure for LSTM parameters
     */
    struct Config {
        int input_size = 50;              // Number of input features
        int hidden_size = 128;            // LSTM hidden state size
        int num_layers = 2;               // Number of LSTM layers
        int output_size = 3;              // Number of output classes (buy/sell/hold)
        double dropout_rate = 0.2;       // Dropout probability
        bool use_batch_norm = true;       // Enable batch normalization
        bool use_attention = false;       // Enable attention mechanism
        bool bidirectional = false;       // Use bidirectional LSTM
        std::string activation = "tanh";  // Activation function (tanh, relu)
    };

    /**
     * @brief Constructor
     * @param config Network configuration
     */
    explicit LSTMNetwork(const Config& config);

    /**
     * @brief Forward pass
     * @param x Input tensor [batch_size, sequence_length, input_size]
     * @return Output tensor [batch_size, output_size]
     */
    torch::Tensor forward(torch::Tensor x);

    /**
     * @brief Get model configuration
     * @return Current configuration
     */
    const Config& getConfig() const { return config_; }

    /**
     * @brief Save model weights and configuration
     * @param path File path to save model
     */
    void saveModel(const std::string& path);

    /**
     * @brief Load model weights and configuration
     * @param path File path to load model from
     */
    void loadModel(const std::string& path);

private:
    Config config_;
    torch::nn::LSTM lstm_{nullptr};
    torch::nn::Linear output_layer_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    torch::nn::BatchNorm1d batch_norm_{nullptr};
    
    // Optional attention mechanism
    torch::nn::Linear attention_query_{nullptr};
    torch::nn::Linear attention_key_{nullptr};
    torch::nn::Linear attention_value_{nullptr};
    
    void buildNetwork();
    torch::Tensor applyAttention(torch::Tensor lstm_output);
};

/**
 * @brief Transformer network for time series prediction
 * 
 * Implements Transformer architecture optimized for financial time series with
 * multi-head attention, positional encoding, and adaptive tokenization for
 * capturing long-range dependencies and market patterns.
 */
class TransformerNetwork : public torch::nn::Module {
public:
    /**
     * @brief Configuration structure for Transformer parameters
     */
    struct Config {
        int input_size = 50;              // Number of input features
        int d_model = 256;               // Model dimension
        int nhead = 8;                   // Number of attention heads
        int num_encoder_layers = 6;       // Number of encoder layers
        int num_decoder_layers = 6;       // Number of decoder layers
        int dim_feedforward = 1024;       // Feed-forward dimension
        int output_size = 3;              // Number of output classes (buy/sell/hold)
        double dropout_rate = 0.1;        // Dropout probability
        int max_seq_length = 1000;        // Maximum sequence length
        bool use_positional_encoding = true; // Enable positional encoding
        bool use_layer_norm = true;        // Enable layer normalization
        std::string activation = "relu";  // Activation function
    };

    /**
     * @brief Constructor
     * @param config Network configuration
     */
    explicit TransformerNetwork(const Config& config);

    /**
     * @brief Forward pass
     * @param x Input tensor [batch_size, sequence_length, input_size]
     * @param tgt Target tensor [batch_size, target_length, input_size] (optional)
     * @return Output tensor [batch_size, output_size]
     */
    torch::Tensor forward(torch::Tensor x, torch::Tensor tgt = torch::Tensor());

    /**
     * @brief Get model configuration
     * @return Current configuration
     */
    const Config& getConfig() const { return config_; }

    /**
     * @brief Save model weights and configuration
     * @param path File path to save model
     */
    void saveModel(const std::string& path);

    /**
     * @brief Load model weights and configuration
     * @param path File path to load model from
     */
    void loadModel(const std::string& path);

private:
    Config config_;
    torch::nn::Transformer transformer_{nullptr};
    torch::nn::Linear input_projection_{nullptr};
    torch::nn::Linear output_projection_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    
    // Positional encoding
    torch::Tensor positional_encoding_;
    
    void buildNetwork();
    void createPositionalEncoding();
    torch::Tensor addPositionalEncoding(torch::Tensor x);
};

/**
 * @brief Hybrid model combining CNN and MLP architectures
 * 
 * Advanced model that uses CNN for pattern extraction followed by
 * MLP for final classification and confidence scoring.
 */
class HybridNetwork : public torch::nn::Module {
public:
    struct Config {
        CNNNetwork::Config cnn_config;
        MLPNetwork::Config mlp_config;
        bool use_attention = false;       // Enable attention mechanism
        double fusion_dropout = 0.1;      // Dropout for feature fusion
    };

    explicit HybridNetwork(const Config& config);
    torch::Tensor forward(torch::Tensor x);
    
    const Config& getConfig() const { return config_; }
    void saveModel(const std::string& path);
    void loadModel(const std::string& path);

private:
    Config config_;
    std::shared_ptr<CNNNetwork> cnn_backbone_;
    std::shared_ptr<MLPNetwork> mlp_head_;
    torch::nn::Linear fusion_layer_{nullptr};
    torch::nn::Dropout fusion_dropout_{nullptr};
};

/**
 * @brief Model factory for creating neural network instances
 */
class ModelFactory {
public:
    enum class ModelType {
        MLP,
        CNN,
        LSTM,
        TRANSFORMER,
        HYBRID
    };

    /**
     * @brief Create a neural network model
     * @param type Model type
     * @param config_path Path to configuration file
     * @return Shared pointer to created model
     */
    static std::shared_ptr<torch::nn::Module> createModel(
        ModelType type, 
        const std::string& config_path = ""
    );

    /**
     * @brief Load model from saved state
     * @param model_path Path to saved model
     * @return Shared pointer to loaded model
     */
    static std::shared_ptr<torch::nn::Module> loadModel(const std::string& model_path);
};

/**
 * @brief Training utilities and metrics
 */
class TrainingUtils {
public:
    struct TrainingMetrics {
        double loss = 0.0;
        double accuracy = 0.0;
        double precision = 0.0;
        double recall = 0.0;
        double f1_score = 0.0;
        std::vector<double> class_accuracies;
    };

    /**
     * @brief Calculate training metrics
     * @param predictions Model predictions
     * @param targets Ground truth labels
     * @return Training metrics
     */
    static TrainingMetrics calculateMetrics(
        const torch::Tensor& predictions,
        const torch::Tensor& targets
    );

    /**
     * @brief Apply learning rate scheduling
     * @param optimizer PyTorch optimizer
     * @param epoch Current epoch
     * @param schedule_type Schedule type ("step", "cosine", "exponential")
     */
    static void applyLRSchedule(
        torch::optim::Optimizer& optimizer,
        int epoch,
        const std::string& schedule_type = "step"
    );

    /**
     * @brief Early stopping checker
     * @param current_loss Current validation loss
     * @param patience Number of epochs to wait
     * @return True if training should stop
     */
    static bool checkEarlyStopping(
        double current_loss,
        int patience = 10
    );
};

} // namespace Models
} // namespace ArchNeuronX

#endif // NEURAL_NETWORKS_H