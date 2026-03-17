#pragma once

#include <memory>
#include <vector>
#include <string>
#include <torch/torch.h>

namespace ArchNeuronX {

class NeuralModel {
public:
    virtual ~NeuralModel() = default;
    virtual torch::Tensor forward(const torch::Tensor& input) = 0;
    virtual void train(const torch::Tensor& input, const torch::Tensor& target) = 0;
    virtual void save(const std::string& path) = 0;
    virtual void load(const std::string& path) = 0;
};

struct TradingSignal {
    enum class Action { BUY, SELL, HOLD };
    
    Action action;
    double confidence;
    double price_target;
    std::string symbol;
    std::string timestamp;
    std::string explanation;
};

class TradingEngine {
public:
    TradingEngine();
    ~TradingEngine();
    
    // Model management
    void loadModel(const std::string& model_path);
    void trainModel(const std::string& data_path, const std::string& config_path);
    
    // Signal generation
    std::vector<TradingSignal> generateSignals(const torch::Tensor& market_data);
    TradingSignal predictSingle(const torch::Tensor& input);
    
    // Performance metrics
    struct Metrics {
        double accuracy;
        double precision;
        double recall;
        double f1_score;
        double sharpe_ratio;
        double max_drawdown;
    };
    
    Metrics evaluateModel(const torch::Tensor& test_data, const torch::Tensor& test_labels);
    
private:
    std::unique_ptr<NeuralModel> model_;
    bool gpu_enabled_;
    torch::Device device_;
};

} // namespace ArchNeuronX