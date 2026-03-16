#pragma once
// ============================================================
// ArchNeuronX v2 - Model Ensemble with Dynamic Weighting
// Combines MLP, CNN, TFT with performance-based weights
// Strategy: rolling accuracy-weighted voting
// ============================================================
#include <torch/torch.h>
#include <vector>
#include <string>
#include <deque>
#include <memory>

namespace archneuronx {
namespace models {

struct ModelWeight {
    std::string name;
    double weight;          // Current weight (sum to 1.0)
    double rolling_accuracy; // Accuracy on last N predictions
    int64_t correct_count;
    int64_t total_count;
};

class EnsembleModel {
public:
    explicit EnsembleModel(int window_size = 100);

    // Add a model to the ensemble
    void add_model(const std::string& name,
                   torch::jit::script::Module model,
                   double initial_weight = 1.0);

    // Predict: weighted average of all models
    // Returns probabilities [3]: BUY, SELL, HOLD
    [[nodiscard]] torch::Tensor predict(
        const torch::Tensor& temporal_input,
        const torch::Tensor& static_input,
        const torch::Device& device);

    // Update weights based on prediction outcome
    // Call after trade result is known
    void update_weights(const std::string& model_name, bool correct);

    // Rebalance all weights based on rolling accuracy
    void rebalance_weights();

    // Get current weight distribution
    [[nodiscard]] std::vector<ModelWeight> get_weights() const;

    // A/B testing: compare two specific models
    [[nodiscard]] std::pair<double, double>
    ab_test(const std::string& model_a,
             const std::string& model_b) const;

private:
    struct ModelEntry {
        std::string name;
        torch::jit::script::Module module;
        ModelWeight weight_info;
        std::deque<bool> recent_predictions;  // Ring buffer
    };

    std::vector<ModelEntry> models_;
    int window_size_;

    void normalize_weights();
};

} // namespace models
} // namespace archneuronx
