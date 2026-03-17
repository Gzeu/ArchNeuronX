/**
 * @file ensemble.cpp
 * @brief Ensemble model implementation with dynamic weighting
 * @author George Pricop
 * @date 2025-10-02
 */

#include "models/ensemble.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace archneuronx {
namespace models {

EnsembleModel::EnsembleModel(int window_size) : window_size_(window_size) {
}

void EnsembleModel::add_model(const std::string& name,
                             torch::jit::script::Module model,
                             double initial_weight) {
    ModelEntry entry;
    entry.name = name;
    entry.module = model;
    entry.weight_info.name = name;
    entry.weight_info.weight = initial_weight;
    entry.weight_info.rolling_accuracy = 0.0;
    entry.weight_info.correct_count = 0;
    entry.weight_info.total_count = 0;
    
    models_.push_back(entry);
    normalize_weights();
    
    std::cout << "Added model '" << name << "' with initial weight " 
              << initial_weight << std::endl;
}

torch::Tensor EnsembleModel::predict(const torch::Tensor& temporal_input,
                                   const torch::Tensor& static_input,
                                   const torch::Device& device) {
    if (models_.empty()) {
        throw std::runtime_error("No models in ensemble");
    }
    
    std::vector<torch::Tensor> predictions;
    std::vector<double> weights;
    
    // Get predictions from all models
    for (const auto& model_entry : models_) {
        try {
            // Move inputs to correct device
            auto temp_input = temporal_input.to(device);
            auto stat_input = static_input.to(device);
            
            // Create input tuple for model
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(temp_input);
            inputs.push_back(stat_input);
            
            // Get model prediction
            auto output = model_entry.module.forward(inputs);
            auto prediction = output.toTensor();
            
            // Apply softmax to get probabilities
            prediction = torch::softmax(prediction, -1);
            
            predictions.push_back(prediction);
            weights.push_back(model_entry.weight_info.weight);
            
        } catch (const std::exception& e) {
            std::cerr << "Error in model '" << model_entry.name 
                      << "': " << e.what() << std::endl;
            // Skip this model and continue
            continue;
        }
    }
    
    if (predictions.empty()) {
        throw std::runtime_error("All models failed to predict");
    }
    
    // Weighted average of predictions
    torch::Tensor ensemble_prediction = torch::zeros_like(predictions[0]);
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        ensemble_prediction += predictions[i] * weights[i];
    }
    
    // Final softmax to ensure valid probabilities
    ensemble_prediction = torch::softmax(ensemble_prediction, -1);
    
    return ensemble_prediction;
}

void EnsembleModel::update_weights(const std::string& model_name, bool correct) {
    // Find the model
    auto it = std::find_if(models_.begin(), models_.end(),
                         [&model_name](const ModelEntry& entry) {
                             return entry.name == model_name;
                         });
    
    if (it == models_.end()) {
        std::cerr << "Model '" << model_name << "' not found in ensemble" << std::endl;
        return;
    }
    
    // Update prediction history
    it->recent_predictions.push_back(correct);
    if (it->recent_predictions.size() > window_size_) {
        it->recent_predictions.pop_front();
    }
    
    // Update statistics
    it->weight_info.correct_count += correct ? 1 : 0;
    it->weight_info.total_count += 1;
    
    // Calculate rolling accuracy
    if (!it->recent_predictions.empty()) {
        int correct_count = std::count(it->recent_predictions.begin(),
                                     it->recent_predictions.end(), true);
        it->weight_info.rolling_accuracy = 
            static_cast<double>(correct_count) / it->recent_predictions.size();
    }
    
    // Rebalance weights periodically
    if (it->weight_info.total_count % 10 == 0) {
        rebalance_weights();
    }
}

void EnsembleModel::rebalance_weights() {
    if (models_.size() < 2) {
        return; // No rebalancing needed for single model
    }
    
    // Calculate new weights based on rolling accuracy
    std::vector<double> new_weights;
    double total_accuracy = 0.0;
    
    for (const auto& model_entry : models_) {
        double accuracy = model_entry.weight_info.rolling_accuracy;
        
        // Apply smoothing to avoid zero weights
        double smoothed_accuracy = std::max(accuracy, 0.1);
        new_weights.push_back(smoothed_accuracy);
        total_accuracy += smoothed_accuracy;
    }
    
    // Normalize weights to sum to 1.0
    if (total_accuracy > 0.0) {
        for (size_t i = 0; i < models_.size(); ++i) {
            models_[i].weight_info.weight = new_weights[i] / total_accuracy;
        }
    }
    
    normalize_weights();
    
    // Print weight distribution
    std::cout << "Rebalanced ensemble weights:" << std::endl;
    for (const auto& model_entry : models_) {
        std::cout << "  " << model_entry.name << ": " 
                  << model_entry.weight_info.weight << " (accuracy: "
                  << model_entry.weight_info.rolling_accuracy << ")" << std::endl;
    }
}

std::vector<ModelWeight> EnsembleModel::get_weights() const {
    std::vector<ModelWeight> weights;
    for (const auto& model_entry : models_) {
        weights.push_back(model_entry.weight_info);
    }
    return weights;
}

std::pair<double, double> EnsembleModel::ab_test(const std::string& model_a,
                                                const std::string& model_b) const {
    auto it_a = std::find_if(models_.begin(), models_.end(),
                            [&model_a](const ModelEntry& entry) {
                                return entry.name == model_a;
                            });
    
    auto it_b = std::find_if(models_.begin(), models_.end(),
                            [&model_b](const ModelEntry& entry) {
                                return entry.name == model_b;
                            });
    
    if (it_a == models_.end() || it_b == models_.end()) {
        throw std::runtime_error("One or both models not found for A/B test");
    }
    
    double accuracy_a = it_a->weight_info.rolling_accuracy;
    double accuracy_b = it_b->weight_info.rolling_accuracy;
    
    // Calculate statistical significance (simplified)
    double n_a = static_cast<double>(it_a->recent_predictions.size());
    double n_b = static_cast<double>(it_b->recent_predictions.size());
    
    if (n_a < 10 || n_b < 10) {
        return {accuracy_a, accuracy_b}; // Insufficient data
    }
    
    // Standard error for proportion
    double se_a = std::sqrt((accuracy_a * (1 - accuracy_a)) / n_a);
    double se_b = std::sqrt((accuracy_b * (1 - accuracy_b)) / n_b);
    
    // Z-score for difference
    double z_score = (accuracy_a - accuracy_b) / std::sqrt(se_a * se_a + se_b * se_b);
    
    // P-value (two-tailed test)
    double p_value = 2.0 * (1.0 - std::abs(std::erf(z_score / std::sqrt(2.0))));
    
    std::cout << "A/B Test Results:" << std::endl;
    std::cout << "  " << model_a << ": " << accuracy_a << " (n=" << n_a << ")" << std::endl;
    std::cout << "  " << model_b << ": " << accuracy_b << " (n=" << n_b << ")" << std::endl;
    std::cout << "  Z-score: " << z_score << ", p-value: " << p_value << std::endl;
    
    return {accuracy_a, accuracy_b};
}

void EnsembleModel::normalize_weights() {
    if (models_.empty()) {
        return;
    }
    
    double total_weight = 0.0;
    for (const auto& model_entry : models_) {
        total_weight += model_entry.weight_info.weight;
    }
    
    if (total_weight > 0.0) {
        for (auto& model_entry : models_) {
            model_entry.weight_info.weight /= total_weight;
        }
    } else {
        // Equal weights if all are zero
        double equal_weight = 1.0 / models_.size();
        for (auto& model_entry : models_) {
            model_entry.weight_info.weight = equal_weight;
        }
    }
}

} // namespace models
} // namespace archneuronx
