/**
 * @file regime_aware_example.cpp
 * @brief Example demonstrating regime-aware ensemble system
 * @author ArchNeuronX Team
 * @date 2026-03-17
 */

#include "models/regime_aware_ensemble.hpp"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace archneuronx;
using namespace archneuronx::models;
using namespace archneuronx::regime;

// Helper function to generate synthetic market data
std::pair<std::vector<double>, std::vector<double>> generate_synthetic_data(
    int num_points, 
    MarketRegime regime,
    double noise_level = 0.01) {
    
    std::vector<double> prices, volumes;
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, noise_level);
    
    double base_price = 100.0;
    double base_volume = 1000.0;
    
    for (int i = 0; i < num_points; ++i) {
        double trend = 0.0;
        double volatility = noise_level;
        
        switch (regime) {
            case MarketRegime::BULL_LOW_VOL:
                trend = 0.001;
                volatility = 0.005;
                break;
            case MarketRegime::BULL_HIGH_VOL:
                trend = 0.001;
                volatility = 0.02;
                break;
            case MarketRegime::BEAR_LOW_VOL:
                trend = -0.0008;
                volatility = 0.005;
                break;
            case MarketRegime::BEAR_HIGH_VOL:
                trend = -0.0008;
                volatility = 0.025;
                break;
            case MarketRegime::SIDEWAYS_LOW_VOL:
                trend = 0.0;
                volatility = 0.003;
                break;
            case MarketRegime::SIDEWAYS_HIGH_VOL:
                trend = 0.0;
                volatility = 0.015;
                break;
            default:
                break;
        }
        
        std::normal_distribution<double> price_change(trend, volatility);
        base_price *= (1.0 + price_change(rng));
        base_volume *= (1.0 + noise(rng));
        
        prices.push_back(base_price);
        volumes.push_back(std::max(100.0, base_volume));
    }
    
    return {prices, volumes};
}

// Helper function to create a simple neural model
torch::jit::script::Module create_simple_model(const std::string& model_name) {
    // Create a simple MLP using TorchScript
    std::string model_def = R"(
        import torch
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self, input_size=50, hidden_size=64, output_size=3):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, temporal, static):
                x = torch.relu(self.fc1(temporal))
                x = self.dropout(x)
                x = torch.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        # Create model instance
        model = SimpleModel()
        model.eval()
        
        # Trace the model
        example_temporal = torch.randn(1, 50)
        example_static = torch.randn(1, 10)
        traced_model = torch.jit.trace(model, (example_temporal, example_static))
        
        # Save the traced model
        traced_model.save("simple_model.pt")
    )";
    
    // For this example, we'll create a dummy module
    // In practice, you would load a pre-trained model
    return torch::jit::compile("def forward(temporal, static): return torch.randn(1, 3)");
}

void demonstrate_regime_aware_ensemble() {
    std::cout << "=== Regime-Aware Ensemble Demonstration ===" << std::endl;
    
    // Configuration
    RegimeEnsembleConfig ensemble_config;
    ensemble_config.adaptation_rate = 0.15;
    ensemble_config.min_weight_threshold = 0.05;
    ensemble_config.regime_boost_factor = 1.5;
    ensemble_config.enable_regime_diversification = true;
    
    RegimeConfig regime_config;
    regime_config.price_window = 60;
    regime_config.volatility_window = 20;
    regime_config.use_ml_classifier = false; // Use statistical for simplicity
    
    // Create ensemble
    RegimeAwareEnsemble ensemble(ensemble_config, regime_config);
    
    if (!ensemble.initialize()) {
        std::cerr << "Failed to initialize ensemble" << std::endl;
        return;
    }
    
    // Create models with regime-specific configurations
    std::vector<std::string> model_names = {"MLP_Model", "CNN_Model", "LSTM_Model"};
    std::vector<MarketRegime> regimes = {
        MarketRegime::BULL_LOW_VOL, MarketRegime::BULL_HIGH_VOL,
        MarketRegime::BEAR_LOW_VOL, MarketRegime::BEAR_HIGH_VOL,
        MarketRegime::SIDEWAYS_LOW_VOL, MarketRegime::SIDEWAYS_HIGH_VOL
    };
    
    for (const auto& model_name : model_names) {
        auto model = create_simple_model(model_name);
        
        std::unordered_map<MarketRegime, RegimeModelConfig> regime_configs;
        
        // Configure model strengths for different regimes
        for (auto regime : regimes) {
            RegimeModelConfig config;
            config.model_name = model_name;
            config.is_active = true;
            config.regime_specific_accuracy = 0.5;
            
            // Give each model different strengths
            if (model_name == "MLP_Model") {
                config.base_weight = (regime == MarketRegime::BULL_LOW_VOL || regime == MarketRegime::BEAR_LOW_VOL) ? 1.2 : 0.8;
                config.performance_multiplier = 1.0;
            } else if (model_name == "CNN_Model") {
                config.base_weight = (regime == MarketRegime::BULL_HIGH_VOL || regime == MarketRegime::BEAR_HIGH_VOL) ? 1.3 : 0.7;
                config.performance_multiplier = 1.1;
            } else { // LSTM_Model
                config.base_weight = (regime == MarketRegime::SIDEWAYS_LOW_VOL || regime == MarketRegime::SIDEWAYS_HIGH_VOL) ? 1.4 : 0.6;
                config.performance_multiplier = 1.2;
            }
            
            regime_configs[regime] = config;
        }
        
        ensemble.add_model_with_regime_config(model_name, model, regime_configs);
    }
    
    std::cout << "Added " << model_names.size() << " models with regime-specific configurations" << std::endl;
    
    // Simulate trading across different regimes
    torch::Device device(torch::kCPU);
    std::vector<MarketRegime> regime_sequence = {
        MarketRegime::BULL_LOW_VOL, MarketRegime::BULL_LOW_VOL, MarketRegime::BULL_HIGH_VOL,
        MarketRegime::TRANSITION, MarketRegime::BEAR_HIGH_VOL, MarketRegime::BEAR_LOW_VOL,
        MarketRegime::SIDEWAYS_LOW_VOL, MarketRegime::SIDEWAYS_HIGH_VOL, MarketRegime::TRANSITION,
        MarketRegime::BULL_LOW_VOL
    };
    
    int total_predictions = 0;
    int correct_predictions = 0;
    
    for (size_t regime_idx = 0; regime_idx < regime_sequence.size(); ++regime_idx) {
        auto current_regime = regime_sequence[regime_idx];
        
        std::cout << "\n--- Testing Regime: " << static_cast<int>(current_regime) << " ---" << std::endl;
        
        // Generate data for current regime
        auto [prices, volumes] = generate_synthetic_data(100, current_regime);
        
        // Create dummy input tensors
        auto temporal_input = torch::randn({1, 50});
        auto static_input = torch::randn({1, 10});
        
        // Get regime-aware prediction
        auto prediction = ensemble.predict_regime_aware(
            temporal_input, static_input, device, prices, volumes
        );
        
        // Get current regime info
        auto regime_result = ensemble.get_current_regime();
        auto metrics = ensemble.get_metrics();
        
        std::cout << "Current regime: " << static_cast<int>(regime_result.regime) << std::endl;
        std::cout << "Regime confidence: " << regime_result.confidence << std::endl;
        std::cout << "Is transition: " << (regime_result.is_transition ? "Yes" : "No") << std::endl;
        std::cout << "Overall accuracy: " << metrics.overall_accuracy << std::endl;
        std::cout << "Weight entropy: " << metrics.weight_entropy << std::endl;
        std::cout << "Regime stability: " << metrics.regime_stability_score << std::endl;
        
        // Show best models for current regime
        auto best_models = ensemble.get_best_models_for_regime(current_regime);
        std::cout << "Best models for this regime: ";
        for (const auto& model : best_models) {
            std::cout << model << " ";
        }
        std::cout << std::endl;
        
        // Simulate some prediction outcomes
        for (int i = 0; i < 10; ++i) {
            // Simulate prediction accuracy based on regime and model
            bool correct = (std::rand() % 100) < 60; // 60% accuracy simulation
            
            // Update ensemble with performance
            std::string model_name = model_names[std::rand() % model_names.size()];
            ensemble.update_performance_regime_aware(model_name, correct, current_regime);
            
            total_predictions++;
            if (correct) correct_predictions++;
        }
        
        // Check for overfitting
        if (ensemble.is_overfitting_detected()) {
            std::cout << "⚠️  Overfitting detected! Risk: " << ensemble.calculate_overfitting_risk() << std::endl;
            ensemble.apply_overfitting_mitigation();
        }
        
        // Show regime performance
        auto regime_performance = ensemble.get_regime_performance();
        std::cout << "Regime-specific accuracies:" << std::endl;
        for (const auto& [regime, accuracy] : regime_performance) {
            if (accuracy > 0.0) {
                std::cout << "  Regime " << static_cast<int>(regime) << ": " << accuracy << std::endl;
            }
        }
    }
    
    // Final summary
    std::cout << "\n=== Final Summary ===" << std::endl;
    auto final_metrics = ensemble.get_metrics();
    std::cout << "Total predictions: " << total_predictions << std::endl;
    std::cout << "Correct predictions: " << correct_predictions << std::endl;
    std::cout << "Overall accuracy: " << (correct_predictions / static_cast<double>(total_predictions)) << std::endl;
    std::cout << "Final ensemble accuracy: " << final_metrics.overall_accuracy << std::endl;
    std::cout << "Regime switches: " << final_metrics.regime_switches << std::endl;
    std::cout << "Weight entropy: " << final_metrics.weight_entropy << std::endl;
    std::cout << "Regime stability: " << final_metrics.regime_stability_score << std::endl;
    std::cout << "Overfitting risk: " << ensemble.calculate_overfitting_risk() << std::endl;
    
    std::cout << "\n✅ Regime-aware ensemble demonstration completed successfully!" << std::endl;
}

int main() {
    try {
        demonstrate_regime_aware_ensemble();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
