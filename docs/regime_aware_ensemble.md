# Regime-Aware Ensemble System

## Overview

The Regime-Aware Ensemble System is a core component of ArchNeuronX v3.0 "Elite" that addresses the #1 survival factor for neural trading systems: **anti-overfitting through regime specialization**.

### Key Features

- **Real-time Market Regime Detection**: Bull/Bear/Sideways + Volatility classification
- **Regime-Specific Model Weighting**: Different models excel in different market conditions
- **Dynamic Ensemble Adaptation**: Weights automatically adjust based on regime changes
- **Overfitting Detection & Mitigation**: Built-in safeguards against over-optimization
- **Performance Tracking per Regime**: Detailed analytics for each market condition

## Architecture

### Core Components

1. **RegimeDetector** (`src/regime/regime_detector.cpp`)
   - Statistical + ML-based regime classification
   - 8 market regimes: Bull/Bear/Sideways × Low/High volatility + Transition/Unknown
   - Real-time feature extraction from price/volume data

2. **RegimeAwareEnsemble** (`src/models/regime_aware_ensemble.cpp`)
   - Extends base EnsembleModel with regime awareness
   - Maintains separate model weights per regime
   - Automatic weight adaptation and diversification

### Market Regimes

| Regime | Description | Typical Characteristics |
|--------|-------------|------------------------|
| BULL_LOW_VOL | Rising market, low volatility | Steady uptrend, predictable |
| BULL_HIGH_VOL | Rising market, high volatility | Volatile uptrend, momentum |
| BEAR_LOW_VOL | Falling market, low volatility | Steady downtrend, predictable |
| BEAR_HIGH_VOL | Falling market, high volatility | Volatile downtrend, panic |
| SIDEWAYS_LOW_VOL | Range-bound, low volatility | Consolidation, mean reversion |
| SIDEWAYS_HIGH_VOL | Range-bound, high volatility | Choppy, uncertain |
| TRANSITION | Regime change period | High entropy, unstable |
| UNKNOWN | Unable to classify | Insufficient data |

## Quick Start

### Building

```bash
# Build the regime-aware example
cmake -B build -S .
cmake --build build --target regime_aware_example

# Run the demonstration
./build/regime_aware_example
```

### Basic Usage

```cpp
#include "models/regime_aware_ensemble.hpp"

// Configure ensemble
RegimeEnsembleConfig config;
config.adaptation_rate = 0.15;
config.enable_regime_diversification = true;

// Configure regime detector
RegimeConfig regime_config;
regime_config.price_window = 60;
regime_config.use_ml_classifier = false;

// Create ensemble
RegimeAwareEnsemble ensemble(config, regime_config);
ensemble.initialize();

// Add models with regime-specific configurations
std::unordered_map<MarketRegime, RegimeModelConfig> regime_configs;
// ... configure for each regime ...

ensemble.add_model_with_regime_config("MLP_Model", model, regime_configs);

// Make regime-aware predictions
auto prediction = ensemble.predict_regime_aware(
    temporal_input, static_input, device, prices, volumes
);
```

## Configuration

### RegimeEnsembleConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `adaptation_rate` | 0.1 | How fast weights adapt to new regime |
| `min_weight_threshold` | 0.05 | Minimum weight any model can have |
| `regime_boost_factor` | 1.5 | Performance boost for regime-specialized models |
| `max_regime_concentration` | 0.6 | Maximum weight concentration in one regime |
| `enable_regime_diversification` | true | Force diversification across regimes |

### RegimeConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `price_window` | 60 | Price history window for feature extraction |
| `volume_window` | 30 | Volume history window |
| `volatility_window` | 20 | Volatility calculation window |
| `trend_threshold` | 0.02 | Trend strength threshold (2%) |
| `volatility_threshold` | 0.015 | Volatility threshold (1.5%) |
| `use_ml_classifier` | true | Use neural classifier vs statistical only |

## Anti-Overfitting Features

### 1. Regime Diversification
- Prevents concentration in single market condition
- Forces models to perform across multiple regimes
- Reduces overfitting to specific market patterns

### 2. Dynamic Weight Adaptation
- Gradual weight changes prevent sudden overreactions
- Performance-based weighting with smoothing
- Automatic rebalancing based on regime-specific accuracy

### 3. Overfitting Detection
- **Weight Entropy**: Low entropy indicates over-concentration
- **Regime Correlation**: High correlation with regime changes
- **Performance Degradation**: Declining accuracy over time

### 4. Automatic Mitigation
- Reduces maximum concentration limits
- Increases adaptation rate for faster recovery
- Applies regularization to overperforming models

## Performance Monitoring

### Key Metrics

```cpp
auto metrics = ensemble.get_metrics();
std::cout << "Overall accuracy: " << metrics.overall_accuracy << std::endl;
std::cout << "Weight entropy: " << metrics.weight_entropy << std::endl;
std::cout << "Regime stability: " << metrics.regime_stability_score << std::endl;
std::cout << "Regime switches: " << metrics.regime_switches << std::endl;
```

### Regime-Specific Performance

```cpp
auto performance = ensemble.get_regime_performance();
for (const auto& [regime, accuracy] : performance) {
    std::cout << "Regime " << regime << ": " << accuracy << std::endl;
}
```

### Best Models per Regime

```cpp
auto best_models = ensemble.get_best_models_for_regime(MarketRegime::BULL_LOW_VOL);
for (const auto& model : best_models) {
    std::cout << "Top model: " << model << std::endl;
}
```

## Advanced Usage

### Custom Regime Classifier

Train your own regime classifier:

```cpp
// Collect historical features and labels
std::vector<MarketFeatures> features;
std::vector<MarketRegime> labels;

// Train classifier
regime_detector.train_classifier(features, labels);
```

### Export/Import Configurations

```cpp
// Export regime configurations
ensemble.export_regime_config("regime_configs.json");

// Import configurations
ensemble.import_regime_config("regime_configs.json");
```

### Optimization for Specific Regime

```cpp
// Optimize ensemble for bull markets
ensemble.optimize_for_regime(MarketRegime::BULL_LOW_VOL, 100);
```

## Integration with Existing Systems

### OpenCLaw Integration

```cpp
// Update ensemble with live market data
ensemble.update_with_market_data(prices, volumes, timestamp);

// Get regime-aware prediction for trading
auto prediction = ensemble.predict_regime_aware(
    temporal_features, static_features, device, prices, volumes
);

// Use prediction in OpenCLaw execution
if (prediction[0].item<float>() > 0.6) { // BUY signal
    openclaw_agent.execute_order("BUY", size, price);
}
```

### Risk Management Integration

```cpp
// Check overfitting risk before trading
if (ensemble.is_overfitting_detected()) {
    std::cout << "High overfitting risk, reducing position size" << std::endl;
    position_size *= 0.5;
}

// Consider regime stability in risk calculations
auto stability = ensemble.get_regime_stability();
if (stability < 0.3) { // Unstable regime
    max_drawdown_limit *= 0.7; // Reduce risk
}
```

## Best Practices

### 1. Model Configuration
- Give each model specific regime strengths
- Avoid making all models good at everything
- Use domain knowledge for regime specialization

### 2. Parameter Tuning
- Start with conservative adaptation rates (0.1-0.2)
- Enable diversification in production
- Monitor overfitting risk closely

### 3. Performance Monitoring
- Track regime-specific accuracy
- Watch for sudden weight concentration
- Monitor regime stability score

### 4. Risk Management
- Reduce position size during transitions
- Increase thresholds during high volatility
- Always have overfitting mitigation enabled

## Troubleshooting

### Low Accuracy in Specific Regime
1. Check if models are configured for that regime
2. Verify regime detection accuracy
3. Consider adding regime-specific models

### High Overfitting Risk
1. Enable regime diversification
2. Reduce max concentration limits
3. Increase adaptation rate
4. Add more diverse models

### Frequent Regime Switches
1. Increase regime detection thresholds
2. Use longer windows for feature extraction
3. Enable ML classifier for smoother transitions

## Future Enhancements

- **Temporal Fusion Transformer (TFT)**: Advanced time-series modeling
- **Meta-Learning**: Fast adaptation to new regimes
- **Graph Neural Networks**: Asset correlation modeling
- **Multi-Asset Regime Detection**: Cross-asset regime analysis

## References

1. "Market Regime Detection Using Machine Learning" - Journal of Financial Data Science
2. "Ensemble Methods for Financial Time Series" - IEEE Transactions on Neural Networks
3. "Anti-Overfitting Strategies in Neural Trading Systems" - Quantitative Finance

---

⚠️ **Financial Risk Warning**: This system is designed for research and paper trading only. Never risk real money without extensive backtesting and validation. Past performance does not guarantee future results.
