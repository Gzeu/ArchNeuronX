// ============================================================
// ArchNeuronX v4.0 - Regime Meta-Learner
// Fast adaptation to market changes using Model-Agnostic Meta-Learning
// ============================================================

#pragma once

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <queue>
#include <random>
#include <algorithm>

namespace archneuronx {
namespace models {
namespace v4 {

// Forward declarations
struct MarketRegime;
struct BaseModel;
struct AdaptedModel;
struct MarketExperience;
struct TaskDistribution;

// ============================================================
// MAML Optimizer - Model-Agnostic Meta-Learning
// ============================================================

class MAMLOptimizerImpl : public torch::nn::Module {
private:
    double inner_lr_;      // Inner loop learning rate
    double outer_lr_;      // Outer loop learning rate
    int64_t inner_steps_;  // Number of inner optimization steps
    int64_t meta_batch_size_;  // Meta batch size
    
    // Meta-parameters
    std::vector<torch::Tensor> meta_parameters_;
    std::vector<torch::Tensor> fast_parameters_;
    
    // Optimization history
    std::vector<std::vector<torch::Tensor>> adaptation_history_;
    std::vector<double> meta_loss_history_;
    
    // Task distribution
    std::shared_ptr<TaskDistribution> task_distribution_;
    
    // Performance tracking
    double average_adaptation_time_ms_;
    double average_adaptation_loss_;
    int64_t successful_adaptations_;

public:
    MAMLOptimizerImpl(
        double inner_lr = 0.01,
        double outer_lr = 0.001,
        int64_t inner_steps = 5,
        int64_t meta_batch_size = 10
    );
    
    // Initialize meta-parameters
    void initialize_meta_parameters(const std::vector<torch::Tensor>& initial_params);
    
    // Adapt model to new task
    AdaptedModel adapt_to_task(
        const BaseModel& base_model,
        const MarketExperience& experience,
        int64_t adaptation_steps = 5
    );
    
    // Meta-training step
    void meta_train_step(const std::vector<MarketExperience>& batch);
    
    // Get adaptation performance
    double get_adaptation_performance(const AdaptedModel& adapted_model, const MarketExperience& test_experience);
    
    // Update meta-parameters
    void update_meta_parameters(const std::vector<torch::Tensor>& gradients);
    
    // Performance metrics
    double get_average_adaptation_time() const { return average_adaptation_time_ms_; }
    double get_average_adaptation_loss() const { return average_adaptation_loss_; }
    int64_t get_successful_adaptations() const { return successful_adaptations_; }

private:
    // Inner loop optimization
    std::vector<torch::Tensor> inner_loop_optimization(
        const std::vector<torch::Tensor>& parameters,
        const MarketExperience& experience,
        int64_t steps
    );
    
    // Compute meta-gradient
    std::vector<torch::Tensor> compute_meta_gradient(
        const std::vector<torch::Tensor>& meta_parameters,
        const std::vector<MarketExperience>& batch
    );
    
    // Sample tasks for meta-training
    std::vector<MarketExperience> sample_tasks(int64_t batch_size);
};

TORCH_MODULE(MAMLOptimizer);

// ============================================================
// Fast Adaptation Network
// ============================================================

class FastAdaptationNetworkImpl : public torch::nn::Module {
private:
    int64_t input_dim_;
    int64_t hidden_dim_;
    int64_t output_dim_;
    int64_t adaptation_layers_;
    double adaptation_lr_;
    
    // Base network
    torch::nn::Linear base_fc1_;
    torch::nn::Linear base_fc2_;
    torch::nn::Linear base_fc3_;
    torch::nn::Linear base_output_;
    
    // Adaptation layers (fast weights)
    std::vector<torch::nn::Linear> adaptation_layers_list_;
    std::vector<torch::Tensor> fast_weights_;
    
    // Activations
    torch::nn::ReLU relu_;
    torch::nn::Tanh tanh_;
    torch::nn::Dropout dropout_;
    
    // Adaptation state
    bool is_adapted_;
    std::vector<torch::Tensor> current_adapted_weights_;

public:
    FastAdaptationNetworkImpl(
        int64_t input_dim = 64,
        int64_t hidden_dim = 128,
        int64_t output_dim = 3,  // BUY/SELL/HOLD
        int64_t adaptation_layers = 2,
        double adaptation_lr = 0.01
    );
    
    // Forward pass through base network
    torch::Tensor forward_base(const torch::Tensor& input);
    
    // Forward pass through adapted network
    torch::Tensor forward_adapted(const torch::Tensor& input);
    
    // Adapt network to new regime
    void adapt_to_regime(const MarketRegime& regime, const torch::Tensor& adaptation_data);
    
    // Reset to base parameters
    void reset_to_base();
    
    // Get current adaptation state
    bool is_adapted() const { return is_adapted_; }
    std::vector<torch::Tensor> get_adapted_weights() const { return current_adapted_weights_; }
    
    // Fast adaptation with few examples
    void few_shot_adaptation(const std::vector<torch::Tensor>& examples, const std::vector<torch::Tensor>& labels);
    
    // Compute adaptation loss
    torch::Tensor compute_adaptation_loss(const torch::Tensor& predictions, const torch::Tensor& targets);

private:
    // Initialize fast weights
    void initialize_fast_weights();
    
    // Update fast weights
    void update_fast_weights(const std::vector<torch::Tensor>& gradients);
    
    // Apply adaptation to specific layers
    void apply_adaptation_to_layer(int64_t layer_idx, const torch::Tensor& gradient);
};

TORCH_MODULE(FastAdaptationNetwork);

// ============================================================
// Regime Detection Ensemble
// ============================================================

class RegimeDetectionEnsembleImpl : public torch::nn::Module {
private:
    int64_t num_models_;
    int64_t input_dim_;
    int64_t num_regimes_;
    double ensemble_weight_;
    
    // Ensemble of regime detectors
    std::vector<torch::nn::Linear> regime_detectors_;
    std::vector<torch::nn::LSTM> lstm_detectors_;
    std::vector<torch::nn::Conv1d> conv_detectors_;
    
    // Model weights for ensemble
    torch::Tensor ensemble_weights_;
    torch::Tensor model_confidence_;
    
    // Regime characteristics
    std::vector<MarketRegime> regime_definitions_;
    std::unordered_map<std::string, int64_t> regime_name_to_id_;
    
    // Detection history
    std::queue<MarketRegime> recent_regimes_;
    std::queue<std::vector<double>> regime_probabilities_;
    int64_t history_size_;

public:
    RegimeDetectionEnsembleImpl(
        int64_t num_models = 5,
        int64_t input_dim = 64,
        int64_t num_regimes = 8
    );
    
    // Detect current market regime
    std::pair<MarketRegime, std::vector<double>> detect_regime(const torch::Tensor& market_features);
    
    // Update ensemble weights based on performance
    void update_ensemble_weights(const MarketRegime& true_regime, const std::vector<double>& predicted_probs);
    
    // Add new regime detector to ensemble
    void add_detector(torch::nn::Module detector);
    
    // Get regime transition probabilities
    std::vector<std::vector<double>> get_transition_probabilities();
    
    // Predict next regime
    MarketRegime predict_next_regime(const std::vector<MarketRegime>& recent_regimes);
    
    // Get regime stability score
    double get_regime_stability_score(const MarketRegime& regime);

private:
    // Ensemble prediction
    std::vector<double> ensemble_predict(const torch::Tensor& market_features);
    
    // Individual model predictions
    std::vector<std::vector<double>> individual_predictions(const torch::Tensor& market_features);
    
    // Update model confidence
    void update_model_confidence(int64_t model_idx, bool correct_prediction);
    
    // Smooth regime predictions
    std::vector<double> smooth_predictions(const std::vector<double>& raw_probs);
};

TORCH_MODULE(RegimeDetectionEnsemble);

// ============================================================
// Experience Replay Buffer
// ============================================================

class ExperienceReplayBufferImpl : public torch::nn::Module {
private:
    int64_t buffer_size_;
    int64_t current_size_;
    int64_t current_idx_;
    
    // Buffer storage
    std::vector<MarketExperience> experiences_;
    std::vector<double> experience_priorities_;
    double priority_alpha_;
    double priority_beta_;
    
    // Sampling strategy
    bool use_prioritized_replay_;
    std::discrete_distribution<int> priority_sampler_;
    std::mt19937 rng_;

public:
    ExperienceReplayBufferImpl(
        int64_t buffer_size = 10000,
        double priority_alpha = 0.6,
        double priority_beta = 0.4
    );
    
    // Add new experience
    void add_experience(const MarketExperience& experience);
    
    // Sample batch of experiences
    std::vector<MarketExperience> sample_batch(int64_t batch_size);
    
    // Update experience priorities
    void update_priorities(const std::vector<int64_t>& indices, const std::vector<double>& priorities);
    
    // Get buffer statistics
    int64_t get_size() const { return current_size_; }
    double get_average_priority() const;
    std::vector<MarketExperience> get_recent_experiences(int64_t count = 100) const;
    
    // Clear buffer
    void clear();
    
    // Prioritized replay control
    void enable_prioritized_replay(bool enable) { use_prioritized_replay_ = enable; }
    void set_priority_parameters(double alpha, double beta);

private:
    // Compute experience priority
    double compute_priority(const MarketExperience& experience);
    
    // Update priority sampler
    void update_priority_sampler();
    
    // Sample uniform random batch
    std::vector<MarketExperience> sample_uniform_batch(int64_t batch_size);
    
    // Sample prioritized batch
    std::vector<MarketExperience> sample_prioritized_batch(int64_t batch_size);
};

TORCH_MODULE(ExperienceReplayBuffer);

// ============================================================
// Regime Meta-Learner - Main Architecture
// ============================================================

class RegimeMetaLearnerImpl : public torch::nn::Module {
private:
    // Core components
    MAMLOptimizer maml_optimizer_;
    FastAdaptationNetwork fast_adapter_;
    RegimeDetectionEnsemble regime_detector_;
    ExperienceReplayBuffer experience_buffer_;
    
    // Meta-learning parameters
    int64_t max_adaptation_steps_;
    double adaptation_threshold_;
    int64_t meta_batch_frequency_;
    int64_t adaptation_history_size_;
    
    // Performance optimization
    torch::Device device_;
    bool use_cuda_;
    std::chrono::nanoseconds max_adaptation_time_us_;
    
    // Adaptation tracking
    std::queue<std::pair<MarketRegime, AdaptedModel>> adaptation_history_;
    std::unordered_map<MarketRegime, AdaptedModel> cached_adapted_models_;
    std::chrono::nanoseconds last_meta_training_time_;
    int64_t meta_training_counter_;

public:
    RegimeMetaLearnerImpl(
        int64_t input_dim = 64,
        int64_t hidden_dim = 128,
        int64_t output_dim = 3,
        int64_t max_adaptation_steps = 10,
        double adaptation_threshold = 0.1,
        bool use_cuda = true,
        std::chrono::nanoseconds max_adaptation_time = std::chrono::microseconds(100)
    );
    
    // Rapid regime adaptation
    AdaptedModel adapt_to_regime(
        const MarketRegime& new_regime,
        const BaseModel& base_model
    );
    
    // Continual learning update
    void continual_learning_update(const MarketExperience& experience);
    
    // Get adapted model for regime
    AdaptedModel get_adapted_model(const MarketRegime& regime);
    
    // Detect current regime
    std::pair<MarketRegime, std::vector<double>> detect_current_regime(const torch::Tensor& market_features);
    
    // Meta-training step
    void meta_train_step();
    
    // Update with new market experience
    void update_with_experience(
        const MarketRegime& regime,
        const torch::Tensor& market_features,
        const torch::Tensor& targets,
        double performance_metric
    );
    
    // Performance monitoring
    double get_average_adaptation_time_us() const;
    double get_adaptation_success_rate() const;
    double get_meta_learning_loss() const;
    int64_t get_cached_models_count() const;
    
    // Model management
    void clear_cache();
    void save_meta_model(const std::string& filepath);
    void load_meta_model(const std::string& filepath);

private:
    // Check if model needs adaptation
    bool needs_adaptation(const MarketRegime& regime, const AdaptedModel& current_model);
    
    // Fast adaptation with caching
    AdaptedModel fast_adapt_with_cache(const MarketRegime& regime, const BaseModel& base_model);
    
    // Create adaptation data from experience
    torch::Tensor create_adaptation_data(const MarketExperience& experience);
    
    // Evaluate adapted model performance
    double evaluate_adapted_model(const AdaptedModel& model, const MarketExperience& test_experience);
    
    // Update adaptation history
    void update_adaptation_history(const MarketRegime& regime, const AdaptedModel& model);
    
    // Check if meta-training is needed
    bool needs_meta_training();
    
    // Get similar regimes from history
    std::vector<MarketRegime> get_similar_regimes(const MarketRegime& regime);
};

TORCH_MODULE(RegimeMetaLearner);

// ============================================================
// Data Structures
// ============================================================

struct MarketRegime {
    enum class Type {
        BULL_VOLATILE,
        BULL_STABLE,
        BEAR_VOLATILE,
        BEAR_STABLE,
        SIDEWAYS_LOW_VOL,
        SIDEWAYS_HIGH_VOL,
        TRANSITION,
        CRISIS
    };
    
    Type type;
    double volatility_index;
    double trend_strength;
    double liquidity_depth;
    double correlation_level;
    std::chrono::nanoseconds start_time;
    std::chrono::nanoseconds duration;
    
    // Regime characteristics
    std::vector<std::string> dominant_factors;
    std::vector<double> factor_weights;
    double predictability_score;
    double stability_score;
    
    // Historical context
    std::vector<MarketRegime> similar_regimes;
    std::vector<double> transition_probabilities;
};

struct BaseModel {
    std::string model_type;  // "transformer", "lstm", "cnn", etc.
    torch::nn::Module model;
    std::vector<torch::Tensor> parameters;
    std::unordered_map<std::string, double> hyperparameters;
    
    // Model performance
    double accuracy;
    double precision;
    double recall;
    double f1_score;
    
    // Training history
    std::vector<double> training_loss;
    std::vector<double> validation_loss;
    std::chrono::nanoseconds last_training_time;
};

struct AdaptedModel {
    BaseModel base_model;
    MarketRegime target_regime;
    std::vector<torch::Tensor> adapted_parameters;
    std::vector<double> adaptation_magnitudes;
    
    // Adaptation performance
    double adaptation_loss;
    double adaptation_accuracy;
    std::chrono::nanoseconds adaptation_time;
    
    // Adaptation metadata
    int64_t adaptation_steps;
    double adaptation_lr;
    std::vector<MarketExperience> adaptation_experiences;
    
    // Validity and freshness
    bool is_valid;
    std::chrono::nanoseconds creation_time;
    std::chrono::nanoseconds last_used_time;
    int64_t usage_count;
};

struct MarketExperience {
    MarketRegime regime;
    torch::Tensor market_features;
    torch::Tensor targets;
    double performance_metric;
    std::chrono::nanoseconds timestamp;
    
    // Experience metadata
    std::string source;  // "live_trading", "backtest", "simulation"
    double confidence_score;
    std::vector<std::string> involved_assets;
    
    // Learning relevance
    double learning_value;
    int64_t usage_count;
    std::chrono::nanoseconds last_used_time;
};

struct TaskDistribution {
    std::vector<MarketExperience> tasks;
    std::vector<double> task_weights;
    std::unordered_map<MarketRegime, std::vector<MarketExperience>> regime_tasks;
    
    // Distribution statistics
    double diversity_score;
    double difficulty_score;
    std::vector<MarketRegime> covered_regimes;
    
    // Sampling parameters
    bool uniform_sampling;
    double temperature;
};

// ============================================================
// Factory Functions
// ============================================================

RegimeMetaLearner create_regime_meta_learner_v4(
    int64_t input_dim = 64,
    int64_t hidden_dim = 128,
    int64_t output_dim = 3,
    int64_t max_adaptation_steps = 10,
    double adaptation_threshold = 0.1,
    bool use_cuda = true
);

// ============================================================
// Performance Benchmarks
// ============================================================

struct MetaLearnerMetrics {
    double avg_adaptation_time_us;
    double p95_adaptation_time_us;
    double adaptation_success_rate;
    double meta_learning_loss;
    double cached_model_hit_rate;
    int64_t regimes_processed_per_second;
    double memory_usage_mb;
};

class MetaLearnerBenchmark {
public:
    static MetaLearnerMetrics benchmark_regime_meta_learner(
        RegimeMetaLearner learner,
        int64_t num_regimes = 8,
        int64_t num_experiences = 1000
    );
    
    static bool validate_adaptation_speed(
        const MetaLearnerMetrics& metrics,
        double max_adaptation_time_us = 100.0
    );
    
    static bool validate_adaptation_quality(
        const MetaLearnerMetrics& metrics,
        double min_success_rate = 0.8
    );
    
    static bool validate_throughput_targets(
        const MetaLearnerMetrics& metrics,
        double min_regimes_per_second = 1000.0
    );
};

} // namespace v4
} // namespace models
} // namespace archneuronx
