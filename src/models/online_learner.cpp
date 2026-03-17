// ============================================================
// ArchNeuronX v2 - Online Learning Implementation
// Continual learning for adapting to market regime changes
// ============================================================
#include "models/neural_networks.hpp"
#include <torch/torch.h>
#include <deque>
#include <memory>
#include <mutex>
#include <algorithm>

namespace ArchNeuronX {
namespace Models {

/**
 * @brief Experience replay buffer for online learning
 */
class ExperienceReplay {
public:
    struct Experience {
        torch::Tensor state;
        torch::Tensor action;
        torch::Tensor reward;
        torch::Tensor next_state;
        bool done;
        double timestamp;
    };

    explicit ExperienceReplay(size_t max_size = 10000) : max_size_(max_size) {}

    void add(const Experience& exp) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (buffer_.size() >= max_size_) {
            buffer_.pop_front();
        }
        
        buffer_.push_back(exp);
        
        // Remove old experiences based on time window
        cleanup_old_experiences();
    }

    std::vector<Experience> sample(size_t batch_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (buffer_.empty()) {
            return {};
        }
        
        std::vector<Experience> batch;
        size_t actual_batch_size = std::min(batch_size, buffer_.size());
        
        // Simple random sampling (could be improved with prioritized replay)
        std::vector<size_t> indices(buffer_.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_shuffle(indices.begin(), indices.end());
        
        for (size_t i = 0; i < actual_batch_size; ++i) {
            batch.push_back(buffer_[indices[i]]);
        }
        
        return batch;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_.clear();
    }

private:
    size_t max_size_;
    std::deque<Experience> buffer_;
    mutable std::mutex mutex_;
    
    void cleanup_old_experiences() {
        const double max_age_seconds = 86400.0; // 24 hours
        auto current_time = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        
        buffer_.erase(
            std::remove_if(buffer_.begin(), buffer_.end(),
                [current_time, max_age_seconds](const Experience& exp) {
                    return (current_time - exp.timestamp) > max_age_seconds;
                }),
            buffer_.end()
        );
    }
};

/**
 * @brief Online learning optimizer with adaptive learning rates
 */
class OnlineOptimizer {
public:
    struct Config {
        double base_lr = 0.001;
        double lr_decay_factor = 0.95;
        double lr_increase_factor = 1.05;
        int patience = 5;
        double min_lr = 1e-6;
        double max_lr = 1e-2;
        bool use_adaptive_lr = true;
    };

    explicit OnlineOptimizer(torch::optim::Optimizer& optimizer, const Config& config = {})
        : optimizer_(optimizer), config_(config), current_lr_(config.base_lr) {}

    void step(double loss) {
        if (!config_.use_adaptive_lr) {
            optimizer_.step();
            return;
        }

        // Adaptive learning rate based on loss trend
        if (loss_history_.size() >= config_.patience) {
            double recent_avg = std::accumulate(loss_history_.end() - config_.patience, 
                                             loss_history_.end(), 0.0) / config_.patience;
            double older_avg = std::accumulate(loss_history_.end() - 2 * config_.patience,
                                             loss_history_.end() - config_.patience, 0.0) / config_.patience;

            if (recent_avg < older_avg * 0.99) { // Improving
                current_lr_ = std::min(current_lr_ * config_.lr_increase_factor, config_.max_lr);
            } else if (recent_avg > older_avg * 1.01) { // Deteriorating
                current_lr_ = std::max(current_lr_ * config_.lr_decay_factor, config_.min_lr);
            }

            update_learning_rate();
        }

        loss_history_.push_back(loss);
        if (loss_history_.size() > 100) {
            loss_history_.pop_front();
        }

        optimizer_.step();
    }

    double get_current_lr() const { return current_lr_; }

private:
    torch::optim::Optimizer& optimizer_;
    Config config_;
    double current_lr_;
    std::deque<double> loss_history_;

    void update_learning_rate() {
        for (auto& param_group : optimizer_.param_groups()) {
            param_group.options().lr(current_lr_);
        }
    }
};

/**
 * @brief Online learning manager for continual model adaptation
 */
class OnlineLearner {
public:
    struct Config {
        size_t replay_buffer_size = 10000;
        size_t batch_size = 32;
        int update_frequency = 10; // Update every N new experiences
        double loss_threshold = 0.1; // Trigger update if loss exceeds this
        bool use_experience_replay = true;
        bool use_early_stopping = true;
        int early_stopping_patience = 20;
        double min_performance = 0.6; // Minimum accuracy to maintain
    };

    explicit OnlineLearner(std::shared_ptr<torch::nn::Module> model, 
                           const Config& config = {})
        : model_(model), config_(config), replay_buffer_(config.replay_buffer_size) {
        
        // Initialize optimizer
        optimizer_ = std::make_unique<torch::optim::Adam>(model_->parameters(), 1e-3);
        online_optimizer_ = std::make_unique<OnlineOptimizer>(*optimizer_);
    }

    void add_experience(const torch::Tensor& state,
                       const torch::Tensor& action,
                       const torch::Tensor& reward,
                       const torch::Tensor& next_state,
                       bool done) {
        
        ExperienceReplay::Experience exp{
            state.clone(),
            action.clone(),
            reward.clone(),
            next_state.clone(),
            done,
            std::chrono::duration<double>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count()
        };

        replay_buffer_.add(exp);
        experience_count_++;

        // Trigger online update if conditions are met
        if (should_update()) {
            perform_online_update();
        }
    }

    void perform_online_update() {
        if (replay_buffer_.size() < config_.batch_size) {
            return;
        }

        auto batch = replay_buffer_.sample(config_.batch_size);
        if (batch.empty()) {
            return;
        }

        // Prepare batch tensors
        std::vector<torch::Tensor> states, actions, rewards, next_states;
        std::vector<bool> dones;

        for (const auto& exp : batch) {
            states.push_back(exp.state);
            actions.push_back(exp.action);
            rewards.push_back(exp.reward);
            next_states.push_back(exp.next_state);
            dones.push_back(exp.done);
        }

        auto states_batch = torch::stack(states);
        auto actions_batch = torch::stack(actions);
        auto rewards_batch = torch::stack(rewards);
        auto next_states_batch = torch::stack(next_states);

        // Forward pass
        model_->train();
        optimizer_->zero_grad();

        auto predictions = model_->forward(states_batch);
        auto loss = compute_loss(predictions, actions_batch, rewards_batch);

        // Backward pass
        loss.backward();
        
        // Gradient clipping for stability
        torch::nn::utils::clip_grad_norm_(model_->parameters(), 1.0);
        
        online_optimizer_->step(loss.item<double>());

        // Update performance tracking
        update_performance_metrics(loss.item<double>());

        last_update_time_ = std::chrono::steady_clock::now();
    }

    bool should_update() const {
        if (!config_.use_experience_replay || replay_buffer_.size() < config_.batch_size) {
            return false;
        }

        // Update based on frequency
        if (experience_count_ % config_.update_frequency != 0) {
            return false;
        }

        // Check if performance degradation triggers update
        if (recent_performance_.size() >= 10) {
            double avg_performance = std::accumulate(recent_performance_.begin(), 
                                                   recent_performance_.end(), 0.0) / recent_performance_.size();
            if (avg_performance < config_.min_performance) {
                return true;
            }
        }

        return true;
    }

    double get_recent_performance() const {
        if (recent_performance_.empty()) {
            return 0.0;
        }
        return std::accumulate(recent_performance_.begin(), recent_performance_.end(), 0.0) / recent_performance_.size();
    }

    size_t get_buffer_size() const { return replay_buffer_.size(); }
    double get_current_lr() const { return online_optimizer_->get_current_lr(); }

private:
    std::shared_ptr<torch::nn::Module> model_;
    Config config_;
    ExperienceReplay replay_buffer_;
    std::unique_ptr<torch::optim::Optimizer> optimizer_;
    std::unique_ptr<OnlineOptimizer> online_optimizer_;
    
    size_t experience_count_ = 0;
    std::deque<double> recent_performance_;
    std::chrono::steady_clock::time_point last_update_time_;

    torch::Tensor compute_loss(const torch::Tensor& predictions,
                              const torch::Tensor& actions,
                              const torch::Tensor& rewards) {
        // Simple MSE loss for demonstration
        // In practice, this would be more sophisticated for trading
        auto target = actions + rewards; // Simplified target
        return torch::mse_loss(predictions, target);
    }

    void update_performance_metrics(double loss) {
        double performance = 1.0 / (1.0 + loss); // Convert loss to performance metric
        recent_performance_.push_back(performance);
        
        if (recent_performance_.size() > 50) {
            recent_performance_.pop_front();
        }
    }
};

// Factory function
std::unique_ptr<OnlineLearner> create_online_learner(
    std::shared_ptr<torch::nn::Module> model,
    const OnlineLearner::Config& config = {}
) {
    return std::make_unique<OnlineLearner>(model, config);
}

} // namespace Models
} // namespace ArchNeuronX
