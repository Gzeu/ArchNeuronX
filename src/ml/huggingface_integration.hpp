#pragma once

#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <functional>

namespace archneuronx {
namespace ml {

/**
 * Hugging Face Integration
 * 
 * This class provides integration with Hugging Face Transformers
 * and Mistral AI models for enhanced trading signal generation
 * and market analysis capabilities.
 */
class HuggingFaceIntegration {
public:
    struct HFModelConfig {
        std::string model_name;           // e.g., "mistralai/Mistral-7B-v0.1"
        std::string model_path;           // Local path or HF Hub path
        std::string cache_dir;           // Cache directory for models
        bool use_cuda = true;             // Use GPU acceleration
        bool use_flash_attention = true; // Use flash attention
        int max_length = 2048;           // Maximum sequence length
        double temperature = 1.0;        // Generation temperature
        int top_k = 50;                  // Top-k sampling
        bool do_sample = true;           // Use sampling instead of greedy
    };

    struct TradingPromptConfig {
        std::string system_prompt = "You are an expert trading assistant. Analyze market data and provide trading recommendations.";
        std::string user_prompt_template = "Market data: {market_data}\nCurrent portfolio: {portfolio}\nProvide trading signals in JSON format with: symbol, action, confidence, expected_return, risk_score.";
        std::string instruction_template = "Generate trading signals in JSON format with: symbol, action, confidence, expected_return, risk_score.";
        bool include_reasoning = true;
        bool include_market_analysis = true;
    };

public:
    explicit HuggingFaceIntegration(const HFModelConfig& config);
    ~HuggingFaceIntegration() = default;

    // Model loading and management
    bool load_model();
    void unload_model();
    bool is_model_loaded() const { return model_loaded_; }
    
    // Text generation for trading
    std::string generate_trading_signals(
        const std::string& market_data,
        const std::string& portfolio_state,
        const TradingPromptConfig& prompt_config
    );
    
    std::string generate_market_analysis(
        const std::vector<std::string>& symbols,
        const std::vector<double>& prices,
        const std::vector<double>& volumes
    );
    
    std::string generate_risk_assessment(
        const std::string& portfolio_composition,
        double portfolio_value,
        const std::string& market_conditions
    );
    
    // Batch processing
    std::vector<std::string> batch_generate_signals(
        const std::vector<std::string>& market_data_batch,
        const std::vector<std::string>& portfolio_states,
        const TradingPromptConfig& prompt_config
    );
    
    // Model information
    std::string get_model_info() const;
    std::vector<std::string> get_available_models() const;
    double get_model_performance() const { return model_performance_; }
    
    // Configuration
    void update_generation_params(double temperature, int top_k, bool do_sample);
    void set_max_length(int max_length);
    void set_device(const std::string& device);

private:
    HFModelConfig config_;
    TradingPromptConfig prompt_config_;
    
    // Model components
    std::unique_ptr<torch::nn::Module> model_;
    std::unique_ptr<torch::nn::Module> tokenizer_;
    std::unique_ptr<torch::Device> device_;
    
    // Model state
    bool model_loaded_ = false;
    double model_performance_ = 0.0;
    std::string current_model_name_;
    
    // Generation parameters
    double temperature_;
    int top_k_;
    bool do_sample_;
    int max_length_;
    
    // Cache and optimization
    std::map<std::string, torch::Tensor> kv_cache_;
    std::map<std::string, torch::Tensor> attention_cache_;
    
    // Private methods
    bool load_model_from_huggingface();
    bool load_model_from_local();
    void setup_model_optimizations();
    void initialize_kv_cache();
    void initialize_attention_cache();
    
    // Prompt engineering
    std::string build_trading_prompt(
        const std::string& market_data,
        const std::string& portfolio_state,
        const TradingPromptConfig& config
    );
    
    std::string build_market_analysis_prompt(
        const std::vector<std::string>& symbols,
        const std::vector<double>& prices,
        const std::vector<double>& volumes
    );
    
    std::string build_risk_assessment_prompt(
        const std::string& portfolio_composition,
        double portfolio_value,
        const std::string& market_conditions
    );
    
    // Text processing
    std::string post_process_generation(const std::string& generated_text);
    std::string extract_json_from_text(const std::string& text);
    std::vector<std::string> parse_trading_signals(const std::string& json_text);
    
    // Tokenization
    std::vector<int> tokenize_text(const std::string& text);
    std::string detokenize_tokens(const std::vector<int>& tokens);
    
    // Generation
    torch::Tensor generate_text_tokens(
        const std::vector<int>& input_ids,
        int max_new_tokens
    );
    
    std::vector<int> sample_from_logits(
        const torch::Tensor& logits,
        int top_k,
        double temperature
    );
};

/**
 * Mistral AI Integration
 * 
 * Specialized integration for Mistral AI models with
 * optimized performance for trading applications.
 */
class MistralIntegration : public HuggingFaceIntegration {
public:
    struct MistralConfig : public HFModelConfig {
        MistralConfig() : HFModelConfig() {
            model_name = "mistralai/Mistral-7B-v0.1";
            use_flash_attention = true;
            use_cuda = true;
            max_length = 2048;
            temperature = 0.7;  // Lower temperature for more deterministic trading
            top_k = 10;
            do_sample = false;  // Use greedy for trading decisions
        }
    };

public:
    explicit MistralIntegration(const MistralConfig& config);
    
    // Trading-specific methods
    std::string generate_quantum_enhanced_signals(
        const std::string& market_data,
        const std::string& quantum_state,
        const std::string& portfolio_state
    );
    
    std::string generate_regime_detection(
        const std::string& market_history,
        const std::string& current_indicators
    );
    
    std::string generate_portfolio_optimization(
        const std::string& current_portfolio,
        const std::string& market_opportunities,
        const std::string& risk_constraints
    );
    
    // Performance optimization
    void optimize_for_trading();
    void enable_quantum_enhancement();
    
private:
    MistralConfig mistral_config_;
    
    // Mistral-specific optimizations
    void setup_mistral_optimizations();
    void initialize_mistral_kv_cache();
    void setup_flash_attention_optimization();
    
    // Trading-specific prompt templates
    std::string build_quantum_trading_prompt(
        const std::string& market_data,
        const std::string& quantum_state,
        const std::string& portfolio_state
    );
    
    std::string build_regime_detection_prompt(
        const std::string& market_history,
        const std::string& current_indicators
    );
    
    std::string build_portfolio_optimization_prompt(
        const std::string& current_portfolio,
        const std::string& market_opportunities,
        const std::string& risk_constraints
    );
};

/**
 * Trading Signal Generator with LLM Integration
 * 
 * Combines quantum neural networks with large language models
 * for enhanced trading signal generation.
 */
class LLMEnhancedSignalGenerator {
public:
    struct LLMConfig {
        std::string llm_provider = "huggingface";  // "huggingface" or "mistral"
        std::string model_name = "mistralai/Mistral-7B-v0.1";
        bool use_llm_for_signals = true;
        bool use_llm_for_analysis = true;
        bool use_llm_for_risk = true;
        double llm_confidence_threshold = 0.8;
        bool enable_fallback = true;  // Fall back to quantum-only if LLM fails
    };

    struct SignalGenerationConfig {
        bool use_quantum_primary = true;
        bool use_llm_enhancement = true;
        double quantum_weight = 0.6;      // Weight for quantum model
        double llm_weight = 0.4;         // Weight for LLM
        bool combine_confidence = true;
        std::string combination_method = "weighted_average"; // "weighted_average", "ensemble", "voting"
    };

public:
    explicit LLMEnhancedSignalGenerator(const LLMConfig& config);
    
    // Enhanced signal generation
    std::vector<TradingSignal> generate_enhanced_signals(
        const torch::Tensor& market_data,
        const std::vector<std::string>& symbols,
        const torch::Tensor& quantum_state
    );
    
    // Market analysis with LLM
    std::string analyze_market_with_llm(
        const std::vector<std::string>& symbols,
        const std::vector<double>& prices,
        const std::vector<double>& volumes
    );
    
    // Risk assessment with LLM
    std::string assess_risk_with_llm(
        const std::string& portfolio_composition,
        double portfolio_value,
        const std::string& market_conditions
    );
    
    // Configuration
    void update_llm_config(const LLMConfig& config);
    void update_signal_config(const SignalGenerationConfig& config);
    
    // Performance monitoring
    double get_llm_performance() const;
    double get_enhanced_performance() const;
    std::map<std::string, double> get_component_performance() const;

private:
    LLMConfig llm_config_;
    SignalGenerationConfig signal_config_;
    
    // Components
    std::unique_ptr<HuggingFaceIntegration> llm_integration_;
    std::unique_ptr<models::QuantumTradingSignals> quantum_model_;
    
    // Performance tracking
    std::map<std::string, double> component_performance_;
    double llm_performance_ = 0.0;
    double enhanced_performance_ = 0.0;
    
    // Signal enhancement methods
    std::vector<TradingSignal> enhance_quantum_signals_with_llm(
        const std::vector<TradingSignal>& quantum_signals,
        const std::string& llm_analysis
    );
    
    std::vector<TradingSignal> combine_signals(
        const std::vector<TradingSignal>& quantum_signals,
        const std::vector<TradingSignal>& llm_signals
    );
    
    double calculate_enhanced_confidence(
        double quantum_confidence,
        double llm_confidence
    );
    
    std::string extract_llm_confidence(const std::string& llm_response);
    std::vector<TradingSignal> parse_llm_signals(const std::string& llm_response);
};

/**
 * Model Manager for Multiple LLMs
 * 
 * Manages multiple models from Hugging Face and Mistral AI
 * with automatic model selection and fallback mechanisms.
 */
class ModelManager {
public:
    struct ModelManagerConfig {
        std::string default_provider = "huggingface";
        std::string default_model = "mistralai < 1; 2;
        std::vector<std::string> available_models = {
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b-chat-hf",
            "microsoft/DialoGPT-medium",
            "google/gemma-7b"
        };
        std::string cache_dir = "./models/cache";
        bool auto_select_best_model = true;
        bool enable_model_switching = true;
    };

public:
    explicit ModelManager(const ModelManagerConfig& config);
    
    // Model management
    bool load_model(const std::string& model_name);
    void unload_model();
    std::string get_current_model() const;
    
    // Automatic model selection
    std::string select_best_model_for_task(const std::string& task);
    std::string select_model_for_trading();
    std::string select_model_for_analysis();
    
    // Model information
    std::vector<std::string> get_available_models() const;
    std::string get_model_info(const std::string& model_name) const;
    double get_model_performance(const std::string& model_name) const;
    
    // Performance optimization
    void optimize_model_performance();
    void enable_model_caching();
    void setup_model_ensemble();

private:
    ModelManagerConfig config_;
    
    // Model registry
    std::map<std::string, std::unique_ptr<HuggingFaceIntegration>> loaded_models_;
    std::string current_model_;
    
    // Performance tracking
    std::map<std::string, double> model_performance_;
    std::map<std::string, std::chrono::system_clock::time_point> last_used_;
    
    // Cache management
    std::string cache_dir_;
    std::map<std::string, std::string> model_cache_paths_;
    
    // Private methods
    bool is_model_cached(const std::string& model_name);
    std::string get_cached_model_path(const std::string& model_name);
    void cache_model(const std::string& model_name, const std::string& path);
    
    void initialize_model_registry();
    void update_model_performance(const std::string& model_name, double performance);
};

} // namespace ml
} // namespace archneuronx
