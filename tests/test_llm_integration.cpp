#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../src/ml/huggingface_integration.hpp"
#include <torch/torch.h>
#include <string>
#include <memory>
#include <chrono>

using namespace archneuronx::ml;
using namespace ::testing;

// Mock classes for testing
class MockHuggingFaceIntegration : public HuggingFaceIntegration {
public:
    MockHuggingFaceIntegration(const HuggingFaceIntegration::HFModelConfig& config)
        : HuggingFaceIntegration(config) {}
    
    MOCK_METHOD(bool, load_model, (), ());
    MOCK_METHOD(void, unload_model, (), ());
    MOCK_METHOD(bool, is_model_loaded, (), (const));
    MOCK_METHOD(std::string, generate_trading_signals, 
        (const std::string&, const std::string&, const HuggingFaceIntegration::TradingPromptConfig&), ());
    MOCK_METHOD(std::string, generate_market_analysis, 
        (const std::vector<std::string>&, const std::vector<double>&, const std::vector<double>&), ());
    MOCK_METHOD(std::string, generate_risk_assessment, 
        (const std::string&, double, const std::string&), ());
    MOCK_METHOD(std::string, get_model_info, (), (const));
    MOCK_METHOD(double, get_model_performance, (), ());
    MOCK_METHOD(void, update_generation_params, (double, int, bool), ());
    MOCK_METHOD(void, set_max_length, (int), ());
    MOCK_METHOD(void, set_device, (const std::string&), ());
};

// Test fixture for LLM Integration
class LLMIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;
    
    std::unique_ptr<HuggingFaceIntegration> llm_integration_;
    std::shared_ptr<MockHuggingFaceIntegration> mock_llm_;
    
    HuggingFaceIntegration::HFModelConfig default_config_;
    HuggingFaceIntegration::TradingPromptConfig default_prompt_config_;
    
    void setup_mocks();
    void expect_llm_initialized();
    void test_model_loading();
    void test_trading_signal_generation();
    void test_market_analysis();
    void test_risk_assessment();
    void test_model_configuration();
    void test_performance_optimization();
    void test_error_handling();
    void test_mistral_integration();
    void test_llm_enhanced_signals();
};

void LLMIntegrationTest::SetUp() {
    default_config_.model_name = "test-model";
    default_config_.cache_dir = "./test_cache";
    default_config_.use_cuda = false;
    default_config_.use_flash_attention = true;
    default_config_.max_length = 2048;
    default_config_.temperature = 0.7;
    default_config_.top_k = 50;
    default_config_.do_sample = false;
    
    default_prompt_config_.system_prompt = "You are an expert trading assistant.";
    default_prompt_config_.user_prompt_template = "Market data: {market_data}\nPortfolio: {portfolio}";
    default_prompt_config_.instruction_template = "Generate trading signals in JSON format.";
    default_prompt_config_.include_reasoning = true;
    default_prompt_config_.include_market_analysis = true;
    
    setup_mocks();
    llm_integration_ = std::make_unique<HuggingFaceIntegration>(default_config_);
}

void LLMIntegrationTest::TearDown() {
    if (llm_integration_) {
        llm_integration_->unload_model();
    }
}

void LLMIntegrationTest::setup_mocks() {
    mock_llm_ = std::make_shared<MockHuggingFaceIntegration>(default_config_);
    
    // Set up mock expectations
    EXPECT_CALL(*mock_llm_, load_model())
        .WillRepeatedly(Return(true));
    EXPECT_CALL(*mock_llm_, is_model_loaded())
        .WillRepeatedly(Return(true));
    EXPECT_CALL(*mock_llm_, get_model_performance())
        .WillRepeatedly(Return(0.88));
}

void LLMIntegrationTest::expect_llm_initialized() {
    EXPECT_TRUE(llm_integration_ != nullptr);
    EXPECT_TRUE(llm_integration_->is_model_loaded());
}

void LLMIntegrationTest::test_model_loading() {
    expect_llm_initialized();
    
    // Test model loading
    EXPECT_CALL(*mock_llm_, load_model())
        .Times(1);
    
    EXPECT_TRUE(llm_integration_->load_model());
    EXPECT_TRUE(llm_integration_->is_model_loaded());
    
    // Test model unloading
    EXPECT_CALL(*mock_llm_, unload_model())
        .Times(1);
    
    llm_integration_->unload_model();
    EXPECT_FALSE(llm_integration_->is_model_loaded());
}

void LLMIntegrationTest::test_trading_signal_generation() {
    expect_llm_initialized();
    
    // Setup test data
    std::string market_data = "AAPL: 175.25, MSFT: 375.50, GOOGL: 150.75";
    std::string portfolio_state = "Portfolio: 100000 USD";
    
    // Mock trading signal generation
    std::string expected_response = R"({
        "signals": [
            {
                "symbol": "AAPL",
                "action": "BUY",
                "confidence": 0.892,
                "expected_return": 0.023,
                "risk_score": 0.156
            },
            {
                "symbol": "MSFT", 
                "action": "SELL",
                "confidence": 0.845,
                "expected_return": -0.018,
                "risk_score": 0.234
            }
        ]
    })";
    
    EXPECT_CALL(*mock_llm_, generate_trading_signals(market_data, portfolio_state, default_prompt_config_))
        .WillOnce(Return(expected_response));
    
    // Generate trading signals
    auto signals = llm_integration_->generate_trading_signals(market_data, portfolio_state, default_prompt_config_);
    
    // Verify response
    EXPECT_FALSE(signals.empty());
    EXPECT_TRUE(signals.find("AAPL") != std::string::npos);
    EXPECT_TRUE(signals.find("MSFT") != std::string::npos);
    EXPECT_TRUE(signals.find("BUY") != std::string::npos);
    EXPECT_TRUE(signals.find("SELL") != std::string::npos);
}

void LLMIntegrationTest::test_market_analysis() {
    expect_llm_initialized();
    
    // Setup test data
    std::vector<std::string> symbols = {"AAPL", "MSFT", "GOOGL"};
    std::vector<double> prices = {175.25, 375.50, 150.75};
    std::vector<double> volumes = {1000000, 1500000, 800000};
    
    // Mock market analysis generation
    std::string expected_analysis = R"({
        "market_trend": "bullish",
        "key_levels": {
            "support": 170.0,
            "resistance": 180.0
        },
        "recommendations": {
            "AAPL": "BUY",
            "MSFT": "HOLD",
            "GOOGL": "SELL"
        },
        "risk_factors": ["market_volatility", "geopolitical_risk"]
    })";
    
    EXPECT_CALL(*mock_llm_, generate_market_analysis(symbols, prices, volumes))
        .WillOnce(Return(expected_analysis));
    
    // Generate market analysis
    auto analysis = llm_integration_->generate_market_analysis(symbols, prices, volumes);
    
    // Verify analysis
    EXPECT_FALSE(analysis.empty());
    EXPECT_TRUE(analysis.find("market_trend") != std::string::npos);
    EXPECT_TRUE(analysis.find("key_levels") != std::string::npos);
    EXPECT_TRUE(analysis.find("recommendations") != std::string::npos);
}

void LLMIntegrationTest::test_risk_assessment() {
    expect_llm_initialized();
    
    // Setup test data
    std::string portfolio_composition = "AAPL: 30%, MSFT: 25%, GOOGL: 20%, AMZN: 15%, TSLA: 10%";
    double portfolio_value = 100000.0;
    std::string market_conditions = "volatile market with high uncertainty";
    
    // Mock risk assessment generation
    std::string expected_assessment = R"({
        "overall_risk": "moderate",
        "value_at_risk": {
            "var_95": 5000.0,
            "var_99": 8000.0
        },
        "max_drawdown": 0.12,
        "concentration_risk": 0.08,
        "liquidity_risk": 0.05,
        "hedging_recommendations": ["options_hedge", "diversification"],
        "stop_loss_recommendations": [
            {"symbol": "AAPL", "stop_loss": 165.0},
            {"symbol": "MSFT", "stop_loss": 350.0}
        ]
    })";
    
    EXPECT_CALL(*mock_llm_, generate_risk_assessment(portfolio_composition, portfolio_value, market_conditions))
        .WillOnce(Return(expected_assessment));
    
    // Generate risk assessment
    auto assessment = llm_integration_->generate_risk_assessment(portfolio_composition, portfolio_value, market_conditions);
    
    // Verify assessment
    EXPECT_FALSE(assessment.empty());
    EXPECT_TRUE(assessment.find("overall_risk") != std::string::npos);
    EXPECT_TRUE(assessment.find("value_at_risk") != std::string::npos);
    EXPECT_TRUE(assessment.find("hedging_recommendations") != std::string::npos);
}

void LLMIntegrationTest::test_model_configuration() {
    expect_llm_initialized();
    
    // Test generation parameters update
    EXPECT_CALL(*mock_llm_, update_generation_params(0.5, 20, false))
        .Times(1);
    
    llm_integration_->update_generation_params(0.5, 20, false);
    
    // Test max length update
    EXPECT_CALL(*mock_llm_, set_max_length(1024))
        .Times(1);
    
    llm_integration_->set_max_length(1024);
    
    // Test device update
    EXPECT_CALL(*mock_llm_, set_device("cpu"))
        .Times(1);
    
    llm_integration_->set_device("cpu");
    
    // Verify model info
    EXPECT_CALL(*mock_llm_, get_model_info())
        .Times(1);
    
    auto model_info = llm_integration_->get_model_info();
    EXPECT_FALSE(model_info.empty());
}

void LLMIntegrationTest::test_performance_optimization() {
    expect_llm_initialized();
    
    // Test performance optimization
    EXPECT_CALL(*mock_llm_, update_generation_params(0.3, 10, false))
        .Times(1);
    EXPECT_CALL(*mock_llm_, set_max_length(512))
        .Times(1);
    
    llm_integration_->optimize_for_trading();
    
    // Verify performance metrics
    EXPECT_CALL(*mock_llm_, get_model_performance())
        .Times(AtLeast(1));
    
    double performance = llm_integration_->get_model_performance();
    EXPECT_GT(performance, 0.0);
    EXPECT_LE(performance, 1.0);
}

void LLMIntegrationTest::test_error_handling() {
    expect_llm_initialized();
    
    // Test model loading failure
    EXPECT_CALL(*mock_llm_, load_model())
        .WillOnce(Return(false));
    
    EXPECT_FALSE(llm_integration_->load_model());
    EXPECT_FALSE(llm_integration_->is_model_loaded());
    
    // Test generation with unloaded model
    EXPECT_CALL(*mock_llm_, is_model_loaded())
        .WillOnce(Return(false));
    
    auto response = llm_integration_->generate_trading_signals("test", "test", default_prompt_config_);
    EXPECT_TRUE(response.find("Error: Model not loaded") != std::string::npos);
    
    // Test with invalid input
    EXPECT_CALL(*mock_llm_, generate_trading_signals(testing::A<std::string>(), testing::A<std::string>(), testing::A<HuggingFaceIntegration::TradingPromptConfig>()))
        .WillOnce(Throw(std::runtime_error("Generation error")));
    
    EXPECT_THROW(std::runtime_error("Generation error"), llm_integration_->generate_trading_signals("test", "test", default_prompt_config_));
}

void LLMIntegrationTest::test_mistral_integration() {
    expect_llm_initialized();
    
    // Create Mistral integration
    MistralIntegration::MistralConfig mistral_config;
    mistral_config.model_name = "mistralai/Mistral-7B-v0.1";
    mistral_config.use_flash_attention = true;
    mistral_config.temperature = 0.7;
    mistral_config.top_k = 10;
    mistral_config.do_sample = false;
    
    auto mistral_integration = std::make_unique<MistralIntegration>(mistral_config);
    
    // Test Mistral-specific features
    EXPECT_CALL(*mock_llm_, load_model())
        .Times(1);
    
    EXPECT_TRUE(mistral_integration->load_model());
    
    // Test quantum-enhanced signals
    std::string market_data = "AAPL: 175.25, MSFT: 375.50";
    std::string quantum_state = "coherence: 0.85, entanglement: 0.92";
    std::string portfolio_state = "Portfolio: 100000 USD";
    
    std::string enhanced_signals = mistral_integration->generate_quantum_enhanced_signals(
        market_data, quantum_state, portfolio_state
    );
    
    EXPECT_FALSE(enhanced_signals.empty());
    EXPECT_TRUE(enhanced_signals.find("quantum") != std::string::npos);
    
    // Test regime detection
    std::string market_history = "AAPL: 170.0->175.0, MSFT: 370.0->375.0";
    std::string current_indicators = "RSI: 65, MACD: bullish";
    
    std::string regime = mistral_integration->generate_regime_detection(market_history, current_indicators);
    
    EXPECT_FALSE(regime.empty());
    EXPECT_TRUE(regime.find("regime") != std::string::npos);
    
    // Test portfolio optimization
    std::string current_portfolio = "AAPL: 30%, MSFT: 25%, GOOGL: 20%";
    std::string market_opportunities = "AAPL: strong_buy, MSFT: hold";
    std::string risk_constraints = "max_risk: 0.05, max_drawdown: 0.1";
    
    std::string optimization = mistral_integration->generate_portfolio_optimization(
        current_portfolio, market_opportunities, risk_constraints
    );
    
    EXPECT_FALSE(optimization.empty());
    EXPECT_TRUE(optimization.find("optimization") != std::string::npos);
}

void LLMIntegrationTest::test_llm_enhanced_signals() {
    expect_llm_initialized();
    
    // Create LLM enhanced signal generator
    LLMEnhancedSignalGenerator::LLMConfig llm_config;
    llm_config.llm_provider = "huggingface";
    llm_config.model_name = "test-model";
    llm_config.use_llm_for_signals = true;
    llm_config.use_llm_for_analysis = true;
    llm_config.use_llm_for_risk = true;
    llm_config.llm_confidence_threshold = 0.8;
    llm_config.enable_fallback = true;
    
    auto llm_generator = std::make_unique<LLMEnhancedSignalGenerator>(llm_config);
    
    // Setup test data
    torch::Tensor market_data = torch::randn({10, 128});
    std::vector<std::string> symbols = {"AAPL", "MSFT", "GOOGL"};
    torch::Tensor quantum_state = torch::ones(8) / std::sqrt(8);
    
    // Mock LLM response
    std::string llm_response = R"({
        "signals": [
            {
                "symbol": "AAPL",
                "action": "BUY",
                "confidence": 0.892,
                "expected_return": 0.023,
                "risk_score": 0.156
            }
        ]
    })";
    
    EXPECT_CALL(*mock_llm_, generate_market_analysis(symbols, testing::A<std::vector<double>>(), testing::A<std::vector<double>>()))
        .WillOnce(Return(llm_response));
    
    // Generate enhanced signals
    auto enhanced_signals = llm_generator->generate_enhanced_signals(market_data, symbols, quantum_state);
    
    // Verify enhanced signals
    EXPECT_FALSE(enhanced_signals.empty());
    EXPECT_EQ(enhanced_signals.size(), symbols.size());
    
    // Verify signal structure
    for (const auto& signal : enhanced_signals) {
        EXPECT_FALSE(signal.symbol.empty());
        EXPECT_FALSE(signal.action.empty());
        EXPECT_GE(signal.confidence, 0.0);
        EXPECT_LE(signal.confidence, 1.0);
        EXPECT_GE(signal.expected_return, -1.0);
        EXPECT_LE(signal.expected_return, 1.0);
        EXPECT_GE(signal.risk_score, 0.0);
        EXPECT_LE(signal.risk_score, 1.0);
    }
    
    // Test market analysis
    std::vector<double> prices = {175.25, 375.50, 150.75};
    std::vector<double> volumes = {1000000, 1500000, 800000};
    
    auto analysis = llm_generator->analyze_market_with_llm(symbols, prices, volumes);
    EXPECT_FALSE(analysis.empty());
    
    // Test risk assessment
    std::string portfolio_composition = "AAPL: 30%, MSFT: 25%, GOOGL: 20%";
    double portfolio_value = 100000.0;
    std::string market_conditions = "volatile market";
    
    auto risk_assessment = llm_generator->assess_risk_with_llm(portfolio_composition, portfolio_value, market_conditions);
    EXPECT_FALSE(risk_assessment.empty());
    
    // Verify performance metrics
    EXPECT_GT(llm_generator->get_llm_performance(), 0.0);
    EXPECT_GT(llm_generator->get_enhanced_performance(), 0.0);
    
    auto component_performance = llm_generator->get_component_performance();
    EXPECT_FALSE(component_performance.empty());
    EXPECT_TRUE(component_performance.find("quantum_model") != component_performance.end());
    EXPECT_TRUE(component_performance.find("llm_model") != component_performance.end());
    EXPECT_TRUE(component_performance.find("enhanced_system") != component_performance.end());
}

// Test LLM integration with different models
TEST_F(LLMIntegrationTest, ModelVariants) {
    // Test with different model configurations
    {
        HuggingFaceIntegration::HFModelConfig config;
        config.model_name = "meta-llama/Llama-2-7b-chat-hf";
        config.use_cuda = true;
        config.use_flash_attention = true;
        
        auto llm = std::make_unique<HuggingFaceIntegration>(config);
        EXPECT_CALL(*mock_llm_, load_model())
            .Times(1);
        
        EXPECT_TRUE(llm->load_model());
        EXPECT_EQ(llm->get_model_info().find("meta-llama/Llama-2-7b-chat-hf"), std::string::npos);
    }
    
    // Test with Google Gemma
    {
        HuggingFaceIntegration::HFModelConfig config;
        config.model_name = "google/gemma-7b";
        config.use_cuda = false;
        config.use_flash_attention = false;
        
        auto llm = std::make_unique<HuggingFaceIntegration>(config);
        EXPECT_CALL(*mock_llm_, load_model())
            .Times(1);
        
        EXPECT_TRUE(llm->load_model());
    }
    
    // Test with Microsoft DialoGPT
    {
        HuggingFaceIntegration::HFModelConfig config;
        config.model_name = "microsoft/DialoGPT-medium";
        config.use_cuda = true;
        config.use_flash_attention = false;
        
        auto llm = std::make_unique<HuggingFaceIntegration>(config);
        EXPECT_CALL(*mock_llm_, load_model())
            .Times(1);
        
        EXPECT_TRUE(llm->load_model());
    }
}

// Test LLM integration performance
TEST_F(LLMIntegrationTest, PerformanceBenchmark) {
    expect_llm_initialized();
    
    const int num_generations = 100;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Performance benchmark for signal generation
    for (int i = 0; i < num_generations; ++i) {
        std::string market_data = "AAPL: 175.25, MSFT: 375.50";
        std::string portfolio_state = "Portfolio: 100000 USD";
        
        auto signals = llm_integration_->generate_trading_signals(market_data, portfolio_state, default_prompt_config_);
        
        // Verify signal generation
        EXPECT_FALSE(signals.empty());
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Calculate performance metrics
    double avg_time = duration.count() / num_generations;
    double throughput = num_generations / (duration.count() / 1000.0);
    
    // Verify performance targets
    EXPECT_LT(avg_time, 100.0);  // <100ms per generation
    EXPECT_GT(throughput, 10.0);  // >10 generations/sec
    
    std::cout << "LLM Integration Performance:" << std::endl;
    std::cout << "  Generations: " << num_generations << std::endl;
    std::cout << "  Total Time: " << duration.count() << "ms" << std::endl;
    std::cout << "  Avg Time: " << avg_time << "ms" << std::endl;
    std::cout << "  Throughput: " << throughput << " gens/sec" << std::endl;
}

// Test LLM integration error recovery
TEST_F(LLMIntegrationTest, ErrorRecovery) {
    expect_llm_initialized();
    
    // Test model loading failure
    EXPECT_CALL(*mock_llm_, load_model())
        .WillOnce(Return(false));
    
    EXPECT_FALSE(llm_integration_->load_model());
    
    // Test fallback behavior
    EXPECT_CALL(*mock_llm_, is_model_loaded())
        .WillOnce(Return(false));
    
    auto response = llm_integration_->generate_trading_signals("test", "test", default_prompt_config_);
    EXPECT_TRUE(response.find("Error: Model not loaded") != std::string::npos);
    
    // Test recovery
    EXPECT_CALL(*mock_llm_, load_model())
        .WillOnce(Return(true));
    
    EXPECT_TRUE(llm_integration_->load_model());
    
    // Verify recovery
    EXPECT_CALL(*mock_llm_, is_model_loaded())
        .WillOnce(Return(true));
    
    response = llm_integration_->generate_trading_signals("test", "test", default_prompt_config_);
    EXPECT_FALSE(response.find("Error: Model not loaded") != std::string::npos);
}

// End of test suite
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
