#include <gtest/gtest.h>
#include <torch/torch.h>
#include "core/trading_engine.h"

using namespace ArchNeuronX;

class TradingEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<TradingEngine>();
    }
    
    void TearDown() override {
        engine.reset();
    }
    
    std::unique_ptr<TradingEngine> engine;
};

TEST_F(TradingEngineTest, InitializationTest) {
    ASSERT_NE(engine, nullptr);
}

TEST_F(TradingEngineTest, SignalGenerationTest) {
    // Create sample market data
    auto market_data = torch::randn({1, 50, 5}); // batch_size=1, sequence_length=50, features=5
    
    // This should not crash (model might not be loaded yet)
    EXPECT_NO_THROW({
        try {
            auto signals = engine->generateSignals(market_data);
        } catch (const std::exception& e) {
            // Expected if no model is loaded yet
            EXPECT_TRUE(std::string(e.what()).find("model") != std::string::npos);
        }
    });
}

TEST_F(TradingEngineTest, TradingSignalStructTest) {
    TradingSignal signal;
    signal.action = TradingSignal::Action::BUY;
    signal.confidence = 0.85;
    signal.price_target = 50000.0;
    signal.symbol = "BTCUSDT";
    signal.timestamp = "2024-01-15T10:30:00Z";
    signal.explanation = "Strong bullish pattern detected";
    
    EXPECT_EQ(signal.action, TradingSignal::Action::BUY);
    EXPECT_DOUBLE_EQ(signal.confidence, 0.85);
    EXPECT_DOUBLE_EQ(signal.price_target, 50000.0);
    EXPECT_EQ(signal.symbol, "BTCUSDT");
}

// Test for different signal actions
TEST_F(TradingEngineTest, SignalActionsTest) {
    EXPECT_EQ(static_cast<int>(TradingSignal::Action::BUY), 0);
    EXPECT_EQ(static_cast<int>(TradingSignal::Action::SELL), 1);
    EXPECT_EQ(static_cast<int>(TradingSignal::Action::HOLD), 2);
}