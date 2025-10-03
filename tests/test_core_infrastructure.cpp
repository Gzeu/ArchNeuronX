/**
 * @file test_core_infrastructure.cpp
 * @brief Unit tests for ArchNeuronX core infrastructure
 * @author George Pricop
 * @date 2025-10-02
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>

#include "core/logger.hpp"
#include "core/config.hpp"
#include "trading/trading_engine.hpp"

using namespace ArchNeuronX::Core;
using namespace ArchNeuronX::Trading;

/**
 * @class LoggerTest
 * @brief Test fixture for logger functionality
 */
class LoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test logs directory
        std::filesystem::create_directories("test_logs");
        
        // Initialize logger for testing
        Logger::getInstance().initialize(
            LogLevel::DEBUG,
            LogOutput::FILE_ONLY,
            "test_logs/test.log"
        );
    }
    
    void TearDown() override {
        // Clean up test files
        std::filesystem::remove_all("test_logs");
    }
};

/**
 * @brief Test logger initialization and basic logging
 */
TEST_F(LoggerTest, BasicLogging) {
    EXPECT_TRUE(Logger::getInstance().isLevelEnabled(LogLevel::DEBUG));
    EXPECT_TRUE(Logger::getInstance().isLevelEnabled(LogLevel::INFO));
    EXPECT_TRUE(Logger::getInstance().isLevelEnabled(LogLevel::ERROR));
    
    // Test logging macros
    LOG_DEBUG("Debug message test");
    LOG_INFO("Info message test");
    LOG_WARN("Warning message test");
    LOG_ERROR("Error message test");
    
    // Flush to ensure messages are written
    Logger::getInstance().flush();
    
    // Check if log file was created and contains messages
    EXPECT_TRUE(std::filesystem::exists("test_logs/test.log"));
    
    std::ifstream log_file("test_logs/test.log");
    std::string content((std::istreambuf_iterator<char>(log_file)),
                        std::istreambuf_iterator<char>());
    
    EXPECT_NE(content.find("Debug message test"), std::string::npos);
    EXPECT_NE(content.find("Info message test"), std::string::npos);
    EXPECT_NE(content.find("Warning message test"), std::string::npos);
    EXPECT_NE(content.find("Error message test"), std::string::npos);
}

/**
 * @brief Test logger level filtering
 */
TEST_F(LoggerTest, LogLevelFiltering) {
    // Set log level to WARNING
    Logger::getInstance().setLogLevel(LogLevel::WARN);
    
    EXPECT_FALSE(Logger::getInstance().isLevelEnabled(LogLevel::DEBUG));
    EXPECT_FALSE(Logger::getInstance().isLevelEnabled(LogLevel::INFO));
    EXPECT_TRUE(Logger::getInstance().isLevelEnabled(LogLevel::WARN));
    EXPECT_TRUE(Logger::getInstance().isLevelEnabled(LogLevel::ERROR));
}

/**
 * @class ConfigTest
 * @brief Test fixture for configuration functionality
 */
class ConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        Config::getInstance().clear();
    }
    
    void TearDown() override {
        Config::getInstance().clear();
        std::filesystem::remove("test_config.json");
    }
};

/**
 * @brief Test configuration set and get operations
 */
TEST_F(ConfigTest, SetAndGet) {
    Config& config = Config::getInstance();
    
    // Test different data types
    config.set("test.string", std::string("hello world"));
    config.set("test.integer", 42);
    config.set("test.double", 3.14159);
    config.set("test.boolean", true);
    
    std::vector<std::string> test_list = {"item1", "item2", "item3"};
    config.set("test.list", test_list);
    
    // Test retrieval
    EXPECT_EQ(config.get<std::string>("test.string"), "hello world");
    EXPECT_EQ(config.get<int>("test.integer"), 42);
    EXPECT_DOUBLE_EQ(config.get<double>("test.double"), 3.14159);
    EXPECT_EQ(config.get<bool>("test.boolean"), true);
    
    auto retrieved_list = config.get<std::vector<std::string>>("test.list");
    EXPECT_EQ(retrieved_list.size(), 3);
    EXPECT_EQ(retrieved_list[0], "item1");
    EXPECT_EQ(retrieved_list[1], "item2");
    EXPECT_EQ(retrieved_list[2], "item3");
}

/**
 * @brief Test configuration default values
 */
TEST_F(ConfigTest, DefaultValues) {
    Config& config = Config::getInstance();
    
    // Test non-existent keys return default values
    EXPECT_EQ(config.get<std::string>("nonexistent.key", "default"), "default");
    EXPECT_EQ(config.get<int>("nonexistent.key", 100), 100);
    EXPECT_DOUBLE_EQ(config.get<double>("nonexistent.key", 2.718), 2.718);
    EXPECT_EQ(config.get<bool>("nonexistent.key", false), false);
}

/**
 * @brief Test configuration file save and load
 */
TEST_F(ConfigTest, SaveAndLoad) {
    Config& config = Config::getInstance();
    
    // Set test values
    config.set("app.name", std::string("TestApp"));
    config.set("app.version", std::string("1.0.0"));
    config.set("settings.debug", true);
    config.set("settings.port", 8080);
    
    // Save to file
    config.saveToFile("test_config.json");
    EXPECT_TRUE(std::filesystem::exists("test_config.json"));
    
    // Clear config and reload
    config.clear();
    EXPECT_FALSE(config.has("app.name"));
    
    config.loadFromFile("test_config.json");
    
    // Verify loaded values
    EXPECT_EQ(config.get<std::string>("app.name"), "TestApp");
    EXPECT_EQ(config.get<std::string>("app.version"), "1.0.0");
    EXPECT_EQ(config.get<bool>("settings.debug"), true);
    EXPECT_EQ(config.get<int>("settings.port"), 8080);
}

/**
 * @class TradingEngineTest
 * @brief Test fixture for trading engine functionality
 */
class TradingEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = std::make_unique<TradingEngine>("TestEngine");
        
        risk_params_.max_position_size = 0.1;
        risk_params_.min_confidence_threshold = 0.7;
        risk_params_.use_stop_loss = true;
        risk_params_.use_take_profit = true;
        
        engine_->initialize(10000.0, risk_params_);
    }
    
    std::unique_ptr<TradingEngine> engine_;
    RiskParameters risk_params_;
};

/**
 * @brief Test trading engine initialization
 */
TEST_F(TradingEngineTest, Initialization) {
    EXPECT_EQ(engine_->getName(), "TestEngine");
    EXPECT_DOUBLE_EQ(engine_->getInitialCapital(), 10000.0);
    EXPECT_DOUBLE_EQ(engine_->getCurrentCapital(), 10000.0);
    EXPECT_FALSE(engine_->isRunning());
}

/**
 * @brief Test signal processing
 */
TEST_F(TradingEngineTest, SignalProcessing) {
    // Create test signal
    TradingSignal signal;
    signal.symbol = "BTCUSDT";
    signal.signal_type = SignalType::BUY;
    signal.confidence = 0.85;
    signal.price_target = 50000.0;
    signal.stop_loss = 48000.0;
    signal.take_profit = 52000.0;
    signal.timestamp = std::chrono::system_clock::now();
    
    // Update market price
    engine_->updatePrice("BTCUSDT", 49500.0);
    
    // Process signal
    bool result = engine_->processSignal(signal);
    EXPECT_TRUE(result);
    
    // Check positions
    auto positions = engine_->getPositions();
    EXPECT_GT(positions.size(), 0);
}

/**
 * @brief Test signal validation
 */
TEST_F(TradingEngineTest, SignalValidation) {
    // Test signal with low confidence (should be rejected)
    TradingSignal low_confidence_signal;
    low_confidence_signal.symbol = "BTCUSDT";
    low_confidence_signal.signal_type = SignalType::BUY;
    low_confidence_signal.confidence = 0.5; // Below threshold
    low_confidence_signal.price_target = 50000.0;
    
    engine_->updatePrice("BTCUSDT", 49500.0);
    bool result = engine_->processSignal(low_confidence_signal);
    EXPECT_FALSE(result); // Should be rejected due to low confidence
}

/**
 * @brief Test portfolio statistics
 */
TEST_F(TradingEngineTest, PortfolioStatistics) {
    auto stats = engine_->getStatistics();
    EXPECT_GE(stats.size(), 0);
    
    double portfolio_value = engine_->getPortfolioValue();
    EXPECT_DOUBLE_EQ(portfolio_value, 10000.0); // Should equal initial capital
}

/**
 * @brief Test signal type conversions
 */
TEST(UtilityTest, SignalTypeConversions) {
    EXPECT_EQ(signalTypeToString(SignalType::BUY), "BUY");
    EXPECT_EQ(signalTypeToString(SignalType::SELL), "SELL");
    EXPECT_EQ(signalTypeToString(SignalType::HOLD), "HOLD");
    
    EXPECT_EQ(stringToSignalType("BUY"), SignalType::BUY);
    EXPECT_EQ(stringToSignalType("SELL"), SignalType::SELL);
    EXPECT_EQ(stringToSignalType("HOLD"), SignalType::HOLD);
}

/**
 * @brief Test risk parameter validation
 */
TEST(UtilityTest, RiskParameterValidation) {
    RiskParameters valid_params;
    valid_params.max_position_size = 0.1;
    valid_params.min_confidence_threshold = 0.7;
    
    // Should not throw
    EXPECT_NO_THROW(valid_params.validate());
    
    RiskParameters invalid_params;
    invalid_params.max_position_size = 1.5; // Invalid (> 1.0)
    invalid_params.min_confidence_threshold = 0.7;
    
    // Should throw
    EXPECT_THROW(invalid_params.validate(), std::invalid_argument);
}

/**
 * @brief Main test runner
 */
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}