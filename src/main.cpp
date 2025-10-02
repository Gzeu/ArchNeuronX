/**
 * @file main.cpp
 * @brief Main application entry point for ArchNeuronX
 * @author George Pricop
 * @date 2025-10-02
 */

#include <iostream>
#include <exception>
#include <chrono>
#include <thread>

// Core infrastructure
#include "core/logger.hpp"
#include "core/config.hpp"
#include "core/base_model.hpp"
#include "trading/trading_engine.hpp"

// LibTorch
#include <torch/torch.h>

using namespace ArchNeuronX;
using namespace ArchNeuronX::Core;
using namespace ArchNeuronX::Trading;

/**
 * @brief Initialize application configuration with default values
 */
void initializeDefaultConfig() {
    LOG_INFO("Initializing default configuration...");
    
    // Application settings
    CONFIG_SET("app.name", std::string("ArchNeuronX"));
    CONFIG_SET("app.version", std::string("1.0.0"));
    CONFIG_SET("app.environment", std::string("development"));
    
    // Logging settings
    CONFIG_SET("logging.level", std::string("INFO"));
    CONFIG_SET("logging.file", std::string("logs/archneuronx.log"));
    CONFIG_SET("logging.max_file_size", 10485760); // 10MB
    
    // Model settings
    CONFIG_SET("model.type", std::string("MLP"));
    CONFIG_SET("model.input_size", 50);
    CONFIG_SET("model.output_size", 3); // BUY, SELL, HOLD
    CONFIG_SET("model.hidden_layers", std::vector<std::string>{"128", "64", "32"});
    CONFIG_SET("model.use_gpu", true);
    
    // Training settings
    CONFIG_SET("training.epochs", 100);
    CONFIG_SET("training.batch_size", 32);
    CONFIG_SET("training.learning_rate", 0.001);
    CONFIG_SET("training.patience", 10);
    
    // Trading settings
    CONFIG_SET("trading.initial_capital", 10000.0);
    CONFIG_SET("trading.max_position_size", 0.1);
    CONFIG_SET("trading.min_confidence", 0.7);
    CONFIG_SET("trading.use_stop_loss", true);
    
    // Data settings
    CONFIG_SET("data.symbols", std::vector<std::string>{"BTCUSDT", "ETHUSDT", "ADAUSDT"});
    CONFIG_SET("data.timeframe", std::string("1h"));
    CONFIG_SET("data.lookback_period", 100);
    
    LOG_INFO("Default configuration initialized successfully");
}

/**
 * @brief Test LibTorch installation and CUDA availability
 */
void testLibTorchSetup() {
    LOG_INFO("Testing LibTorch setup...");
    
    try {
        // Test basic tensor operations
        torch::Tensor test_tensor = torch::randn({3, 4});
        LOG_INFO("Created test tensor with shape: [" << test_tensor.size(0) << ", " << test_tensor.size(1) << "]");
        
        // Check CUDA availability
        bool cuda_available = torch::cuda::is_available();
        LOG_INFO("CUDA available: " << (cuda_available ? "Yes" : "No"));
        
        if (cuda_available) {
            int cuda_device_count = torch::cuda::device_count();
            LOG_INFO("CUDA devices available: " << cuda_device_count);
            
            for (int i = 0; i < cuda_device_count; ++i) {
                auto props = torch::cuda::getDeviceProperties(i);
                LOG_INFO("GPU " << i << ": " << props.name << " (" << props.totalGlobalMem / (1024*1024) << " MB)");
            }
            
            // Test GPU tensor operations
            torch::Device cuda_device(torch::kCUDA, 0);
            torch::Tensor gpu_tensor = test_tensor.to(cuda_device);
            torch::Tensor result = torch::matmul(gpu_tensor, gpu_tensor.transpose(0, 1));
            LOG_INFO("GPU tensor operations successful");
        }
        
        // Test automatic differentiation
        torch::Tensor x = torch::randn({2, 2}, torch::requires_grad());
        torch::Tensor y = x * x * 3;
        torch::Tensor z = y.mean();
        z.backward();
        LOG_INFO("Automatic differentiation test successful");
        
        LOG_INFO("LibTorch setup test completed successfully");
        
    } catch (const std::exception& e) {
        LOG_ERROR("LibTorch test failed: " << e.what());
        throw;
    }
}

/**
 * @brief Test trading engine functionality
 */
void testTradingEngine() {
    LOG_INFO("Testing trading engine...");
    
    try {
        // Create trading engine
        TradingEngine engine("TestEngine");
        
        // Setup risk parameters
        RiskParameters risk_params;
        risk_params.max_position_size = CONFIG_GET("trading.max_position_size", 0.1);
        risk_params.min_confidence_threshold = CONFIG_GET("trading.min_confidence", 0.7);
        risk_params.use_stop_loss = CONFIG_GET("trading.use_stop_loss", true);
        
        // Initialize engine
        double initial_capital = CONFIG_GET("trading.initial_capital", 10000.0);
        engine.initialize(initial_capital, risk_params);
        
        LOG_INFO("Trading engine initialized with capital: $" << initial_capital);
        
        // Create test trading signal
        TradingSignal test_signal;
        test_signal.symbol = "BTCUSDT";
        test_signal.signal_type = SignalType::BUY;
        test_signal.confidence = 0.85;
        test_signal.price_target = 50000.0;
        test_signal.stop_loss = 48000.0;
        test_signal.take_profit = 52000.0;
        test_signal.timestamp = std::chrono::system_clock::now();
        test_signal.explanation = "Strong bullish momentum detected";
        
        // Update market price
        engine.updatePrice("BTCUSDT", 49500.0);
        
        // Process signal
        bool signal_processed = engine.processSignal(test_signal);
        LOG_INFO("Test signal processed: " << (signal_processed ? "Success" : "Failed"));
        
        // Get portfolio statistics
        auto stats = engine.getStatistics();
        LOG_INFO("Portfolio value: $" << engine.getPortfolioValue());
        
        LOG_INFO("Trading engine test completed successfully");
        
    } catch (const std::exception& e) {
        LOG_ERROR("Trading engine test failed: " << e.what());
        throw;
    }
}

/**
 * @brief Display application banner
 */
void displayBanner() {
    std::cout << "\n";
    std::cout << "  █████╗ ██████╗  ██████╗██╗  ██╗███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ███╗   ██╗██╗  ██╗\n";
    std::cout << " ██╔══██╗██╔══██╗██╔════╝██║  ██║████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗████╗  ██║╚██╗██╔╝\n";
    std::cout << " ███████║██████╔╝██║     ███████║██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██╔██╗ ██║ ╚███╔╝ \n";
    std::cout << " ██╔══██║██╔══██╗██║     ██╔══██║██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██║╚██╗██║ ██╔██╗ \n";
    std::cout << " ██║  ██║██║  ██║╚██████╗██║  ██║██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝██║ ╚████║██╔╝ ██╗\n";
    std::cout << " ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝\n";
    std::cout << "\n";
    std::cout << "         Automated Neural Network Trading System v1.0.0\n";
    std::cout << "         Author: George Pricop | 2025 | C++17 + LibTorch\n";
    std::cout << "\n";
}

/**
 * @brief Main application entry point
 * @param argc Argument count
 * @param argv Argument values
 * @return Exit code
 */
int main(int argc, char* argv[]) {
    try {
        // Display application banner
        displayBanner();
        
        // Initialize logging system
        Logger::getInstance().initialize(
            LogLevel::INFO,
            LogOutput::CONSOLE_AND_FILE,
            "logs/archneuronx.log"
        );
        
        LOG_INFO("=== ArchNeuronX Starting ===");
        LOG_INFO("Application started with " << argc << " arguments");
        
        // Initialize configuration
        initializeDefaultConfig();
        
        // Test LibTorch setup
        testLibTorchSetup();
        
        // Test trading engine
        testTradingEngine();
        
        // Save configuration to file
        CONFIG.saveToFile("config/default.json");
        LOG_INFO("Configuration saved to config/default.json");
        
        LOG_INFO("=== Core Infrastructure Test Completed ===");
        LOG_INFO("All systems operational. Ready for development.");
        
        // Keep application running for demonstration
        std::cout << "\nPress Enter to exit...";
        std::cin.get();
        
        LOG_INFO("=== ArchNeuronX Shutting Down ===");
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        LOG_FATAL("Application crashed: " << e.what());
        return 1;
    } catch (...) {
        std::cerr << "Unknown fatal error occurred" << std::endl;
        LOG_FATAL("Application crashed with unknown error");
        return 2;
    }
}