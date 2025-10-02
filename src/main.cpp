#include <iostream>
#include <string>
#include <memory>
#include "core/trading_engine.h"
#include "api/rest_server.h"
#include "utils/config_manager.h"
#include "utils/logger.h"

int main(int argc, char* argv[]) {
    try {
        // Initialize logger
        ArchNeuronX::Logger::initialize();
        
        // Parse command line arguments
        if (argc < 2) {
            std::cout << "ArchNeuronX - Automated Neural Network Trading System\n";
            std::cout << "Usage: " << argv[0] << " <command> [options]\n";
            std::cout << "\nCommands:\n";
            std::cout << "  train    - Train neural network models\n";
            std::cout << "  predict  - Generate trading signals\n";
            std::cout << "  server   - Start REST API server\n";
            std::cout << "  backtest - Run backtesting analysis\n";
            return 0;
        }
        
        std::string command = argv[1];
        
        if (command == "train") {
            // Model training logic
            LOG_INFO("Starting model training...");
            // TODO: Implement training pipeline
            
        } else if (command == "predict") {
            // Prediction logic
            LOG_INFO("Generating trading signals...");
            // TODO: Implement prediction pipeline
            
        } else if (command == "server") {
            // Start API server
            LOG_INFO("Starting REST API server...");
            
            auto config = ArchNeuronX::ConfigManager::load("config/server.json");
            auto server = std::make_unique<ArchNeuronX::RestServer>(config);
            
            server->start();
            
        } else if (command == "backtest") {
            // Backtesting logic
            LOG_INFO("Starting backtesting analysis...");
            // TODO: Implement backtesting
            
        } else {
            std::cerr << "Unknown command: " << command << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("Fatal error: " + std::string(e.what()));
        return 1;
    }
    
    return 0;
}