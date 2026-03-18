#include "trading/live_trading_engine.hpp"
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <signal.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>

using namespace trading::live;

// Global flag for graceful shutdown
std::atomic<bool> keep_running(true);

// Signal handler for graceful shutdown
void signal_handler(int signal) {
    std::cout << "\n🛑 Received signal " << signal << ". Shutting down gracefully..." << std::endl;
    keep_running = false;
}

// Print banner
void print_banner() {
    std::cout << R"(
    ████████╗ ██████╗ ██╗  ██╗███████╗██████╗ ██████╗ 
    ╚══██╔══╝██╔═══██╗██║ ██╔╝██╔════╝██╔══██╗██╔══██╗
       ██║   ██║   ██║█████╔╝ █████╗  ██████╔╝██████╔╝
       ██║   ██║   ██║██╔═██╗ ██╔══╝  ██╔══██╗██╔═══╝ 
       ██║   ╚██████╔╝██║  ██╗███████╗██║  ██║██║     
       ╚═╝    ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     
                                                       
    🚀 LIVE TRADING ENGINE v4.0 🚀
    🧠 Quantum-Enhanced AI Trading
    🤖 Autonomous Trading Agents
    🤖 LLM-Enhanced Market Analysis
    🌐 Real-Time Risk Management
    )" << std::endl;
}

// Print system status
void print_system_status(const LiveTradingEngine& engine) {
    std::cout << "\n📊 SYSTEM STATUS" << std::endl;
    std::cout << "================" << std::endl;
    
    Portfolio portfolio = engine.get_portfolio();
    RiskMetrics metrics = engine.get_risk_metrics();
    
    std::cout << "💰 Portfolio Value: $" << std::fixed << std::setprecision(2) << portfolio.total_value << std::endl;
    std::cout << "💵 Cash Balance: $" << std::fixed << std::setprecision(2) << portfolio.cash_balance << std::endl;
    std::cout << "📈 Total P&L: $" << std::fixed << std::setprecision(2) << portfolio.total_pnl << std::endl;
    std::cout << "📅 Daily P&L: $" << std::fixed << std::setprecision(2) << portfolio.daily_pnl << std::endl;
    std::cout << "📊 Win Rate: " << std::fixed << std::setprecision(1) << (metrics.win_rate * 100) << "%" << std::endl;
    std::cout << "🎯 Total Trades: " << metrics.total_trades << std::endl;
    std::cout << "✅ Winning Trades: " << metrics.winning_trades << std::endl;
    std::cout << "❌ Losing Trades: " << metrics.losing_trades << std::endl;
    std::cout << "⚡ Sharpe Ratio: " << std::fixed << std::setprecision(2) << metrics.sharpe_ratio << std::endl;
    std::cout << "🛡️ Max Drawdown: " << std::fixed << std::setprecision(2) << (metrics.max_drawdown * 100) << "%" << std::endl;
}

// Print market data
void print_market_data(const LiveTradingEngine& engine, const std::vector<std::string>& symbols) {
    std::cout << "\n📈 MARKET DATA" << std::endl;
    std::cout << "===============" << std::endl;
    
    for (const auto& symbol : symbols) {
        MarketData data = engine.get_market_data(symbol);
        std::cout << "📊 " << symbol << ": $" << std::fixed << std::setprecision(2) << data.price
                  << " | Vol: " << std::fixed << std::setprecision(0) << data.volume
                  << " | RSI: " << std::fixed << std::setprecision(1) << data.rsi
                  << " | Volatility: " << std::fixed << std::setprecision(2) << (data.volatility * 100) << "%" << std::endl;
    }
}

// Print open orders
void print_open_orders(const LiveTradingEngine& engine) {
    std::vector<Order> orders = engine.get_open_orders();
    
    std::cout << "\n📋 OPEN ORDERS" << std::endl;
    std::cout << "===============" << std::endl;
    
    if (orders.empty()) {
        std::cout << "📭 No open orders" << std::endl;
        return;
    }
    
    for (const auto& order : orders) {
        std::cout << "📋 " << order.id << " | " << order.symbol << " | "
                  << (order.side == Order::Side::BUY ? "BUY" : "SELL") << " | "
                  << order.quantity << " @ $" << std::fixed << std::setprecision(2) << order.price << std::endl;
    }
}

// Print positions
void print_positions(const LiveTradingEngine& engine) {
    Portfolio portfolio = engine.get_portfolio();
    
    std::cout << "\n📊 POSITIONS" << std::endl;
    std::cout << "=============" << std::endl;
    
    if (portfolio.positions.empty()) {
        std::cout << "📭 No open positions" << std::endl;
        return;
    }
    
    for (const auto& [symbol, position] : portfolio.positions) {
        std::cout << "📊 " << symbol << ": " << position.quantity << " @ $" 
                  << std::fixed << std::setprecision(2) << position.avg_price
                  << " | P&L: $" << std::fixed << std::setprecision(2) << position.unrealized_pnl << std::endl;
    }
}

// Interactive command interface
void interactive_mode(LiveTradingEngine& engine) {
    std::string command;
    
    while (keep_running.load()) {
        std::cout << "\n🎯 Enter command (help for commands): ";
        std::getline(std::cin, command);
        
        if (command == "help") {
            std::cout << "\n📋 AVAILABLE COMMANDS:" << std::endl;
            std::cout << "=====================" << std::endl;
            std::cout << "status     - Show system status" << std::endl;
            std::cout << "market     - Show market data" << std::endl;
            std::cout << "orders     - Show open orders" << std::endl;
            std::cout << "positions  - Show positions" << std::endl;
            std::cout << "buy <sym> <qty> <price> - Place buy order" << std::endl;
            std::cout << "sell <sym> <qty> <price> - Place sell order" << std::endl;
            std::cout << "cancel <id> - Cancel order" << std::endl;
            std::cout << "symbols    - Show available symbols" << std::endl;
            std::cout << "risk       - Show risk metrics" << std::endl;
            std::cout << "stop       - Stop trading engine" << std::endl;
            std::cout << "help       - Show this help" << std::endl;
            
        } else if (command == "status") {
            print_system_status(engine);
            
        } else if (command == "market") {
            std::vector<std::string> symbols = {"BTCUSDT", "ETHUSDT", "BNBUSDT"};
            print_market_data(engine, symbols);
            
        } else if (command == "orders") {
            print_open_orders(engine);
            
        } else if (command == "positions") {
            print_positions(engine);
            
        } else if (command == "symbols") {
            std::vector<std::string> symbols = engine.get_available_symbols();
            std::cout << "\n📊 AVAILABLE SYMBOLS" << std::endl;
            std::cout << "===================" << std::endl;
            for (size_t i = 0; i < std::min(symbols.size(), size_t(20)); ++i) {
                std::cout << "📊 " << symbols[i] << std::endl;
            }
            if (symbols.size() > 20) {
                std::cout << "... and " << (symbols.size() - 20) << " more" << std::endl;
            }
            
        } else if (command == "risk") {
            RiskMetrics metrics = engine.get_risk_metrics();
            std::cout << "\n🛡️ RISK METRICS" << std::endl;
            std::cout << "===============" << std::endl;
            std::cout << "📊 Max Drawdown: " << std::fixed << std::setprecision(2) << (metrics.max_drawdown * 100) << "%" << std::endl;
            std::cout << "⚡ Sharpe Ratio: " << std::fixed << std::setprecision(2) << metrics.sharpe_ratio << std::endl;
            std::cout << "🎯 Sortino Ratio: " << std::fixed << std::setprecision(2) << metrics.sortino_ratio << std::endl;
            std::cout << "📈 VaR 95%: " << std::fixed << std::setprecision(2) << (metrics.var_95 * 100) << "%" << std::endl;
            std::cout << "📊 Win Rate: " << std::fixed << std::setprecision(1) << (metrics.win_rate * 100) << "%" << std::endl;
            std::cout << "💰 Profit Factor: " << std::fixed << std::setprecision(2) << metrics.profit_factor << std::endl;
            
        } else if (command == "stop") {
            std::cout << "🛑 Stopping trading engine..." << std::endl;
            keep_running = false;
            
        } else if (command.substr(0, 3) == "buy") {
            // Parse buy command: buy <symbol> <quantity> <price>
            std::istringstream iss(command);
            std::string cmd, symbol;
            double quantity, price;
            
            if (iss >> cmd >> symbol >> quantity >> price) {
                Order order;
                order.symbol = symbol;
                order.type = Order::Type::MARKET;
                order.side = Order::Side::BUY;
                order.quantity = quantity;
                order.price = price;
                order.created_at = std::chrono::system_clock::now();
                
                std::string order_id = engine.place_order(order);
                if (!order_id.empty()) {
                    std::cout << "✅ Buy order placed: " << order_id << std::endl;
                } else {
                    std::cout << "❌ Failed to place buy order" << std::endl;
                }
            } else {
                std::cout << "❌ Invalid buy command format. Use: buy <symbol> <quantity> <price>" << std::endl;
            }
            
        } else if (command.substr(0, 4) == "sell") {
            // Parse sell command: sell <symbol> <quantity> <price>
            std::istringstream iss(command);
            std::string cmd, symbol;
            double quantity, price;
            
            if (iss >> cmd >> symbol >> quantity >> price) {
                Order order;
                order.symbol = symbol;
                order.type = Order::Type::MARKET;
                order.side = Order::Side::SELL;
                order.quantity = quantity;
                order.price = price;
                order.created_at = std::chrono::system_clock::now();
                
                std::string order_id = engine.place_order(order);
                if (!order_id.empty()) {
                    std::cout << "✅ Sell order placed: " << order_id << std::endl;
                } else {
                    std::cout << "❌ Failed to place sell order" << std::endl;
                }
            } else {
                std::cout << "❌ Invalid sell command format. Use: sell <symbol> <quantity> <price>" << std::endl;
            }
            
        } else if (command.substr(0, 6) == "cancel") {
            // Parse cancel command: cancel <order_id>
            std::istringstream iss(command);
            std::string cmd, order_id;
            
            if (iss >> cmd >> order_id) {
                if (engine.cancel_order(order_id)) {
                    std::cout << "✅ Order cancelled: " << order_id << std::endl;
                } else {
                    std::cout << "❌ Failed to cancel order: " << order_id << std::endl;
                }
            } else {
                std::cout << "❌ Invalid cancel command format. Use: cancel <order_id>" << std::endl;
            }
            
        } else if (command.empty()) {
            // Continue
        } else {
            std::cout << "❌ Unknown command: " << command << ". Type 'help' for available commands." << std::endl;
        }
    }
}

// Setup callbacks
void setup_callbacks(LiveTradingEngine& engine, AlertSystem& alert_system) {
    // Order filled callback
    engine.set_order_filled_callback([&alert_system](const Order& order) {
        alert_system.send_trade_alert(order);
    });
    
    // Market data callback
    engine.set_market_data_callback([&alert_system](const MarketData& data) {
        // Send alert for significant price movements
        if (std::abs(data.volatility) > 0.05) {
            alert_system.send_alert("VOLATILITY", 
                "High volatility detected in " + data.symbol + ": " + 
                std::to_string(data.volatility * 100) + "%");
        }
    });
    
    // Portfolio update callback
    engine.set_portfolio_update_callback([&alert_system](const Portfolio& portfolio) {
        // Send alert for significant P&L changes
        if (std::abs(portfolio.daily_pnl) > 1000) {
            alert_system.send_performance_alert(
                "Daily P&L: $" + std::to_string(portfolio.daily_pnl));
        }
    });
    
    // Error callback
    engine.set_error_callback([&alert_system](const std::string& error) {
        alert_system.send_alert("ERROR", error);
    });
}

// Main function
int main(int argc, char* argv[]) {
    print_banner();
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    try {
        std::cout << "🚀 Initializing ArchNeuronX Live Trading Engine..." << std::endl;
        
        // Create exchange interface (mock implementation)
        auto exchange = std::make_unique<BinanceExchange>("demo_api_key", "demo_api_secret");
        
        // Create quantum trading signals (mock implementation)
        auto quantum_signals = std::make_unique<models::QuantumTradingSignals>();
        
        // Create quantum trading agent (mock implementation)
        auto trading_agent = std::make_unique<agents::QuantumTradingAgent>(10, 3);
        
        // Create LLM integration (mock implementation)
        auto llm_integration = std::make_unique<ml::HuggingFaceIntegration>();
        
        // Create live trading engine
        LiveTradingEngine engine(
            std::move(exchange),
            std::move(quantum_signals),
            std::move(trading_agent),
            std::move(llm_integration)
        );
        
        // Create alert system
        AlertSystem alert_system;
        
        // Setup callbacks
        setup_callbacks(engine, alert_system);
        
        // Configure trading parameters
        std::vector<std::string> trading_symbols = {"BTCUSDT", "ETHUSDT", "BNBUSDT"};
        engine.set_trading_symbols(trading_symbols);
        engine.set_trading_interval(std::chrono::milliseconds(1000));
        engine.set_risk_parameters(100.0, 1000.0);
        
        std::cout << "📊 Trading symbols: ";
        for (const auto& symbol : trading_symbols) {
            std::cout << symbol << " ";
        }
        std::cout << std::endl;
        
        std::cout << "⏱️ Trading interval: 1000ms" << std::endl;
        std::cout << "🛡️ Risk per trade: $100.00" << std::endl;
        std::cout << "📊 Max position size: $1000.00" << std::endl;
        
        // Start trading engine
        if (!engine.start()) {
            std::cerr << "❌ Failed to start trading engine" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Trading engine started successfully!" << std::endl;
        
        // Wait a moment for initialization
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // Show initial status
        print_system_status(engine);
        print_market_data(engine, trading_symbols);
        
        // Interactive mode
        std::cout << "\n🎯 Entering interactive mode..." << std::endl;
        std::cout << "Type 'help' for available commands or 'stop' to shutdown." << std::endl;
        
        interactive_mode(engine);
        
        // Stop trading engine
        std::cout << "🛑 Stopping trading engine..." << std::endl;
        engine.stop();
        
        // Final status
        std::cout << "\n📊 FINAL STATUS" << std::endl;
        std::cout << "===============" << std::endl;
        print_system_status(engine);
        
        std::cout << "\n🎉 ArchNeuronX Live Trading Engine shutdown complete!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
