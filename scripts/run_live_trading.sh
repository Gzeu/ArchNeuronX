#!/bin/bash

# ArchNeuronX v4.0 Live Trading Runner
# This script runs the live trading system with real market data

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    print_error "Please run this script from the ArchNeuronX root directory"
    exit 1
}

# Function to check dependencies
check_dependencies() {
    print_header "CHECKING LIVE TRADING DEPENDENCIES"
    
    # Check for CMake
    if ! command -v cmake &> /dev/null; then
        print_error "CMake is not installed. Please install CMake 3.20+"
        exit 1
    fi
    
    # Check for Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8+"
        exit 1
    fi
    
    # Check for PyTorch
    if ! python3 -c "import torch" &> /dev/null; then
        print_warning "PyTorch is not installed. Installing..."
        python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Check for curl (for exchange API)
    if ! command -v curl &> /dev/null; then
        print_error "curl is not installed. Please install curl"
        exit 1
    fi
    
    # Check for nlohmann/json
    if ! python3 -c "import json" &> /dev/null; then
        print_error "JSON library is not available"
        exit 1
    fi
    
    print_status "Live trading dependencies check completed"
}

# Function to build live trading system
build_live_trading() {
    print_header "BUILDING LIVE TRADING SYSTEM"
    
    # Create build directory
    if [ ! -d "build" ]; then
        print_status "Creating build directory"
        mkdir build
    fi
    
    cd build
    
    # Configure with CMake
    print_status "Configuring live trading system with CMake..."
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DUSE_CUDA=OFF \
          -DBUILD_V4_QUANTUM=ON \
          -DBUILD_LLM_INTEGRATION=ON \
          -DBUILD_LIVE_TRADING=ON \
          ..
    
    # Build live trading system
    print_status "Building live trading system..."
    make -j$(nproc) archneuronx_live_trading
    
    # Check if build was successful
    if [ $? -eq 0 ]; then
        print_status "Live trading system built successfully"
    else
        print_error "Live trading system build failed"
        exit 1
    fi
    
    cd ..
}

# Function to run live trading in interactive mode
run_interactive() {
    print_header "STARTING LIVE TRADING - INTERACTIVE MODE"
    
    if [ ! -f "build/archneuronx_live_trading" ]; then
        print_error "Live trading executable not found. Please build first."
        exit 1
    fi
    
    print_status "Starting live trading in interactive mode..."
    print_warning "This is a DEMO version with simulated market data"
    print_warning "DO NOT use with real money without proper testing"
    
    # Set environment variables
    export ARCHNEURONX_MODE=live
    export TRADING_SYMBOLS="BTCUSDT,ETHUSDT,BNBUSDT"
    export RISK_PER_TRADE=100.0
    export MAX_POSITION_SIZE=1000.0
    export TRADING_INTERVAL=1000
    
    # Run live trading
    ./build/archneuronx_live_trading
}

# Function to run live trading in automated mode
run_automated() {
    print_header "STARTING LIVE TRADING - AUTOMATED MODE"
    
    if [ ! -f "build/archneuronx_live_trading" ]; then
        print_error "Live trading executable not found. Please build first."
        exit 1
    fi
    
    print_status "Starting live trading in automated mode..."
    print_warning "This is a DEMO version with simulated market data"
    print_warning "DO NOT use with real money without proper testing"
    
    # Set environment variables
    export ARCHNEURONX_MODE=live
    export TRADING_SYMBOLS="BTCUSDT,ETHUSDT,BNBUSDT"
    export RISK_PER_TRADE=100.0
    export MAX_POSITION_SIZE=1000.0
    export TRADING_INTERVAL=1000
    export AUTOMATED_MODE=true
    
    # Run live trading in background
    nohup ./build/archneuronx_live_trading > logs/live_trading.log 2>&1 &
    
    # Get PID
    PID=$!
    echo $PID > logs/live_trading.pid
    
    print_status "Live trading started in background with PID: $PID"
    print_status "Logs are being written to: logs/live_trading.log"
    print_status "To stop: ./scripts/run_live_trading.sh stop"
}

# Function to stop live trading
stop_live_trading() {
    print_header "STOPPING LIVE TRADING"
    
    if [ -f "logs/live_trading.pid" ]; then
        PID=$(cat logs/live_trading.pid)
        
        if kill -0 $PID 2>/dev/null; then
            print_status "Stopping live trading process (PID: $PID)..."
            kill -TERM $PID
            
            # Wait for graceful shutdown
            for i in {1..10}; do
                if ! kill -0 $PID 2>/dev/null; then
                    print_status "Live trading stopped gracefully"
                    break
                fi
                sleep 1
            done
            
            # Force kill if still running
            if kill -0 $PID 2>/dev/null; then
                print_warning "Force killing live trading process..."
                kill -KILL $PID
            fi
            
            rm logs/live_trading.pid
        else
            print_warning "Live trading process not found"
        fi
    else
        print_warning "No live trading PID file found"
    fi
}

# Function to show live trading status
show_status() {
    print_header "LIVE TRADING STATUS"
    
    if [ -f "logs/live_trading.pid" ]; then
        PID=$(cat logs/live_trading.pid)
        
        if kill -0 $PID 2>/dev/null; then
            print_status "Live trading is RUNNING (PID: $PID)"
            
            # Show recent logs
            if [ -f "logs/live_trading.log" ]; then
                print_status "Recent logs:"
                tail -n 20 logs/live_trading.log
            fi
        else
            print_status "Live trading is NOT running"
        fi
    else
        print_status "Live trading is NOT running"
    fi
}

# Function to show logs
show_logs() {
    print_header "LIVE TRADING LOGS"
    
    if [ -f "logs/live_trading.log" ]; then
        print_status "Showing live trading logs:"
        tail -f logs/live_trading.log
    else
        print_error "No log file found"
    fi
}

# Function to run backtesting
run_backtest() {
    print_header "RUNNING BACKTESTING"
    
    if [ ! -f "build/archneuronx_live_trading" ]; then
        print_error "Live trading executable not found. Please build first."
        exit 1
    fi
    
    print_status "Running backtesting on historical data..."
    
    # Set environment variables
    export ARCHNEURONX_MODE=backtest
    export BACKTEST_DATA_PATH="data/historical/"
    export BACKTEST_START_DATE="2024-01-01"
    export BACKTEST_END_DATE="2024-12-31"
    
    # Run backtesting
    ./build/archneuronx_live_trading --backtest
}

# Function to run paper trading
run_paper_trading() {
    print_header "STARTING PAPER TRADING"
    
    if [ ! -f "build/archneuronx_live_trading" ]; then
        print_error "Live trading executable not found. Please build first."
        exit 1
    fi
    
    print_status "Starting paper trading with simulated money..."
    
    # Set environment variables
    export ARCHNEURONX_MODE=paper
    export PAPER_BALANCE=10000.0
    export TRADING_SYMBOLS="BTCUSDT,ETHUSDT,BNBUSDT"
    export RISK_PER_TRADE=100.0
    export MAX_POSITION_SIZE=1000.0
    
    # Run paper trading
    ./build/archneuronx_live_trading --paper
}

# Function to clean build
clean_build() {
    print_header "CLEANING LIVE TRADING BUILD"
    
    if [ -d "build" ]; then
        print_status "Removing build directory..."
        rm -rf build
        print_status "✅ Build directory removed"
    else
        print_status "Build directory does not exist"
    fi
    
    # Clean logs
    if [ -f "logs/live_trading.log" ]; then
        print_status "Cleaning logs..."
        rm logs/live_trading.log
        print_status "✅ Logs cleaned"
    fi
    
    # Clean PID file
    if [ -f "logs/live_trading.pid" ]; then
        rm logs/live_trading.pid
        print_status "✅ PID file cleaned"
    fi
}

# Function to show help
show_help() {
    echo "ArchNeuronX v4.0 Live Trading Runner"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build           Build the live trading system"
    echo "  interactive     Run live trading in interactive mode"
    echo "  automated      Run live trading in automated mode"
    echo "  stop            Stop automated live trading"
    echo "  status          Show live trading status"
    echo "  logs            Show live trading logs"
    echo "  backtest        Run backtesting on historical data"
    echo "  paper           Run paper trading with simulated money"
    echo "  clean           Clean build and logs"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Build live trading system"
    echo "  $0 interactive              # Start interactive trading"
    echo "  $0 automated                # Start automated trading"
    echo "  $0 stop                     # Stop automated trading"
    echo "  $0 status                   # Show trading status"
    echo "  $0 logs                     # Show live logs"
    echo "  $0 backtest                 # Run backtesting"
    echo "  $0 paper                    # Start paper trading"
    echo ""
    echo "Trading Modes:"
    echo "  🎯 Interactive Mode - Manual control with commands"
    echo "  🤖 Automated Mode - Fully automated trading"
    echo "  📊 Backtesting Mode - Test on historical data"
    echo "  📝 Paper Trading - Test with simulated money"
    echo ""
    echo "Risk Management:"
    echo "  🛡️ Built-in risk management with stop-loss and take-profit"
    echo "  📊 Real-time portfolio monitoring"
    echo "  🚨 Alert system for risk events"
    echo "  📈 Performance tracking and analytics"
}

# Create logs directory
mkdir -p logs

# Main script logic
case "${1:-help}" in
    "build")
        check_dependencies
        build_live_trading
        ;;
    "interactive")
        run_interactive
        ;;
    "automated")
        run_automated
        ;;
    "stop")
        stop_live_trading
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs
        ;;
    "backtest")
        run_backtest
        ;;
    "paper")
        run_paper_trading
        ;;
    "clean")
        clean_build
        ;;
    "help"|*)
        show_help
        ;;
esac
