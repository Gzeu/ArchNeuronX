#!/bin/bash

# ArchNeuronX v4.0 Complete Trading System Runner
# This script builds and runs the complete trading system with all components

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
fi

# Function to check dependencies
check_dependencies() {
    print_header "CHECKING DEPENDENCIES"
    
    # Check for CMake
    if ! command -v cmake &> /dev/null; then
        print_error "CMake is not installed. Please install CMake 3.20+"
        exit 1
    fi
    
    # Check for C++ compiler
    if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
        print_error "C++ compiler is not installed. Please install g++ or clang++"
        exit 1
    fi
    
    # Check for Python (for LibTorch)
    if ! command -v python3 &> /dev/null; then
        print_warning "Python3 is not installed. LibTorch may not be available"
    fi
    
    # Check for CUDA (optional)
    if command -v nvcc &> /dev/null; then
        print_status "CUDA found: $(nvcc --version | head -1)"
    else
        print_warning "CUDA not found. GPU acceleration will be disabled"
    fi
    
    print_status "Dependencies check completed"
}

# Function to build the system
build_system() {
    print_header "BUILDING COMPLETE TRADING SYSTEM"
    
    # Create build directory
    if [ ! -d "build" ]; then
        print_status "Creating build directory"
        mkdir build
    fi
    
    cd build
    
    # Configure with CMake
    print_status "Configuring with CMake..."
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DUSE_CUDA=ON \
          -DBUILD_V4_QUANTUM=ON \
          -DBUILD_LLM_INTEGRATION=ON \
          -DENABLE_GPU_ACCELERATION=ON \
          -DENABLE_FLASH_ATTENTION=ON \
          -DENABLE_MODEL_CACHING=ON \
          ..
    
    # Build
    print_status "Building complete system..."
    make -j$(nproc) archneuronx_complete
    
    # Check if build was successful
    if [ $? -eq 0 ]; then
        print_status "Build completed successfully"
    else
        print_error "Build failed"
        exit 1
    fi
    
    cd ..
}

# Function to run the system
run_system() {
    print_header "RUNNING COMPLETE TRADING SYSTEM"
    
    # Check if binary exists
    if [ ! -f "build/archneuronx_complete" ]; then
        print_error "Binary not found. Please build the system first."
        exit 1
    fi
    
    # Create models cache directory
    if [ ! -d "models/cache" ]; then
        print_status "Creating models cache directory"
        mkdir -p models/cache
    fi
    
    # Create logs directory
    if [ ! -d "logs" ]; then
        print_status "Creating logs directory"
        mkdir -p logs
    fi
    
    # Run the system
    print_status "Starting ArchNeuronX v4.0 Complete Trading System..."
    print_status "Web Interface: http://localhost:8080"
    print_status "WebSocket: ws://localhost:3001"
    print_status "Press Ctrl+C to stop"
    echo
    
    # Run with environment variables
    export CUDA_VISIBLE_DEVICES=0
    export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"
    export PYTHONPATH="$PYTHONPATH:$(python3 -c 'import torch; print(torch.__file__[:torch.__file__.rfind("/lib")]' 2>/dev/null || echo "")"
    
    ./build/archneuronx_complete "$@"
}

# Function to run in continuous mode
run_continuous() {
    print_header "RUNNING CONTINUOUS TRADING MODE"
    
    if [ ! -f "build/archneuronx_complete" ]; then
        print_error "Binary not found. Please build the system first."
        exit 1
    fi
    
    print_status "Starting continuous trading mode..."
    print_status "Web Interface: http://localhost:8080"
    print_status "WebSocket: ws://localhost:3001"
    print_status "Press Ctrl+C to stop"
    echo
    
    ./build/archneuronx_complete --continuous
}

# Function to show system status
show_status() {
    print_header "SYSTEM STATUS"
    
    if [ ! -f "build/archneuronx_complete" ]; then
        print_error "Binary not found. Please build the system first."
        exit 1
    fi
    
    print_status "Checking system status..."
    
    # Check if system is running
    if pgrep -f "archneuronx_complete" > /dev/null; then
        print_status "✅ ArchNeuronX is running"
        print_status "Web Interface: http://localhost:8080"
        print_status "WebSocket: ws://localhost:3001"
    else
        print_warning "ArchNeuronX is not running"
    fi
    
    # Show system info if available
    if [ -f "logs/system_info.json" ]; then
        print_status "System Information:"
        cat logs/system_info.json
    fi
}

# Function to stop the system
stop_system() {
    print_header "STOPPING SYSTEM"
    
    if pgrep -f "archneuronx_complete" > /dev/null; then
        print_status "Stopping ArchNeuronX..."
        pkill -f "archneuronx_complete"
        sleep 2
        
        if pgrep -f "archneuronx_complete" > /dev/null; then
            print_warning "System still running, forcing stop..."
            pkill -9 -f "archneuronx_complete"
        fi
        
        print_status "✅ ArchNeuronX stopped"
    else
        print_warning "ArchNeuronX is not running"
    fi
}

# Function to clean build
clean_build() {
    print_header "CLEANING BUILD"
    
    if [ -d "build" ]; then
        print_status "Removing build directory..."
        rm -rf build
        print_status "✅ Build directory removed"
    else
        print_status "Build directory does not exist"
    fi
    
    # Clean cache directories
    if [ -d "models/cache" ]; then
        print_status "Cleaning model cache..."
        rm -rf models/cache/*
        print_status "✅ Model cache cleaned"
    fi
    
    if [ -d "logs" ]; then
        print_status "Cleaning logs..."
        rm -f logs/*
        print_status "✅ Logs cleaned"
    fi
}

# Function to show help
show_help() {
    echo "ArchNeuronX v4.0 Complete Trading System Runner"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       Build the complete trading system"
    echo "  run         Run the system in interactive mode"
    echo "  continuous   Run the system in continuous trading mode"
    echo "  status      Show system status"
    echo "  stop        Stop the running system"
    echo "  clean       Clean build and cache"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Build the system"
    echo "  $0 run                      # Run in interactive mode"
    echo "  $0 continuous                # Run in continuous mode"
    echo "  $0 status                   # Show system status"
    echo "  $0 stop                     # Stop the system"
    echo ""
    echo "Web Interface:"
    echo "  HTTP Server: http://localhost:8080"
    echo "  WebSocket: ws://localhost:3001"
    echo "  API Endpoints: http://localhost:8080/api/v4/"
    echo ""
    echo "System Components:"
    echo "  🧠 Quantum Neural Networks"
    echo "  🤖 Quantum Trading Agents"
    echo "  🤖 HuggingFace LLM Integration"
    echo "  🌐 Web Interface"
    echo "  🤝 Multi-Agent Coordination"
    echo "  📊 Real-time Monitoring"
}

# Main script logic
main() {
    case "${1:-help}" in
        "build")
            check_dependencies
            build_system
            ;;
        "run")
            run_system
            ;;
        "continuous")
            run_continuous
            ;;
        "status")
            show_status
            ;;
        "stop")
            stop_system
            ;;
        "clean")
            clean_build
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function
main "$@"
