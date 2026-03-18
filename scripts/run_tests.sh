#!/bin/bash

# ArchNeuronX v4.0 Test Runner
# This script runs all tests for the complete trading system

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
    print_header "CHECKING TEST DEPENDENCIES"
    
    # Check for CMake
    if ! command -v cmake &> /dev/null; then
        print_error "CMake is not installed. Please install CMake 3.20+"
        exit 1
    fi
    
    # Check for gtest/gmock
    if ! ldconfig -p | grep -q "libgtest" || ! ldconfig -p | grep -q "libgmock"; then
        print_warning "Google Test/Mock may not be installed. Installing..."
        
        # Try to install gtest/gmock
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y libgtest-dev libgmock-dev
        elif command -v yum &> /dev/null; then
            sudo yum install -y gtest-devel gmock-devel
        elif command -v brew &> /dev/null; then
            brew install googletest
        else
            print_error "Please install Google Test/Mock manually"
            exit 1
        fi
    fi
    
    # Check for torch
    if ! python3 -c "import torch" &> /dev/null; then
        print_warning "PyTorch may not be installed. Please install LibTorch 2.6+"
    fi
    
    print_status "Test dependencies check completed"
}

# Function to build tests
build_tests() {
    print_header "BUILDING TESTS"
    
    # Create build directory
    if [ ! -d "build" ]; then
        print_status "Creating build directory"
        mkdir build
    fi
    
    cd build
    
    # Configure with CMake
    print_status "Configuring tests with CMake..."
    cmake -DCMAKE_BUILD_TYPE=Debug \
          -DENABLE_COVERAGE=ON \
          -DUSE_CUDA=ON \
          -DBUILD_V4_QUANTUM=ON \
          -DBUILD_LLM_INTEGRATION=ON \
          ..
    
    # Build tests
    print_status "Building tests..."
    make -j$(nproc) archneuronx_tests
    
    # Check if build was successful
    if [ $? -eq 0 ]; then
        print_status "Tests built successfully"
    else
        print_error "Test build failed"
        exit 1
    fi
    
    cd ..
}

# Function to run all tests
run_all_tests() {
    print_header "RUNNING ALL TESTS"
    
    if [ ! -f "build/archneuronx_tests" ]; then
        print_error "Test executable not found. Please build tests first."
        exit 1
    fi
    
    print_status "Running all tests..."
    ./build/archneuronx_tests --gtest_output=xml
    
    # Check test results
    if [ $? -eq 0 ]; then
        print_status "All tests passed!"
    else
        print_error "Some tests failed!"
        exit 1
    fi
}

# Function to run unit tests
run_unit_tests() {
    print_header "RUNNING UNIT TESTS"
    
    if [ ! -f "build/archneuronx_tests" ]; then
        print_error "Test executable not found. Please build tests first."
        exit 1
    fi
    
    print_status "Running unit tests..."
    ./build/archneuronx_tests --gtest_filter="Unit*" --gtest_output=xml
    
    if [ $? -eq 0 ]; then
        print_status "Unit tests passed!"
    else
        print_error "Unit tests failed!"
        exit 1
    fi
}

# Function to run integration tests
run_integration_tests() {
    print_header "RUNNING INTEGRATION TESTS"
    
    if [ ! -f "build/archneuronx_tests" ]; then
        print_error "Test executable not found. Please build tests first."
        exit 1
    fi
    
    print_status "Running integration tests..."
    ./build/archneuronx_tests --gtest_filter="Integration*" --gtest_output=xml
    
    if [ $? -eq 0 ]; then
        print_status "Integration tests passed!"
    else
        print_error "Integration tests failed!"
        exit 1
    fi
}

# Function to run quantum tests
run_quantum_tests() {
    print_header "RUNNING QUANTUM TESTS"
    
    if [ ! -f "build/archneuronx_tests" ]; then
        print_error "Test executable not found. Please build tests first."
        exit 1
    fi
    
    print_status "Running quantum tests..."
    ./build/archneuronx_tests --gtest_filter="*Quantum*" --gtest_output=xml
    
    if [ $? -eq 0 ]; then
        print_status "Quantum tests passed!"
    else
        print_error "Quantum tests failed!"
        exit 1
    fi
}

# Function to run LLM tests
run_llm_tests() {
    print_header "RUNNING LLM TESTS"
    
    if [ ! -f "build/archneuronx_tests" ]; then
        print_error "Test executable not found. Please build tests first."
        exit 1
    fi
    
    print_status "Running LLM tests..."
    ./build/archneuronx_tests --gtest_filter="*LLM*" --gtest_output=xml
    
    if [ $? -eq 0 ]; then
        print_status "LLM tests passed!"
    else
        print_error "LLM tests failed!"
        exit 1
    fi
}

# Function to run agent tests
run_agent_tests() {
    print_header "RUNNING AGENT TESTS"
    
    if [ ! -f "build/archneuronx_tests" ]; then
        print_error "Test executable not found. Please build tests first."
        exit 1
    fi
    
    print_status "Running agent tests..."
    ./build/archneuronx_tests --gtest_filter="*Agent*" --gtest_output=xml
    
    if [ $? -eq 0 ]; then
        print_status "Agent tests passed!"
    else
        print_error "Agent tests failed!"
        exit 1
    fi
}

# Function to run performance tests
run_performance_tests() {
    print_header "RUNNING PERFORMANCE TESTS"
    
    if [ ! -f "build/archneuronx_tests" ]; then
        print_error "Test executable not found. Please build tests first."
        exit 1
    fi
    
    print_status "Running performance tests..."
    ./build/archneuronx_tests --gtest_filter="*Performance*" --gtest_output=xml
    
    if [ $? -eq 0 ]; then
        print_status "Performance tests passed!"
    else
        print_error "Performance tests failed!"
        exit 1
    fi
}

# Function to generate coverage report
generate_coverage() {
    print_header "GENERATING COVERAGE REPORT"
    
    if [ ! -f "build/archneuronx_tests" ]; then
        print_error "Test executable not found. Please build tests first."
        exit 1
    fi
    
    # Run tests with coverage
    print_status "Running tests with coverage..."
    ./build/archneuronx_tests --gtest_output=xml
    
    # Generate coverage report
    if command -v gcov &> /dev/null; then
        print_status "Generating coverage report..."
        
        cd build
        gcov ../src/**/*.cpp
        lcov --capture --directory . --output-file coverage.info
        lcov --remove coverage.info '/usr/*' --output-file coverage.info
        lcov --list coverage.info
        
        # Generate HTML report
        if command -v genhtml &> /dev/null; then
            genhtml coverage.info --output-directory coverage_html
            print_status "Coverage report generated in build/coverage_html/"
        else
            print_warning "genhtml not found. HTML report not generated."
        fi
        
        cd ..
        
        # Show coverage summary
        print_status "Coverage Summary:"
        lcov --summary coverage.info
        
    else
        print_warning "gcov not found. Coverage report not generated."
    fi
}

# Function to run memory tests
run_memory_tests() {
    print_header "RUNNING MEMORY TESTS"
    
    if [ ! -f "build/archneuronx_tests" ]; then
        print_error "Test executable not found. Please build tests first."
        exit 1
    fi
    
    # Check for valgrind
    if ! command -v valgrind &> /dev/null; then
        print_warning "Valgrind not found. Memory tests not available."
        return
    fi
    
    print_status "Running memory tests with Valgrind..."
    valgrind --tool=memcheck --leak-check=full ./build/archneuronx_tests --gtest_filter="*Quantum*" --gtest_output=xml
    
    if [ $? -eq 0 ]; then
        print_status "Memory tests completed!"
    else
        print_error "Memory tests failed!"
        exit 1
    fi
}

# Function to run stress tests
run_stress_tests() {
    print_header "RUNNING STRESS TESTS"
    
    if [ ! -f "build/archneuronx_tests" ]; then
        print_error "Test executable not found. Please build tests first."
        exit 1
    fi
    
    print_status "Running stress tests..."
    
    # Run tests multiple times to check for memory leaks
    for i in {1..10}; do
        print_status "Stress test iteration $i/10..."
        ./build/archneuronx_tests --gtest_filter="*Performance*" --gtest_output=xml
        
        if [ $? -ne 0 ]; then
            print_error "Stress test failed at iteration $i!"
            exit 1
        fi
    done
    
    print_status "Stress tests completed successfully!"
}

# Function to clean test build
clean_tests() {
    print_header "CLEANING TEST BUILD"
    
    if [ -d "build" ]; then
        print_status "Removing build directory..."
        rm -rf build
        print_status "✅ Build directory removed"
    else
        print_status "Build directory does not exist"
    fi
    
    # Clean test outputs
    if [ -d "test_results" ]; then
        print_status "Cleaning test results..."
        rm -rf test_results
        print_status "✅ Test results cleaned"
    fi
    
    # Clean coverage files
    if [ -d "coverage_html" ]; then
        print_status "Cleaning coverage files..."
        rm -rf coverage_html
        print_status "✅ Coverage files cleaned"
    fi
    
    # Clean gcov files
    find . -name "*.gcov" -delete 2>/dev/null || true
    find . -name "*.gcda" -delete 2>/dev/null || true
    find . -name "*.gcno" -delete 2>/dev/null || true
}

# Function to show help
show_help() {
    echo "ArchNeuronX v4.0 Test Runner"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build           Build the test suite"
    echo "  all             Run all tests"
    echo "  unit            Run unit tests only"
    echo "  integration     Run integration tests only"
    echo "  quantum         Run quantum tests only"
    echo "  llm             Run LLM tests only"
    echo "  agents          Run agent tests only"
    echo "  performance     Run performance tests only"
    echo "  coverage        Generate coverage report"
    echo "  memory          Run memory tests with Valgrind"
    echo "  stress          Run stress tests"
    echo "  clean           Clean test build and outputs"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Build tests"
    echo "  $0 all                      # Run all tests"
    echo "  $0 quantum                  # Run quantum tests"
    echo "  $0 coverage                 # Generate coverage report"
    echo ""
    echo "Test Categories:"
    echo "  🧠 Quantum Neural Networks"
    echo "  🤖 Quantum Trading Agents"
    echo "  🤖 LLM Integration"
    echo "  🌐 Web Interface"
    echo "  🤝 Multi-Agent Coordination"
    echo "  📊 Performance Benchmarks"
}

# Main script logic
main() {
    case "${1:-help}" in
        "build")
            check_dependencies
            build_tests
            ;;
        "all")
            run_all_tests
            ;;
        "unit")
            run_unit_tests
            ;;
        "integration")
            run_integration_tests
            ;;
        "quantum")
            run_quantum_tests
            ;;
        "llm")
            run_llm_tests
            ;;
        "agents")
            run_agent_tests
            ;;
        "performance")
            run_performance_tests
            ;;
        "coverage")
            generate_coverage
            ;;
        "memory")
            run_memory_tests
            ;;
        "stress")
            run_stress_tests
            ;;
        "clean")
            clean_tests
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function
main "$@"
