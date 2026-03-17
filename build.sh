#!/bin/bash

# ArchNeuronX Build Script
# Usage: ./build.sh [release|debug] [cuda|cpu]

set -e

# Default parameters
BUILD_TYPE="Release"
CUDA_SUPPORT="ON"

# Parse arguments
if [ "$1" = "debug" ]; then
    BUILD_TYPE="Debug"
fi

if [ "$2" = "cpu" ]; then
    CUDA_SUPPORT="OFF"
fi

echo "🚀 Building ArchNeuronX v2.0"
echo "Build Type: $BUILD_TYPE"
echo "CUDA Support: $CUDA_SUPPORT"
echo "=================================="

# Check dependencies
echo "📋 Checking dependencies..."

# Check CMake
if ! command -v cmake &> /dev/null; then
    echo "❌ CMake not found. Please install CMake 3.20+"
    exit 1
fi

# Check CUDA if enabled
if [ "$CUDA_SUPPORT" = "ON" ]; then
    if ! command -v nvcc &> /dev/null; then
        echo "❌ CUDA compiler not found. Please install CUDA 11.8+"
        exit 1
    fi
    
    # Check GPU
    if ! command -v nvidia-smi &> /dev/null; then
        echo "⚠️  nvidia-smi not found. GPU may not be available."
    else
        echo "✅ GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    fi
fi

# Create build directory
BUILD_DIR="build-$BUILD_TYPE"
echo "📁 Creating build directory: $BUILD_DIR"
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR

# Configure
echo "⚙️  Configuring build..."
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE -DUSE_CUDA=$CUDA_SUPPORT"

if [ "$BUILD_TYPE" = "Release" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON"
fi

cmake .. $CMAKE_ARGS

# Build
echo "🔨 Building..."
make -j$(nproc)

# Run tests if available
if [ -f "test/archneuronx_test" ]; then
    echo "🧪 Running tests..."
    ./test/archneuronx_test
fi

echo ""
echo "✅ Build completed successfully!"
echo "📦 Binary location: $BUILD_DIR/src/archneuronx"
echo ""
echo "🎯 Quick start commands:"
echo "  CPU-only:     ./archneuronx --config ../config/profiles/development.json"
echo "  With CUDA:    ./archneuronx --config ../config/deployment.json --cuda"
echo "  Backtest:      ./archneuronx --backtest --data ../data/sample/"
echo "  Help:          ./archneuronx --help"
