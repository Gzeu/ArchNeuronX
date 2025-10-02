#!/bin/bash

# ArchNeuronX Setup Script
# This script sets up the development environment

set -e

echo "Setting up ArchNeuronX development environment..."

# Check if running on Arch Linux
if ! command -v pacman &> /dev/null; then
    echo "Warning: This script is optimized for Arch Linux"
fi

# Install system dependencies
echo "Installing system dependencies..."
sudo pacman -Syu --noconfirm
sudo pacman -S --noconfirm base-devel cmake gcc git python python-pip cuda cudnn

# Create directories
echo "Creating project directories..."
mkdir -p build logs data/models data/datasets config/models

# Download LibTorch if not present
if [ ! -d "libtorch" ]; then
    echo "Downloading LibTorch..."
    wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
    unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip
    rm libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip
fi

# Set environment variables
echo "Setting environment variables..."
export Torch_DIR=$(pwd)/libtorch/share/cmake/Torch
export LD_LIBRARY_PATH=$(pwd)/libtorch/lib:$LD_LIBRARY_PATH

# Build the project
echo "Building project..."
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=$(pwd)/../libtorch
make -j$(nproc)

echo "Setup completed successfully!"
echo "To run the application:"
echo "  cd build && ./archneuronx server"
echo "To run tests:"
echo "  cd build && ctest --output-on-failure"