# ArchNeuronX v2.0 - Entry Points Documentation

## Overview

ArchNeuronX provides multiple entry points for different use cases. Each main file serves a specific purpose:

## Main Entry Points

### 1. `src/main.cpp` - **Primary CLI Interface**
**Purpose**: Complete command-line interface for all operations

**Usage**:
```bash
./archneuronx <command> [options]
```

**Commands**:
- `train` - Train neural network models
- `predict` - Generate trading signals
- `server` - Start REST API server  
- `backtest` - Run backtesting analysis

**When to use**: Production deployments, full-featured CLI operations

### 2. `src/main_http.cpp` - **HTTP Server Mode**
**Purpose**: Standalone HTTP server with REST endpoints

**Features**:
- Built-in HTTP server implementation
- 5 REST endpoints for trading operations
- Real-time signal streaming
- Web dashboard integration

**Endpoints**:
- `GET /api/v1/status` - System health check
- `GET /api/v1/signals` - Get trading signals
- `POST /api/v1/predict` - Generate prediction
- `GET /api/v1/portfolio` - Portfolio status
- `GET /api/v1/metrics` - Performance metrics

**When to use**: Dedicated API server deployments, microservices architecture

### 3. `src/main_simple.cpp` - **Minimal Testing Interface**
**Purpose**: Lightweight testing and development

**Features**:
- Minimal dependencies
- Fast startup time
- Basic server functionality
- Development debugging

**When to use**: Development, debugging, quick testing

## Build Targets

### CMake Configuration
```cmake
# Main CLI executable
add_executable(archneuronx src/main.cpp)

# HTTP server executable  
add_executable(archneuronx-server src/main_http.cpp)

# Simple testing executable
add_executable(archneuronx-simple src/main_simple.cpp)
```

### Docker Images

#### Production Image (with CUDA)
```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04
# Builds main.cpp with full features
```

#### CPU-only Image
```dockerfile  
FROM ubuntu:20.04
# Uses CMakeLists.txt with USE_CUDA=OFF
```

## Deployment Scenarios

### 1. Full Trading System
```bash
# Build
cmake -DUSE_CUDA=ON -DBUILD_TESTS=ON ..
make archneuronx

# Run training
./archneuronx train --model transformer --data btc_1h.csv

# Start prediction service
./archneuronx predict --model transformer --real-time

# Run backtest
./archneuronx backtest --model transformer --start 2023-01-01 --end 2023-12-31
```

### 2. API Server Deployment
```bash
# Build HTTP server
cmake -DUSE_CUDA=ON ..
make archneuronx-server

# Start API server
./archneuronx-server --port 8080 --metrics-port 9090
```

### 3. Development/Testing
```bash
# Quick build for testing
cmake -DUSE_CUDA=OFF ..
make archneuronx-simple

# Run simple server
./archneuronx-simple server
```

## Configuration Files

Each entry point respects the same configuration hierarchy:

1. `config/default.json` - Default settings
2. `config/production.json` - Production overrides  
3. `config/development.json` - Development settings
4. Command line arguments - Highest priority

## Environment Variables

Common across all entry points:

```bash
# CUDA configuration
USE_CUDA=1
CUDA_VISIBLE_DEVICES=0,1

# API configuration  
API_PORT=8080
API_HOST=0.0.0.0
API_KEY_REQUIRED=1

# Logging
LOG_LEVEL=INFO
LOG_DIR=/var/log/archneuronx

# Monitoring
PROMETHEUS_PORT=9090
METRICS_ENABLED=1
```

## Choosing the Right Entry Point

| Use Case | Recommended Entry Point | Reason |
|----------|------------------------|---------|
| Production trading system | `main.cpp` | Full CLI, all features |
| Microservices API | `main_http.cpp` | Dedicated HTTP server |
| Development/debugging | `main_simple.cpp` | Lightweight, fast startup |
| Docker container | `main_http.cpp` | Single process, easy containerization |
| Cloud deployment | `main_http.cpp` | HTTP-native, load balancer friendly |

## Migration Guide

### From v1.x to v2.0

**Old**: Single `main.cpp` with hardcoded server
**New**: Multiple specialized entry points

```bash
# v1.x
./archneuronx

# v2.0 equivalent
./archneuronx server
# or
./archneuronx-server
```

### Container Migration

**Before**:
```dockerfile
CMD ["./archneuronx"]
```

**After**:
```dockerfile
CMD ["./archneuronx-server", "--port", "8080"]
```

## Performance Considerations

- `main.cpp`: Full feature set, higher memory usage
- `main_http.cpp`: Optimized for HTTP throughput
- `main_simple.cpp`: Minimal footprint, fastest startup

Choose based on your specific deployment requirements and resource constraints.
