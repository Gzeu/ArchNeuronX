# ArchNeuronX v2.0 - Automated Neural Network Trading System

[![CI/CD Pipeline](https://github.com/Gzeu/ArchNeuronX/actions/workflows/ci.yml/badge.svg)](https://github.com/Gzeu/ArchNeuronX/actions)
[![Docker](https://img.shields.io/badge/Docker-CUDA%2012.4-blue)](https://hub.docker.com/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-orange)](https://en.cppreference.com/)
[![LibTorch](https://img.shields.io/badge/LibTorch-2.6.0-red)](https://pytorch.org/cppdocs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900)](https://developer.nvidia.com/cuda-toolkit)

> High-performance algorithmic trading system using neural networks (MLP, CNN, **Transformer**), GPU acceleration, and comprehensive risk management.

## What's New in v2.0

| Feature | v1.x | v2.0 |
|---------|------|------|
| C++ Standard | C++17 | **C++20** |
| LibTorch | 2.1.0 | **2.6.0** |
| CUDA | 11.8+ | **12.4** |
| Models | MLP, CNN | MLP, CNN, **Transformer, TFT** |
| Technical Indicators | Basic | **RSI, MACD, BB, ATR, OBV, VWAP, Stochastic, CCI, ADX** |
| Risk Management | None | **VaR, Kelly sizing, Stop-loss, Circuit breaker** |
| API Endpoints | 5 | **10+ including /backtest, /portfolio, /risk** |
| WebSocket | None | **Real-time market + signal streams** |
| Auth | None | **API keys + rate limiting (token bucket)** |
| Docker | Single stage | **Multi-stage, non-root, <500MB runtime** |
| Kubernetes | None | **Deployment, HPA, PDB, GPU support** |
| CI/CD | Basic | **Matrix builds, CodeQL, Trivy, Codecov** |
| Logging | std::cout | **spdlog structured logging** |
| Metrics | None | **Prometheus /metrics endpoint** |

## Overview

ArchNeuronX is a production-grade financial time series analysis system for algorithmic trading. It leverages neural networks with CUDA GPU acceleration to generate buy/sell/hold signals for crypto, forex, and equity markets.

## Tech Stack

- **C++20** - Core language with coroutines, concepts, ranges
- **LibTorch 2.6.0** - PyTorch C++ API with Flash Attention
- **CUDA 12.4 + cuDNN 9** - GPU acceleration + Tensor Cores
- **Docker** - Multi-stage build, ~450MB runtime image
- **Kubernetes** - Production deployment with HPA + GPU scheduling
- **spdlog** - Structured JSON logging
- **Prometheus** - Metrics exposition
- **nlohmann/json** - JSON serialization
- **OpenSSL** - API authentication

## Key Features

### Neural Network Models
- **MLP** - Multi-layer perceptron for fast signal generation
- **CNN** - Convolutional feature extraction from price patterns
- **Transformer** - Multi-head attention with Flash Attention + RoPE
- **TFT** - Temporal Fusion Transformer for multi-horizon forecasting
- **Ensemble** - Dynamic model weighting by recent performance
- **Online Learning** - Incremental updates without full retraining

### Technical Indicators (GPU-accelerated)
```
Moving Averages : SMA, EMA, WMA
Momentum        : RSI(14), MACD(12,26,9), Stochastic, Williams %R, ROC
Volatility      : Bollinger Bands(20,2), ATR(14), Historical Volatility
Volume          : OBV, VWAP, ADL, Chaikin Money Flow
Trend           : CCI, ADX
Feature Matrix  : 30+ features, z-score normalized, [T, F] tensor
```

### Risk Management
- **Position Sizing**: Kelly Criterion, Half-Kelly, Volatility-adjusted, Risk Parity
- **Stop-Loss**: Fixed %, ATR-based dynamic, Trailing stop
- **VaR**: Historical simulation, Parametric, 95%/99% confidence
- **Circuit Breaker**: Auto-halt if drawdown > 15%
- **Trade Validation**: Exposure limits, correlation checks, regime filtering
- **Metrics**: Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, Profit Factor

### REST API v2
```
POST /api/v1/predict          # Generate trading signals
POST /api/v1/train            # Start model training
GET  /api/v1/models           # List available models
GET  /api/v1/status           # System health
GET  /api/v1/reports          # Performance reports
POST /api/v1/backtest         # Strategy backtesting NEW
GET  /api/v1/portfolio        # Portfolio state NEW
POST /api/v1/portfolio/rebalance # Rebalancing NEW
GET  /api/v1/risk             # Risk metrics NEW
GET  /api/v1/risk/var         # VaR calculation NEW
GET  /api/v1/docs             # OpenAPI 3.1 spec NEW

WS   /ws/v1/market            # Real-time market data NEW
WS   /ws/v1/signals           # Real-time signals NEW
```

**Authentication**: `X-API-Key: <your_key>` header  
**Rate Limiting**: 100 req/min per key (token bucket algorithm)

## Project Structure

```
ArchNeuronX/
├── src/
│   ├── core/           # Trading engine, risk manager, portfolio, ensemble
│   ├── models/         # MLP, CNN, Transformer, TFT, online learner
│   ├── data/           # Market feeds, preprocessor, technical indicators
│   ├── api/            # REST handlers, WebSocket, auth, rate limiter
│   └── utils/          # Logger, metrics, CUDA utils, Prometheus
├── include/            # Public headers (mirrors src/)
├── tests/
│   ├── unit/           # Unit tests (GTest/GMock)
│   ├── integration/    # Pipeline integration tests
│   └── performance/    # Latency benchmarks
├── benchmarks/         # Google Benchmark suites
├── k8s/                # Kubernetes manifests NEW
│   └── deployment.yaml # Deployment, Service, HPA, PDB
├── docs/               # Architecture, API, training, deployment guides
├── config/             # JSON configuration files
├── scripts/            # Build, deployment, data scripts
└── .github/workflows/  # CI/CD pipelines
```

## Prerequisites

- C++20 compatible compiler (GCC 12+ or Clang 15+)
- CMake 3.20+
- LibTorch 2.6.0 (CUDA 12.4 build)
- CUDA 12.4+ Toolkit + cuDNN 9 (optional but recommended)
- Docker 24+ (for containerized deployment)
- OpenSSL, libcurl, Boost, spdlog, nlohmann/json

## Quick Start

### Build from Source

```bash
git clone https://github.com/Gzeu/ArchNeuronX.git
cd ArchNeuronX

# Download LibTorch 2.6.0 + CUDA 12.4
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.6.0+cu124.zip
unzip libtorch-*.zip -d /opt/

# Configure and build
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON \
  -DTorch_DIR=/opt/libtorch/share/cmake/Torch
cmake --build build --parallel $(nproc)

# Run tests
cd build && ctest --output-on-failure
```

### Docker (Recommended)

```bash
# Build
docker build -t archneuronx:2.0.0 .

# Run with GPU
docker run --gpus all -p 8080:8080 -p 9090:9090 archneuronx:2.0.0

# Run CPU-only
docker run -p 8080:8080 archneuronx:2.0.0
```

### Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl -n trading get pods
kubectl -n trading port-forward svc/archneuronx 8080:80
```

## Usage

### Train a Transformer Model

```bash
./build/archneuronx train \
  --model transformer \
  --config config/transformer_config.json \
  --data data/BTCUSDT_1h.csv \
  --epochs 100 \
  --device cuda
```

### Generate Signals

```bash
./build/archneuronx predict \
  --model models/transformer_btc.pt \
  --input data/live_feed.csv \
  --output signals.json
```

### Run Backtest via API

```bash
curl -X POST http://localhost:8080/api/v1/backtest \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": "transformer",
    "symbol": "BTCUSDT",
    "start_date": "2024-01-01",
    "end_date": "2025-12-31",
    "initial_capital": 10000,
    "include_costs": true
  }'
```

### Start API Server

```bash
./build/archneuronx server \
  --port 8080 \
  --metrics-port 9090 \
  --log-level info
```

## API Authentication

All endpoints require authentication via API key:

```http
X-API-Key: anx_your_api_key_here
```

Or Bearer token:

```http
Authorization: Bearer your_jwt_token
```

## Performance

| Operation | CPU (i9-13900K) | GPU (RTX 4090) |
|-----------|-----------------|----------------|
| MLP inference (batch=1) | ~0.3ms | ~0.05ms |
| Transformer inference (seq=64, batch=1) | ~2ms | ~0.3ms |
| Feature matrix computation (30 indicators, T=1000) | ~15ms | ~1.5ms |
| Full pipeline (data→features→model→signal) | ~25ms | ~3ms |

## Broker Integration Examples

| Broker | API Type | Example Config |
|--------|----------|----------------|
| Binance | REST + WebSocket | `config/binance_config.json` |
| Coinbase Advanced | REST + WebSocket | `config/coinbase_config.json` |
| Interactive Brokers | TWS API | `config/ibkr_config.json` |
| Kraken | REST + WebSocket | `config/kraken_config.json` |
| Bybit | REST + WebSocket | `config/bybit_config.json` |

## Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api.md)
- [Model Training Guide](docs/training.md)
- [Deployment Guide](docs/deployment.md)
- [Risk Management Guide](docs/risk_management.md) NEW
- [Contributing Guidelines](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Fork and clone
git checkout -b feature/my-improvement

# Make changes, ensure tests pass
cmake --build build && cd build && ctest

# Submit PR against develop branch
```

## License

MIT License - see [LICENSE](LICENSE)

---

**Built for algorithmic trading with neural networks - March 2026**
