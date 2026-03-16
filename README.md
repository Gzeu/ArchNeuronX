# ArchNeuronX v2.0 - Automated Neural Network Trading System

[![Build Status](https://github.com/Gzeu/ArchNeuronX/workflows/CI/badge.svg)](https://github.com/Gzeu/ArchNeuronX/actions)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![LibTorch](https://img.shields.io/badge/LibTorch-2.6-orange.svg)](https://pytorch.org/cppdocs/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)

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

ArchNeuronX is a high-performance automated trading system leveraging neural networks (MLP/CNN/LSTM/Transformer) with GPU acceleration, real-time risk management, backtesting engine, portfolio optimization, and a full REST + WebSocket API.

### Tech Stack

- **C++20** - Core development language with concepts, coroutines, ranges
- **LibTorch 2.6** - PyTorch C++ API for neural networks
- **CUDA 12.4** - GPU acceleration for training and inference
- **Ubuntu 22.04** - Primary build environment (multi-stage Docker)
- **Docker** - Containerized multi-stage deployment
- **REST + WebSocket API** - Real-time trading integration
- **Kubernetes (k8s)** - Production orchestration

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

- **Real-time Data Processing** - Crypto & Forex APIs with WebSocket feeds
- **Neural Network Models** - MLP, CNN, LSTM, Transformer architectures
- **GPU Acceleration** - CUDA 12.4-enabled training and inference
- **Signal Generation** - Buy/sell/hold with confidence scores and VaR
- **Risk Management** - VaR, CVaR, position sizing, circuit breakers
- **Backtesting Engine** - Full historical simulation with Sharpe/Sortino metrics
- **Portfolio Optimization** - Mean-variance, Kelly criterion, rebalancing
- **REST + WebSocket API** - v2 endpoints for all subsystems
- **Automated Reports** - Visual performance analytics
- **CI/CD Pipeline** - GitHub Actions with security scanning
- **Kubernetes** - Production-grade deployment manifests

- C++20 compatible compiler (GCC 12+ or Clang 15+)
- CMake 3.20+
- LibTorch 2.6.0 (CUDA 12.4 build)
- CUDA 12.4+ Toolkit + cuDNN 9 (optional but recommended)
- Docker 24+ (for containerized deployment)
- OpenSSL, libcurl, Boost, spdlog, nlohmann/json

```
ArchNeuronX/
├── src/                    # Source code
│   ├── core/               # Core trading engine
│   ├── models/             # Neural network models (MLP/CNN/LSTM/Transformer)
│   ├── data/               # Data acquisition & preprocessing
│   ├── api/                # REST + WebSocket API endpoints
│   ├── risk/               # Risk management (VaR, circuit breakers)
│   ├── trading/            # Signal generation & execution
│   ├── backtest/           # Backtesting engine
│   ├── monitoring/         # System monitoring & metrics
│   └── utils/              # Utility functions
├── include/                # Header files
├── tests/                  # Unit and integration tests
├── k8s/                    # Kubernetes deployment manifests
├── docs/                   # Documentation
├── scripts/                # Build and deployment scripts
├── config/                 # Configuration files
├── .github/workflows/      # CI/CD pipelines
├── CMakeLists.txt          # C++20 + LibTorch 2.6 + CUDA 12.4
├── Dockerfile              # Multi-stage CUDA 12.4 + Ubuntu 22.04
└── docker-compose.yml      # Full stack compose
```

## 🔧 Quick Start

### Prerequisites

- C++20 compatible compiler (GCC 12+ or Clang 14+)
- CMake 3.25+
- LibTorch 2.6
- CUDA 12.4+ (optional, for GPU support)
- Docker 24+ (for containerized deployment)

### Building

```bash
git clone https://github.com/Gzeu/ArchNeuronX.git
cd ArchNeuronX

# Build with CMake (C++20 + LibTorch 2.6)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20
make -j$(nproc)

# Run tests
cd build && ctest --output-on-failure
```

### Docker (Recommended)

```bash
# Build multi-stage Docker image (CUDA 12.4 + Ubuntu 22.04)
docker build -t archneuronx:v2.0 .

# Run with GPU support
docker run --gpus all -p 8080:8080 -p 8081:8081 archneuronx:v2.0

# Full stack with docker-compose
docker-compose up -d
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/
```

## Usage

### Train a Transformer Model

```bash
./build/archneuronx train --config config/mlp_config.json --data data/crypto_data.csv --model lstm
```

### Generate Signals

```bash
./build/archneuronx predict \
  --model models/transformer_btc.pt \
  --input data/live_feed.csv \
  --output signals.json
```

### Backtesting

```bash
./build/archneuronx backtest --config config/backtest.json --from 2023-01-01 --to 2024-12-31
```

### API Server

```bash
./build/archneuronx server --port 8080 --ws-port 8081
```

## 🔌 API Endpoints (v2)

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v2/predict` | Generate trading signals |
| GET | `/api/v2/models` | List available models |
| POST | `/api/v2/train` | Start model training |
| GET | `/api/v2/status` | System health check |
| GET | `/api/v2/reports` | Performance reports |
| POST | `/api/v2/backtest` | Run backtesting |
| GET | `/api/v2/portfolio` | Portfolio status |
| POST | `/api/v2/risk/var` | Calculate Value at Risk |
| GET | `/api/v2/metrics` | Prometheus metrics |

### WebSocket

| Channel | Description |
|---------|-------------|
| `ws://host:8081/ws/signals` | Real-time trading signals |
| `ws://host:8081/ws/portfolio` | Portfolio updates |
| `ws://host:8081/ws/metrics` | Live performance metrics |

## API Authentication

| Model | Use Case | Inference Latency |
|-------|----------|------------------|
| **MLP** | Pattern recognition in time series | < 1ms |
| **CNN** | Feature extraction from OHLCV data | < 2ms |
| **LSTM** | Sequential market data prediction | < 5ms |
| **Transformer** | Attention-based multi-asset analysis | < 10ms |
| **Ensemble** | Hybrid model voting | < 15ms |
All endpoints require authentication via API key:

```http
X-API-Key: anx_your_api_key_here
```

- Accuracy, Precision, Recall, F1-Score
- Sharpe Ratio, Sortino Ratio, Maximum Drawdown
- Value at Risk (VaR 95%/99%), CVaR
- Win Rate, Profit Factor, Risk-Adjusted Returns
- Real-time GPU inference latency
- Portfolio beta, alpha, correlation

```http
Authorization: Bearer your_jwt_token
```

- **GitHub Actions** - Automated C++20 builds and tests
- **Docker Hub** - Multi-arch container registry
- **Security Scans** - Trivy + CodeQL vulnerability assessment
- **Performance Tests** - Automated benchmarking with Google Benchmark
- **Dependabot** - Automated dependency updates

## 📖 Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api.md)
- [Model Training Guide](docs/training.md)
- [Deployment Guide](docs/deployment.md)
- [Risk Management](docs/risk_management.md)
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

For questions and support:

- Create an [issue](https://github.com/Gzeu/ArchNeuronX/issues)
- Discussion forum: [GitHub Discussions](https://github.com/Gzeu/ArchNeuronX/discussions)

---
**Built with ❤️ for algorithmic trading enthusiasts | v2.0 | C++20 | LibTorch 2.6 | CUDA 12.4**
MIT License - see [LICENSE](LICENSE)

---

**Built for algorithmic trading with neural networks - March 2026**
