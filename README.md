# ArchNeuronX - Automated Neural Network Trading System

[![Build Status](https://github.com/Gzeu/ArchNeuronX/workflows/CI/badge.svg)](https://github.com/Gzeu/ArchNeuronX/actions)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🚀 Overview

ArchNeuronX is an automated financial time series analysis system for algorithmic trading using neural networks (MLP/CNN) with GPU acceleration, CI/CD automation, Docker deployment, and REST API integration.

### Tech Stack
- **C++17** - Core development language
- **LibTorch 2.1.0** - PyTorch C++ API for neural networks
- **CUDA** - GPU acceleration support
- **Arch Linux** - Primary development environment
- **Docker** - Containerized deployment
- **REST API** - External integration

## 🎯 Key Features

- **Real-time Data Processing** - Crypto & Forex APIs integration
- **Neural Network Models** - MLP and CNN architectures
- **GPU Acceleration** - CUDA-enabled training and inference
- **Signal Generation** - Buy/sell/hold recommendations with confidence scores
- **REST API** - Trading bot integration endpoints
- **Automated Reports** - Visual performance analytics
- **CI/CD Pipeline** - Automated testing and deployment

## 📋 Project Structure

```
ArchNeuronX/
├── src/                    # Source code
│   ├── core/              # Core trading engine
│   ├── models/            # Neural network models
│   ├── data/              # Data acquisition & preprocessing
│   ├── api/               # REST API endpoints
│   └── utils/             # Utility functions
├── include/               # Header files
├── tests/                 # Unit and integration tests
├── docker/                # Docker configuration
├── docs/                  # Documentation
├── scripts/               # Build and deployment scripts
├── config/                # Configuration files
└── data/                  # Sample data and models
```

## 🔧 Quick Start

### Prerequisites
- C++17 compatible compiler (GCC 9+ or Clang 10+)
- CMake 3.18+
- LibTorch 2.1.0
- CUDA 11.8+ (optional, for GPU support)
- Docker (for containerized deployment)

### Building

```bash
# Clone the repository
git clone https://github.com/Gzeu/ArchNeuronX.git
cd ArchNeuronX

# Build with CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

### Docker Deployment

```bash
# Build Docker image
docker build -t archneuronx .

# Run container
docker run -p 8080:8080 archneuronx
```

## 📊 Usage

### Training Models

```bash
./build/archneuronx train --config config/mlp_config.json --data data/crypto_data.csv
```

### Generating Signals

```bash
./build/archneuronx predict --model models/trained_model.pt --input data/live_feed.csv
```

### API Server

```bash
./build/archneuronx server --port 8080
```

## 🔌 API Endpoints

- `POST /api/v1/predict` - Generate trading signals
- `GET /api/v1/models` - List available models
- `POST /api/v1/train` - Start model training
- `GET /api/v1/status` - System health check
- `GET /api/v1/reports` - Performance reports

## 🧠 Supported Models

- **MLP (Multi-Layer Perceptron)** - For pattern recognition in time series
- **CNN (Convolutional Neural Network)** - For feature extraction from market data
- **Hybrid Models** - Combining MLP and CNN architectures

## 📈 Performance Metrics

- Accuracy, Precision, Recall, F1-Score
- Sharpe Ratio, Maximum Drawdown
- Win Rate, Risk-Adjusted Returns
- Real-time inference latency

## 🔄 CI/CD Pipeline

- **GitHub Actions** - Automated builds and tests
- **Docker Hub** - Container registry
- **Security Scans** - Vulnerability assessment
- **Performance Tests** - Automated benchmarking

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api.md)
- [Model Training Guide](docs/training.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions and support:
- Create an [issue](https://github.com/Gzeu/ArchNeuronX/issues)
- Discussion forum: [GitHub Discussions](https://github.com/Gzeu/ArchNeuronX/discussions)

---

**Built with ❤️ for algorithmic trading enthusiasts**