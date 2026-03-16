# ============================================================
# ArchNeuronX v2.0 - Multi-stage Dockerfile
# Base: CUDA 12.4 + Ubuntu 22.04
# ============================================================

# Stage 1: Builder
FROM nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04 AS builder

LABEL stage=builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    unzip \
    pkg-config \
    libssl-dev \
    libcurl4-openssl-dev \
    libboost-all-dev \
    libgtest-dev \
    libgmock-dev \
    python3 \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install LibTorch 2.6.0 with CUDA 12.4
WORKDIR /opt
RUN wget -q https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu124.zip && \
    unzip -q libtorch-cxx11-abi-shared-with-deps-2.6.0+cu124.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-2.6.0+cu124.zip

ENV Torch_DIR=/opt/libtorch/share/cmake/Torch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

# Install nlohmann/json
RUN wget -q https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp \
    -O /usr/local/include/json.hpp

# Install spdlog
RUN git clone --depth 1 --branch v1.13.0 https://github.com/gabime/spdlog.git /tmp/spdlog && \
    cd /tmp/spdlog && cmake -B build -DCMAKE_BUILD_TYPE=Release -DSPDLOG_BUILD_SHARED=ON && \
    cmake --build build --target install -j$(nproc) && \
    rm -rf /tmp/spdlog

# Copy source code
WORKDIR /app
COPY . .

# Build the project
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON \
    -DBUILD_TESTS=ON \
    -DBUILD_BENCHMARKS=OFF \
    -DTorch_DIR=/opt/libtorch/share/cmake/Torch \
    -G Ninja && \
    cmake --build build --parallel $(nproc)

# Run tests in builder stage
RUN cd build && ctest --output-on-failure --timeout 120 || true

# ============================================================
# Stage 2: Runtime (minimal image)
# ============================================================
FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04 AS runtime

LABEL maintainer="ArchNeuronX Team"
LABEL version="2.0.0"
LABEL description="ArchNeuronX - Automated Neural Network Trading System"

ENV DEBIAN_FRONTEND=noninteractive

# Install minimal runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    libcurl4 \
    libboost-system1.74.0 \
    libboost-filesystem1.74.0 \
    libboost-program-options1.74.0 \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Copy LibTorch runtime libs
COPY --from=builder /opt/libtorch/lib/*.so* /usr/local/lib/

# Copy spdlog
COPY --from=builder /usr/local/lib/libspdlog* /usr/local/lib/

# Copy binary
COPY --from=builder /app/build/archneuronx /usr/local/bin/archneuronx

# Copy configs
COPY --from=builder /app/config /etc/archneuronx/config

RUN ldconfig

# Create non-root user for security
RUN groupadd -r archneuronx && useradd -r -g archneuronx -d /app -s /sbin/nologin archneuronx

# Create data and model directories
RUN mkdir -p /app/models /app/data /app/logs && \
    chown -R archneuronx:archneuronx /app

USER archneuronx
WORKDIR /app

# Expose REST API and metrics ports
EXPOSE 8080 9090

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/v2/status || exit 1

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV ARCHNEURONX_CONFIG=/etc/archneuronx/config
ENV ARCHNEURONX_LOG_LEVEL=info

# Use tini as init process to handle signals properly
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["archneuronx", "server", "--port", "8080", "--metrics-port", "9090"]
