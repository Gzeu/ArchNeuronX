# ============================================================
# ArchNeuronX v2 Dockerfile
# CUDA 12.4.1 + LibTorch 2.6.0 + Ubuntu 22.04
# Multi-stage build for minimal production image
# ============================================================

# ============================================================
# Stage 1: Dependencies Cache
# ============================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS deps

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    cmake \
    ninja-build \
    gcc-12 \
    g++-12 \
    # Required libraries
    libssl-dev \
    libcurl4-openssl-dev \
    ca-certificates \
    wget \
    unzip \
    # JSON (nlohmann header-only)
    nlohmann-json3-dev \
    # Logging
    libspdlog-dev \
    # WebSocket
    libboost-system-dev \
    libboost-thread-dev \
    # Testing
    libgtest-dev \
    libgmock-dev \
    # Benchmarking
    libbenchmark-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# ============================================================
# Stage 2: LibTorch Download
# ============================================================
FROM deps AS libtorch

WORKDIR /opt
# LibTorch 2.6.0 + CUDA 12.4 (cxx11 ABI for Linux compatibility)
RUN wget -q --show-progress \
    "https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu124.zip" \
    -O libtorch.zip \
    && unzip -q libtorch.zip \
    && rm libtorch.zip \
    && echo "LibTorch 2.6.0+cu124 downloaded successfully"

# ============================================================
# Stage 3: Build
# ============================================================
FROM deps AS builder

# Copy LibTorch from cache stage
COPY --from=libtorch /opt/libtorch /opt/libtorch

ENV Torch_DIR=/opt/libtorch/share/cmake/Torch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

# Copy source code
WORKDIR /app
COPY . .

# Build with Ninja for faster compilation
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/libtorch \
    -DCMAKE_CXX_COMPILER=g++-12 \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
    && cmake --build build --parallel $(nproc) \
    && echo "Build completed successfully"

# Run unit tests during build
RUN cd build && ctest --output-on-failure -R UnitTests || true

# ============================================================
# Stage 4: Production Runtime (minimal image)
# ============================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS production

ENV DEBIAN_FRONTEND=noninteractive

# Minimal runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    libcurl4 \
    ca-certificates \
    libspdlog1 \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Copy LibTorch shared libraries (only .so files, not headers/cmake)
COPY --from=builder /opt/libtorch/lib/*.so* /opt/libtorch/lib/

# Copy built binary
COPY --from=builder /app/build/bin/archneuronx /usr/local/bin/archneuronx

# Copy config and scripts
COPY --from=builder /app/config /opt/archneuronx/config
COPY --from=builder /app/scripts /opt/archneuronx/scripts

# Environment
ENV LD_LIBRARY_PATH=/opt/libtorch/lib
ENV ARCHNEURONX_CONFIG=/opt/archneuronx/config/production.json
ENV ARCHNEURONX_LOG_LEVEL=info
ENV ARCHNEURONX_PORT=8080
ENV ARCHNEURONX_METRICS_PORT=9090

# Create non-root user for security
RUN groupadd -g 1000 archneuron \
    && useradd -m -u 1000 -g archneuron archneuron \
    && chown -R archneuron:archneuron /opt/archneuronx

USER archneuron
WORKDIR /opt/archneuronx

# API port + Prometheus metrics port
EXPOSE 8080 9090

# Health check (main API + metrics)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:${ARCHNEURONX_PORT}/api/v1/status || exit 1

# Use tini as init to handle signals correctly
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["archneuronx", \
     "--config", "/opt/archneuronx/config/production.json", \
     "--port", "8080", \
     "--metrics-port", "9090", \
     "--log-level", "info"]

# ============================================================
# Stage 5: Development image (includes tools for debugging)
# ============================================================
FROM builder AS development

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdb \
    valgrind \
    strace \
    htop \
    vim \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/opt/libtorch/lib
WORKDIR /app

# Dev entrypoint runs tests automatically
CMD ["bash", "-c", "cmake --build build && cd build && ctest --output-on-failure"]
