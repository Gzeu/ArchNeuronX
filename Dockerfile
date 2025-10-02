# Multi-stage build for ArchNeuronX
FROM archlinux:latest as builder

# Install build dependencies
RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
    base-devel \
    cmake \
    gcc \
    git \
    python \
    python-pip \
    cuda \
    cudnn && \
    pacman -Scc --noconfirm

# Install LibTorch
WORKDIR /opt
RUN wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip

# Set environment variables
ENV Torch_DIR=/opt/libtorch/share/cmake/Torch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

# Copy source code
WORKDIR /app
COPY . .

# Build the application
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch && \
    make -j$(nproc)

# Production stage
FROM archlinux:latest as production

# Install runtime dependencies
RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
    gcc-libs \
    cuda-runtime \
    curl && \
    pacman -Scc --noconfirm

# Copy LibTorch libraries
COPY --from=builder /opt/libtorch/lib /opt/libtorch/lib

# Copy built application
COPY --from=builder /app/build/archneuronx /usr/local/bin/
COPY --from=builder /app/config /opt/archneuronx/config
COPY --from=builder /app/scripts /opt/archneuronx/scripts

# Set environment variables
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/archneuronx/scripts:$PATH

# Create non-root user
RUN useradd -m -u 1000 archneuron && \
    chown -R archneuron:archneuron /opt/archneuronx

USER archneuron
WORKDIR /opt/archneuronx

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/status || exit 1

# Default command
CMD ["archneuronx", "server", "--port", "8080", "--config", "config/production.json"]