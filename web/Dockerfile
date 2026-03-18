# ============================================================
# ArchNeuronX v4.0 - Web Application Dockerfile
# Quantum Neural Trading Dashboard & API Gateway
# ============================================================

FROM node:18-alpine

LABEL maintainer="ArchNeuronX Team"
LABEL version="4.0.0"
LABEL description="ArchNeuronX v4.0 - Quantum Neural Trading Web Dashboard"

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3000
ENV ARCHNEURONX_API_URL=http://archneuronx-engine:8080

# Install system dependencies
RUN apk add --no-cache \
    curl \
    && rm -rf /var/cache/apk/*

# Create app directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy application files
COPY . .

# Create logs directory
RUN mkdir -p logs

# Set permissions
RUN chown -R node:node /app
USER node

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Expose ports
EXPOSE 3000 3001

# Start the application
CMD ["npm", "start"]
