#!/bin/bash
# ============================================================
# ArchNeuronX v4.0 - Web Application Deployment Script
# Quantum Neural Trading Dashboard Deployment
# ============================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WEB_DIR="$PROJECT_ROOT/web"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if ArchNeuronX engine is running
    if ! curl -f http://localhost:8080/api/v4/health &> /dev/null; then
        log_warning "ArchNeuronX engine is not running on port 8080"
        log "Starting ArchNeuronX engine first..."
        start_archneuronx_engine
    fi
    
    log_success "Prerequisites check completed"
}

# Start ArchNeuronX engine if not running
start_archneuronx_engine() {
    log "Starting ArchNeuronX v4.0 engine..."
    
    cd "$PROJECT_ROOT"
    
    # Check if Docker image exists
    if ! docker images | grep -q "archneuronx:v4.0"; then
        log "Building ArchNeuronX v4.0 Docker image..."
        docker build -f Dockerfile.v4.0.simple -t archneuronx:v4.0 .
    fi
    
    # Start the container
    if ! docker ps | grep -q "archneuronx_v4_demo"; then
        log "Starting ArchNeuronX engine container..."
        docker run -d --name archneuronx_v4_engine -p 8080:8080 archneuronx:v4.0
    else
        log "ArchNeuronX engine is already running"
    fi
    
    # Wait for engine to be healthy
    log "Waiting for ArchNeuronX engine to be healthy..."
    local retries=0
    while [ $retries -lt 30 ]; do
        if curl -f http://localhost:8080/api/v4/health &> /dev/null; then
            log_success "ArchNeuronX engine is healthy"
            break
        fi
        sleep 2
        retries=$((retries + 1))
    done
    
    if [ $retries -eq 30 ]; then
        log_error "ArchNeuronX engine failed to become healthy"
        exit 1
    fi
}

# Install web dependencies
install_dependencies() {
    log "Installing web application dependencies..."
    
    cd "$WEB_DIR"
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        log "Installing Node.js dependencies..."
        npm install
    else
        log "Dependencies already installed"
    fi
    
    log_success "Dependencies installation completed"
}

# Build web application
build_web_application() {
    log "Building web application..."
    
    cd "$WEB_DIR"
    
    # Build Docker image
    log "Building web application Docker image..."
    docker build -t archneuronx-web:v4.0 .
    
    log_success "Web application build completed"
}

# Deploy web application
deploy_web_application() {
    log "Deploying web application..."
    
    cd "$PROJECT_ROOT"
    
    # Stop existing services
    log "Stopping existing services..."
    docker-compose -f docker-compose.web.yml down
    
    # Start new services
    log "Starting web application services..."
    docker-compose -f docker-compose.web.yml up -d
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    check_service_health
}

# Check service health
check_service_health() {
    log "Checking service health..."
    
    local services=("web-gateway:3000" "grafana:3002" "prometheus:9090")
    
    for service in "${services[@]}"; do
        local service_name=$(echo "$service" | cut -d: -f1)
        local service_port=$(echo "$service" | cut -d: -f2)
        
        log "Checking $service_name (port $service_port)..."
        
        local retries=0
        while [ $retries -lt 30 ]; do
            if curl -f "http://localhost:$service_port/health" &> /dev/null || \
               curl -f "http://localhost:$service_port" &> /dev/null; then
                log_success "$service_name is healthy"
                break
            fi
            sleep 2
            retries=$((retries + 1))
        done
        
        if [ $retries -eq 30 ]; then
            log_warning "$service_name may not be fully ready yet"
        fi
    done
}

# Run smoke tests
run_smoke_tests() {
    log "Running smoke tests..."
    
    # Test web gateway
    log "Testing web gateway..."
    if curl -f http://localhost:3000/health &> /dev/null; then
        log_success "Web gateway health check passed"
    else
        log_error "Web gateway health check failed"
        return 1
    fi
    
    # Test API proxy
    log "Testing API proxy..."
    if curl -f http://localhost:3000/api/v4/status &> /dev/null; then
        log_success "API proxy test passed"
    else
        log_error "API proxy test failed"
        return 1
    fi
    
    # Test dashboard
    log "Testing dashboard..."
    if curl -f http://localhost:3000/ | grep -q "ArchNeuronX v4.0"; then
        log_success "Dashboard test passed"
    else
        log_error "Dashboard test failed"
        return 1
    fi
    
    # Test WebSocket
    log "Testing WebSocket connection..."
    if curl -f http://localhost:3001 &> /dev/null; then
        log_success "WebSocket test passed"
    else
        log_warning "WebSocket test may need manual verification"
    fi
    
    log_success "Smoke tests completed"
}

# Show deployment status
show_deployment_status() {
    log "Deployment Status:"
    echo ""
    
    echo "🚀 Services:"
    docker-compose -f docker-compose.web.yml ps
    echo ""
    
    echo "📊 Service URLs:"
    echo "  - Web Dashboard: http://localhost:3000"
    echo "  - WebSocket: ws://localhost:3001"
    echo "  - Grafana: http://localhost:3002"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - ArchNeuronX API: http://localhost:8080"
    echo ""
    
    echo "📈 Monitoring:"
    echo "  - Grafana Dashboard: http://localhost:3002"
    echo "  - Prometheus Metrics: http://localhost:9090"
    echo ""
    
    echo "🔧 Logs:"
    echo "  - Web Gateway: docker logs archneuronx_web_gateway"
    echo "  - ArchNeuronX Engine: docker logs archneuronx_v4_engine"
    echo "  - Grafana: docker logs archneuronx_grafana"
    echo "  - Prometheus: docker logs archneuronx_prometheus"
    echo ""
    
    echo "⚡ Quick Commands:"
    echo "  - View logs: docker-compose -f docker-compose.web.yml logs -f"
    echo "  - Stop services: docker-compose -f docker-compose.web.yml down"
    echo "  - Restart services: docker-compose -f docker-compose.web.yml restart"
    echo "  - Update services: docker-compose -f docker-compose.web.yml pull && docker-compose -f docker-compose.web.yml up -d"
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    # Add any cleanup tasks here
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log "Starting ArchNeuronX v4.0 Web Application Deployment..."
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Check prerequisites
    check_prerequisites
    
    # Install dependencies
    install_dependencies
    
    # Build web application
    build_web_application
    
    # Deploy web application
    deploy_web_application
    
    # Run smoke tests
    run_smoke_tests
    
    # Show deployment status
    show_deployment_status
    
    log_success "ArchNeuronX v4.0 Web Application deployment completed successfully!"
    log "🌐 Access the dashboard at: http://localhost:3000"
    log "📊 Access Grafana at: http://localhost:3002"
    log "🔍 Access Prometheus at: http://localhost:9090"
}

# Parse command line arguments
case "${1:-}" in
    "build-only")
        check_prerequisites
        install_dependencies
        build_web_application
        ;;
    "deploy-only")
        deploy_web_application
        ;;
    "test-only")
        run_smoke_tests
        ;;
    "status")
        show_deployment_status
        ;;
    "cleanup")
        cd "$PROJECT_ROOT"
        docker-compose -f docker-compose.web.yml down
        ;;
    "restart")
        cd "$PROJECT_ROOT"
        docker-compose -f docker-compose.web.yml restart
        ;;
    "logs")
        cd "$PROJECT_ROOT"
        docker-compose -f docker-compose.web.yml logs -f
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [build-only|deploy-only|test-only|status|cleanup|restart|logs|help]"
        echo "  build-only     - Build the web application without deploying"
        echo "  deploy-only    - Deploy the web application without building"
        echo "  test-only      - Run smoke tests only"
        echo "  status         - Show deployment status"
        echo "  cleanup        - Stop and remove all services"
        echo "  restart        - Restart all services"
        echo "  logs           - Show logs for all services"
        echo "  help           - Show this help message"
        exit 0
        ;;
    *)
        main
        ;;
esac
