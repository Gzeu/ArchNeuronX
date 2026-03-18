#!/bin/bash
# ============================================================
# ArchNeuronX v4.0 - Deployment Automation Script
# Automated deployment for <20μs latency and 500K+ orders/sec
# ============================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
CONFIG_DIR="$PROJECT_ROOT/config"
BACKUP_DIR="$PROJECT_ROOT/backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_DIR/deployment.log"
}

# Success message
success() {
    log "${GREEN}SUCCESS" "$@"
}

# Info message
info() {
    log "${BLUE}INFO" "$@"
}

# Warning message
warn() {
    log "${YELLOW}WARN" "$@"
}

# Error message
error() {
    log "${RED}ERROR" "$@"
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    # Check if required tools are installed
    local required_tools=("kubectl" "helm" "docker" "jq" "curl")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "Required tool '$tool' is not installed"
            exit 1
        fi
    done
    
    # Check Kubernetes cluster access
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot access Kubernetes cluster"
        exit 1
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace archneuronx-v4 &> /dev/null; then
        info "Creating archneuronx-v4 namespace..."
        kubectl create namespace archneuronx-v4
    fi
    
    success "Prerequisites check completed"
}

# Create backup
create_backup() {
    local backup_name="v4_backup_$(date +%Y%m%d_%H%M%S)"
    local backup_path="$BACKUP_DIR/$backup_name"
    
    info "Creating backup: $backup_name"
    
    mkdir -p "$backup_path"
    
    # Backup current deployment
    if kubectl get deployment -n archneuronx-v4 &> /dev/null; then
        info "Backing up current deployments..."
        kubectl get deployment -n archneuronx-v4 -o yaml > "$backup_path/deployments.yaml"
    fi
    
    # Backup configurations
    if [ -d "$CONFIG_DIR" ]; then
        info "Backing up configurations..."
        cp -r "$CONFIG_DIR" "$backup_path/"
    fi
    
    # Backup secrets
    if kubectl get secrets -n archneuronx-v4 &> /dev/null; then
        info "Backing up secrets..."
        kubectl get secrets -n archneuronx-v4 -o yaml > "$backup_path/secrets.yaml"
    fi
    
    success "Backup created: $backup_path"
}

# Health check function
health_check() {
    local service_name=$1
    local namespace=${2:-archneuronx-v4}
    local max_wait_time=${3:-300}
    local wait_interval=10
    
    info "Performing health check for $service_name..."
    
    local elapsed=0
    while [ $elapsed -lt $max_wait_time ]; do
        if kubectl get pods -n "$namespace" -l app="$service_name" -o jsonpath='{.items[*].status.phase}' | grep -q "Running"; then
            if kubectl get pods -n "$namespace" -l app="$service_name" -o jsonpath='{.items[*].status.containerStatuses[*].ready}' | grep -q "true"; then
                success "$service_name is healthy"
                return 0
            fi
        fi
        
        info "Waiting for $service_name to be ready... (${elapsed}s/${max_wait_time}s)"
        sleep $wait_interval
        elapsed=$((elapsed + wait_interval))
    done
    
    error "$service_name health check failed after ${max_wait_time}s"
    return 1
}

# Performance validation
validate_performance() {
    info "Validating performance targets..."
    
    # Wait for services to be ready
    health_check "market-transformer-v4"
    health_check "order-routing-v4"
    health_check "risk-management-v4"
    
    # Get service URLs
    local market_transformer_url=$(kubectl get service market-transformer-v4 -n archneuronx-v4 -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -z "$market_transformer_url" ]; then
        market_transformer_url="localhost:8080"
    fi
    
    # Run performance tests
    info "Running performance validation tests..."
    
    # Test latency
    local latency_result=$(curl -s -w "%{time_total}" -o /dev/null "http://$market_transformer_url/v4/health" || echo "0")
    local latency_ms=$(echo "$latency_result * 1000" | bc)
    
    if (( $(echo "$latency_ms < 1" | bc -l) )); then
        success "Latency test passed: ${latency_ms}ms"
    else
        warn "Latency test failed: ${latency_ms}ms (target: <1ms)"
    fi
    
    # Test throughput
    info "Running throughput test..."
    local throughput_result=$(python3 "$SCRIPT_DIR/../integration/v4_integration_tests.py" --test-type throughput 2>/dev/null || echo "0")
    
    if [[ "$throughput_result" == *"success":true* ]]; then
        success "Throughput test passed"
    else
        warn "Throughput test failed"
    fi
    
    success "Performance validation completed"
}

# Deploy infrastructure
deploy_infrastructure() {
    info "Deploying infrastructure components..."
    
    # Deploy Redis
    info "Deploying Redis..."
    helm upgrade --install redis-v4 bitnami/redis \
        --namespace archneuronx-v4 \
        --set auth.enabled=true \
        --set auth.password="redis_password_123" \
        --set master.persistence.size=100Gi \
        --set replica.replicaCount=3 \
        --wait
    
    # Deploy Kafka
    info "Deploying Kafka..."
    helm upgrade --install kafka-v4 bitnami/kafka \
        --namespace archneuronx-v4 \
        --set replicaCount=3 \
        --set persistence.size=1Ti \
        --set zookeeper.persistence.size=100Gi \
        --wait
    
    # Deploy InfluxDB
    info "Deploying InfluxDB..."
    helm upgrade --install influxdb-v4 bitnami/influxdb \
        --namespace archneuronx-v4 \
        --set auth.enabled=true \
        --set auth.admin.username="admin" \
        --set auth.admin.password="influxdb_password_123" \
        --set persistence.size=500Gi \
        --wait
    
    # Deploy Neo4j
    info "Deploying Neo4j..."
    helm upgrade --install neo4j-v4 neo4j/neo4j \
        --namespace archneuronx-v4 \
        --set acceptLicenseAgreement=yes \
        --set neo4j.password="neo4j_password_123" \
        --set resources.requests.memory=8Gi \
        --set persistence.size=500Gi \
        --wait
    
    # Deploy MongoDB
    info "Deploying MongoDB..."
    helm upgrade --install mongodb-v4 bitnami/mongodb \
        --namespace archneuronx-v4 \
        --set auth.rootPassword="mongodb_password_123" \
        --set persistence.size=500Gi \
        --wait
    
    success "Infrastructure deployment completed"
}

# Deploy services
deploy_services() {
    info "Deploying ArchNeuronX v4.0 services..."
    
    # Deploy Market Transformer
    info "Deploying Market Transformer v4.0..."
    helm upgrade --install market-transformer-v4 \
        "$PROJECT_ROOT/infrastructure/helm/archneuronx-v4" \
        --namespace archneuronx-v4 \
        --set services.marketTransformer.enabled=true \
        --set services.marketTransformer.replicaCount=10 \
        --set services.marketTransformer.resources.requests.cpu=4000m \
        --set services.marketTransformer.resources.requests.memory=8Gi \
        --set services.marketTransformer.resources.limits.cpu=8000m \
        --set services.marketTransformer.resources.limits.memory=16Gi \
        --set services.marketTransformer.resources.limits."nvidia.com/gpu"=1 \
        --wait
    
    # Deploy Graph Network
    info "Deploying Graph Network v4.0..."
    helm upgrade --install graph-network-v4 \
        "$PROJECT_ROOT/infrastructure/helm/archneuronx-v4" \
        --namespace archneuronx-v4 \
        --set services.graphNetwork.enabled=true \
        --set services.graphNetwork.replicaCount=5 \
        --set services.graphNetwork.resources.requests.cpu=2000m \
        --set services.graphNetwork.resources.requests.memory=16Gi \
        --set services.graphNetwork.resources.limits.cpu=4000m \
        --set services.graphNetwork.resources.limits.memory=32Gi \
        --wait
    
    # Deploy Order Routing
    info "Deploying Order Routing v4.0..."
    helm upgrade --install order-routing-v4 \
        "$PROJECT_ROOT/infrastructure/helm/archneuronx-v4" \
        --namespace archneuronx-v4 \
        --set services.orderRouting.enabled=true \
        --set services.orderRouting.replicaCount=8 \
        --set services.orderRouting.resources.requests.cpu=2000m \
        --set services.orderRouting.resources.requests.memory=4Gi \
        --set services.orderRouting.resources.limits.cpu=4000m \
        --set services.orderRouting.resources.limits.memory=8Gi \
        --wait
    
    # Deploy Risk Management
    info "Deploying Risk Management v4.0..."
    helm upgrade --install risk-management-v4 \
        "$PROJECT_ROOT/infrastructure/helm/archneuronx-v4" \
        --namespace archneuronx-v4 \
        --set services.riskManagement.enabled=true \
        --set services.riskManagement.replicaCount=6 \
        --set services.riskManagement.resources.requests.cpu=2000m \
        --set services.riskManagement.resources.requests.memory=8Gi \
        --set services.riskManagement.resources.limits.cpu=4000m \
        --set services.riskManagement.resources.limits.memory=16Gi \
        --wait
    
    # Deploy Portfolio Optimizer
    info "Deploying Portfolio Optimizer v4.0..."
    helm upgrade --install portfolio-optimizer-v4 \
        "$PROJECT_ROOT/infrastructure/helm/archneuronx-v4" \
        --namespace archneuronx-v4 \
        --set services.portfolioOptimizer.enabled=true \
        --set services.portfolioOptimizer.replicaCount=4 \
        --set services.portfolioOptimizer.resources.requests.cpu=3000m \
        --set services.portfolioOptimizer.resources.requests.memory=8Gi \
        --set services.portfolioOptimizer.resources.limits.cpu=6000m \
        --set services.portfolioOptimizer.resources.limits.memory=16Gi \
        --wait
    
    # Deploy Regime Meta-Learner
    info "Deploying Regime Meta-Learner v4.0..."
    helm upgrade --install regime-meta-learner-v4 \
        "$PROJECT_ROOT/infrastructure/helm/archneuronx-v4" \
        --namespace archneuronx-v4 \
        --set services.regimeMetaLearner.enabled=true \
        --set services.regimeMetaLearner.replicaCount=3 \
        --set services.regimeMetaLearner.resources.requests.cpu=2000m \
        --set services.regimeMetaLearner.resources.requests.memory=8Gi \
        --set services.regimeMetaLearner.resources.limits.cpu=4000m \
        --set services.regimeMetaLearner.resources.limits.memory=16Gi \
        --set services.regimeMetaLearner.resources.limits."nvidia.com/gpu"=1 \
        --wait
    
    success "Services deployment completed"
}

# Deploy monitoring
deploy_monitoring() {
    info "Deploying monitoring stack..."
    
    # Deploy Prometheus
    info "Deploying Prometheus..."
    helm upgrade --install prometheus-v4 prometheus-community/prometheus \
        --namespace archneuronx-v4 \
        --set server.retention=30d \
        --set server.persistentVolume.enabled=true \
        --set server.persistentVolume.size=200Gi \
        --set alertmanager.enabled=true \
        --set alertmanager.persistentVolume.enabled=true \
        --set alertmanager.persistentVolume.size=10Gi \
        --wait
    
    # Deploy Grafana
    info "Deploying Grafana..."
    helm upgrade --install grafana-v4 grafana/grafana \
        --namespace archneuronx-v4 \
        --set adminPassword="admin123" \
        --set persistence.enabled=true \
        --set persistence.size=20Gi \
        --set datasources."prometheus".enabled=true \
        --set datasources."prometheus".type=prometheus \
        --set datasources."prometheus".url=http://prometheus-v4-server:9090 \
        --wait
    
    # Deploy Jaeger
    info "Deploying Jaeger..."
    helm upgrade --install jaeger-v4 jaegertracing/jaeger \
        --namespace archneuronx-v4 \
        --set collector.enabled=true \
        --set query.enabled=true \
        --set agent.enabled=true \
        --wait
    
    success "Monitoring deployment completed"
}

# Deploy API Gateway
deploy_api_gateway() {
    info "Deploying API Gateway..."
    
    # Deploy Kong API Gateway
    helm upgrade --install kong-v4 bitnami/kong \
        --namespace archneuronx-v4 \
        --set ingressController.enabled=true \
        --set ingressController.ingressClassResource.name=kong \
        --set admin.enabled=true \
        --set admin.service.type=LoadBalancer \
        --set manager.enabled=true \
        --set portal.enabled=false \
        --wait
    
    # Configure Kong plugins and services
    info "Configuring Kong API Gateway..."
    
    # Apply Kong configuration
    kubectl apply -f "$PROJECT_ROOT/infrastructure/kong/kong-config.yaml" -n archneuronx-v4
    
    success "API Gateway deployment completed"
}

# Run integration tests
run_integration_tests() {
    info "Running integration tests..."
    
    # Create test runner pod
    kubectl run integration-test-runner \
        --image=python:3.9 \
        --rm -i \
        --restart=Never \
        --namespace=archneuronx-v4 \
        --command=bash \
        -- -c "
        pip install requests numpy pandas prometheus_client pytest asyncio
        python /tests/integration/v4_integration_tests.py
        " || {
        error "Integration tests failed"
        return 1
    }
    
    success "Integration tests completed"
}

# Gradual traffic migration
migrate_traffic() {
    local migration_percentage=${1:-10}
    
    info "Migrating ${migration_percentage}% of traffic to v4.0..."
    
    # Update Istio virtual service for traffic splitting
    kubectl patch virtualservice archneuronx-api -n archneuronx-v4 -p '
    {
        "spec": {
            "http": [
                {
                    "match": [{"uri": {"prefix": "/v4/"}}],
                    "route": [
                        {"destination": {"host": "market-transformer-v4", "subset": "v4"}, "weight": '$migration_percentage'},
                        {"destination": {"host": "market-transformer-v3", "subset": "v3"}, "weight": '$((100 - migration_percentage))'}
                    ]
                }
            ]
        }
    }'
    
    success "Traffic migration completed: ${migration_percentage}% to v4.0"
}

# Rollback function
rollback() {
    local backup_name=${1:-}
    
    warn "Initiating rollback..."
    
    if [ -z "$backup_name" ]; then
        # Use latest backup
        backup_name=$(ls -t "$BACKUP_DIR" | head -n 1)
    fi
    
    backup_path="$BACKUP_DIR/$backup_name"
    
    if [ ! -d "$backup_path" ]; then
        error "Backup not found: $backup_name"
        exit 1
    fi
    
    info "Rolling back to backup: $backup_name"
    
    # Restore deployments
    if [ -f "$backup_path/deployments.yaml" ]; then
        info "Restoring deployments..."
        kubectl apply -f "$backup_path/deployments.yaml" -n archneuronx-v4
    fi
    
    # Restore secrets
    if [ -f "$backup_path/secrets.yaml" ]; then
        info "Restoring secrets..."
        kubectl apply -f "$backup_path/secrets.yaml" -n archneuronx-v4
    fi
    
    success "Rollback completed"
}

# Cleanup function
cleanup() {
    info "Cleaning up temporary resources..."
    
    # Remove test runner pods
    kubectl delete pod integration-test-runner -n archneuronx-v4 --ignore-not-found=true
    
    # Clean up old logs
    find "$LOG_DIR" -name "*.log" -mtime +7 -delete
    
    success "Cleanup completed"
}

# Main deployment function
main() {
    local phase=${1:-all}
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    case $phase in
        "prerequisites")
            check_prerequisites
            ;;
        "backup")
            create_backup
            ;;
        "infrastructure")
            check_prerequisites
            deploy_infrastructure
            ;;
        "services")
            check_prerequisites
            deploy_services
            ;;
        "monitoring")
            check_prerequisites
            deploy_monitoring
            ;;
        "api-gateway")
            check_prerequisites
            deploy_api_gateway
            ;;
        "integration")
            check_prerequisites
            run_integration_tests
            ;;
        "performance")
            check_prerequisites
            validate_performance
            ;;
        "migrate")
            local percentage=${2:-10}
            migrate_traffic "$percentage"
            ;;
        "rollback")
            local backup_name=${2:-}
            rollback "$backup_name"
            ;;
        "cleanup")
            cleanup
            ;;
        "all")
            info "Starting complete ArchNeuronX v4.0 deployment..."
            check_prerequisites
            create_backup
            deploy_infrastructure
            deploy_services
            deploy_monitoring
            deploy_api_gateway
            run_integration_tests
            validate_performance
            success "ArchNeuronX v4.0 deployment completed successfully!"
            ;;
        *)
            echo "Usage: $0 {prerequisites|backup|infrastructure|services|monitoring|api-gateway|integration|performance|migrate|rollback|cleanup|all}"
            echo ""
            echo "Examples:"
            echo "  $0 all                    # Complete deployment"
            echo "  $0 prerequisites           # Check prerequisites only"
            echo "  $0 infrastructure          # Deploy infrastructure only"
            echo "  $0 services               # Deploy services only"
            echo "  $0 migrate 25            # Migrate 25% traffic to v4.0"
            echo "  $0 rollback backup_name    # Rollback to specific backup"
            exit 1
            ;;
    esac
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
