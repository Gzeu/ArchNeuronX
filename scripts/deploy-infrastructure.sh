#!/bin/bash
# ============================================================
# ArchNeuronX v4.0 - Infrastructure Deployment Script
# Automated deployment of complete infrastructure stack
# ============================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INFRA_DIR="$PROJECT_ROOT/infrastructure"
DEPLOYMENT_DIR="$PROJECT_ROOT/deployment"

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
    
    # Check required tools
    local tools=("terraform" "kubectl" "helm" "aws" "docker")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    # Check kubernetes cluster access
    if ! kubectl cluster-info &> /dev/null; then
        log_warning "Kubernetes cluster not accessible, will create new cluster"
    fi
    
    log_success "Prerequisites check completed"
}

# Initialize Terraform
init_terraform() {
    log "Initializing Terraform..."
    
    cd "$INFRA_DIR/terraform"
    
    # Initialize Terraform
    terraform init \
        -input=false \
        -backend-config="bucket=archneuronx-terraform-state" \
        -backend-config="key=v4/terraform.tfstate" \
        -backend-config="region=us-east-1" \
        -backend-config="encrypt=true" \
        -backend-config="dynamodb_table=terraform-locks"
    
    # Validate Terraform configuration
    terraform validate
    
    log_success "Terraform initialization completed"
}

# Plan Terraform deployment
plan_terraform() {
    log "Planning Terraform deployment..."
    
    cd "$INFRA_DIR/terraform"
    
    # Create execution plan
    terraform plan \
        -input=false \
        -out=tfplan \
        -var-file="environments/production.tfvars"
    
    log_success "Terraform plan created: tfplan"
}

# Apply Terraform deployment
apply_terraform() {
    log "Applying Terraform deployment..."
    
    cd "$INFRA_DIR/terraform"
    
    # Apply the plan
    terraform apply \
        -input=false \
        -auto-approve \
        tfplan
    
    # Get outputs
    terraform output -json > "$INFRA_DIR/terraform-outputs.json"
    
    log_success "Terraform deployment completed"
}

# Configure kubectl
configure_kubectl() {
    log "Configuring kubectl..."
    
    cd "$INFRA_DIR/terraform"
    
    # Get cluster endpoint and certificate
    local cluster_endpoint=$(terraform output -raw cluster_endpoint)
    local cluster_ca=$(terraform output -raw cluster_certificate_authority_data)
    local cluster_name=$(terraform output -raw cluster_name)
    
    # Update kubeconfig
    aws eks update-kubeconfig \
        --region us-east-1 \
        --name "$cluster_name"
    
    # Verify cluster access
    kubectl cluster-info
    kubectl get nodes
    
    log_success "kubectl configuration completed"
}

# Install Helm charts
install_helm_charts() {
    log "Installing Helm charts..."
    
    # Add required Helm repositories
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo add jetstack https://charts.jetstack.io
    helm repo add nvidia https://nvidia.github.io/gpu-operator
    helm repo update
    
    # Install NVIDIA GPU Operator
    log "Installing NVIDIA GPU Operator..."
    helm upgrade --install nvidia-gpu-operator nvidia/gpu-operator \
        --namespace gpu-operator \
        --create-namespace \
        --values "$INFRA_DIR/helm/gpu-operator-values.yaml" \
        --wait \
        --timeout 10m
    
    # Install Cert-Manager
    log "Installing Cert-Manager..."
    helm upgrade --install cert-manager jetstack/cert-manager \
        --namespace cert-manager \
        --create-namespace \
        --set installCRDs=true \
        --wait \
        --timeout 5m
    
    # Install Prometheus Stack
    log "Installing Prometheus monitoring stack..."
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --values "$INFRA_DIR/helm/prometheus-values.yaml" \
        --wait \
        --timeout 15m
    
    # Install Ingress-Nginx
    log "Installing Ingress-Nginx controller..."
    helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
        --namespace ingress-nginx \
        --create-namespace \
        --values "$INFRA_DIR/helm/ingress-nginx-values.yaml" \
        --wait \
        --timeout 10m
    
    # Install Elasticsearch and Kibana
    log "Installing Elasticsearch and Kibana..."
    helm upgrade --install elasticsearch elastic/elasticsearch \
        --namespace logging \
        --create-namespace \
        --values "$INFRA_DIR/helm/elasticsearch-values.yaml" \
        --wait \
        --timeout 15m
    
    helm upgrade --install kibana elastic/kibana \
        --namespace logging \
        --values "$INFRA_DIR/helm/kibana-values.yaml" \
        --wait \
        --timeout 10m
    
    log_success "Helm charts installation completed"
}

# Deploy ArchNeuronX v4.0 application
deploy_application() {
    log "Deploying ArchNeuronX v4.0 application..."
    
    # Create namespaces
    kubectl apply -f "$DEPLOYMENT_DIR/k8s/production/namespace.yaml"
    
    # Apply ConfigMaps and Secrets
    kubectl apply -f "$DEPLOYMENT_DIR/k8s/production/configmaps.yaml"
    kubectl apply -f "$DEPLOYMENT_DIR/k8s/production/secrets.yaml"
    
    # Apply storage classes
    kubectl apply -f "$DEPLOYMENT_DIR/k8s/production/storage-classes.yaml"
    
    # Apply Persistent Volume Claims
    kubectl apply -f "$DEPLOYMENT_DIR/k8s/production/pvcs.yaml"
    
    # Deploy the application
    kubectl apply -f "$DEPLOYMENT_DIR/k8s/production/archneuronx-v4-deployment.yaml"
    
    # Wait for deployments to be ready
    log "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=600s deployment/archneuronx-v4-gpu -n archneuronx-production
    kubectl wait --for=condition=available --timeout=600s deployment/archneuronx-v4-cpu -n archneuronx-production
    
    log_success "ArchNeuronX v4.0 deployment completed"
}

# Setup monitoring and alerting
setup_monitoring() {
    log "Setting up monitoring and alerting..."
    
    # Apply Service Monitors
    kubectl apply -f "$DEPLOYMENT_DIR/k8s/production/servicemonitors.yaml"
    
    # Apply Prometheus Rules
    kubectl apply -f "$DEPLOYMENT_DIR/k8s/production/prometheus-rules.yaml"
    
    # Apply Grafana Dashboards
    kubectl apply -f "$DEPLOYMENT_DIR/k8s/production/grafana-dashboards.yaml"
    
    # Apply AlertManager Config
    kubectl apply -f "$DEPLOYMENT_DIR/k8s/production/alertmanager-config.yaml"
    
    # Wait for monitoring components
    kubectl wait --for=condition=available --timeout=300s deployment/prometheus-kube-prometheus-stack-prometheus -n monitoring
    kubectl wait --for=condition=available --timeout=300s deployment/prometheus-kube-prometheus-stack-grafana -n monitoring
    
    log_success "Monitoring setup completed"
}

# Run smoke tests
run_smoke_tests() {
    log "Running smoke tests..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app=archneuronx-v4 -n archneuronx-production --timeout=300s
    
    # Test API endpoints
    local gpu_service_ip=$(kubectl get service archneuronx-v4-gpu-service -n archneuronx-production -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    local cpu_service_ip=$(kubectl get service archneuronx-v4-cpu-service -n archneuronx-production -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    # Test health endpoints
    log "Testing GPU service health endpoint..."
    if curl -f -s "http://$gpu_service_ip:8080/api/v4/health" > /dev/null; then
        log_success "GPU service health check passed"
    else
        log_error "GPU service health check failed"
        return 1
    fi
    
    log "Testing CPU service health endpoint..."
    if curl -f -s "http://$cpu_service_ip:8080/api/v4/health" > /dev/null; then
        log_success "CPU service health check passed"
    else
        log_error "CPU service health check failed"
        return 1
    fi
    
    # Test metrics endpoints
    log "Testing metrics endpoints..."
    if curl -f -s "http://$gpu_service_ip:9090/metrics" > /dev/null; then
        log_success "GPU metrics endpoint accessible"
    else
        log_warning "GPU metrics endpoint not accessible"
    fi
    
    log_success "Smoke tests completed"
}

# Run performance tests
run_performance_tests() {
    log "Running performance tests..."
    
    # Get service URLs
    local gpu_service_ip=$(kubectl get service archneuronx-v4-gpu-service -n archneuronx-production -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    # Test latency
    log "Testing API latency..."
    local latency_result=$(curl -w "%{time_total}" -s -o /dev/null "http://$gpu_service_ip:8080/api/v4/status")
    local latency_ms=$(echo "$latency_result * 1000" | bc)
    
    if (( $(echo "$latency_ms < 20" | bc -l) )); then
        log_success "Latency test passed: ${latency_ms}ms (<20ms target)"
    else
        log_warning "Latency test warning: ${latency_ms}ms (target <20ms)"
    fi
    
    # Test throughput
    log "Testing API throughput..."
    local start_time=$(date +%s.%N)
    
    # Send 100 concurrent requests
    for i in {1..100}; do
        curl -s "http://$gpu_service_ip:8080/api/v4/status" > /dev/null &
    done
    wait
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    local throughput=$(echo "100 / $duration" | bc)
    
    log_success "Throughput test: ${throughput} requests/second"
    
    log_success "Performance tests completed"
}

# Generate deployment report
generate_report() {
    log "Generating deployment report..."
    
    local report_file="$INFRA_DIR/deployment-report-$(date +%Y%m%d-%H%M%S).md"
    
    cat > "$report_file" << EOF
# ArchNeuronX v4.0 Deployment Report

**Deployment Date:** $(date)
**Environment:** Production

## Infrastructure Components

### Kubernetes Cluster
- **Region:** us-east-1
- **Version:** $(kubectl version --short | grep 'Server Version' | cut -d' ' -f3)
- **Node Count:** $(kubectl get nodes --no-headers | wc -l)

### Node Types
- **GPU Nodes:** $(kubectl get nodes --no-headers --selector=node-type=gpu | wc -l)
- **CPU Nodes:** $(kubectl get nodes --no-headers --selector=node-type=cpu | wc -l)
- **Service Nodes:** $(kubectl get nodes --no-headers --selector=node-type=service | wc -l)

### ArchNeuronX v4.0 Pods
- **GPU Pods:** $(kubectl get pods -n archneuronx-production --selector=component=gpu-inference --no-headers | wc -l)
- **CPU Pods:** $(kubectl get pods -n archneuronx-production --selector=component=cpu-execution --no-headers | wc -l)
- **Total Pods:** $(kubectl get pods -n archneuronx-production --no-headers | wc -l)

### Services
- **GPU Service:** $(kubectl get service archneuronx-v4-gpu-service -n archneuronx-production -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
- **CPU Service:** $(kubectl get service archneuronx-v4-cpu-service -n archneuronx-production -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
- **Ingress:** https://api.archneuronx.com

### Monitoring
- **Grafana:** https://grafana.archneuronx.com
- **Prometheus:** http://prometheus.archneuronx.com
- **AlertManager:** http://alertmanager.archneuronx.com

## Performance Metrics

### API Performance
- **Target Latency:** <20μs
- **Target Throughput:** >500K ops/sec
- **Actual Latency:** ${latency_ms}ms
- **Actual Throughput:** ${throughput} req/sec

### Resource Utilization
- **GPU Utilization:** $(kubectl top nodes --selector=node-type=gpu --no-headers | awk '{sum+=$2} END {print sum/NR}'%)
- **CPU Utilization:** $(kubectl top nodes --selector=node-type=cpu --no-headers | awk '{sum+=$2} END {print sum/NR}'%)
- **Memory Utilization:** $(kubectl top nodes --no-headers | awk '{sum+=$3} END {print sum/NR}'Mi)

## Health Status

### Pod Status
\`\`\`
$(kubectl get pods -n archneuronx-production -o wide)
\`\`\`

### Service Status
\`\`\`
$(kubectl get services -n archneuronx-production)
\`\`\`

### Node Status
\`\`\`
$(kubectl get nodes -o wide)
\`\`\`

## Next Steps

1. **Monitor Performance:** Keep an eye on latency and throughput metrics
2. **Scale as Needed:** Use HPA to scale based on load
3. **Backup Data:** Ensure regular backups of all data
4. **Security Updates:** Keep all components updated
5. **Performance Tuning:** Optimize based on actual usage patterns

## Troubleshooting

### Common Issues
1. **Pod Not Starting:** Check resource limits and node availability
2. **High Latency:** Check GPU utilization and network connectivity
3. **Scaling Issues:** Review HPA configuration and resource requests

### Commands
- **Check Pod Logs:** \`kubectl logs -f deployment/archneuronx-v4-gpu -n archneuronx-production\`
- **Check Events:** \`kubectl get events -n archneuronx-production --sort-by=.metadata.creationTimestamp\`
- **Scale Deployment:** \`kubectl scale deployment archneuronx-v4-gpu --replicas=4 -n archneuronx-production\`

---

**Report Generated:** $(date)
**Deployment Status:** Success
EOF

    log_success "Deployment report generated: $report_file"
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    rm -f "$INFRA_DIR/terraform/tfplan"
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log "Starting ArchNeuronX v4.0 infrastructure deployment..."
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Check prerequisites
    check_prerequisites
    
    # Initialize and apply Terraform
    init_terraform
    plan_terraform
    apply_terraform
    
    # Configure kubectl
    configure_kubectl
    
    # Install Helm charts
    install_helm_charts
    
    # Deploy application
    deploy_application
    
    # Setup monitoring
    setup_monitoring
    
    # Run tests
    run_smoke_tests
    run_performance_tests
    
    # Generate report
    generate_report
    
    log_success "ArchNeuronX v4.0 infrastructure deployment completed successfully!"
    log "Access the application at: https://api.archneuronx.com"
    log "Access Grafana at: https://grafana.archneuronx.com"
}

# Parse command line arguments
case "${1:-}" in
    "terraform-only")
        check_prerequisites
        init_terraform
        plan_terraform
        apply_terraform
        ;;
    "helm-only")
        install_helm_charts
        ;;
    "app-only")
        deploy_application
        setup_monitoring
        ;;
    "test-only")
        run_smoke_tests
        run_performance_tests
        ;;
    "cleanup")
        cd "$INFRA_DIR/terraform"
        terraform destroy -auto-approve
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [terraform-only|helm-only|app-only|test-only|cleanup|help]"
        echo "  terraform-only: Deploy only Terraform infrastructure"
        echo "  helm-only: Install only Helm charts"
        echo "  app-only: Deploy only ArchNeuronX application"
        echo "  test-only: Run only smoke and performance tests"
        echo "  cleanup: Destroy all infrastructure"
        echo "  help: Show this help message"
        exit 0
        ;;
    *)
        main
        ;;
esac
