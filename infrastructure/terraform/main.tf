# ============================================================
# ArchNeuronX v4.0 - Infrastructure as Code
# Terraform Configuration for Multi-Region Deployment
# ============================================================

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.2"
    }
  }
  
  backend "s3" {
    bucket = "archneuronx-terraform-state"
    key    = "v4/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
    dynamodb_table = "terraform-locks"
  }
}

# ============================================================
# Provider Configuration
# ============================================================

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "ArchNeuronX"
      Environment = var.environment
      Version     = "v4.0"
      ManagedBy   = "Terraform"
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

# ============================================================
# VPC Configuration
# ============================================================

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  
  name = "${var.project_name}-vpc"
  cidr = var.vpc_cidr
  
  azs             = local.availability_zones
  private_subnets = [for i, az in local.availability_zones : cidrsubnet(var.vpc_cidr, 8, i * 2)]
  public_subnets  = [for i, az in local.availability_zones : cidrsubnet(var.vpc_cidr, 8, i * 2 + 1)]
  
  enable_nat_gateway   = true
  enable_vpn_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  # High-performance networking for trading
  private_subnet_tags = {
    "kubernetes.io/cluster/${module.eks.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"                  = "1"
    "NetworkPerformance"                             = "High"
  }
  
  public_subnet_tags = {
    "kubernetes.io/cluster/${module.eks.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                         = "1"
    "NetworkPerformance"                             = "High"
  }
  
  tags = {
    Environment = var.environment
    Version     = "v4.0"
  }
}

# ============================================================
# EKS Cluster Configuration
# ============================================================

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"
  
  cluster_name    = "${var.project_name}-eks"
  cluster_version = var.kubernetes_version
  
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets
  
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true
  
  # Cluster add-ons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
  
  # Managed node groups for different workloads
  node_groups = {
    
    # GPU nodes for neural network inference
    gpu_nodes = {
      desired_capacity = var.gpu_node_count
      max_capacity     = var.gpu_node_count + 2
      min_capacity     = var.gpu_node_count
      
      instance_types = ["p5.48xlarge", "p4d.24xlarge"]
      
      k8s_labels = {
        role          = "gpu"
        workload      = "neural-inference"
        node-type     = "gpu"
        performance   = "ultra-low-latency"
      }
      
      k8s_taints = {
        gpu = {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
      
      user_data = templatefile("${path.module}/templates/gpu-node-user-data.sh", {
        cluster_name = module.eks.cluster_name
        region       = var.aws_region
      })
      
      block_device_mappings = {
        nvme = {
          device_name = "/dev/nvme1n1"
          volume_type = "io2"
          volume_size = 1000
          iops        = 100000
          throughput  = 1000
        }
      }
    }
    
    # High-performance CPU nodes for trading execution
    cpu_nodes = {
      desired_capacity = var.cpu_node_count
      max_capacity     = var.cpu_node_count + 4
      min_capacity     = var.cpu_node_count
      
      instance_types = ["c7i.4xlarge", "c6i.8xlarge"]
      
      k8s_labels = {
        role        = "cpu"
        workload    = "trading-execution"
        node-type   = "high-performance"
        performance = "real-time"
      }
      
      user_data = templatefile("${path.module}/templates/cpu-node-user-data.sh", {
        cluster_name = module.eks.cluster_name
        region       = var.aws_region
      })
      
      block_device_mappings = {
        nvme = {
          device_name = "/dev/nvme1n1"
          volume_type = "io2"
          volume_size = 500
          iops        = 50000
          throughput  = 500
        }
      }
    }
    
    # General purpose nodes for API and services
    service_nodes = {
      desired_capacity = var.service_node_count
      max_capacity     = var.service_node_count + 2
      min_capacity     = var.service_node_count
      
      instance_types = ["m6i.xlarge", "m5.xlarge"]
      
      k8s_labels = {
        role      = "service"
        workload  = "api-services"
        node-type = "general-purpose"
      }
      
      user_data = templatefile("${path.module}/templates/service-node-user-data.sh", {
        cluster_name = module.eks.cluster_name
        region       = var.aws_region
      })
    }
  }
  
  tags = {
    Environment = var.environment
    Version     = "v4.0"
  }
}

# ============================================================
# NVIDIA GPU Operator
# ============================================================

resource "helm_release" "nvidia_gpu_operator" {
  name       = "nvidia-gpu-operator"
  repository = "https://nvidia.github.io/gpu-operator"
  chart      = "gpu-operator"
  version    = "v23.9.1"
  namespace  = "gpu-operator"
  
  create_namespace = true
  
  set {
    name  = "operator.defaultRuntime"
    value = "containerd"
  }
  
  set {
    name  = "driver.enabled"
    value = "true"
  }
  
  set {
    name  = "toolkit.enabled"
    value = "true"
  }
  
  set {
    name  = "devicePlugin.enabled"
    value = "true"
  }
  
  depends_on = [module.eks]
}

# ============================================================
# Monitoring Stack
# ============================================================

resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  version    = "48.1.1"
  namespace  = "monitoring"
  
  create_namespace = true
  
  values = [
    file("${path.module}/helm/prometheus-values.yaml")
  ]
  
  depends_on = [module.eks]
}

# ============================================================
# Logging Stack
# ============================================================

resource "helm_release" "elasticsearch" {
  name       = "elasticsearch"
  repository = "https://helm.elastic.co"
  chart      = "elasticsearch"
  version    = "8.5.1"
  namespace  = "logging"
  
  create_namespace = true
  
  values = [
    file("${path.module}/helm/elasticsearch-values.yaml")
  ]
  
  depends_on = [module.eks]
}

resource "helm_release" "kibana" {
  name       = "kibana"
  repository = "https://helm.elastic.co"
  chart      = "kibana"
  version    = "8.5.1"
  namespace  = "logging"
  
  values = [
    file("${path.module}/helm/kibana-values.yaml")
  ]
  
  depends_on = [helm_release.elasticsearch]
}

# ============================================================
# Ingress Controller
# ============================================================

resource "helm_release" "ingress_nginx" {
  name       = "ingress-nginx"
  repository = "https://kubernetes.github.io/ingress-nginx"
  chart      = "ingress-nginx"
  version    = "4.8.0"
  namespace  = "ingress-nginx"
  
  create_namespace = true
  
  values = [
    file("${path.module}/helm/ingress-nginx-values.yaml")
  ]
  
  depends_on = [module.eks]
}

# ============================================================
# Cert-Manager for SSL
# ============================================================

resource "helm_release" "cert_manager" {
  name       = "cert-manager"
  repository = "https://charts.jetstack.io"
  chart      = "cert-manager"
  version    = "v1.13.2"
  namespace  = "cert-manager"
  
  create_namespace = true
  
  set {
    name  = "installCRDs"
    value = "true"
  }
  
  depends_on = [module.eks]
}

# ============================================================
# ArchNeuronX v4.0 Application
# ============================================================

resource "kubernetes_namespace" "archneuronx" {
  metadata {
    name = "archneuronx"
    
    labels = {
      name     = "archneuronx"
      version  = "v4.0"
      project  = "archneuronx"
    }
  }
}

# ConfigMap for application configuration
resource "kubernetes_config_map" "archneuronx_config" {
  metadata {
    name      = "archneuronx-v4-config"
    namespace = kubernetes_namespace.archneuronx.metadata.name
  }
  
  data = {
    "production.json" = file("${path.root}/config/v4_production.json")
  }
}

# Secrets for sensitive data
resource "kubernetes_secret" "archneuronx_secrets" {
  metadata {
    name      = "archneuronx-v4-secrets"
    namespace = kubernetes_namespace.archneuronx.metadata.name
  }
  
  type = "Opaque"
  
  data = {
    "api-key"        = var.api_key
    "database-url"   = var.database_url
    "redis-password" = var.redis_password
  }
}

# ============================================================
# Storage Classes
# ============================================================

resource "kubernetes_storage_class" "nvme_ssd" {
  metadata {
    name = "nvme-ssd"
  }
  
  storage_provisioner = "ebs.csi.aws.com"
  
  parameters = {
    type      = "io2"
    iops      = "100000"
    throughput = "1000"
    fsType    = "ext4"
  }
  
  allow_volume_expansion = true
  
  volume_binding_mode = "WaitForFirstConsumer"
}

resource "kubernetes_storage_class" "gpu_memory" {
  metadata {
    name = "gpu-memory"
  }
  
  storage_provisioner = "ebs.csi.aws.com"
  
  parameters = {
    type      = "io2"
    iops      = "200000"
    throughput = "2000"
    fsType    = "ext4"
  }
  
  allow_volume_expansion = true
  
  volume_binding_mode = "WaitForFirstConsumer"
}

# ============================================================
# Network Policies
# ============================================================

resource "kubernetes_network_policy" "archneuronx_network_policy" {
  metadata {
    name      = "archneuronx-network-policy"
    namespace = kubernetes_namespace.archneuronx.metadata.name
  }
  
  policy_types = ["Ingress", "Egress"]
  
  ingress {
    from {
      namespace_selector {
        match_labels = {
          name = "ingress-nginx"
        }
      }
    }
    
    ports {
      protocol = "TCP"
      port     = 8080
    }
  }
  
  egress {
    to {
      namespace_selector {}
    }
    
    ports {
      protocol = "TCP"
      port     = 443
    }
    
    ports {
      protocol = "TCP"
      port     = 53
    }
    
    ports {
      protocol = "UDP"
      port     = 53
    }
  }
}

# ============================================================
# Pod Disruption Budgets
# ============================================================

resource "kubernetes_pod_disruption_budget" "archneuronx_pdb" {
  metadata {
    name      = "archneuronx-pdb"
    namespace = kubernetes_namespace.archneuronx.metadata.name
  }
  
  spec {
    min_available = 2
    
    selector {
      match_labels = {
        app = "archneuronx-v4"
      }
    }
  }
}

# ============================================================
# Horizontal Pod Autoscaler
# ============================================================

resource "kubernetes_horizontal_pod_autoscaler" "archneuronx_hpa" {
  metadata {
    name      = "archneuronx-v4-hpa"
    namespace = kubernetes_namespace.archneuronx.metadata.name
  }
  
  spec {
    max_replicas = 10
    min_replicas = 2
    
    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = 70
        }
      }
    }
    
    metric {
      type = "Resource"
      resource {
        name = "memory"
        target {
          type                = "Utilization"
          average_utilization = 80
        }
      }
    }
    
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = "archneuronx-v4"
    }
  }
}

# ============================================================
# Service Monitor for Prometheus
# ============================================================

resource "kubernetes_manifest" "archneuronx_servicemonitor" {
  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name      = "archneuronx-v4"
      namespace = kubernetes_namespace.archneuronx.metadata.name
      labels = {
        release = "prometheus"
      }
    }
    spec = {
      selector = {
        match_labels = {
          app = "archneuronx-v4"
        }
      }
      endpoints = [
        {
          port     = "metrics"
          path     = "/metrics"
          interval = "15s"
        }
      ]
    }
  }
}

# ============================================================
# Outputs
# ============================================================

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group id attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "region" {
  description = "AWS region"
  value       = var.aws_region
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "node_groups" {
  description = "EKS node groups"
  value       = module.eks.eks_managed_node_groups
}
