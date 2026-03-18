# ============================================================
# ArchNeuronX v4.0 - Terraform Variables
# ============================================================

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "archneuronx-v4"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "production"
  
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
  
  validation {
    condition     = contains(["us-east-1", "us-west-2", "eu-west-1"], var.aws_region)
    error_message = "AWS region must be one of: us-east-1, us-west-2, eu-west-1."
  }
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
  
  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid CIDR block."
  }
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
  
  validation {
    condition     = can(regex("^1\\.[2-9][0-9]$", var.kubernetes_version))
    error_message = "Kubernetes version must be in format 1.xx where xx is 20 or higher."
  }
}

# ============================================================
# Node Group Configuration
# ============================================================

variable "gpu_node_count" {
  description = "Number of GPU nodes for neural network inference"
  type        = number
  default     = 2
  
  validation {
    condition     = var.gpu_node_count >= 1 && var.gpu_node_count <= 10
    error_message = "GPU node count must be between 1 and 10."
  }
}

variable "cpu_node_count" {
  description = "Number of high-performance CPU nodes for trading execution"
  type        = number
  default     = 4
  
  validation {
    condition     = var.cpu_node_count >= 2 && var.cpu_node_count <= 20
    error_message = "CPU node count must be between 2 and 20."
  }
}

variable "service_node_count" {
  description = "Number of general purpose nodes for API services"
  type        = number
  default     = 3
  
  validation {
    condition     = var.service_node_count >= 1 && var.service_node_count <= 10
    error_message = "Service node count must be between 1 and 10."
  }
}

# ============================================================
# Application Configuration
# ============================================================

variable "api_key" {
  description = "API key for external services"
  type        = string
  sensitive   = true
}

variable "database_url" {
  description = "Database connection URL"
  type        = string
  sensitive   = true
}

variable "redis_password" {
  description = "Redis password"
  type        = string
  sensitive   = true
}

# ============================================================
# Performance Configuration
# ============================================================

variable "enable_high_performance_networking" {
  description = "Enable enhanced networking for trading applications"
  type        = bool
  default     = true
}

variable "enable_gpu_optimization" {
  description = "Enable GPU optimization for neural network workloads"
  type        = bool
  default     = true
}

variable "enable_monitoring" {
  description = "Enable comprehensive monitoring and observability"
  type        = bool
  default     = true
}

# ============================================================
# Security Configuration
# ============================================================

variable "enable_private_endpoint" {
  description = "Enable private endpoints for enhanced security"
  type        = bool
  default     = true
}

variable "enable_encryption" {
  description = "Enable encryption for all storage and communications"
  type        = bool
  default     = true
}

variable "enable_audit_logging" {
  description = "Enable comprehensive audit logging"
  type        = bool
  default     = true
}

# ============================================================
# Cost Optimization
# ============================================================

variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "enable_autoscaling" {
  description = "Enable automatic scaling based on load"
  type        = bool
  default     = true
}

variable "enable_cost_monitoring" {
  description = "Enable detailed cost monitoring and alerts"
  type        = bool
  default     = true
}

# ============================================================
# Backup and Disaster Recovery
# ============================================================

variable "enable_backups" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
  
  validation {
    condition     = var.backup_retention_days >= 7 && var.backup_retention_days <= 365
    error_message = "Backup retention must be between 7 and 365 days."
  }
}

variable "enable_cross_region_replication" {
  description = "Enable cross-region replication for disaster recovery"
  type        = bool
  default     = true
}

# ============================================================
# Local Variables
# ============================================================

locals {
  availability_zones = [
    "${var.aws_region}a",
    "${var.aws_region}b",
    "${var.aws_region}c"
  ]
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    Version     = "v4.0"
    ManagedBy   = "Terraform"
  }
  
  gpu_instance_types = [
    "p5.48xlarge",
    "p4d.24xlarge",
    "p3.16xlarge"
  ]
  
  cpu_instance_types = [
    "c7i.4xlarge",
    "c6i.8xlarge",
    "c5.9xlarge"
  ]
  
  service_instance_types = [
    "m6i.xlarge",
    "m5.xlarge",
    "m6a.xlarge"
  ]
}
