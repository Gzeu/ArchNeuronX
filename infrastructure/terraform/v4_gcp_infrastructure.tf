# ============================================================
# ArchNeuronX v4.0 - GCP Infrastructure as Code
# Secondary region deployment for disaster recovery and load balancing
# ============================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.80"
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
  }
  
  backend "gcs" {
    bucket = "archneuronx-v4-terraform-state"
    prefix = "infrastructure/gcp/terraform"
  }
}

# ============================================================
# Provider Configuration
# ============================================================

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
  
  default_tags {
    tags = {
      Project     = "ArchNeuronX"
      Environment = var.environment
      Version     = "v4.0"
      ManagedBy   = "Terraform"
    }
  }
}

# ============================================================
# Variables
# ============================================================

variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
  default     = "archneuronx-v4"
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "archneuronx-v4-gcp"
}

variable "network_name" {
  description = "VPC network name"
  type        = string
  default     = "archneuronx-v4-network"
}

variable "subnet_cidrs" {
  description = "Subnet CIDR blocks"
  type        = list(string)
  default     = ["10.1.0.0/24", "10.1.1.0/24", "10.1.2.0/24"]
}

# ============================================================
# VPC Network
# ============================================================

# VPC Network
resource "google_compute_network" "main" {
  name                    = var.network_name
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"
  
  description = "ArchNeuronX v4.0 primary VPC network"
}

# Subnets
resource "google_compute_subnetwork" "private" {
  count         = length(var.subnet_cidrs)
  name          = "${var.cluster_name}-private-${count.index}"
  ip_cidr_range = var.subnet_cidrs[count.index]
  region        = var.gcp_region
  network       = google_compute_network.main.id
  
  private_ip_google_access = true
  
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.100.${count.index}.0/18"
  }
  
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.200.${count.index}.0/24"
  }
  
  description = "Private subnet ${count.index} for ArchNeuronX v4.0"
}

# Firewall Rules
resource "google_compute_firewall" "allow_internal" {
  name    = "${var.cluster_name}-allow-internal"
  network = google_compute_network.main.name
  
  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }
  
  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }
  
  allow {
    protocol = "icmp"
  }
  
  source_tags = ["archneuronx-internal"]
  target_tags = ["archneuronx-internal"]
  
  description = "Allow internal traffic"
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "${var.cluster_name}-allow-ssh"
  network = google_compute_network.main.name
  
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  
  source_ranges = ["35.235.240.0/20"] # IAP-to-VPC connectors
  target_tags   = ["archneuronx-ssh"]
  
  description = "Allow SSH from IAP"
}

resource "google_compute_firewall" "allow_kubernetes" {
  name    = "${var.cluster_name}-allow-kubernetes"
  network = google_compute_network.main.name
  
  allow {
    protocol = "tcp"
    ports    = ["443", "8443"]
  }
  
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["archneuronx-gke-master"]
  
  description = "Allow Kubernetes API access"
}

# ============================================================
# GKE Cluster
# ============================================================

# GKE Cluster
resource "google_container_cluster" "main" {
  name     = var.cluster_name
  location = var.gcp_region
  
  initial_node_count = 1
  
  remove_default_node_pool = true
  
  network    = google_compute_network.main.id
  subnetwork = google_compute_subnetwork.private[0].id
  
  ip_allocation_policy {
    use_ip_aliases = true
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }
  
  master_authorized_networks_config {
    cidr_blocks {
      cidr_block   = "0.0.0.0/0"
      display_name = "All"
    }
  }
  
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }
  
  database_encryption {
    state    = "ENCRYPTED"
    key_name = google_kms_crypto_key.gke.id
  }
  
  resource_usage_export_config {
    enable_network_egress_export = true
  }
  
  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
    network_policy_config {
      disabled = false
    }
    gce_persistent_disk_csi_driver {
      disabled = false
    }
    gcp_filestore_csi_driver {
      disabled = false
    }
    gke_backup_agent {
      disabled = false
    }
  }
  
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }
  
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS", "APISERVER", "CONTROLLER_MANAGER", "SCHEDULER"]
  }
  
  workload_identity_config {
    workload_pool = "${var.gcp_project_id}.svc.id.goog"
  }
  
  node_config {
    machine_type = "e2-standard-4"
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    labels = {
      environment = var.environment
      cluster     = var.cluster_name
      role        = "default"
    }
    
    tags = ["archneuronx-gke-node"]
  }
  
  depends_on = [
    google_kms_crypto_key_iam_binding.gke
  ]
}

# ============================================================
# GKE Node Pools
# ============================================================

# Compute Optimized Node Pool
resource "google_container_node_pool" "compute_optimized" {
  name       = "compute-optimized"
  location   = var.gcp_region
  cluster    = google_container_cluster.main.name
  node_count = 3
  
  node_config {
    machine_type = "c2-standard-8"
    disk_size_gb = 200
    disk_type    = "pd-ssd"
    
    preemptible  = false
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    labels = {
      environment          = var.environment
      cluster              = var.cluster_name
      role                 = "compute-optimized"
      "archneuronx/node-type" = "high-performance"
    }
    
    taints {
      key    = "archneuronx/node-type"
      value  = "high-performance"
      effect = "NO_SCHEDULE"
    }
    
    tags = ["archneuronx-compute-node"]
    
    metadata = {
      disable-legacy-endpoint = "true"
    }
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

# GPU Optimized Node Pool
resource "google_container_node_pool" "gpu_optimized" {
  name       = "gpu-optimized"
  location   = var.gcp_region
  cluster    = google_container_cluster.main.name
  node_count = 2
  
  node_config {
    machine_type = "n1-standard-8"
    disk_size_gb = 500
    disk_type    = "pd-ssd"
    
    preemptible  = false
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    labels = {
      environment          = var.environment
      cluster              = var.cluster_name
      role                 = "gpu-optimized"
      "archneuronx/node-type" = "ml-workload"
    }
    
    tags = ["archneuronx-gpu-node"]
    
    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
    }
    
    metadata = {
      disable-legacy-endpoint = "true"
    }
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

# Memory Optimized Node Pool
resource "google_container_node_pool" "memory_optimized" {
  name       = "memory-optimized"
  location   = var.gcp_region
  cluster    = google_container_cluster.main.name
  node_count = 3
  
  node_config {
    machine_type = "n1-highmem-32"
    disk_size_gb = 1000
    disk_type    = "pd-ssd"
    
    preemptible  = false
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    labels = {
      environment          = var.environment
      cluster              = var.cluster_name
      role                 = "memory-optimized"
      "archneuronx/node-type" = "data-intensive"
    }
    
    tags = ["archneuronx-memory-node"]
    
    metadata = {
      disable-legacy-endpoint = "true"
    }
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

# ============================================================
# KMS Encryption
# ============================================================

# KMS Key Ring
resource "google_kms_key_ring" "main" {
  name     = "${var.cluster_name}-keyring"
  location = var.gcp_region
}

# KMS Crypto Key for GKE
resource "google_kms_crypto_key" "gke" {
  name     = "${var.cluster_name}-gke-key"
  key_ring = google_kms_key_ring.main.id
  
  purpose = "ENCRYPT_DECRYPT"
  
  version_template {
    algorithm        = "GOOGLE_SYMMETRIC_ENCRYPTION"
    protection_level = "SOFTWARE"
  }
  
  rotation_period = "7776000s" # 90 days
}

# KMS Key IAM Binding
resource "google_kms_crypto_key_iam_binding" "gke" {
  crypto_key_id = google_kms_crypto_key.gke.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  
  members = [
    "serviceAccount:service-${data.google_project.current.number}@container-engine.googleapis.com"
  ]
}

# ============================================================
# Service Accounts
# ============================================================

# GKE Service Account
resource "google_service_account" "gke" {
  account_id   = "${var.cluster_name}-gke"
  display_name = "ArchNeuronX v4.0 GKE Service Account"
}

resource "google_project_iam_member" "gke" {
  project = var.gcp_project_id
  role    = "roles/container.admin"
  member  = "serviceAccount:${google_service_account.gke.email}"
}

# ============================================================
# Storage
# ============================================================

# GCS Bucket for Terraform State
resource "google_storage_bucket" "terraform_state" {
  name          = "archneuronx-v4-terraform-state"
  location      = var.gcp_region
  storage_class = "STANDARD"
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
  
  labels = {
    environment = var.environment
    project     = "ArchNeuronX"
    version     = "v4.0"
  }
}

# GCS Bucket for Application Data
resource "google_storage_bucket" "app_data" {
  name          = "archneuronx-v4-app-data"
  location      = var.gcp_region
  storage_class = "STANDARD"
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Delete"
    }
  }
  
  labels = {
    environment = var.environment
    project     = "ArchNeuronX"
    version     = "v4.0"
  }
}

# ============================================================
# Cloud SQL (PostgreSQL)
# ============================================================

# Cloud SQL Instance
resource "google_sql_database_instance" "main" {
  name             = "${var.cluster_name}-postgres"
  database_version = "POSTGRES_14"
  region           = var.gcp_region
  
  settings {
    tier = "db-n1-standard-8"
    
    disk_size = 1000
    disk_type = "PD_SSD"
    
    ip_configuration {
      ipv4_enabled = false
      private_network = google_compute_network.main.id
      require_ssl = true
    }
    
    backup_configuration {
      enabled            = true
      binary_log_enabled = true
      location          = var.gcp_region
      start_time        = "02:00"
    }
    
    maintenance_window {
      day  = 7  # Sunday
      hour = 3  # 3 AM
    }
    
    database_flags {
      name  = "max_connections"
      value = "500"
    }
    
    database_flags {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements"
    }
    
    database_flags {
      name  = "log_min_duration_statement"
      value = "1000"
    }
  }
  
  deletion_protection = false
  
  depends_on = [
    google_service_networking_connection.private_vpc_connection
  ]
}

# Cloud SQL Database
resource "google_sql_database" "main" {
  name     = "archneuronx_v4"
  instance = google_sql_database_instance.main.name
}

# Cloud SQL User
resource "google_sql_user" "app_user" {
  name     = "archneuronx_app"
  instance = google_sql_database_instance.main.name
  password = random_password.db_password.result
}

# ============================================================
# Memorystore (Redis)
# ============================================================

# Redis Instance
resource "google_redis_instance" "main" {
  name           = "${var.cluster_name}-redis"
  tier           = "STANDARD_HA"
  memory_size_gb = 32
  
  location_id             = "${var.gcp_region}-a"
  alternative_location_id  = "${var.gcp_region}-b"
  
  authorized_network = google_compute_subnetwork.private[0].id
  
  redis_version     = "REDIS_7_0"
  display_name      = "ArchNeuronX v4.0 Redis Cluster"
  
  labels = {
    environment = var.environment
    project     = "ArchNeuronX"
    version     = "v4.0"
  }
}

# ============================================================
# VPC Peering for Cloud SQL
# ============================================================

# Service Networking Connection
resource "google_service_networking_connection" "private_vpc_connection" {
  service = "servicenetworking.googleapis.com"
  network = google_compute_network.main.id
  
  reserved_peering_ranges = [
    google_compute_global_address.private_ip_range.name
  ]
  
  depends_on = [google_project_service.enable_service_networking]
}

# Global Address for Private IP Range
resource "google_compute_global_address" "private_ip_range" {
  name          = "${var.cluster_name}-private-ip-range"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.main.id
}

# ============================================================
# Monitoring and Logging
# ============================================================

# Monitoring Workspace
resource "google_monitoring_workspace" "main" {
  display_name = "ArchNeuronX v4.0 Monitoring"
}

# Logging Bucket
resource "google_logging_bucket" "main" {
  name        = "${var.cluster_name}-logging-bucket"
  location    = var.gcp_region
  
  retention_days = 30
  
  lifecycle {
    ignore_changes = [retention_days]
  }
}

# Logging Sink
resource "google_logging_sink" "main" {
  name          = "${var.cluster_name}-logging-sink"
  destination   = "logging.googleapis.com/projects/${var.gcp_project_id}/locations/${var.gcp_region}/buckets/${google_logging_bucket.main.name}"
  
  filter = "resource.type=\"k8s_cluster\""
  
  depends_on = [google_project_service.enable_logging_api]
}

# ============================================================
# IAM and Service Accounts
# ============================================================

# Enable Required APIs
resource "google_project_service" "enable_container_api" {
  service = "container.googleapis.com"
}

resource "google_project_service" "enable_compute_api" {
  service = "compute.googleapis.com"
}

resource "google_project_service" "enable_sqladmin_api" {
  service = "sqladmin.googleapis.com"
}

resource "google_project_service" "enable_redis_api" {
  service = "redis.googleapis.com"
}

resource "google_project_service" "enable_service_networking_api" {
  service = "servicenetworking.googleapis.com"
}

resource "google_project_service" "enable_logging_api" {
  service = "logging.googleapis.com"
}

resource "google_project_service" "enable_monitoring_api" {
  service = "monitoring.googleapis.com"
}

resource "google_project_service" "enable_kms_api" {
  service = "cloudkms.googleapis.com"
}

# ============================================================
# Random Resources
# ============================================================

resource "random_password" "db_password" {
  length  = 32
  special = true
}

# ============================================================
# Data Sources
# ============================================================

data "google_project" "current" {}

# ============================================================
# Outputs
# ============================================================

output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.main.name
}

output "cluster_location" {
  description = "GKE cluster location"
  value       = google_container_cluster.main.location
}

output "network_name" {
  description = "VPC network name"
  value       = google_compute_network.main.name
}

output "subnet_names" {
  description = "Subnet names"
  value       = google_compute_subnetwork.private[*].name
}

output "redis_instance_name" {
  description = "Redis instance name"
  value       = google_redis_instance.main.name
}

output "redis_host" {
  description = "Redis host"
  value       = google_redis_instance.main.host
}

output "redis_port" {
  description = "Redis port"
  value       = google_redis_instance.main.port
}

output "sql_instance_name" {
  description = "Cloud SQL instance name"
  value       = google_sql_database_instance.main.name
}

output "sql_instance_connection_name" {
  description = "Cloud SQL instance connection name"
  value       = google_sql_database_instance.main.connection_name
}

output "sql_database_name" {
  description = "Cloud SQL database name"
  value       = google_sql_database.main.name
}

output "gke_service_account_email" {
  description = "GKE service account email"
  value       = google_service_account.gke.email
}
