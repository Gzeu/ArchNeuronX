# ============================================================
# ArchNeuronX v4.0 - AWS Infrastructure as Code
# Multi-region, multi-availability zone deployment for <20μs latency
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
  }
  
  backend "s3" {
    bucket = "archneuronx-v4-terraform-state"
    key    = "infrastructure/aws/terraform.tfstate"
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

provider "aws" {
  alias  = "us_west_2"
  region = "us-west-2"
  
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

variable "aws_region" {
  description = "Primary AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "archneuronx-v4"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnets" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnets" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "database_subnets" {
  description = "Database subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.201.0/24", "10.0.202.0/24", "10.0.203.0/24"]
}

# ============================================================
# Networking
# ============================================================

# VPC
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "${var.cluster_name}-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  
  tags = {
    Name = "${var.cluster_name}-igw"
  }
}

# NAT Gateways
resource "aws_eip" "nat" {
  count  = length(var.public_subnets)
  domain = "vpc"
  
  tags = {
    Name = "${var.cluster_name}-nat-eip-${count.index}"
  }
  
  depends_on = [aws_internet_gateway.main]
}

resource "aws_nat_gateway" "main" {
  count         = length(var.public_subnets)
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id
  
  tags = {
    Name = "${var.cluster_name}-nat-${count.index}"
  }
  
  depends_on = [aws_internet_gateway.main]
}

# Subnets
resource "aws_subnet" "private" {
  count             = length(var.private_subnets)
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnets[count.index]
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  enable_resource_name_dns_a_record_on_launch    = true
  enable_resource_name_dns_aaaa_record_on_launch = false
  
  tags = {
    Name                                              = "${var.cluster_name}-private-${count.index}"
    "kubernetes.io/cluster/${var.cluster_name}"       = "shared"
    "kubernetes.io/role/internal-elb"                 = "1"
    "karpenter.sh/capacity-type"                      = "on-demand"
  }
}

resource "aws_subnet" "public" {
  count                   = length(var.public_subnets)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnets[count.index]
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  
  tags = {
    Name                                              = "${var.cluster_name}-public-${count.index}"
    "kubernetes.io/cluster/${var.cluster_name}"       = "shared"
    "kubernetes.io/role/elb"                          = "1"
  }
}

resource "aws_subnet" "database" {
  count             = length(var.database_subnets)
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.database_subnets[count.index]
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {
    Name                                              = "${var.cluster_name}-database-${count.index}"
    "kubernetes.io/cluster/${var.cluster_name}"       = "shared"
  }
}

# Route Tables
resource "aws_route_table" "private" {
  count  = length(var.private_subnets)
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }
  
  tags = {
    Name = "${var.cluster_name}-private-rt-${count.index}"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  
  tags = {
    Name = "${var.cluster_name}-public-rt"
  }
}

resource "aws_route_table" "database" {
  vpc_id = aws_vpc.main.id
  
  tags = {
    Name = "${var.cluster_name}-database-rt"
  }
}

# Route Table Associations
resource "aws_route_table_association" "private" {
  count          = length(var.private_subnets)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

resource "aws_route_table_association" "public" {
  count          = length(var.public_subnets)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "database" {
  count          = length(var.database_subnets)
  subnet_id      = aws_subnet.database[count.index].id
  route_table_id = aws_route_table.database.id
}

# VPC Endpoints for S3 and DynamoDB
resource "aws_vpc_endpoint" "s3" {
  vpc_id       = aws_vpc.main.id
  service_name = "com.amazonaws.${var.aws_region}.s3"
  
  route_table_ids = concat(
    aws_route_table.private[*].id,
    aws_route_table.database[*].id
  )
  
  tags = {
    Name = "${var.cluster_name}-s3-endpoint"
  }
}

resource "aws_vpc_endpoint" "dynamodb" {
  vpc_id       = aws_vpc.main.id
  service_name = "com.amazonaws.${var.aws_region}.dynamodb"
  
  route_table_ids = concat(
    aws_route_table.private[*].id,
    aws_route_table.database[*].id
  )
  
  tags = {
    Name = "${var.cluster_name}-dynamodb-endpoint"
  }
}

# ============================================================
# EKS Cluster
# ============================================================

# EKS Cluster
resource "aws_eks_cluster" "main" {
  name     = var.cluster_name
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.28"
  
  vpc_config {
    subnet_ids = concat(
      aws_subnet.private[*].id,
      aws_subnet.public[*].id
    )
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = ["0.0.0.0/0"]
  }
  
  kubernetes_network_config {
    service_ipv4_cidr = "10.100.0.0/16"
  }
  
  enabled_cluster_log_types = [
    "api",
    "audit",
    "authenticator",
    "controllerManager",
    "scheduler"
  ]
  
  encryption_config {
    resources = [
      "secrets",
      "configmaps"
    ]
    provider {
      key_arn = aws_kms_key.eks.arn
    }
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_cloudwatch_log_group.eks
  ]
  
  tags = {
    Name = var.cluster_name
  }
}

# EKS Node Groups
resource "aws_eks_node_group" "compute_optimized" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "compute-optimized"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = aws_subnet.private[*].id
  
  instance_types = ["c5n.18xlarge", "c5n.9xlarge"]
  
  scaling_config {
    desired_size = 20
    max_size     = 50
    min_size     = 10
  }
  
  update_config {
    max_unavailable = 1
  }
  
  remote_access {
    ec2_ssh_key               = aws_key_pair.deployer.key_name
    source_security_group_ids = [aws_security_group.node_ssh.id]
  }
  
  labels = {
    role = "compute-optimized"
    archneuronx/node-type = "high-performance"
  }
  
  taint {
    key    = "archneuronx/node-type"
    value  = "high-performance"
    effect = "NO_SCHEDULE"
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
    aws_iam_role_policy_attachment.eks_auto_scaling_policy
  ]
  
  tags = {
    Name = "${var.cluster_name}-compute-optimized"
  }
}

resource "aws_eks_node_group" "gpu_optimized" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "gpu-optimized"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = aws_subnet.private[*].id
  
  instance_types = ["p3.8xlarge", "p4d.24xlarge"]
  
  scaling_config {
    desired_size = 10
    max_size     = 20
    min_size     = 5
  }
  
  update_config {
    max_unavailable = 1
  }
  
  remote_access {
    ec2_ssh_key               = aws_key_pair.deployer.key_name
    source_security_group_ids = [aws_security_group.node_ssh.id]
  }
  
  labels = {
    role = "gpu-optimized"
    archneuronx/node-type = "ml-workload"
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
    aws_iam_role_policy_attachment.eks_auto_scaling_policy
  ]
  
  tags = {
    Name = "${var.cluster_name}-gpu-optimized"
  }
}

resource "aws_eks_node_group" "memory_optimized" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "memory-optimized"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = aws_subnet.private[*].id
  
  instance_types = ["r5.24xlarge", "r5.16xlarge"]
  
  scaling_config {
    desired_size = 15
    max_size     = 30
    min_size     = 8
  }
  
  update_config {
    max_unavailable = 1
  }
  
  remote_access {
    ec2_ssh_key               = aws_key_pair.deployer.key_name
    source_security_group_ids = [aws_security_group.node_ssh.id]
  }
  
  labels = {
    role = "memory-optimized"
    archneuronx/node-type = "data-intensive"
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
    aws_iam_role_policy_attachment.eks_auto_scaling_policy
  ]
  
  tags = {
    Name = "${var.cluster_name}-memory-optimized"
  }
}

# ============================================================
# KMS Keys
# ============================================================

resource "aws_kms_key" "eks" {
  description             = "EKS cluster encryption key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow EKS Service"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:GenerateDataKey",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })
  
  tags = {
    Name = "${var.cluster_name}-eks-key"
  }
}

resource "aws_kms_alias" "eks" {
  name          = "alias/${var.cluster_name}-eks"
  target_key_id = aws_kms_key.eks.key_id
}

# ============================================================
# IAM Roles
# ============================================================

# EKS Cluster Role
resource "aws_iam_role" "eks_cluster" {
  name = "${var.cluster_name}-eks-cluster"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
  
  tags = {
    Name = "${var.cluster_name}-eks-cluster"
  }
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster.name
}

# EKS Node Role
resource "aws_iam_role" "eks_node" {
  name = "${var.cluster_name}-eks-node"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
  
  tags = {
    Name = "${var.cluster_name}-eks-node"
  }
}

resource "aws_iam_role_policy_attachment" "eks_worker_node_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_node.name
}

resource "aws_iam_role_policy_attachment" "eks_cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_node.name
}

resource "aws_iam_role_policy_attachment" "eks_container_registry_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_node.name
}

resource "aws_iam_role_policy_attachment" "eks_auto_scaling_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AutoScalingFullAccess"
  role       = aws_iam_role.eks_node.name
}

# ============================================================
# Security Groups
# ============================================================

# Cluster Security Group
resource "aws_security_group" "cluster" {
  name        = "${var.cluster_name}-cluster"
  description = "EKS cluster security group"
  vpc_id      = aws_vpc.main.id
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.cluster_name}-cluster"
  }
}

# Node Security Group
resource "aws_security_group" "node" {
  name        = "${var.cluster_name}-node"
  description = "EKS node security group"
  vpc_id      = aws_vpc.main.id
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.cluster_name}-node"
  }
}

# Node SSH Security Group
resource "aws_security_group" "node_ssh" {
  name        = "${var.cluster_name}-node-ssh"
  description = "EKS node SSH security group"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.cluster_name}-node-ssh"
  }
}

# ============================================================
# Additional Resources
# ============================================================

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "eks" {
  name              = "/aws/eks/${var.cluster_name}/cluster"
  retention_in_days = 7
  
  tags = {
    Name = "${var.cluster_name}-logs"
  }
}

# SSH Key Pair
resource "aws_key_pair" "deployer" {
  key_name   = "${var.cluster_name}-deployer"
  public_key = file("${path.module}/ssh/deployer.pub")
}

# Data Sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# ============================================================
# Outputs
# ============================================================

output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.main.name
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_certificate_authority_data" {
  description = "EKS cluster certificate authority data"
  value       = aws_eks_cluster.main.certificate_authority[0].data
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "database_subnet_ids" {
  description = "Database subnet IDs"
  value       = aws_subnet.database[*].id
}

output "node_role_arn" {
  description = "EKS node role ARN"
  value       = aws_iam_role.eks_node.arn
}
