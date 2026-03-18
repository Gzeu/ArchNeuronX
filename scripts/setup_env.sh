#!/bin/bash

# ArchNeuronX v4.0 Environment Setup Script
# Secure API Key Management System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Create .env file
create_env_file() {
    print_header "CREATING SECURE ENVIRONMENT FILE"
    
    if [ -f ".env" ]; then
        print_warning ".env file already exists. Creating backup..."
        cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
    fi
    
    cat > .env << 'EOF'
# ArchNeuronX v4.0 Environment Variables
# SECURE API KEYS - DO NOT COMMIT TO GIT!
# This file contains sensitive information and should never be shared.

# ============================================================================
# EXCHANGE API KEYS
# ============================================================================
# Get your API keys from: https://www.binance.com/en/my/settings/api-management
# IMPORTANT: Enable "Enable Spot & Margin Trading" and "Enable Futures"
# IMPORTANT: Restrict IP access to your server IP for security

# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

# ============================================================================
# LLM API KEYS
# ============================================================================
# Get your HuggingFace token from: https://huggingface.co/settings/tokens
# Get your Mistral API key from: https://console.mistral.ai/

# HuggingFace Configuration
HUGGINGFACE_TOKEN=your_huggingface_token_here
HUGGINGFACE_MODEL=mistralai/Mistral-7B-v0.1

# Mistral AI Configuration
MISTRAL_API_KEY=your_mistral_api_key_here
MISTRAL_MODEL=mistral-large-latest

# ============================================================================
# ALERT SYSTEM CONFIGURATION
# ============================================================================
# Email alerts (Gmail example - use App Passwords!)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_gmail_app_password_here

# Twilio SMS alerts
TWILIO_ACCOUNT_SID=your_twilio_account_sid_here
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here

# Webhook alerts
WEBHOOK_URL=https://your-webhook-url.com/alerts
WEBHOOK_TOKEN=your_webhook_token_here

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================
# For production database (PostgreSQL example)
DATABASE_URL=postgresql://username:password@localhost:5432/archneuronx

# Redis for caching
REDIS_URL=redis://localhost:6379

# ============================================================================
# SECURITY CONFIGURATION
# ============================================================================
# JWT secret for API authentication
JWT_SECRET=your_jwt_secret_here_minimum_32_characters

# Encryption key for sensitive data (32 characters minimum)
ENCRYPTION_KEY=your_32_character_encryption_key_here

# ============================================================================
# MONITORING CONFIGURATION
# ============================================================================
# Prometheus metrics
PROMETHEUS_PORT=9090

# Grafana dashboard
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your_grafana_password_here

# ============================================================================
# DEVELOPMENT CONFIGURATION
# ============================================================================
# Environment mode: development, staging, production
NODE_ENV=development

# Debug mode: true/false
DEBUG_MODE=false

# Verbose logging: true/false
VERBOSE_LOGGING=false

# Mock data for testing: true/false
MOCK_DATA=true

# Paper trading mode: true/false
PAPER_TRADING=true

# ============================================================================
# DOCKER CONFIGURATION
# ============================================================================
# Docker registry
DOCKER_REGISTRY=your-registry.com

# Docker image tag
DOCKER_TAG=latest

# ============================================================================
# KUBERNETES CONFIGURATION
# ============================================================================
# Kubernetes namespace
K8S_NAMESPACE=archneuronx

# Kubernetes cluster
K8S_CLUSTER=your-cluster-name

# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================
# Maximum concurrent connections
MAX_CONNECTIONS=1000

# Request timeout in seconds
REQUEST_TIMEOUT=30

# Cache TTL in seconds
CACHE_TTL=300

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO

# Log file path
LOG_FILE_PATH=logs/

# Maximum log file size in MB
MAX_LOG_FILE_SIZE=10

# ============================================================================
# TRADING CONFIGURATION
# ============================================================================
# Risk per trade in USD
RISK_PER_TRADE=100

# Maximum position size in USD
MAX_POSITION_SIZE=1000

# Maximum daily loss in USD
MAX_DAILY_LOSS=1000

# Maximum drawdown percentage
MAX_DRAWDOWN=0.1

# Stop loss percentage
STOP_LOSS_PCT=0.02

# Take profit percentage
TAKE_PROFIT_PCT=0.05

# ============================================================================
# WEB INTERFACE CONFIGURATION
# ============================================================================
# HTTP port
HTTP_PORT=8080

# WebSocket port
WEBSOCKET_PORT=3001

# API rate limit per minute
API_RATE_LIMIT=1000

# ============================================================================
# BACKUP CONFIGURATION
# ============================================================================
# Backup storage path
BACKUP_PATH=/backup/archneuronx/

# Backup retention days
BACKUP_RETENTION_DAYS=30

# ============================================================================
# NOTIFICATION CONFIGURATION
# ============================================================================
# Slack webhook for notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK

# Discord webhook for notifications
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR/DISCORD/WEBHOOK

# Telegram bot token
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

EOF

    print_status "✅ .env file created successfully"
    print_warning "🔒 Please edit .env file with your actual API keys"
    print_warning "🚨 NEVER commit .env file to Git!"
}

# Create .gitignore entry
setup_gitignore() {
    print_header "SETTING UP GITIGNORE"
    
    if [ -f ".gitignore" ]; then
        if ! grep -q ".env" .gitignore; then
            echo "" >> .gitignore
            echo "# Environment variables - SECURE!" >> .gitignore
            echo ".env" >> .gitignore
            echo ".env.*" >> .gitignore
            echo "config/live_trading.yaml" >> .gitignore
            print_status "✅ Added .env to .gitignore"
        else
            print_status "✅ .env already in .gitignore"
        fi
    else
        cat > .gitignore << 'EOF'
# Environment variables - SECURE!
.env
.env.*
config/live_trading.yaml

# Build directories
build/
dist/

# Logs
logs/
*.log

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Temporary files
*.tmp
*.temp
*.bak
*.backup

# Security files
*.key
*.pem
*.crt
*.p12
*.pfx

# Database
*.db
*.sqlite
*.sqlite3

# Cache
.cache/
cache/

# Docker
.dockerignore

# Kubernetes
secrets/
EOF
        print_status "✅ Created .gitignore file"
    fi
}

# Create configuration loader
create_config_loader() {
    print_header "CREATING CONFIGURATION LOADER"
    
    cat > scripts/load_config.py << 'EOF'
#!/usr/bin/env python3
"""
ArchNeuronX v4.0 Configuration Loader
Secure environment variable substitution for YAML files
"""

import os
import re
import sys
import yaml
from pathlib import Path

def substitute_env_vars(text):
    """Replace ${VAR} patterns with environment variables"""
    pattern = re.compile(r'\$\{([^}]+)\}')
    
    def replace_match(match):
        var_name = match.group(1)
        value = os.getenv(var_name)
        if value is None:
            print(f"⚠️  Environment variable {var_name} not found!")
            return f"${{{var_name}}}"  # Return original if not found
        return value
    
    return pattern.sub(replace_match, text)

def load_config(template_path, output_path):
    """Load template YAML and substitute environment variables"""
    try:
        # Read template file
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Substitute environment variables
        processed_content = substitute_env_vars(template_content)
        
        # Parse YAML to validate
        config = yaml.safe_load(processed_content)
        
        # Write processed configuration
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Configuration loaded: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

def main():
    """Main function"""
    if len(sys.argv) != 3:
        print("Usage: python load_config.py <template_file> <output_file>")
        sys.exit(1)
    
    template_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Check if template file exists
    if not os.path.exists(template_file):
        print(f"❌ Template file not found: {template_file}")
        sys.exit(1)
    
    # Load environment variables from .env file
    env_file = Path('.env')
    if env_file.exists():
        print("📁 Loading environment variables from .env file...")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Environment variables loaded")
    else:
        print("⚠️  .env file not found. Using system environment variables.")
    
    # Load configuration
    success = load_config(template_file, output_file)
    
    if success:
        print("🎉 Configuration loaded successfully!")
        
        # Verify critical variables
        critical_vars = ['BINANCE_API_KEY', 'BINANCE_API_SECRET']
        missing_vars = []
        
        for var in critical_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"⚠️  Missing critical environment variables: {', '.join(missing_vars)}")
            print("📝 Please update your .env file with these variables.")
        else:
            print("✅ All critical environment variables are set!")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
EOF

    chmod +x scripts/load_config.py
    print_status "✅ Configuration loader created"
}

# Create secure startup script
create_secure_startup() {
    print_header "CREATING SECURE STARTUP SCRIPT"
    
    cat > scripts/start_secure.sh << 'EOF'
#!/bin/bash

# ArchNeuronX v4.0 Secure Startup Script
# Loads environment variables and starts live trading

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Check if .env file exists
check_env_file() {
    if [ ! -f ".env" ]; then
        print_error "❌ .env file not found!"
        print_status "📝 Please run: ./scripts/setup_env.sh"
        exit 1
    fi
    
    print_status "✅ .env file found"
}

# Load environment variables
load_env_vars() {
    print_status "📁 Loading environment variables..."
    
    # Load from .env file
    set -a
    source .env
    set +a
    
    print_status "✅ Environment variables loaded"
}

# Verify critical variables
verify_critical_vars() {
    print_status "🔍 Verifying critical environment variables..."
    
    critical_vars=("BINANCE_API_KEY" "BINANCE_API_SECRET")
    missing_vars=()
    
    for var in "${critical_vars[@]}"; do
        if [ -z "${!var}" ] || [ "${!var}" = "your_${var,,}_here" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        print_error "❌ Missing or unset critical variables:"
        for var in "${missing_vars[@]}"; do
            echo "   - $var"
        done
        print_status "📝 Please update your .env file with these variables."
        exit 1
    fi
    
    print_status "✅ All critical environment variables are set!"
}

# Generate configuration
generate_config() {
    print_status "🔧 Generating configuration from template..."
    
    # Check if template exists
    if [ ! -f "config/live_trading_template.yaml" ]; then
        print_error "❌ Configuration template not found!"
        exit 1
    fi
    
    # Use Python script to substitute variables
    python3 scripts/load_config.py config/live_trading_template.yaml config/live_trading.yaml
    
    if [ $? -eq 0 ]; then
        print_status "✅ Configuration generated successfully"
    else
        print_error "❌ Failed to generate configuration"
        exit 1
    fi
}

# Start live trading
start_trading() {
    print_header "🚀 STARTING ARCHNEURONX LIVE TRADING"
    
    # Check if executable exists
    if [ ! -f "build/archneuronx_live_trading" ]; then
        print_error "❌ Live trading executable not found!"
        print_status "🏗️  Please build first: ./scripts/run_live_trading.sh build"
        exit 1
    fi
    
    # Set environment variables
    export ARCHNEURONX_CONFIG_FILE="config/live_trading.yaml"
    export ARCHNEURONX_ENV_LOADED="true"
    
    print_status "🎯 Starting live trading system..."
    print_warning "🔒 Using secure configuration with API keys from environment"
    
    # Start live trading
    ./build/archneuronx_live_trading
}

# Main execution
main() {
    print_header "🔒 ARCHNEURONX SECURE STARTUP"
    
    check_env_file
    load_env_vars
    verify_critical_vars
    generate_config
    start_trading
}

# Run main function
main "$@"
EOF

    chmod +x scripts/start_secure.sh
    print_status "✅ Secure startup script created"
}

# Main function
main() {
    print_header "🔒 ARCHNEURONX V4.0 ENVIRONMENT SETUP"
    
    print_status "🚀 Setting up secure environment for API keys..."
    
    create_env_file
    setup_gitignore
    create_config_loader
    create_secure_startup
    
    print_header "✅ SETUP COMPLETED"
    print_status "🎉 Secure environment setup completed!"
    echo ""
    print_status "📝 NEXT STEPS:"
    echo "   1. Edit .env file with your actual API keys"
    echo "   2. Run: ./scripts/start_secure.sh"
    echo "   3. Or use: python3 scripts/load_config.py config/live_trading_template.yaml config/live_trading.yaml"
    echo ""
    print_warning "🔒 IMPORTANT: Never commit .env file to Git!"
    print_warning "🔒 Your API keys are now secure and isolated from version control!"
}

# Run main function
main "$@"
