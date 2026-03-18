@echo off
REM ArchNeuronX v4.0 Environment Setup Script for Windows
REM Secure API Key Management System

setlocal enabledelayedexpansion

REM Colors for output
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Print colored output
:print_status
echo %GREEN%[INFO]%NC% %~1
goto :eof

:print_warning
echo %YELLOW%[WARNING]%NC% %~1
goto :eof

:print_error
echo %RED%[ERROR]%NC% %~1
goto :eof

:print_header
echo %BLUE%========================================%NC%
echo %BLUE%%~1%NC%
echo %BLUE%========================================%NC%
goto :eof

REM Create .env file
:create_env_file
call :print_header "CREATING SECURE ENVIRONMENT FILE"

if exist ".env" (
    call :print_warning ".env file already exists. Creating backup..."
    copy .env .env.backup.%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2% >nul
)

(
echo # ArchNeuronX v4.0 Environment Variables
echo # SECURE API KEYS - DO NOT COMMIT TO GIT!
echo # This file contains sensitive information and should never be shared.
echo.
echo # ============================================================================
echo # EXCHANGE API KEYS
echo # ============================================================================
echo # Get your API keys from: https://www.binance.com/en/my/settings/api-management
echo # IMPORTANT: Enable "Enable Spot ^& Margin Trading" and "Enable Futures"
echo # IMPORTANT: Restrict IP access to your server IP for security
echo.
echo # Binance API Configuration
echo BINANCE_API_KEY=your_binance_api_key_here
echo BINANCE_API_SECRET=your_binance_api_secret_here
echo.
echo # ============================================================================
echo # LLM API KEYS
echo # ============================================================================
echo # Get your HuggingFace token from: https://huggingface.co/settings/tokens
echo # Get your Mistral API key from: https://console.mistral.ai/
echo.
echo # HuggingFace Configuration
echo HUGGINGFACE_TOKEN=your_huggingface_token_here
echo HUGGINGFACE_MODEL=mistralai/Mistral-7B-v0.1
echo.
echo # Mistral AI Configuration
echo MISTRAL_API_KEY=your_mistral_api_key_here
echo MISTRAL_MODEL=mistral-large-latest
echo.
echo # ============================================================================
echo # ALERT SYSTEM CONFIGURATION
echo # ============================================================================
echo # Email alerts ^(Gmail example - use App Passwords!^)
echo SMTP_SERVER=smtp.gmail.com
echo SMTP_PORT=587
echo EMAIL_USERNAME=your_email@gmail.com
echo EMAIL_PASSWORD=your_gmail_app_password_here
echo.
echo # Twilio SMS alerts
echo TWILIO_ACCOUNT_SID=your_twilio_account_sid_here
echo TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
echo.
echo # Webhook alerts
echo WEBHOOK_URL=https://your-webhook-url.com/alerts
echo WEBHOOK_TOKEN=your_webhook_token_here
echo.
echo # ============================================================================
echo # DATABASE CONFIGURATION
echo # ============================================================================
echo # For production database ^(PostgreSQL example^)
echo DATABASE_URL=postgresql://username:password@localhost:5432/archneuronx
echo.
echo # Redis for caching
echo REDIS_URL=redis://localhost:6379
echo.
echo # ============================================================================
echo # SECURITY CONFIGURATION
echo # ============================================================================
echo # JWT secret for API authentication
echo JWT_SECRET=your_jwt_secret_here_minimum_32_characters
echo.
echo # Encryption key for sensitive data ^(32 characters minimum^)
echo ENCRYPTION_KEY=your_32_character_encryption_key_here
echo.
echo # ============================================================================
echo # MONITORING CONFIGURATION
echo # ============================================================================
echo # Prometheus metrics
echo PROMETHEUS_PORT=9090
echo.
echo # Grafana dashboard
echo GRAFANA_PORT=3000
echo GRAFANA_ADMIN_USER=admin
echo GRAFANA_ADMIN_PASSWORD=your_grafana_password_here
echo.
echo # ============================================================================
echo # DEVELOPMENT CONFIGURATION
echo # ============================================================================
echo # Environment mode: development, staging, production
echo NODE_ENV=development
echo.
echo # Debug mode: true/false
echo DEBUG_MODE=false
echo.
echo # Verbose logging: true/false
echo VERBOSE_LOGGING=false
echo.
echo # Mock data for testing: true/false
echo MOCK_DATA=true
echo.
echo # Paper trading mode: true/false
echo PAPER_TRADING=true
echo.
echo # ============================================================================
echo # TRADING CONFIGURATION
echo # ============================================================================
echo # Risk per trade in USD
echo RISK_PER_TRADE=100
echo.
echo # Maximum position size in USD
echo MAX_POSITION_SIZE=1000
echo.
echo # Maximum daily loss in USD
echo MAX_DAILY_LOSS=1000
echo.
echo # Maximum drawdown percentage
echo MAX_DRAWDOWN=0.1
echo.
echo # Stop loss percentage
echo STOP_LOSS_PCT=0.02
echo.
echo # Take profit percentage
echo TAKE_PROFIT_PCT=0.05
echo.
echo # ============================================================================
echo # WEB INTERFACE CONFIGURATION
echo # ============================================================================
echo # HTTP port
echo HTTP_PORT=8080
echo.
echo # WebSocket port
echo WEBSOCKET_PORT=3001
echo.
echo # API rate limit per minute
echo API_RATE_LIMIT=1000
) > .env

call :print_status "✅ .env file created successfully"
call :print_warning "🔒 Please edit .env file with your actual API keys"
call :print_warning "🚨 NEVER commit .env file to Git!"

goto :eof

REM Setup .gitignore
:setup_gitignore
call :print_header "SETTING UP GITIGNORE"

if exist ".gitignore" (
    findstr /C:".env" .gitignore >nul
    if errorlevel 1 (
        echo. >> .gitignore
        echo # Environment variables - SECURE! >> .gitignore
        echo .env >> .gitignore
        echo .env.* >> .gitignore
        echo config/live_trading.yaml >> .gitignore
        call :print_status "✅ Added .env to .gitignore"
    ) else (
        call :print_status "✅ .env already in .gitignore"
    )
) else (
    (
echo # Environment variables - SECURE!
echo .env
echo .env.*
echo config/live_trading.yaml
echo.
echo # Build directories
echo build/
echo dist/
echo.
echo # Logs
echo logs/
echo *.log
echo.
echo # OS generated files
echo .DS_Store
echo .DS_Store?
echo ._*
echo .Spotlight-V100
echo .Trashes
echo ehthumbs.db
echo Thumbs.db
echo.
echo # IDE files
echo .vscode/
echo .idea/
echo *.swp
echo *.swo
echo *~
echo.
echo # Python
echo __pycache__/
echo *.py[cod]
echo *$py.class
echo *.so
echo .Python
echo env/
echo venv/
echo ENV/
echo env.bak/
echo venv.bak/
echo.
echo # Node.js
echo node_modules/
echo npm-debug.log*
echo yarn-debug.log*
echo yarn-error.log*
echo.
echo # Temporary files
echo *.tmp
echo *.temp
echo *.bak
echo *.backup
echo.
echo # Security files
echo *.key
echo *.pem
echo *.crt
echo *.p12
echo *.pfx
echo.
echo # Database
echo *.db
echo *.sqlite
echo *.sqlite3
echo.
echo # Cache
echo .cache/
echo cache/
echo.
echo # Docker
echo .dockerignore
echo.
echo # Kubernetes
echo secrets/
    ) > .gitignore
    call :print_status "✅ Created .gitignore file"
)

goto :eof

REM Create configuration loader
:create_config_loader
call :print_header "CREATING CONFIGURATION LOADER"

(
echo @echo off
echo REM ArchNeuronX v4.0 Configuration Loader for Windows
echo REM Secure environment variable substitution for YAML files
echo.
echo python -c "
echo import os
echo import re
echo import sys
echo import yaml
echo from pathlib import Path
echo.
echo def substitute_env_vars^(text^):
echo     """Replace ${VAR} patterns with environment variables"""
echo     pattern = re.compile^(r'\$\{([^}]+)\}'^)
echo     
echo     def replace_match^(match^):
echo         var_name = match.group^(1^)
echo         value = os.getenv^(var_name^)
echo         if value is None:
echo             print^(f'⚠️  Environment variable {var_name} not found!'^)
echo             return f'${{{var_name}}}'
echo         return value
echo     
echo     return pattern.sub^(replace_match, text^)
echo.
echo def load_config^(template_path, output_path^):
echo     """Load template YAML and substitute environment variables"""
echo     try:
echo         # Read template file
echo         with open^(template_path, 'r'^) as f:
echo             template_content = f.read^(^)
echo         
echo         # Substitute environment variables
echo         processed_content = substitute_env_vars^(template_content^)
echo         
echo         # Parse YAML to validate
echo         config = yaml.safe_load^(processed_content^)
echo         
echo         # Write processed configuration
echo         with open^(output_path, 'w'^) as f:
echo             yaml.dump^(config, f, default_flow_style=False, indent=2^)
echo         
echo         print^(f'✅ Configuration loaded: {output_path}'^)
echo         return True
echo         
echo     except Exception as e:
echo         print^(f'❌ Error loading configuration: {e}'^)
echo         return False
echo.
echo def main^(^):
echo     """Main function"""
echo     if len^(sys.argv^) != 3:
echo         print^('Usage: python load_config.py ^<template_file^> ^<output_file^>'^)
echo         sys.exit^(1^)
echo     
echo     template_file = sys.argv[1]
echo     output_file = sys.argv[2]
echo     
echo     # Check if template file exists
echo     if not os.path.exists^(template_file^):
echo         print^(f'❌ Template file not found: {template_file}'^)
echo         sys.exit^(1^)
echo     
echo     # Load environment variables from .env file
echo     env_file = Path^('.env'^)
echo     if env_file.exists^(^):
echo         print^('📁 Loading environment variables from .env file...'^)
echo         with open^(env_file, 'r'^) as f:
echo             for line in f:
echo                 line = line.strip^(^)
echo                 if line and not line.startswith^('#'^) and '=' in line:
echo                     key, value = line.split^('=', 1^)
echo                     os.environ[key] = value
echo         print^('✅ Environment variables loaded'^)
echo     else:
echo         print^('⚠️  .env file not found. Using system environment variables.'^)
echo     
echo     # Load configuration
echo     success = load_config^(template_file, output_file^)
echo     
echo     if success:
echo         print^('🎉 Configuration loaded successfully!'^)
echo         
echo         # Verify critical variables
echo         critical_vars = ['BINANCE_API_KEY', 'BINANCE_API_SECRET']
echo         missing_vars = []
echo         
echo         for var in critical_vars:
echo             if not os.getenv^(var^):
echo                 missing_vars.append^(var^)
echo         
echo         if missing_vars:
echo             print^(f'⚠️  Missing critical environment variables: {", ".join^(missing_vars^)}'^)
echo             print^('📝 Please update your .env file with these variables.'^)
echo         else:
echo             print^('✅ All critical environment variables are set!'^)
echo     
echo     sys.exit^(0 if success else 1^)
echo.
echo if __name__ == "__main__":
echo     main^(^)
echo " %%*
echo.
echo REM Load environment variables from .env file
echo if exist ".env" (
echo     echo 📁 Loading environment variables from .env file...
echo     for /f "usebackq tokens=1,2 delims==" %%a in ^(".env"^) do (
echo         if not "%%a"=="" if not "%%a:~0,1%"=="#" set "%%a=%%b"
echo     )
echo     echo ✅ Environment variables loaded
echo ^) else (
echo     echo ⚠️  .env file not found. Using system environment variables.
echo ^)
echo.
echo REM Check arguments
echo if "%~1"=="" (
echo     echo Usage: %~nx0 ^<template_file^> ^<output_file^>
echo     exit /b 1
echo ^)
echo.
echo if "%~2"=="" (
echo     echo Usage: %~nx0 ^<template_file^> ^<output_file^>
echo     exit /b 1
echo ^)
echo.
echo REM Run Python script
echo python "%~dp0load_config.py" %1 %2
) > scripts\load_config.bat

call :print_status "✅ Configuration loader created"

goto :eof

REM Create secure startup script
:create_secure_startup
call :print_header "CREATING SECURE STARTUP SCRIPT"

(
echo @echo off
echo REM ArchNeuronX v4.0 Secure Startup Script for Windows
echo REM Loads environment variables and starts live trading
echo.
echo setlocal enabledelayedexpansion
echo.
echo REM Colors for output
echo set "RED=[91m"
echo set "GREEN=[92m"
echo set "YELLOW=[93m"
echo set "BLUE=[94m"
echo set "NC=[0m"
echo.
echo REM Print colored output
echo :print_status
echo echo %%GREEN%%[INFO]%%NC%% %%~1
echo goto :eof
echo.
echo :print_warning
echo echo %%YELLOW%%[WARNING]%%NC%% %%~1
echo goto :eof
echo.
echo :print_error
echo echo %%RED%%[ERROR]%%NC%% %%~1
echo goto :eof
echo.
echo :print_header
echo echo %%BLUE%%========================================%%NC%%
echo echo %%BLUE%%%%~1%%NC%%
echo echo %%BLUE%%========================================%%NC%%
echo goto :eof
echo.
echo REM Check if .env file exists
echo :check_env_file
echo if not exist ".env" (
echo     call :print_error "❌ .env file not found!"
echo     call :print_status "📝 Please run: scripts\setup_env.bat"
echo     exit /b 1
echo ^)
echo.
echo call :print_status "✅ .env file found"
echo goto :eof
echo.
echo REM Load environment variables
echo :load_env_vars
echo call :print_status "📁 Loading environment variables..."
echo.
echo REM Load from .env file
echo for /f "usebackq tokens=1,2 delims==" %%a in ^(".env"^) do (
echo     if not "%%a"=="" if not "%%a:~0,1%"=="#" set "%%a=%%b"
echo ^)
echo.
echo call :print_status "✅ Environment variables loaded"
echo goto :eof
echo.
echo REM Verify critical variables
echo :verify_critical_vars
echo call :print_status "🔍 Verifying critical environment variables..."
echo.
echo set "missing_vars="
echo.
echo REM Check BINANCE_API_KEY
echo if "!BINANCE_API_KEY!"=="" set "missing_vars=!missing_vars! BINANCE_API_KEY"
echo if "!BINANCE_API_KEY!"=="your_binance_api_key_here" set "missing_vars=!missing_vars! BINANCE_API_KEY"
echo.
echo REM Check BINANCE_API_SECRET
echo if "!BINANCE_API_SECRET!"=="" set "missing_vars=!missing_vars! BINANCE_API_SECRET"
echo if "!BINANCE_API_SECRET!"=="your_binance_api_secret_here" set "missing_vars=!missing_vars! BINANCE_API_SECRET"
echo.
echo if not "!missing_vars!"=="" (
echo     call :print_error "❌ Missing or unset critical variables:!missing_vars!"
echo     call :print_status "📝 Please update your .env file with these variables."
echo     exit /b 1
echo ^)
echo.
echo call :print_status "✅ All critical environment variables are set!"
echo goto :eof
echo.
echo REM Generate configuration
echo :generate_config
echo call :print_status "🔧 Generating configuration from template..."
echo.
echo REM Check if template exists
echo if not exist "config\live_trading_template.yaml" (
echo     call :print_error "❌ Configuration template not found!"
echo     exit /b 1
echo ^)
echo.
echo REM Use batch script to substitute variables
echo call scripts\load_config.bat config\live_trading_template.yaml config\live_trading.yaml
echo.
echo if !errorlevel! neq 0 (
echo     call :print_error "❌ Failed to generate configuration"
echo     exit /b 1
echo ^)
echo.
echo call :print_status "✅ Configuration generated successfully"
echo goto :eof
echo.
echo REM Start live trading
echo :start_trading
echo call :print_header "🚀 STARTING ARCHNEURONX LIVE TRADING"
echo.
echo REM Check if executable exists
echo if not exist "build\archneuronx_live_trading.exe" (
echo     call :print_error "❌ Live trading executable not found!"
echo     call :print_status "🏗️  Please build first: scripts\run_live_trading.bat build"
echo     exit /b 1
echo ^)
echo.
echo REM Set environment variables
echo set ARCHNEURONX_CONFIG_FILE=config\live_trading.yaml
echo set ARCHNEURONX_ENV_LOADED=true
echo.
echo call :print_status "🎯 Starting live trading system..."
echo call :print_warning "🔒 Using secure configuration with API keys from environment"
echo.
echo REM Start live trading
echo build\archneuronx_live_trading.exe
echo goto :eof
echo.
echo REM Main execution
echo :main
echo call :print_header "🔒 ARCHNEURONX SECURE STARTUP"
echo.
echo call :check_env_file
echo call :load_env_vars
echo call :verify_critical_vars
echo call :generate_config
echo call :start_trading
echo goto :eof
echo.
echo REM Run main function
echo call :main %%*
) > scripts\start_secure.bat

call :print_status "✅ Secure startup script created"

goto :eof

REM Main function
:main
call :print_header "🔒 ARCHNEURONX V4.0 ENVIRONMENT SETUP"

call :print_status "🚀 Setting up secure environment for API keys..."

call :create_env_file
call :setup_gitignore
call :create_config_loader
call :create_secure_startup

call :print_header "✅ SETUP COMPLETED"
call :print_status "🎉 Secure environment setup completed!"
echo.
call :print_status "📝 NEXT STEPS:"
echo "   1. Edit .env file with your actual API keys"
echo "   2. Run: scripts\start_secure.bat"
echo "   3. Or use: scripts\load_config.bat config\live_trading_template.yaml config\live_trading.yaml"
echo.
call :print_warning "🔒 IMPORTANT: Never commit .env file to Git!"
call :print_warning "🔒 Your API keys are now secure and isolated from version control!"

goto :eof

REM Run main function
call :main %*
