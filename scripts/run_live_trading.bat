@echo off
REM ArchNeuronX v4.0 Live Trading Runner for Windows
REM This script runs the live trading system with real market data

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

REM Check if we're in the right directory
if not exist "CMakeLists.txt" (
    call :print_error "Please run this script from the ArchNeuronX root directory"
    exit /b 1
)

REM Function to check dependencies
:check_dependencies
call :print_header "CHECKING LIVE TRADING DEPENDENCIES"

REM Check for CMake
cmake --version >nul 2>&1
if errorlevel 1 (
    call :print_error "CMake is not installed. Please install CMake 3.20+"
    exit /b 1
)

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Python is not installed. Please install Python 3.8+"
    exit /b 1
)

REM Check for PyTorch
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    call :print_warning "PyTorch is not installed. Installing..."
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

REM Check for curl
curl --version >nul 2>&1
if errorlevel 1 (
    call :print_error "curl is not installed. Please install curl"
    exit /b 1
)

call :print_status "Live trading dependencies check completed"
goto :eof

REM Function to build live trading system
:build_live_trading
call :print_header "BUILDING LIVE TRADING SYSTEM"

REM Create build directory
if not exist "build" (
    call :print_status "Creating build directory"
    mkdir build
)

cd build

REM Configure with CMake
call :print_status "Configuring live trading system with CMake..."
cmake -DCMAKE_BUILD_TYPE=Release ^
      -DUSE_CUDA=OFF ^
      -DBUILD_V4_QUANTUM=ON ^
      -DBUILD_LLM_INTEGRATION=ON ^
      -DBUILD_LIVE_TRADING=ON ^
      ..

REM Build live trading system
call :print_status "Building live trading system..."
cmake --build . --config Release --target archneuronx_live_trading

REM Check if build was successful
if errorlevel 1 (
    call :print_error "Live trading system build failed"
    cd ..
    exit /b 1
)

call :print_status "Live trading system built successfully"
cd ..

goto :eof

REM Function to run live trading in interactive mode
:run_interactive
call :print_header "STARTING LIVE TRADING - INTERACTIVE MODE"

if not exist "build\archneuronx_live_trading.exe" (
    call :print_error "Live trading executable not found. Please build first."
    exit /b 1
)

call :print_status "Starting live trading in interactive mode..."
call :print_warning "This is a DEMO version with simulated market data"
call :print_warning "DO NOT use with real money without proper testing"

REM Set environment variables
set ARCHNEURONX_MODE=live
set TRADING_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT
set RISK_PER_TRADE=100.0
set MAX_POSITION_SIZE=1000.0
set TRADING_INTERVAL=1000

REM Run live trading
build\archneuronx_live_trading.exe

goto :eof

REM Function to run live trading in automated mode
:run_automated
call :print_header "STARTING LIVE TRADING - AUTOMATED MODE"

if not exist "build\archneuronx_live_trading.exe" (
    call :print_error "Live trading executable not found. Please build first."
    exit /b 1
)

call :print_status "Starting live trading in automated mode..."
call :print_warning "This is a DEMO version with simulated market data"
call :print_warning "DO NOT use with real money without proper testing"

REM Set environment variables
set ARCHNEURONX_MODE=live
set TRADING_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT
set RISK_PER_TRADE=100.0
set MAX_POSITION_SIZE=1000.0
set TRADING_INTERVAL=1000
set AUTOMATED_MODE=true

REM Run live trading in background
start /B build\archneuronx_live_trading.exe > logs\live_trading.log 2>&1

REM Store PID (simplified for Windows)
echo %TIME% > logs\live_trading.pid

call :print_status "Live trading started in background"
call :print_status "Logs are being written to: logs\live_trading.log"
call :print_status "To stop: run_live_trading.bat stop"

goto :eof

REM Function to stop live trading
:stop_live_trading
call :print_header "STOPPING LIVE TRADING"

REM Try to stop the process
taskkill /F /IM archneuronx_live_trading.exe >nul 2>&1
if errorlevel 1 (
    call :print_warning "Live trading process not found"
) else (
    call :print_status "Live trading stopped"
)

REM Clean PID file
if exist "logs\live_trading.pid" (
    del logs\live_trading.pid
)

goto :eof

REM Function to show live trading status
:show_status
call :print_header "LIVE TRADING STATUS"

REM Check if process is running
tasklist /FI "IMAGENAME eq archneuronx_live_trading.exe" | find "archneuronx_live_trading.exe" >nul
if errorlevel 1 (
    call :print_status "Live trading is NOT running"
) else (
    call :print_status "Live trading is RUNNING"
    
    REM Show recent logs
    if exist "logs\live_trading.log" (
        call :print_status "Recent logs:"
        powershell "Get-Content 'logs\live_trading.log' | Select-Object -Last 20"
    )
)

goto :eof

REM Function to show logs
:show_logs
call :print_header "LIVE TRADING LOGS"

if exist "logs\live_trading.log" (
    call :print_status "Showing live trading logs:"
    powershell "Get-Content 'logs\live_trading.log' -Wait"
) else (
    call :print_error "No log file found"
)

goto :eof

REM Function to run backtesting
:run_backtest
call :print_header "RUNNING BACKTESTING"

if not exist "build\archneuronx_live_trading.exe" (
    call :print_error "Live trading executable not found. Please build first."
    exit /b 1
)

call :print_status "Running backtesting on historical data..."

REM Set environment variables
set ARCHNEURONX_MODE=backtest
set BACKTEST_DATA_PATH=data\historical\
set BACKTEST_START_DATE=2024-01-01
set BACKTEST_END_DATE=2024-12-31

REM Run backtesting
build\archneuronx_live_trading.exe --backtest

goto :eof

REM Function to run paper trading
:run_paper_trading
call :print_header "STARTING PAPER TRADING"

if not exist "build\archneuronx_live_trading.exe" (
    call :print_error "Live trading executable not found. Please build first."
    exit /b 1
)

call :print_status "Starting paper trading with simulated money..."

REM Set environment variables
set ARCHNEURONX_MODE=paper
set PAPER_BALANCE=10000.0
set TRADING_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT
set RISK_PER_TRADE=100.0
set MAX_POSITION_SIZE=1000.0

REM Run paper trading
build\archneuronx_live_trading.exe --paper

goto :eof

REM Function to clean build
:clean_build
call :print_header "CLEANING LIVE TRADING BUILD"

if exist "build" (
    call :print_status "Removing build directory..."
    rmdir /s /q build
    call :print_status "✅ Build directory removed"
) else (
    call :print_status "Build directory does not exist"
)

REM Clean logs
if exist "logs\live_trading.log" (
    call :print_status "Cleaning logs..."
    del logs\live_trading.log
    call :print_status "✅ Logs cleaned"
)

REM Clean PID file
if exist "logs\live_trading.pid" (
    del logs\live_trading.pid
)

goto :eof

REM Function to show help
:show_help
echo ArchNeuronX v4.0 Live Trading Runner for Windows
echo.
echo Usage: %~nx0 [COMMAND]
echo.
echo Commands:
echo   build           Build the live trading system
echo   interactive     Run live trading in interactive mode
echo   automated      Run live trading in automated mode
echo   stop            Stop automated live trading
echo   status          Show live trading status
echo   logs            Show live trading logs
echo   backtest        Run backtesting on historical data
echo   paper           Run paper trading with simulated money
echo   clean           Clean build and logs
echo   help            Show this help message
echo.
echo Examples:
echo   %~nx0 build                    # Build live trading system
echo   %~nx0 interactive              # Start interactive trading
echo   %~nx0 automated                # Start automated trading
echo   %~nx0 stop                     # Stop automated trading
echo   %~nx0 status                   # Show trading status
echo   %~nx0 logs                     # Show live logs
echo   %~nx0 backtest                 # Run backtesting
echo   %~nx0 paper                    # Start paper trading
echo.
echo Trading Modes:
echo   🎯 Interactive Mode - Manual control with commands
echo   🤖 Automated Mode - Fully automated trading
echo   📊 Backtesting Mode - Test on historical data
echo   📝 Paper Trading - Test with simulated money
echo.
echo Risk Management:
echo   🛡️ Built-in risk management with stop-loss and take-profit
echo   📊 Real-time portfolio monitoring
echo   🚨 Alert system for risk events
echo   📈 Performance tracking and analytics

goto :eof

REM Create logs directory
if not exist "logs" mkdir logs

REM Main script logic
if "%1"=="" goto :help
if "%1"=="build" goto :build_live_trading
if "%1"=="interactive" goto :run_interactive
if "%1"=="automated" goto :run_automated
if "%1"=="stop" goto :stop_live_trading
if "%1"=="status" goto :show_status
if "%1"=="logs" goto :show_logs
if "%1"=="backtest" goto :run_backtest
if "%1"=="paper" goto :run_paper_trading
if "%1"=="clean" goto :clean_build
if "%1"=="help" goto :help
goto :help

:eof
