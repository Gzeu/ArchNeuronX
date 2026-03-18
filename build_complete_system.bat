@echo off
REM ArchNeuronX v4.0 Complete Trading System Builder for Windows
REM This script builds and runs the complete trading system with all components

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
call :print_header "CHECKING DEPENDENCIES"

REM Check for CMake
cmake --version >nul 2>&1
if errorlevel 1 (
    call :print_error "CMake is not installed. Please install CMake 3.20+"
    exit /b 1
)
call :print_status "CMake found: $(cmake --version ^| head -1)"

REM Check for C++ compiler
cl >nul 2>&1
if errorlevel 1 (
    g++ --version >nul 2>&1
    if errorlevel 1 (
        call :print_error "C++ compiler is not installed. Please install Visual Studio or g++"
        exit /b 1
    )
    call :print_status "G++ compiler found"
) else (
    call :print_status "MSVC compiler found"
)

REM Check for Python (for LibTorch)
python --version >nul 2>&1
if errorlevel 1 (
    call :print_warning "Python is not installed. LibTorch may not be available"
) else (
    call :print_status "Python found: $(python --version)"
)

REM Check for CUDA (optional)
nvcc --version >nul 2>&1
if errorlevel 1 (
    call :print_warning "CUDA not found. GPU acceleration will be disabled"
) else (
    call :print_status "CUDA found: $(nvcc --version ^| head -1)"
)

call :print_status "Dependencies check completed"
goto :eof

REM Function to build the system
:build_system
call :print_header "BUILDING COMPLETE TRADING SYSTEM"

REM Create build directory
if not exist "build" (
    call :print_status "Creating build directory"
    mkdir build
)

cd build

REM Configure with CMake
call :print_status "Configuring with CMake..."
cmake -DCMAKE_BUILD_TYPE=Release ^
      -DUSE_CUDA=ON ^
      -DBUILD_V4_QUANTUM=ON ^
      -DBUILD_LLM_INTEGRATION=ON ^
      -DENABLE_GPU_ACCELERATION=ON ^
      -DENABLE_FLASH_ATTENTION=ON ^
      -DENABLE_MODEL_CACHING=ON ^
      ..

REM Build
call :print_status "Building complete system..."
cmake --build . --config Release --target archneuronx_complete

REM Check if build was successful
if errorlevel 1 (
    call :print_error "Build failed"
    cd ..
    exit /b 1
)

call :print_status "Build completed successfully"
cd ..

goto :eof

REM Function to run the system
:run_system
call :print_header "RUNNING COMPLETE TRADING SYSTEM"

REM Check if binary exists
if not exist "build\archneuronx_complete.exe" (
    call :print_error "Binary not found. Please build the system first."
    exit /b 1
)

REM Create models cache directory
if not exist "models\cache" (
    call :print_status "Creating models cache directory"
    mkdir models\cache
)

REM Create logs directory
if not exist "logs" (
    call :print_status "Creating logs directory"
    mkdir logs
)

REM Run the system
call :print_status "Starting ArchNeuronX v4.0 Complete Trading System..."
call :print_status "Web Interface: http://localhost:8080"
call :print_status "WebSocket: ws://localhost:3001"
call :print_status "Press Ctrl+C to stop"
echo.

REM Set environment variables
set CUDA_VISIBLE_DEVICES=0
set TORCH_CUDA_ARCH_LIST=8.6;8.9;9.0

REM Run the system
build\archneuronx_complete.exe %*

goto :eof

REM Function to run in continuous mode
:run_continuous
call :print_header "RUNNING CONTINUOUS TRADING MODE"

if not exist "build\archneuronx_complete.exe" (
    call :print_error "Binary not found. Please build the system first."
    exit /b 1
)

call :print_status "Starting continuous trading mode..."
call :print_status "Web Interface: http://localhost:8080"
call :print_status "WebSocket: ws://localhost:3001"
call :print_status "Press Ctrl+C to stop"
echo.

build\archneuronx_complete.exe --continuous

goto :eof

REM Function to show system status
:show_status
call :print_header "SYSTEM STATUS"

if not exist "build\archneuronx_complete.exe" (
    call :print_error "Binary not found. Please build the system first."
    exit /b 1
)

call :print_status "Checking system status..."

REM Check if system is running
tasklist /FI "IMAGENAME eq archneuronx_complete.exe" 2>nul | findstr "archneuronx_complete.exe" >nul
if errorlevel 1 (
    call :print_warning "ArchNeuronX is not running"
) else (
    call :print_status "✅ ArchNeuronX is running"
    call :print_status "Web Interface: http://localhost:8080"
    call :print_status "WebSocket: ws://localhost:3001"
)

goto :eof

REM Function to stop the system
:stop_system
call :print_header "STOPPING SYSTEM"

tasklist /FI "IMAGENAME eq archneuronx_complete.exe" 2>nul | findstr "archneuronx_complete.exe" >nul
if errorlevel 1 (
    call :print_warning "ArchNeuronX is not running"
) else (
    call :print_status "Stopping ArchNeuronX..."
    taskkill /F /IM archneuronx_complete.exe >nul 2>&1
    timeout /t 2 /nobreak >nul
    
    REM Check if still running and force stop
    tasklist /FI "IMAGENAME eq archneuronx_complete.exe" 2>nul | findstr "archneuronx_complete.exe" >nul
    if not errorlevel 1 (
        call :print_warning "System still running, forcing stop..."
        taskkill /F /IM archneuronx_complete.exe /T >nul 2>&1
    )
    
    call :print_status "✅ ArchNeuronX stopped"
)

goto :eof

REM Function to clean build
:clean_build
call :print_header "CLEANING BUILD"

if exist "build" (
    call :print_status "Removing build directory..."
    rmdir /s /q build
    call :print_status "✅ Build directory removed"
) else (
    call :print_status "Build directory does not exist"
)

REM Clean cache directories
if exist "models\cache" (
    call :print_status "Cleaning model cache..."
    del /q models\cache\* 2>nul
    call :print_status "✅ Model cache cleaned"
)

if exist "logs" (
    call :print_status "Cleaning logs..."
    del /q logs\* 2>nul
    call :print_status "✅ Logs cleaned"
)

goto :eof

REM Function to show help
:show_help
echo ArchNeuronX v4.0 Complete Trading System Builder for Windows
echo.
echo Usage: %~nx0 [COMMAND]
echo.
echo Commands:
echo   build       Build the complete trading system
echo   run         Run the system in interactive mode
echo   continuous   Run the system in continuous trading mode
echo   status      Show system status
echo   stop        Stop the running system
echo   clean       Clean build and cache
echo   help        Show this help message
echo.
echo Examples:
echo   %~nx0 build                    # Build the system
echo   %~nx0 run                      # Run in interactive mode
echo   %~nx0 continuous                # Run in continuous mode
echo   %~nx0 status                   # Show system status
echo   %~nx0 stop                     # Stop the system
echo.
echo Web Interface:
echo   HTTP Server: http://localhost:8080
echo   WebSocket: ws://localhost:3001
echo   API Endpoints: http://localhost:8080/api/v4/
echo.
echo System Components:
echo   🧠 Quantum Neural Networks
echo   🤖 Quantum Trading Agents
echo   🤖 HuggingFace LLM Integration
echo   🌐 Web Interface
echo   🤝 Multi-Agent Coordination
echo   📊 Real-time Monitoring

goto :eof

REM Main script logic
if "%1"=="" goto :help
if "%1"=="build" goto :build_system
if "%1"=="run" goto :run_system
if "%1"=="continuous" goto :run_continuous
if "%1"=="status" goto :show_status
if "%1"=="stop" goto :stop_system
if "%1"=="clean" goto :clean_build
if "%1"=="help" goto :help
goto :help

:eof
