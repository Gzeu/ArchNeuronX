@echo off
REM ArchNeuronX Build Script for Windows
REM Usage: build.bat [release|debug] [cuda|cpu]

setlocal enabledelayedexpansion

REM Default parameters
set BUILD_TYPE=Release
set CUDA_SUPPORT=ON

REM Parse arguments
if "%1"=="debug" set BUILD_TYPE=Debug
if "%2"=="cpu" set CUDA_SUPPORT=OFF

echo 🚀 Building ArchNeuronX v2.0
echo Build Type: %BUILD_TYPE%
echo CUDA Support: %CUDA_SUPPORT%
echo ==================================

REM Check dependencies
echo 📋 Checking dependencies...

REM Check CMake
cmake --version >nul 2>&1
if errorlevel 1 (
    echo ❌ CMake not found. Please install CMake 3.20+
    exit /b 1
)

REM Check CUDA if enabled
if "%CUDA_SUPPORT%"=="ON" (
    nvcc --version >nul 2>&1
    if errorlevel 1 (
        echo ❌ CUDA compiler not found. Please install CUDA 11.8+
        exit /b 1
    )
    
    REM Check GPU
    nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo ⚠️  nvidia-smi not found. GPU may not be available.
    ) else (
        echo ✅ GPU detected:
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    )
)

REM Create build directory
set BUILD_DIR=build-%BUILD_TYPE%
echo 📁 Creating build directory: %BUILD_DIR%
if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
mkdir %BUILD_DIR%
cd %BUILD_DIR%

REM Configure
echo ⚙️  Configuring build...
set CMAKE_ARGS=-DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DUSE_CUDA=%CUDA_SUPPORT%

if "%BUILD_TYPE%"=="Release" (
    set CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
)

cmake .. %CMAKE_ARGS%
if errorlevel 1 (
    echo ❌ CMake configuration failed
    exit /b 1
)

REM Build
echo 🔨 Building...
cmake --build . --config %BUILD_TYPE% --parallel
if errorlevel 1 (
    echo ❌ Build failed
    exit /b 1
)

REM Run tests if available
if exist "test\%BUILD_TYPE%\archneuronx_test.exe" (
    echo 🧪 Running tests...
    test\%BUILD_TYPE%\archneuronx_test.exe
)

echo.
echo ✅ Build completed successfully!
echo 📦 Binary location: %BUILD_DIR%\src\%BUILD_TYPE%\archneuronx.exe
echo.
echo 🎯 Quick start commands:
echo   CPU-only:     archneuronx.exe --config ..\config\profiles\development.json
echo   With CUDA:    archneuronx.exe --config ..\config\deployment.json --cuda
echo   Backtest:      archneuronx.exe --backtest --data ..\data\sample\
echo   Help:          archneuronx.exe --help
pause
