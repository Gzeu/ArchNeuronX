@echo off
REM ArchNeuronX v4.0 Test Runner for Windows
REM This script runs all tests for the complete trading system

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
call :print_header "CHECKING TEST DEPENDENCIES"

REM Check for CMake
cmake --version >nul 2>&1
if errorlevel 1 (
    call :print_error "CMake is not installed. Please install CMake 3.20+"
    exit /b 1
)

REM Check for vcpkg (for Google Test)
vcpkg version >nul 2>&1
if errorlevel 1 (
    call :print_warning "vcpkg may not be installed. Installing Google Test..."
    
    REM Try to install Google Test via vcpkg
    vcpkg install gtest gtest:x64-windows gmock gmock:x64-windows
    
    if errorlevel 1 (
        call :print_error "Please install Google Test/Mock manually"
        exit /b 1
    )
)

REM Check for torch
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    call :print_warning "PyTorch may not be installed. Please install LibTorch 2.6+"
)

call :print_status "Test dependencies check completed"
goto :eof

REM Function to build tests
:build_tests
call :print_header "BUILDING TESTS"

REM Create build directory
if not exist "build" (
    call :print_status "Creating build directory"
    mkdir build
)

cd build

REM Configure with CMake
call :print_status "Configuring tests with CMake..."
cmake -DCMAKE_BUILD_TYPE=Debug ^
      -DENABLE_COVERAGE=ON ^
      -DUSE_CUDA=ON ^
      -DBUILD_V4_QUANTUM=ON ^
      -DBUILD_LLM_INTEGRATION=ON ^
      ..

REM Build tests
call :print_status "Building tests..."
cmake --build . --config Debug --target archneuronx_tests

REM Check if build was successful
if errorlevel 1 (
    call :print_error "Test build failed"
    cd ..
    exit /b 1
)

call :print_status "Tests built successfully"
cd ..

goto :eof

REM Function to run all tests
:run_all_tests
call :print_header "RUNNING ALL TESTS"

if not exist "build\archneuronx_tests.exe" (
    call :print_error "Test executable not found. Please build tests first."
    exit /b 1
)

call :print_status "Running all tests..."
build\archneuronx_tests.exe --gtest_output=xml

REM Check test results
if errorlevel 1 (
    call :print_error "Some tests failed!"
    exit /b 1
) else (
    call :print_status "All tests passed!"
)

goto :eof

REM Function to run unit tests
:run_unit_tests
call :print_header "RUNNING UNIT TESTS"

if not exist "build\archneuronx_tests.exe" (
    call :print_error "Test executable not found. Please build tests first."
    exit /b 1
)

call :print_status "Running unit tests..."
build\archneuronx_tests.exe --gtest_filter="Unit*" --gtest_output=xml

if errorlevel 1 (
    call :print_error "Unit tests failed!"
    exit /b 1
) else (
    call :print_status "Unit tests passed!"
)

goto :eof

REM Function to run integration tests
:run_integration_tests
call :print_header "RUNNING INTEGRATION TESTS"

if not exist "build\archneuronx_tests.exe" (
    call :print_error "Test executable not found. Please build tests first."
    exit /b 1
)

call :print_status "Running integration tests..."
build\archneuronx_tests.exe --gtest_filter="Integration*" --gtest_output=xml

if errorlevel 1 (
    call :print_error "Integration tests failed!"
    exit /b 1
) else (
    call :print_status "Integration tests passed!"
)

goto :eof

REM Function to run quantum tests
:run_quantum_tests
call :print_header "RUNNING QUANTUM TESTS"

if not exist "build\archneuronx_tests.exe" (
    call :print_error "Test executable not found. Please build tests first."
    exit /b 1
)

call :print_status "Running quantum tests..."
build\archneuronx_tests.exe --gtest_filter="*Quantum*" --gtest_output=xml

if errorlevel 1 (
    call :print_error "Quantum tests failed!"
    exit /b 1
) else (
    call :print_status "Quantum tests passed!"
)

goto :eof

REM Function to run LLM tests
:run_llm_tests
call :print_header "RUNNING LLM TESTS"

if not exist "build\archneuronx_tests.exe" (
    call :print_error "Test executable not found. Please build tests first."
    exit /b 1
)

call :print_status "Running LLM tests..."
build\archneuronx_tests.exe --gtest_filter="*LLM*" --gtest_output=xml

if errorlevel 1 (
    call :print_error "LLM tests failed!"
    exit /b 1
) else (
    call :print_status "LLM tests passed!"
)

goto :eof

REM Function to run agent tests
:run_agent_tests
call :print_header "RUNNING AGENT TESTS"

if not exist "build\archneuronx_tests.exe" (
    call :print_error "Test executable not found. Please build tests first."
    exit /b 1
)

call :print_status "Running agent tests..."
build\archneuronx_tests.exe --gtest_filter="*Agent*" --gtest_output=xml

if errorlevel 1 (
    call :print_error "Agent tests failed!"
    exit /b 1
) else (
    call :print_status "Agent tests passed!"
)

goto :eof

REM Function to run performance tests
:run_performance_tests
call :print_header "RUNNING PERFORMANCE TESTS"

if not exist "build\archneuronx_tests.exe" (
    call :print_error "Test executable not found. Please build tests first."
    exit /b 1
)

call :print_status "Running performance tests..."
build\archneuronx_tests.exe --gtest_filter="*Performance*" --gtest_output=xml

if errorlevel 1 (
    call :print_error "Performance tests failed!"
    exit /b 1
) else (
    call :print_status "Performance tests passed!"
)

goto :eof

REM Function to run stress tests
:run_stress_tests
call :print_header "RUNNING STRESS TESTS"

if not exist "build\archneuronx_tests.exe" (
    call :print_error "Test executable not found. Please build tests first."
    exit /b 1
)

call :print_status "Running stress tests..."

REM Run tests multiple times to check for memory leaks
for /l %%i in (1,1,10) do (
    call :print_status "Stress test iteration %%i/10..."
    build\archneuronx_tests.exe --gtest_filter="*Performance*" --gtest_output=xml
    
    if errorlevel 1 (
        call :print_error "Stress test failed at iteration %%i!"
        exit /b 1
    )
)

call :print_status "Stress tests completed successfully!"
goto :eof

REM Function to clean test build
:clean_tests
call :print_header "CLEANING TEST BUILD"

if exist "build" (
    call :print_status "Removing build directory..."
    rmdir /s /q build
    call :print_status "✅ Build directory removed"
) else (
    call :print_status "Build directory does not exist"
)

REM Clean test outputs
if exist "test_results" (
    call :print_status "Cleaning test results..."
    rmdir /s /q test_results
    call :print_status "✅ Test results cleaned"
)

REM Clean coverage files
if exist "coverage_html" (
    call :print_status "Cleaning coverage files..."
    rmdir /s /q coverage_html
    call :print_status "✅ Coverage files cleaned"
)

goto :eof

REM Function to show help
:show_help
echo ArchNeuronX v4.0 Test Runner for Windows
echo.
echo Usage: %~nx0 [COMMAND]
echo.
echo Commands:
echo   build           Build the test suite
echo   all             Run all tests
echo   unit            Run unit tests only
echo   integration     Run integration tests only
echo   quantum         Run quantum tests only
echo   llm             Run LLM tests only
echo   agents          Run agent tests only
echo   performance     Run performance tests only
echo   stress          Run stress tests
echo   clean           Clean test build and outputs
echo   help            Show this help message
echo.
echo Examples:
echo   %~nx0 build                    # Build tests
echo   %~nx0 all                      # Run all tests
echo   %~nx0 quantum                  # Run quantum tests
echo   %~nx0 stress                   # Run stress tests
echo.
echo Test Categories:
echo   🧠 Quantum Neural Networks
echo   🤖 Quantum Trading Agents
echo   🤖 LLM Integration
echo   🌐 Web Interface
echo   🤝 Multi-Agent Coordination
echo   📊 Performance Benchmarks

goto :eof

REM Main script logic
if "%1"=="" goto :help
if "%1"=="build" goto :build_tests
if "%1"=="all" goto :run_all_tests
if "%1"=="unit" goto :run_unit_tests
if "%1"=="integration" goto :run_integration_tests
if "%1"=="quantum" goto :run_quantum_tests
if "%1"=="llm" goto :run_llm_tests
if "%1"=="agents" goto :run_agent_tests
if "%1"=="performance" goto :run_performance_tests
if "%1"=="stress" goto :run_stress_tests
if "%1"=="clean" goto :clean_tests
if "%1"=="help" goto :help
goto :help

:eof
