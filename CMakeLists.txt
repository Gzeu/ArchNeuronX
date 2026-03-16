cmake_minimum_required(VERSION 3.20)
project(ArchNeuronX VERSION 2.0.0 LANGUAGES CXX CUDA)

# ============================================================
# C++20 standard (upgrade from C++17)
# ============================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================
# Compiler options
# ============================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native -ffast-math)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0 -DDEBUG -fsanitize=address)
        add_link_options(-fsanitize=address)
    endif()
    # Enable LTO for Release builds
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()

# ============================================================
# Options
# ============================================================
option(USE_CUDA "Enable CUDA GPU acceleration" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(ENABLE_PROFILING "Enable CUDA profiling" OFF)
option(USE_TENSORRT "Enable TensorRT optimization" OFF)

# ============================================================
# Find required packages
# ============================================================
find_package(Torch REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

# CUDA support
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
    if(CUDAToolkit_VERSION VERSION_LESS "12.0")
        message(WARNING "CUDA 12.0+ recommended, found ${CUDAToolkit_VERSION}")
    endif()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 --use_fast_math -arch=sm_80)
endif()

# HTTP client/server
find_package(cpprestsdk QUIET)
if(NOT cpprestsdk_FOUND)
    message(STATUS "cpprestsdk not found, using alternative HTTP implementation")
endif()

# WebSocket++ for real-time data
find_package(websocketpp QUIET)
if(NOT websocketpp_FOUND)
    message(STATUS "websocketpp not found, will use bundled header-only version")
endif()

# nlohmann/json for JSON handling
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# spdlog for structured logging
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.13.0
    )
    FetchContent_MakeAvailable(spdlog)
endif()

find_package(prometheus-cpp QUIET)
find_package(CURL REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem program_options)

# ============================================================
# Source files (v2: includes risk/, trading/, backtest/, monitoring/)
# ============================================================
set(CORE_SOURCES
    src/core/trading_engine.cpp
    src/core/order_manager.cpp
    src/core/portfolio_manager.cpp
    src/core/market_regime_detector.cpp
    src/core/ensemble_manager.cpp
)

set(MODEL_SOURCES
    src/models/lstm_model.cpp
    src/models/mlp_model.cpp
    src/models/cnn_model.cpp
    src/models/transformer_model.cpp
    src/models/temporal_fusion_transformer.cpp
    src/models/model_factory.cpp
    src/models/online_learner.cpp
)

set(DATA_SOURCES
    src/data/data_manager.cpp
    src/data/market_data_feed.cpp
    src/data/preprocessor.cpp
    src/data/technical_indicators.cpp
    src/data/feature_engineer.cpp
    src/data/websocket_feed.cpp
)

set(API_SOURCES
    src/api/rest_server.cpp
    src/api/predict_handler.cpp
    src/api/train_handler.cpp
    src/api/backtest_handler.cpp
    src/api/portfolio_handler.cpp
    src/api/websocket_handler.cpp
    src/api/auth_middleware.cpp
    src/api/rate_limiter.cpp
)

# v2: new dedicated modules
set(RISK_SOURCES
    src/risk/risk_manager.cpp
)

set(TRADING_SOURCES
    src/trading/signal_generator.cpp
)

set(BACKTEST_SOURCES
    src/backtest/backtest_engine.cpp
)

set(UTILS_SOURCES
    src/utils/logger.cpp
    src/utils/metrics.cpp
    src/utils/config_parser.cpp
    src/utils/cuda_utils.cpp
    src/utils/math_utils.cpp
    src/utils/prometheus_exporter.cpp
)

set(ALL_SOURCES
    ${CORE_SOURCES}
    ${MODEL_SOURCES}
    ${DATA_SOURCES}
    ${API_SOURCES}
    ${RISK_SOURCES}
    ${TRADING_SOURCES}
    ${BACKTEST_SOURCES}
    ${UTILS_SOURCES}
    src/main.cpp
)

# ============================================================
# Main executable
# ============================================================
add_executable(archneuronx ${ALL_SOURCES})

target_include_directories(archneuronx PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
)

target_link_libraries(archneuronx PRIVATE
    ${TORCH_LIBRARIES}
    ${CURL_LIBRARIES}
    ${Boost_LIBRARIES}
    OpenSSL::SSL
    OpenSSL::Crypto
    Threads::Threads
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)

if(USE_CUDA)
    target_link_libraries(archneuronx PRIVATE CUDA::cudart CUDA::cublas CUDA::curand)
    target_compile_definitions(archneuronx PRIVATE USE_CUDA=1)
endif()

if(prometheus-cpp_FOUND)
    target_link_libraries(archneuronx PRIVATE prometheus-cpp::core prometheus-cpp::pull)
    target_compile_definitions(archneuronx PRIVATE USE_PROMETHEUS=1)
endif()

target_compile_features(archneuronx PRIVATE cxx_std_20)

# ============================================================
# Tests
# ============================================================
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    set(TEST_SOURCES
        tests/unit/test_technical_indicators.cpp
        tests/unit/test_risk_manager.cpp
        tests/unit/test_signal_generator.cpp
        tests/unit/test_data_preprocessor.cpp
        tests/unit/test_transformer_model.cpp
        tests/unit/test_portfolio_manager.cpp
        tests/integration/test_pipeline.cpp
        tests/integration/test_api_endpoints.cpp
        tests/performance/test_inference_latency.cpp
    )
    add_executable(archneuronx_tests ${TEST_SOURCES})
    target_include_directories(archneuronx_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${TORCH_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
    )
    target_link_libraries(archneuronx_tests PRIVATE
        GTest::gtest_main
        GTest::gmock
        ${TORCH_LIBRARIES}
        ${Boost_LIBRARIES}
        nlohmann_json::nlohmann_json
        spdlog::spdlog
        Threads::Threads
    )

    include(GoogleTest)
    gtest_discover_tests(archneuronx_tests)
endif()

# ============================================================
# Benchmarks
# ============================================================
if(BUILD_BENCHMARKS)
    find_package(benchmark QUIET)
    if(benchmark_FOUND)
        add_executable(archneuronx_bench
            benchmarks/bench_inference.cpp
            benchmarks/bench_indicators.cpp
            benchmarks/bench_pipeline.cpp
        )
        target_link_libraries(archneuronx_bench PRIVATE
            benchmark::benchmark
            ${TORCH_LIBRARIES}
            nlohmann_json::nlohmann_json
            spdlog::spdlog
            Threads::Threads
        )
    else()
        message(STATUS "Google Benchmark not found, skipping benchmarks")
    endif()
endif()

# ============================================================
# Install targets
# ============================================================
install(TARGETS archneuronx DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/archneuronx)
install(DIRECTORY config/ DESTINATION etc/archneuronx)

# ============================================================
# Package configuration
# ============================================================
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CPACK_GENERATOR "DEB;TGZ")
include(CPack)

message(STATUS "")
message(STATUS "ArchNeuronX v${PROJECT_VERSION} - C++20 | LibTorch 2.6 | CUDA 12.4")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA       : ${USE_CUDA} (${CUDAToolkit_VERSION})")
message(STATUS "  Tests      : ${BUILD_TESTS}")
message(STATUS "  Benchmarks : ${BUILD_BENCHMARKS}")
message(STATUS "")
