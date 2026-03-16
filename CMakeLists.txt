cmake_minimum_required(VERSION 3.20)
project(ArchNeuronX VERSION 2.0.0 LANGUAGES CXX)

# ============================================================
# C++20 Standard (upgrade from C++17)
# Enables: ranges, concepts, coroutines, std::span, std::format
# ============================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # GCC 12+ required for full C++20 support
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "12.0")
        message(WARNING "GCC 12+ recommended for full C++20 support. Found: ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0 -DDEBUG -fsanitize=address,undefined)
        add_link_options(-fsanitize=address,undefined)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang 14+ required for full C++20 support
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0")
        message(WARNING "Clang 14+ recommended for full C++20 support.")
    endif()
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

# ============================================================
# Find Required Packages
# LibTorch 2.6.0 + CUDA 12.4 (upgrade from 2.1.0 + CUDA 11.8)
# Download CPU: https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcpu.zip
# Download CUDA: https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu124.zip
# ============================================================
find_package(Torch 2.0 REQUIRED)
message(STATUS "Found LibTorch: ${Torch_VERSION}")

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

# CURL for HTTP requests
find_package(CURL REQUIRED)
message(STATUS "Found CURL: ${CURL_VERSION_STRING}")

# ============================================================
# nlohmann/json (modern, header-only JSON - replaces jsoncpp)
# Install: apt install nlohmann-json3-dev  OR  vcpkg install nlohmann-json
# ============================================================
find_package(nlohmann_json 3.11 QUIET)
if(nlohmann_json_FOUND)
    message(STATUS "Found nlohmann/json: ${nlohmann_json_VERSION}")
else()
    message(STATUS "nlohmann/json not found via CMake, trying pkg-config...")
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
        PATHS /usr/include /usr/local/include
    )
    if(NLOHMANN_JSON_INCLUDE_DIR)
        message(STATUS "Found nlohmann/json headers: ${NLOHMANN_JSON_INCLUDE_DIR}")
    else()
        # Fallback: use bundled header
        message(WARNING "nlohmann/json not found - will use bundled single-header version")
    endif()
endif()

# ============================================================
# cpp-httplib (header-only HTTP/WebSocket server - replaces cpprestsdk)
# Install: apt install libcpp-httplib-dev  OR copy httplib.h to include/
# ============================================================
find_path(HTTPLIB_INCLUDE_DIR httplib.h
    PATHS /usr/include /usr/local/include ${CMAKE_SOURCE_DIR}/include/third_party
)
if(HTTPLIB_INCLUDE_DIR)
    message(STATUS "Found cpp-httplib: ${HTTPLIB_INCLUDE_DIR}")
    add_definitions(-DUSE_HTTPLIB)
else()
    message(WARNING "cpp-httplib not found - HTTP server will use fallback implementation")
endif()

# WebSocketPP (header-only WebSocket)
find_path(WEBSOCKETPP_INCLUDE_DIR
    NAMES websocketpp/config/asio_no_tls.hpp
    PATHS /usr/include /usr/local/include
)
if(WEBSOCKETPP_INCLUDE_DIR)
    message(STATUS "Found WebSocketPP: ${WEBSOCKETPP_INCLUDE_DIR}")
    add_definitions(-DUSE_WEBSOCKETPP)
else()
    message(WARNING "WebSocketPP not found, WebSocket functionality will be disabled")
endif()

# Prometheus C++ client for metrics
find_package(prometheus-cpp QUIET)
if(prometheus-cpp_FOUND)
    message(STATUS "Found prometheus-cpp")
    add_definitions(-DUSE_PROMETHEUS)
else()
    message(STATUS "prometheus-cpp not found, Prometheus metrics disabled")
endif()

# spdlog for structured logging
find_package(spdlog QUIET)
if(spdlog_FOUND)
    message(STATUS "Found spdlog: ${spdlog_VERSION}")
    add_definitions(-DUSE_SPDLOG)
else()
    message(STATUS "spdlog not found, using basic logging")
endif()

# ============================================================
# Optional CUDA support (CUDA 12.4+ recommended)
# ============================================================
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Target architectures: Turing(75), Ampere(80,86), Ada(89), Hopper(90)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA support enabled: ${CUDAToolkit_VERSION}")
    if(CUDAToolkit_VERSION VERSION_LESS "12.0")
        message(WARNING "CUDA 12.0+ recommended (found ${CUDAToolkit_VERSION}). Tensor Core optimizations may be limited.")
    endif()
else()
    message(STATUS "CUDA not found, building CPU-only version")
endif()

# ============================================================
# Include Directories
# ============================================================
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/core
    ${CMAKE_SOURCE_DIR}/include/api
    ${CMAKE_SOURCE_DIR}/include/models
    ${CMAKE_SOURCE_DIR}/include/data
    ${CMAKE_SOURCE_DIR}/include/trading
    ${CMAKE_SOURCE_DIR}/include/risk
    ${CMAKE_SOURCE_DIR}/include/backtest
    ${CMAKE_SOURCE_DIR}/include/monitoring
    ${CMAKE_SOURCE_DIR}/include/utils
    ${CMAKE_SOURCE_DIR}/include/third_party
    ${TORCH_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
)

if(HTTPLIB_INCLUDE_DIR)
    include_directories(${HTTPLIB_INCLUDE_DIR})
endif()
if(WEBSOCKETPP_INCLUDE_DIR)
    include_directories(${WEBSOCKETPP_INCLUDE_DIR})
endif()
if(NLOHMANN_JSON_INCLUDE_DIR)
    include_directories(${NLOHMANN_JSON_INCLUDE_DIR})
endif()

# Preprocessor definitions
add_definitions(-DTORCH_VERSION_MAJOR=${Torch_VERSION_MAJOR})
add_definitions(-DTORCH_VERSION_MINOR=${Torch_VERSION_MINOR})
add_definitions(-DARCHNEURONX_VERSION="${PROJECT_VERSION}")

# ============================================================
# Source files (v2 adds: risk/, backtest/, monitoring/)
# ============================================================
file(GLOB_RECURSE CORE_SOURCES      "src/core/*.cpp")
file(GLOB_RECURSE API_SOURCES        "src/api/*.cpp")
file(GLOB_RECURSE MODEL_SOURCES      "src/models/*.cpp")
file(GLOB_RECURSE DATA_SOURCES       "src/data/*.cpp")
file(GLOB_RECURSE TRADING_SOURCES    "src/trading/*.cpp")
file(GLOB_RECURSE RISK_SOURCES       "src/risk/*.cpp")       # NEW v2
file(GLOB_RECURSE BACKTEST_SOURCES   "src/backtest/*.cpp")   # NEW v2
file(GLOB_RECURSE MONITORING_SOURCES "src/monitoring/*.cpp") # NEW v2
file(GLOB_RECURSE UTILS_SOURCES      "src/utils/*.cpp")
file(GLOB MAIN_SOURCE                "src/main.cpp")
file(GLOB_RECURSE ALL_HEADERS        "include/*.h" "include/*.hpp")

# ============================================================
# Static library (for reuse in tests and benchmarks)
# ============================================================
add_library(archneuronx_lib STATIC
    ${CORE_SOURCES}
    ${API_SOURCES}
    ${MODEL_SOURCES}
    ${DATA_SOURCES}
    ${TRADING_SOURCES}
    ${RISK_SOURCES}
    ${BACKTEST_SOURCES}
    ${MONITORING_SOURCES}
    ${UTILS_SOURCES}
)

target_link_libraries(archneuronx_lib PUBLIC
    ${TORCH_LIBRARIES}
    Threads::Threads
    OpenSSL::SSL
    OpenSSL::Crypto
    ${CURL_LIBRARIES}
)

if(nlohmann_json_FOUND)
    target_link_libraries(archneuronx_lib PUBLIC nlohmann_json::nlohmann_json)
endif()
if(CUDAToolkit_FOUND)
    target_link_libraries(archneuronx_lib PUBLIC
        CUDA::cudart CUDA::cublas CUDA::curand CUDA::cufft
    )
endif()
if(prometheus-cpp_FOUND)
    target_link_libraries(archneuronx_lib PUBLIC prometheus-cpp::pull)
endif()
if(spdlog_FOUND)
    target_link_libraries(archneuronx_lib PUBLIC spdlog::spdlog)
endif()

# ============================================================
# Main executable
# ============================================================
add_executable(archneuronx
    ${MAIN_SOURCE}
    ${ALL_HEADERS}
)
target_link_libraries(archneuronx PRIVATE archneuronx_lib)

set_target_properties(archneuronx PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Windows: copy torch DLLs
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET archneuronx POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS} $<TARGET_FILE_DIR:archneuronx>)
endif()

# ============================================================
# Testing with Google Test + Google Benchmark
# ============================================================
enable_testing()
find_package(GTest QUIET)
if(GTest_FOUND)
    message(STATUS "Google Test found, enabling unit tests")
    file(GLOB_RECURSE TEST_SOURCES "tests/unit/*.cpp")
    file(GLOB_RECURSE INTEGRATION_TEST_SOURCES "tests/integration/*.cpp")

    # Unit tests
    add_executable(run_unit_tests ${TEST_SOURCES})
    target_link_libraries(run_unit_tests PRIVATE archneuronx_lib GTest::gtest GTest::gtest_main)
    set_target_properties(run_unit_tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    add_test(NAME UnitTests COMMAND run_unit_tests --gtest_output=xml:unit_results.xml)

    # Integration tests
    if(INTEGRATION_TEST_SOURCES)
        add_executable(run_integration_tests ${INTEGRATION_TEST_SOURCES})
        target_link_libraries(run_integration_tests PRIVATE archneuronx_lib GTest::gtest GTest::gtest_main)
        set_target_properties(run_integration_tests PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
        add_test(NAME IntegrationTests COMMAND run_integration_tests --gtest_output=xml:integration_results.xml)
    endif()
else()
    message(STATUS "Google Test not found, unit tests disabled")
endif()

# Benchmarks
find_package(benchmark QUIET)
if(benchmark_FOUND)
    message(STATUS "Google Benchmark found, enabling performance benchmarks")
    file(GLOB_RECURSE BENCH_SOURCES "tests/benchmarks/*.cpp")
    if(BENCH_SOURCES)
        add_executable(run_benchmarks ${BENCH_SOURCES})
        target_link_libraries(run_benchmarks PRIVATE archneuronx_lib benchmark::benchmark benchmark::benchmark_main)
        set_target_properties(run_benchmarks PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks)
    endif()
endif()

# ============================================================
# Documentation (Doxygen)
# ============================================================
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_SOURCE_DIR}/docs/Doxyfile.in
                   ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(docs
        ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
endif()

# ============================================================
# Install targets
# ============================================================
install(TARGETS archneuronx DESTINATION bin)
install(TARGETS archneuronx_lib DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/archneuronx
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    PATTERN "third_party" EXCLUDE)
install(DIRECTORY config/ DESTINATION share/archneuronx/config)
install(FILES README.md LICENSE CHANGELOG.md DESTINATION share/doc/archneuronx)

# ============================================================
# Development helper targets
# ============================================================
add_custom_target(format
    COMMAND find ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/tests -name "*.cpp" -o -name "*.hpp" -o -name "*.h"
            | xargs clang-format -i --style=file
    COMMENT "Formatting source code with clang-format"
)

add_custom_target(lint
    COMMAND find ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include
            -name "*.cpp" -o -name "*.hpp"
            | xargs clang-tidy -p ${CMAKE_BINARY_DIR}
    COMMENT "Running static analysis with clang-tidy"
)

add_custom_target(memcheck
    COMMAND valgrind --leak-check=full --error-exitcode=1
            ${CMAKE_BINARY_DIR}/tests/run_unit_tests
    COMMENT "Running memory check with Valgrind"
    DEPENDS run_unit_tests
)

# ============================================================
# CPack
# ============================================================
set(CPACK_PACKAGE_NAME "ArchNeuronX")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Automated Neural Network Trading System - C++20 + LibTorch + CUDA")
set(CPACK_PACKAGE_VENDOR "George Pricop")
set(CPACK_PACKAGE_CONTACT "https://github.com/Gzeu/ArchNeuronX")
set(CPACK_GENERATOR "TGZ;ZIP;DEB")
include(CPack)

# ============================================================
# Configuration Summary
# ============================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "  ArchNeuronX v${PROJECT_VERSION} Configuration")
message(STATUS "========================================")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:         ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  LibTorch:         ${Torch_VERSION}")
message(STATUS "  CUDA support:     ${CUDAToolkit_FOUND} (${CUDAToolkit_VERSION})")
message(STATUS "  nlohmann/json:    ${nlohmann_json_FOUND}")
message(STATUS "  cpp-httplib:      ${HTTPLIB_INCLUDE_DIR}")
message(STATUS "  WebSocketPP:      ${WEBSOCKETPP_INCLUDE_DIR}")
message(STATUS "  prometheus-cpp:   ${prometheus-cpp_FOUND}")
message(STATUS "  spdlog:           ${spdlog_FOUND}")
message(STATUS "  Google Test:      ${GTest_FOUND}")
message(STATUS "  Google Benchmark: ${benchmark_FOUND}")
message(STATUS "  Doxygen:          ${DOXYGEN_FOUND}")
message(STATUS "========================================")
message(STATUS "")
