cmake_minimum_required(VERSION 3.20)
project(ArchNeuronX VERSION 2.0.0 LANGUAGES CXX)

# ============================================================
# C++20 standard (upgrade from C++17)
# ============================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
option(USE_CUDA "Enable CUDA GPU acceleration" OFF)
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)
option(ENABLE_PROFILING "Enable CUDA profiling" OFF)
option(USE_TENSORRT "Enable TensorRT optimization" OFF)

# ============================================================
# Find required packages
# ============================================================
find_package(Threads REQUIRED)

# LibTorch for neural networks
find_package(Torch REQUIRED)

# Optional: CUDA for GPU acceleration
if(USE_CUDA)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # CUDA capabilities
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86) # Turing, Ampere, Ada
    
    # Add CUDA definitions
    add_definitions(-DUSE_CUDA)
endif()

# ============================================================
# Include directories
# ============================================================
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# ============================================================
# Add subdirectories
# ============================================================
add_subdirectory(src)

# Regime-aware ensemble example executable
add_executable(regime_aware_example
    examples/regime_aware_example.cpp
    src/regime/regime_detector.cpp
    src/models/regime_aware_ensemble.cpp
    src/models/ensemble.cpp
)

target_link_libraries(regime_aware_example 
    PRIVATE 
    ${TORCH_LIBRARIES}
    Threads::Threads
)

target_include_directories(regime_aware_example PRIVATE include)

# Main executable
add_executable(archneuronx
    src/trading/openclaw_agent.cpp
    src/trading/openclaw_integration.cpp
    src/trading/openclaw_integration.hpp
    src/openclaw/openclaw_core.cpp
    src/openclaw/openclaw_core.hpp
)

target_link_libraries(archneuronx 
    PRIVATE 
    openclaw_integration
    openclaw_core
    ${TORCH_LIBRARIES}
)

# OpenCLaw Core Library
add_library(openclaw_core
    src/openclaw/openclaw_core.cpp
    src/openclaw/openclaw_core.hpp
)

target_include_directories(openclaw_core PRIVATE include)
target_include_directories(openclaw_integration PRIVATE include)

# ============================================================
# Testing setup
# ============================================================
if(BUILD_TESTS)
    enable_testing()
    # add_subdirectory(tests)  # Commented out for now
endif()

# ============================================================
# Installation
# ============================================================
install(TARGETS archneuronx
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY config/
    DESTINATION etc/archneuronx/config
    FILES_MATCHING PATTERN "*.json"
)

# ============================================================
# CPack configuration
# ============================================================
set(CPACK_PACKAGE_NAME "ArchNeuronX")
set(CPACK_PACKAGE_VERSION "2.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Automated Neural Network Trading System")
set(CPACK_PACKAGE_VENDOR "ArchNeuronX Team")
set(CPACK_GENERATOR "TGZ;DEB")

include(CPack)
