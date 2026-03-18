cmake_minimum_required(VERSION 3.20)
project(ArchNeuronX VERSION 4.0.0 LANGUAGES CXX CUDA)

# ============================================================
# C++20 standard with v4.0 optimizations
# ============================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================
# v4.0 Compiler optimizations
# ============================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native -ffast-math)
    # v4.0 specific optimizations
    add_compile_options(-mavx2 -mfma -mbmi2)  # Modern CPU features
    add_compile_options(-flto -fno-fat-lto-objects)  # Link-time optimization
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0 -DDEBUG -fsanitize=address)
        add_link_options(-fsanitize=address)
    endif()
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        # v4.0 aggressive optimizations
        add_compile_options(-DNDEBUG -funroll-loops -fprefetch-loop-arrays)
    endif()
endif()

# ============================================================
# v4.0 Options
# ============================================================
option(USE_CUDA "Enable CUDA 12.4 GPU acceleration" ON)
option(USE_LIBTORCH_26 "Enable LibTorch 2.6" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(ENABLE_PROFILING "Enable CUDA profiling" ON)
option(USE_TENSORRT "Enable TensorRT optimization" ON)
option(BUILD_V4_QUANTUM "Build v4.0 quantum networks" ON)
option(ENABLE_ULTRA_LOW_LATENCY "Enable <20μs latency optimizations" ON)

# ============================================================
# v4.0 Package finding
# ============================================================
find_package(Threads REQUIRED)

# LibTorch 2.6 for v4.0 neural networks
if(USE_LIBTORCH_26)
    # Prefer LibTorch 2.6
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch-2.6")
    find_package(Torch REQUIRED)
    message(STATUS "Found LibTorch ${TORCH_VERSION}")
else()
    find_package(Torch REQUIRED)
endif()

# CUDA 12.4 for v4.0 GPU acceleration
if(USE_CUDA)
    # Check for CUDA availability first
    find_package(CUDA QUIET)
    if(NOT CUDA_FOUND)
        message(WARNING "CUDA not found. GPU acceleration will be disabled.")
        set(USE_CUDA OFF CACHE BOOL "Disable CUDA" FORCE)
    else()
        enable_language(CUDA)
        message(STATUS "Found CUDA ${CUDA_VERSION}")
        
        # v4.0 CUDA optimizations
        set(CMAKE_CUDA_ARCHITECTURES 80;86;89;90)  # Ampere, Hopper
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        
        # CUDA compiler options
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-O3 --expt-relaxed-constexpr")
    endif()
endif()

# TensorRT for v4.0 deployment
if(USE_TENSORRT)
    find_package(TensorRT REQUIRED)
    message(STATUS "Found TensorRT ${TensorRT_VERSION}")
endif()

# Additional CUDA setup for v4.0
if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # CUDA capabilities
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86) # Turing, Ampere, Ada
    
    # Add CUDA definitions
    add_definitions(-DUSE_CUDA)
endif()

# ============================================================
# v4.0 Include directories
# ============================================================
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# ============================================================
# v4.0 Add subdirectories
# ============================================================
add_subdirectory(src)

# ============================================================
# v4.0 Quantum Network Library
# ============================================================
if(BUILD_V4_QUANTUM)
    add_library(v4_quantum_network
        src/models/v4_quantum_neural_network.cpp
        include/models/v4_quantum_neural_network.hpp
    )
    
    target_link_libraries(v4_quantum_network
        PRIVATE
        ${TORCH_LIBRARIES}
        Threads::Threads
    )
    
    if(USE_CUDA)
        target_link_libraries(v4_quantum_network PRIVATE CUDA::cudart CUDA::cublas)
    endif()
    
    if(USE_TENSORRT)
        target_link_libraries(v4_quantum_network PRIVATE nvinfer::nvinfer)
    endif()
    
    target_include_directories(v4_quantum_network PRIVATE include)
    target_compile_features(v4_quantum_network PRIVATE cxx_std_20)
    
    # v4.0 optimizations
    if(ENABLE_ULTRA_LOW_LATENCY)
        target_compile_definitions(v4_quantum_network PRIVATE ULTRA_LOW_LATENCY)
    endif()
endif()

# ============================================================
# v4.0 Ultra Low Latency Engine
# ============================================================
if(ENABLE_ULTRA_LOW_LATENCY)
    add_library(v4_ultra_low_latency_engine
        src/core/v4_ultra_low_latency_engine.cpp
        include/core/v4_ultra_low_latency_engine.hpp
    )
    
    target_link_libraries(v4_ultra_low_latency_engine
        PRIVATE
        v4_quantum_network
        ${TORCH_LIBRARIES}
        Threads::Threads
    )
    
    if(USE_CUDA)
        target_link_libraries(v4_ultra_low_latency_engine PRIVATE CUDA::cudart CUDA::cublas)
    endif()
    
    target_include_directories(v4_ultra_low_latency_engine PRIVATE include)
    target_compile_features(v4_ultra_low_latency_engine PRIVATE cxx_std_20)
endif()

# ============================================================
# v4.0 Main executable with quantum integration
# ============================================================
add_executable(archneuronx_v4
    src/main_v4.cpp
    src/api/v4_rest_server.cpp
    src/api/v4_rest_server.hpp
)

target_link_libraries(archneuronx_v4
    PRIVATE
    v4_ultra_low_latency_engine
    v4_quantum_network
    ${TORCH_LIBRARIES}
    Threads::Threads
)

if(USE_CUDA)
    target_link_libraries(archneuronx_v4 PRIVATE CUDA::cudart CUDA::cublas)
endif()

if(USE_TENSORRT)
    target_link_libraries(archneuronx_v4 PRIVATE nvinfer::nvinfer)
endif()

target_include_directories(archneuronx_v4 PRIVATE include)
target_compile_features(archneuronx_v4 PRIVATE cxx_std_20)

# ============================================================
# v4.0 Examples and benchmarks
# ============================================================
if(BUILD_BENCHMARKS)
    add_executable(v4_quantum_example
        examples/v4_quantum_network_example.cpp
    )
    
    target_link_libraries(v4_quantum_example
        PRIVATE
        v4_ultra_low_latency_engine
        v4_quantum_network
        ${TORCH_LIBRARIES}
        Threads::Threads
    )
    
    target_include_directories(v4_quantum_example PRIVATE include)
endif()

# ============================================================
# Legacy v3.0 support (optional)
# ============================================================
add_executable(archneuronx_legacy
    src/main_http.cpp
)

target_link_libraries(archneuronx_legacy 
    PRIVATE 
    ${TORCH_LIBRARIES}
    Threads::Threads
)

target_include_directories(archneuronx_legacy PRIVATE include)

# ============================================================
# v4.0 Testing setup
# ============================================================
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
    
    # v4.0 performance tests
    add_executable(v4_performance_test
        tests/v4_performance_test.cpp
        tests/v4_latency_test.cpp
        tests/v4_throughput_test.cpp
    )
    
    target_link_libraries(v4_performance_test
        PRIVATE
        v4_ultra_low_latency_engine
        v4_quantum_network
        ${TORCH_LIBRARIES}
        Threads::Threads
    )
    
    target_include_directories(v4_performance_test PRIVATE include)
endif()

# ============================================================
# v4.0 Installation
# ============================================================
install(TARGETS archneuronx_v4 archneuronx_legacy
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

if(BUILD_V4_QUANTUM)
    install(TARGETS v4_quantum_network v4_ultra_low_latency_engine
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif()

install(DIRECTORY config/
    DESTINATION etc/archneuronx/config
    FILES_MATCHING PATTERN "*.json"
)

install(DIRECTORY models/
    DESTINATION var/lib/archneuronx/models
)

# ============================================================
# v4.0 CPack configuration
# ============================================================
set(CPACK_PACKAGE_NAME "ArchNeuronX")
set(CPACK_PACKAGE_VERSION "4.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "ArchNeuronX v4.0 - Market-Dominating Execution Engine")
set(CPACK_PACKAGE_VENDOR "ArchNeuronX Team")
set(CPACK_PACKAGE_CONTACT "dev@archneuronx.com")
set(CPACK_GENERATOR "TGZ;DEB;RPM")

# v4.0 dependencies
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libtorch-dev (>= 2.6), cuda-runtime-12-4, libcudnn8")
set(CPACK_DEBIAN_PACKAGE_SECTION "science")

include(CPack)
