cmake_minimum_required(VERSION 3.18)
project(ArchNeuronX VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3)
endif()

# Find required packages
find_package(Torch REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# Optional CUDA support
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA support enabled")
else()
    message(STATUS "CUDA not found, building CPU-only version")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

# Create main executable
add_executable(archneuronx ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(archneuronx 
    ${TORCH_LIBRARIES}
    Threads::Threads
)

# Set properties
set_property(TARGET archneuronx PROPERTY CXX_STANDARD 17)

# Copy torch dlls if on windows
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET archneuronx
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:archneuronx>)
endif (MSVC)

# Testing
enable_testing()
add_subdirectory(tests)

# Install targets
install(TARGETS archneuronx DESTINATION bin)
install(DIRECTORY config/ DESTINATION share/archneuronx/config)
install(FILES README.md LICENSE DESTINATION share/doc/archneuronx)