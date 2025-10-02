cmake_minimum_required(VERSION 3.18)
project(ArchNeuronX VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0 -DDEBUG)
    endif()
endif()

# Find required packages
find_package(Torch REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

# HTTP client/server libraries
find_package(cpprestsdk QUIET)
if(NOT cpprestsdk_FOUND)
    message(STATUS "cpprestsdk not found, using alternative HTTP implementation")
endif()

# JSON processing
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found, will use bundled version")
endif()

# Optional CUDA support
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "CUDA support enabled")
else()
    message(STATUS "CUDA not found, building CPU-only version")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/core
    ${CMAKE_SOURCE_DIR}/include/api
    ${CMAKE_SOURCE_DIR}/include/models
    ${CMAKE_SOURCE_DIR}/include/data
    ${CMAKE_SOURCE_DIR}/include/trading
    ${CMAKE_SOURCE_DIR}/include/utils
    ${TORCH_INCLUDE_DIRS}
)

# Preprocessor definitions
add_definitions(-DTORCH_VERSION_MAJOR=${Torch_VERSION_MAJOR})
add_definitions(-DTORCH_VERSION_MINOR=${Torch_VERSION_MINOR})

# Source files organization
file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp")
file(GLOB_RECURSE API_SOURCES "src/api/*.cpp")
file(GLOB_RECURSE MODEL_SOURCES "src/models/*.cpp")
file(GLOB_RECURSE DATA_SOURCES "src/data/*.cpp")
file(GLOB_RECURSE TRADING_SOURCES "src/trading/*.cpp")
file(GLOB_RECURSE UTILS_SOURCES "src/utils/*.cpp")
file(GLOB MAIN_SOURCE "src/main.cpp")

file(GLOB_RECURSE ALL_HEADERS "include/*.h" "include/*.hpp")

# Create main executable
add_executable(archneuronx 
    ${MAIN_SOURCE}
    ${CORE_SOURCES}
    ${API_SOURCES}
    ${MODEL_SOURCES}
    ${DATA_SOURCES}
    ${TRADING_SOURCES}
    ${UTILS_SOURCES}
    ${ALL_HEADERS}
)

# Link libraries
target_link_libraries(archneuronx 
    ${TORCH_LIBRARIES}
    Threads::Threads
    OpenSSL::SSL
    OpenSSL::Crypto
)

# Add optional libraries
if(cpprestsdk_FOUND)
    target_link_libraries(archneuronx cpprestsdk::cpprest)
    add_definitions(-DUSE_CPPRESTSDK)
endif()

if(nlohmann_json_FOUND)
    target_link_libraries(archneuronx nlohmann_json::nlohmann_json)
    add_definitions(-DUSE_NLOHMANN_JSON)
endif()

# CUDA linking
if(CUDA_FOUND)
    target_link_libraries(archneuronx ${CUDA_LIBRARIES})
endif()

# Set properties
set_property(TARGET archneuronx PROPERTY CXX_STANDARD 17)
set_target_properties(archneuronx PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Copy torch dlls if on windows
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET archneuronx
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:archneuronx>)
endif(MSVC)

# Create separate library for reusable components
add_library(archneuronx_lib STATIC
    ${CORE_SOURCES}
    ${API_SOURCES}
    ${MODEL_SOURCES}
    ${DATA_SOURCES}
    ${TRADING_SOURCES}
    ${UTILS_SOURCES}
)

target_link_libraries(archneuronx_lib 
    ${TORCH_LIBRARIES}
    Threads::Threads
    OpenSSL::SSL
    OpenSSL::Crypto
)

# Testing
enable_testing()
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
    add_subdirectory(tests)
endif()

# Documentation
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_SOURCE_DIR}/docs/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(docs
        ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# Install targets
install(TARGETS archneuronx DESTINATION bin)
install(TARGETS archneuronx_lib DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/archneuronx FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
install(DIRECTORY config/ DESTINATION share/archneuronx/config)
install(FILES README.md LICENSE DESTINATION share/doc/archneuronx)

# CPack configuration
set(CPACK_PACKAGE_NAME "ArchNeuronX")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Automated Neural Network Trading System")
set(CPACK_PACKAGE_VENDOR "George Pricop")
set(CPACK_GENERATOR "TGZ;ZIP")
include(CPack)