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
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
if(NOT JSONCPP_FOUND)
    find_package(jsoncpp QUIET)
    if(jsoncpp_FOUND)
        set(JSONCPP_LIBRARIES jsoncpp_lib)
    else()
        message(STATUS "jsoncpp not found, will use bundled version")
    endif()
endif()

# CURL for HTTP requests
find_package(CURL REQUIRED)
if(CURL_FOUND)
    message(STATUS "Found CURL: ${CURL_LIBRARIES}")
else()
    message(FATAL_ERROR "CURL library is required for data providers")
endif()

# WebSocketPP (header-only library)
find_path(WEBSOCKETPP_INCLUDE_DIR
    NAMES websocketpp/config/asio_no_tls.hpp
    PATHS /usr/include /usr/local/include
)
if(WEBSOCKETPP_INCLUDE_DIR)
    message(STATUS "Found WebSocketPP: ${WEBSOCKETPP_INCLUDE_DIR}")
else()
    message(WARNING "WebSocketPP not found, WebSocket functionality will be disabled")
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
    ${CURL_INCLUDE_DIRS}
)

# Add WebSocketPP include directory if found
if(WEBSOCKETPP_INCLUDE_DIR)
    include_directories(${WEBSOCKETPP_INCLUDE_DIR})
    add_definitions(-DUSE_WEBSOCKETPP)
endif()

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
    ${CURL_LIBRARIES}
)

# Add optional libraries
if(cpprestsdk_FOUND)
    target_link_libraries(archneuronx cpprestsdk::cpprest)
    add_definitions(-DUSE_CPPRESTSDK)
endif()

if(JSONCPP_FOUND)
    target_link_libraries(archneuronx ${JSONCPP_LIBRARIES})
    target_include_directories(archneuronx PRIVATE ${JSONCPP_INCLUDE_DIRS})
    add_definitions(-DUSE_JSONCPP)
elseif(jsoncpp_FOUND)
    target_link_libraries(archneuronx ${JSONCPP_LIBRARIES})
    add_definitions(-DUSE_JSONCPP)
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
    ${CURL_LIBRARIES}
)

# Add optional libraries to static lib
if(JSONCPP_FOUND)
    target_link_libraries(archneuronx_lib ${JSONCPP_LIBRARIES})
elseif(jsoncpp_FOUND)
    target_link_libraries(archneuronx_lib ${JSONCPP_LIBRARIES})
endif()

# Testing with Google Test
enable_testing()
find_package(GTest QUIET)
if(GTest_FOUND AND EXISTS ${CMAKE_SOURCE_DIR}/tests/)
    message(STATUS "Google Test found, enabling unit tests")
    
    # Test executables
    file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
    
    add_executable(run_tests ${TEST_SOURCES})
    target_link_libraries(run_tests 
        archneuronx_lib
        GTest::gtest 
        GTest::gtest_main
    )
    
    # Add tests
    add_test(NAME CoreInfrastructureTests COMMAND run_tests)
    
    # Test properties
    set_target_properties(run_tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    )
else()
    message(STATUS "Google Test not found, unit tests disabled")
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

# Development tools
add_custom_target(format
    COMMAND find ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i
    COMMENT "Formatting source code"
)

add_custom_target(lint
    COMMAND find ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include -name "*.cpp" -o -name "*.hpp" | xargs clang-tidy
    COMMENT "Running static analysis"
)

# CPack configuration
set(CPACK_PACKAGE_NAME "ArchNeuronX")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Automated Neural Network Trading System")
set(CPACK_PACKAGE_VENDOR "George Pricop")
set(CPACK_GENERATOR "TGZ;ZIP")
include(CPack)

# Print configuration summary
message(STATUS "")
message(STATUS "ArchNeuronX Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Torch version: ${Torch_VERSION}")
message(STATUS "  CUDA support: ${CUDA_FOUND}")
message(STATUS "  cpprestsdk: ${cpprestsdk_FOUND}")
message(STATUS "  jsoncpp: ${JSONCPP_FOUND}")
message(STATUS "  CURL: ${CURL_FOUND}")
message(STATUS "  WebSocketPP: ${WEBSOCKETPP_INCLUDE_DIR}")
message(STATUS "  Google Test: ${GTest_FOUND}")
message(STATUS "  Doxygen: ${DOXYGEN_FOUND}")
message(STATUS "")