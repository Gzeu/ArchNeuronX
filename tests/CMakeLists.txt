# Testing configuration
find_package(GTest REQUIRED)

# Test sources
file(GLOB_RECURSE TEST_SOURCES "*.cpp")

# Create test executable
add_executable(archneuronx_tests ${TEST_SOURCES})

# Link libraries
target_link_libraries(archneuronx_tests
    ${TORCH_LIBRARIES}
    GTest::gtest_main
    Threads::Threads
)

# Include directories
target_include_directories(archneuronx_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Register tests
include(GoogleTest)
gtest_discover_tests(archneuronx_tests)

# Add custom test targets
add_custom_target(test_unit
    COMMAND $<TARGET_FILE:archneuronx_tests> --gtest_filter="Unit*"
    DEPENDS archneuronx_tests
)

add_custom_target(test_integration
    COMMAND $<TARGET_FILE:archneuronx_tests> --gtest_filter="Integration*"
    DEPENDS archneuronx_tests
)