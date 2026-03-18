# Testing configuration
find_package(GTest REQUIRED)
find_package(GMock REQUIRED)

# Test sources
set(TEST_SOURCES
    test_complete_system.cpp
    test_quantum_neural_networks.cpp
    test_quantum_agents.cpp
    test_llm_integration.cpp
)

# Create test executable
add_executable(archneuronx_tests ${TEST_SOURCES})

# Link libraries
target_link_libraries(archneuronx_tests
    ${TORCH_LIBRARIES}
    GTest::gtest_main
    GMock::gmock_main
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

add_custom_target(test_quantum
    COMMAND $<TARGET_FILE:archneuronx_tests> --gtest_filter="*Quantum*"
    DEPENDS archneuronx_tests
)

add_custom_target(test_llm
    COMMAND $<TARGET_FILE:archneuronx_tests> --gtest_filter="*LLM*"
    DEPENDS archneuronx_tests
)

add_custom_target(test_agents
    COMMAND $<TARGET_FILE:archneuronx_tests> --gtest_filter="*Agent*"
    DEPENDS archneuronx_tests
)

add_custom_target(test_performance
    COMMAND $<TARGET_FILE:archneuronx_tests> --gtest_filter="*Performance*"
    DEPENDS archneuronx_tests
)

add_custom_target(test_all
    COMMAND $<TARGET_FILE:archneuronx_tests>
    DEPENDS archneuronx_tests
)

# Add coverage support
option(ENABLE_COVERAGE "Enable code coverage" OFF)
if(ENABLE_COVERAGE)
    target_compile_options(archneuronx_tests PRIVATE --coverage)
    target_link_options(archneuronx_tests PRIVATE --coverage)
endif()