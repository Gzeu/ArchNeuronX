# ============================================================
# ArchNeuronX Source CMakeLists.txt
# ============================================================

# Create executable with HTTP server
add_executable(archneuronx
    main_http.cpp
)

# Set properties
set_target_properties(archneuronx PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
