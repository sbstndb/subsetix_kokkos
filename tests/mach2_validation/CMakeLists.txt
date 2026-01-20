# CMakeLists.txt for mach2_validation tests
# Phase 0: Validation Infrastructure

# Add executable for validation tests
add_executable(mach2_validation
    validation_main.cpp
)

# Link against required libraries
target_link_libraries(mach2_validation
    PRIVATE
        subsetix::core
        subsetix::csr_ops
        subsetix::fvd  # Only if FVD is enabled
        Kokkos::kokkos
)

# Include directories
target_include_directories(mach2_validation
    PRIVATE
        ${CMAKE_SOURCE_DIR}/examples/mach2_cylinder  # For mach2_config.hpp, mach2_utils.hpp
)

# Compile options
target_compile_options(mach2_validation
    PRIVATE
        ${SUBSETIX_COPTIES}
)

# Test discovery (if using CTest)
include(CTest)
include(GoogleTest)

# Note: For now, this is a manual test executable
# TODO: Convert to proper unit tests in Phase 0.5
