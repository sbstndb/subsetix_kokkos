# CMakeLists.txt for mach2_validation tests
# Phase 0: Validation Infrastructure

# Add executable for type safety tests (Phase 0.5)
add_executable(mach2_type_safety_test
    type_safety_tests.cpp
)

# Link against required libraries
target_link_libraries(mach2_type_safety_test
    PRIVATE
        subsetix::core
        subsetix::csr_ops
        Kokkos::kokkos
)

# Include directories
target_include_directories(mach2_type_safety_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/examples/mach2_cylinder  # For mach2_config.hpp, mach2_utils.hpp
        ${CMAKE_SOURCE_DIR}/include                     # For FVD layer
)

# Compile options
target_compile_options(mach2_type_safety_test
    PRIVATE
        ${SUBSETIX_COPTIES}
)

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

# Add to CTest (if enabled)
include(CTest)
if(BUILD_TESTING)
    add_test(NAME Mach2TypeSafety COMMAND mach2_type_safety_test)
    # Note: mach2_validation is a manual test framework, not automated yet
endif()
