# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

# Experimental algorithms tests
# Only compiled if SUBSETIX_ENABLE_EXPERIMENTAL=ON

message(STATUS "Experimental tests will be built")

# ============================================================================
# Special Purpose Tests (keep for specific validation)
# ============================================================================

# Sorted rows test - verifies that mesh generation produces sorted row_keys
add_executable(experimental_sorted_rows_test
  test_main.cpp
  set_algebra/test_sorted_rows.cpp
)

target_link_libraries(experimental_sorted_rows_test
  PRIVATE
    experimental_subsetix_csr
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME ExperimentalSortedRowsTest COMMAND experimental_sorted_rows_test)

# Overlap patterns test - test each overlap pattern with v1/v2
add_executable(experimental_overlap_patterns_test
  test_main.cpp
  set_algebra/test_overlap_patterns.cpp
)

target_link_libraries(experimental_overlap_patterns_test
  PRIVATE
    experimental_subsetix_csr
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME ExperimentalOverlapPatternsTest COMMAND experimental_overlap_patterns_test)

# Large mesh test - test intersection with large mesh sizes (same as benchmark ranges)
add_executable(experimental_large_mesh_test
  test_main.cpp
  set_algebra/test_large_mesh.cpp
)

target_link_libraries(experimental_large_mesh_test
  PRIVATE
    experimental_subsetix_csr
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME ExperimentalLargeMeshTest COMMAND experimental_large_mesh_test)

message(STATUS "  - experimental_sorted_rows_test")
message(STATUS "  - experimental_overlap_patterns_test")
message(STATUS "  - experimental_large_mesh_test")

# ============================================================================
# New Architecture Tests with Common Format and Random Generation
# ============================================================================

# baseline unitary tests - comprehensive baseline-specific tests
add_executable(experimental_baseline_unitary_test
  test_main.cpp
  set_algebra/test_v1_unitary.cpp
)

target_link_libraries(experimental_baseline_unitary_test
  PRIVATE
    experimental_subsetix_csr
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME ExperimentalBaselineUnitaryTest COMMAND experimental_baseline_unitary_test)

# optimized unitary tests - optimized-specific tests
add_executable(experimental_optimized_unitary_test
  test_main.cpp
  set_algebra/test_v2_unitary.cpp
)

target_link_libraries(experimental_optimized_unitary_test
  PRIVATE
    experimental_subsetix_csr
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME ExperimentalOptimizedUnitaryTest COMMAND experimental_optimized_unitary_test)

# Cross-version test - compares baseline/optimized using random geometries
add_executable(experimental_cross_version_test
  test_main.cpp
  set_algebra/test_cross_version.cpp
)

target_link_libraries(experimental_cross_version_test
  PRIVATE
    experimental_subsetix_csr
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME ExperimentalCrossVersionTest COMMAND experimental_cross_version_test)

# Random mesh generator test - validates sparsity-based unique sampling
add_executable(experimental_random_mesh_generator_test
  test_main.cpp
  set_algebra/test_random_mesh_generator.cpp
)

target_link_libraries(experimental_random_mesh_generator_test
  PRIVATE
    experimental_subsetix_csr
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME ExperimentalRandomMeshGeneratorTest COMMAND experimental_random_mesh_generator_test)

message(STATUS "  - experimental_baseline_unitary_test")
message(STATUS "  - experimental_optimized_unitary_test")
message(STATUS "  - experimental_cross_version_test")
message(STATUS "  - experimental_random_mesh_generator_test")
