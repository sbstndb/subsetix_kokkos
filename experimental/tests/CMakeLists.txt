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

# v1 unitary tests - comprehensive v1-specific tests
add_executable(experimental_v1_unitary_test
  test_main.cpp
  set_algebra/test_v1_unitary.cpp
)

target_link_libraries(experimental_v1_unitary_test
  PRIVATE
    experimental_subsetix_csr
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME ExperimentalV1UnitaryTest COMMAND experimental_v1_unitary_test)

# v2 unitary tests - v2-specific tests
add_executable(experimental_v2_unitary_test
  test_main.cpp
  set_algebra/test_v2_unitary.cpp
)

target_link_libraries(experimental_v2_unitary_test
  PRIVATE
    experimental_subsetix_csr
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME ExperimentalV2UnitaryTest COMMAND experimental_v2_unitary_test)

# v3 unitary tests - v3-specific tests
add_executable(experimental_v3_unitary_test
  test_main.cpp
  set_algebra/test_v3_unitary.cpp
)

target_link_libraries(experimental_v3_unitary_test
  PRIVATE
    experimental_subsetix_csr
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME ExperimentalV3UnitaryTest COMMAND experimental_v3_unitary_test)

# Cross-version test - compares v1/v2/v3 using random geometries
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

message(STATUS "  - experimental_v1_unitary_test")
message(STATUS "  - experimental_v2_unitary_test")
message(STATUS "  - experimental_v3_unitary_test")
message(STATUS "  - experimental_cross_version_test")
message(STATUS "  - experimental_random_mesh_generator_test")
