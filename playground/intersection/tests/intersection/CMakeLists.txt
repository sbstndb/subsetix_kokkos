# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

# Playground intersection algorithm tests
# Only compiled if SUBSETIX_ENABLE_PLAYGROUND=ON

message(STATUS "Playground intersection tests will be built")

# ============================================================================
# Special Purpose Tests (keep for specific validation)
# ============================================================================

# Sorted rows test - verifies that mesh generation produces sorted row_keys
add_executable(playground_intersection_sorted_rows_test
  test_main.cpp
  test_sorted_rows.cpp
)

target_link_libraries(playground_intersection_sorted_rows_test
  PRIVATE
    playground_subsetix_csr_intersection
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME PlaygroundIntersectionSortedRowsTest COMMAND playground_intersection_sorted_rows_test)

# Overlap patterns test - test each overlap pattern with v1/v2
add_executable(playground_intersection_overlap_patterns_test
  test_main.cpp
  test_overlap_patterns.cpp
)

target_link_libraries(playground_intersection_overlap_patterns_test
  PRIVATE
    playground_subsetix_csr_intersection
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME PlaygroundIntersectionOverlapPatternsTest COMMAND playground_intersection_overlap_patterns_test)

# Large mesh test - test intersection with large mesh sizes (same as benchmark ranges)
add_executable(playground_intersection_large_mesh_test
  test_main.cpp
  test_large_mesh.cpp
)

target_link_libraries(playground_intersection_large_mesh_test
  PRIVATE
    playground_subsetix_csr_intersection
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME PlaygroundIntersectionLargeMeshTest COMMAND playground_intersection_large_mesh_test)

message(STATUS "  - playground_intersection_sorted_rows_test")
message(STATUS "  - playground_intersection_overlap_patterns_test")
message(STATUS "  - playground_intersection_large_mesh_test")

# ============================================================================
# New Architecture Tests with Common Format and Random Generation
# ============================================================================

# v1 unitary tests - comprehensive v1-specific tests
add_executable(playground_intersection_v1_unitary_test
  test_main.cpp
  test_v1_unitary.cpp
)

target_link_libraries(playground_intersection_v1_unitary_test
  PRIVATE
    playground_subsetix_csr_intersection
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME PlaygroundIntersectionV1UnitaryTest COMMAND playground_intersection_v1_unitary_test)

# v2 unitary tests - v2-specific tests
add_executable(playground_intersection_v2_unitary_test
  test_main.cpp
  test_v2_unitary.cpp
)

target_link_libraries(playground_intersection_v2_unitary_test
  PRIVATE
    playground_subsetix_csr_intersection
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME PlaygroundIntersectionV2UnitaryTest COMMAND playground_intersection_v2_unitary_test)

# v3 unitary tests - v3-specific tests
add_executable(playground_intersection_v3_unitary_test
  test_main.cpp
  test_v3_unitary.cpp
)

target_link_libraries(playground_intersection_v3_unitary_test
  PRIVATE
    playground_subsetix_csr_intersection
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME PlaygroundIntersectionV3UnitaryTest COMMAND playground_intersection_v3_unitary_test)

# Cross-version test - compares v1/v2/v3 using random geometries
add_executable(playground_intersection_cross_version_test
  test_main.cpp
  test_cross_version.cpp
)

target_link_libraries(playground_intersection_cross_version_test
  PRIVATE
    playground_subsetix_csr_intersection
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME PlaygroundIntersectionCrossVersionTest COMMAND playground_intersection_cross_version_test)

# Random mesh generator test - validates sparsity-based unique sampling
add_executable(playground_intersection_random_mesh_generator_test
  test_main.cpp
  test_random_mesh_generator.cpp
)

target_link_libraries(playground_intersection_random_mesh_generator_test
  PRIVATE
    playground_subsetix_csr_intersection
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME PlaygroundIntersectionRandomMeshGeneratorTest COMMAND playground_intersection_random_mesh_generator_test)

message(STATUS "  - playground_intersection_v1_unitary_test")
message(STATUS "  - playground_intersection_v2_unitary_test")
message(STATUS "  - playground_intersection_v3_unitary_test")
message(STATUS "  - playground_intersection_cross_version_test")
message(STATUS "  - playground_intersection_random_mesh_generator_test")
