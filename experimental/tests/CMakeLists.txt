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

# v1 tests with conversion - comprehensive v1-specific tests using common format
add_executable(experimental_v1_with_conversion_test
  test_main.cpp
  set_algebra/test_v1_with_conversion.cpp
)

target_link_libraries(experimental_v1_with_conversion_test
  PRIVATE
    experimental_subsetix_csr
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME ExperimentalV1WithConversionTest COMMAND experimental_v1_with_conversion_test)

# v2 tests with conversion - v2-specific tests using common format
add_executable(experimental_v2_with_conversion_test
  test_main.cpp
  set_algebra/test_v2_with_conversion.cpp
)

target_link_libraries(experimental_v2_with_conversion_test
  PRIVATE
    experimental_subsetix_csr
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME ExperimentalV2WithConversionTest COMMAND experimental_v2_with_conversion_test)

# v3 tests with conversion - v3-specific tests using common format
add_executable(experimental_v3_with_conversion_test
  test_main.cpp
  set_algebra/test_v3_with_conversion.cpp
)

target_link_libraries(experimental_v3_with_conversion_test
  PRIVATE
    experimental_subsetix_csr
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME ExperimentalV3WithConversionTest COMMAND experimental_v3_with_conversion_test)

# Random comparison test - compares v1/v2/v3 using random geometries ONLY
add_executable(experimental_random_comparison_test
  test_main.cpp
  set_algebra/test_random_comparison.cpp
)

target_link_libraries(experimental_random_comparison_test
  PRIVATE
    experimental_subsetix_csr
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME ExperimentalRandomComparisonTest COMMAND experimental_random_comparison_test)

message(STATUS "  - experimental_v1_with_conversion_test")
message(STATUS "  - experimental_v2_with_conversion_test")
message(STATUS "  - experimental_v3_with_conversion_test")
message(STATUS "  - experimental_random_comparison_test")
