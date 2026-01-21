# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

# Experimental algorithms tests
# Only compiled if SUBSETIX_ENABLE_EXPERIMENTAL=ON

message(STATUS "Experimental tests will be built")

# v1 tests - use gtest_main (no actual Views created in tests, so Kokkos init not needed)
add_executable(experimental_v1_test
  set_algebra/v1_test.cpp
)

target_link_libraries(experimental_v1_test
  PRIVATE
    experimental_subsetix_csr
    gtest_main
    gmock_main
)

add_test(NAME ExperimentalV1Test COMMAND experimental_v1_test)

# v2 tests - use custom test_main that initializes Kokkos
add_executable(experimental_v2_test
  test_main.cpp
  set_algebra/v2_test.cpp
)

target_link_libraries(experimental_v2_test
  PRIVATE
    experimental_subsetix_csr
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME ExperimentalV2Test COMMAND experimental_v2_test)

# v3 tests - use custom test_main that initializes Kokkos
add_executable(experimental_v3_test
  test_main.cpp
  set_algebra/v3_test.cpp
)

target_link_libraries(experimental_v3_test
  PRIVATE
    experimental_subsetix_csr
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME ExperimentalV3Test COMMAND experimental_v3_test)

message(STATUS "  - experimental_v1_test")
message(STATUS "  - experimental_v2_test")
message(STATUS "  - experimental_v3_test")
message(STATUS "  - experimental_benchmark_smoke_test")

# Benchmark smoke test - quick check that benchmark code doesn't segfault
# Uses test_main.cpp to initialize Kokkos (same as v2/v3 tests)
add_executable(experimental_benchmark_smoke_test
  test_main.cpp
  set_algebra/benchmark_smoke_test.cpp
)

target_link_libraries(experimental_benchmark_smoke_test
  PRIVATE
    experimental_subsetix_csr
    gtest
    gmock
    Kokkos::kokkos
)

add_test(NAME ExperimentalBenchmarkSmokeTest COMMAND experimental_benchmark_smoke_test)

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
