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
