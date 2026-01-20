# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

# Experimental algorithms tests
# Only compiled if SUBSETIX_ENABLE_EXPERIMENTAL=ON

message(STATUS "Experimental tests will be built")

# v1 tests
add_executable(experimental_v1_test
  set_algebra/v1_test.cpp
)

target_link_libraries(experimental_v1_test
  PRIVATE
    subsetix::core
    experimental_subsetix_csr
    gtest_main
    gmock_main
)

add_test(NAME ExperimentalV1Test COMMAND experimental_v1_test)

# v2 tests
add_executable(experimental_v2_test
  set_algebra/v2_test.cpp
)

target_link_libraries(experimental_v2_test
  PRIVATE
    subsetix::core
    experimental_subsetix_csr
    gtest_main
    gmock_main
)

add_test(NAME ExperimentalV2Test COMMAND experimental_v2_test)

message(STATUS "  - experimental_v1_test")
message(STATUS "  - experimental_v2_test")
