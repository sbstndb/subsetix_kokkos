# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

# Experimental algorithms tests
# Only compiled if SUBSETIX_ENABLE_EXPERIMENTAL=ON

message(STATUS "Experimental tests will be built")

# Create test executable
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

# Register with CTest
add_test(NAME ExperimentalV1Test COMMAND experimental_v1_test)

message(STATUS "  - experimental_v1_test")
