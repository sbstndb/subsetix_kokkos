# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

# Playground intersection tests

if(NOT BUILD_TESTING)
  return()
endif()

message(STATUS "=== Configuring Playground Intersection Tests ===")

# Create test executable
add_executable(playground_intersection_tests
  intersection/test_main.cpp
  intersection/test_baseline_unitary.cpp
  intersection/test_optimized_unitary.cpp
  intersection/test_cross_version.cpp
  intersection/test_overlap_patterns.cpp
  intersection/test_sorted_rows.cpp
  intersection/test_large_mesh.cpp
  intersection/test_random_mesh_generator.cpp
  intersection/test_workspace_validation.cpp
)

target_link_libraries(playground_intersection_tests
  PRIVATE
    playground::intersection
    Kokkos::kokkos
    gtest
    gmock
    gmock_main
)

target_include_directories(playground_intersection_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Register with CTest
add_test(NAME playground_intersection_tests COMMAND playground_intersection_tests)

# Set test properties
set_tests_properties(playground_intersection_tests PROPERTIES
  LABELS "playground;intersection"
)

message(STATUS "  - playground_intersection_tests configured")

message(STATUS "=== Playground Intersection Tests Configured ===")
