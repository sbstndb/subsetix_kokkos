# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

# Playground algorithms module
# Standalone module for playground set algebra algorithms
# Only built if SUBSETIX_ENABLE_PLAYGROUND=ON

if(NOT SUBSETIX_ENABLE_PLAYGROUND)
  return()
endif()

message(STATUS "=== Configuring Playground Module ===")

# Create alias library for playground intersection headers (interface library)
add_library(playground_subsetix_intersection INTERFACE)

target_link_libraries(playground_subsetix_intersection
  INTERFACE
    Kokkos::kokkos
)

target_include_directories(playground_subsetix_intersection
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/intersection/include
)

# Set alias for convenience
add_library(playground::intersection ALIAS playground_subsetix_intersection)

message(STATUS "  - playground::intersection library configured")

# Create alias library for playground fields headers (interface library)
add_library(playground_subsetix_fields INTERFACE)

target_link_libraries(playground_subsetix_fields
  INTERFACE
    Kokkos::kokkos
    playground::intersection
)

target_include_directories(playground_subsetix_fields
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/fields/include
)

# Set alias for convenience
add_library(playground::fields ALIAS playground_subsetix_fields)

message(STATUS "  - playground::fields library configured")

# Tests
if(BUILD_TESTING)
  add_subdirectory(intersection/tests)
  add_subdirectory(fields/tests)
endif()

# Benchmarks
option(SUBSETIX_BUILD_BENCHMARKS "Build playground benchmarks" ON)
if(SUBSETIX_BUILD_BENCHMARKS)
  add_subdirectory(intersection/benchmarks)
  add_subdirectory(fields/benchmarks)
endif()

message(STATUS "=== Playground Module Configured ===")
