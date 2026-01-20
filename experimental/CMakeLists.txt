# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

# Experimental algorithms module
# Standalone module for experimental set algebra algorithms
# Only built if SUBSETIX_ENABLE_EXPERIMENTAL=ON

if(NOT SUBSETIX_ENABLE_EXPERIMENTAL)
  return()
endif()

message(STATUS "=== Configuring Experimental Module ===")

# Create alias library for experimental headers (interface library)
add_library(experimental_subsetix_csr INTERFACE)

target_include_directories(experimental_subsetix_csr
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set alias for convenience
add_library(experimental::csr ALIAS experimental_subsetix_csr)

message(STATUS "  - experimental::csr library configured")

# Tests
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# Benchmarks
if(SUBSETIX_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

message(STATUS "=== Experimental Module Configured ===")
