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

# ============================================================================
# Row Mapping Alternatives Libraries
# ============================================================================

# v4: Hash-based row mapping
add_library(playground_intersection_hash_based INTERFACE)
target_link_libraries(playground_intersection_hash_based
  INTERFACE
    playground_subsetix_intersection
    subsetix::geometry
)
target_include_directories(playground_intersection_hash_based
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/intersection/include
)
add_library(playground::intersection::hash_based ALIAS playground_intersection_hash_based)

# v5: Parallel merge row mapping
add_library(playground_intersection_parallel_merge INTERFACE)
target_link_libraries(playground_intersection_parallel_merge
  INTERFACE
    playground_subsetix_intersection
    subsetix::geometry
)
target_include_directories(playground_intersection_parallel_merge
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/intersection/include
)
add_library(playground::intersection::parallel_merge ALIAS playground_intersection_parallel_merge)

# v6: Direct index row mapping
add_library(playground_intersection_direct_index INTERFACE)
target_link_libraries(playground_intersection_direct_index
  INTERFACE
    playground_subsetix_intersection
    subsetix::geometry
)
target_include_directories(playground_intersection_direct_index
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/intersection/include
)
add_library(playground::intersection::direct_index ALIAS playground_intersection_direct_index)

# v7: SoA optimized row mapping
add_library(playground_intersection_soa_optimized INTERFACE)
target_link_libraries(playground_intersection_soa_optimized
  INTERFACE
    playground_subsetix_intersection
    subsetix::geometry
)
target_include_directories(playground_intersection_soa_optimized
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/intersection/include
)
add_library(playground::intersection::soa_optimized ALIAS playground_intersection_soa_optimized)

# v8: Hybrid CPU-GPU row mapping
add_library(playground_intersection_hybrid_cpu_gpu INTERFACE)
target_link_libraries(playground_intersection_hybrid_cpu_gpu
  INTERFACE
    playground_subsetix_intersection
    subsetix::geometry
)
target_include_directories(playground_intersection_hybrid_cpu_gpu
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/intersection/include
)
add_library(playground::intersection::hybrid_cpu_gpu ALIAS playground_intersection_hybrid_cpu_gpu)

# v9: Adaptive row mapping (depends on all others)
add_library(playground_intersection_adaptive INTERFACE)
target_link_libraries(playground_intersection_adaptive
  INTERFACE
    playground_subsetix_intersection
    subsetix::geometry
    playground_intersection_hash_based
    playground_intersection_parallel_merge
    playground_intersection_direct_index
    playground_intersection_soa_optimized
    playground_intersection_hybrid_cpu_gpu
)
target_include_directories(playground_intersection_adaptive
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/intersection/include
)
add_library(playground::intersection::adaptive ALIAS playground_intersection_adaptive)

message(STATUS "  - playground::intersection library configured")
message(STATUS "  - playground::intersection::hash_based library configured")
message(STATUS "  - playground::intersection::parallel_merge library configured")
message(STATUS "  - playground::intersection::direct_index library configured")
message(STATUS "  - playground::intersection::soa_optimized library configured")
message(STATUS "  - playground::intersection::hybrid_cpu_gpu library configured")
message(STATUS "  - playground::intersection::adaptive library configured")

# Tests
if(BUILD_TESTING)
  add_subdirectory(intersection/tests)
endif()

# Benchmarks
option(SUBSETIX_BUILD_BENCHMARKS "Build playground benchmarks" ON)
if(SUBSETIX_BUILD_BENCHMARKS)
  add_subdirectory(intersection/benchmarks)
endif()

message(STATUS "=== Playground Module Configured ===")
