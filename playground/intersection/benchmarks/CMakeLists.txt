# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

# Playground intersection benchmarks

if(NOT SUBSETIX_BUILD_BENCHMARKS)
  return()
endif()

message(STATUS "=== Configuring Playground Intersection Benchmarks ===")

# Find Google Benchmark
find_package(benchmark QUIET)

if(NOT benchmark_FOUND)
  message(STATUS "  - Google Benchmark not found, skipping playground benchmarks")
  return()
endif()

message(STATUS "  - Google Benchmark found, configuring playground intersection benchmarks")

# Create benchmark executables
add_executable(playground_intersection_regular_benchmark
  intersection/regular_mesh_benchmark.cpp
)

add_executable(playground_intersection_comparison_benchmark
  intersection/unified_comparison_benchmark.cpp
)

add_executable(playground_intersection_workspace_benchmark
  intersection/workspace_benchmark.cpp
)

add_executable(playground_intersection_phase_benchmark
  intersection/phase_benchmark.cpp
)

target_link_libraries(playground_intersection_regular_benchmark
  PRIVATE
    playground::intersection
    Kokkos::kokkos
    benchmark::benchmark
    benchmark::benchmark_main
)

target_link_libraries(playground_intersection_comparison_benchmark
  PRIVATE
    playground::intersection
    Kokkos::kokkos
    benchmark::benchmark
    benchmark::benchmark_main
)

target_link_libraries(playground_intersection_workspace_benchmark
  PRIVATE
    playground::intersection
    Kokkos::kokkos
    benchmark::benchmark
    benchmark::benchmark_main
)

target_link_libraries(playground_intersection_phase_benchmark
  PRIVATE
    playground::intersection
    Kokkos::kokkos
    benchmark::benchmark
    benchmark::benchmark_main
)

target_include_directories(playground_intersection_regular_benchmark
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/playground/intersection/tests
)

target_include_directories(playground_intersection_comparison_benchmark
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/playground/intersection/tests
)

target_include_directories(playground_intersection_workspace_benchmark
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/playground/intersection/tests
)

target_include_directories(playground_intersection_phase_benchmark
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/playground/intersection/tests
)

message(STATUS "  - playground_intersection_regular_benchmark configured")
message(STATUS "  - playground_intersection_comparison_benchmark configured")
message(STATUS "  - playground_intersection_workspace_benchmark configured")
message(STATUS "  - playground_intersection_phase_benchmark configured")

message(STATUS "=== Playground Intersection Benchmarks Configured ===")
