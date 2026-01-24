# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

if(NOT SUBSETIX_BUILD_BENCHMARKS)
  return()
endif()

message(STATUS "=== Configuring Playground Intersection Benchmarks ===")

find_package(benchmark QUIET)

if(NOT benchmark_FOUND)
  message(STATUS "  - Google Benchmark not found, skipping playground benchmarks")
  return()
endif()

message(STATUS "  - Google Benchmark found, configuring playground intersection benchmarks")

add_executable(playground_intersection_regular_benchmark
  intersection/regular_mesh_benchmark.cpp
)

add_executable(playground_intersection_comparison_benchmark
  intersection/unified_comparison_benchmark.cpp
)

add_executable(playground_benchmark_v4_hash
  intersection/benchmark_v4_hash.cpp
)

add_executable(playground_benchmark_v5_parallel_merge
  intersection/benchmark_v5_parallel_merge.cpp
)

add_executable(playground_benchmark_v6_direct_index
  intersection/benchmark_v6_direct_index.cpp
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

target_link_libraries(playground_benchmark_v4_hash
  PRIVATE
    playground::intersection
    Kokkos::kokkos
    benchmark::benchmark
    benchmark::benchmark_main
)

target_link_libraries(playground_benchmark_v5_parallel_merge
  PRIVATE
    playground::intersection
    Kokkos::kokkos
    benchmark::benchmark
    benchmark::benchmark_main
)

target_link_libraries(playground_benchmark_v6_direct_index
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

target_include_directories(playground_benchmark_v4_hash
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/playground/intersection/tests
)

target_include_directories(playground_benchmark_v5_parallel_merge
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/playground/intersection/tests
)

target_include_directories(playground_benchmark_v6_direct_index
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/playground/intersection/tests
)

message(STATUS "  - playground_intersection_regular_benchmark configured")
message(STATUS "  - playground_intersection_comparison_benchmark configured")
message(STATUS "  - playground_benchmark_v4_hash configured")
message(STATUS "  - playground_benchmark_v5_parallel_merge configured")
message(STATUS "  - playground_benchmark_v6_direct_index configured")
message(STATUS "=== Playground Intersection Benchmarks Configured ===")
