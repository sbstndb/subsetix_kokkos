# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

# Playground fields benchmarks

if(NOT SUBSETIX_BUILD_BENCHMARKS)
  return()
endif()

message(STATUS "=== Configuring Playground Fields Benchmarks ===")

if(NOT TARGET benchmark::benchmark)
  message(STATUS "  - Google Benchmark target not found, skipping playground fields benchmarks")
  return()
endif()

message(STATUS "  - Google Benchmark found, configuring playground fields benchmarks")

add_executable(playground_fields_stencil_benchmark
  fields/stencil_benchmark.cpp
)

target_link_libraries(playground_fields_stencil_benchmark
  PRIVATE
    playground::fields
    playground::intersection
    Kokkos::kokkos
    benchmark::benchmark
)

target_include_directories(playground_fields_stencil_benchmark
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/playground/intersection/tests
)

message(STATUS "  - playground_fields_stencil_benchmark configured")
message(STATUS "=== Playground Fields Benchmarks Configured ===")
