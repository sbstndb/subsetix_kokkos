# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

# Experimental algorithms benchmarks
# Only compiled if SUBSETIX_ENABLE_EXPERIMENTAL=ON

message(STATUS "Experimental benchmarks will be built")

# Unified comparison benchmark (compares v1/v2/v3 using configs)
add_executable(experimental_comparison_benchmark
  set_algebra/unified_comparison_benchmark.cpp
)

# Include test headers for RandomMeshGenerator and configs
get_filename_component(EXPERIMENTAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../" ABSOLUTE)
target_include_directories(experimental_comparison_benchmark
  PRIVATE
    ${EXPERIMENTAL_ROOT}/tests
)

target_link_libraries(experimental_comparison_benchmark
  PRIVATE
    experimental_subsetix_csr
    benchmark::benchmark
    benchmark::benchmark_main
)

message(STATUS "  - experimental_comparison_benchmark")

# Regular mesh benchmark (optimal performance scenario)
add_executable(experimental_regular_mesh_benchmark
  set_algebra/regular_mesh_benchmark.cpp
)

target_include_directories(experimental_regular_mesh_benchmark
  PRIVATE
    ${EXPERIMENTAL_ROOT}/tests
)

target_link_libraries(experimental_regular_mesh_benchmark
  PRIVATE
    experimental_subsetix_csr
    benchmark::benchmark
    benchmark::benchmark_main
)

message(STATUS "  - experimental_regular_mesh_benchmark")
