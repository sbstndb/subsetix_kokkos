# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

# Experimental algorithms benchmarks
# Only compiled if SUBSETIX_ENABLE_EXPERIMENTAL=ON

message(STATUS "Experimental benchmarks will be built")

# Create benchmark executable
add_executable(experimental_v1_benchmark
  set_algebra/v1_benchmark.cpp
)

target_link_libraries(experimental_v1_benchmark
  PRIVATE
    subsetix::core
    experimental_subsetix_csr
    benchmark::benchmark
    benchmark::benchmark_main
)

message(STATUS "  - experimental_v1_benchmark")
