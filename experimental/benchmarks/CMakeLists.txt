# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

# Experimental algorithms benchmarks
# Only compiled if SUBSETIX_ENABLE_EXPERIMENTAL=ON

message(STATUS "Experimental benchmarks will be built")

# Unified comparison benchmark (compares v1/v2/v3 across all patterns)
add_executable(experimental_comparison_benchmark
  set_algebra/unified_comparison_benchmark.cpp
)

target_link_libraries(experimental_comparison_benchmark
  PRIVATE
    experimental_subsetix_csr
    benchmark::benchmark
    benchmark::benchmark_main
)

message(STATUS "  - experimental_comparison_benchmark")
