# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

# Experimental algorithms benchmarks
# Only compiled if SUBSETIX_ENABLE_EXPERIMENTAL=ON

message(STATUS "Experimental benchmarks will be built")

# v1 benchmark (original)
add_executable(experimental_v1_benchmark
  set_algebra/v1_benchmark.cpp
)

target_link_libraries(experimental_v1_benchmark
  PRIVATE
    experimental_subsetix_csr
    benchmark::benchmark
    benchmark::benchmark_main
)

# v2 benchmark
add_executable(experimental_v2_benchmark
  set_algebra/v2_benchmark.cpp
)

target_link_libraries(experimental_v2_benchmark
  PRIVATE
    experimental_subsetix_csr
    benchmark::benchmark
    benchmark::benchmark_main
)

# Unified comparison benchmark (v1 vs v2)
add_executable(experimental_comparison_benchmark
  set_algebra/unified_comparison_benchmark.cpp
)

target_link_libraries(experimental_comparison_benchmark
  PRIVATE
    experimental_subsetix_csr
    benchmark::benchmark
    benchmark::benchmark_main
)

message(STATUS "  - experimental_v1_benchmark")
message(STATUS "  - experimental_v2_benchmark")
message(STATUS "  - experimental_comparison_benchmark")
