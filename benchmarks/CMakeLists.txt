add_executable(kokkos_dummy_benchmark
    kokkos_dummy_benchmark.cpp
)

target_link_libraries(kokkos_dummy_benchmark
    PRIVATE
        subsetix_core
)

add_executable(csr_intersection_bench
    csr_intersection_bench.cpp
)

target_link_libraries(csr_intersection_bench
    PRIVATE
        subsetix_core
        benchmark::benchmark
)
