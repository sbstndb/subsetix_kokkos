add_executable(kokkos_dummy_benchmark
    kokkos_dummy_benchmark.cpp
)

target_link_libraries(kokkos_dummy_benchmark
    PRIVATE
        subsetix_core
)

add_executable(csr_set_algebra_benchmark
    csr_set_algebra_benchmark.cpp
)

target_link_libraries(csr_set_algebra_benchmark
    PRIVATE
        subsetix_core
        benchmark::benchmark
)

add_executable(csr_set_transform_benchmark
    csr_set_transform_benchmark.cpp
)

target_link_libraries(csr_set_transform_benchmark
    PRIVATE
        subsetix_core
        benchmark::benchmark
)

add_executable(csr_field_ops_benchmark
    csr_field_ops_benchmark.cpp
)

target_link_libraries(csr_field_ops_benchmark
    PRIVATE
        subsetix_core
        benchmark::benchmark
)
