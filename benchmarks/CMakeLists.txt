add_executable(kokkos_dummy_benchmark
    kokkos_dummy_benchmark.cpp
)

target_link_libraries(kokkos_dummy_benchmark
    PRIVATE
        subsetix_core
)

