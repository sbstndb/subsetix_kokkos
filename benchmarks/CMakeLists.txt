add_executable(csr_set_algebra_benchmark
    csr_set_algebra_benchmark.cpp
)

target_link_libraries(csr_set_algebra_benchmark
    PRIVATE
        subsetix::core
        benchmark::benchmark
)

add_executable(csr_set_transform_benchmark
    csr_set_transform_benchmark.cpp
)

target_link_libraries(csr_set_transform_benchmark
    PRIVATE
        subsetix::core
        benchmark::benchmark
)

add_executable(csr_field_ops_benchmark
    csr_field_ops_benchmark.cpp
)

target_link_libraries(csr_field_ops_benchmark
    PRIVATE
        subsetix::core
        benchmark::benchmark
)

add_executable(csr_morphology_benchmark
    csr_morphology_benchmark.cpp
)

target_link_libraries(csr_morphology_benchmark
    PRIVATE
        subsetix::core
        benchmark::benchmark
)

add_executable(csr_threshold_benchmark
    csr_threshold_benchmark.cpp
)

target_link_libraries(csr_threshold_benchmark
    PRIVATE
        subsetix::core
        benchmark::benchmark
)

add_executable(multilevel_overhead_benchmark
    multilevel_overhead_benchmark.cpp
)

target_link_libraries(multilevel_overhead_benchmark
    PRIVATE
        subsetix::core
        benchmark::benchmark
)

# Experimental benchmarks (only if SUBSETIX_ENABLE_EXPERIMENTAL=ON)
add_subdirectory(experimental)
