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

# Phase 6: FVD Layer Performance Benchmarks
add_executable(fvd_benchmark
    fvd_benchmark.cpp
)

target_link_libraries(fvd_benchmark
    PRIVATE
        subsetix::core
        benchmark::benchmark
)
target_include_directories(fvd_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Phase 7: Productivity Comparison Benchmark (Old vs New API)
add_executable(productivity_comparison
    productivity_comparison.cpp
)

target_link_libraries(productivity_comparison
    PRIVATE
        subsetix::core
        benchmark::benchmark
)
target_include_directories(productivity_comparison PRIVATE ${CMAKE_SOURCE_DIR}/include)
