# Experimental algorithms benchmarks
# Only compiled if SUBSETIX_ENABLE_EXPERIMENTAL=ON

if(SUBSETIX_ENABLE_EXPERIMENTAL)
  message(STATUS "Experimental benchmarks will be built")

  # Create benchmark executable
  add_executable(experimental_v1_benchmark
    set_algebra/v1_benchmark.cpp
  )

  target_link_libraries(experimental_v1_benchmark
    PRIVATE
    subsetix_core
    benchmark::benchmark
    benchmark::benchmark_main
  )

  target_include_directories(experimental_v1_benchmark
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
  )

  message(STATUS "  - experimental_v1_benchmark")
else()
  message(STATUS "Experimental benchmarks disabled (SUBSETIX_ENABLE_EXPERIMENTAL=OFF)")
endif()
