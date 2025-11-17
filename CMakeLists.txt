cmake_minimum_required(VERSION 3.16)

project(subsetix_kokkos
    VERSION 0.1.0
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SUBSETIX_KOKKOS_OPENMP "Enable Kokkos OpenMP backend" OFF)
option(SUBSETIX_KOKKOS_CUDA "Enable Kokkos CUDA backend" OFF)

include(FetchContent)

FetchContent_Declare(
    kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG 4.5.00
)

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Kokkos SERIAL backend" FORCE)

if(SUBSETIX_KOKKOS_OPENMP)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable Kokkos OPENMP backend" FORCE)
else()
    set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "Disable Kokkos OPENMP backend" FORCE)
endif()

if(SUBSETIX_KOKKOS_CUDA)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Kokkos CUDA backend" FORCE)
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable Kokkos CUDA lambdas" FORCE)
else()
    set(Kokkos_ENABLE_CUDA OFF CACHE BOOL "Disable Kokkos CUDA backend" FORCE)
    set(Kokkos_ENABLE_CUDA_LAMBDA OFF CACHE BOOL "Disable Kokkos CUDA lambdas" FORCE)
endif()

FetchContent_MakeAvailable(kokkos)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)
