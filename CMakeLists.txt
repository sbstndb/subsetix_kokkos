cmake_minimum_required(VERSION 3.16)

project(subsetix_kokkos
    VERSION 0.1.0
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SUBSETIX_KOKKOS_OPENMP "Enable Kokkos OpenMP backend" OFF)
option(SUBSETIX_KOKKOS_CUDA "Enable Kokkos CUDA backend" OFF)

option(SUBSETIX_EXECSPACE_FORCE_CUDA "Force ExecSpace = Kokkos::Cuda" OFF)
option(SUBSETIX_EXECSPACE_FORCE_OPENMP "Force ExecSpace = Kokkos::OpenMP" OFF)
option(SUBSETIX_EXECSPACE_FORCE_SERIAL "Force ExecSpace = Kokkos::Serial" OFF)

option(SUBSETIX_MEMORYSPACE_FORCE_UVM "Force DeviceMemorySpace = Kokkos::CudaUVMSpace" OFF)
option(SUBSETIX_MEMORYSPACE_FORCE_HOSTPINNED "Force DeviceMemorySpace = Kokkos::HostPinnedSpace" OFF)

set(_SUBSETIX_EXECSPACE_FORCE_COUNT 0)
if(SUBSETIX_EXECSPACE_FORCE_CUDA)
  math(EXPR _SUBSETIX_EXECSPACE_FORCE_COUNT "${_SUBSETIX_EXECSPACE_FORCE_COUNT} + 1")
endif()
if(SUBSETIX_EXECSPACE_FORCE_OPENMP)
  math(EXPR _SUBSETIX_EXECSPACE_FORCE_COUNT "${_SUBSETIX_EXECSPACE_FORCE_COUNT} + 1")
endif()
if(SUBSETIX_EXECSPACE_FORCE_SERIAL)
  math(EXPR _SUBSETIX_EXECSPACE_FORCE_COUNT "${_SUBSETIX_EXECSPACE_FORCE_COUNT} + 1")
endif()

if(_SUBSETIX_EXECSPACE_FORCE_COUNT GREATER 1)
  message(FATAL_ERROR "Only one of SUBSETIX_EXECSPACE_FORCE_{CUDA,OPENMP,SERIAL} can be ON.")
endif()

set(_SUBSETIX_MEMORYSPACE_FORCE_COUNT 0)
if(SUBSETIX_MEMORYSPACE_FORCE_UVM)
  math(EXPR _SUBSETIX_MEMORYSPACE_FORCE_COUNT "${_SUBSETIX_MEMORYSPACE_FORCE_COUNT} + 1")
endif()
if(SUBSETIX_MEMORYSPACE_FORCE_HOSTPINNED)
  math(EXPR _SUBSETIX_MEMORYSPACE_FORCE_COUNT "${_SUBSETIX_MEMORYSPACE_FORCE_COUNT} + 1")
endif()

if(_SUBSETIX_MEMORYSPACE_FORCE_COUNT GREATER 1)
  message(FATAL_ERROR "Only one of SUBSETIX_MEMORYSPACE_FORCE_{UVM,HOSTPINNED} can be ON.")
endif()

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

if(SUBSETIX_EXECSPACE_FORCE_OPENMP)
    set(SUBSETIX_KOKKOS_OPENMP ON CACHE BOOL "" FORCE)
endif()

if(SUBSETIX_KOKKOS_CUDA)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Kokkos CUDA backend" FORCE)
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable Kokkos CUDA lambdas" FORCE)
else()
    set(Kokkos_ENABLE_CUDA OFF CACHE BOOL "Disable Kokkos CUDA backend" FORCE)
    set(Kokkos_ENABLE_CUDA_LAMBDA OFF CACHE BOOL "Disable Kokkos CUDA lambdas" FORCE)
endif()

if(SUBSETIX_EXECSPACE_FORCE_CUDA)
    set(SUBSETIX_KOKKOS_CUDA ON CACHE BOOL "" FORCE)
endif()

if(SUBSETIX_EXECSPACE_FORCE_SERIAL)
    # Serial is enabled by default; nothing else needed here.
endif()

FetchContent_MakeAvailable(kokkos)

if(SUBSETIX_EXECSPACE_FORCE_CUDA)
    add_compile_definitions(SUBSETIX_EXECSPACE_CUDA)
elseif(SUBSETIX_EXECSPACE_FORCE_OPENMP)
    add_compile_definitions(SUBSETIX_EXECSPACE_OPENMP)
elseif(SUBSETIX_EXECSPACE_FORCE_SERIAL)
    add_compile_definitions(SUBSETIX_EXECSPACE_SERIAL)
endif()

if(SUBSETIX_MEMORYSPACE_FORCE_UVM)
    add_compile_definitions(SUBSETIX_MEMORYSPACE_FORCE_UVM)
elseif(SUBSETIX_MEMORYSPACE_FORCE_HOSTPINNED)
    add_compile_definitions(SUBSETIX_MEMORYSPACE_FORCE_HOSTPINNED)
endif()

# GoogleTest for unit tests
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Google Benchmark for micro-benchmarks
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.4
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)
add_subdirectory(examples)
