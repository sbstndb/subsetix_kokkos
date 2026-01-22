cmake_minimum_required(VERSION 3.16)

project(subsetix_kokkos
    VERSION 0.1.0
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SUBSETIX_KOKKOS_OPENMP "Enable Kokkos OpenMP backend" OFF)
option(SUBSETIX_KOKKOS_CUDA "Enable Kokkos CUDA backend" OFF)

option(SUBSETIX_USE_MPI "Enable MPI support for FVD" OFF)

option(SUBSETIX_EXECSPACE_FORCE_CUDA "Force ExecSpace = Kokkos::Cuda" OFF)
option(SUBSETIX_EXECSPACE_FORCE_OPENMP "Force ExecSpace = Kokkos::OpenMP" OFF)
option(SUBSETIX_EXECSPACE_FORCE_SERIAL "Force ExecSpace = Kokkos::Serial" OFF)

option(SUBSETIX_MEMORYSPACE_FORCE_UVM "Force DeviceMemorySpace = Kokkos::CudaUVMSpace" OFF)
option(SUBSETIX_MEMORYSPACE_FORCE_HOSTPINNED "Force DeviceMemorySpace = Kokkos::HostPinnedSpace" OFF)

option(SUBSETIX_ENABLE_COVERAGE "Enable code coverage analysis (GCC only)" OFF)

# ========================================================================
# Profiling Configuration
# ========================================================================

option(SUBSETIX_ENABLE_PROFILING "Enable profiling support (adds debug symbols for nvprof/Nsight)" OFF)

if(SUBSETIX_ENABLE_PROFILING)
  message(STATUS "=== Profiling Enabled ===")
  add_compile_definitions(SUBSETIX_ENABLE_PROFILING)

  # Add debug symbols for profiling tools
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-g>)
  add_link_options(-g)

  # For nvprof, we want Release optimization with debug symbols
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(WARNING "Profiling with Debug build may produce misleading results. Consider using RelWithDebInfo or Release.")
  endif()

  message(STATUS "Profiling support enabled (debug symbols added)")
endif()

# ========================================================================
# Experimental Algorithms Configuration
# ========================================================================

option(SUBSETIX_ENABLE_EXPERIMENTAL "Enable experimental set algebra algorithms" OFF)

if(SUBSETIX_ENABLE_EXPERIMENTAL)
  message(STATUS "=== Experimental Algorithms Enabled ===")
  add_compile_definitions(SUBSETIX_ENABLE_EXPERIMENTAL)
  message(STATUS "Experimental algorithms will be compiled and tested")
else()
  message(STATUS "Experimental algorithms disabled (default)")
endif()

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

# ========================================================================
# Code Coverage Configuration
# ========================================================================

if(SUBSETIX_ENABLE_COVERAGE)
  message(STATUS "=== Code Coverage Enabled ===")

  # Check if using GCC (gcov only works with GCC)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    message(FATAL_ERROR "Coverage analysis requires GCC compiler")
  endif()

  # Add coverage flags
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:--coverage>)
  add_link_options(--coverage)

  # Ensure debug build for accurate coverage
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type for coverage" FORCE)
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0" CACHE STRING "" FORCE)

  message(STATUS "Coverage flags added for GCC")
endif()

# ========================================================================
# Compile Commands for clang-tidy
# ========================================================================

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

# ========================================================================
# MPI Configuration
# ========================================================================

if(SUBSETIX_USE_MPI)
    # Find MPI
    find_package(MPI REQUIRED)

    # Add compile definition
    add_compile_definitions(SUBSETIX_USE_MPI)

    # Link MPI libraries to targets that need them
    # (This will be done in the subdirectories)

    message(STATUS "MPI enabled:")
    message(STATUS "  MPI_CXX_COMPILER: ${MPI_CXX_COMPILER}")
    message(STATUS "  MPI_CXX_LIBRARIES: ${MPI_CXX_LIBRARIES}")
    message(STATUS "  MPI_CXX_INCLUDE_PATH: ${MPI_CXX_INCLUDE_PATH}")
else()
    message(STATUS "MPI disabled - FVD will use single-rank mode")
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

# Stable libraries can be disabled to only build experimental
option(SUBSETIX_BUILD_STABLE_LIBS "Build stable (non-experimental) libraries" ON)
if(SUBSETIX_BUILD_STABLE_LIBS)
  add_subdirectory(src)
endif()

# Stable tests can be disabled to only build experimental
option(SUBSETIX_BUILD_STABLE_TESTS "Build stable (non-experimental) tests" ON)
if(SUBSETIX_BUILD_STABLE_TESTS)
  add_subdirectory(tests)
endif()

# Stable benchmarks can be disabled to only build experimental
option(SUBSETIX_BUILD_STABLE_BENCHMARKS "Build stable (non-experimental) benchmarks" ON)
if(SUBSETIX_BUILD_STABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# Examples depend on stable libraries
if(SUBSETIX_BUILD_STABLE_LIBS)
  add_subdirectory(examples)
endif()

add_subdirectory(experimental)
