# Fixed memory profiling tools for Subsetix
# These tools use atexit() to work around Google Benchmark's std::_Exit(0)

cmake_minimum_required(VERSION 3.20)

project(FixedProfilingTools CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Kokkos headers (we only need the interface headers)
find_path(KOKKOS_TOOLS_INCLUDE_DIR
    NAMES kp_core.hpp
    PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../build-experimental-serial-profile/_deps/kokkos_tools-src
        ${CMAKE_CURRENT_SOURCE_DIR}/../../build-experimental-openmp-profile/_deps/kokkos_tools-src
        ${CMAKE_CURRENT_SOURCE_DIR}/../../build-experimental-cuda-gcc12-profile/_deps/kokkos_tools-src
    PATH_SUFFIXES profiling/all
    NO_DEFAULT_PATH
)

if(NOT KOKKOS_TOOLS_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find Kokkos tools headers. Please build a profiling preset first.")
endif()

message(STATUS "Found Kokkos tools headers at: ${KOKKOS_TOOLS_INCLUDE_DIR}")

# Get the base kokkos-tools source directory
get_filename_component(KOKKOS_TOOLS_BASE_DIR ${KOKKOS_TOOLS_INCLUDE_DIR} DIRECTORY)
get_filename_component(KOKKOS_TOOLS_SRC_DIR ${KOKKOS_TOOLS_BASE_DIR} DIRECTORY)

# Include directories
include_directories(
    ${KOKKOS_TOOLS_INCLUDE_DIR}
    ${KOKKOS_TOOLS_BASE_DIR}  # For core/impl subdirectories
    ${KOKKOS_TOOLS_SRC_DIR}/common/makefile-only  # For kp_config.hpp
)

# Shared library for fixed memory-hwm
add_library(kp_hwm_fixed SHARED kp_hwm_fixed.cpp)
set_target_properties(kp_hwm_fixed PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    PREFIX ""
)

# Shared library for fixed memory-usage
add_library(kp_memory_usage_fixed SHARED kp_memory_usage_fixed.cpp)
set_target_properties(kp_memory_usage_fixed PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    PREFIX ""
)

# Link pthread
target_link_libraries(kp_hwm_fixed PRIVATE pthread)
target_link_libraries(kp_memory_usage_fixed PRIVATE pthread)

# Installation
install(TARGETS kp_hwm_fixed kp_memory_usage_fixed
    LIBRARY DESTINATION lib
)
