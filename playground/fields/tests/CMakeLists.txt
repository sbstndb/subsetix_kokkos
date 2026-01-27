# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

# Playground fields tests

if(NOT BUILD_TESTING)
  return()
endif()

message(STATUS "=== Configuring Playground Fields Tests ===")

add_executable(playground_fields_tests
  fields/test_main.cpp
  fields/test_stencil_2d.cpp
  fields/test_stencil_3d.cpp
)

target_link_libraries(playground_fields_tests
  PRIVATE
    playground::fields
    Kokkos::kokkos
    GTest::gtest
)

target_include_directories(playground_fields_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(NAME playground_fields_tests COMMAND playground_fields_tests)

set_tests_properties(playground_fields_tests PROPERTIES
  LABELS "playground;fields"
)

message(STATUS "  - playground_fields_tests configured")
message(STATUS "=== Playground Fields Tests Configured ===")
