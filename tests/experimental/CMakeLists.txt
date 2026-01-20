# Experimental algorithms tests
# Only compiled if SUBSETIX_ENABLE_EXPERIMENTAL=ON

if(SUBSETIX_ENABLE_EXPERIMENTAL)
  message(STATUS "Experimental tests will be built")

  # Create test executable
  add_executable(experimental_v1_test
    set_algebra/v1_test.cpp
  )

  target_link_libraries(experimental_v1_test
    PRIVATE
    subsetix_core
    gtest_main
    gmock_main
  )

  target_include_directories(experimental_v1_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
  )

  # Register with CTest
  add_test(NAME ExperimentalV1Test COMMAND experimental_v1_test)

  message(STATUS "  - experimental_v1_test")
else()
  message(STATUS "Experimental tests disabled (SUBSETIX_ENABLE_EXPERIMENTAL=OFF)")
endif()
