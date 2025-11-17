set(SUBSETIX_TEST_SOURCES
    test_main.cpp
    kokkos_smoke_test.cpp
    csr_interval_set_smoke_test.cpp
    csr_builders_smoke_test.cpp
    csr_field_smoke_test.cpp
    csr_field_ops_smoke_test.cpp
    csr_field_stencil_smoke_test.cpp
    csr_field_amr_ops_smoke_test.cpp
    csr_union_smoke_test.cpp
    csr_union_components_smoke_test.cpp
    csr_intersection_smoke_test.cpp
    csr_intersection_components_smoke_test.cpp
    csr_difference_smoke_test.cpp
    csr_difference_components_smoke_test.cpp
    csr_row_ops_components_smoke_test.cpp
    csr_set_algebra_properties_test.cpp
    csr_translation_smoke_test.cpp
    csr_amr_refine_project_smoke_test.cpp
)

add_executable(subsetix_tests
    ${SUBSETIX_TEST_SOURCES}
)

target_link_libraries(subsetix_tests
    PRIVATE
        subsetix_core
        GTest::gtest
)

add_test(NAME SubsetixAllTests
         COMMAND subsetix_tests)
