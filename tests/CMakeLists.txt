# Group tests by functionality to avoid ODR/CUDA linking issues in monolithic builds

# 1. Core Tests: Builders, Basic Data Structures
add_executable(subsetix_test_core
    test_main.cpp
    kokkos_test.cpp
    csr_interval_set_test.cpp
    csr_interval_subset_test.cpp
    csr_builders_test.cpp
    csr_field_test.cpp
    workspace_capacity_test.cpp
    vtk_export_test.cpp
)
target_link_libraries(subsetix_test_core PRIVATE subsetix::core GTest::gtest)
add_test(NAME CoreTests COMMAND subsetix_test_core)

# 2. Set Algebra Tests: Union, Intersection, Difference
add_executable(subsetix_test_ops
    test_main.cpp
    csr_union_test.cpp
    csr_union_components_test.cpp
    csr_intersection_test.cpp
    csr_intersection_components_test.cpp
    csr_difference_test.cpp
    csr_symmetric_difference_test.cpp
    csr_difference_components_test.cpp
    csr_set_algebra_properties_test.cpp
)
target_link_libraries(subsetix_test_ops PRIVATE subsetix::core GTest::gtest)
add_test(NAME OpsTests COMMAND subsetix_test_ops)

# 3. Field & Advanced Ops Tests: Field Ops, Morpho, Threshold
add_executable(subsetix_test_advanced
    test_main.cpp
    csr_field_ops_test.cpp
    csr_field_stencil_test.cpp
    csr_field_stencil_subset_test.cpp
    csr_field_algebra_test.cpp
    csr_field_api_compat_test.cpp
    csr_row_ops_components_test.cpp
    csr_morphology_test.cpp
    csr_threshold_test.cpp
    csr_translation_test.cpp
    csr_field_subview_test.cpp
)
target_link_libraries(subsetix_test_advanced PRIVATE subsetix::core GTest::gtest)
add_test(NAME AdvancedTests COMMAND subsetix_test_advanced)

# 4. AMR & Multilevel Tests: Refine/Project, Remap, Multilevel
# Grouping these ensures they are compiled in a context where heavy templates/kernels
# don't conflict with basic ops.
add_executable(subsetix_test_amr
    test_main.cpp
    csr_field_amr_ops_test.cpp
    csr_field_remap_test.cpp
    csr_amr_refine_project_test.cpp
    multilevel_test.cpp
)
target_link_libraries(subsetix_test_amr PRIVATE subsetix::core GTest::gtest)
add_test(NAME AmrTests COMMAND subsetix_test_amr)
