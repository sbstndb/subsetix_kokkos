# Group tests by functionality to avoid ODR/CUDA linking issues in monolithic builds

# 1. Core Tests: Builders, Basic Data Structures
add_executable(subsetix_test_core
    test_main.cpp
    kokkos_smoke_test.cpp
    csr_interval_set_smoke_test.cpp
    csr_builders_smoke_test.cpp
    csr_field_smoke_test.cpp
    workspace_capacity_test.cpp
    vtk_export_test.cpp
)
target_link_libraries(subsetix_test_core PRIVATE subsetix_core GTest::gtest)
add_test(NAME CoreTests COMMAND subsetix_test_core)

# 2. Set Algebra Tests: Union, Intersection, Difference
add_executable(subsetix_test_ops
    test_main.cpp
    csr_union_smoke_test.cpp
    csr_union_components_smoke_test.cpp
    csr_intersection_smoke_test.cpp
    csr_intersection_components_smoke_test.cpp
    csr_difference_smoke_test.cpp
    csr_symmetric_difference_smoke_test.cpp
    csr_difference_components_smoke_test.cpp
    csr_set_algebra_properties_test.cpp
)
target_link_libraries(subsetix_test_ops PRIVATE subsetix_core GTest::gtest)
add_test(NAME OpsTests COMMAND subsetix_test_ops)

# 3. Field & Advanced Ops Tests: Field Ops, Morpho, Threshold
add_executable(subsetix_test_advanced
    test_main.cpp
    csr_field_ops_smoke_test.cpp
    csr_field_stencil_smoke_test.cpp
    csr_field_algebra_smoke_test.cpp
    csr_field_new_smoke_test.cpp
    csr_row_ops_components_smoke_test.cpp
    csr_morphology_smoke_test.cpp
    csr_threshold_smoke_test.cpp
    csr_translation_smoke_test.cpp
    csr_field_subview_smoke_test.cpp
)
target_link_libraries(subsetix_test_advanced PRIVATE subsetix_core GTest::gtest)
add_test(NAME AdvancedTests COMMAND subsetix_test_advanced)

# 4. AMR & Multilevel Tests: Refine/Project, Remap, Multilevel
# Grouping these ensures they are compiled in a context where heavy templates/kernels
# don't conflict with basic ops.
add_executable(subsetix_test_amr
    test_main.cpp
    csr_field_amr_ops_smoke_test.cpp
    csr_field_remap_smoke_test.cpp
    csr_amr_refine_project_smoke_test.cpp
    multilevel_smoke_test.cpp
)
target_link_libraries(subsetix_test_amr PRIVATE subsetix_core GTest::gtest)
add_test(NAME AmrTests COMMAND subsetix_test_amr)
