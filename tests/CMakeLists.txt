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

# 5. FVD High-Level API Tests: Compilation tests for FVD layer
# These tests verify that the FVD high-level API compiles correctly
# Note: Does not use test_main.cpp - has its own main()
add_executable(subsetix_test_fvd_api
    fvd_high_level_api_test.cpp
)
target_link_libraries(subsetix_test_fvd_api PRIVATE subsetix::core GTest::gtest)
target_include_directories(subsetix_test_fvd_api PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME FvdApiTests COMMAND subsetix_test_fvd_api)

# 6. FVD Execution Tests: Numerical validation and GPU tests
# These tests actually execute code to verify correctness
# Note: Does not use test_main.cpp - has its own main()
add_executable(subsetix_test_fvd_execution
    fvd_execution_tests.cpp
)
target_link_libraries(subsetix_test_fvd_execution PRIVATE subsetix::core GTest::gtest)
target_include_directories(subsetix_test_fvd_execution PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME FvdExecutionTests COMMAND subsetix_test_fvd_execution)

# 7. FVD Integrators Tests: Time integrators, AMR with coarsening, time-dependent BCs
# Tests for the new high-level features
# Note: Does not use test_main.cpp - has its own main()
add_executable(subsetix_test_fvd_integrators
    fvd_integrators_test.cpp
)
target_link_libraries(subsetix_test_fvd_integrators PRIVATE subsetix::core GTest::gtest)
target_include_directories(subsetix_test_fvd_integrators PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME FvdIntegratorsTests COMMAND subsetix_test_fvd_integrators)

# 8. Mach2 Validation Tests: Type safety and validation framework
# Tests for mach2_cylinder FVD migration validation
add_executable(subsetix_test_mach2_validation
    mach2_validation/type_safety_tests.cpp
)
target_link_libraries(subsetix_test_mach2_validation PRIVATE subsetix::core GTest::gtest)
target_include_directories(subsetix_test_mach2_validation PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(subsetix_test_mach2_validation PRIVATE ${CMAKE_SOURCE_DIR}/examples/mach2_cylinder)
add_test(NAME Mach2ValidationTests COMMAND subsetix_test_mach2_validation)

# 9. FVD Multi-System Tests: Phase 6 - Extensibility Validation
# Tests that AdaptiveSolver works with ANY System (Euler2D, Advection2D, etc.)
add_executable(subsetix_test_fvd_multi_system
    fvd_multi_system_test.cpp
)
target_link_libraries(subsetix_test_fvd_multi_system PRIVATE subsetix::core GTest::gtest)
target_include_directories(subsetix_test_fvd_multi_system PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME FvdMultiSystemTests COMMAND subsetix_test_fvd_multi_system)

# 10. FVD Accuracy Tests: Phase 6 - Accuracy and Validation
# Tests for numerical accuracy, mass conservation, and stability
add_executable(subsetix_test_fvd_accuracy
    fvd_accuracy_test.cpp
)
target_link_libraries(subsetix_test_fvd_accuracy PRIVATE subsetix::core GTest::gtest)
target_include_directories(subsetix_test_fvd_accuracy PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME FvdAccuracyTests COMMAND subsetix_test_fvd_accuracy)

# 10.5. FVD Numerical Accuracy Tests: Comprehensive numerical validation
# Tests for analytical solution comparison, convergence rates, flux schemes, time integrators
add_executable(subsetix_test_fvd_numerical_accuracy
    fvd_numerical_accuracy_test.cpp
)
target_link_libraries(subsetix_test_fvd_numerical_accuracy PRIVATE subsetix::core GTest::gtest)
target_include_directories(subsetix_test_fvd_numerical_accuracy PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME FvdNumericalAccuracyTests COMMAND subsetix_test_fvd_numerical_accuracy)

# 11. FVD Regression Suite: Comprehensive regression testing
# Tests for all flux schemes, reconstruction methods, time integrators,
# AMR operations, boundary conditions, and performance regression detection
add_executable(subsetix_test_fvd_regression
    fvd_regression_suite.cpp
)
target_link_libraries(subsetix_test_fvd_regression PRIVATE subsetix::core GTest::gtest)
target_include_directories(subsetix_test_fvd_regression PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME FvdRegressionSuite COMMAND subsetix_test_fvd_regression)

# 12. FVD Cross-Backend Tests: Validation across Serial, OpenMP, CUDA
# Tests that verify bit-identical results across different Kokkos backends
# Note: Does not use test_main.cpp - has its own main()
add_executable(subsetix_test_fvd_cross_backend
    fvd_cross_backend_test.cpp
)
target_link_libraries(subsetix_test_fvd_cross_backend PRIVATE subsetix::core GTest::gtest)
target_include_directories(subsetix_test_fvd_cross_backend PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME FvdCrossBackendTests COMMAND subsetix_test_fvd_cross_backend)

# 13. FVD WENO Reconstruction Tests: High-Order Reconstruction Validation
# Tests for WENO-JS and WENO-Z reconstruction methods
# Note: Does not use test_main.cpp - has its own main()
add_executable(subsetix_test_fvd_weno
    fvd_weno_reconstruction_test.cpp
)
target_link_libraries(subsetix_test_fvd_weno PRIVATE subsetix::core GTest::gtest)
target_include_directories(subsetix_test_fvd_weno PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME FvdWenoTests COMMAND subsetix_test_fvd_weno)
