add_executable(csr_generators_to_vtk
    csr_generators_to_vtk.cpp
)

target_link_libraries(csr_generators_to_vtk
    PRIVATE
        subsetix::core
)

add_executable(csr_bitmap_to_vtk
    csr_bitmap_to_vtk.cpp
)

target_link_libraries(csr_bitmap_to_vtk
    PRIVATE
        subsetix::core
)

add_executable(csr_set_ops_to_vtk
    csr_set_ops_to_vtk.cpp
)

target_link_libraries(csr_set_ops_to_vtk
    PRIVATE
        subsetix::core
)

add_executable(field_mask_fill_to_vtk
    field_mask_fill_to_vtk.cpp
)

target_link_libraries(field_mask_fill_to_vtk
    PRIVATE
        subsetix::core
)

add_executable(field_stencil_smoothing_to_vtk
    field_stencil_smoothing_to_vtk.cpp
)

target_link_libraries(field_stencil_smoothing_to_vtk
    PRIVATE
        subsetix::core
)

add_executable(amr_field_restrict_prolong_to_vtk
    amr_field_restrict_prolong_to_vtk.cpp
)

target_link_libraries(amr_field_restrict_prolong_to_vtk
    PRIVATE
        subsetix::core
)

add_executable(set_algebra_masks_to_vtk
    set_algebra_masks_to_vtk.cpp
)

target_link_libraries(set_algebra_masks_to_vtk
    PRIVATE
        subsetix::core
)

add_executable(csr_morphology_to_vtk
    csr_morphology_to_vtk.cpp
)

target_link_libraries(csr_morphology_to_vtk
    PRIVATE
        subsetix::core
)

add_executable(threshold_to_vtk
    threshold_to_vtk.cpp
)

target_link_libraries(threshold_to_vtk
    PRIVATE
        subsetix::core
)

add_executable(multilevel_amr_demo
    multilevel_amr_demo.cpp
)

target_link_libraries(multilevel_amr_demo
    PRIVATE
        subsetix::core
)

add_executable(field_subview_workflow
    field_subview_workflow.cpp
)

target_link_libraries(field_subview_workflow
    PRIVATE
        subsetix::core
)

add_executable(field_subview_overlap_stencil
    field_subview_overlap_stencil.cpp
)

target_link_libraries(field_subview_overlap_stencil
    PRIVATE
        subsetix::core
)

add_executable(amr_advection_2d
    amr_advection/amr_advection_2d.cpp
)

target_link_libraries(amr_advection_2d
    PRIVATE
        subsetix::core
)

add_executable(mach2_cylinder
    mach2_cylinder/mach2_cylinder.cpp
)

target_link_libraries(mach2_cylinder
    PRIVATE
        subsetix::core
)

add_executable(smoke_plume
    smoke_plume/smoke_plume.cpp
)

target_link_libraries(smoke_plume
    PRIVATE
        subsetix::core
)

# ============================================================================
# FVD High-Level API Examples
# ============================================================================

add_executable(fvd_mach2_cylinder_example
    fvd_mach2_cylinder_example.cpp
)

target_link_libraries(fvd_mach2_cylinder_example
    PRIVATE
        subsetix::core
)
target_include_directories(fvd_mach2_cylinder_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
