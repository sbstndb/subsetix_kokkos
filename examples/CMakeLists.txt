add_executable(csr_generators_to_vtk
    csr_generators_to_vtk.cpp
)

target_link_libraries(csr_generators_to_vtk
    PRIVATE
        subsetix_core
)

add_executable(csr_set_ops_to_vtk
    csr_set_ops_to_vtk.cpp
)

target_link_libraries(csr_set_ops_to_vtk
    PRIVATE
        subsetix_core
)
