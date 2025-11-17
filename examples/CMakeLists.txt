add_executable(csr_generators_to_vtk
    csr_generators_to_vtk.cpp
)

target_link_libraries(csr_generators_to_vtk
    PRIVATE
        subsetix_core
)

