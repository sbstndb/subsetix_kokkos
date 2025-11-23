add_library(subsetix_geometry INTERFACE)
add_library(subsetix::geometry ALIAS subsetix_geometry)

target_link_libraries(subsetix_geometry
    INTERFACE
        Kokkos::kokkos
)

target_include_directories(subsetix_geometry
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)

add_library(subsetix_field INTERFACE)
add_library(subsetix::field ALIAS subsetix_field)

target_link_libraries(subsetix_field
    INTERFACE
        subsetix_geometry
)

target_include_directories(subsetix_field
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)

add_library(subsetix_multilevel INTERFACE)
add_library(subsetix::multilevel ALIAS subsetix_multilevel)

target_link_libraries(subsetix_multilevel
    INTERFACE
        subsetix_field
)

target_include_directories(subsetix_multilevel
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)

add_library(subsetix_vtk INTERFACE)
add_library(subsetix::vtk ALIAS subsetix_vtk)

target_link_libraries(subsetix_vtk
    INTERFACE
        subsetix_field
        subsetix_multilevel
)

target_include_directories(subsetix_vtk
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)

add_library(subsetix_core INTERFACE)
add_library(subsetix::core ALIAS subsetix_core)

target_link_libraries(subsetix_core
    INTERFACE
        subsetix_geometry
        subsetix_field
        subsetix_multilevel
        subsetix_vtk
)

target_include_directories(subsetix_core
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)
