add_library(subsetix_core INTERFACE)

target_link_libraries(subsetix_core
    INTERFACE
        Kokkos::kokkos
)

target_include_directories(subsetix_core
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)
