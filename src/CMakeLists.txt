add_library(subsetix_core INTERFACE)

target_link_libraries(subsetix_core
    INTERFACE
        Kokkos::kokkos
)

