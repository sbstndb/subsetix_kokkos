add_executable(kokkos_smoke_test
    kokkos_smoke_test.cpp
)

target_link_libraries(kokkos_smoke_test
    PRIVATE
        subsetix_core
)

add_test(NAME KokkosSmokeTest
         COMMAND kokkos_smoke_test)

add_executable(tiled_interval_set_smoke_test
    tiled_interval_set_smoke_test.cpp
)

target_link_libraries(tiled_interval_set_smoke_test
    PRIVATE
        subsetix_core
)

add_test(NAME TiledIntervalSetSmokeTest
         COMMAND tiled_interval_set_smoke_test)
