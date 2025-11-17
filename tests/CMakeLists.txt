add_executable(kokkos_smoke_test
    kokkos_smoke_test.cpp
)

target_link_libraries(kokkos_smoke_test
    PRIVATE
        subsetix_core
)

add_test(NAME KokkosSmokeTest
         COMMAND kokkos_smoke_test)

