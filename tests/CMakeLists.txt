add_executable(kokkos_smoke_test
    kokkos_smoke_test.cpp
)

target_link_libraries(kokkos_smoke_test
    PRIVATE
        subsetix_core
)

add_test(NAME KokkosSmokeTest
         COMMAND kokkos_smoke_test)

add_executable(csr_interval_set_smoke_test
    csr_interval_set_smoke_test.cpp
)

target_link_libraries(csr_interval_set_smoke_test
    PRIVATE
        subsetix_core
)

add_test(NAME CSRIntervalSetSmokeTest
         COMMAND csr_interval_set_smoke_test)

add_executable(csr_builders_smoke_test
    csr_builders_smoke_test.cpp
)

target_link_libraries(csr_builders_smoke_test
    PRIVATE
        subsetix_core
)

add_test(NAME CSRBuildersSmokeTest
         COMMAND csr_builders_smoke_test)

add_executable(csr_field_smoke_test
    csr_field_smoke_test.cpp
)

target_link_libraries(csr_field_smoke_test
    PRIVATE
        subsetix_core
)

add_test(NAME CSRFieldSmokeTest
         COMMAND csr_field_smoke_test)

add_executable(csr_union_smoke_test
    csr_union_smoke_test.cpp
)

target_link_libraries(csr_union_smoke_test
    PRIVATE
        subsetix_core
)

add_test(NAME CSRUnionSmokeTest
         COMMAND csr_union_smoke_test)

add_executable(csr_union_components_smoke_test
    csr_union_components_smoke_test.cpp
)

target_link_libraries(csr_union_components_smoke_test
    PRIVATE
        subsetix_core
)

add_test(NAME CSRUnionComponentsSmokeTest
         COMMAND csr_union_components_smoke_test)
