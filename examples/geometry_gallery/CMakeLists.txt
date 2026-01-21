# Geometry Gallery Examples
# Small executables demonstrating Geometry2D API

# ============================================================================
# Example 01: Simple Box
# ============================================================================
add_executable(01_box
    01_box.cpp
)

target_link_libraries(01_box
    PRIVATE
        subsetix::core
)
target_include_directories(01_box PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ============================================================================
# Example 02: Box with Cylinder
# ============================================================================
add_executable(02_box_with_cylinder
    02_box_with_cylinder.cpp
)

target_link_libraries(02_box_with_cylinder
    PRIVATE
        subsetix::core
)
target_include_directories(02_box_with_cylinder PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ============================================================================
# Example 03: Multiple Cylinders
# ============================================================================
add_executable(03_multiple_cylinders
    03_multiple_cylinders.cpp
)

target_link_libraries(03_multiple_cylinders
    PRIVATE
        subsetix::core
)
target_include_directories(03_multiple_cylinders PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ============================================================================
# Example 04: Complex Geometry
# ============================================================================
add_executable(04_complex_geometry
    04_complex_geometry.cpp
)

target_link_libraries(04_complex_geometry
    PRIVATE
        subsetix::core
)
target_include_directories(04_complex_geometry PRIVATE ${CMAKE_SOURCE_DIR}/include)
