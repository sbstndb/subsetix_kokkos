/**
 * @file 02_box_with_cylinder.cpp
 * @brief Example 02: Box with cylinder obstacle
 *
 * Demonstrates adding a circular obstacle to a rectangular domain.
 * This is the classic "flow around cylinder" CFD test case.
 */

#include <Kokkos_Core.hpp>
#include <subsetix/fvd/geometry/geometry_builder.hpp>
#include <subsetix/io/vtk_export.hpp>
#include <iostream>

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);

    using Real = float;
    using Geometry2D = subsetix::fvd::Geometry2D<Real>;

    std::cout << "=== Example 02: Box with Cylinder ===" << std::endl;

    // Create a rectangular domain
    // 200 cells in x, 100 cells in y
    // Cell size: 5mm x 5mm
    auto geom = Geometry2D::build_box(200, 100, 0.005f, 0.005f);

    // Add a cylinder obstacle (in physical coordinates)
    // Center at (0.5, 0.5) meters
    // Radius: 0.1 meters (10cm)
    // is_obstacle = true (solid, removed from fluid domain)
    geom.add_cylinder(0.5f, 0.5f, 0.1f, true);

    // Build the CSR geometry
    auto fluid = geom.build();

    std::cout << "Domain: " << geom.nx() << "x" << geom.ny() << " cells" << std::endl;
    std::cout << "Cylinder: center=(0.5, 0.5), radius=0.1m" << std::endl;
    std::cout << "Fluid cells: " << fluid.num_rows << std::endl;

    // Export to VTK for visualization
    subsetix::vtk::write_legacy_quads(fluid, "02_box_with_cylinder.vtk");

    std::cout << "Exported: 02_box_with_cylinder.vtk" << std::endl;

    return 0;
}
