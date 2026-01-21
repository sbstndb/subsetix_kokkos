/**
 * @file 04_complex_geometry.cpp
 * @brief Example 04: Complex geometry with multiple shapes
 *
 * Demonstrates combining different primitive types:
 * - Rectangular obstacles
 * - Circular obstacles
 * - Mixed patterns
 */

#include <Kokkos_Core.hpp>
#include <subsetix/fvd/geometry/geometry_builder.hpp>
#include <subsetix/io/vtk_export.hpp>
#include <iostream>

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);

    using Real = float;
    using Geometry2D = subsetix::fvd::Geometry2D<Real>;

    std::cout << "=== Example 04: Complex Geometry ===" << std::endl;

    // Create a rectangular domain
    auto geom = Geometry2D::build_box(200, 100, 0.005f, 0.005f);

    // Add a large rectangular obstacle (wall with gap)
    geom.add_rectangle(0.4f, 0.45f, 0.2f, 0.8f, true);

    // Add cylinders around the gap
    geom.add_cylinder(0.4f, 0.2f, 0.04f, true);
    geom.add_cylinder(0.4f, 0.8f, 0.04f, true);
    geom.add_cylinder(0.45f, 0.2f, 0.04f, true);
    geom.add_cylinder(0.45f, 0.8f, 0.04f, true);

    // Add inlet channel constriction
    geom.add_rectangle(0.0f, 0.1f, 0.3f, 0.7f, true);
    geom.add_rectangle(0.0f, 0.1f, 0.3f, 0.35f, false);  // fluid region
    geom.add_rectangle(0.0f, 0.1f, 0.65f, 0.7f, false); // fluid region

    // Build the CSR geometry
    auto fluid = geom.build();

    std::cout << "Domain: " << geom.nx() << "x" << geom.ny() << " cells" << std::endl;
    std::cout << "Obstacles: 1 wall + 4 cylinders + inlet constriction" << std::endl;
    std::cout << "Fluid cells: " << fluid.num_rows << std::endl;

    // Export to VTK for visualization
    subsetix::vtk::write_legacy_quads(fluid, "04_complex_geometry.vtk");

    std::cout << "Exported: 04_complex_geometry.vtk" << std::endl;

    return 0;
}
