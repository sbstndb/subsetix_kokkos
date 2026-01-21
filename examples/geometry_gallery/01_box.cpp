/**
 * @file 01_box.cpp
 * @brief Example 01: Simple rectangular domain
 *
 * Demonstrates creating a basic rectangular computational domain.
 */

#include <Kokkos_Core.hpp>
#include <subsetix/fvd/geometry/geometry_builder.hpp>
#include <subsetix/io/vtk_export.hpp>
#include <iostream>

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);

    using Real = float;
    using Geometry2D = subsetix::fvd::Geometry2D<Real>;

    std::cout << "=== Example 01: Simple Box ===" << std::endl;

    // Create a rectangular domain
    // 100 cells in x, 50 cells in y
    // Cell size: 1cm x 1cm
    auto geom = Geometry2D::build_box(100, 50, 0.01f, 0.01f);

    // Build the CSR geometry
    auto fluid = geom.build();

    std::cout << "Domain: " << geom.nx() << "x" << geom.ny() << " cells" << std::endl;
    std::cout << "Fluid cells: " << fluid.num_rows << std::endl;

    // Export to VTK for visualization
    subsetix::vtk::write_legacy_quads(fluid, "01_box.vtk");

    std::cout << "Exported: 01_box.vtk" << std::endl;

    return 0;
}
