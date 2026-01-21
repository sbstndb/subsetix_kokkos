# Box with Cylinder Obstacle

Demonstrates adding a circular obstacle using `add_cylinder()`.

        This is the classic "flow around cylinder" test case, widely used in CFD for
        validation of solvers. The cylinder is removed from the fluid domain.

## API Usage

```cpp
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
```

## Result

![Box with Cylinder Obstacle](output/02_box_with_cylinder.png)

*Figure: Box with Cylinder Obstacle*



## Full Example Code

<details>
<summary>Click to expand full source code</summary>

```cpp


#include <Kokkos_Core.hpp>
#include <subsetix/fvd/geometry/geometry_builder.hpp>
#include <subsetix/io/vtk_export.hpp>
#include <iostream>

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);

    using Real = float;
    using Geometry2D = subsetix::fvd::Geometry2D<Real>;

    std::cout << "=== Example 02: Box with Cylinder ===" << std::endl;

    
    
    
    auto geom = Geometry2D::build_box(200, 100, 0.005f, 0.005f);

    
    
    
    
    geom.add_cylinder(0.5f, 0.5f, 0.1f, true);

    
    auto fluid = geom.build();

    std::cout << "Domain: " << geom.nx() << "x" << geom.ny() << " cells" << std::endl;
    std::cout << "Cylinder: center=(0.5, 0.5), radius=0.1m" << std::endl;
    std::cout << "Fluid cells: " << fluid.num_rows << std::endl;

    
    subsetix::vtk::write_legacy_quads(fluid, "02_box_with_cylinder.vtk");

    std::cout << "Exported: 02_box_with_cylinder.vtk" << std::endl;

    return 0;
}

```

</details>

---
