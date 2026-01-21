# Multiple Cylinder Obstacles

Shows adding multiple circular obstacles in a staggered pattern.

        This configuration could represent flow through a tube bundle or heat exchanger,
        where fluid flows around an array of cylinders.

## API Usage

```cpp
auto geom = Geometry2D::build_box(200, 100, 0.005f, 0.005f);

    // Add multiple cylinders in a staggered pattern
    // Row 1
    geom.add_cylinder(0.3f, 0.3f, 0.05f, true);
    geom.add_cylinder(0.5f, 0.3f, 0.05f, true);
    geom.add_cylinder(0.7f, 0.3f, 0.05f, true);

    // Row 2 (offset)
    geom.add_cylinder(0.4f, 0.5f, 0.05f, true);
    geom.add_cylinder(0.6f, 0.5f, 0.05f, true);

    // Row 3
    geom.add_cylinder(0.3f, 0.7f, 0.05f, true);
    geom.add_cylinder(0.5f, 0.7f, 0.05f, true);
    geom.add_cylinder(0.7f, 0.7f, 0.05f, true);

    // Build the CSR geometry
    auto fluid = geom.build();

    std::cout << "Domain: " << geom.nx() << "x" << geom.ny() << " cells" << std::endl;
    std::cout << "Cylinders: 8 (staggered pattern)" << std::endl;
    std::cout << "Fluid cells: " << fluid.num_rows << std::endl;

    // Export to VTK for visualization
```

## Result

![Multiple Cylinder Obstacles](output/03_multiple_cylinders.png)

*Figure: Multiple Cylinder Obstacles*



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

    std::cout << "=== Example 03: Multiple Cylinders ===" << std::endl;

    
    auto geom = Geometry2D::build_box(200, 100, 0.005f, 0.005f);

    
    
    geom.add_cylinder(0.3f, 0.3f, 0.05f, true);
    geom.add_cylinder(0.5f, 0.3f, 0.05f, true);
    geom.add_cylinder(0.7f, 0.3f, 0.05f, true);

    
    geom.add_cylinder(0.4f, 0.5f, 0.05f, true);
    geom.add_cylinder(0.6f, 0.5f, 0.05f, true);

    
    geom.add_cylinder(0.3f, 0.7f, 0.05f, true);
    geom.add_cylinder(0.5f, 0.7f, 0.05f, true);
    geom.add_cylinder(0.7f, 0.7f, 0.05f, true);

    
    auto fluid = geom.build();

    std::cout << "Domain: " << geom.nx() << "x" << geom.ny() << " cells" << std::endl;
    std::cout << "Cylinders: 8 (staggered pattern)" << std::endl;
    std::cout << "Fluid cells: " << fluid.num_rows << std::endl;

    
    subsetix::vtk::write_legacy_quads(fluid, "03_multiple_cylinders.vtk");

    std::cout << "Exported: 03_multiple_cylinders.vtk" << std::endl;

    return 0;
}

```

</details>

---
