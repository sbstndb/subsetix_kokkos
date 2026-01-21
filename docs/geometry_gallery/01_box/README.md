# Simple Rectangular Domain

Creates a basic rectangular computational domain using `build_box()`.

        This is the starting point for any simulation - a simple box filled with fluid cells.

## API Usage

```cpp
auto geom = Geometry2D::build_box(100, 50, 0.01f, 0.01f);

    // Build the CSR geometry
    auto fluid = geom.build();

    std::cout << "Domain: " << geom.nx() << "x" << geom.ny() << " cells" << std::endl;
    std::cout << "Fluid cells: " << fluid.num_rows << std::endl;

    // Export to VTK for visualization
```

## Result

![Simple Rectangular Domain](output/01_box.png)

*Figure: Simple Rectangular Domain*



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

    std::cout << "=== Example 01: Simple Box ===" << std::endl;

    
    
    
    auto geom = Geometry2D::build_box(100, 50, 0.01f, 0.01f);

    
    auto fluid = geom.build();

    std::cout << "Domain: " << geom.nx() << "x" << geom.ny() << " cells" << std::endl;
    std::cout << "Fluid cells: " << fluid.num_rows << std::endl;

    
    subsetix::vtk::write_legacy_quads(fluid, "01_box.vtk");

    std::cout << "Exported: 01_box.vtk" << std::endl;

    return 0;
}

```

</details>

---
