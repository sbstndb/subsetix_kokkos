#include <Kokkos_Core.hpp>

#include "example_output.hpp"

#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>
#include <subsetix/io/vtk_export.hpp>

#include <string_view>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  {
    using namespace subsetix::csr;
    using subsetix::vtk::write_legacy_quads;

    const auto output_dir =
        subsetix_examples::make_example_output_dir("csr_morphology_to_vtk",
                                                   argc,
                                                   argv);
    const auto output_path = [&output_dir](std::string_view filename) {
      return subsetix_examples::output_file(output_dir, filename);
    };

    // Base shape: A disk with a hole (difference of two disks)
    Disk2D disk_outer{50, 50, 30};
    Disk2D disk_inner{50, 50, 10};
    
    auto d_outer = make_disk_device(disk_outer);
    auto d_inner = make_disk_device(disk_inner);
    
    CsrSetAlgebraContext ctx;
    
    IntervalSet2DDevice shape =
        allocate_interval_set_device(d_outer.num_rows, d_outer.num_intervals + d_inner.num_intervals);
    set_difference_device(d_outer, d_inner, shape, ctx);
    
    auto h_shape = build_host_from_device(shape);
    write_legacy_quads(h_shape, output_path("original_shape.vtk"));

    // Expand (Dilation)
    // Allocate a buffer large enough. A simple heuristic is to add 2*radius to dims.
    // Or just use a large buffer for this example.
    auto expanded = allocate_interval_set_device(200, 1000);
    expand_device(shape, 2, 2, expanded, ctx);
    
    auto h_expanded = build_host_from_device(expanded);
    write_legacy_quads(h_expanded, output_path("expanded_r2.vtk"));
    
    // Shrink (Erosion)
    auto shrunk = allocate_interval_set_device(200, 1000);
    shrink_device(shape, 2, 2, shrunk, ctx);
    
    auto h_shrunk = build_host_from_device(shrunk);
    write_legacy_quads(h_shrunk, output_path("shrunk_r2.vtk"));
    
    // Opening (Shrink then Expand) - removes small noise/objects
    // We'll start with a noisy shape (random + shape) to demonstrate.
    // For simplicity here, just show opening on the clean shape.
    auto opened = allocate_interval_set_device(200, 1000);
    // Reuse shrunk as input to expand
    expand_device(shrunk, 2, 2, opened, ctx);
    
    auto h_opened = build_host_from_device(opened);
    write_legacy_quads(h_opened, output_path("opened_r2.vtk"));
    
    // Closing (Expand then Shrink) - fills small holes
    auto closed = allocate_interval_set_device(200, 1000);
    // Reuse expanded as input to shrink
    shrink_device(expanded, 2, 2, closed, ctx);
    
    auto h_closed = build_host_from_device(closed);
    write_legacy_quads(h_closed, output_path("closed_r2.vtk"));
  }

  Kokkos::finalize();
  return 0;
}
