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
        subsetix_examples::make_example_output_dir("csr_set_ops_to_vtk",
                                                   argc,
                                                   argv);
    const auto output_path = [&output_dir](std::string_view filename) {
      return subsetix_examples::output_file(output_dir, filename);
    };

    Box2D box;
    box.x_min = 0;
    box.x_max = 64;
    box.y_min = 0;
    box.y_max = 32;

    Disk2D disk;
    disk.cx = 32;
    disk.cy = 16;
    disk.radius = 12;

    auto box_dev = make_box_device(box);
    auto disk_dev = make_disk_device(disk);

    CsrSetAlgebraContext ctx;

    auto u = allocate_interval_set_device(box_dev.num_rows + disk_dev.num_rows,
                                          box_dev.num_intervals + disk_dev.num_intervals);
    set_union_device(box_dev, disk_dev, u, ctx);

    auto i = allocate_interval_set_device(std::min(box_dev.num_rows, disk_dev.num_rows),
                                          box_dev.num_intervals + disk_dev.num_intervals);
    set_intersection_device(box_dev, disk_dev, i, ctx);

    auto d = allocate_interval_set_device(box_dev.num_rows,
                                          box_dev.num_intervals + disk_dev.num_intervals);
    set_difference_device(box_dev, disk_dev, d, ctx);

    auto x = allocate_interval_set_device(box_dev.num_rows + disk_dev.num_rows,
                                          box_dev.num_intervals + disk_dev.num_intervals);
    set_symmetric_difference_device(box_dev, disk_dev, x, ctx);

    auto u_host = to<HostMemorySpace>(u);
    auto i_host = to<HostMemorySpace>(i);
    auto d_host = to<HostMemorySpace>(d);
    auto x_host = to<HostMemorySpace>(x);

    write_legacy_quads(u_host, output_path("box_disk_union.vtk"));
    write_legacy_quads(i_host, output_path("box_disk_intersection.vtk"));
    write_legacy_quads(d_host, output_path("box_disk_difference.vtk"));
    write_legacy_quads(x_host, output_path("box_disk_symmetric_difference.vtk"));
  }

  Kokkos::finalize();
  return 0;
}
