#include <Kokkos_Core.hpp>

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_set_ops.hpp>
#include <subsetix/vtk_export.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  {
    using namespace subsetix::csr;
    using subsetix::vtk::write_legacy_quads;

    // Base shapes: rectangle (box) and disk on the same domain.
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

    // Set operations on device.
    auto u = set_union_device(box_dev, disk_dev);
    auto i = set_intersection_device(box_dev, disk_dev);
    auto d = set_difference_device(box_dev, disk_dev); // box \ disk

    // Export results to VTK for visualization.
    auto u_host = build_host_from_device(u);
    auto i_host = build_host_from_device(i);
    auto d_host = build_host_from_device(d);

    write_legacy_quads(u_host, "box_disk_union.vtk");
    write_legacy_quads(i_host, "box_disk_intersection.vtk");
    write_legacy_quads(d_host, "box_disk_difference.vtk");
  }

  Kokkos::finalize();
  return 0;
}

