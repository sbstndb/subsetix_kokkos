#include <Kokkos_Core.hpp>

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/vtk_export.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  {
    using namespace subsetix::csr;
    using subsetix::vtk::write_legacy_quads;

    // Example 1: simple box (upscaled by 4x).
    Box2D box;
    box.x_min = 0;
    box.x_max = 64;
    box.y_min = 0;
    box.y_max = 32;

    auto box_dev = make_box_device(box);
    auto box_host = build_host_from_device(box_dev);
    write_legacy_quads(box_host, "box.vtk");

    // Example 2: disk centered in the box (upscaled by 4x).
    Disk2D disk;
    disk.cx = 32;
    disk.cy = 16;
    disk.radius = 12;

    auto disk_dev = make_disk_device(disk);
    auto disk_host = build_host_from_device(disk_dev);
    write_legacy_quads(disk_host, "disk.vtk");

    // Example 3: random geometry on a larger domain (upscaled by 4x).
    Domain2D dom;
    dom.x_min = 0;
    dom.x_max = 128;
    dom.y_min = 0;
    dom.y_max = 64;

    auto rand_dev = make_random_device(dom, 0.2, 123456);
    auto rand_host = build_host_from_device(rand_dev);
    write_legacy_quads(rand_host, "random.vtk");
  }

  Kokkos::finalize();
  return 0;
}
