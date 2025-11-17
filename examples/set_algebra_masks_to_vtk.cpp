#include <Kokkos_Core.hpp>

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_field.hpp>
#include <subsetix/csr_field_ops.hpp>
#include <subsetix/csr_set_ops.hpp>
#include <subsetix/vtk_export.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  {
    using namespace subsetix::csr;
    using subsetix::vtk::write_legacy_quads;

    Box2D domain;
    domain.x_min = 0;
    domain.x_max = 64;
    domain.y_min = 0;
    domain.y_max = 32;

    auto geom_dev = make_box_device(domain);
    auto geom_host = build_host_from_device(geom_dev);

    auto field_host =
        make_field_like_geometry<double>(geom_host, 0.0);
    auto field_dev =
        build_device_field_from_host(field_host);

    Box2D box;
    box.x_min = 0;
    box.x_max = 64;
    box.y_min = 0;
    box.y_max = 32;
    auto box_dev = make_box_device(box);

    Disk2D disk;
    disk.cx = 32;
    disk.cy = 16;
    disk.radius = 10;
    auto disk_dev = make_disk_device(disk);

    auto u = set_union_device(box_dev, disk_dev);
    auto inter = set_intersection_device(box_dev, disk_dev);
    auto diff = set_difference_device(box_dev, disk_dev);

    fill_on_set_device(field_dev, diff, 1.0);
    fill_on_set_device(field_dev, inter, 2.0);

    auto scaled_union =
        set_difference_device(u, inter);
    scale_on_set_device(field_dev, scaled_union, 0.5);

    auto result_host =
        build_host_field_from_device(field_dev);
    write_legacy_quads(result_host,
                       "set_algebra_masks_field.vtk",
                       "value");

    auto u_host = build_host_from_device(u);
    auto inter_host = build_host_from_device(inter);
    auto diff_host = build_host_from_device(diff);
    write_legacy_quads(u_host,
                       "set_algebra_union.vtk");
    write_legacy_quads(inter_host,
                       "set_algebra_intersection.vtk");
    write_legacy_quads(diff_host,
                       "set_algebra_difference.vtk");
  }

  Kokkos::finalize();
  return 0;
}

