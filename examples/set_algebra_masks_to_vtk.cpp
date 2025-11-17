#include <Kokkos_Core.hpp>

#include "example_output.hpp"

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_field.hpp>
#include <subsetix/csr_field_ops.hpp>
#include <subsetix/csr_set_ops.hpp>
#include <subsetix/vtk_export.hpp>

#include <string_view>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  {
    using namespace subsetix::csr;
    using subsetix::vtk::write_legacy_quads;

    const auto output_dir =
        subsetix_examples::make_example_output_dir("set_algebra_masks_to_vtk",
                                                   argc,
                                                   argv);
    const auto output_path = [&output_dir](std::string_view filename) {
      return subsetix_examples::output_file(output_dir, filename);
    };

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

    CsrSetAlgebraContext ctx;

    auto u = allocate_union_output_buffer(box_dev, disk_dev);
    set_union_device(box_dev, disk_dev, u, ctx);

    auto inter = allocate_intersection_output_buffer(box_dev, disk_dev);
    set_intersection_device(box_dev, disk_dev, inter, ctx);

    auto diff = allocate_difference_output_buffer(box_dev, disk_dev);
    set_difference_device(box_dev, disk_dev, diff, ctx);

    fill_on_set_device(field_dev, diff, 1.0);
    fill_on_set_device(field_dev, inter, 2.0);

    auto scaled_union =
        allocate_difference_output_buffer(u, inter);
    set_difference_device(u, inter, scaled_union, ctx);
    scale_on_set_device(field_dev, scaled_union, 0.5);

    auto result_host =
        build_host_field_from_device(field_dev);
    write_legacy_quads(result_host,
                       output_path("set_algebra_masks_field.vtk"),
                       "value");

    auto u_host = build_host_from_device(u);
    auto inter_host = build_host_from_device(inter);
    auto diff_host = build_host_from_device(diff);
    write_legacy_quads(u_host,
                       output_path("set_algebra_union.vtk"));
    write_legacy_quads(inter_host,
                       output_path("set_algebra_intersection.vtk"));
    write_legacy_quads(diff_host,
                       output_path("set_algebra_difference.vtk"));
  }

  Kokkos::finalize();
  return 0;
}
