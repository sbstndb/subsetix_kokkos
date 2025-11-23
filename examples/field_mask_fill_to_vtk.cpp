#include <Kokkos_Core.hpp>

#include "example_output.hpp"

#include <subsetix/geometry.hpp>
#include <subsetix/field.hpp>
#include <subsetix/io.hpp>

#include <algorithm>
#include <string_view>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  {
    using namespace subsetix::csr;
    using subsetix::vtk::write_legacy_quads;

    const auto output_dir =
        subsetix_examples::make_example_output_dir("field_mask_fill_to_vtk",
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

    Disk2D disk;
    disk.cx = 32;
    disk.cy = 16;
    disk.radius = 10;
    auto disk_mask = make_disk_device(disk);

    Domain2D rnd_dom;
    rnd_dom.x_min = domain.x_min;
    rnd_dom.x_max = domain.x_max;
    rnd_dom.y_min = domain.y_min;
    rnd_dom.y_max = domain.y_max;
    auto random_mask =
        make_random_device(rnd_dom, 0.15, 123456);

    auto disk_view = make_subview(field_dev, disk_mask, "disk_mask");
    auto random_view = make_subview(field_dev, random_mask, "random_mask");

    fill_subview_device(disk_view, 1.0);
    scale_subview_device(random_view, 2.0);

    auto field_modified_host =
        build_host_field_from_device(field_dev);
    write_legacy_quads(field_modified_host,
                       output_path("field_mask_fill.vtk"), "value");

    auto field_copy_host =
        make_field_like_geometry<double>(geom_host, 0.0);
    auto field_copy_dev =
        build_device_field_from_host(field_copy_host);

    auto copy_src = make_subview(field_dev, disk_mask, "copy_src");
    auto copy_dst = make_subview(field_copy_dev, disk_mask, "copy_dst");
    copy_subview_device(copy_dst, copy_src);

    auto field_copy_result_host =
        build_host_field_from_device(field_copy_dev);
    write_legacy_quads(field_copy_result_host,
                       output_path("field_mask_fill_copy.vtk"),
                       "value");
  }

  Kokkos::finalize();
  return 0;
}
