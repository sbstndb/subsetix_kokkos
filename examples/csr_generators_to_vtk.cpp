#include <Kokkos_Core.hpp>

#include "example_output.hpp"

#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>
#include <subsetix/field/csr_field.hpp>
#include <subsetix/field/csr_field_ops.hpp>
#include <subsetix/io/vtk_export.hpp>

#include <string_view>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  {
    using namespace subsetix::csr;
    using subsetix::vtk::write_legacy_quads;

    const auto output_dir =
        subsetix_examples::make_example_output_dir("csr_generators_to_vtk",
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

    auto box_dev = make_box_device(box);
    auto box_host = to<HostMemorySpace>(box_dev);
    write_legacy_quads(box_host, output_path("box.vtk"));

    auto box_field_host =
        make_field_like_geometry<float>(box_host, 1.0f);
    write_legacy_quads(box_field_host, output_path("box_field.vtk"),
                       "value");

    Disk2D disk;
    disk.cx = 32;
    disk.cy = 16;
    disk.radius = 12;

    auto disk_dev = make_disk_device(disk);
    auto disk_host = to<HostMemorySpace>(disk_dev);
    write_legacy_quads(disk_host, output_path("disk.vtk"));

    auto disk_field_host =
        make_field_like_geometry<float>(disk_host, 2.0f);
    write_legacy_quads(disk_field_host, output_path("disk_field.vtk"),
                       "value");

    Domain2D dom;
    dom.x_min = 0;
    dom.x_max = 128;
    dom.y_min = 0;
    dom.y_max = 64;

    auto rand_dev = make_random_device(dom, 0.2, 123456);
    auto rand_host = to<HostMemorySpace>(rand_dev);
    write_legacy_quads(rand_host, output_path("random.vtk"));

    auto rand_field_host =
        make_field_like_geometry<float>(rand_host, 3.0f);
    write_legacy_quads(rand_field_host, output_path("random_field.vtk"),
                       "value");

    Domain2D cb_dom = dom;
    auto cb_dev = make_checkerboard_device(cb_dom, 4);
    auto cb_host = to<HostMemorySpace>(cb_dev);
    write_legacy_quads(cb_host, output_path("checkerboard.vtk"));

    auto cb_field_host =
        make_field_like_geometry<float>(cb_host, 4.0f);
    write_legacy_quads(cb_field_host,
                       output_path("checkerboard_field.vtk"),
                       "value");
  }

  Kokkos::finalize();
  return 0;
}
