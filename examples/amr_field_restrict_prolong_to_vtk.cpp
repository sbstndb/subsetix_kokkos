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
        subsetix_examples::make_example_output_dir(
            "amr_field_restrict_prolong_to_vtk", argc, argv);
    const auto output_path = [&output_dir](std::string_view filename) {
      return subsetix_examples::output_file(output_dir, filename);
    };

    Box2D coarse_box;
    coarse_box.x_min = 0;
    coarse_box.x_max = 16;
    coarse_box.y_min = 0;
    coarse_box.y_max = 8;

    auto coarse_geom_dev = make_box_device(coarse_box);
    CsrSetAlgebraContext ctx;
    IntervalSet2DDevice fine_geom_dev;
    refine_level_up_device(coarse_geom_dev, fine_geom_dev, ctx);

    auto coarse_geom_host =
        build_host_from_device(coarse_geom_dev);
    auto fine_geom_host =
        build_host_from_device(fine_geom_dev);

    auto coarse_field_host =
        make_field_like_geometry<double>(
            coarse_geom_host, 0.0);
    auto fine_field_host =
        make_field_like_geometry<double>(
            fine_geom_host, 0.0);

    write_legacy_quads(coarse_field_host,
                       output_path("amr_coarse_geometry.vtk"),
                       "value");
    write_legacy_quads(fine_field_host,
                       output_path("amr_fine_geometry.vtk"),
                       "value");

    for (std::size_t row = 0;
         row < fine_field_host.row_keys.size(); ++row) {
      const Coord y = fine_field_host.row_keys[row].y;
      const std::size_t begin =
          fine_field_host.row_ptr[row];
      const std::size_t end =
          fine_field_host.row_ptr[row + 1];
      for (std::size_t k = begin; k < end; ++k) {
        const auto fi = fine_field_host.intervals[k];
        for (Coord x = fi.begin; x < fi.end; ++x) {
          const std::size_t offset =
              fi.value_offset +
              static_cast<std::size_t>(x - fi.begin);
          fine_field_host.values[offset] =
              static_cast<double>(x + 10 * y);
        }
      }
    }

    auto coarse_field_dev =
        build_device_field_from_host(coarse_field_host);
    auto fine_field_dev =
        build_device_field_from_host(fine_field_host);
    auto coarse_mask = coarse_geom_dev;

    restrict_field_on_set_device(coarse_field_dev,
                                 fine_field_dev,
                                 coarse_mask);

    auto restricted_host =
        build_host_field_from_device(coarse_field_dev);
    write_legacy_quads(restricted_host,
                       output_path("amr_coarse_restricted.vtk"),
                       "value");

    auto coarse_field2_host =
        make_field_like_geometry<double>(
            coarse_geom_host, 0.0);
    for (std::size_t row = 0;
         row < coarse_field2_host.row_keys.size();
         ++row) {
      const Coord y = coarse_field2_host.row_keys[row].y;
      const std::size_t begin =
          coarse_field2_host.row_ptr[row];
      const std::size_t end =
          coarse_field2_host.row_ptr[row + 1];
      for (std::size_t k = begin; k < end; ++k) {
        const auto fi = coarse_field2_host.intervals[k];
        for (Coord x = fi.begin; x < fi.end; ++x) {
          const std::size_t offset =
              fi.value_offset +
              static_cast<std::size_t>(x - fi.begin);
          coarse_field2_host.values[offset] =
              static_cast<double>(100 + 3 * x + 5 * y);
        }
      }
    }

    auto coarse_field2_dev =
        build_device_field_from_host(coarse_field2_host);
    auto fine_field2_host =
        make_field_like_geometry<double>(
            fine_geom_host, 0.0);
    auto fine_field2_dev =
        build_device_field_from_host(fine_field2_host);

    auto fine_mask = fine_geom_dev;
    prolong_field_on_set_device(fine_field2_dev,
                                coarse_field2_dev,
                                fine_mask);

    auto prolonged_host =
        build_host_field_from_device(fine_field2_dev);
    write_legacy_quads(coarse_field2_host,
                       output_path("amr_coarse_source.vtk"), "value");
    write_legacy_quads(prolonged_host,
                       output_path("amr_fine_prolonged.vtk"),
                       "value");
  }

  Kokkos::finalize();
  return 0;
}
