#include <Kokkos_Core.hpp>

#include "example_output.hpp"

#include <subsetix/csr_field.hpp>
#include <subsetix/csr_field_ops.hpp>
#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_ops/workspace.hpp>
#include <subsetix/vtk_export.hpp>

#include <string_view>
#include <vector>

namespace {

struct FivePointAverage {
  KOKKOS_INLINE_FUNCTION
  double operator()(subsetix::csr::Coord x,
                    subsetix::csr::Coord /*y*/,
                    std::size_t idx,
                    int interval_index,
                    const subsetix::csr::detail::FieldStencilContext<double>& ctx) const {
    const double center = ctx.center(idx);
    const double east = ctx.right(idx);
    const double west = ctx.left(idx);
    const double north = ctx.north(x, interval_index);
    const double south = ctx.south(x, interval_index);
    return (center + east + west + north + south) / 5.0;
  }
};

} // namespace

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  {
    using namespace subsetix::csr;
    using subsetix::vtk::write_legacy_quads;

    const auto output_dir =
        subsetix_examples::make_example_output_dir("field_subview_workflow",
                                                   argc,
                                                   argv);
    const auto output_path = [&output_dir](std::string_view filename) {
      return subsetix_examples::output_file(output_dir, filename);
    };

    Box2D domain{0, 128, 0, 64};
    auto geom_dev = make_box_device(domain);
    auto geom_host = build_host_from_device(geom_dev);

    auto field_host =
        make_field_like_geometry<double>(geom_host, 0.0);
    // Initialize host values with a smooth ramp.
    for (std::size_t row = 0; row < field_host.row_keys.size(); ++row) {
      const Coord y = field_host.row_keys[row].y;
      const std::size_t begin = field_host.row_ptr[row];
      const std::size_t end = field_host.row_ptr[row + 1];
      for (std::size_t k = begin; k < end; ++k) {
        const auto fi = field_host.intervals[k];
        for (Coord x = fi.begin; x < fi.end; ++x) {
          const std::size_t offset =
              fi.value_offset +
              static_cast<std::size_t>(x - fi.begin);
          field_host.values[offset] =
              static_cast<double>(x) * 0.02 +
              static_cast<double>(y) * 0.05;
        }
      }
    }

    auto field_dev =
        build_device_field_from_host(field_host);
    auto filtered_field_dev =
        build_device_field_from_host(field_host);

    // Build masks for the patch (disk) and the interior stencil zone.
    Disk2D patch;
    patch.cx = 64;
    patch.cy = 32;
    patch.radius = 14;
    auto patch_mask = make_disk_device(patch);

    Box2D interior = domain;
    interior.x_min += 2;
    interior.x_max -= 2;
    interior.y_min += 2;
    interior.y_max -= 2;
    auto interior_mask = make_box_device(interior);

    // Modify the patch via subview and keep a copy for filtered field.
    auto patch_view = make_subview(field_dev, patch_mask, "patch_values");
    auto patch_filtered_view =
        make_subview(filtered_field_dev, patch_mask, "patch_filtered");

    CsrSetAlgebraContext patch_ctx;
    fill_subview_device(patch_view, 5.0, patch_ctx);
    scale_subview_device(patch_view, 1.2);

    // Run a smoothing stencil on the interior region via subviews.
    auto interior_src =
        make_subview(field_dev, interior_mask, "interior_src");
    auto interior_dst =
        make_subview(filtered_field_dev, interior_mask, "interior_dst");

    CsrSetAlgebraContext interior_ctx;
    apply_stencil_on_subview_device(interior_dst,
                                    interior_src,
                                    FivePointAverage{},
                                    interior_ctx);

    // Restore the high-frequency patch inside the filtered field.
    copy_subview_device(patch_filtered_view, patch_view, patch_ctx);

    auto output_original =
        build_host_field_from_device(field_dev);
    auto output_filtered =
        build_host_field_from_device(filtered_field_dev);

    write_legacy_quads(output_original,
                       output_path("field_subview_original.vtk"),
                       "value");
    write_legacy_quads(output_filtered,
                       output_path("field_subview_filtered.vtk"),
                       "value");
  }

  Kokkos::finalize();
  return 0;
}
