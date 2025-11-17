#include <Kokkos_Core.hpp>

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_field.hpp>
#include <subsetix/csr_field_ops.hpp>
#include <subsetix/vtk_export.hpp>

namespace {

using namespace subsetix::csr;

struct FivePointAverage {
  KOKKOS_INLINE_FUNCTION
  double operator()(Coord x,
                    Coord y,
                    std::size_t idx,
                    int interval_index,
                    const detail::FieldStencilContext<double>& ctx) const {
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

    Box2D box;
    box.x_min = 0;
    box.x_max = 64;
    box.y_min = 0;
    box.y_max = 64;

    auto geom_dev = make_box_device(box);
    auto geom_host = build_host_from_device(geom_dev);

    auto field_host =
        make_field_like_geometry<double>(geom_host, 0.0);

    for (std::size_t row = 0;
         row < field_host.row_keys.size(); ++row) {
      const Coord y = field_host.row_keys[row].y;
      const std::size_t begin = field_host.row_ptr[row];
      const std::size_t end = field_host.row_ptr[row + 1];
      for (std::size_t k = begin; k < end; ++k) {
        const auto fi = field_host.intervals[k];
        for (Coord x = fi.begin; x < fi.end; ++x) {
          const std::size_t offset =
              fi.value_offset +
              static_cast<std::size_t>(x - fi.begin);
          const double cx = static_cast<double>(x - 32);
          const double cy = static_cast<double>(y - 32);
          const double r2 = cx * cx + cy * cy;
          field_host.values[offset] =
              (r2 < 8.0 * 8.0) ? 1.0 : 0.0;
        }
      }
    }

    write_legacy_quads(field_host,
                       "field_stencil_initial.vtk",
                       "value");

    auto field_curr_dev =
        build_device_field_from_host(field_host);
    auto field_next_dev =
        build_device_field_from_host(field_host);

    Box2D inner;
    inner.x_min = 1;
    inner.x_max = 63;
    inner.y_min = 1;
    inner.y_max = 63;
    auto mask_dev = make_box_device(inner);

    const int iterations = 10;
    for (int it = 0; it < iterations; ++it) {
      apply_stencil_on_set_device(field_next_dev,
                                  field_curr_dev,
                                  mask_dev,
                                  FivePointAverage{});
      std::swap(field_curr_dev, field_next_dev);
    }

    auto smoothed_host =
        build_host_field_from_device(field_curr_dev);
    write_legacy_quads(smoothed_host,
                       "field_stencil_smoothed.vtk",
                       "value");
  }

  Kokkos::finalize();
  return 0;
}

