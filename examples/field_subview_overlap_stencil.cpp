#include <Kokkos_Core.hpp>

#include "example_output.hpp"

#include <cmath>
#include <string_view>

#include <subsetix/field.hpp>
#include <subsetix/geometry.hpp>
#include <subsetix/csr_ops/workspace.hpp>
#include <subsetix/io.hpp>

namespace {

using namespace subsetix::csr;

struct FivePointAverage {
  KOKKOS_INLINE_FUNCTION
  double operator()(Coord x, Coord /*y*/, std::size_t idx,
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

IntervalSet2DHost split_geometry(const IntervalSet2DHost& base) {
  IntervalSet2DHost out;
  out.row_keys = base.row_keys;
  out.row_ptr.resize(base.row_ptr.size());
  out.intervals.clear();

  for (std::size_t row = 0; row < base.row_keys.size(); ++row) {
    const std::size_t begin = base.row_ptr[row];
    const std::size_t end = base.row_ptr[row + 1];
    out.row_ptr[row] = out.intervals.size();
    for (std::size_t k = begin; k < end; ++k) {
      const Interval iv = base.intervals[k];
      const Coord length =
          static_cast<Coord>(iv.end - iv.begin);
      if (length <= 1) {
        out.intervals.push_back(iv);
        continue;
      }
      const Coord mid = iv.begin + length / 2;
      out.intervals.push_back(Interval{iv.begin, mid});
      out.intervals.push_back(Interval{mid, iv.end});
    }
    out.row_ptr[row + 1] = out.intervals.size();
  }

  out.rebuild_mapping();
  return out;
}

void fill_gaussian_ridge(IntervalField2DHost<double>& field) {
  const double cx = 32.0;
  const double cy = 16.0;
  for (std::size_t row = 0; row < field.row_keys.size(); ++row) {
    const Coord y = field.row_keys[row].y;
    const std::size_t begin = field.row_ptr[row];
    const std::size_t end = field.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = field.intervals[k];
      for (Coord x = fi.begin; x < fi.end; ++x) {
        const std::size_t off =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        const double dx = static_cast<double>(x) - cx;
        const double dy = static_cast<double>(y) - cy;
        const double r2 = dx * dx + dy * dy;
        const double ridge =
            std::sin(static_cast<double>(x) * 0.1) *
            std::cos(static_cast<double>(y) * 0.15);
        field.values[off] =
            std::exp(-r2 / 400.0) + 0.25 * ridge;
      }
    }
  }
}

} // namespace

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  {
    using subsetix::vtk::write_legacy_quads;

    const auto output_dir =
        subsetix_examples::make_example_output_dir(
            "field_subview_overlap_stencil", argc, argv);
    const auto output_path = [&output_dir](std::string_view name) {
      return subsetix_examples::output_file(output_dir, name);
    };

    Box2D domain{0, 64, 0, 32};
    auto dense_geom_dev = make_box_device(domain);
    auto dense_geom_host = build_host_from_device(dense_geom_dev);
    auto split_geom_host = split_geometry(dense_geom_host);
    auto split_geom_dev = build_device_from_host(split_geom_host);

    auto source_host =
        make_field_like_geometry<double>(dense_geom_host, 0.0);
    fill_gaussian_ridge(source_host);
    auto sink_host =
        make_field_like_geometry<double>(split_geom_host, 0.0);

    auto source_field = build_device_field_from_host(source_host);
    auto sink_field = build_device_field_from_host(sink_host);

    Box2D interior = domain;
    interior.x_min = 8;
    interior.x_max = 56;
    interior.y_min = 6;
    interior.y_max = 26;
    auto overlap_mask = make_box_device(interior);

    auto src_view =
        make_subview(source_field, overlap_mask, "overlap_src");
    auto dst_view =
        make_subview(sink_field, overlap_mask, "overlap_dst");

    CsrSetAlgebraContext ctx;
    apply_stencil_on_subview_device(dst_view, src_view,
                                    FivePointAverage{}, ctx);

    auto src_result = build_host_field_from_device(source_field);
    auto dst_result = build_host_field_from_device(sink_field);
    write_legacy_quads(src_result,
                       output_path("overlap_source.vtk"),
                       "source");
    write_legacy_quads(dst_result,
                       output_path("overlap_smoothed.vtk"),
                       "smoothed");
  }

  Kokkos::finalize();
  return 0;
}
