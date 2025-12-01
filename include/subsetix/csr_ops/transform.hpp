#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/csr_ops/workspace.hpp>

namespace subsetix {
namespace csr {
namespace detail {

/**
 * @brief Apply a row-key transform on device with workspace,
 *        preserving row_ptr and intervals.
 */
template <class Transform>
inline void
apply_row_key_transform_device(const IntervalSet2DDevice& in,
                               Transform transform,
                               IntervalSet2DDevice& out,
                               UnifiedCsrWorkspace& /* workspace */) {
  out.num_rows = in.num_rows;
  out.num_intervals = in.num_intervals;
  out.row_ptr = in.row_ptr;
  out.intervals = in.intervals;

  if (in.num_rows == 0) {
    out.row_keys = IntervalSet2DDevice::RowKeyView();
    return;
  }

  if (out.row_keys.extent(0) < in.num_rows) {
    out.row_keys = IntervalSet2DDevice::RowKeyView(
        "subsetix_transform_row_keys", in.num_rows);
  }

  auto in_sub = Kokkos::subview(in.row_keys, std::make_pair(std::size_t(0), in.num_rows));
  auto out_sub = Kokkos::subview(out.row_keys, std::make_pair(std::size_t(0), in.num_rows));

  Kokkos::Experimental::transform(ExecSpace(), in_sub, out_sub, transform);
  ExecSpace().fence();
}

/**
 * @brief Apply an interval transform on device with workspace,
 *        preserving row structure.
 */
template <class Transform>
inline void
apply_interval_transform_device(const IntervalSet2DDevice& in,
                                Transform transform,
                                IntervalSet2DDevice& out,
                                UnifiedCsrWorkspace& /* workspace */) {
  out.num_rows = in.num_rows;
  out.num_intervals = in.num_intervals;
  out.row_keys = in.row_keys;
  out.row_ptr = in.row_ptr;

  if (in.num_intervals == 0) {
    out.intervals = IntervalSet2DDevice::IntervalView();
    return;
  }

  if (out.intervals.extent(0) < in.num_intervals) {
    out.intervals = IntervalSet2DDevice::IntervalView(
        "subsetix_transform_intervals", in.num_intervals);
  }

  auto in_sub = Kokkos::subview(in.intervals, std::make_pair(std::size_t(0), in.num_intervals));
  auto out_sub = Kokkos::subview(out.intervals, std::make_pair(std::size_t(0), in.num_intervals));

  Kokkos::Experimental::transform(ExecSpace(), in_sub, out_sub, transform);
  ExecSpace().fence();
}

struct TranslateXTransform {
  Coord dx;

  KOKKOS_INLINE_FUNCTION
  Interval operator()(const Interval& iv) const {
    Interval out_iv = iv;
    out_iv.begin = static_cast<Coord>(out_iv.begin + dx);
    out_iv.end = static_cast<Coord>(out_iv.end + dx);
    return out_iv;
  }
};

struct TranslateYTransform {
  Coord dy;

  KOKKOS_INLINE_FUNCTION
  RowKey2D operator()(const RowKey2D& key) const {
    RowKey2D out_key = key;
    out_key.y = static_cast<Coord>(out_key.y + dy);
    return out_key;
  }
};

} // namespace detail

/**
 * @brief Translate all intervals of a CSR interval set by a constant
 *        offset along X.
 */
inline void
translate_x_device(const IntervalSet2DDevice& in,
                   Coord dx,
                   IntervalSet2DDevice& out,
                   CsrSetAlgebraContext& ctx) {
  if (dx == 0) {
    out = in;
    return;
  }
  detail::TranslateXTransform transform{dx};
  detail::apply_interval_transform_device(in, transform, out,
                                          ctx.workspace);
}

/**
 * @brief Translate all rows of a CSR interval set by a constant offset
 *        along Y.
 */
inline void
translate_y_device(const IntervalSet2DDevice& in,
                   Coord dy,
                   IntervalSet2DDevice& out,
                   CsrSetAlgebraContext& ctx) {
  if (dy == 0 || in.num_rows == 0) {
    out = in;
    return;
  }
  detail::TranslateYTransform transform{dy};
  detail::apply_row_key_transform_device(in, transform, out,
                                         ctx.workspace);
}

} // namespace csr
} // namespace subsetix
