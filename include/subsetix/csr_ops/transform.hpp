#pragma once

#include <Kokkos_Core.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/csr_ops/workspace.hpp>

namespace subsetix {
namespace csr {
namespace detail {

template <class Transform>
struct RowKeyTransformFunctor {
  IntervalSet2DDevice::RowKeyView row_keys_out;
  IntervalSet2DDevice::RowKeyView row_keys_in;
  Transform transform;

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t i) const {
    row_keys_out(i) = transform(row_keys_in(i));
  }
};

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

  RowKeyTransformFunctor<Transform> functor;
  functor.row_keys_out = out.row_keys;
  functor.row_keys_in = in.row_keys;
  functor.transform = transform;

  Kokkos::parallel_for(
      "subsetix_csr_row_key_transform_kernel",
      Kokkos::RangePolicy<ExecSpace>(0, in.num_rows),
      functor);

  ExecSpace().fence();
}

template <class Transform>
struct IntervalTransformFunctor {
  IntervalSet2DDevice::IntervalView intervals_out;
  IntervalSet2DDevice::IntervalView intervals_in;
  Transform transform;

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t i) const {
    intervals_out(i) = transform(intervals_in(i));
  }
};

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

  IntervalTransformFunctor<Transform> functor;
  functor.intervals_out = out.intervals;
  functor.intervals_in = in.intervals;
  functor.transform = transform;

  Kokkos::parallel_for(
      "subsetix_csr_interval_transform_kernel",
      Kokkos::RangePolicy<ExecSpace>(0, in.num_intervals),
      functor);

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
