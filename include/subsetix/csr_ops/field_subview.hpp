#pragma once

#include <stdexcept>

#include <subsetix/csr_field.hpp>
#include <subsetix/csr_ops/field_core.hpp>
#include <subsetix/csr_ops/field_stencil.hpp>
#include <subsetix/csr_ops/field_amr.hpp>

namespace subsetix {
namespace csr {

namespace detail {

template <typename T>
inline bool subview_region_empty(const Field2DSubViewDevice<T>& sub) {
  return sub.region.num_rows == 0 || sub.region.num_intervals == 0;
}

template <typename T>
inline void ensure_matching_regions(const Field2DSubViewDevice<T>& a,
                                    const Field2DSubViewDevice<T>& b) {
#ifndef NDEBUG
  if (a.region.row_keys.data() != b.region.row_keys.data() ||
      a.region.row_ptr.data() != b.region.row_ptr.data() ||
      a.region.intervals.data() != b.region.intervals.data()) {
    throw std::runtime_error(
        "Field2DSubView regions must point to the same geometry");
  }
#else
  (void)a;
  (void)b;
#endif
}

} // namespace detail

/**
 * @brief Apply a functor on all cells of a subview.
 */
template <typename T, class Functor>
inline void apply_on_subview_device(Field2DSubViewDevice<T>& subfield,
                                    Functor func) {
  if (detail::subview_region_empty(subfield)) {
    return;
  }
  apply_on_set_device(subfield.parent, subfield.region, func);
}

/**
 * @brief Fill a subview with a constant value.
 */
template <typename T>
inline void fill_subview_device(Field2DSubViewDevice<T>& subfield,
                                const T& value) {
  if (detail::subview_region_empty(subfield)) {
    return;
  }
  fill_on_set_device(subfield.parent, subfield.region, value);
}

/**
 * @brief Scale all values inside a subview.
 */
template <typename T>
inline void scale_subview_device(Field2DSubViewDevice<T>& subfield,
                                 const T& alpha) {
  if (detail::subview_region_empty(subfield)) {
    return;
  }
  scale_on_set_device(subfield.parent, subfield.region, alpha);
}

/**
 * @brief Copy values from one subview to another subview sharing the same region.
 */
template <typename T>
inline void copy_subview_device(Field2DSubViewDevice<T>& dst,
                                const Field2DSubViewDevice<T>& src) {
  detail::ensure_matching_regions(dst, src);
  if (detail::subview_region_empty(dst)) {
    return;
  }
  copy_on_set_device(dst.parent, src.parent, dst.region);
}

/**
 * @brief Apply a stencil functor from in-subview to out-subview.
 */
template <typename T, class StencilFunctor>
inline void apply_stencil_on_subview_device(
    Field2DSubViewDevice<T>& dst,
    const Field2DSubViewDevice<T>& src,
    StencilFunctor stencil) {
  detail::ensure_matching_regions(dst, src);
  if (detail::subview_region_empty(dst)) {
    return;
  }
  apply_stencil_on_set_device(dst.parent, src.parent, dst.region, stencil);
}

/**
 * @brief Restrict fine field values onto a coarse subview.
 */
template <typename T>
inline void restrict_field_subview_device(
    Field2DSubViewDevice<T>& coarse_subview,
    const Field2DDevice<T>& fine_field) {
  if (detail::subview_region_empty(coarse_subview)) {
    return;
  }
  restrict_field_on_set_device(coarse_subview.parent, fine_field,
                               coarse_subview.region);
}

/**
 * @brief Prolong coarse field values onto a fine subview (injection).
 */
template <typename T>
inline void prolong_field_subview_device(
    Field2DSubViewDevice<T>& fine_subview,
    const Field2DDevice<T>& coarse_field) {
  if (detail::subview_region_empty(fine_subview)) {
    return;
  }
  prolong_field_on_set_device(fine_subview.parent, coarse_field,
                              fine_subview.region);
}

/**
 * @brief Prolong coarse field values onto a fine subview using prediction.
 */
template <typename T>
inline void prolong_field_prediction_subview_device(
    Field2DSubViewDevice<T>& fine_subview,
    const Field2DDevice<T>& coarse_field) {
  if (detail::subview_region_empty(fine_subview)) {
    return;
  }
  prolong_field_prediction_device(fine_subview.parent, coarse_field,
                                  fine_subview.region);
}

} // namespace csr
} // namespace subsetix
