#pragma once

#include <stdexcept>

#include <subsetix/field/csr_field.hpp>
#include <subsetix/csr_ops/field_mapping.hpp>
#include <subsetix/csr_ops/field_stencil.hpp>
#include <subsetix/csr_ops/field_amr.hpp>
#include <subsetix/csr_ops/field_subset.hpp>
#include <subsetix/csr_ops/workspace.hpp>

namespace subsetix {
namespace csr {

namespace detail {

template <typename T>
inline bool subview_region_empty(const Field2DSubViewDevice<T>& sub) {
  return sub.region.num_rows == 0 || sub.region.num_intervals == 0;
}

template <typename T>
inline void ensure_subview_subset(Field2DSubViewDevice<T>& sub,
                                  CsrSetAlgebraContext* ctx) {
  if (sub.has_subset() || !sub.valid() || !ctx) return;
  build_interval_subset_device(sub.parent.geometry, sub.region, sub.subset, ctx);
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

template <typename T, typename SubsetOp, typename SetOp>
inline void dispatch_subview(Field2DSubViewDevice<T>& sub,
                             CsrSetAlgebraContext* ctx,
                             SubsetOp&& on_subset, SetOp&& on_set) {
  if (subview_region_empty(sub)) return;
  if (ctx) ensure_subview_subset(sub, ctx);

  if (sub.has_subset()) { on_subset(); }
  else { on_set(); }
}

} // namespace detail

/** @brief Apply a functor on all cells of a subview. */
template <typename T, class Functor>
inline void apply_on_subview_device(Field2DSubViewDevice<T>& sub,
                                    Functor func,
                                    CsrSetAlgebraContext* ctx = nullptr) {
  detail::dispatch_subview(sub, ctx,
    [&]{ apply_on_subset_device(sub.parent, sub.subset, func); },
    [&]{ apply_on_set_device(sub.parent, sub.region, func); });
}

/** @brief Fill a subview with a constant value. */
template <typename T>
inline void fill_subview_device(Field2DSubViewDevice<T>& sub,
                                const T& value,
                                CsrSetAlgebraContext* ctx = nullptr) {
  detail::dispatch_subview(sub, ctx,
    [&]{ fill_on_subset_device(sub.parent, sub.subset, value); },
    [&]{ fill_on_set_device(sub.parent, sub.region, value); });
}

/** @brief Scale all values inside a subview. */
template <typename T>
inline void scale_subview_device(Field2DSubViewDevice<T>& sub,
                                 const T& alpha,
                                 CsrSetAlgebraContext* ctx = nullptr) {
  detail::dispatch_subview(sub, ctx,
    [&]{ scale_on_subset_device(sub.parent, sub.subset, alpha); },
    [&]{ scale_on_set_device(sub.parent, sub.region, alpha); });
}

/** @brief Copy values from one subview to another sharing the same region. */
template <typename T>
inline void copy_subview_device(Field2DSubViewDevice<T>& dst,
                                const Field2DSubViewDevice<T>& src,
                                CsrSetAlgebraContext* ctx = nullptr) {
  detail::ensure_matching_regions(dst, src);
  detail::dispatch_subview(dst, ctx,
    [&]{ copy_on_subset_device(dst.parent, src.parent, dst.subset); },
    [&]{ copy_on_set_device(dst.parent, src.parent, dst.region); });
}

/** @brief Apply a stencil functor from in-subview to out-subview. */
template <typename T, class StencilFunctor>
inline void apply_stencil_on_subview_device(Field2DSubViewDevice<T>& dst,
                                            const Field2DSubViewDevice<T>& src,
                                            StencilFunctor stencil,
                                            CsrSetAlgebraContext* ctx = nullptr) {
  detail::ensure_matching_regions(dst, src);
  detail::dispatch_subview(dst, ctx,
    [&]{ apply_stencil_on_subset_device(dst.parent, src.parent, dst.subset, stencil); },
    [&]{ apply_stencil_on_set_device(dst.parent, src.parent, dst.region, stencil); });
}

/** @brief Apply a CSR-friendly stencil functor on a subview region. */
template <typename OutT, typename InT, class StencilFunctor>
inline void apply_csr_stencil_on_subview_device(Field2DSubViewDevice<OutT>& dst,
                                                const Field2DSubViewDevice<InT>& src,
                                                StencilFunctor stencil,
                                                bool strict_check = true) {
  detail::ensure_matching_regions(dst, src);
  if (detail::subview_region_empty(dst)) return;
  apply_csr_stencil_on_set_device(dst.parent, src.parent, dst.region, stencil, strict_check);
}

/** @brief Restrict fine field values onto a coarse subview. */
template <typename T>
inline void restrict_field_subview_device(Field2DSubViewDevice<T>& sub,
                                          const Field2DDevice<T>& fine,
                                          CsrSetAlgebraContext* ctx = nullptr) {
  detail::dispatch_subview(sub, ctx,
    [&]{ restrict_field_on_subset_device(sub.parent, fine, sub.subset); },
    [&]{ restrict_field_on_set_device(sub.parent, fine, sub.region); });
}

/** @brief Prolong coarse field values onto a fine subview (injection). */
template <typename T>
inline void prolong_field_subview_device(Field2DSubViewDevice<T>& sub,
                                         const Field2DDevice<T>& coarse,
                                         CsrSetAlgebraContext* ctx = nullptr) {
  detail::dispatch_subview(sub, ctx,
    [&]{ prolong_field_on_subset_device(sub.parent, coarse, sub.subset); },
    [&]{ prolong_field_on_set_device(sub.parent, coarse, sub.region); });
}

/** @brief Prolong coarse field values onto a fine subview using prediction. */
template <typename T>
inline void prolong_field_prediction_subview_device(Field2DSubViewDevice<T>& sub,
                                                    const Field2DDevice<T>& coarse,
                                                    CsrSetAlgebraContext* ctx = nullptr) {
  detail::dispatch_subview(sub, ctx,
    [&]{ prolong_field_prediction_on_subset_device(sub.parent, coarse, sub.subset); },
    [&]{ prolong_field_prediction_device(sub.parent, coarse, sub.region); });
}

} // namespace csr
} // namespace subsetix
