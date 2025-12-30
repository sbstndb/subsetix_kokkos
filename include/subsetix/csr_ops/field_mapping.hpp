#pragma once

#include <cstddef>

#include <Kokkos_Core.hpp>

#include <subsetix/field/csr_field.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/detail/csr_utils.hpp>

namespace subsetix {
namespace csr {

namespace detail {

// ---------------------------------------------------------------------------
// Helpers to map a mask IntervalSet to a field geometry
// (Used by field_stencil.hpp and field_amr.hpp for pre-computed mappings)
// ---------------------------------------------------------------------------

template <typename T>
using MaskFieldMapping = FieldMaskMapping;

template <typename T>
inline Kokkos::View<int*, DeviceMemorySpace>
build_mask_row_to_field_row_mapping(const IntervalSet2DDevice& mask,
                                    const Field2DDevice<T>& field) {
  return build_row_map_y(mask.row_keys, field.geometry.row_keys,
                         field.geometry.num_rows);
}

inline Kokkos::View<int*, DeviceMemorySpace>
build_mask_interval_to_row_mapping(const IntervalSet2DDevice& mask) {
  if (mask.num_rows == 0 || mask.num_intervals == 0) {
    return Kokkos::View<int*, DeviceMemorySpace>(
        "subsetix_mask_interval_rows", mask.num_intervals);
  }

  Kokkos::View<int*, DeviceMemorySpace> interval_rows(
      "subsetix_mask_interval_rows", mask.num_intervals);
  auto mask_row_ptr = mask.row_ptr;
  Kokkos::parallel_for(
      "subsetix_fill_mask_interval_rows",
      Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(mask.num_rows)),
      KOKKOS_LAMBDA(const int row_idx) {
        const std::size_t begin = mask_row_ptr(row_idx);
        const std::size_t end = mask_row_ptr(row_idx + 1);
        for (std::size_t k = begin; k < end; ++k) {
          interval_rows(k) = row_idx;
        }
      });
  ExecSpace().fence();
  return interval_rows;
}

template <typename T>
inline Kokkos::View<int*, DeviceMemorySpace>
build_mask_interval_to_field_interval_mapping(
    const IntervalSet2DDevice& mask,
    const Field2DDevice<T>& field,
    const Kokkos::View<int*, DeviceMemorySpace>& row_map) {
  if (mask.num_rows == 0 || mask.num_intervals == 0) {
    return Kokkos::View<int*, DeviceMemorySpace>(
        "subsetix_mask_interval_to_field2d_interval", mask.num_intervals);
  }

  Kokkos::View<int*, DeviceMemorySpace> mapping(
      "subsetix_mask_interval_to_field2d_interval", mask.num_intervals);
  Kokkos::deep_copy(mapping, -1);

  auto mask_row_ptr = mask.row_ptr;
  auto mask_intervals = mask.intervals;
  auto field_row_ptr = field.geometry.row_ptr;
  auto field_intervals = field.geometry.intervals;

  Kokkos::parallel_for(
      "subsetix_fill_mask_interval_field2d_interval",
      Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(mask.num_rows)),
      KOKKOS_LAMBDA(const int row_idx) {
        const int field_row = row_map(row_idx);
        if (field_row < 0) return;

        const std::size_t mask_begin = mask_row_ptr(row_idx);
        const std::size_t mask_end = mask_row_ptr(row_idx + 1);
        const std::size_t field_begin = field_row_ptr(field_row);
        const std::size_t field_end = field_row_ptr(field_row + 1);

        std::size_t fi = field_begin;
        for (std::size_t mi = mask_begin; mi < mask_end; ++mi) {
          const auto mask_iv = mask_intervals(mi);
          while (fi < field_end) {
            const auto field_iv = field_intervals(fi);
            if (mask_iv.begin >= field_iv.begin &&
                mask_iv.end <= field_iv.end) {
              mapping(mi) = static_cast<int>(fi);
              break;
            }
            ++fi;
          }
        }
      });
  ExecSpace().fence();
  return mapping;
}

template <typename T>
inline MaskFieldMapping<T>
build_mask_field_mapping(const Field2DDevice<T>& field,
                         const IntervalSet2DDevice& mask) {
  MaskFieldMapping<T> mapping;
  const FieldMaskMapping base =
      build_field_mask_mapping(mask, field.geometry);
  mapping.row_map = base.row_map;
  mapping.interval_to_row = base.interval_to_row;
  mapping.interval_to_field_interval = base.interval_to_field_interval;
  return mapping;
}

} // namespace detail

} // namespace csr
} // namespace subsetix

// Include unified implementations from field_subset.hpp
#include <subsetix/csr_ops/field_subset.hpp>
