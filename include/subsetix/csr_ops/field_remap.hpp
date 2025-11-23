#pragma once

#include <cstddef>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include <subsetix/csr_field.hpp>
#include <subsetix/csr_interval_set.hpp>
#include <subsetix/detail/csr_utils.hpp>

namespace subsetix {
namespace csr {

namespace detail {

// ---------------------------------------------------------------------------
// Helpers for field remapping
// ---------------------------------------------------------------------------

/**
 * @brief Build a mapping from destination field intervals to source field
 *        intervals based on geometric overlap.
 *
 * For each interval in dst, we find the corresponding row in src (if any)
 * and then find overlapping intervals within that row.
 */
template <typename T>
struct RemapMapping {
  Kokkos::View<int*, DeviceMemorySpace> dst_interval_to_src_row;
  Kokkos::View<int*, DeviceMemorySpace> dst_interval_to_src_interval_start;
  Kokkos::View<int*, DeviceMemorySpace> dst_interval_to_src_interval_count;
};

template <typename T>
inline RemapMapping<T>
build_remap_mapping(const Field2DDevice<T>& dst,
                    const Field2DDevice<T>& src) {
  RemapMapping<T> mapping;

  const std::size_t num_dst_intervals = dst.geometry.num_intervals;
  if (num_dst_intervals == 0) {
    return mapping;
  }

  mapping.dst_interval_to_src_row = Kokkos::View<int*, DeviceMemorySpace>(
      "subsetix_remap_dst_to_src_row", num_dst_intervals);
  mapping.dst_interval_to_src_interval_start = Kokkos::View<int*, DeviceMemorySpace>(
      "subsetix_remap_dst_to_src_interval_start", num_dst_intervals);
  mapping.dst_interval_to_src_interval_count = Kokkos::View<int*, DeviceMemorySpace>(
      "subsetix_remap_dst_to_src_interval_count", num_dst_intervals);

  Kokkos::deep_copy(mapping.dst_interval_to_src_row, -1);
  Kokkos::deep_copy(mapping.dst_interval_to_src_interval_start, -1);
  Kokkos::deep_copy(mapping.dst_interval_to_src_interval_count, 0);

  auto dst_row_keys = dst.geometry.row_keys;
  auto dst_row_ptr = dst.geometry.row_ptr;
  auto dst_intervals = dst.geometry.intervals;
  auto dst_offsets = dst.geometry.cell_offsets;

  auto src_row_keys = src.geometry.row_keys;
  auto src_row_ptr = src.geometry.row_ptr;
  auto src_intervals = src.geometry.intervals;
  auto src_offsets = src.geometry.cell_offsets;

  const std::size_t num_dst_rows = dst.geometry.num_rows;
  const std::size_t num_src_rows = src.geometry.num_rows;

  auto dst_to_src_row = mapping.dst_interval_to_src_row;
  auto dst_to_src_start = mapping.dst_interval_to_src_interval_start;
  auto dst_to_src_count = mapping.dst_interval_to_src_interval_count;

  // Step 1: For each dst row, find corresponding src row
  Kokkos::parallel_for(
      "subsetix_remap_find_src_rows",
      Kokkos::RangePolicy<ExecSpace>(0, num_dst_rows),
      KOKKOS_LAMBDA(const std::size_t dst_row_idx) {
        const Coord y_dst = dst_row_keys(dst_row_idx).y;

        // Binary search in src rows
        std::size_t lo = 0;
        std::size_t hi = num_src_rows;
        while (lo < hi) {
          const std::size_t mid = lo + (hi - lo) / 2;
          if (src_row_keys(mid).y < y_dst) {
            lo = mid + 1;
          } else {
            hi = mid;
          }
        }

        int src_row_idx = -1;
        if (lo < num_src_rows && src_row_keys(lo).y == y_dst) {
          src_row_idx = static_cast<int>(lo);
        }

        // Mark all intervals in this dst row
        const std::size_t dst_begin = dst_row_ptr(dst_row_idx);
        const std::size_t dst_end = dst_row_ptr(dst_row_idx + 1);
        for (std::size_t di = dst_begin; di < dst_end; ++di) {
          dst_to_src_row(di) = src_row_idx;
        }
      });

  ExecSpace().fence();

  // Step 2: For each dst interval, find overlapping src intervals
  Kokkos::parallel_for(
      "subsetix_remap_find_src_intervals",
      Kokkos::RangePolicy<ExecSpace>(0, num_dst_intervals),
      KOKKOS_LAMBDA(const std::size_t di) {
        const int src_row_idx = dst_to_src_row(di);
        if (src_row_idx < 0) {
          return;
        }

        const auto dst_iv = dst_intervals(di);
        const std::size_t src_begin = src_row_ptr(src_row_idx);
        const std::size_t src_end = src_row_ptr(src_row_idx + 1);

        // Find first overlapping src interval
        int first_overlap = -1;
        int overlap_count = 0;

        for (std::size_t si = src_begin; si < src_end; ++si) {
          const auto src_iv = src_intervals(si);

          // Check for overlap: [dst.begin, dst.end) ∩ [src.begin, src.end)
          const Coord overlap_begin = (dst_iv.begin > src_iv.begin) ? dst_iv.begin : src_iv.begin;
          const Coord overlap_end = (dst_iv.end < src_iv.end) ? dst_iv.end : src_iv.end;

          if (overlap_begin < overlap_end) {
            if (first_overlap < 0) {
              first_overlap = static_cast<int>(si);
            }
            ++overlap_count;
          }
        }

        dst_to_src_start(di) = first_overlap;
        dst_to_src_count(di) = overlap_count;
      });

  ExecSpace().fence();

  return mapping;
}

} // namespace detail

// ---------------------------------------------------------------------------
// Field Remapping / Projection
// ---------------------------------------------------------------------------

/**
 * @brief Transfer field values from src to dst where geometries overlap.
 *
 * For each cell (x, y) in dst:
 *   - If (x, y) exists in src, copy the value from src.
 *   - Otherwise, set the value to default_value.
 *
 * This operation is useful after set algebra operations that change the
 * geometry, allowing you to project field values onto the new geometry.
 */
template <typename T>
inline void remap_field_device(Field2DDevice<T>& dst,
                               const Field2DDevice<T>& src,
                               const T& default_value) {
  if (dst.geometry.num_intervals == 0) {
    return;
  }

  // Build mapping
  const auto mapping = detail::build_remap_mapping(dst, src);

  auto dst_row_keys = dst.geometry.row_keys;
  auto dst_row_ptr = dst.geometry.row_ptr;
  auto dst_intervals = dst.geometry.intervals;
  auto dst_offsets = dst.geometry.cell_offsets;
  auto dst_values = dst.values;

  auto src_row_ptr = src.geometry.row_ptr;
  auto src_intervals = src.geometry.intervals;
  auto src_offsets = src.geometry.cell_offsets;
  auto src_values = src.values;

  auto dst_to_src_row = mapping.dst_interval_to_src_row;
  auto dst_to_src_start = mapping.dst_interval_to_src_interval_start;
  auto dst_to_src_count = mapping.dst_interval_to_src_interval_count;

  const T default_val = default_value;

  Kokkos::parallel_for(
      "subsetix_remap_field",
      Kokkos::RangePolicy<ExecSpace>(0, dst.geometry.num_intervals),
      KOKKOS_LAMBDA(const std::size_t di) {
        const auto dst_iv = dst_intervals(di);
        const std::size_t dst_offset = dst_offsets(di);
        const Coord dst_begin = dst_iv.begin;
        const Coord dst_end = dst_iv.end;

        const int src_row_idx = dst_to_src_row(di);
        if (src_row_idx < 0) {
          // No src row: fill with default
          for (Coord x = dst_begin; x < dst_end; ++x) {
            const std::size_t idx = dst_offset + static_cast<std::size_t>(x - dst_begin);
            dst_values(idx) = default_val;
          }
          return;
        }

        const int first_src_interval = dst_to_src_start(di);
        const int num_src_intervals = dst_to_src_count(di);

        if (first_src_interval < 0 || num_src_intervals == 0) {
          // No overlap: fill with default
          for (Coord x = dst_begin; x < dst_end; ++x) {
            const std::size_t idx = dst_offset + static_cast<std::size_t>(x - dst_begin);
            dst_values(idx) = default_val;
          }
          return;
        }

        // Process each cell in dst interval
        for (Coord x = dst_begin; x < dst_end; ++x) {
          const std::size_t dst_idx = dst_offset + static_cast<std::size_t>(x - dst_begin);
          bool found = false;

          // Search in overlapping src intervals
          for (int k = 0; k < num_src_intervals; ++k) {
            const std::size_t si = static_cast<std::size_t>(first_src_interval + k);
            const auto src_iv = src_intervals(si);

            if (x >= src_iv.begin && x < src_iv.end) {
              // Found: copy value
              const std::size_t src_idx = src_offsets(si) +
                                          static_cast<std::size_t>(x - src_iv.begin);
              dst_values(dst_idx) = src_values(src_idx);
              found = true;
              break;
            }
          }

          if (!found) {
            dst_values(dst_idx) = default_val;
          }
        }
      });

  ExecSpace().fence();
}

/**
 * @brief Accumulate field values from src into dst where geometries overlap.
 *
 * For each cell (x, y) in dst:
 *   - If (x, y) exists in src, add src value to dst: dst += src.
 *   - Otherwise, leave dst unchanged.
 *
 * This is useful for accumulating contributions from multiple sources.
 */
template <typename T>
inline void accumulate_field_device(Field2DDevice<T>& dst,
                                    const Field2DDevice<T>& src) {
  if (dst.geometry.num_intervals == 0) {
    return;
  }

  // Build mapping
  const auto mapping = detail::build_remap_mapping(dst, src);

  auto dst_intervals = dst.geometry.intervals;
  auto dst_offsets = dst.geometry.cell_offsets;
  auto dst_values = dst.values;

  auto src_intervals = src.geometry.intervals;
  auto src_offsets = src.geometry.cell_offsets;
  auto src_values = src.values;

  auto dst_to_src_row = mapping.dst_interval_to_src_row;
  auto dst_to_src_start = mapping.dst_interval_to_src_interval_start;
  auto dst_to_src_count = mapping.dst_interval_to_src_interval_count;

  Kokkos::parallel_for(
      "subsetix_accumulate_field",
      Kokkos::RangePolicy<ExecSpace>(0, dst.geometry.num_intervals),
      KOKKOS_LAMBDA(const std::size_t di) {
        const auto dst_iv = dst_intervals(di);
        const std::size_t dst_offset = dst_offsets(di);
        const Coord dst_begin = dst_iv.begin;
        const Coord dst_end = dst_iv.end;

        const int src_row_idx = dst_to_src_row(di);
        if (src_row_idx < 0) {
          return;
        }

        const int first_src_interval = dst_to_src_start(di);
        const int num_src_intervals = dst_to_src_count(di);

        if (first_src_interval < 0 || num_src_intervals == 0) {
          return;
        }

        // Process each cell in dst interval
        for (Coord x = dst_begin; x < dst_end; ++x) {
          const std::size_t dst_idx = dst_offset + static_cast<std::size_t>(x - dst_begin);

          // Search in overlapping src intervals
          for (int k = 0; k < num_src_intervals; ++k) {
            const std::size_t si = static_cast<std::size_t>(first_src_interval + k);
            const auto src_iv = src_intervals(si);

            if (x >= src_iv.begin && x < src_iv.end) {
              // Found: accumulate value
              const std::size_t src_idx = src_offsets(si) +
                                          static_cast<std::size_t>(x - src_iv.begin);
              dst_values(dst_idx) += src_values(src_idx);
              break;
            }
          }
        }
      });

  ExecSpace().fence();
}

} // namespace csr
} // namespace subsetix
