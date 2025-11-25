#pragma once

#include <Kokkos_Core.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/csr_ops/workspace.hpp>
#include <subsetix/csr_ops/core.hpp> // for row_union_count/fill
#include <subsetix/detail/csr_utils.hpp>

namespace subsetix {
namespace csr {
namespace detail {

template <class IntervalView>
struct CoarsenIntervalView {
  IntervalView base;

  KOKKOS_INLINE_FUNCTION
  Interval operator()(const std::size_t i) const {
    const Interval iv = base(i);
    const Coord b = detail::floor_div2(iv.begin);
    const Coord e = detail::ceil_div2(iv.end);
    return Interval{b, e};
  }
};

struct RowCoarsenResult {
  IntervalSet2DDevice::RowKeyView row_keys;
  Kokkos::View<int*, DeviceMemorySpace> row_index_first;
  Kokkos::View<int*, DeviceMemorySpace> row_index_second;
  std::size_t num_rows = 0;
};

/**
 * @brief Build coarse-row mapping for a level-down projection (fine -> coarse).
 */
inline RowCoarsenResult
build_row_coarsen_mapping(const IntervalSet2DDevice& fine,
                          UnifiedCsrWorkspace& workspace) {
  RowCoarsenResult result;

  const std::size_t num_rows_fine = fine.num_rows;
  if (num_rows_fine == 0) {
    return result;
  }

  // Use workspace buffers instead of allocating
  // tmp_rows: row_key_buf_0 [num_rows_fine]
  // tmp_first: int_buf_0 [num_rows_fine]
  // tmp_second: int_buf_1 [num_rows_fine]
  // d_num_rows: scalar_size_t_buf_0
  // is_boundary: int_buf_2 [num_rows_fine]
  // coarse_y: int_buf_3 [num_rows_fine] - Note: Coord is int32_t
  // boundary_scan: int_buf_4 [num_rows_fine]
  // d_total_boundaries: scalar_int_buf_0
  
  auto tmp_rows = workspace.get_row_key_buf_0(num_rows_fine);
  auto tmp_first = workspace.get_int_buf_0(num_rows_fine);
  auto tmp_second = workspace.get_int_buf_1(num_rows_fine);
  auto is_boundary = workspace.get_int_buf_2(num_rows_fine);
  auto boundary_scan = workspace.get_int_buf_4(num_rows_fine);
  
  // We need to reinterpret cast coarse_y if Coord is not int.
  // But Coord is int32_t, so we can use int_buf.
  static_assert(sizeof(Coord) == sizeof(int), "Coord size mismatch");
  auto coarse_y_int = workspace.get_int_buf_3(num_rows_fine);
  auto coarse_y = Kokkos::subview(coarse_y_int, Kokkos::make_pair(std::size_t(0), num_rows_fine));

  auto rows_in = fine.row_keys;

  // Parallel: compute coarse Y for each fine row and mark boundaries
  Kokkos::parallel_for(
      "subsetix_csr_coarsen_compute_boundaries",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_fine),
      KOKKOS_LAMBDA(const std::size_t i) {
        const Coord y_f = rows_in(i).y;
        const Coord y_c = detail::floor_div2(y_f);
        coarse_y(i) = y_c;
        
        // Mark boundary if first row or coarse Y changes
        if (i == 0) {
          is_boundary(i) = 1;
        } else {
          const Coord y_c_prev = detail::floor_div2(rows_in(i - 1).y);
          is_boundary(i) = (y_c != y_c_prev) ? 1 : 0;
        }
      });

  ExecSpace().fence();

  // Scan to get output indices for boundaries
  int num_rows_out = detail::exclusive_scan_with_total<int>(
      "subsetix_csr_coarsen_scan_boundaries",
      num_rows_fine,
      is_boundary,
      boundary_scan);

  result.num_rows = static_cast<std::size_t>(num_rows_out);
  if (num_rows_out == 0) {
    return result;
  }

  // Fill output arrays in parallel
  Kokkos::parallel_for(
      "subsetix_csr_coarsen_fill_output",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_fine),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (is_boundary(i)) {
          const int out_idx = boundary_scan(i);
          const Coord y_c = coarse_y(i);
          
          tmp_rows(out_idx) = RowKey2D{y_c};
          tmp_first(out_idx) = static_cast<int>(i);
          
          // Check if next row has same coarse Y
          if (i + 1 < num_rows_fine && coarse_y(i + 1) == y_c) {
            tmp_second(out_idx) = static_cast<int>(i + 1);
          } else {
            tmp_second(out_idx) = -1;
          }
        }
      });

  // We need to copy tmp_* to dedicated output buffers if we want to return valid result.
  // Wait, result.row_keys needs to be valid output.
  // We can reuse the unified workspace buffers tmp_rows, tmp_first, tmp_second IF they are not needed further.
  // In this case, result is used by project_level_down_device.
  // project_level_down_device uses result.row_index_first/second to index into fine structure.
  // So we can just set pointers.
  // BUT: result members are Views of specific size.
  // We should subview the large buffers.
  
  // However, row_keys is an output of project_level_down_device.
  // It seems project_level_down_device sets out.row_keys = row_keys_coarse;
  // So if we return workspace buffer, it's fine as long as it persists.
  
  result.row_keys = Kokkos::subview(tmp_rows, Kokkos::make_pair(0, num_rows_out));
  result.row_index_first = Kokkos::subview(tmp_first, Kokkos::make_pair(0, num_rows_out));
  result.row_index_second = Kokkos::subview(tmp_second, Kokkos::make_pair(0, num_rows_out));

  return result;
}

} // namespace detail

/**
 * @brief Refine a CSR interval set to the next finer level (×2 in X and Y).
 */
inline void
refine_level_up_device(const IntervalSet2DDevice& in,
                       IntervalSet2DDevice& out,
                       CsrSetAlgebraContext& ctx) {
  // ctx is unused in original code except for potential allocation?
  // The original code allocated `out` buffers directly if needed.
  // It seems ctx was passed but unused in the previous implementation too.
  // We keep the signature for consistency but we don't need workspace here
  // unless we want to use it for something temporary (none here).
  
  const std::size_t num_rows_in = in.num_rows;
  const std::size_t num_intervals_in = in.num_intervals;

  if (num_rows_in == 0) {
    out.num_rows = 0;
    out.num_intervals = 0;
    return;
  }

  const std::size_t num_rows_out = num_rows_in * 2;
  const std::size_t num_intervals_out = num_intervals_in * 2;

  // Allocate output buffers if needed
  if (out.row_keys.extent(0) < num_rows_out) {
    out.row_keys = IntervalSet2DDevice::RowKeyView(
        "subsetix_csr_refine_row_keys_out", num_rows_out);
  }
  if (out.row_ptr.extent(0) < num_rows_out + 1) {
    out.row_ptr = IntervalSet2DDevice::IndexView(
        "subsetix_csr_refine_row_ptr_out", num_rows_out + 1);
  }
  if (num_intervals_out > 0 && out.intervals.extent(0) < num_intervals_out) {
    out.intervals = IntervalSet2DDevice::IntervalView(
        "subsetix_csr_refine_intervals_out", num_intervals_out);
  }

  auto row_keys_in = in.row_keys;
  auto row_ptr_in = in.row_ptr;
  auto intervals_in = in.intervals;
  auto row_keys_out = out.row_keys;
  auto row_ptr_out = out.row_ptr;
  auto intervals_out = out.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_refine_row_keys",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_in),
      KOKKOS_LAMBDA(const std::size_t i) {
        const Coord y = row_keys_in(i).y;
        row_keys_out(2 * i) = RowKey2D{
            static_cast<Coord>(2 * y)};
        row_keys_out(2 * i + 1) = RowKey2D{
            static_cast<Coord>(2 * y + 1)};
      });

  Kokkos::parallel_for(
      "subsetix_csr_refine_row_ptr",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_in),
      KOKKOS_LAMBDA(const std::size_t i) {
        const std::size_t start = row_ptr_in(i);
        const std::size_t end = row_ptr_in(i + 1);
        const std::size_t count = end - start;
        row_ptr_out(2 * i) = 2 * start;
        row_ptr_out(2 * i + 1) = 2 * start + count;
        if (i + 1 == num_rows_in) {
          row_ptr_out(2 * num_rows_in) =
              2 * row_ptr_in(num_rows_in);
        }
      });

  if (num_intervals_out > 0) {
    Kokkos::parallel_for(
        "subsetix_csr_refine_intervals",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_in),
        KOKKOS_LAMBDA(const std::size_t i) {
          const std::size_t start = row_ptr_in(i);
          const std::size_t end = row_ptr_in(i + 1);
          const std::size_t count = end - start;
          const std::size_t out_even = 2 * start;
          const std::size_t out_odd = out_even + count;
          for (std::size_t k = 0; k < count; ++k) {
            const Interval iv = intervals_in(start + k);
            const Interval refined{
                static_cast<Coord>(iv.begin * 2),
                static_cast<Coord>(iv.end * 2)};
            intervals_out(out_even + k) = refined;
            intervals_out(out_odd + k) = refined;
          }
        });
  }

  ExecSpace().fence();

  out.num_rows = num_rows_out;
  out.num_intervals = num_intervals_out;
}

/**
 * @brief Project a CSR interval set to the next coarser level (÷2 in X and Y).
 */
inline void
project_level_down_device(const IntervalSet2DDevice& fine,
                           IntervalSet2DDevice& out,
                           CsrSetAlgebraContext& ctx) {
  const std::size_t num_rows_fine = fine.num_rows;
  const std::size_t num_intervals_fine = fine.num_intervals;

  if (num_rows_fine == 0 || num_intervals_fine == 0) {
    out.num_rows = 0;
    out.num_intervals = 0;
    return;
  }

  detail::RowCoarsenResult rows =
      detail::build_row_coarsen_mapping(fine, ctx.workspace);
  const std::size_t num_rows_coarse = rows.num_rows;
  if (num_rows_coarse == 0) {
    out.num_rows = 0;
    out.num_intervals = 0;
    return;
  }

  auto row_keys_coarse = rows.row_keys;
  auto row_index_first = rows.row_index_first;
  auto row_index_second = rows.row_index_second;

  auto row_ptr_fine = fine.row_ptr;
  auto intervals_fine = fine.intervals;

  using CoarsenView =
      detail::CoarsenIntervalView<IntervalSet2DDevice::IntervalView>;
  CoarsenView coarsen_view{intervals_fine};

  // Use workspace buffers to avoid allocation
  auto row_counts = ctx.workspace.get_size_t_buf_0(num_rows_coarse);

  // Count intervals needed for each coarse row
  Kokkos::parallel_for(
      "subsetix_csr_project_count",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_coarse),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = row_index_first(i);
        const int ib = row_index_second(i);

        std::size_t begin_a = 0;
        std::size_t end_a = 0;
        std::size_t begin_b = 0;
        std::size_t end_b = 0;

        if (ia >= 0) {
          const std::size_t row_a = static_cast<std::size_t>(ia);
          begin_a = row_ptr_fine(row_a);
          end_a = row_ptr_fine(row_a + 1);
        }
        if (ib >= 0) {
          const std::size_t row_b = static_cast<std::size_t>(ib);
          begin_b = row_ptr_fine(row_b);
          end_b = row_ptr_fine(row_b + 1);
        }

        if (begin_a == end_a && begin_b == end_b) {
          row_counts(i) = 0;
          return;
        }

        row_counts(i) = detail::row_union_count(
            coarsen_view, begin_a, end_a,
            coarsen_view, begin_b, end_b);
      });

  ExecSpace().fence();

  // Allocate output row_ptr buffer if needed
  if (out.row_ptr.extent(0) < num_rows_coarse + 1) {
    out.row_ptr = IntervalSet2DDevice::IndexView(
        "subsetix_csr_project_row_ptr_out", num_rows_coarse + 1);
  }
  auto row_ptr_out = out.row_ptr;
  std::size_t num_intervals_coarse = detail::exclusive_scan_csr_row_ptr<std::size_t>(
      "subsetix_csr_project_scan",
      num_rows_coarse,
      row_counts,
      row_ptr_out);

  out.num_rows = num_rows_coarse;
  out.num_intervals = num_intervals_coarse;
  out.row_keys = row_keys_coarse;

  if (num_intervals_coarse == 0) {
    out.intervals = IntervalSet2DDevice::IntervalView();
    return;
  }

  if (out.intervals.extent(0) < num_intervals_coarse) {
    out.intervals = IntervalSet2DDevice::IntervalView(
        "subsetix_csr_project_intervals_out", num_intervals_coarse);
  }

  auto intervals_out = out.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_project_fill",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_coarse),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = row_index_first(i);
        const int ib = row_index_second(i);

        std::size_t begin_a = 0;
        std::size_t end_a = 0;
        std::size_t begin_b = 0;
        std::size_t end_b = 0;

        if (ia >= 0) {
          const std::size_t row_a = static_cast<std::size_t>(ia);
          begin_a = row_ptr_fine(row_a);
          end_a = row_ptr_fine(row_a + 1);
        }
        if (ib >= 0) {
          const std::size_t row_b = static_cast<std::size_t>(ib);
          begin_b = row_ptr_fine(row_b);
          end_b = row_ptr_fine(row_b + 1);
        }

        if (begin_a == end_a && begin_b == end_b) {
          return;
        }

        const std::size_t out_offset = row_ptr_out(i);
        detail::row_union_fill(coarsen_view, begin_a, end_a,
                               coarsen_view, begin_b, end_b,
                               intervals_out, out_offset);
      });

  ExecSpace().fence();
}

} // namespace csr
} // namespace subsetix
