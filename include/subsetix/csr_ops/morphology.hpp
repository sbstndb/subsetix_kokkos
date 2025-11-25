#pragma once

#include <Kokkos_Core.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/csr_ops/workspace.hpp>
#include <subsetix/detail/csr_utils.hpp>

namespace subsetix {
namespace csr {
namespace detail {

// Max expected N for N-way operations (radius 3 -> N=7). 
// We use a small static buffer in kernels.
constexpr int MAX_MORPH_N = 16;

/**
 * @brief Count intervals in the union of N expanded rows.
 * 
 * Each row i contributes intervals [begin - rx, end + rx).
 */
template <class IntervalView>
KOKKOS_INLINE_FUNCTION
std::size_t row_n_way_union_count(const IntervalView& intervals,
                                  const std::size_t* row_ptrs, // Array of current ptrs
                                  const std::size_t* row_ends, // Array of end ptrs
                                  int n_rows,
                                  Coord rx) {
  std::size_t ptrs[MAX_MORPH_N];
  for(int i=0; i<n_rows; ++i) ptrs[i] = row_ptrs[i];

  std::size_t count = 0;
  bool active = true;

  // Current merged interval
  Coord cur_end = 0;
  bool have_current = false;

  while(active) {
    active = false;
    Coord min_start = 0;
    bool first = true;
    int best_idx = -1;

    // Find the interval that starts earliest among all rows
    for(int i=0; i<n_rows; ++i) {
      if(ptrs[i] < row_ends[i]) {
        active = true;
        const auto iv = intervals(ptrs[i]);
        const Coord b = iv.begin - rx;
        if(first || b < min_start) {
          min_start = b;
          best_idx = i;
          first = false;
        }
      }
    }

    if(!active) break;

    // Process the best interval
    const auto iv = intervals(ptrs[best_idx]);
    const Coord b = iv.begin - rx;
    const Coord e = iv.end + rx;
    
    // Advance the pointer
    ptrs[best_idx]++;

    if(!have_current) {
      cur_end = e;
      have_current = true;
    } else {
      // Overlap or adjacent?
      if(b <= cur_end) {
        // Merge
        if(e > cur_end) cur_end = e;
      } else {
        // Emit current
        count++;
        cur_end = e;
      }
    }
  }

  if(have_current) count++;
  return count;
}

template <class IntervalView, class IntervalViewOut>
KOKKOS_INLINE_FUNCTION
void row_n_way_union_fill(const IntervalView& intervals,
                          const std::size_t* row_ptrs,
                          const std::size_t* row_ends,
                          int n_rows,
                          Coord rx,
                          const IntervalViewOut& intervals_out,
                          std::size_t out_offset) {
  std::size_t ptrs[MAX_MORPH_N];
  for(int i=0; i<n_rows; ++i) ptrs[i] = row_ptrs[i];

  std::size_t write_idx = 0;
  bool active = true;
  Coord cur_begin = 0;
  Coord cur_end = 0;
  bool have_current = false;

  while(active) {
    active = false;
    Coord min_start = 0;
    bool first = true;
    int best_idx = -1;

    for(int i=0; i<n_rows; ++i) {
      if(ptrs[i] < row_ends[i]) {
        active = true;
        const auto iv = intervals(ptrs[i]);
        const Coord b = iv.begin - rx;
        if(first || b < min_start) {
          min_start = b;
          best_idx = i;
          first = false;
        }
      }
    }

    if(!active) break;

    const auto iv = intervals(ptrs[best_idx]);
    const Coord b = iv.begin - rx;
    const Coord e = iv.end + rx;
    ptrs[best_idx]++;

    if(!have_current) {
      cur_begin = b;
      cur_end = e;
      have_current = true;
    } else {
      if(b <= cur_end) {
        if(e > cur_end) cur_end = e;
      } else {
        intervals_out(out_offset + write_idx) = Interval{cur_begin, cur_end};
        write_idx++;
        cur_begin = b;
        cur_end = e;
      }
    }
  }

  if(have_current) {
    intervals_out(out_offset + write_idx) = Interval{cur_begin, cur_end};
  }
}

/**
 * @brief Count intervals in the intersection of N shrunk rows.
 * 
 * Each row i contributes intervals [begin + rx, end - rx).
 * If begin + rx >= end - rx, the interval is ignored.
 */
template <class IntervalView>
KOKKOS_INLINE_FUNCTION
std::size_t row_n_way_intersection_count(const IntervalView& intervals,
                                         const std::size_t* row_ptrs,
                                         const std::size_t* row_ends,
                                         int n_rows,
                                         Coord rx) {
  if(n_rows == 0) return 0;

  std::size_t ptrs[MAX_MORPH_N];
  for(int i=0; i<n_rows; ++i) ptrs[i] = row_ptrs[i];

  std::size_t count = 0;
  
  // Skip empty/invalid intervals initially
  for(int i=0; i<n_rows; ++i) {
    while(ptrs[i] < row_ends[i]) {
      const auto iv = intervals(ptrs[i]);
      const Coord b = iv.begin + rx;
      const Coord e = iv.end - rx;
      if(b < e) break; // Valid
      ptrs[i]++;
    }
    if(ptrs[i] == row_ends[i]) return 0; // One row empty -> intersection empty
  }

  while(true) {
    // Find max start and min end of current intervals
    Coord max_start = -2147483648; 
    Coord min_end = 2147483647; 
    
    // We also need to identify which interval ends earliest to advance it
    int earliest_end_idx = -1;
    Coord earliest_end_val = 2147483647;

    for(int i=0; i<n_rows; ++i) {
      const auto iv = intervals(ptrs[i]);
      const Coord b = iv.begin + rx;
      const Coord e = iv.end - rx;

      if(i==0 || b > max_start) max_start = b;
      if(i==0 || e < min_end) min_end = e;

      if(e < earliest_end_val) {
        earliest_end_val = e;
        earliest_end_idx = i;
      }
    }

    if(max_start < min_end) {
      count++;
    }

    // Advance the one that ends earliest
    int idx = earliest_end_idx;
    ptrs[idx]++;
    
    // Skip invalid intervals for this row
    while(ptrs[idx] < row_ends[idx]) {
      const auto iv = intervals(ptrs[idx]);
      const Coord b = iv.begin + rx;
      const Coord e = iv.end - rx;
      if(b < e) break;
      ptrs[idx]++;
    }

    if(ptrs[idx] == row_ends[idx]) break; // Exhausted one row
  }

  return count;
}

template <class IntervalView, class IntervalViewOut>
KOKKOS_INLINE_FUNCTION
void row_n_way_intersection_fill(const IntervalView& intervals,
                                 const std::size_t* row_ptrs,
                                 const std::size_t* row_ends,
                                 int n_rows,
                                 Coord rx,
                                 const IntervalViewOut& intervals_out,
                                 std::size_t out_offset) {
  if(n_rows == 0) return;

  std::size_t ptrs[MAX_MORPH_N];
  for(int i=0; i<n_rows; ++i) ptrs[i] = row_ptrs[i];

  std::size_t write_idx = 0;

  for(int i=0; i<n_rows; ++i) {
    while(ptrs[i] < row_ends[i]) {
      const auto iv = intervals(ptrs[i]);
      const Coord b = iv.begin + rx;
      const Coord e = iv.end - rx;
      if(b < e) break;
      ptrs[i]++;
    }
    if(ptrs[i] == row_ends[i]) return;
  }

  while(true) {
    Coord max_start = -2147483648;
    Coord min_end = 2147483647;
    int earliest_end_idx = -1;
    Coord earliest_end_val = 2147483647;

    for(int i=0; i<n_rows; ++i) {
      const auto iv = intervals(ptrs[i]);
      const Coord b = iv.begin + rx;
      const Coord e = iv.end - rx;

      if(i==0 || b > max_start) max_start = b;
      if(i==0 || e < min_end) min_end = e;

      if(e < earliest_end_val) {
        earliest_end_val = e;
        earliest_end_idx = i;
      }
    }

    if(max_start < min_end) {
      intervals_out(out_offset + write_idx) = Interval{max_start, min_end};
      write_idx++;
    }

    int idx = earliest_end_idx;
    ptrs[idx]++;
    while(ptrs[idx] < row_ends[idx]) {
      const auto iv = intervals(ptrs[idx]);
      const Coord b = iv.begin + rx;
      const Coord e = iv.end - rx;
      if(b < e) break;
      ptrs[idx]++;
    }

    if(ptrs[idx] == row_ends[idx]) break;
  }
}

struct RowMorphologyResult {
    IntervalSet2DDevice::RowKeyView row_keys;
    Kokkos::View<int*, DeviceMemorySpace> map_start;
    Kokkos::View<int*, DeviceMemorySpace> map_end;
    std::size_t num_rows = 0;
};

/**
 * @brief Build row mapping for Expand (Dilation).
 * Output rows are Union of [y - ry, y + ry] for all input rows y.
 */
inline RowMorphologyResult
build_expand_row_mapping(const IntervalSet2DDevice& in,
                         Coord ry,
                         UnifiedCsrWorkspace& workspace) {
    RowMorphologyResult result;
    std::size_t num_rows_in = in.num_rows;
    if (num_rows_in == 0) return result;

    // Checkout buffers
    // input_row_exists: int_buf_0 [num_rows_in]
    // row_block_counts: size_t_buf_0 [num_rows_in]
    // scan_offsets: size_t_buf_1 [num_rows_in]

    auto in_exists = workspace.get_int_buf_0(num_rows_in);
    auto row_block_counts = workspace.get_size_t_buf_0(num_rows_in);
    auto scan_offsets = workspace.get_size_t_buf_1(num_rows_in);
    
    auto row_keys_in = in.row_keys;
    
    // 1. Identify where overlaps break

    Kokkos::parallel_for("morph_expand_counts", Kokkos::RangePolicy<ExecSpace>(0, num_rows_in),
        KOKKOS_LAMBDA(const std::size_t i) {
            const Coord y = row_keys_in(i).y;
            const Coord start = y - ry;
            const Coord end = y + ry + 1;
            
            bool is_start = true;
            Coord prev_end = 0;
            
            if (i > 0) {
                const Coord prev_y = row_keys_in(i-1).y;
                prev_end = prev_y + ry + 1;
                if (start < prev_end) {
                    is_start = false;
                }
            }
            
            Coord actual_start = start;
            if (!is_start) {
                actual_start = prev_end;
            }
            
            if (actual_start < end) {
                row_block_counts(i) = (end - actual_start);
            } else {
                row_block_counts(i) = 0;
            }
            
            in_exists(i) = is_start ? 1 : 0;
    });
    
    ExecSpace().fence();

    std::size_t total_rows_out = detail::exclusive_scan_with_total<std::size_t>(
        "morph_expand_scan",
        num_rows_in,
        row_block_counts,
        scan_offsets);
    
    result.num_rows = total_rows_out;
    if (total_rows_out == 0) return result;
    
    // Checkout output buffers
    // row_keys: row_key_buf_0 [total_rows_out]
    // map_start: int_buf_1 [total_rows_out]
    // map_end: int_buf_2 [total_rows_out]
    
    auto out_rows = workspace.get_row_key_buf_0(total_rows_out);
    auto map_start = workspace.get_int_buf_1(total_rows_out);
    auto map_end = workspace.get_int_buf_2(total_rows_out);
    
    result.row_keys = out_rows;
    result.map_start = map_start;
    result.map_end = map_end;
    
    // 2. Fill output rows
    Kokkos::parallel_for("morph_expand_fill_rows", Kokkos::RangePolicy<ExecSpace>(0, num_rows_in),
        KOKKOS_LAMBDA(const std::size_t i) {
            const std::size_t count = row_block_counts(i);
            if (count == 0) return;
            
            const Coord y_in = row_keys_in(i).y;
            const std::size_t offset = scan_offsets(i);
            const bool is_start = (in_exists(i) == 1);
            
            Coord start_y = y_in - ry;
            if (!is_start) {
                 const Coord prev_y = row_keys_in(i-1).y;
                 start_y = prev_y + ry + 1;
            }
            
            for(std::size_t k=0; k<count; ++k) {
                out_rows(offset + k) = RowKey2D{static_cast<Coord>(start_y + k)};
            }
    });
    
    ExecSpace().fence();
    
    // 3. Compute map_start and map_end
    Kokkos::parallel_for("morph_expand_map", Kokkos::RangePolicy<ExecSpace>(0, total_rows_out),
        KOKKOS_LAMBDA(const std::size_t j) {
            const Coord y = out_rows(j).y;
            
            std::size_t start_idx = num_rows_in;
            {
                std::size_t l = 0, r = num_rows_in;
                while(l < r) {
                    std::size_t m = l + (r - l)/2;
                    if(row_keys_in(m).y >= y - ry) r = m;
                    else l = m + 1;
                }
                start_idx = l;
            }
            
            std::size_t end_idx = num_rows_in;
            {
                std::size_t l = 0, r = num_rows_in;
                while(l < r) {
                    std::size_t m = l + (r - l)/2;
                    if(row_keys_in(m).y > y + ry) r = m;
                    else l = m + 1;
                }
                end_idx = l;
            }
            
            map_start(j) = static_cast<int>(start_idx);
            map_end(j) = static_cast<int>(end_idx);
    });
    
    ExecSpace().fence();
    
    return result;
}

/**
 * @brief Build row mapping for Shrink (Erosion).
 * Output rows are subset of input rows.
 */
inline RowMorphologyResult
build_shrink_row_mapping(const IntervalSet2DDevice& in,
                         Coord ry,
                         UnifiedCsrWorkspace& workspace) {
    RowMorphologyResult result;
    std::size_t num_rows_in = in.num_rows;
    if (num_rows_in == 0) return result;
    
    // Checkout buffers
    // valid_flags: int_buf_0 [num_rows_in]
    // offsets: size_t_buf_0 [num_rows_in]

    auto valid_flags = workspace.get_int_buf_0(num_rows_in);
    auto offsets = workspace.get_size_t_buf_0(num_rows_in);
    
    auto row_keys_in = in.row_keys;
    
    // Check validity of each row
    Kokkos::parallel_for("morph_shrink_check", Kokkos::RangePolicy<ExecSpace>(0, num_rows_in),
        KOKKOS_LAMBDA(const std::size_t i) {
            const Coord y = row_keys_in(i).y;
            bool possible = true;
            
            std::size_t idx_L = 0;
            {
                std::size_t l = 0, r = num_rows_in;
                while(l < r) {
                    std::size_t m = l + (r - l)/2;
                    if(row_keys_in(m).y >= y - ry) r = m;
                    else l = m + 1;
                }
                idx_L = l;
            }
            
            if (idx_L >= num_rows_in || row_keys_in(idx_L).y != y - ry) {
                possible = false;
            }
            
            if (possible) {
                std::size_t idx_R_candidate = idx_L + 2*ry;
                if (idx_R_candidate < num_rows_in && row_keys_in(idx_R_candidate).y == y + ry) {
                    possible = true;
                } else {
                    possible = false;
                }
            }
            
            valid_flags(i) = possible ? 1 : 0;
    });
    
    ExecSpace().fence();

    std::size_t total_rows_out = detail::exclusive_scan_with_total<std::size_t>(
        "morph_shrink_scan",
        num_rows_in,
        valid_flags,
        offsets);
    
    result.num_rows = total_rows_out;
    if (total_rows_out == 0) return result;
    
    // Checkout output buffers
    // out_keys: row_key_buf_0 [total_rows_out]
    // out_map_start: int_buf_1 [total_rows_out]
    
    auto out_keys = workspace.get_row_key_buf_0(total_rows_out);
    auto out_map_start = workspace.get_int_buf_1(total_rows_out);
    
    Kokkos::parallel_for("morph_shrink_fill", Kokkos::RangePolicy<ExecSpace>(0, num_rows_in),
        KOKKOS_LAMBDA(const std::size_t i) {
            if (valid_flags(i)) {
                const std::size_t off = offsets(i);
                out_keys(off) = row_keys_in(i);
                out_map_start(off) = static_cast<int>(i);
            }
    });
    
    ExecSpace().fence();
    
    result.row_keys = out_keys;
    result.map_start = out_map_start;
    
    return result;
}

} // namespace detail

/**
 * @brief Expand (dilate) a CSR set by rx in X and ry in Y.
 */
inline void
expand_device(const IntervalSet2DDevice& in,
              Coord rx,
              Coord ry,
              IntervalSet2DDevice& out,
              CsrSetAlgebraContext& ctx) {
  detail::reset_preallocated_interval_set(out);
  if (in.num_rows == 0) return;

  if (rx < 0) rx = 0;
  if (ry < 0) ry = 0;

  // 1. Determine output rows and mapping
  detail::RowMorphologyResult map_result =
      detail::build_expand_row_mapping(in, ry, ctx.workspace);
  
  const std::size_t num_rows_out = map_result.num_rows;
  if (num_rows_out == 0) return;
  
  // Allocate rows
  if (out.row_keys.extent(0) < num_rows_out) {
    out.row_keys = IntervalSet2DDevice::RowKeyView("subsetix_expand_rows", num_rows_out);
  }
  
  auto row_keys_out = out.row_keys;
  auto map_keys = map_result.row_keys;
  
  Kokkos::parallel_for("subsetix_expand_copy_keys", Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
    KOKKOS_LAMBDA(const std::size_t i) {
      row_keys_out(i) = map_keys(i);
  });

  // 2. Count intervals
  auto row_counts = ctx.workspace.get_size_t_buf_0(num_rows_out);

  auto map_start = map_result.map_start;
  auto map_end = map_result.map_end;
  auto row_ptr_in = in.row_ptr;
  auto intervals_in = in.intervals;
  
  Kokkos::parallel_for("subsetix_expand_count", Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
    KOKKOS_LAMBDA(const std::size_t i) {
      const int start_idx = map_start(i);
      const int end_idx = map_end(i);
      const int n_rows = end_idx - start_idx;
      const int capped_rows = (n_rows > detail::MAX_MORPH_N)
                                  ? detail::MAX_MORPH_N
                                  : n_rows;
      
      if (capped_rows <= 0) {
        row_counts(i) = 0;
        return;
      }
      
      std::size_t ptrs[detail::MAX_MORPH_N];
      std::size_t ends[detail::MAX_MORPH_N];
      
      for(int k=0; k<capped_rows; ++k) {
        const std::size_t r = static_cast<std::size_t>(start_idx + k);
        ptrs[k] = row_ptr_in(r);
        ends[k] = row_ptr_in(r+1);
      }
      
      row_counts(i) = detail::row_n_way_union_count(intervals_in, ptrs, ends, capped_rows, rx);
  });
  
  ExecSpace().fence();

  // 3. Scan
  if (out.row_ptr.extent(0) < num_rows_out + 1) {
      out.row_ptr = IntervalSet2DDevice::IndexView("subsetix_expand_row_ptr", num_rows_out + 1);
  }
  auto row_ptr_out = out.row_ptr;
  std::size_t num_intervals_out = detail::exclusive_scan_csr_row_ptr<std::size_t>(
      "subsetix_expand_scan",
      num_rows_out,
      row_counts,
      row_ptr_out);
  
  out.num_rows = num_rows_out;
  out.num_intervals = num_intervals_out;
  
  if (num_intervals_out == 0) return;
  
  if (out.intervals.extent(0) < num_intervals_out) {
      out.intervals = IntervalSet2DDevice::IntervalView("subsetix_expand_intervals", num_intervals_out);
  }
  auto intervals_out = out.intervals;
  
  // 4. Fill
  Kokkos::parallel_for("subsetix_expand_fill", Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
    KOKKOS_LAMBDA(const std::size_t i) {
      const int start_idx = map_start(i);
      const int end_idx = map_end(i);
      const int n_rows = end_idx - start_idx;
      const int capped_rows = (n_rows > detail::MAX_MORPH_N)
                                  ? detail::MAX_MORPH_N
                                  : n_rows;
      
      if (capped_rows <= 0) return;
      
      std::size_t ptrs[detail::MAX_MORPH_N];
      std::size_t ends[detail::MAX_MORPH_N];
      
      for(int k=0; k<capped_rows; ++k) {
        const std::size_t r = static_cast<std::size_t>(start_idx + k);
        ptrs[k] = row_ptr_in(r);
        ends[k] = row_ptr_in(r+1);
      }
      
      const std::size_t offset = row_ptr_out(i);
      detail::row_n_way_union_fill(intervals_in, ptrs, ends, capped_rows, rx, intervals_out, offset);
  });
  
  ExecSpace().fence();
}

/**
 * @brief Shrink (erode) a CSR set by rx in X and ry in Y.
 */
inline void
shrink_device(const IntervalSet2DDevice& in,
              Coord rx,
              Coord ry,
              IntervalSet2DDevice& out,
              CsrSetAlgebraContext& ctx) {
  detail::reset_preallocated_interval_set(out);
  if (in.num_rows == 0) return;

  if (rx < 0) rx = 0;
  if (ry < 0) ry = 0;

  // 1. Determine output rows (valid subset of input rows)
  detail::RowMorphologyResult map_result =
      detail::build_shrink_row_mapping(in, ry, ctx.workspace);
  
  const std::size_t num_rows_out = map_result.num_rows;
  if (num_rows_out == 0) return;
  
  if (out.row_keys.extent(0) < num_rows_out) {
      out.row_keys = IntervalSet2DDevice::RowKeyView("subsetix_shrink_rows", num_rows_out);
  }
  auto row_keys_out = out.row_keys;
  auto map_keys = map_result.row_keys;
  
  Kokkos::parallel_for("subsetix_shrink_copy_keys", Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
    KOKKOS_LAMBDA(const std::size_t i) {
      row_keys_out(i) = map_keys(i);
  });
  
  // 2. Count intervals
  auto row_counts = ctx.workspace.get_size_t_buf_0(num_rows_out);

  auto map_start = map_result.map_start; // Contains index of central row
  auto row_ptr_in = in.row_ptr;
  auto intervals_in = in.intervals;
  
  Kokkos::parallel_for("subsetix_shrink_count", Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
    KOKKOS_LAMBDA(const std::size_t i) {
      const int central_idx = map_start(i);
      
      int n_rows = 2 * static_cast<int>(ry) + 1;
      if (n_rows > detail::MAX_MORPH_N) n_rows = detail::MAX_MORPH_N; 
      
      int start_row_idx = central_idx - static_cast<int>(ry);
      
      std::size_t ptrs[detail::MAX_MORPH_N];
      std::size_t ends[detail::MAX_MORPH_N];
      
      for(int k=0; k<n_rows; ++k) {
        std::size_t r = static_cast<std::size_t>(start_row_idx + k);
        ptrs[k] = row_ptr_in(r);
        ends[k] = row_ptr_in(r+1);
      }
      
      row_counts(i) = detail::row_n_way_intersection_count(intervals_in, ptrs, ends, n_rows, rx);
  });
  
  ExecSpace().fence();

  // 3. Scan
  if (out.row_ptr.extent(0) < num_rows_out + 1) {
      out.row_ptr = IntervalSet2DDevice::IndexView("subsetix_shrink_row_ptr", num_rows_out + 1);
  }
  auto row_ptr_out = out.row_ptr;
  std::size_t num_intervals_out = detail::exclusive_scan_csr_row_ptr<std::size_t>(
      "subsetix_shrink_scan",
      num_rows_out,
      row_counts,
      row_ptr_out);
  
  out.num_rows = num_rows_out;
  out.num_intervals = num_intervals_out;
  
  if (num_intervals_out == 0) return;
  
  if (out.intervals.extent(0) < num_intervals_out) {
      out.intervals = IntervalSet2DDevice::IntervalView("subsetix_shrink_intervals", num_intervals_out);
  }
  auto intervals_out = out.intervals;
  
  // 4. Fill
  Kokkos::parallel_for("subsetix_shrink_fill", Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
    KOKKOS_LAMBDA(const std::size_t i) {
      const int central_idx = map_start(i);
      int n_rows = 2 * static_cast<int>(ry) + 1;
      if (n_rows > detail::MAX_MORPH_N) n_rows = detail::MAX_MORPH_N;
      
      int start_row_idx = central_idx - static_cast<int>(ry);
      
      std::size_t ptrs[detail::MAX_MORPH_N];
      std::size_t ends[detail::MAX_MORPH_N];
      
      for(int k=0; k<n_rows; ++k) {
        std::size_t r = static_cast<std::size_t>(start_row_idx + k);
        ptrs[k] = row_ptr_in(r);
        ends[k] = row_ptr_in(r+1);
      }
      
      const std::size_t offset = row_ptr_out(i);
      detail::row_n_way_intersection_fill(intervals_in, ptrs, ends, n_rows, rx, intervals_out, offset);
  });
  
  ExecSpace().fence();
}

} // namespace csr
} // namespace subsetix
