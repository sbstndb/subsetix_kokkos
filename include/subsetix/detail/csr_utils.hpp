#pragma once

#include <Kokkos_Core.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>

namespace subsetix {
namespace csr {
namespace detail {

KOKKOS_INLINE_FUNCTION
Coord floor_div2(Coord x) {
  // Floor division by 2 for both positive and negative coordinates.
  return (x >= 0) ? (x / 2) : ((x - 1) / 2);
}

KOKKOS_INLINE_FUNCTION
Coord ceil_div2(Coord x) {
  // Ceil division by 2 for both positive and negative coordinates.
  return (x >= 0) ? ((x + 1) / 2) : (x / 2);
}

/**
 * @brief Reset a preallocated IntervalSet2DDevice buffer.
 *
 * For now we simply fill the full capacity with zeros for all views.
 * TODO: consider restricting the zeroing to the portions actually used.
 */
inline void reset_preallocated_interval_set(IntervalSet2DDevice& out) {
  out.num_rows = 0;
  out.num_intervals = 0;

  const std::size_t rows_cap = out.row_keys.extent(0);
  const std::size_t row_ptr_cap = out.row_ptr.extent(0);
  const std::size_t intervals_cap = out.intervals.extent(0);

  if (rows_cap > 0) {
    RowKey2D zero_key{};
    Kokkos::deep_copy(out.row_keys, zero_key);
  }
  if (row_ptr_cap > 0) {
    Kokkos::deep_copy(out.row_ptr, std::size_t(0));
  }
  if (intervals_cap > 0) {
    Interval zero_interval{};
    Kokkos::deep_copy(out.intervals, zero_interval);
  }
}

} // namespace detail
} // namespace csr
} // namespace subsetix
