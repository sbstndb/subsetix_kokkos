#pragma once

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>

// Utilitaires de tests pour comparer et construire des CSR 2D simples.

namespace subsetix {
namespace csr_test {

using namespace subsetix::csr;

inline IntervalSet2DHost make_host_csr(
    const std::vector<std::pair<Coord, std::vector<Interval>>>& rows) {
  if (rows.empty()) {
    return IntervalSet2DHost{};
  }

  // Count total intervals
  std::size_t total_intervals = 0;
  for (const auto& row : rows) {
    total_intervals += row.second.size();
  }

  const std::size_t num_rows = rows.size();

  IntervalSet2DHost h;
  h.num_rows = num_rows;
  h.num_intervals = total_intervals;

  h.row_keys = IntervalSet2DHost::RowKeyView("test_row_keys", num_rows);
  h.row_ptr = IntervalSet2DHost::IndexView("test_row_ptr", num_rows + 1);
  h.intervals = IntervalSet2DHost::IntervalView("test_intervals", total_intervals);

  std::size_t row_idx = 0;
  std::size_t interval_idx = 0;
  h.row_ptr(0) = 0;

  for (const auto& row : rows) {
    const Coord y = row.first;
    const auto& ivs = row.second;

    h.row_keys(row_idx) = RowKey2D{y};

    for (const auto& iv : ivs) {
      h.intervals(interval_idx) = iv;
      ++interval_idx;
    }

    h.row_ptr(row_idx + 1) = interval_idx;
    ++row_idx;
  }

  compute_cell_offsets_host(h);

  return h;
}

inline void expect_equal_csr(const IntervalSet2DHost& a,
                             const IntervalSet2DHost& b) {
  ASSERT_EQ(a.num_rows, b.num_rows)
      << "num_rows mismatch";
  ASSERT_EQ(a.num_intervals, b.num_intervals)
      << "num_intervals mismatch";

  for (std::size_t i = 0; i < a.num_rows; ++i) {
    EXPECT_EQ(a.row_keys(i).y, b.row_keys(i).y)
        << "row_keys mismatch at index " << i;
  }
  for (std::size_t i = 0; i < a.num_rows + 1; ++i) {
    EXPECT_EQ(a.row_ptr(i), b.row_ptr(i))
        << "row_ptr mismatch at index " << i;
  }
  for (std::size_t i = 0; i < a.num_intervals; ++i) {
    EXPECT_EQ(a.intervals(i).begin, b.intervals(i).begin)
        << "interval begin mismatch at index " << i;
    EXPECT_EQ(a.intervals(i).end, b.intervals(i).end)
        << "interval end mismatch at index " << i;
  }
}

inline std::size_t cardinality(const IntervalSet2DHost& s) {
  std::size_t total = 0;
  for (std::size_t i = 0; i < s.num_intervals; ++i) {
    const Coord width = s.intervals(i).end - s.intervals(i).begin;
    if (width > 0) {
      total += static_cast<std::size_t>(width);
    }
  }
  return total;
}

} // namespace csr_test
} // namespace subsetix
