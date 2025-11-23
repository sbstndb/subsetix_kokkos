#pragma once

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <subsetix/geometry.hpp>

// Utilitaires de tests pour comparer et construire des CSR 2D simples.

namespace subsetix {
namespace csr_test {

using namespace subsetix::csr;

inline IntervalSet2DHost make_host_csr(
    const std::vector<std::pair<Coord, std::vector<Interval>>>& rows) {
  IntervalSet2DHost h;
  h.row_keys.reserve(rows.size());
  h.row_ptr.reserve(rows.size() + 1);

  std::size_t offset = 0;
  h.row_ptr.push_back(offset);

  for (const auto& row : rows) {
    const Coord y = row.first;
    const auto& ivs = row.second;
    h.row_keys.push_back(RowKey2D{y});
    for (const auto& iv : ivs) {
      h.intervals.push_back(iv);
      ++offset;
    }
    h.row_ptr.push_back(offset);
  }

  return h;
}

inline void expect_equal_csr(const IntervalSet2DHost& a,
                             const IntervalSet2DHost& b) {
  ASSERT_EQ(a.row_keys.size(), b.row_keys.size())
      << "row_keys size mismatch";
  ASSERT_EQ(a.row_ptr.size(), b.row_ptr.size())
      << "row_ptr size mismatch";
  ASSERT_EQ(a.intervals.size(), b.intervals.size())
      << "intervals size mismatch";

  for (std::size_t i = 0; i < a.row_keys.size(); ++i) {
    EXPECT_EQ(a.row_keys[i].y, b.row_keys[i].y)
        << "row_keys mismatch at index " << i;
  }
  for (std::size_t i = 0; i < a.row_ptr.size(); ++i) {
    EXPECT_EQ(a.row_ptr[i], b.row_ptr[i])
        << "row_ptr mismatch at index " << i;
  }
  for (std::size_t i = 0; i < a.intervals.size(); ++i) {
    EXPECT_EQ(a.intervals[i].begin, b.intervals[i].begin)
        << "interval begin mismatch at index " << i;
    EXPECT_EQ(a.intervals[i].end, b.intervals[i].end)
        << "interval end mismatch at index " << i;
  }
}

inline std::size_t cardinality(const IntervalSet2DHost& s) {
  std::size_t total = 0;
  for (const auto& iv : s.intervals) {
    const Coord width = iv.end - iv.begin;
    if (width > 0) {
      total += static_cast<std::size_t>(width);
    }
  }
  return total;
}

} // namespace csr_test
} // namespace subsetix

