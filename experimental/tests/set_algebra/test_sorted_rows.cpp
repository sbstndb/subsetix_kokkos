// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <gtest/gtest.h>
#include <experimental/subsetix/csr/set_algebra/v1.hpp>
#include <experimental/subsetix/csr/mesh.hpp>
#include <Kokkos_Core.hpp>

using namespace experimental::subsetix::csr;

// ============================================================================
// Test: Verify row_keys are sorted for all overlap patterns
// ============================================================================

TEST(SortedRowsTest, FullOverlapRowKeysAreSorted) {
  const int n = 64;

  Mesh2DDevice mesh;
  mesh.num_rows = n;
  mesh.num_intervals = n;
  mesh.row_keys = Mesh2DDevice::RowKeyView("row_keys", n);
  mesh.row_ptr = Mesh2DDevice::IndexView("row_ptr", n + 1);
  mesh.intervals = Mesh2DDevice::IntervalView("intervals", n);

  auto row_keys_host = Kokkos::create_mirror_view(mesh.row_keys);
  auto row_ptr_host = Kokkos::create_mirror_view(mesh.row_ptr);
  auto intervals_host = Kokkos::create_mirror_view(mesh.intervals);

  // FULL_OVERLAP pattern
  for (int i = 0; i < n; ++i) {
    row_keys_host(i) = RowKey2D{i};
    row_ptr_host(i) = i;
    intervals_host(i) = Interval{0, 100};
  }
  row_ptr_host(n) = n;

  Kokkos::deep_copy(mesh.row_keys, row_keys_host);
  Kokkos::deep_copy(mesh.row_ptr, row_ptr_host);
  Kokkos::deep_copy(mesh.intervals, intervals_host);

  // Verify sorted
  for (int i = 1; i < n; ++i) {
    EXPECT_GE(row_keys_host(i).y, row_keys_host(i-1).y)
        << "Row keys should be sorted at index " << i;
  }
}

TEST(SortedRowsTest, PartialOverlapPatternIsNOTSorted_Buggy) {
  const int n = 64;

  Mesh2DDevice mesh_a, mesh_b;
  mesh_a.num_rows = n;
  mesh_a.num_intervals = n;
  mesh_a.row_keys = Mesh2DDevice::RowKeyView("row_keys_a", n);
  mesh_a.row_ptr = Mesh2DDevice::IndexView("row_ptr_a", n + 1);
  mesh_a.intervals = Mesh2DDevice::IntervalView("intervals_a", n);

  mesh_b.num_rows = n;
  mesh_b.num_intervals = n;
  mesh_b.row_keys = Mesh2DDevice::RowKeyView("row_keys_b", n);
  mesh_b.row_ptr = Mesh2DDevice::IndexView("row_ptr_b", n + 1);
  mesh_b.intervals = Mesh2DDevice::IntervalView("intervals_b", n);

  auto row_keys_a = Kokkos::create_mirror_view(mesh_a.row_keys);
  auto row_keys_b = Kokkos::create_mirror_view(mesh_b.row_keys);
  auto row_ptr_a = Kokkos::create_mirror_view(mesh_a.row_ptr);
  auto row_ptr_b = Kokkos::create_mirror_view(mesh_b.row_ptr);
  auto intervals_a = Kokkos::create_mirror_view(mesh_a.intervals);
  auto intervals_b = Kokkos::create_mirror_view(mesh_b.intervals);

  // PARTIAL_OVERLAP pattern (BUGGY - generates unsorted row_keys!)
  for (int i = 0; i < n; ++i) {
    const int offset_a = (i % 2 == 0) ? 0 : n/2;
    const int offset_b = (i % 2 == 0) ? 0 : n/2 + n;

    row_keys_a(i) = RowKey2D{i + offset_a};
    row_keys_b(i) = RowKey2D{i + offset_b};

    row_ptr_a(i) = i;
    row_ptr_b(i) = i;
    intervals_a(i) = Interval{0, 100};
    intervals_b(i) = Interval{0, 100};
  }
  row_ptr_a(n) = n;
  row_ptr_b(n) = n;

  // Check if sorted - this should FAIL with the buggy implementation
  bool a_sorted = true;
  bool b_sorted = true;
  for (int i = 1; i < n; ++i) {
    if (row_keys_a(i).y < row_keys_a(i-1).y) a_sorted = false;
    if (row_keys_b(i).y < row_keys_b(i-1).y) b_sorted = false;
  }

  EXPECT_FALSE(a_sorted) << "Mesh A row_keys should NOT be sorted with buggy pattern";
  EXPECT_FALSE(b_sorted) << "Mesh B row_keys should NOT be sorted with buggy pattern";

  // This demonstrates why the benchmark segfaults - binary search on unsorted data!
}

TEST(SortedRowsTest, FixedPartialOverlapPatternIsSorted) {
  const int n = 64;

  Mesh2DDevice mesh_a, mesh_b;
  mesh_a.num_rows = n;
  mesh_a.num_intervals = n;
  mesh_a.row_keys = Mesh2DDevice::RowKeyView("row_keys_a", n);
  mesh_a.row_ptr = Mesh2DDevice::IndexView("row_ptr_a", n + 1);
  mesh_a.intervals = Mesh2DDevice::IntervalView("intervals_a", n);

  mesh_b.num_rows = n;
  mesh_b.num_intervals = n;
  mesh_b.row_keys = Mesh2DDevice::RowKeyView("row_keys_b", n);
  mesh_b.row_ptr = Mesh2DDevice::IndexView("row_ptr_b", n + 1);
  mesh_b.intervals = Mesh2DDevice::IntervalView("intervals_b", n);

  auto row_keys_a = Kokkos::create_mirror_view(mesh_a.row_keys);
  auto row_keys_b = Kokkos::create_mirror_view(mesh_b.row_keys);
  auto row_ptr_a = Kokkos::create_mirror_view(mesh_a.row_ptr);
  auto row_ptr_b = Kokkos::create_mirror_view(mesh_b.row_ptr);
  auto intervals_a = Kokkos::create_mirror_view(mesh_a.intervals);
  auto intervals_b = Kokkos::create_mirror_view(mesh_b.intervals);

  // FIXED PARTIAL_OVERLAP pattern - generate sorted row_keys with 50% overlap
  // Mesh A: rows 0, 2, 4, 6, ..., 126 (even numbers, first half only)
  // Mesh B: rows 32, 34, 36, ..., 158 (shifted even numbers)
  // Overlap: rows 32, 34, 36, ..., 126 (50% of Mesh A)
  for (int i = 0; i < n; ++i) {
    // Mesh A: even rows in range [0, n)
    row_keys_a(i) = RowKey2D{2 * i};

    // Mesh B: even rows in range [n/2, 3n/2)
    row_keys_b(i) = RowKey2D{n/2 + 2 * i};

    row_ptr_a(i) = i;
    row_ptr_b(i) = i;
    intervals_a(i) = Interval{0, 100};
    intervals_b(i) = Interval{0, 100};
  }
  row_ptr_a(n) = n;
  row_ptr_b(n) = n;

  // Verify sorted
  for (int i = 1; i < n; ++i) {
    EXPECT_GE(row_keys_a(i).y, row_keys_a(i-1).y)
        << "Mesh A row_keys should be sorted";
    EXPECT_GE(row_keys_b(i).y, row_keys_b(i-1).y)
        << "Mesh B row_keys should be sorted";
  }
}

#endif
