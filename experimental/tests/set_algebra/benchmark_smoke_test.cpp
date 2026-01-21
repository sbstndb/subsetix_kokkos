// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <gtest/gtest.h>
#include <experimental/subsetix/csr/set_algebra/v1.hpp>
#include <experimental/subsetix/csr/set_algebra/v2.hpp>
#include <Kokkos_Core.hpp>

using namespace experimental::subsetix::csr;

// ============================================================================
// Benchmark Smoke Test - Quick sanity check that benchmark code works
// ============================================================================

TEST(ExperimentalBenchmarkSmokeTest, BenchmarkCodeDoesNotSegfault) {
  // This test reproduces what the benchmark does but with assertions
  // instead of benchmark framework, making it easier to debug in CI

  const int n = 64;

  // Generate meshes (same as benchmark)
  Mesh2DDevice mesh_a, mesh_b;
  mesh_a.num_rows = n;
  mesh_a.num_intervals = n;
  mesh_a.row_keys = Mesh2DDevice::RowKeyView("test_row_keys", n);
  mesh_a.row_ptr = Mesh2DDevice::IndexView("test_row_ptr", n + 1);
  mesh_a.intervals = Mesh2DDevice::IntervalView("test_intervals", n);

  mesh_b.num_rows = n;
  mesh_b.num_intervals = n;
  mesh_b.row_keys = Mesh2DDevice::RowKeyView("test_row_keys_b", n);
  mesh_b.row_ptr = Mesh2DDevice::IndexView("test_row_ptr_b", n + 1);
  mesh_b.intervals = Mesh2DDevice::IntervalView("test_intervals_b", n);

  // Fill with data
  auto row_keys_host = Kokkos::create_mirror_view(mesh_a.row_keys);
  auto row_ptr_host = Kokkos::create_mirror_view(mesh_a.row_ptr);
  auto intervals_host = Kokkos::create_mirror_view(mesh_a.intervals);

  for (int i = 0; i < n; ++i) {
    row_keys_host(i) = RowKey2D{i};
    row_ptr_host(i) = i;
    intervals_host(i) = Interval{0, 100};
  }
  row_ptr_host(n) = n;

  Kokkos::deep_copy(mesh_a.row_keys, row_keys_host);
  Kokkos::deep_copy(mesh_a.row_ptr, row_ptr_host);
  Kokkos::deep_copy(mesh_a.intervals, intervals_host);

  Kokkos::deep_copy(mesh_b.row_keys, row_keys_host);
  Kokkos::deep_copy(mesh_b.row_ptr, row_ptr_host);
  Kokkos::deep_copy(mesh_b.intervals, intervals_host);

  // Run intersection (same as benchmark)
  auto result_v1 = v1::intersect_meshes_2d(mesh_a, mesh_b);

  // Basic sanity checks
  EXPECT_GT(result_v1.num_rows, 0);
  EXPECT_GT(result_v1.num_intervals, 0);
  EXPECT_EQ(result_v1.num_intervals, result_v1.num_rows);  // 1:1 for FULL_OVERLAP

  // Test v2 with workspace
  v2::MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space> workspace;
  auto result_v2 = v2::intersect_meshes_2d(mesh_a, mesh_b, workspace);

  EXPECT_GT(result_v2.num_rows, 0);
  EXPECT_GT(result_v2.num_intervals, 0);

  // V1 and V2 should produce same results
  EXPECT_EQ(result_v1.num_rows, result_v2.num_rows);
  EXPECT_EQ(result_v1.num_intervals, result_v2.num_intervals);
}

#endif
