// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <benchmark/benchmark.h>
#include <experimental/subsetix/csr/set_algebra.hpp>
#include <Kokkos_Core.hpp>

using namespace experimental::subsetix::csr;

// ============================================================================
// 2D Mesh Intersection Benchmark - v2
// ============================================================================

static void BM_V2_Intersection2D_Overlapping(benchmark::State& state) {
  const int n = state.range(0);

  // Create two overlapping 2D meshes
  Mesh2DDevice A, B;
  A.num_rows = n;
  A.num_intervals = n;
  A.row_keys = Mesh2DDevice::RowKeyView("A_row_keys", n);
  A.row_ptr = Mesh2DDevice::IndexView("A_row_ptr", n + 1);
  A.intervals = Mesh2DDevice::IntervalView("A_intervals", n);

  B.num_rows = n;
  B.num_intervals = n;
  B.row_keys = Mesh2DDevice::RowKeyView("B_row_keys", n);
  B.row_ptr = Mesh2DDevice::IndexView("B_row_ptr", n + 1);
  B.intervals = Mesh2DDevice::IntervalView("B_intervals", n);

  auto A_row_keys_host = Kokkos::create_mirror_view(A.row_keys);
  auto A_row_ptr_host = Kokkos::create_mirror_view(A.row_ptr);
  auto A_intervals_host = Kokkos::create_mirror_view(A.intervals);
  auto B_row_keys_host = Kokkos::create_mirror_view(B.row_keys);
  auto B_row_ptr_host = Kokkos::create_mirror_view(B.row_ptr);
  auto B_intervals_host = Kokkos::create_mirror_view(B.intervals);

  for (int i = 0; i < n; ++i) {
    A_row_keys_host(i) = RowKey2D{i};
    A_row_ptr_host(i) = i;
    A_intervals_host(i) = Interval{0, 100};

    B_row_keys_host(i) = RowKey2D{i};  // Same rows = full overlap
    B_row_ptr_host(i) = i;
    B_intervals_host(i) = Interval{50, 150};
  }
  A_row_ptr_host(n) = n;
  B_row_ptr_host(n) = n;

  Kokkos::deep_copy(A.row_keys, A_row_keys_host);
  Kokkos::deep_copy(A.row_ptr, A_row_ptr_host);
  Kokkos::deep_copy(A.intervals, A_intervals_host);
  Kokkos::deep_copy(B.row_keys, B_row_keys_host);
  Kokkos::deep_copy(B.row_ptr, B_row_ptr_host);
  Kokkos::deep_copy(B.intervals, B_intervals_host);

  for (auto _ : state) {
    auto result = v2::intersect_meshes_2d(A, B);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (A.num_intervals + B.num_intervals) * sizeof(Interval));
}

BENCHMARK(BM_V2_Intersection2D_Overlapping)->Range(64, 8192);

// ============================================================================
// 3D Mesh Intersection Benchmark - v2
// ============================================================================

static void BM_V2_Intersection3D_Overlapping(benchmark::State& state) {
  const int n = state.range(0);

  // Create two overlapping 3D meshes (simplified: same (y,z) rows)
  Mesh3DDevice A, B;
  A.num_rows = n;
  A.num_intervals = n;
  A.row_keys = Mesh3DDevice::RowKeyView("A_row_keys", n);
  A.row_ptr = Mesh3DDevice::IndexView("A_row_ptr", n + 1);
  A.intervals = Mesh3DDevice::IntervalView("A_intervals", n);

  B.num_rows = n;
  B.num_intervals = n;
  B.row_keys = Mesh3DDevice::RowKeyView("B_row_keys", n);
  B.row_ptr = Mesh3DDevice::IndexView("B_row_ptr", n + 1);
  B.intervals = Mesh3DDevice::IntervalView("B_intervals", n);

  auto A_row_keys_host = Kokkos::create_mirror_view(A.row_keys);
  auto A_row_ptr_host = Kokkos::create_mirror_view(A.row_ptr);
  auto A_intervals_host = Kokkos::create_mirror_view(A.intervals);
  auto B_row_keys_host = Kokkos::create_mirror_view(B.row_keys);
  auto B_row_ptr_host = Kokkos::create_mirror_view(B.row_ptr);
  auto B_intervals_host = Kokkos::create_mirror_view(B.intervals);

  for (int i = 0; i < n; ++i) {
    A_row_keys_host(i) = RowKey3D{i, i % 10};
    A_row_ptr_host(i) = i;
    A_intervals_host(i) = Interval{0, 100};

    B_row_keys_host(i) = RowKey3D{i, i % 10};
    B_row_ptr_host(i) = i;
    B_intervals_host(i) = Interval{50, 150};
  }
  A_row_ptr_host(n) = n;
  B_row_ptr_host(n) = n;

  Kokkos::deep_copy(A.row_keys, A_row_keys_host);
  Kokkos::deep_copy(A.row_ptr, A_row_ptr_host);
  Kokkos::deep_copy(A.intervals, A_intervals_host);
  Kokkos::deep_copy(B.row_keys, B_row_keys_host);
  Kokkos::deep_copy(B.row_ptr, B_row_ptr_host);
  Kokkos::deep_copy(B.intervals, B_intervals_host);

  for (auto _ : state) {
    auto result = v2::intersect_meshes_3d(A, B);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (A.num_intervals + B.num_intervals) * sizeof(Interval));
}

BENCHMARK(BM_V2_Intersection3D_Overlapping)->Range(64, 8192);

BENCHMARK_MAIN();

#endif // SUBSETIX_ENABLE_EXPERIMENTAL
