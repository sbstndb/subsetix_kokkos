// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <benchmark/benchmark.h>
#include <experimental/subsetix/csr/set_algebra/v1.hpp>
#include <experimental/subsetix/csr/set_algebra/v2.hpp>
#include <Kokkos_Core.hpp>
#include <random>

using namespace experimental::subsetix::csr;

// ============================================================================
// Benchmark Fixtures
// ============================================================================

template <int DIM>
struct MeshBenchmark : public benchmark::Fixture {
  using MeshType = Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space>;

  void SetUp(const ::benchmark::State& state) override {
    const std::size_t n_rows = state.range(0);
    const std::size_t intervals_per_row = state.range(1);

    // Create mesh A with regular intervals
    A = create_regular_mesh<DIM>(n_rows, intervals_per_row, 0, 100);

    // Create mesh B with overlapping intervals (offset by 50%)
    B = create_regular_mesh<DIM>(n_rows, intervals_per_row, 25, 75);
  }

  void TearDown(const ::benchmark::State&) override {
    // Kokkos Views will be automatically cleaned up
  }

  static MeshType create_regular_mesh(std::size_t n_rows, std::size_t intervals_per_row,
                                      int offset, int max_coord) {
    MeshType mesh;
    mesh.num_rows = n_rows;
    mesh.num_intervals = n_rows * intervals_per_row;

    mesh.row_keys = typename MeshType::RowKeyView("row_keys", n_rows);
    mesh.row_ptr = typename MeshType::IndexView("row_ptr", n_rows + 1);
    mesh.intervals = typename MeshType::IntervalView("intervals", mesh.num_intervals);

    auto keys_h = Kokkos::create_mirror_view(mesh.row_keys);
    auto ptr_h = Kokkos::create_mirror_view(mesh.row_ptr);
    auto int_h = Kokkos::create_mirror_view(mesh.intervals);

    std::size_t idx = 0;
    for (std::size_t i = 0; i < n_rows; ++i) {
      if constexpr (DIM == 2) {
        keys_h(i) = RowKey2D{static_cast<int>(i)};
      } else {
        keys_h(i) = RowKey3D{static_cast<int>(i), 0};
      }
      ptr_h(i) = idx;

      // Create intervals with some gaps
      for (std::size_t j = 0; j < intervals_per_row; ++j) {
        int start = offset + static_cast<int>(j * (max_coord - offset) / intervals_per_row);
        int end = start + (max_coord - offset) / intervals_per_row / 2;
        int_h(idx++) = Interval{start, end};
      }
    }
    ptr_h(n_rows) = idx;

    Kokkos::deep_copy(mesh.row_keys, keys_h);
    Kokkos::deep_copy(mesh.row_ptr, ptr_h);
    Kokkos::deep_copy(mesh.intervals, int_h);

    return mesh;
  }

  MeshType A;
  MeshType B;
};

// ============================================================================
// 2D Benchmarks
// ============================================================================

BENCHMARK_TEMPLATE_DEFINE_F(MeshBenchmark, V1_2D_Small)(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v1::intersect_meshes<2>(A, B);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(MeshBenchmark, V2_2D_Small)(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v2::intersect_meshes<2>(A, B);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(MeshBenchmark, V2_2D_NoWS_Small)(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v2::intersect_meshes<2>(A, B);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }
}

// ============================================================================
// 3D Benchmarks
// ============================================================================

BENCHMARK_TEMPLATE_DEFINE_F(MeshBenchmark, V1_3D_Small)(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v1::intersect_meshes<3>(A, B);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(MeshBenchmark, V2_3D_Small)(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v2::intersect_meshes<3>(A, B);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(MeshBenchmark, V2_3D_NoWS_Small)(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v2::intersect_meshes<3>(A, B);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }
}

// ============================================================================
// Register Benchmarks
// ============================================================================

// Small meshes: 10 rows, 5 intervals per row
BENCHMARK_REGISTER_F(MeshBenchmark, V1_2D_Small)->Args({10, 5})->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(MeshBenchmark, V2_2D_Small)->Args({10, 5})->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(MeshBenchmark, V2_2D_NoWS_Small)->Args({10, 5})->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(MeshBenchmark, V1_3D_Small)->Args({10, 5})->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(MeshBenchmark, V2_3D_Small)->Args({10, 5})->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(MeshBenchmark, V2_3D_NoWS_Small)->Args({10, 5})->Unit(benchmark::kMicrosecond);

// Medium meshes: 100 rows, 10 intervals per row
BENCHMARK_REGISTER_F(MeshBenchmark, V1_2D_Small)->Args({100, 10})->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(MeshBenchmark, V2_2D_Small)->Args({100, 10})->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(MeshBenchmark, V2_2D_NoWS_Small)->Args({100, 10})->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(MeshBenchmark, V1_3D_Small)->Args({100, 10})->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(MeshBenchmark, V2_3D_Small)->Args({100, 10})->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(MeshBenchmark, V2_3D_NoWS_Small)->Args({100, 10})->Unit(benchmark::kMicrosecond);

// Large meshes: 1000 rows, 20 intervals per row
BENCHMARK_REGISTER_F(MeshBenchmark, V1_2D_Small)->Args({1000, 20})->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(MeshBenchmark, V2_2D_Small)->Args({1000, 20})->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(MeshBenchmark, V2_2D_NoWS_Small)->Args({1000, 20})->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(MeshBenchmark, V1_3D_Small)->Args({1000, 20})->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(MeshBenchmark, V2_3D_Small)->Args({1000, 20})->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(MeshBenchmark, V2_3D_NoWS_Small)->Args({1000, 20})->Unit(benchmark::kMillisecond);

#endif // SUBSETIX_ENABLE_EXPERIMENTAL
