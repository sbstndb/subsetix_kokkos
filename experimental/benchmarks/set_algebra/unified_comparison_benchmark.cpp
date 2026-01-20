// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <benchmark/benchmark.h>
#include <experimental/subsetix/csr/set_algebra/v1.hpp>
#include <experimental/subsetix/csr/set_algebra/v2.hpp>
#include <Kokkos_Core.hpp>

using namespace experimental::subsetix::csr;

// ============================================================================
// Test Data Generation
// ============================================================================

enum class OverlapPattern {
  FULL_OVERLAP,      // 100% overlap (same rows)
  PARTIAL_OVERLAP,   // 50% overlap
  MINIMAL_OVERLAP,   // 10% overlap
  NO_OVERLAP         // 0% overlap
};

/**
 * @brief Generate test meshes for v1/v2 (experimental Mesh types)
 */
template <class MemorySpace>
Mesh<2, MemorySpace> generate_mesh_2d(
    int n,
    OverlapPattern pattern,
    int offset_shift = 0) {

  Mesh<2, MemorySpace> mesh;
  mesh.num_rows = n;
  mesh.num_intervals = n;
  mesh.row_keys = typename Mesh<2, MemorySpace>::RowKeyView("gen_row_keys", n);
  mesh.row_ptr = typename Mesh<2, MemorySpace>::IndexView("gen_row_ptr", n + 1);
  mesh.intervals = typename Mesh<2, MemorySpace>::IntervalView("gen_intervals", n);

  auto row_keys_host = Kokkos::create_mirror_view(mesh.row_keys);
  auto row_ptr_host = Kokkos::create_mirror_view(mesh.row_ptr);
  auto intervals_host = Kokkos::create_mirror_view(mesh.intervals);

  for (int i = 0; i < n; ++i) {
    // Apply pattern-based offset
    int row_offset = 0;
    switch (pattern) {
      case OverlapPattern::FULL_OVERLAP:
        row_offset = 0;
        break;
      case OverlapPattern::PARTIAL_OVERLAP:
        row_offset = (i % 2 == 0) ? 0 : n/2;
        break;
      case OverlapPattern::MINIMAL_OVERLAP:
        row_offset = (i % 10 == 0) ? 0 : n + offset_shift;
        break;
      case OverlapPattern::NO_OVERLAP:
        row_offset = n + offset_shift;
        break;
    }

    row_keys_host(i) = RowKey2D{i + row_offset};
    row_ptr_host(i) = i;
    intervals_host(i) = Interval{0, 100};
  }
  row_ptr_host(n) = n;

  Kokkos::deep_copy(mesh.row_keys, row_keys_host);
  Kokkos::deep_copy(mesh.row_ptr, row_ptr_host);
  Kokkos::deep_copy(mesh.intervals, intervals_host);

  return mesh;
}

// ============================================================================
// Unified Benchmark - v1, v2, Stable
// ============================================================================

template <OverlapPattern Pattern>
class UnifiedIntersectionBenchmark : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State& state) override {
    const int n = state.range(0);

    // Generate test data once (shared by all algorithms)
    mesh_a_ = generate_mesh_2d<Kokkos::DefaultExecutionSpace::memory_space>(n, Pattern, 0);
    mesh_b_ = generate_mesh_2d<Kokkos::DefaultExecutionSpace::memory_space>(n, Pattern, 1);

    // Create v2 workspace (reused across iterations)
    workspace_ = v2::MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space>();
  }

  void TearDown(const benchmark::State&) override {}

protected:
  Mesh2DDevice mesh_a_, mesh_b_;
  v2::MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space> workspace_;
};

// ========================================================================
// v1 Benchmark
// ========================================================================

BENCHMARK_TEMPLATE_F(UnifiedIntersectionBenchmark, V1, OverlapPattern::FULL_OVERLAP)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v1::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (mesh_a_.num_intervals + mesh_b_.num_intervals) * sizeof(Interval));
}

BENCHMARK_REGISTER_F(UnifiedIntersectionBenchmark, V1)
    ->Range(64, 8192)
    ->Unit(benchmark::kMillisecond);

// ========================================================================
// v2 Benchmark (with workspace)
// ========================================================================

BENCHMARK_TEMPLATE_F(UnifiedIntersectionBenchmark, V2, OverlapPattern::FULL_OVERLAP)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v2::intersect_meshes_2d(mesh_a_, mesh_b_, workspace_);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (mesh_a_.num_intervals + mesh_b_.num_intervals) * sizeof(Interval));
}

BENCHMARK_REGISTER_F(UnifiedIntersectionBenchmark, V2)
    ->Range(64, 8192)
    ->Unit(benchmark::kMillisecond);

// ========================================================================
// Partial Overlap Benchmarks
// ========================================================================

using PartialOverlapBench = UnifiedIntersectionBenchmark<OverlapPattern::PARTIAL_OVERLAP>;

BENCHMARK_F(PartialOverlapBench, V1)(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v1::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_F(PartialOverlapBench, V2)(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v2::intersect_meshes_2d(mesh_a_, mesh_b_, workspace_);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(PartialOverlapBench, V1)->Range(64, 8192)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(PartialOverlapBench, V2)->Range(64, 8192)->Unit(benchmark::kMillisecond);

// ========================================================================
// Minimal Overlap Benchmarks
// ========================================================================

using MinimalOverlapBench = UnifiedIntersectionBenchmark<OverlapPattern::MINIMAL_OVERLAP>;

BENCHMARK_F(MinimalOverlapBench, V1)(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v1::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_F(MinimalOverlapBench, V2)(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v2::intersect_meshes_2d(mesh_a_, mesh_b_, workspace_);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(MinimalOverlapBench, V1)->Range(64, 8192)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(MinimalOverlapBench, V2)->Range(64, 8192)->Unit(benchmark::kMillisecond);

// ========================================================================
// 3D Benchmarks
// ========================================================================

static void BM_Intersection3D_V1_V2_FullOverlap(benchmark::State& state) {
  const int n = state.range(0);

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

  auto A_keys_h = Kokkos::create_mirror_view(A.row_keys);
  auto A_ptr_h = Kokkos::create_mirror_view(A.row_ptr);
  auto A_int_h = Kokkos::create_mirror_view(A.intervals);
  auto B_keys_h = Kokkos::create_mirror_view(B.row_keys);
  auto B_ptr_h = Kokkos::create_mirror_view(B.row_ptr);
  auto B_int_h = Kokkos::create_mirror_view(B.intervals);

  for (int i = 0; i < n; ++i) {
    A_keys_h(i) = RowKey3D{i, i % 10};
    A_ptr_h(i) = i;
    A_int_h(i) = Interval{0, 100};

    B_keys_h(i) = RowKey3D{i, i % 10};
    B_ptr_h(i) = i;
    B_int_h(i) = Interval{50, 150};
  }
  A_ptr_h(n) = n;
  B_ptr_h(n) = n;

  Kokkos::deep_copy(A.row_keys, A_keys_h);
  Kokkos::deep_copy(A.row_ptr, A_ptr_h);
  Kokkos::deep_copy(A.intervals, A_int_h);
  Kokkos::deep_copy(B.row_keys, B_keys_h);
  Kokkos::deep_copy(B.row_ptr, B_ptr_h);
  Kokkos::deep_copy(B.intervals, B_int_h);

  for (auto _ : state) {
    auto result = v1::intersect_meshes_3d(A, B);
    benchmark::DoNotOptimize(result.num_intervals);
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (A.num_intervals + B.num_intervals) * sizeof(Interval));
}

static void BM_Intersection3D_V2_FullOverlap(benchmark::State& state) {
  const int n = state.range(0);

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

  auto A_keys_h = Kokkos::create_mirror_view(A.row_keys);
  auto A_ptr_h = Kokkos::create_mirror_view(A.row_ptr);
  auto A_int_h = Kokkos::create_mirror_view(A.intervals);
  auto B_keys_h = Kokkos::create_mirror_view(B.row_keys);
  auto B_ptr_h = Kokkos::create_mirror_view(B.row_ptr);
  auto B_int_h = Kokkos::create_mirror_view(B.intervals);

  for (int i = 0; i < n; ++i) {
    A_keys_h(i) = RowKey3D{i, i % 10};
    A_ptr_h(i) = i;
    A_int_h(i) = Interval{0, 100};

    B_keys_h(i) = RowKey3D{i, i % 10};
    B_ptr_h(i) = i;
    B_int_h(i) = Interval{50, 150};
  }
  A_ptr_h(n) = n;
  B_ptr_h(n) = n;

  Kokkos::deep_copy(A.row_keys, A_keys_h);
  Kokkos::deep_copy(A.row_ptr, A_ptr_h);
  Kokkos::deep_copy(A.intervals, A_int_h);
  Kokkos::deep_copy(B.row_keys, B_keys_h);
  Kokkos::deep_copy(B.row_ptr, B_ptr_h);
  Kokkos::deep_copy(B.intervals, B_int_h);

  v2::MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space> ws;

  for (auto _ : state) {
    auto result = v2::intersect_meshes_3d(A, B, ws);
    benchmark::DoNotOptimize(result.num_intervals);
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (A.num_intervals + B.num_intervals) * sizeof(Interval));
}

BENCHMARK(BM_Intersection3D_V1_V2_FullOverlap)->Range(64, 8192)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Intersection3D_V2_FullOverlap)->Range(64, 8192)->Unit(benchmark::kMillisecond);

// ========================================================================
// Large-scale 2D-only Benchmarks (2D can handle much larger meshes)
// ========================================================================

static void BM_Intersection2D_Large_V1(benchmark::State& state) {
  const int n = state.range(0);

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

  auto A_keys_h = Kokkos::create_mirror_view(A.row_keys);
  auto A_ptr_h = Kokkos::create_mirror_view(A.row_ptr);
  auto A_int_h = Kokkos::create_mirror_view(A.intervals);
  auto B_keys_h = Kokkos::create_mirror_view(B.row_keys);
  auto B_ptr_h = Kokkos::create_mirror_view(B.row_ptr);
  auto B_int_h = Kokkos::create_mirror_view(B.intervals);

  for (int i = 0; i < n; ++i) {
    A_keys_h(i) = RowKey2D{i};
    A_ptr_h(i) = i;
    A_int_h(i) = Interval{0, 100};

    B_keys_h(i) = RowKey2D{i};
    B_ptr_h(i) = i;
    B_int_h(i) = Interval{50, 150};
  }
  A_ptr_h(n) = n;
  B_ptr_h(n) = n;

  Kokkos::deep_copy(A.row_keys, A_keys_h);
  Kokkos::deep_copy(A.row_ptr, A_ptr_h);
  Kokkos::deep_copy(A.intervals, A_int_h);
  Kokkos::deep_copy(B.row_keys, B_keys_h);
  Kokkos::deep_copy(B.row_ptr, B_ptr_h);
  Kokkos::deep_copy(B.intervals, B_int_h);

  for (auto _ : state) {
    auto result = v1::intersect_meshes_2d(A, B);
    benchmark::DoNotOptimize(result.num_intervals);
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (A.num_intervals + B.num_intervals) * sizeof(Interval));
}

static void BM_Intersection2D_Large_V2(benchmark::State& state) {
  const int n = state.range(0);

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

  auto A_keys_h = Kokkos::create_mirror_view(A.row_keys);
  auto A_ptr_h = Kokkos::create_mirror_view(A.row_ptr);
  auto A_int_h = Kokkos::create_mirror_view(A.intervals);
  auto B_keys_h = Kokkos::create_mirror_view(B.row_keys);
  auto B_ptr_h = Kokkos::create_mirror_view(B.row_ptr);
  auto B_int_h = Kokkos::create_mirror_view(B.intervals);

  for (int i = 0; i < n; ++i) {
    A_keys_h(i) = RowKey2D{i};
    A_ptr_h(i) = i;
    A_int_h(i) = Interval{0, 100};

    B_keys_h(i) = RowKey2D{i};
    B_ptr_h(i) = i;
    B_int_h(i) = Interval{50, 150};
  }
  A_ptr_h(n) = n;
  B_ptr_h(n) = n;

  Kokkos::deep_copy(A.row_keys, A_keys_h);
  Kokkos::deep_copy(A.row_ptr, A_ptr_h);
  Kokkos::deep_copy(A.intervals, A_int_h);
  Kokkos::deep_copy(B.row_keys, B_keys_h);
  Kokkos::deep_copy(B.row_ptr, B_ptr_h);
  Kokkos::deep_copy(B.intervals, B_int_h);

  v2::MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space> ws;

  for (auto _ : state) {
    auto result = v2::intersect_meshes_2d(A, B, ws);
    benchmark::DoNotOptimize(result.num_intervals);
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (A.num_intervals + B.num_intervals) * sizeof(Interval));
}

// Large 2D benchmarks: 16K, 32K, 64K, 128K rows
BENCHMARK(BM_Intersection2D_Large_V1)->RangeMultiplier(2)->Range(16384, 131072)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Intersection2D_Large_V2)->RangeMultiplier(2)->Range(16384, 131072)->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    Kokkos::finalize();
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  // Note: Not calling Kokkos::finalize() here because Fixtures may still hold Views
  // that will be destroyed after main() returns.
  return 0;
}

#endif // SUBSETIX_ENABLE_EXPERIMENTAL
