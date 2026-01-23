// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#include <benchmark/benchmark.h>
#include <playground/subsetix/csr/intersection/algorithm/baseline.hpp>
#include <playground/subsetix/csr/intersection/algorithm/optimized.hpp>
#include <intersection/test_random_mesh_generator.hpp>
#include <Kokkos_Core.hpp>
#include <vector>

// Bring version namespaces into scope
using namespace playground::subsetix::csr::intersection;
using namespace playground::subsetix::csr::intersection::baseline;
using namespace playground::subsetix::csr::intersection::optimized;
using namespace playground::subsetix::csr::intersection::test;

// Type aliases for convenience
using Coord = int32_t;
using IntervalType = playground::subsetix::csr::intersection::Interval<Coord>;

// ============================================================================
// Conversion helpers for different versions
// ============================================================================

namespace benchmark_helpers {

// baseline conversion
inline baseline::Mesh2DDevice from_common_2d_baseline(const DefaultCommonMesh2D& mesh) {
  return MeshConverter2D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}
inline baseline::Mesh3DDevice from_common_3d_baseline(const DefaultCommonMesh3D& mesh) {
  return MeshConverter3D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}

// optimized conversion
inline optimized::Mesh2DDevice from_common_2d_optimized(const DefaultCommonMesh2D& mesh) {
  return MeshConverter2D<optimized::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}
inline optimized::Mesh3DDevice from_common_3d_optimized(const DefaultCommonMesh3D& mesh) {
  return MeshConverter3D<optimized::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}

} // namespace benchmark_helpers

// ============================================================================
// Regular Mesh Benchmark with Config-based Sizes
// ============================================================================

/**
 * @brief Benchmark fixture using regular (dense) meshes for optimal performance
 *
 * Strategy:
 * - Generate ONE regular mesh in SetUp (memory efficient)
 * - Reuse the same mesh for all iterations (self-intersection A ∩ A)
 * - Different benchmarks use different configs (Small/Medium/Large)
 *
 * Regular meshes provide "best case" performance:
 * - All rows are present (100% density, no gaps)
 * - Each row has exactly one interval covering the full X range
 * - Self-intersection means perfect row/interval alignment
 * - No binary search misses
 * - Minimal memory overhead (1 interval per row)
 *
 * Configurations (matching random config grid extent but 100% dense):
 * - SmallRegularConfig: 64 rows (2D) or 64×64=4096 rows (3D)
 * - MediumRegularConfig: 512 rows (2D) or 512×512=262144 rows (3D)
 * - LargeRegularConfig: 4096 rows (2D) or 4096×4096=16.8M rows (3D)
 */

// ============================================================================
// Version-specific benchmark fixtures
// ============================================================================

// baseline 2D fixture
template <typename GetConfigFunc>
class BaselineRegularMeshBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common = RegularMeshGenerator::generate_2d(cfg);
    mesh_a_ = benchmark_helpers::from_common_2d_baseline(common);
    // Self-intersection: mesh_b_ is a copy of mesh_a_
    mesh_b_ = mesh_a_;
  }
  void TearDown(const benchmark::State&) override {}
protected:
  baseline::Mesh2DDevice mesh_a_, mesh_b_;
};

// optimized 2D fixture
template <typename GetConfigFunc>
class OptimizedRegularMeshBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common = RegularMeshGenerator::generate_2d(cfg);
    mesh_a_ = benchmark_helpers::from_common_2d_optimized(common);
    mesh_b_ = mesh_a_;
  }
  void TearDown(const benchmark::State&) override {}
protected:
  optimized::Mesh2DDevice mesh_a_, mesh_b_;
};

// baseline 3D fixture
template <typename GetConfigFunc>
class BaselineRegularMeshBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common = RegularMeshGenerator::generate_3d(cfg);
    mesh_a_ = benchmark_helpers::from_common_3d_baseline(common);
    mesh_b_ = mesh_a_;
  }
  void TearDown(const benchmark::State&) override {}
protected:
  baseline::Mesh3DDevice mesh_a_, mesh_b_;
};

// optimized 3D fixture
template <typename GetConfigFunc>
class OptimizedRegularMeshBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common = RegularMeshGenerator::generate_3d(cfg);
    mesh_a_ = benchmark_helpers::from_common_3d_optimized(common);
    mesh_b_ = mesh_a_;
  }
  void TearDown(const benchmark::State&) override {}
protected:
  optimized::Mesh3DDevice mesh_a_, mesh_b_;
};

// ============================================================================
// Config Providers
// ============================================================================

struct GetSmallRegularConfig {
  RegularMeshConfig operator()() const { return SmallRegularConfig(); }
};

struct GetMediumRegularConfig {
  RegularMeshConfig operator()() const { return MediumRegularConfig(); }
};

struct GetLargeRegularConfig {
  RegularMeshConfig operator()() const { return LargeRegularConfig(); }
};

struct GetExtraLargeRegularConfig {
  RegularMeshConfig operator()() const { return ExtraLargeRegularConfig(); }
};

// ============================================================================
// 2D Benchmarks
// ============================================================================

BENCHMARK_TEMPLATE_F(BaselineRegularMeshBenchmark2D, Baseline_Regular_SmallConfig, GetSmallRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = baseline::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(OptimizedRegularMeshBenchmark2D, Optimized_Regular_SmallConfig, GetSmallRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = optimized::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(BaselineRegularMeshBenchmark2D, Baseline_Regular_MediumConfig, GetMediumRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = baseline::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(OptimizedRegularMeshBenchmark2D, Optimized_Regular_MediumConfig, GetMediumRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = optimized::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(BaselineRegularMeshBenchmark2D, Baseline_Regular_LargeConfig, GetLargeRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = baseline::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(OptimizedRegularMeshBenchmark2D, Optimized_Regular_LargeConfig, GetLargeRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = optimized::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(BaselineRegularMeshBenchmark2D, Baseline_Regular_ExtraLargeConfig, GetExtraLargeRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = baseline::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(OptimizedRegularMeshBenchmark2D, Optimized_Regular_ExtraLargeConfig, GetExtraLargeRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = optimized::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

// ============================================================================
// 3D Benchmarks
// ============================================================================

BENCHMARK_TEMPLATE_F(BaselineRegularMeshBenchmark3D, Baseline_3D_Regular_SmallConfig, GetSmallRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = baseline::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(OptimizedRegularMeshBenchmark3D, Optimized_3D_Regular_SmallConfig, GetSmallRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = optimized::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(BaselineRegularMeshBenchmark3D, Baseline_3D_Regular_MediumConfig, GetMediumRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = baseline::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(OptimizedRegularMeshBenchmark3D, Optimized_3D_Regular_MediumConfig, GetMediumRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = optimized::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(BaselineRegularMeshBenchmark3D, Baseline_3D_Regular_LargeConfig, GetLargeRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = baseline::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(OptimizedRegularMeshBenchmark3D, Optimized_3D_Regular_LargeConfig, GetLargeRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = optimized::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

// Note: 3D ExtraLarge benchmarks removed due to GPU OOM (67M rows requires >512MB GPU memory)
// 2D ExtraLarge benchmarks are safe and functional

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    Kokkos::finalize();
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  // WORKAROUND: Call Kokkos::finalize() and use _exit() to skip static destructors.
  Kokkos::finalize();
  std::_Exit(0);
}
