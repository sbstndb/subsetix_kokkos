// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#include <benchmark/benchmark.h>
#include <playground/subsetix/csr/intersection/algorithm/baseline.hpp>
#include <playground/subsetix/csr/intersection/algorithm/optimized.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v4_hash.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v5_parallel_merge.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v6_direct_index.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v7_soa_optimized.hpp>
#include <intersection/test_random_mesh_generator.hpp>
#include <Kokkos_Core.hpp>
#include <vector>

// Bring version namespaces into scope
using namespace playground::subsetix::csr::intersection;
using namespace playground::subsetix::csr::intersection::baseline;
using namespace playground::subsetix::csr::intersection::optimized;
using namespace playground::subsetix::csr::intersection::hash_based;
using namespace playground::subsetix::csr::intersection::parallel_merge;
using namespace playground::subsetix::csr::intersection::direct_index;
using namespace playground::subsetix::csr::intersection::soa_optimized;
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

// v4 (hash-based) conversion
inline hash_based::Mesh2DDevice from_common_2d_v4(const DefaultCommonMesh2D& mesh) {
  return MeshConverter2D<hash_based::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}
inline hash_based::Mesh3DDevice from_common_3d_v4(const DefaultCommonMesh3D& mesh) {
  return MeshConverter3D<hash_based::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}

// v5 (parallel merge) conversion
inline parallel_merge::Mesh2DDevice from_common_2d_v5(const DefaultCommonMesh2D& mesh) {
  return MeshConverter2D<parallel_merge::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}
inline parallel_merge::Mesh3DDevice from_common_3d_v5(const DefaultCommonMesh3D& mesh) {
  return MeshConverter3D<parallel_merge::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}

// v6 (direct index) conversion
inline direct_index::Mesh2DDevice from_common_2d_v6(const DefaultCommonMesh2D& mesh) {
  return MeshConverter2D<direct_index::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}
inline direct_index::Mesh3DDevice from_common_3d_v6(const DefaultCommonMesh3D& mesh) {
  return MeshConverter3D<direct_index::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}

// v7 (SOA optimized) uses optimized format with soa_optimized functions
inline optimized::Mesh2DDevice from_common_2d_v7(const DefaultCommonMesh2D& mesh) {
  return MeshConverter2D<optimized::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}
inline optimized::Mesh3DDevice from_common_3d_v7(const DefaultCommonMesh3D& mesh) {
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

// v4 (hash-based) 2D fixture
template <typename GetConfigFunc>
class V4HashRegularMeshBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common = RegularMeshGenerator::generate_2d(cfg);
    mesh_a_ = benchmark_helpers::from_common_2d_v4(common);
    mesh_b_ = mesh_a_;
  }
  void TearDown(const benchmark::State&) override {}
protected:
  hash_based::Mesh2DDevice mesh_a_, mesh_b_;
};

// v4 3D fixture
template <typename GetConfigFunc>
class V4HashRegularMeshBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common = RegularMeshGenerator::generate_3d(cfg);
    mesh_a_ = benchmark_helpers::from_common_3d_v4(common);
    mesh_b_ = mesh_a_;
  }
  void TearDown(const benchmark::State&) override {}
protected:
  hash_based::Mesh3DDevice mesh_a_, mesh_b_;
};

// v5 (parallel merge) 2D fixture
template <typename GetConfigFunc>
class V5ParallelMergeRegularMeshBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common = RegularMeshGenerator::generate_2d(cfg);
    mesh_a_ = benchmark_helpers::from_common_2d_v5(common);
    mesh_b_ = mesh_a_;
  }
  void TearDown(const benchmark::State&) override {}
protected:
  parallel_merge::Mesh2DDevice mesh_a_, mesh_b_;
};

// v5 3D fixture
template <typename GetConfigFunc>
class V5ParallelMergeRegularMeshBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common = RegularMeshGenerator::generate_3d(cfg);
    mesh_a_ = benchmark_helpers::from_common_3d_v5(common);
    mesh_b_ = mesh_a_;
  }
  void TearDown(const benchmark::State&) override {}
protected:
  parallel_merge::Mesh3DDevice mesh_a_, mesh_b_;
};

// v6 (direct index) 2D fixture
template <typename GetConfigFunc>
class V6DirectIndexRegularMeshBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common = RegularMeshGenerator::generate_2d(cfg);
    mesh_a_ = benchmark_helpers::from_common_2d_v6(common);
    mesh_b_ = mesh_a_;
  }
  void TearDown(const benchmark::State&) override {}
protected:
  direct_index::Mesh2DDevice mesh_a_, mesh_b_;
};

// v6 3D fixture
template <typename GetConfigFunc>
class V6DirectIndexRegularMeshBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common = RegularMeshGenerator::generate_3d(cfg);
    mesh_a_ = benchmark_helpers::from_common_3d_v6(common);
    mesh_b_ = mesh_a_;
  }
  void TearDown(const benchmark::State&) override {}
protected:
  direct_index::Mesh3DDevice mesh_a_, mesh_b_;
};

// v7 (SOA optimized) 2D fixture
template <typename GetConfigFunc>
class V7SoaOptimizedRegularMeshBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common = RegularMeshGenerator::generate_2d(cfg);
    mesh_a_ = benchmark_helpers::from_common_2d_v7(common);
    mesh_b_ = mesh_a_;
  }
  void TearDown(const benchmark::State&) override {}
protected:
  optimized::Mesh2DDevice mesh_a_, mesh_b_;
};

// v7 3D fixture
template <typename GetConfigFunc>
class V7SoaOptimizedRegularMeshBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common = RegularMeshGenerator::generate_3d(cfg);
    mesh_a_ = benchmark_helpers::from_common_3d_v7(common);
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

// ============================================================================
// v4-v7 Benchmarks (Large only for comparison with random benchmarks)
// ============================================================================

// 2D Large benchmarks for v4-v7
BENCHMARK_TEMPLATE_F(V4HashRegularMeshBenchmark2D, V4Hash_Regular_LargeConfig, GetLargeRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = hash_based::intersect_meshes<2>(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V5ParallelMergeRegularMeshBenchmark2D, V5ParallelMerge_Regular_LargeConfig, GetLargeRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = parallel_merge::intersect_meshes<2>(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V6DirectIndexRegularMeshBenchmark2D, V6DirectIndex_Regular_LargeConfig, GetLargeRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = direct_index::intersect_meshes<2>(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V7SoaOptimizedRegularMeshBenchmark2D, V7SoaOptimized_Regular_LargeConfig, GetLargeRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = soa_optimized::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

// 3D Large benchmarks for v4-v7
BENCHMARK_TEMPLATE_F(V4HashRegularMeshBenchmark3D, V4Hash_3D_Regular_LargeConfig, GetLargeRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = hash_based::intersect_meshes<3>(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V5ParallelMergeRegularMeshBenchmark3D, V5ParallelMerge_3D_Regular_LargeConfig, GetLargeRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = parallel_merge::intersect_meshes<3>(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V6DirectIndexRegularMeshBenchmark3D, V6DirectIndex_3D_Regular_LargeConfig, GetLargeRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = direct_index::intersect_meshes<3>(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V7SoaOptimizedRegularMeshBenchmark3D, V7SoaOptimized_3D_Regular_LargeConfig, GetLargeRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = soa_optimized::intersect_meshes_3d(mesh_a_, mesh_b_);
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
