// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#include <benchmark/benchmark.h>
#include <playground/subsetix/csr/intersection/algorithm/baseline.hpp>
#include <playground/subsetix/csr/intersection/algorithm/optimized.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v4_hash.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v5_parallel_merge.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v6_direct_index.hpp>
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

// v4_hash conversion
inline hash_based::Mesh2DDevice from_common_2d_v4_hash(const DefaultCommonMesh2D& mesh) {
  return MeshConverter2D<hash_based::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}
inline hash_based::Mesh3DDevice from_common_3d_v4_hash(const DefaultCommonMesh3D& mesh) {
  return MeshConverter3D<hash_based::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}

// v5_parallel_merge conversion
inline parallel_merge::Mesh2DDevice from_common_2d_v5_parallel_merge(const DefaultCommonMesh2D& mesh) {
  return MeshConverter2D<parallel_merge::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}
inline parallel_merge::Mesh3DDevice from_common_3d_v5_parallel_merge(const DefaultCommonMesh3D& mesh) {
  return MeshConverter3D<parallel_merge::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}

// v6_direct_index conversion
inline direct_index::Mesh2DDevice from_common_2d_v6_direct_index(const DefaultCommonMesh2D& mesh) {
  return MeshConverter2D<direct_index::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}
inline direct_index::Mesh3DDevice from_common_3d_v6_direct_index(const DefaultCommonMesh3D& mesh) {
  return MeshConverter3D<direct_index::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}

} // namespace benchmark_helpers

// ============================================================================
// Random Mesh Benchmark with Config-based Sizes
// ============================================================================

/**
 * @brief Benchmark fixture using a single random mesh with predefined configuration
 *
 * Strategy:
 * - Generate ONE random mesh in SetUp (memory efficient)
 * - Reuse the same mesh for all iterations (cache warm, but reproducible)
 * - Different benchmarks use different configs (Small/Medium/Large)
 *
 * Configurations (30% sparsity):
 * - SmallConfig: ~19 rows (2D), ~1229 rows (3D), y_max=64, z_max=64
 * - MediumConfig: ~154 rows (2D), ~78643 rows (3D), y_max=512, z_max=512
 * - LargeConfig: ~1229 rows (2D), ~5.0M rows (3D), y_max=4096, z_max=4096
 */
// ============================================================================
// Version-specific benchmark fixtures
// ============================================================================

// baseline 2D fixture
template <typename GetConfigFunc>
class BaselineRandomMeshBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common_a = RandomMeshGenerator::generate_2d(cfg);
    cfg.seed++;
    auto common_b = RandomMeshGenerator::generate_2d(cfg);
    mesh_a_ = benchmark_helpers::from_common_2d_baseline(common_a);
    mesh_b_ = benchmark_helpers::from_common_2d_baseline(common_b);
  }
  void TearDown(const benchmark::State&) override {}
protected:
  baseline::Mesh2DDevice mesh_a_, mesh_b_;
};

// optimized 2D fixture
template <typename GetConfigFunc>
class OptimizedRandomMeshBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common_a = RandomMeshGenerator::generate_2d(cfg);
    cfg.seed++;
    auto common_b = RandomMeshGenerator::generate_2d(cfg);
    mesh_a_ = benchmark_helpers::from_common_2d_optimized(common_a);
    mesh_b_ = benchmark_helpers::from_common_2d_optimized(common_b);
  }
  void TearDown(const benchmark::State&) override {}
protected:
  optimized::Mesh2DDevice mesh_a_, mesh_b_;
};

// baseline 3D fixture
template <typename GetConfigFunc>
class BaselineRandomMeshBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common_a = RandomMeshGenerator::generate_3d(cfg);
    cfg.seed++;
    auto common_b = RandomMeshGenerator::generate_3d(cfg);
    mesh_a_ = benchmark_helpers::from_common_3d_baseline(common_a);
    mesh_b_ = benchmark_helpers::from_common_3d_baseline(common_b);
  }
  void TearDown(const benchmark::State&) override {}
protected:
  baseline::Mesh3DDevice mesh_a_, mesh_b_;
};

// optimized 3D fixture
template <typename GetConfigFunc>
class OptimizedRandomMeshBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common_a = RandomMeshGenerator::generate_3d(cfg);
    cfg.seed++;
    auto common_b = RandomMeshGenerator::generate_3d(cfg);
    mesh_a_ = benchmark_helpers::from_common_3d_optimized(common_a);
    mesh_b_ = benchmark_helpers::from_common_3d_optimized(common_b);
  }
  void TearDown(const benchmark::State&) override {}
protected:
  optimized::Mesh3DDevice mesh_a_, mesh_b_;
};

// v4_hash 2D fixture
template <typename GetConfigFunc>
class V4HashRandomMeshBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common_a = RandomMeshGenerator::generate_2d(cfg);
    cfg.seed++;
    auto common_b = RandomMeshGenerator::generate_2d(cfg);
    mesh_a_ = benchmark_helpers::from_common_2d_v4_hash(common_a);
    mesh_b_ = benchmark_helpers::from_common_2d_v4_hash(common_b);
  }
  void TearDown(const benchmark::State&) override {}
protected:
  hash_based::Mesh2DDevice mesh_a_, mesh_b_;
};

// v5_parallel_merge 2D fixture
template <typename GetConfigFunc>
class V5ParallelMergeRandomMeshBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common_a = RandomMeshGenerator::generate_2d(cfg);
    cfg.seed++;
    auto common_b = RandomMeshGenerator::generate_2d(cfg);
    mesh_a_ = benchmark_helpers::from_common_2d_v5_parallel_merge(common_a);
    mesh_b_ = benchmark_helpers::from_common_2d_v5_parallel_merge(common_b);
  }
  void TearDown(const benchmark::State&) override {}
protected:
  parallel_merge::Mesh2DDevice mesh_a_, mesh_b_;
};

// v6_direct_index 2D fixture
template <typename GetConfigFunc>
class V6DirectIndexRandomMeshBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common_a = RandomMeshGenerator::generate_2d(cfg);
    cfg.seed++;
    auto common_b = RandomMeshGenerator::generate_2d(cfg);
    mesh_a_ = benchmark_helpers::from_common_2d_v6_direct_index(common_a);
    mesh_b_ = benchmark_helpers::from_common_2d_v6_direct_index(common_b);
  }
  void TearDown(const benchmark::State&) override {}
protected:
  direct_index::Mesh2DDevice mesh_a_, mesh_b_;
};

// v4_hash 3D fixture
template <typename GetConfigFunc>
class V4HashRandomMeshBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common_a = RandomMeshGenerator::generate_3d(cfg);
    cfg.seed++;
    auto common_b = RandomMeshGenerator::generate_3d(cfg);
    mesh_a_ = benchmark_helpers::from_common_3d_v4_hash(common_a);
    mesh_b_ = benchmark_helpers::from_common_3d_v4_hash(common_b);
  }
  void TearDown(const benchmark::State&) override {}
protected:
  hash_based::Mesh3DDevice mesh_a_, mesh_b_;
};

// v5_parallel_merge 3D fixture
template <typename GetConfigFunc>
class V5ParallelMergeRandomMeshBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common_a = RandomMeshGenerator::generate_3d(cfg);
    cfg.seed++;
    auto common_b = RandomMeshGenerator::generate_3d(cfg);
    mesh_a_ = benchmark_helpers::from_common_3d_v5_parallel_merge(common_a);
    mesh_b_ = benchmark_helpers::from_common_3d_v5_parallel_merge(common_b);
  }
  void TearDown(const benchmark::State&) override {}
protected:
  parallel_merge::Mesh3DDevice mesh_a_, mesh_b_;
};

// v6_direct_index 3D fixture
template <typename GetConfigFunc>
class V6DirectIndexRandomMeshBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common_a = RandomMeshGenerator::generate_3d(cfg);
    cfg.seed++;
    auto common_b = RandomMeshGenerator::generate_3d(cfg);
    mesh_a_ = benchmark_helpers::from_common_3d_v6_direct_index(common_a);
    mesh_b_ = benchmark_helpers::from_common_3d_v6_direct_index(common_b);
  }
  void TearDown(const benchmark::State&) override {}
protected:
  direct_index::Mesh3DDevice mesh_a_, mesh_b_;
};

// ============================================================================
// Config Providers
// ============================================================================

struct GetSmallConfig {
  RandomMeshConfig operator()() const { return SmallConfig(); }
};

struct GetMediumConfig {
  RandomMeshConfig operator()() const { return MediumConfig(); }
};

struct GetLargeConfig {
  RandomMeshConfig operator()() const { return LargeConfig(); }
};

struct GetExtraLargeConfig {
  RandomMeshConfig operator()() const { return ExtraLargeConfig(); }
};

// ============================================================================
// 2D Benchmarks
// ============================================================================

BENCHMARK_TEMPLATE_F(BaselineRandomMeshBenchmark2D, Baseline_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = baseline::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  // Set items processed to total intervals across all iterations
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // Google Benchmark now displays:
  // - items_per_second = intervals processed per second
  // - ns_per_interval can be computed as: 1e9 / items_per_second
}

BENCHMARK_TEMPLATE_F(OptimizedRandomMeshBenchmark2D, Optimized_SmallConfig, GetSmallConfig)
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
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V4HashRandomMeshBenchmark2D, V4Hash_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = hash_based::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V5ParallelMergeRandomMeshBenchmark2D, V5ParallelMerge_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = parallel_merge::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V6DirectIndexRandomMeshBenchmark2D, V6DirectIndex_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = direct_index::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(BaselineRandomMeshBenchmark2D, Baseline_MediumConfig, GetMediumConfig)
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
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(OptimizedRandomMeshBenchmark2D, Optimized_MediumConfig, GetMediumConfig)
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
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V4HashRandomMeshBenchmark2D, V4Hash_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = hash_based::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V5ParallelMergeRandomMeshBenchmark2D, V5ParallelMerge_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = parallel_merge::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V6DirectIndexRandomMeshBenchmark2D, V6DirectIndex_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = direct_index::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(BaselineRandomMeshBenchmark2D, Baseline_LargeConfig, GetLargeConfig)
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
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(OptimizedRandomMeshBenchmark2D, Optimized_LargeConfig, GetLargeConfig)
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
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V4HashRandomMeshBenchmark2D, V4Hash_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = hash_based::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V5ParallelMergeRandomMeshBenchmark2D, V5ParallelMerge_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = parallel_merge::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V6DirectIndexRandomMeshBenchmark2D, V6DirectIndex_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = direct_index::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

// ============================================================================
// 3D Benchmarks
// ============================================================================

BENCHMARK_TEMPLATE_F(BaselineRandomMeshBenchmark3D, Baseline_3D_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = baseline::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(OptimizedRandomMeshBenchmark3D, Optimized_3D_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = optimized::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V4HashRandomMeshBenchmark3D, V4Hash_3D_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = hash_based::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V5ParallelMergeRandomMeshBenchmark3D, V5ParallelMerge_3D_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = parallel_merge::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V6DirectIndexRandomMeshBenchmark3D, V6DirectIndex_3D_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = direct_index::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(BaselineRandomMeshBenchmark3D, Baseline_3D_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = baseline::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(OptimizedRandomMeshBenchmark3D, Optimized_3D_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = optimized::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V4HashRandomMeshBenchmark3D, V4Hash_3D_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = hash_based::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V5ParallelMergeRandomMeshBenchmark3D, V5ParallelMerge_3D_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = parallel_merge::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V6DirectIndexRandomMeshBenchmark3D, V6DirectIndex_3D_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = direct_index::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(BaselineRandomMeshBenchmark3D, Baseline_3D_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = baseline::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(OptimizedRandomMeshBenchmark3D, Optimized_3D_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = optimized::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V4HashRandomMeshBenchmark3D, V4Hash_3D_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = hash_based::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V5ParallelMergeRandomMeshBenchmark3D, V5ParallelMerge_3D_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = parallel_merge::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V6DirectIndexRandomMeshBenchmark3D, V6DirectIndex_3D_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = direct_index::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

// ============================================================================
// Extra Large Benchmarks (2x Large)
// ============================================================================

BENCHMARK_TEMPLATE_F(BaselineRandomMeshBenchmark2D, Baseline_ExtraLargeConfig, GetExtraLargeConfig)
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

BENCHMARK_TEMPLATE_F(OptimizedRandomMeshBenchmark2D, Optimized_ExtraLargeConfig, GetExtraLargeConfig)
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

BENCHMARK_TEMPLATE_F(V4HashRandomMeshBenchmark2D, V4Hash_ExtraLargeConfig, GetExtraLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = hash_based::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V5ParallelMergeRandomMeshBenchmark2D, V5ParallelMerge_ExtraLargeConfig, GetExtraLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = parallel_merge::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V6DirectIndexRandomMeshBenchmark2D, V6DirectIndex_ExtraLargeConfig, GetExtraLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = direct_index::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

// 3D ExtraLarge Config (~10M rows with 15% sparsity)
BENCHMARK_TEMPLATE_F(BaselineRandomMeshBenchmark3D, Baseline_3D_ExtraLargeConfig, GetExtraLargeConfig)
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

BENCHMARK_TEMPLATE_F(OptimizedRandomMeshBenchmark3D, Optimized_3D_ExtraLargeConfig, GetExtraLargeConfig)
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

BENCHMARK_TEMPLATE_F(V4HashRandomMeshBenchmark3D, V4Hash_3D_ExtraLargeConfig, GetExtraLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = hash_based::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V5ParallelMergeRandomMeshBenchmark3D, V5ParallelMerge_3D_ExtraLargeConfig, GetExtraLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = parallel_merge::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V6DirectIndexRandomMeshBenchmark3D, V6DirectIndex_3D_ExtraLargeConfig, GetExtraLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = direct_index::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

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
