// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#include <benchmark/benchmark.h>
#include <playground/subsetix/csr/intersection/algorithm/v1.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v2.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v3.hpp>
#include <intersection/test_random_mesh_generator.hpp>
#include <Kokkos_Core.hpp>
#include <vector>

// Bring version namespaces into scope
using namespace playground::subsetix::csr::intersection;
using namespace playground::subsetix::csr::intersection::v1;
using namespace playground::subsetix::csr::intersection::v2;
using namespace playground::subsetix::csr::intersection::v3;
using namespace playground::subsetix::csr::intersection::test;

// Type aliases for convenience
using Coord = int32_t;
using IntervalType = playground::subsetix::csr::intersection::Interval<Coord>;

// ============================================================================
// Conversion helpers for different versions
// ============================================================================

namespace benchmark_helpers {

// v1 conversion
inline v1::Mesh2DDevice from_common_2d_v1(const DefaultCommonMesh2D& mesh) {
  return MeshConverter2D<v1::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}
inline v1::Mesh3DDevice from_common_3d_v1(const DefaultCommonMesh3D& mesh) {
  return MeshConverter3D<v1::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}

// v2 conversion
inline v2::Mesh2DDevice from_common_2d_v2(const DefaultCommonMesh2D& mesh) {
  return MeshConverter2D<v2::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}
inline v2::Mesh3DDevice from_common_3d_v2(const DefaultCommonMesh3D& mesh) {
  return MeshConverter3D<v2::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}

// v3 conversion
inline v3::Mesh2DDevice from_common_2d_v3(const DefaultCommonMesh2D& mesh) {
  return MeshConverter2D<v3::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}
inline v3::Mesh3DDevice from_common_3d_v3(const DefaultCommonMesh3D& mesh) {
  return MeshConverter3D<v3::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
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

// v1 2D fixture
template <typename GetConfigFunc>
class V1RandomMeshBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common_a = RandomMeshGenerator::generate_2d(cfg);
    cfg.seed++;
    auto common_b = RandomMeshGenerator::generate_2d(cfg);
    mesh_a_ = benchmark_helpers::from_common_2d_v1(common_a);
    mesh_b_ = benchmark_helpers::from_common_2d_v1(common_b);
  }
  void TearDown(const benchmark::State&) override {}
protected:
  v1::Mesh2DDevice mesh_a_, mesh_b_;
};

// v2 2D fixture
template <typename GetConfigFunc>
class V2RandomMeshBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common_a = RandomMeshGenerator::generate_2d(cfg);
    cfg.seed++;
    auto common_b = RandomMeshGenerator::generate_2d(cfg);
    mesh_a_ = benchmark_helpers::from_common_2d_v2(common_a);
    mesh_b_ = benchmark_helpers::from_common_2d_v2(common_b);
  }
  void TearDown(const benchmark::State&) override {}
protected:
  v2::Mesh2DDevice mesh_a_, mesh_b_;
};

// v3 2D fixture
template <typename GetConfigFunc>
class V3RandomMeshBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common_a = RandomMeshGenerator::generate_2d(cfg);
    cfg.seed++;
    auto common_b = RandomMeshGenerator::generate_2d(cfg);
    mesh_a_ = benchmark_helpers::from_common_2d_v3(common_a);
    mesh_b_ = benchmark_helpers::from_common_2d_v3(common_b);
  }
  void TearDown(const benchmark::State&) override {}
protected:
  v3::Mesh2DDevice mesh_a_, mesh_b_;
};

// v1 3D fixture
template <typename GetConfigFunc>
class V1RandomMeshBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common_a = RandomMeshGenerator::generate_3d(cfg);
    cfg.seed++;
    auto common_b = RandomMeshGenerator::generate_3d(cfg);
    mesh_a_ = benchmark_helpers::from_common_3d_v1(common_a);
    mesh_b_ = benchmark_helpers::from_common_3d_v1(common_b);
  }
  void TearDown(const benchmark::State&) override {}
protected:
  v1::Mesh3DDevice mesh_a_, mesh_b_;
};

// v2 3D fixture
template <typename GetConfigFunc>
class V2RandomMeshBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common_a = RandomMeshGenerator::generate_3d(cfg);
    cfg.seed++;
    auto common_b = RandomMeshGenerator::generate_3d(cfg);
    mesh_a_ = benchmark_helpers::from_common_3d_v2(common_a);
    mesh_b_ = benchmark_helpers::from_common_3d_v2(common_b);
  }
  void TearDown(const benchmark::State&) override {}
protected:
  v2::Mesh3DDevice mesh_a_, mesh_b_;
};

// v3 3D fixture
template <typename GetConfigFunc>
class V3RandomMeshBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common_a = RandomMeshGenerator::generate_3d(cfg);
    cfg.seed++;
    auto common_b = RandomMeshGenerator::generate_3d(cfg);
    mesh_a_ = benchmark_helpers::from_common_3d_v3(common_a);
    mesh_b_ = benchmark_helpers::from_common_3d_v3(common_b);
  }
  void TearDown(const benchmark::State&) override {}
protected:
  v3::Mesh3DDevice mesh_a_, mesh_b_;
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

BENCHMARK_TEMPLATE_F(V1RandomMeshBenchmark2D, V1_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v1::intersect_meshes_2d(mesh_a_, mesh_b_);
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

BENCHMARK_TEMPLATE_F(V2RandomMeshBenchmark2D, V2_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v2::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V3RandomMeshBenchmark2D, V3_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v3::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V1RandomMeshBenchmark2D, V1_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v1::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V2RandomMeshBenchmark2D, V2_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v2::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V3RandomMeshBenchmark2D, V3_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v3::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V1RandomMeshBenchmark2D, V1_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v1::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V2RandomMeshBenchmark2D, V2_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v2::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V3RandomMeshBenchmark2D, V3_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v3::intersect_meshes_2d(mesh_a_, mesh_b_);
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

BENCHMARK_TEMPLATE_F(V1RandomMeshBenchmark3D, V1_3D_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v1::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V2RandomMeshBenchmark3D, V2_3D_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v2::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V3RandomMeshBenchmark3D, V3_3D_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v3::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V1RandomMeshBenchmark3D, V1_3D_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v1::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V2RandomMeshBenchmark3D, V2_3D_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v2::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V3RandomMeshBenchmark3D, V3_3D_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v3::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V1RandomMeshBenchmark3D, V1_3D_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v1::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V2RandomMeshBenchmark3D, V2_3D_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v2::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
  // ns_per_interval is automatically computed by Google Benchmark as time / items_processed
}

BENCHMARK_TEMPLATE_F(V3RandomMeshBenchmark3D, V3_3D_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v3::intersect_meshes_3d(mesh_a_, mesh_b_);
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

BENCHMARK_TEMPLATE_F(V1RandomMeshBenchmark2D, V1_ExtraLargeConfig, GetExtraLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v1::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V2RandomMeshBenchmark2D, V2_ExtraLargeConfig, GetExtraLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v2::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V3RandomMeshBenchmark2D, V3_ExtraLargeConfig, GetExtraLargeConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v3::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
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
