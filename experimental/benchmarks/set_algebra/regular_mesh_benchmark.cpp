// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <benchmark/benchmark.h>
#include <experimental/subsetix/csr/set_algebra/v1.hpp>
#include <experimental/subsetix/csr/set_algebra/v2.hpp>
#include <experimental/subsetix/csr/set_algebra/v3.hpp>
#include <set_algebra/test_random_mesh_generator.hpp>
#include <Kokkos_Core.hpp>
#include <vector>

// Bring version namespaces into scope
using namespace experimental::subsetix::csr;
using namespace experimental::subsetix::csr::v1;
using namespace experimental::subsetix::csr::v2;
using namespace experimental::subsetix::csr::v3;
using namespace experimental::subsetix::csr::test;

// Type aliases for convenience
using Coord = int32_t;
using IntervalType = experimental::subsetix::csr::Interval<Coord>;

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

// v1 2D fixture
template <typename GetConfigFunc>
class V1RegularMeshBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common = RegularMeshGenerator::generate_2d(cfg);
    mesh_a_ = benchmark_helpers::from_common_2d_v1(common);
    // Self-intersection: mesh_b_ is a copy of mesh_a_
    mesh_b_ = mesh_a_;
  }
  void TearDown(const benchmark::State&) override {}
protected:
  v1::Mesh2DDevice mesh_a_, mesh_b_;
};

// v2 2D fixture
template <typename GetConfigFunc>
class V2RegularMeshBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common = RegularMeshGenerator::generate_2d(cfg);
    mesh_a_ = benchmark_helpers::from_common_2d_v2(common);
    mesh_b_ = mesh_a_;
  }
  void TearDown(const benchmark::State&) override {}
protected:
  v2::Mesh2DDevice mesh_a_, mesh_b_;
};

// v3 2D fixture
template <typename GetConfigFunc>
class V3RegularMeshBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common = RegularMeshGenerator::generate_2d(cfg);
    mesh_a_ = benchmark_helpers::from_common_2d_v3(common);
    mesh_b_ = mesh_a_;
  }
  void TearDown(const benchmark::State&) override {}
protected:
  v3::Mesh2DDevice mesh_a_, mesh_b_;
};

// v1 3D fixture
template <typename GetConfigFunc>
class V1RegularMeshBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common = RegularMeshGenerator::generate_3d(cfg);
    mesh_a_ = benchmark_helpers::from_common_3d_v1(common);
    mesh_b_ = mesh_a_;
  }
  void TearDown(const benchmark::State&) override {}
protected:
  v1::Mesh3DDevice mesh_a_, mesh_b_;
};

// v2 3D fixture
template <typename GetConfigFunc>
class V2RegularMeshBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common = RegularMeshGenerator::generate_3d(cfg);
    mesh_a_ = benchmark_helpers::from_common_3d_v2(common);
    mesh_b_ = mesh_a_;
  }
  void TearDown(const benchmark::State&) override {}
protected:
  v2::Mesh3DDevice mesh_a_, mesh_b_;
};

// v3 3D fixture
template <typename GetConfigFunc>
class V3RegularMeshBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    auto common = RegularMeshGenerator::generate_3d(cfg);
    mesh_a_ = benchmark_helpers::from_common_3d_v3(common);
    mesh_b_ = mesh_a_;
  }
  void TearDown(const benchmark::State&) override {}
protected:
  v3::Mesh3DDevice mesh_a_, mesh_b_;
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

// ============================================================================
// 2D Benchmarks
// ============================================================================

BENCHMARK_TEMPLATE_F(V1RegularMeshBenchmark2D, V1_Regular_SmallConfig, GetSmallRegularConfig)
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

BENCHMARK_TEMPLATE_F(V2RegularMeshBenchmark2D, V2_Regular_SmallConfig, GetSmallRegularConfig)
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

BENCHMARK_TEMPLATE_F(V3RegularMeshBenchmark2D, V3_Regular_SmallConfig, GetSmallRegularConfig)
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

BENCHMARK_TEMPLATE_F(V1RegularMeshBenchmark2D, V1_Regular_MediumConfig, GetMediumRegularConfig)
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

BENCHMARK_TEMPLATE_F(V2RegularMeshBenchmark2D, V2_Regular_MediumConfig, GetMediumRegularConfig)
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

BENCHMARK_TEMPLATE_F(V3RegularMeshBenchmark2D, V3_Regular_MediumConfig, GetMediumRegularConfig)
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

BENCHMARK_TEMPLATE_F(V1RegularMeshBenchmark2D, V1_Regular_LargeConfig, GetLargeRegularConfig)
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

BENCHMARK_TEMPLATE_F(V2RegularMeshBenchmark2D, V2_Regular_LargeConfig, GetLargeRegularConfig)
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

BENCHMARK_TEMPLATE_F(V3RegularMeshBenchmark2D, V3_Regular_LargeConfig, GetLargeRegularConfig)
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
// 3D Benchmarks
// ============================================================================

BENCHMARK_TEMPLATE_F(V1RegularMeshBenchmark3D, V1_3D_Regular_SmallConfig, GetSmallRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v1::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V2RegularMeshBenchmark3D, V2_3D_Regular_SmallConfig, GetSmallRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v2::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V3RegularMeshBenchmark3D, V3_3D_Regular_SmallConfig, GetSmallRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v3::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V1RegularMeshBenchmark3D, V1_3D_Regular_MediumConfig, GetMediumRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v1::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V2RegularMeshBenchmark3D, V2_3D_Regular_MediumConfig, GetMediumRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v2::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V3RegularMeshBenchmark3D, V3_3D_Regular_MediumConfig, GetMediumRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v3::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V1RegularMeshBenchmark3D, V1_3D_Regular_LargeConfig, GetLargeRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v1::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V2RegularMeshBenchmark3D, V2_3D_Regular_LargeConfig, GetLargeRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v2::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(V3RegularMeshBenchmark3D, V3_3D_Regular_LargeConfig, GetLargeRegularConfig)
(benchmark::State& state) {
  std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;
  for (auto _ : state) {
    auto result = v3::intersect_meshes_3d(mesh_a_, mesh_b_);
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

#endif // SUBSETIX_ENABLE_EXPERIMENTAL
