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

using namespace experimental::subsetix::csr;
using namespace experimental::subsetix::csr::test;

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
 * Configurations:
 * - SmallConfig: 1250 rows, y_max=64, z_max=64
 * - MediumConfig: 78643 rows, y_max=512, z_max=512
 * - LargeConfig: 5M rows, y_max=4096, z_max=4096
 */
template <typename GetConfigFunc>
class RandomMeshBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    cfg.num_rows_min = cfg.num_rows_max;  // Exact row count

    // Generate a single pair of meshes
    auto common_a = RandomMeshGenerator::generate_2d(cfg);
    cfg.seed++;
    auto common_b = RandomMeshGenerator::generate_2d(cfg);

    // Convert to device meshes
    mesh_a_ = from_common_2d(common_a);
    mesh_b_ = from_common_2d(common_b);
  }

  void TearDown(const benchmark::State&) override {}

protected:
  Mesh2DDevice mesh_a_, mesh_b_;
};

template <typename GetConfigFunc>
class RandomMeshBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();
    cfg.num_rows_min = cfg.num_rows_max;

    auto common_a = RandomMeshGenerator::generate_3d(cfg);
    cfg.seed++;
    auto common_b = RandomMeshGenerator::generate_3d(cfg);

    mesh_a_ = from_common_3d(common_a);
    mesh_b_ = from_common_3d(common_b);
  }

  void TearDown(const benchmark::State&) override {}

protected:
  Mesh3DDevice mesh_a_, mesh_b_;
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

// ============================================================================
// 2D Benchmarks
// ============================================================================

BENCHMARK_TEMPLATE_F(RandomMeshBenchmark2D, V1_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v1::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (mesh_a_.num_intervals + mesh_b_.num_intervals) * sizeof(Interval));
}

BENCHMARK_TEMPLATE_F(RandomMeshBenchmark2D, V2_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v2::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (mesh_a_.num_intervals + mesh_b_.num_intervals) * sizeof(Interval));
}

BENCHMARK_TEMPLATE_F(RandomMeshBenchmark2D, V3_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v3::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (mesh_a_.num_intervals + mesh_b_.num_intervals) * sizeof(Interval));
}

BENCHMARK_TEMPLATE_F(RandomMeshBenchmark2D, V1_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v1::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (mesh_a_.num_intervals + mesh_b_.num_intervals) * sizeof(Interval));
}

BENCHMARK_TEMPLATE_F(RandomMeshBenchmark2D, V2_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v2::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (mesh_a_.num_intervals + mesh_b_.num_intervals) * sizeof(Interval));
}

BENCHMARK_TEMPLATE_F(RandomMeshBenchmark2D, V3_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v3::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (mesh_a_.num_intervals + mesh_b_.num_intervals) * sizeof(Interval));
}

BENCHMARK_TEMPLATE_F(RandomMeshBenchmark2D, V1_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v1::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (mesh_a_.num_intervals + mesh_b_.num_intervals) * sizeof(Interval));
}

BENCHMARK_TEMPLATE_F(RandomMeshBenchmark2D, V2_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v2::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (mesh_a_.num_intervals + mesh_b_.num_intervals) * sizeof(Interval));
}

BENCHMARK_TEMPLATE_F(RandomMeshBenchmark2D, V3_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v3::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (mesh_a_.num_intervals + mesh_b_.num_intervals) * sizeof(Interval));
}

// ============================================================================
// 3D Benchmarks
// ============================================================================

BENCHMARK_TEMPLATE_F(RandomMeshBenchmark3D, V1_3D_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v1::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_F(RandomMeshBenchmark3D, V2_3D_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v2::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_F(RandomMeshBenchmark3D, V3_3D_SmallConfig, GetSmallConfig)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v3::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_F(RandomMeshBenchmark3D, V1_3D_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v1::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_F(RandomMeshBenchmark3D, V2_3D_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v2::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_F(RandomMeshBenchmark3D, V3_3D_MediumConfig, GetMediumConfig)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v3::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_F(RandomMeshBenchmark3D, V1_3D_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v1::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_F(RandomMeshBenchmark3D, V2_3D_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v2::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE_F(RandomMeshBenchmark3D, V3_3D_LargeConfig, GetLargeConfig)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = v3::intersect_meshes_3d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_intervals);
  }
  state.SetItemsProcessed(state.iterations());
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
