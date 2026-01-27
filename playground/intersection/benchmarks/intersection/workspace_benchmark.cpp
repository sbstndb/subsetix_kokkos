// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

/**
 * @file workspace_benchmark.cpp
 *
 * Benchmarks using pre-allocated workspace to eliminate allocation overhead.
 *
 * This demonstrates the "true" algorithm performance without memory allocation
 * costs. All temporary buffers are allocated once in SetUp() and reused across
 * all benchmark iterations.
 */

#include <benchmark/benchmark.h>
#include <playground/subsetix/csr/intersection/algorithm/baseline.hpp>
#include <playground/subsetix/csr/intersection/workspace.hpp>
#include <intersection/test_random_mesh_generator.hpp>
#include <Kokkos_Core.hpp>

// Bring version namespaces into scope
using namespace playground::subsetix::csr::intersection;
using namespace playground::subsetix::csr::intersection::baseline;
using namespace playground::subsetix::csr::intersection::test;

// Type aliases for convenience
using Coord = int32_t;
using IntervalType = playground::subsetix::csr::intersection::Interval<Coord>;
using Workspace2D = IntersectionWorkspace2D<Kokkos::DefaultExecutionSpace>;
using Workspace3D = IntersectionWorkspace3D<Kokkos::DefaultExecutionSpace>;

// ============================================================================
// Conversion helpers
// ============================================================================

namespace benchmark_helpers {

inline baseline::Mesh2DDevice from_common_2d_baseline(const DefaultCommonMesh2D& mesh) {
  return MeshConverter2D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}

inline baseline::Mesh3DDevice from_common_3d_baseline(const DefaultCommonMesh3D& mesh) {
  return MeshConverter3D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}

} // namespace benchmark_helpers

// ============================================================================
// Workspace Benchmark Fixture (NO allocations per iteration)
// ============================================================================

/**
 * @brief Benchmark fixture using pre-allocated workspace
 *
 * Key differences from regular benchmark:
 * - Workspace allocated ONCE in SetUp()
 * - Result mesh allocated ONCE in SetUp()
 * - Each iteration reuses the same buffers
 * - Measures ONLY algorithm performance, not allocation overhead
 */
template <typename GetConfigFunc>
class WorkspaceBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State&) override {
    auto cfg = GetConfigFunc()();

    // Generate input meshes
    auto common_a = RegularMeshGenerator::generate_2d(cfg);
    auto common_b = RegularMeshGenerator::generate_2d(cfg);

    // Convert to device format
    mesh_a_ = benchmark_helpers::from_common_2d_baseline(common_a);
    mesh_b_ = benchmark_helpers::from_common_2d_baseline(common_b);

    // Calculate maximum required size (intersection cannot exceed inputs)
    std::size_t max_rows = std::max(mesh_a_.num_rows, mesh_b_.num_rows);
    std::size_t max_intervals = std::max(mesh_a_.num_intervals, mesh_b_.num_intervals);

    // Pre-allocate workspace with sufficient capacity
    workspace_.ensure_capacity(max_rows, max_intervals);

    // Pre-allocate result mesh (NO allocation during benchmark!)
    result_.row_keys = Kokkos::View<RowKey2D<Coord>*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_keys", max_rows);
    result_.row_ptr = Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_ptr", max_rows + 1);
    result_.intervals = Kokkos::View<IntervalType*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_intervals", max_intervals);
  }

  void TearDown(const ::benchmark::State&) override {
    // Nothing to cleanup - views will be destroyed automatically
  }

protected:
  baseline::Mesh2DDevice mesh_a_, mesh_b_;
  baseline::Mesh2DDevice result_;
  Workspace2D workspace_;
};

// ============================================================================
// 3D Workspace Benchmark Fixture
// ============================================================================

template <typename GetConfigFunc>
class WorkspaceBenchmark3D : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State&) override {
    auto cfg = GetConfigFunc()();

    // Generate input meshes
    auto common_a = RegularMeshGenerator::generate_3d(cfg);
    auto common_b = RegularMeshGenerator::generate_3d(cfg);

    // Convert to device format
    mesh_a_ = benchmark_helpers::from_common_3d_baseline(common_a);
    mesh_b_ = benchmark_helpers::from_common_3d_baseline(common_b);

    // Calculate maximum required size
    std::size_t max_rows = std::max(mesh_a_.num_rows, mesh_b_.num_rows);
    std::size_t max_intervals = std::max(mesh_a_.num_intervals, mesh_b_.num_intervals);

    // Pre-allocate workspace with sufficient capacity
    workspace_.ensure_capacity(max_rows, max_intervals);

    // Pre-allocate result mesh (NO allocation during benchmark!)
    result_.row_keys = Kokkos::View<RowKey3D<Coord>*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_keys", max_rows);
    result_.row_ptr = Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_ptr", max_rows + 1);
    result_.intervals = Kokkos::View<IntervalType*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_intervals", max_intervals);
  }

  void TearDown(const ::benchmark::State&) override {}

protected:
  baseline::Mesh3DDevice mesh_a_, mesh_b_;
  baseline::Mesh3DDevice result_;
  Workspace3D workspace_;
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
// 2D Workspace Benchmarks (NO allocations per iteration)
// ============================================================================

BENCHMARK_TEMPLATE_F(WorkspaceBenchmark2D, Workspace_2D_SmallConfig, GetSmallRegularConfig)
(benchmark::State& state) {
  // Total intervals processed per iteration
  const std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;

  for (auto _ : state) {
    // This call performs ZERO allocations - all buffers pre-allocated!
    baseline::intersect_meshes_2d_in_place(mesh_a_, mesh_b_, result_, workspace_);

    benchmark::DoNotOptimize(result_.num_rows);
    benchmark::DoNotOptimize(result_.num_intervals);
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(WorkspaceBenchmark2D, Workspace_2D_MediumConfig, GetMediumRegularConfig)
(benchmark::State& state) {
  const std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;

  for (auto _ : state) {
    baseline::intersect_meshes_2d_in_place(mesh_a_, mesh_b_, result_, workspace_);

    benchmark::DoNotOptimize(result_.num_rows);
    benchmark::DoNotOptimize(result_.num_intervals);
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(WorkspaceBenchmark2D, Workspace_2D_LargeConfig, GetLargeRegularConfig)
(benchmark::State& state) {
  const std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;

  for (auto _ : state) {
    baseline::intersect_meshes_2d_in_place(mesh_a_, mesh_b_, result_, workspace_);

    benchmark::DoNotOptimize(result_.num_rows);
    benchmark::DoNotOptimize(result_.num_intervals);
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(WorkspaceBenchmark2D, Workspace_2D_ExtraLargeConfig, GetExtraLargeRegularConfig)
(benchmark::State& state) {
  const std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;

  for (auto _ : state) {
    baseline::intersect_meshes_2d_in_place(mesh_a_, mesh_b_, result_, workspace_);

    benchmark::DoNotOptimize(result_.num_rows);
    benchmark::DoNotOptimize(result_.num_intervals);
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

// ============================================================================
// 3D Workspace Benchmarks (NO allocations per iteration)
// ============================================================================

BENCHMARK_TEMPLATE_F(WorkspaceBenchmark3D, Workspace_3D_SmallConfig, GetSmallRegularConfig)
(benchmark::State& state) {
  const std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;

  for (auto _ : state) {
    baseline::intersect_meshes_3d_in_place(mesh_a_, mesh_b_, result_, workspace_);

    benchmark::DoNotOptimize(result_.num_rows);
    benchmark::DoNotOptimize(result_.num_intervals);
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(WorkspaceBenchmark3D, Workspace_3D_MediumConfig, GetMediumRegularConfig)
(benchmark::State& state) {
  const std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;

  for (auto _ : state) {
    baseline::intersect_meshes_3d_in_place(mesh_a_, mesh_b_, result_, workspace_);

    benchmark::DoNotOptimize(result_.num_rows);
    benchmark::DoNotOptimize(result_.num_intervals);
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK_TEMPLATE_F(WorkspaceBenchmark3D, Workspace_3D_LargeConfig, GetLargeRegularConfig)
(benchmark::State& state) {
  const std::size_t total_intervals = mesh_a_.num_intervals + mesh_b_.num_intervals;

  for (auto _ : state) {
    baseline::intersect_meshes_3d_in_place(mesh_a_, mesh_b_, result_, workspace_);

    benchmark::DoNotOptimize(result_.num_rows);
    benchmark::DoNotOptimize(result_.num_intervals);
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
