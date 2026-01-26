// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

/**
 * @file phase_benchmark.cpp
 *
 * Phase-by-phase micro-benchmarks for intersection algorithm.
 *
 * Each benchmark measures a single phase of the intersection algorithm
 * to understand where time is being spent. All use pre-allocated workspace
 * to eliminate allocation overhead.
 *
 * Phases benchmarked:
 * 1. Row Mapping - binary search to find matching rows
 * 2. Row Scan - count matching rows and compute positions
 * 3. Row Compaction - extract matching rows
 * 4. Interval Counting - count intersections per row
 * 5. Scan (row_ptr) - compute CSR offsets
 * 6. Fill Intervals - compute actual intersections
 * 7. Mark Non-Empty - identify rows with intervals
 * 8. Compute Final Positions - second scan for compaction
 * 9. Compact Final - remove empty rows
 */

#include <benchmark/benchmark.h>
#include <playground/subsetix/csr/intersection/algorithm/baseline.hpp>
#include <playground/subsetix/csr/intersection/workspace.hpp>
#include <intersection/test_random_mesh_generator.hpp>
#include <Kokkos_Core.hpp>

using namespace playground::subsetix::csr::intersection;
using namespace playground::subsetix::csr::intersection::baseline;
using namespace playground::subsetix::csr::intersection::test;

// Type aliases
using Coord = int32_t;
using IntervalType = playground::subsetix::csr::intersection::Interval<Coord>;
using Workspace = IntersectionWorkspace<Kokkos::DefaultExecutionSpace>;

// ============================================================================
// Phase Benchmark Fixture
// ============================================================================

template <typename GetConfigFunc>
class PhaseBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State&) override {
    auto cfg = GetConfigFunc()();

    // Generate input meshes
    auto common_a = RegularMeshGenerator::generate_2d(cfg);
    auto common_b = RegularMeshGenerator::generate_2d(cfg);

    // Convert to device format
    mesh_a_ = MeshConverter2D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(common_a);
    mesh_b_ = MeshConverter2D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(common_b);

    // Allocate workspace and result
    std::size_t max_rows = std::max(mesh_a_.num_rows, mesh_b_.num_rows);
    std::size_t max_intervals = std::max(mesh_a_.num_intervals, mesh_b_.num_intervals);

    workspace_.ensure_capacity(max_rows, max_intervals);

    result_.row_keys = Kokkos::View<RowKey2D<Coord>*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_keys", max_rows);
    result_.row_ptr = Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_ptr", max_rows + 1);
    result_.intervals = Kokkos::View<IntervalType*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_intervals", max_intervals);

    // Pre-compute row mapping phase data (used by multiple phases)
    run_row_mapping_phase();
  }

  void TearDown(const ::benchmark::State&) override {}

protected:
  baseline::Mesh2DDevice mesh_a_, mesh_b_;
  baseline::Mesh2DDevice result_;
  Workspace workspace_;

  // Intermediate data from row mapping phase
  std::size_t num_rows_out_ = 0;

  void run_row_mapping_phase() {
    const std::size_t num_rows_a = mesh_a_.num_rows;

    // Phase 1: Row Mapping
    Kokkos::parallel_for(
        "phase_benchmark_row_mapping",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey2D<Coord> key = mesh_a_.row_keys(i);
          const int idx_b = detail::find_row_by_y(
              mesh_b_.row_keys.data(), mesh_b_.num_rows, key.y);

          workspace_.flags(i) = (idx_b >= 0) ? 1 : 0;
          workspace_.tmp_idx_a(i) = (idx_b >= 0) ? static_cast<int>(i) : -1;
          workspace_.tmp_idx_b(i) = idx_b;
        });

    Kokkos::fence();

    // Phase 2: Row Scan
    Kokkos::parallel_scan(
        "phase_benchmark_row_scan",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = workspace_.flags(i);
          if (final_pass) {
            workspace_.positions(i) = update;
          }
          update += count;
        },
        workspace_.num_rows_out_view);

    Kokkos::fence();

    // Get count
    std::size_t num_rows_out_host = 0;
    Kokkos::deep_copy(num_rows_out_host, workspace_.num_rows_out_view);
    num_rows_out_ = num_rows_out_host;

    // Phase 3: Row Compaction
    Kokkos::parallel_for(
        "phase_benchmark_row_compact",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          if (workspace_.flags(i) == 1) {
            const std::size_t pos = workspace_.positions(i);
            workspace_.out_rows(pos) = static_cast<int>(mesh_a_.row_keys(i).y);
            workspace_.out_idx_a(pos) = workspace_.tmp_idx_a(i);
            workspace_.out_idx_b(pos) = workspace_.tmp_idx_b(i);
          }
        });

    Kokkos::fence();
  }
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
// Phase 1: Row Mapping Benchmark
// ============================================================================

BENCHMARK_TEMPLATE_F(PhaseBenchmark2D, Phase1_RowMapping_Small, GetSmallRegularConfig)
(benchmark::State& state) {
  const std::size_t num_rows_a = mesh_a_.num_rows;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_row_mapping",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey2D<Coord> key = mesh_a_.row_keys(i);
          const int idx_b = detail::find_row_by_y(
              mesh_b_.row_keys.data(), mesh_b_.num_rows, key.y);
          benchmark::DoNotOptimize(idx_b);
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK_TEMPLATE_F(PhaseBenchmark2D, Phase1_RowMapping_Medium, GetMediumRegularConfig)
(benchmark::State& state) {
  const std::size_t num_rows_a = mesh_a_.num_rows;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_row_mapping",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey2D<Coord> key = mesh_a_.row_keys(i);
          const int idx_b = detail::find_row_by_y(
              mesh_b_.row_keys.data(), mesh_b_.num_rows, key.y);
          benchmark::DoNotOptimize(idx_b);
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

// ============================================================================
// Phase 2: Row Scan Benchmark
// ============================================================================

BENCHMARK_TEMPLATE_F(PhaseBenchmark2D, Phase2_RowScan_Small, GetSmallRegularConfig)
(benchmark::State& state) {
  const std::size_t num_rows_a = mesh_a_.num_rows;

  for (auto _ : state) {
    Kokkos::parallel_scan(
        "phase_row_scan",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = workspace_.flags(i);
          if (final_pass) {
            workspace_.positions(i) = update;
          }
          update += count;
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK_TEMPLATE_F(PhaseBenchmark2D, Phase2_RowScan_Medium, GetMediumRegularConfig)
(benchmark::State& state) {
  const std::size_t num_rows_a = mesh_a_.num_rows;

  for (auto _ : state) {
    Kokkos::parallel_scan(
        "phase_row_scan",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = workspace_.flags(i);
          if (final_pass) {
            workspace_.positions(i) = update;
          }
          update += count;
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

// ============================================================================
// Full Intersection (for comparison)
// ============================================================================

BENCHMARK_TEMPLATE_F(PhaseBenchmark2D, Full_Intersection_Small, GetSmallRegularConfig)
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

BENCHMARK_TEMPLATE_F(PhaseBenchmark2D, Full_Intersection_Medium, GetMediumRegularConfig)
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

  Kokkos::finalize();
  std::_Exit(0);
}
