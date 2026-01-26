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
 * IMPORTANT: Uses regular BENCHMARK (not BENCHMARK_TEMPLATE_F) to be
 * CUDA-compatible. CUDA nvc++ doesn't allow __host__ __device__ lambdas
 * in private member functions (which BENCHMARK_TEMPLATE_F generates).
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
using Workspace = IntersectionWorkspace2D<Kokkos::DefaultExecutionSpace>;

// ============================================================================
// Helper function to generate benchmark data
// ============================================================================

struct PhaseBenchmarkData {
  baseline::Mesh2DDevice mesh_a, mesh_b;
  baseline::Mesh2DDevice result;
  Workspace workspace;
  std::size_t num_rows_out;  // Host copy of number of matching rows

  PhaseBenchmarkData(const RegularMeshConfig& cfg) {
    // Generate input meshes
    auto common_a = RegularMeshGenerator::generate_2d(cfg);
    auto common_b = RegularMeshGenerator::generate_2d(cfg);

    // Convert to device format
    mesh_a = MeshConverter2D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(common_a);
    mesh_b = MeshConverter2D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(common_b);

    // Allocate workspace and result
    std::size_t max_rows = std::max(mesh_a.num_rows, mesh_b.num_rows);
    std::size_t max_intervals = std::max(mesh_a.num_intervals, mesh_b.num_intervals);

    workspace.ensure_capacity(max_rows, max_intervals);

    result.row_keys = Kokkos::View<RowKey2D<Coord>*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_keys", max_rows);
    result.row_ptr = Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_ptr", max_rows + 1);
    result.intervals = Kokkos::View<IntervalType*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_intervals", max_intervals);

    // Pre-compute row mapping phase data
    precompute_row_mapping();
  }

  void precompute_row_mapping() {
    const std::size_t num_rows_a = mesh_a.num_rows;
    const auto row_keys_a = mesh_a.row_keys;
    const auto row_keys_b = mesh_b.row_keys;
    const auto num_rows_b = mesh_b.num_rows;
    const auto flags = workspace.flags;
    const auto tmp_idx_a = workspace.tmp_idx_a;
    const auto tmp_idx_b = workspace.tmp_idx_b;
    const auto positions = workspace.positions;
    const auto out_rows = workspace.out_rows;
    const auto out_idx_a = workspace.out_idx_a;
    const auto out_idx_b = workspace.out_idx_b;
    const auto num_rows_out_view = workspace.num_rows_out_view;

    // Phase 1: Row Mapping
    Kokkos::parallel_for(
        "phase_benchmark_row_mapping",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey2D<Coord> key = row_keys_a(i);
          const int idx_b = playground::subsetix::csr::intersection::detail::find_row_by_y(
              row_keys_b, num_rows_b, key.y);

          flags(i) = (idx_b >= 0) ? 1 : 0;
          tmp_idx_a(i) = (idx_b >= 0) ? static_cast<int>(i) : -1;
          tmp_idx_b(i) = idx_b;
        });

    Kokkos::fence();

    // Phase 2: Row Scan
    Kokkos::parallel_scan(
        "phase_benchmark_row_scan",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = flags(i);
          if (final_pass) {
            positions(i) = update;
          }
          update += count;
        },
        num_rows_out_view);

    Kokkos::fence();

    // Extract num_rows_out from device view to host
    Kokkos::deep_copy(num_rows_out, workspace.num_rows_out_view);

    // Phase 3: Row Compaction
    Kokkos::parallel_for(
        "phase_benchmark_row_compact",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          if (flags(i) == 1) {
            const std::size_t pos = positions(i);
            out_rows(pos) = row_keys_a(i);
            out_idx_a(pos) = tmp_idx_a(i);
            out_idx_b(pos) = tmp_idx_b(i);
          }
        });

    Kokkos::fence();
  }
};

// ============================================================================
// Phase 1: Row Mapping Benchmark
// ============================================================================

static void Phase1_RowMapping_Small(benchmark::State& state) {
  PhaseBenchmarkData data(SmallRegularConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto row_keys_a = data.mesh_a.row_keys;
  const auto row_keys_b = data.mesh_b.row_keys;
  const auto num_rows_b = data.mesh_b.num_rows;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_row_mapping",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey2D<Coord> key = row_keys_a(i);
          const int idx_b = playground::subsetix::csr::intersection::detail::find_row_by_y(
              row_keys_b, num_rows_b, key.y);
          benchmark::DoNotOptimize(idx_b);
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase1_RowMapping_Small)->Unit(benchmark::kMicrosecond);

static void Phase1_RowMapping_Medium(benchmark::State& state) {
  PhaseBenchmarkData data(MediumRegularConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto row_keys_a = data.mesh_a.row_keys;
  const auto row_keys_b = data.mesh_b.row_keys;
  const auto num_rows_b = data.mesh_b.num_rows;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_row_mapping",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey2D<Coord> key = row_keys_a(i);
          const int idx_b = playground::subsetix::csr::intersection::detail::find_row_by_y(
              row_keys_b, num_rows_b, key.y);
          benchmark::DoNotOptimize(idx_b);
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase1_RowMapping_Medium)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Phase 2: Row Scan Benchmark
// ============================================================================

static void Phase2_RowScan_Small(benchmark::State& state) {
  PhaseBenchmarkData data(SmallRegularConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto flags = data.workspace.flags;
  const auto positions = data.workspace.positions;

  for (auto _ : state) {
    Kokkos::parallel_scan(
        "phase_row_scan",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = flags(i);
          if (final_pass) {
            positions(i) = update;
          }
          update += count;
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase2_RowScan_Small)->Unit(benchmark::kMicrosecond);

static void Phase2_RowScan_Medium(benchmark::State& state) {
  PhaseBenchmarkData data(MediumRegularConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto flags = data.workspace.flags;
  const auto positions = data.workspace.positions;

  for (auto _ : state) {
    Kokkos::parallel_scan(
        "phase_row_scan",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = flags(i);
          if (final_pass) {
            positions(i) = update;
          }
          update += count;
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase2_RowScan_Medium)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Full Intersection (for comparison)
// ============================================================================

static void Full_Intersection_Small(benchmark::State& state) {
  PhaseBenchmarkData data(SmallRegularConfig());
  const auto total_intervals = data.mesh_a.num_intervals + data.mesh_b.num_intervals;
  const auto& mesh_a = data.mesh_a;
  const auto& mesh_b = data.mesh_b;
  auto& result = data.result;
  auto& workspace = data.workspace;

  for (auto _ : state) {
    baseline::intersect_meshes_2d_in_place(mesh_a, mesh_b, result, workspace);

    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK(Full_Intersection_Small)->Unit(benchmark::kMicrosecond);

static void Full_Intersection_Medium(benchmark::State& state) {
  PhaseBenchmarkData data(MediumRegularConfig());
  const auto total_intervals = data.mesh_a.num_intervals + data.mesh_b.num_intervals;
  const auto& mesh_a = data.mesh_a;
  const auto& mesh_b = data.mesh_b;
  auto& result = data.result;
  auto& workspace = data.workspace;

  for (auto _ : state) {
    baseline::intersect_meshes_2d_in_place(mesh_a, mesh_b, result, workspace);

    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK(Full_Intersection_Medium)->Unit(benchmark::kMicrosecond);

// ============================================================================
// 3D Helper Structure
// ============================================================================

struct PhaseBenchmarkData3D {
  baseline::Mesh3DDevice mesh_a, mesh_b;
  baseline::Mesh3DDevice result;
  IntersectionWorkspace3D<Kokkos::DefaultExecutionSpace> workspace;
  std::size_t num_rows_out;  // Host copy of number of matching rows

  PhaseBenchmarkData3D(const RegularMeshConfig& cfg) {
    // Generate input meshes
    auto common_a = RegularMeshGenerator::generate_3d(cfg);
    auto common_b = RegularMeshGenerator::generate_3d(cfg);

    // Convert to device format
    mesh_a = MeshConverter3D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(common_a);
    mesh_b = MeshConverter3D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(common_b);

    // Allocate workspace and result
    std::size_t max_rows = std::max(mesh_a.num_rows, mesh_b.num_rows);
    std::size_t max_intervals = std::max(mesh_a.num_intervals, mesh_b.num_intervals);

    workspace.ensure_capacity(max_rows, max_intervals);

    result.row_keys = Kokkos::View<RowKey3D<Coord>*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_keys", max_rows);
    result.row_ptr = Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_ptr", max_rows + 1);
    result.intervals = Kokkos::View<IntervalType*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_intervals", max_intervals);

    // Pre-compute row mapping phase data
    precompute_row_mapping();
  }

  void precompute_row_mapping() {
    const std::size_t num_rows_a = mesh_a.num_rows;
    const auto row_keys_a = mesh_a.row_keys;
    const auto row_keys_b = mesh_b.row_keys;
    const auto num_rows_b = mesh_b.num_rows;
    const auto flags = workspace.flags;
    const auto tmp_idx_a = workspace.tmp_idx_a;
    const auto tmp_idx_b = workspace.tmp_idx_b;
    const auto positions = workspace.positions;
    const auto out_rows = workspace.out_rows;
    const auto out_idx_a = workspace.out_idx_a;
    const auto out_idx_b = workspace.out_idx_b;
    const auto num_rows_out_view = workspace.num_rows_out_view;

    // Phase 1: Row Mapping (3D: search by y,z)
    Kokkos::parallel_for(
        "phase_benchmark_row_mapping_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey3D<Coord> key = row_keys_a(i);
          const int idx_b = playground::subsetix::csr::intersection::detail::find_row_by_yz(
              row_keys_b, num_rows_b, key.y, key.z);

          flags(i) = (idx_b >= 0) ? 1 : 0;
          tmp_idx_a(i) = (idx_b >= 0) ? static_cast<int>(i) : -1;
          tmp_idx_b(i) = idx_b;
        });

    Kokkos::fence();

    // Phase 2: Row Scan
    Kokkos::parallel_scan(
        "phase_benchmark_row_scan_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = flags(i);
          if (final_pass) {
            positions(i) = update;
          }
          update += count;
        },
        num_rows_out_view);

    Kokkos::fence();

    // Extract num_rows_out from device view to host
    Kokkos::deep_copy(num_rows_out, workspace.num_rows_out_view);

    // Phase 3: Row Compaction
    Kokkos::parallel_for(
        "phase_benchmark_row_compact_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          if (flags(i) == 1) {
            const std::size_t pos = positions(i);
            out_rows(pos) = row_keys_a(i);
            out_idx_a(pos) = tmp_idx_a(i);
            out_idx_b(pos) = tmp_idx_b(i);
          }
        });

    Kokkos::fence();
  }
};

// ============================================================================
// Phase 1: Row Mapping Benchmark 3D
// ============================================================================

static void Phase1_RowMapping_3D_Small(benchmark::State& state) {
  PhaseBenchmarkData3D data(SmallRegularConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto row_keys_a = data.mesh_a.row_keys;
  const auto row_keys_b = data.mesh_b.row_keys;
  const auto num_rows_b = data.mesh_b.num_rows;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_row_mapping_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey3D<Coord> key = row_keys_a(i);
          const int idx_b = playground::subsetix::csr::intersection::detail::find_row_by_yz(
              row_keys_b, num_rows_b, key.y, key.z);
          benchmark::DoNotOptimize(idx_b);
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase1_RowMapping_3D_Small)->Unit(benchmark::kMicrosecond);

static void Phase1_RowMapping_3D_Medium(benchmark::State& state) {
  PhaseBenchmarkData3D data(MediumRegularConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto row_keys_a = data.mesh_a.row_keys;
  const auto row_keys_b = data.mesh_b.row_keys;
  const auto num_rows_b = data.mesh_b.num_rows;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_row_mapping_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey3D<Coord> key = row_keys_a(i);
          const int idx_b = playground::subsetix::csr::intersection::detail::find_row_by_yz(
              row_keys_b, num_rows_b, key.y, key.z);
          benchmark::DoNotOptimize(idx_b);
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase1_RowMapping_3D_Medium)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Phase 2: Row Scan Benchmark 3D
// ============================================================================

static void Phase2_RowScan_3D_Small(benchmark::State& state) {
  PhaseBenchmarkData3D data(SmallRegularConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto flags = data.workspace.flags;
  const auto positions = data.workspace.positions;

  for (auto _ : state) {
    Kokkos::parallel_scan(
        "phase_row_scan_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = flags(i);
          if (final_pass) {
            positions(i) = update;
          }
          update += count;
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase2_RowScan_3D_Small)->Unit(benchmark::kMicrosecond);

static void Phase2_RowScan_3D_Medium(benchmark::State& state) {
  PhaseBenchmarkData3D data(MediumRegularConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto flags = data.workspace.flags;
  const auto positions = data.workspace.positions;

  for (auto _ : state) {
    Kokkos::parallel_scan(
        "phase_row_scan_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = flags(i);
          if (final_pass) {
            positions(i) = update;
          }
          update += count;
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase2_RowScan_3D_Medium)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Full Intersection 3D
// ============================================================================

static void Full_Intersection_3D_Small(benchmark::State& state) {
  PhaseBenchmarkData3D data(SmallRegularConfig());
  const auto total_intervals = data.mesh_a.num_intervals + data.mesh_b.num_intervals;
  const auto& mesh_a = data.mesh_a;
  const auto& mesh_b = data.mesh_b;
  auto& result = data.result;
  auto& workspace = data.workspace;

  for (auto _ : state) {
    baseline::intersect_meshes_3d_in_place(mesh_a, mesh_b, result, workspace);

    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK(Full_Intersection_3D_Small)->Unit(benchmark::kMicrosecond);

static void Full_Intersection_3D_Medium(benchmark::State& state) {
  PhaseBenchmarkData3D data(MediumRegularConfig());
  const auto total_intervals = data.mesh_a.num_intervals + data.mesh_b.num_intervals;
  const auto& mesh_a = data.mesh_a;
  const auto& mesh_b = data.mesh_b;
  auto& result = data.result;
  auto& workspace = data.workspace;

  for (auto _ : state) {
    baseline::intersect_meshes_3d_in_place(mesh_a, mesh_b, result, workspace);

    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK(Full_Intersection_3D_Medium)->Unit(benchmark::kMicrosecond);

static void Full_Intersection_3D_Large(benchmark::State& state) {
  PhaseBenchmarkData3D data(LargeRegularConfig());
  const auto total_intervals = data.mesh_a.num_intervals + data.mesh_b.num_intervals;
  const auto& mesh_a = data.mesh_a;
  const auto& mesh_b = data.mesh_b;
  auto& result = data.result;
  auto& workspace = data.workspace;

  for (auto _ : state) {
    baseline::intersect_meshes_3d_in_place(mesh_a, mesh_b, result, workspace);

    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK(Full_Intersection_3D_Large)->Unit(benchmark::kMicrosecond);

static void Phase1_RowMapping_3D_Large(benchmark::State& state) {
  PhaseBenchmarkData3D data(LargeRegularConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto row_keys_a = data.mesh_a.row_keys;
  const auto row_keys_b = data.mesh_b.row_keys;
  const auto num_rows_b = data.mesh_b.num_rows;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_row_mapping_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey3D<Coord> key = row_keys_a(i);
          const int idx_b = playground::subsetix::csr::intersection::detail::find_row_by_yz(
              row_keys_b, num_rows_b, key.y, key.z);
          benchmark::DoNotOptimize(idx_b);
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase1_RowMapping_3D_Large)->Unit(benchmark::kMicrosecond);

static void Phase2_RowScan_3D_Large(benchmark::State& state) {
  PhaseBenchmarkData3D data(LargeRegularConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto flags = data.workspace.flags;
  const auto positions = data.workspace.positions;

  for (auto _ : state) {
    Kokkos::parallel_scan(
        "phase_row_scan_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = flags(i);
          if (final_pass) {
            positions(i) = update;
          }
          update += count;
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase2_RowScan_3D_Large)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Phase 3: Row Compaction Benchmark (2D)
// ============================================================================

static void Phase3_RowCompaction_2D_Small(benchmark::State& state) {
  PhaseBenchmarkData data(SmallRegularConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto flags = data.workspace.flags;
  const auto positions = data.workspace.positions;
  const auto row_keys_a = data.mesh_a.row_keys;
  const auto tmp_idx_a = data.workspace.tmp_idx_a;
  const auto tmp_idx_b = data.workspace.tmp_idx_b;
  const auto out_rows = data.workspace.out_rows;
  const auto out_idx_a = data.workspace.out_idx_a;
  const auto out_idx_b = data.workspace.out_idx_b;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_row_compact",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          if (flags(i) == 1) {
            const std::size_t pos = positions(i);
            out_rows(pos) = row_keys_a(i);
            out_idx_a(pos) = tmp_idx_a(i);
            out_idx_b(pos) = tmp_idx_b(i);
          }
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase3_RowCompaction_2D_Small)->Unit(benchmark::kMicrosecond);

static void Phase3_RowCompaction_2D_Medium(benchmark::State& state) {
  PhaseBenchmarkData data(MediumRegularConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto flags = data.workspace.flags;
  const auto positions = data.workspace.positions;
  const auto row_keys_a = data.mesh_a.row_keys;
  const auto tmp_idx_a = data.workspace.tmp_idx_a;
  const auto tmp_idx_b = data.workspace.tmp_idx_b;
  const auto out_rows = data.workspace.out_rows;
  const auto out_idx_a = data.workspace.out_idx_a;
  const auto out_idx_b = data.workspace.out_idx_b;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_row_compact",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          if (flags(i) == 1) {
            const std::size_t pos = positions(i);
            out_rows(pos) = row_keys_a(i);
            out_idx_a(pos) = tmp_idx_a(i);
            out_idx_b(pos) = tmp_idx_b(i);
          }
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase3_RowCompaction_2D_Medium)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Phase 3: Row Compaction Benchmark (3D)
// ============================================================================

static void Phase3_RowCompaction_3D_Small(benchmark::State& state) {
  PhaseBenchmarkData3D data(SmallRegularConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto flags = data.workspace.flags;
  const auto positions = data.workspace.positions;
  const auto row_keys_a = data.mesh_a.row_keys;
  const auto tmp_idx_a = data.workspace.tmp_idx_a;
  const auto tmp_idx_b = data.workspace.tmp_idx_b;
  const auto out_rows = data.workspace.out_rows;
  const auto out_idx_a = data.workspace.out_idx_a;
  const auto out_idx_b = data.workspace.out_idx_b;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_row_compact_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          if (flags(i) == 1) {
            const std::size_t pos = positions(i);
            out_rows(pos) = row_keys_a(i);
            out_idx_a(pos) = tmp_idx_a(i);
            out_idx_b(pos) = tmp_idx_b(i);
          }
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase3_RowCompaction_3D_Small)->Unit(benchmark::kMicrosecond);

static void Phase3_RowCompaction_3D_Medium(benchmark::State& state) {
  PhaseBenchmarkData3D data(MediumRegularConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto flags = data.workspace.flags;
  const auto positions = data.workspace.positions;
  const auto row_keys_a = data.mesh_a.row_keys;
  const auto tmp_idx_a = data.workspace.tmp_idx_a;
  const auto tmp_idx_b = data.workspace.tmp_idx_b;
  const auto out_rows = data.workspace.out_rows;
  const auto out_idx_a = data.workspace.out_idx_a;
  const auto out_idx_b = data.workspace.out_idx_b;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_row_compact_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          if (flags(i) == 1) {
            const std::size_t pos = positions(i);
            out_rows(pos) = row_keys_a(i);
            out_idx_a(pos) = tmp_idx_a(i);
            out_idx_b(pos) = tmp_idx_b(i);
          }
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase3_RowCompaction_3D_Medium)->Unit(benchmark::kMicrosecond);

static void Phase3_RowCompaction_3D_Large(benchmark::State& state) {
  PhaseBenchmarkData3D data(LargeRegularConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto flags = data.workspace.flags;
  const auto positions = data.workspace.positions;
  const auto row_keys_a = data.mesh_a.row_keys;
  const auto tmp_idx_a = data.workspace.tmp_idx_a;
  const auto tmp_idx_b = data.workspace.tmp_idx_b;
  const auto out_rows = data.workspace.out_rows;
  const auto out_idx_a = data.workspace.out_idx_a;
  const auto out_idx_b = data.workspace.out_idx_b;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_row_compact_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          if (flags(i) == 1) {
            const std::size_t pos = positions(i);
            out_rows(pos) = row_keys_a(i);
            out_idx_a(pos) = tmp_idx_a(i);
            out_idx_b(pos) = tmp_idx_b(i);
          }
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase3_RowCompaction_3D_Large)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Phase 4: Count Intervals Per Row Benchmark (2D)
// ============================================================================

static void Phase4_CountIntervals_2D_Small(benchmark::State& state) {
  PhaseBenchmarkData data(SmallRegularConfig());
  const auto num_rows_out = data.num_rows_out;
  const auto out_idx_a = data.workspace.out_idx_a;
  const auto out_idx_b = data.workspace.out_idx_b;
  const auto row_ptr_a = data.mesh_a.row_ptr;
  const auto row_ptr_b = data.mesh_b.row_ptr;
  const auto intervals_a = data.mesh_a.intervals;
  const auto intervals_b = data.mesh_b.intervals;
  const auto row_counts = data.workspace.flags;  // reuse as row_counts

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_count",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i) {
          const int ia = out_idx_a(i);
          const int ib = out_idx_b(i);

          if (ib < 0) {
            row_counts(i) = 0;
            return;
          }

          const auto r = playground::subsetix::csr::intersection::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

          if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
            row_counts(i) = 0;
            return;
          }

          row_counts(i) = playground::subsetix::csr::intersection::baseline::detail::row_intersection_impl<true>(
              intervals_a, r.begin_a, r.end_a,
              intervals_b, r.begin_b, r.end_b,
              Kokkos::View<playground::subsetix::csr::intersection::Interval<Coord>*, Kokkos::DefaultExecutionSpace::memory_space>(), 0);
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_out);
}

BENCHMARK(Phase4_CountIntervals_2D_Small)->Unit(benchmark::kMicrosecond);

static void Phase4_CountIntervals_2D_Medium(benchmark::State& state) {
  PhaseBenchmarkData data(MediumRegularConfig());
  const auto num_rows_out = data.num_rows_out;
  const auto out_idx_a = data.workspace.out_idx_a;
  const auto out_idx_b = data.workspace.out_idx_b;
  const auto row_ptr_a = data.mesh_a.row_ptr;
  const auto row_ptr_b = data.mesh_b.row_ptr;
  const auto intervals_a = data.mesh_a.intervals;
  const auto intervals_b = data.mesh_b.intervals;
  const auto row_counts = data.workspace.flags;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_count",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i) {
          const int ia = out_idx_a(i);
          const int ib = out_idx_b(i);

          if (ib < 0) {
            row_counts(i) = 0;
            return;
          }

          const auto r = playground::subsetix::csr::intersection::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

          if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
            row_counts(i) = 0;
            return;
          }

          row_counts(i) = playground::subsetix::csr::intersection::baseline::detail::row_intersection_impl<true>(
              intervals_a, r.begin_a, r.end_a,
              intervals_b, r.begin_b, r.end_b,
              Kokkos::View<playground::subsetix::csr::intersection::Interval<Coord>*, Kokkos::DefaultExecutionSpace::memory_space>(), 0);
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_out);
}

BENCHMARK(Phase4_CountIntervals_2D_Medium)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Phase 4: Count Intervals Per Row Benchmark (3D)
// ============================================================================

static void Phase4_CountIntervals_3D_Small(benchmark::State& state) {
  PhaseBenchmarkData3D data(SmallRegularConfig());
  const auto num_rows_out = data.num_rows_out;
  const auto out_idx_a = data.workspace.out_idx_a;
  const auto out_idx_b = data.workspace.out_idx_b;
  const auto row_ptr_a = data.mesh_a.row_ptr;
  const auto row_ptr_b = data.mesh_b.row_ptr;
  const auto intervals_a = data.mesh_a.intervals;
  const auto intervals_b = data.mesh_b.intervals;
  const auto row_counts = data.workspace.flags;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_count_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i) {
          const int ia = out_idx_a(i);
          const int ib = out_idx_b(i);

          if (ib < 0) {
            row_counts(i) = 0;
            return;
          }

          const auto r = playground::subsetix::csr::intersection::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

          if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
            row_counts(i) = 0;
            return;
          }

          row_counts(i) = playground::subsetix::csr::intersection::baseline::detail::row_intersection_impl<true>(
              intervals_a, r.begin_a, r.end_a,
              intervals_b, r.begin_b, r.end_b,
              Kokkos::View<playground::subsetix::csr::intersection::Interval<Coord>*, Kokkos::DefaultExecutionSpace::memory_space>(), 0);
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_out);
}

BENCHMARK(Phase4_CountIntervals_3D_Small)->Unit(benchmark::kMicrosecond);

static void Phase4_CountIntervals_3D_Medium(benchmark::State& state) {
  PhaseBenchmarkData3D data(MediumRegularConfig());
  const auto num_rows_out = data.num_rows_out;
  const auto out_idx_a = data.workspace.out_idx_a;
  const auto out_idx_b = data.workspace.out_idx_b;
  const auto row_ptr_a = data.mesh_a.row_ptr;
  const auto row_ptr_b = data.mesh_b.row_ptr;
  const auto intervals_a = data.mesh_a.intervals;
  const auto intervals_b = data.mesh_b.intervals;
  const auto row_counts = data.workspace.flags;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_count_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i) {
          const int ia = out_idx_a(i);
          const int ib = out_idx_b(i);

          if (ib < 0) {
            row_counts(i) = 0;
            return;
          }

          const auto r = playground::subsetix::csr::intersection::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

          if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
            row_counts(i) = 0;
            return;
          }

          row_counts(i) = playground::subsetix::csr::intersection::baseline::detail::row_intersection_impl<true>(
              intervals_a, r.begin_a, r.end_a,
              intervals_b, r.begin_b, r.end_b,
              Kokkos::View<playground::subsetix::csr::intersection::Interval<Coord>*, Kokkos::DefaultExecutionSpace::memory_space>(), 0);
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_out);
}

BENCHMARK(Phase4_CountIntervals_3D_Medium)->Unit(benchmark::kMicrosecond);

static void Phase4_CountIntervals_3D_Large(benchmark::State& state) {
  PhaseBenchmarkData3D data(LargeRegularConfig());
  const auto num_rows_out = data.num_rows_out;
  const auto out_idx_a = data.workspace.out_idx_a;
  const auto out_idx_b = data.workspace.out_idx_b;
  const auto row_ptr_a = data.mesh_a.row_ptr;
  const auto row_ptr_b = data.mesh_b.row_ptr;
  const auto intervals_a = data.mesh_a.intervals;
  const auto intervals_b = data.mesh_b.intervals;
  const auto row_counts = data.workspace.flags;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_count_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i) {
          const int ia = out_idx_a(i);
          const int ib = out_idx_b(i);

          if (ib < 0) {
            row_counts(i) = 0;
            return;
          }

          const auto r = playground::subsetix::csr::intersection::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

          if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
            row_counts(i) = 0;
            return;
          }

          row_counts(i) = playground::subsetix::csr::intersection::baseline::detail::row_intersection_impl<true>(
              intervals_a, r.begin_a, r.end_a,
              intervals_b, r.begin_b, r.end_b,
              Kokkos::View<playground::subsetix::csr::intersection::Interval<Coord>*, Kokkos::DefaultExecutionSpace::memory_space>(), 0);
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_out);
}

BENCHMARK(Phase4_CountIntervals_3D_Large)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Phase 5: Scan to Compute row_ptr Offsets (2D)
// ============================================================================

static void Phase5_ScanRowPtr_2D_Small(benchmark::State& state) {
  PhaseBenchmarkData data(SmallRegularConfig());
  const auto num_rows_out = data.num_rows_out;
  const auto row_counts = data.workspace.flags;  // reuse as row_counts
  auto result_row_ptr = data.result.row_ptr;

  for (auto _ : state) {
    Kokkos::parallel_scan(
        "phase_scan_row_ptr",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = row_counts(i);
          if (final_pass) {
            result_row_ptr(i) = update;
            if (i + 1 == num_rows_out) {
              result_row_ptr(num_rows_out) = update + count;
            }
          }
          update += count;
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_out);
}

BENCHMARK(Phase5_ScanRowPtr_2D_Small)->Unit(benchmark::kMicrosecond);

static void Phase5_ScanRowPtr_2D_Medium(benchmark::State& state) {
  PhaseBenchmarkData data(MediumRegularConfig());
  const auto num_rows_out = data.num_rows_out;
  const auto row_counts = data.workspace.flags;
  auto result_row_ptr = data.result.row_ptr;

  for (auto _ : state) {
    Kokkos::parallel_scan(
        "phase_scan_row_ptr",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = row_counts(i);
          if (final_pass) {
            result_row_ptr(i) = update;
            if (i + 1 == num_rows_out) {
              result_row_ptr(num_rows_out) = update + count;
            }
          }
          update += count;
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_out);
}

BENCHMARK(Phase5_ScanRowPtr_2D_Medium)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Phase 5: Scan to Compute row_ptr Offsets (3D)
// ============================================================================

static void Phase5_ScanRowPtr_3D_Small(benchmark::State& state) {
  PhaseBenchmarkData3D data(SmallRegularConfig());
  const auto num_rows_out = data.num_rows_out;
  const auto row_counts = data.workspace.flags;
  auto result_row_ptr = data.result.row_ptr;

  for (auto _ : state) {
    Kokkos::parallel_scan(
        "phase_scan_row_ptr_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = row_counts(i);
          if (final_pass) {
            result_row_ptr(i) = update;
            if (i + 1 == num_rows_out) {
              result_row_ptr(num_rows_out) = update + count;
            }
          }
          update += count;
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_out);
}

BENCHMARK(Phase5_ScanRowPtr_3D_Small)->Unit(benchmark::kMicrosecond);

static void Phase5_ScanRowPtr_3D_Medium(benchmark::State& state) {
  PhaseBenchmarkData3D data(MediumRegularConfig());
  const auto num_rows_out = data.num_rows_out;
  const auto row_counts = data.workspace.flags;
  auto result_row_ptr = data.result.row_ptr;

  for (auto _ : state) {
    Kokkos::parallel_scan(
        "phase_scan_row_ptr_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = row_counts(i);
          if (final_pass) {
            result_row_ptr(i) = update;
            if (i + 1 == num_rows_out) {
              result_row_ptr(num_rows_out) = update + count;
            }
          }
          update += count;
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_out);
}

BENCHMARK(Phase5_ScanRowPtr_3D_Medium)->Unit(benchmark::kMicrosecond);

static void Phase5_ScanRowPtr_3D_Large(benchmark::State& state) {
  PhaseBenchmarkData3D data(LargeRegularConfig());
  const auto num_rows_out = data.num_rows_out;
  const auto row_counts = data.workspace.flags;
  auto result_row_ptr = data.result.row_ptr;

  for (auto _ : state) {
    Kokkos::parallel_scan(
        "phase_scan_row_ptr_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = row_counts(i);
          if (final_pass) {
            result_row_ptr(i) = update;
            if (i + 1 == num_rows_out) {
              result_row_ptr(num_rows_out) = update + count;
            }
          }
          update += count;
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_out);
}

BENCHMARK(Phase5_ScanRowPtr_3D_Large)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Phase 6: Fill Intersected Intervals (2D)
// ============================================================================

static void Phase6_FillIntervals_2D_Small(benchmark::State& state) {
  PhaseBenchmarkData data(SmallRegularConfig());
  const auto num_rows_out = data.num_rows_out;
  const auto out_idx_a = data.workspace.out_idx_a;
  const auto out_idx_b = data.workspace.out_idx_b;
  const auto row_ptr_a = data.mesh_a.row_ptr;
  const auto row_ptr_b = data.mesh_b.row_ptr;
  const auto intervals_a = data.mesh_a.intervals;
  const auto intervals_b = data.mesh_b.intervals;
  auto result_intervals = data.result.intervals;
  auto result_row_ptr = data.result.row_ptr;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_fill",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i) {
          const int ia = out_idx_a(i);
          const int ib = out_idx_b(i);

          if (ib < 0) {
            return;
          }

          const auto r = playground::subsetix::csr::intersection::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

          if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
            return;
          }

          playground::subsetix::csr::intersection::baseline::detail::row_intersection_impl<false>(
              intervals_a, r.begin_a, r.end_a,
              intervals_b, r.begin_b, r.end_b,
              result_intervals, result_row_ptr(i));
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_out);
}

BENCHMARK(Phase6_FillIntervals_2D_Small)->Unit(benchmark::kMicrosecond);

static void Phase6_FillIntervals_2D_Medium(benchmark::State& state) {
  PhaseBenchmarkData data(MediumRegularConfig());
  const auto num_rows_out = data.num_rows_out;
  const auto out_idx_a = data.workspace.out_idx_a;
  const auto out_idx_b = data.workspace.out_idx_b;
  const auto row_ptr_a = data.mesh_a.row_ptr;
  const auto row_ptr_b = data.mesh_b.row_ptr;
  const auto intervals_a = data.mesh_a.intervals;
  const auto intervals_b = data.mesh_b.intervals;
  auto result_intervals = data.result.intervals;
  auto result_row_ptr = data.result.row_ptr;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_fill",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i) {
          const int ia = out_idx_a(i);
          const int ib = out_idx_b(i);

          if (ib < 0) {
            return;
          }

          const auto r = playground::subsetix::csr::intersection::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

          if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
            return;
          }

          playground::subsetix::csr::intersection::baseline::detail::row_intersection_impl<false>(
              intervals_a, r.begin_a, r.end_a,
              intervals_b, r.begin_b, r.end_b,
              result_intervals, result_row_ptr(i));
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_out);
}

BENCHMARK(Phase6_FillIntervals_2D_Medium)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Phase 6: Fill Intersected Intervals (3D)
// ============================================================================

static void Phase6_FillIntervals_3D_Small(benchmark::State& state) {
  PhaseBenchmarkData3D data(SmallRegularConfig());
  const auto num_rows_out = data.num_rows_out;
  const auto out_idx_a = data.workspace.out_idx_a;
  const auto out_idx_b = data.workspace.out_idx_b;
  const auto row_ptr_a = data.mesh_a.row_ptr;
  const auto row_ptr_b = data.mesh_b.row_ptr;
  const auto intervals_a = data.mesh_a.intervals;
  const auto intervals_b = data.mesh_b.intervals;
  auto result_intervals = data.result.intervals;
  auto result_row_ptr = data.result.row_ptr;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_fill_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i) {
          const int ia = out_idx_a(i);
          const int ib = out_idx_b(i);

          if (ib < 0) {
            return;
          }

          const auto r = playground::subsetix::csr::intersection::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

          if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
            return;
          }

          playground::subsetix::csr::intersection::baseline::detail::row_intersection_impl<false>(
              intervals_a, r.begin_a, r.end_a,
              intervals_b, r.begin_b, r.end_b,
              result_intervals, result_row_ptr(i));
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_out);
}

BENCHMARK(Phase6_FillIntervals_3D_Small)->Unit(benchmark::kMicrosecond);

static void Phase6_FillIntervals_3D_Medium(benchmark::State& state) {
  PhaseBenchmarkData3D data(MediumRegularConfig());
  const auto num_rows_out = data.num_rows_out;
  const auto out_idx_a = data.workspace.out_idx_a;
  const auto out_idx_b = data.workspace.out_idx_b;
  const auto row_ptr_a = data.mesh_a.row_ptr;
  const auto row_ptr_b = data.mesh_b.row_ptr;
  const auto intervals_a = data.mesh_a.intervals;
  const auto intervals_b = data.mesh_b.intervals;
  auto result_intervals = data.result.intervals;
  auto result_row_ptr = data.result.row_ptr;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_fill_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i) {
          const int ia = out_idx_a(i);
          const int ib = out_idx_b(i);

          if (ib < 0) {
            return;
          }

          const auto r = playground::subsetix::csr::intersection::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

          if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
            return;
          }

          playground::subsetix::csr::intersection::baseline::detail::row_intersection_impl<false>(
              intervals_a, r.begin_a, r.end_a,
              intervals_b, r.begin_b, r.end_b,
              result_intervals, result_row_ptr(i));
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_out);
}

BENCHMARK(Phase6_FillIntervals_3D_Medium)->Unit(benchmark::kMicrosecond);

static void Phase6_FillIntervals_3D_Large(benchmark::State& state) {
  PhaseBenchmarkData3D data(LargeRegularConfig());
  const auto num_rows_out = data.num_rows_out;
  const auto out_idx_a = data.workspace.out_idx_a;
  const auto out_idx_b = data.workspace.out_idx_b;
  const auto row_ptr_a = data.mesh_a.row_ptr;
  const auto row_ptr_b = data.mesh_b.row_ptr;
  const auto intervals_a = data.mesh_a.intervals;
  const auto intervals_b = data.mesh_b.intervals;
  auto result_intervals = data.result.intervals;
  auto result_row_ptr = data.result.row_ptr;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_fill_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i) {
          const int ia = out_idx_a(i);
          const int ib = out_idx_b(i);

          if (ib < 0) {
            return;
          }

          const auto r = playground::subsetix::csr::intersection::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

          if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
            return;
          }

          playground::subsetix::csr::intersection::baseline::detail::row_intersection_impl<false>(
              intervals_a, r.begin_a, r.end_a,
              intervals_b, r.begin_b, r.end_b,
              result_intervals, result_row_ptr(i));
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_out);
}

BENCHMARK(Phase6_FillIntervals_3D_Large)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Random Mesh Benchmark Data Structures
// ============================================================================

struct PhaseBenchmarkDataRandom2D {
  baseline::Mesh2DDevice mesh_a, mesh_b;
  baseline::Mesh2DDevice result;
  Workspace workspace;
  std::size_t num_rows_out;

  PhaseBenchmarkDataRandom2D(const RandomMeshConfig& cfg) {
    // Generate input meshes with different seeds
    auto common_a = RandomMeshGenerator::generate_2d(cfg);
    RandomMeshConfig cfg_b = cfg;
    ++cfg_b.seed;
    auto common_b = RandomMeshGenerator::generate_2d(cfg_b);

    // Convert to device format
    mesh_a = MeshConverter2D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(common_a);
    mesh_b = MeshConverter2D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(common_b);

    // Allocate workspace and result
    std::size_t max_rows = std::max(mesh_a.num_rows, mesh_b.num_rows);
    std::size_t max_intervals = std::max(mesh_a.num_intervals, mesh_b.num_intervals);

    workspace.ensure_capacity(max_rows, max_intervals);

    result.row_keys = Kokkos::View<RowKey2D<Coord>*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_keys", max_rows);
    result.row_ptr = Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_ptr", max_rows + 1);
    result.intervals = Kokkos::View<IntervalType*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_intervals", max_intervals);

    // Pre-compute row mapping phase data
    precompute_row_mapping();
  }

  void precompute_row_mapping() {
    const std::size_t num_rows_a = mesh_a.num_rows;
    const auto row_keys_a = mesh_a.row_keys;
    const auto row_keys_b = mesh_b.row_keys;
    const auto num_rows_b = mesh_b.num_rows;
    const auto flags = workspace.flags;
    const auto tmp_idx_a = workspace.tmp_idx_a;
    const auto tmp_idx_b = workspace.tmp_idx_b;
    const auto positions = workspace.positions;
    const auto out_rows = workspace.out_rows;
    const auto out_idx_a = workspace.out_idx_a;
    const auto out_idx_b = workspace.out_idx_b;
    const auto num_rows_out_view = workspace.num_rows_out_view;

    // Phase 1: Row Mapping
    Kokkos::parallel_for(
        "phase_benchmark_row_mapping_random",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey2D<Coord> key = row_keys_a(i);
          const int idx_b = playground::subsetix::csr::intersection::detail::find_row_by_y(
              row_keys_b, num_rows_b, key.y);

          flags(i) = (idx_b >= 0) ? 1 : 0;
          tmp_idx_a(i) = (idx_b >= 0) ? static_cast<int>(i) : -1;
          tmp_idx_b(i) = idx_b;
        });

    Kokkos::fence();

    // Phase 2: Row Scan
    Kokkos::parallel_scan(
        "phase_benchmark_row_scan_random",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = flags(i);
          if (final_pass) {
            positions(i) = update;
          }
          update += count;
        },
        num_rows_out_view);

    Kokkos::fence();

    // Extract num_rows_out from device view to host
    Kokkos::deep_copy(num_rows_out, workspace.num_rows_out_view);

    // Phase 3: Row Compaction
    Kokkos::parallel_for(
        "phase_benchmark_row_compact_random",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          if (flags(i) == 1) {
            const std::size_t pos = positions(i);
            out_rows(pos) = row_keys_a(i);
            out_idx_a(pos) = tmp_idx_a(i);
            out_idx_b(pos) = tmp_idx_b(i);
          }
        });

    Kokkos::fence();
  }
};

struct PhaseBenchmarkDataRandom3D {
  baseline::Mesh3DDevice mesh_a, mesh_b;
  baseline::Mesh3DDevice result;
  IntersectionWorkspace3D<Kokkos::DefaultExecutionSpace> workspace;
  std::size_t num_rows_out;

  PhaseBenchmarkDataRandom3D(const RandomMeshConfig& cfg) {
    // Generate input meshes with different seeds
    auto common_a = RandomMeshGenerator::generate_3d(cfg);
    RandomMeshConfig cfg_b = cfg;
    ++cfg_b.seed;
    auto common_b = RandomMeshGenerator::generate_3d(cfg_b);

    // Convert to device format
    mesh_a = MeshConverter3D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(common_a);
    mesh_b = MeshConverter3D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(common_b);

    // Allocate workspace and result
    std::size_t max_rows = std::max(mesh_a.num_rows, mesh_b.num_rows);
    std::size_t max_intervals = std::max(mesh_a.num_intervals, mesh_b.num_intervals);

    workspace.ensure_capacity(max_rows, max_intervals);

    result.row_keys = Kokkos::View<RowKey3D<Coord>*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_keys", max_rows);
    result.row_ptr = Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_ptr", max_rows + 1);
    result.intervals = Kokkos::View<IntervalType*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_intervals", max_intervals);

    // Pre-compute row mapping phase data
    precompute_row_mapping();
  }

  void precompute_row_mapping() {
    const std::size_t num_rows_a = mesh_a.num_rows;
    const auto row_keys_a = mesh_a.row_keys;
    const auto row_keys_b = mesh_b.row_keys;
    const auto num_rows_b = mesh_b.num_rows;
    const auto flags = workspace.flags;
    const auto tmp_idx_a = workspace.tmp_idx_a;
    const auto tmp_idx_b = workspace.tmp_idx_b;
    const auto positions = workspace.positions;
    const auto out_rows = workspace.out_rows;
    const auto out_idx_a = workspace.out_idx_a;
    const auto out_idx_b = workspace.out_idx_b;
    const auto num_rows_out_view = workspace.num_rows_out_view;

    // Phase 1: Row Mapping (3D: search by y,z)
    Kokkos::parallel_for(
        "phase_benchmark_row_mapping_random_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey3D<Coord> key = row_keys_a(i);
          const int idx_b = playground::subsetix::csr::intersection::detail::find_row_by_yz(
              row_keys_b, num_rows_b, key.y, key.z);

          flags(i) = (idx_b >= 0) ? 1 : 0;
          tmp_idx_a(i) = (idx_b >= 0) ? static_cast<int>(i) : -1;
          tmp_idx_b(i) = idx_b;
        });

    Kokkos::fence();

    // Phase 2: Row Scan
    Kokkos::parallel_scan(
        "phase_benchmark_row_scan_random_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = flags(i);
          if (final_pass) {
            positions(i) = update;
          }
          update += count;
        },
        num_rows_out_view);

    Kokkos::fence();

    // Extract num_rows_out from device view to host
    Kokkos::deep_copy(num_rows_out, workspace.num_rows_out_view);

    // Phase 3: Row Compaction
    Kokkos::parallel_for(
        "phase_benchmark_row_compact_random_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          if (flags(i) == 1) {
            const std::size_t pos = positions(i);
            out_rows(pos) = row_keys_a(i);
            out_idx_a(pos) = tmp_idx_a(i);
            out_idx_b(pos) = tmp_idx_b(i);
          }
        });

    Kokkos::fence();
  }
};

// ============================================================================
// Random Mesh Benchmarks - Full Intersection (2D)
// ============================================================================

static void Full_Intersection_Random_2D_Small(benchmark::State& state) {
  PhaseBenchmarkDataRandom2D data(SmallConfig());
  const auto total_intervals = data.mesh_a.num_intervals + data.mesh_b.num_intervals;
  const auto& mesh_a = data.mesh_a;
  const auto& mesh_b = data.mesh_b;
  auto& result = data.result;
  auto& workspace = data.workspace;

  for (auto _ : state) {
    baseline::intersect_meshes_2d_in_place(mesh_a, mesh_b, result, workspace);

    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK(Full_Intersection_Random_2D_Small)->Unit(benchmark::kMicrosecond);

static void Full_Intersection_Random_2D_Medium(benchmark::State& state) {
  PhaseBenchmarkDataRandom2D data(MediumConfig());
  const auto total_intervals = data.mesh_a.num_intervals + data.mesh_b.num_intervals;
  const auto& mesh_a = data.mesh_a;
  const auto& mesh_b = data.mesh_b;
  auto& result = data.result;
  auto& workspace = data.workspace;

  for (auto _ : state) {
    baseline::intersect_meshes_2d_in_place(mesh_a, mesh_b, result, workspace);

    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK(Full_Intersection_Random_2D_Medium)->Unit(benchmark::kMicrosecond);

static void Full_Intersection_Random_2D_Large(benchmark::State& state) {
  PhaseBenchmarkDataRandom2D data(LargeConfig());
  const auto total_intervals = data.mesh_a.num_intervals + data.mesh_b.num_intervals;
  const auto& mesh_a = data.mesh_a;
  const auto& mesh_b = data.mesh_b;
  auto& result = data.result;
  auto& workspace = data.workspace;

  for (auto _ : state) {
    baseline::intersect_meshes_2d_in_place(mesh_a, mesh_b, result, workspace);

    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK(Full_Intersection_Random_2D_Large)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Random Mesh Benchmarks - Full Intersection (3D)
// ============================================================================

static void Full_Intersection_Random_3D_Small(benchmark::State& state) {
  PhaseBenchmarkDataRandom3D data(SmallConfig());
  const auto total_intervals = data.mesh_a.num_intervals + data.mesh_b.num_intervals;
  const auto& mesh_a = data.mesh_a;
  const auto& mesh_b = data.mesh_b;
  auto& result = data.result;
  auto& workspace = data.workspace;

  for (auto _ : state) {
    baseline::intersect_meshes_3d_in_place(mesh_a, mesh_b, result, workspace);

    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK(Full_Intersection_Random_3D_Small)->Unit(benchmark::kMicrosecond);

static void Full_Intersection_Random_3D_Medium(benchmark::State& state) {
  PhaseBenchmarkDataRandom3D data(MediumConfig());
  const auto total_intervals = data.mesh_a.num_intervals + data.mesh_b.num_intervals;
  const auto& mesh_a = data.mesh_a;
  const auto& mesh_b = data.mesh_b;
  auto& result = data.result;
  auto& workspace = data.workspace;

  for (auto _ : state) {
    baseline::intersect_meshes_3d_in_place(mesh_a, mesh_b, result, workspace);

    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK(Full_Intersection_Random_3D_Medium)->Unit(benchmark::kMicrosecond);

static void Full_Intersection_Random_3D_Large(benchmark::State& state) {
  PhaseBenchmarkDataRandom3D data(LargeConfig());
  const auto total_intervals = data.mesh_a.num_intervals + data.mesh_b.num_intervals;
  const auto& mesh_a = data.mesh_a;
  const auto& mesh_b = data.mesh_b;
  auto& result = data.result;
  auto& workspace = data.workspace;

  for (auto _ : state) {
    baseline::intersect_meshes_3d_in_place(mesh_a, mesh_b, result, workspace);

    benchmark::DoNotOptimize(result.num_rows);
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * total_intervals);
  state.SetBytesProcessed(state.iterations() * total_intervals * sizeof(IntervalType));
}

BENCHMARK(Full_Intersection_Random_3D_Large)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Random Mesh Benchmarks - Phase 1: Row Mapping (2D)
// ============================================================================

static void Phase1_RowMapping_Random_2D_Small(benchmark::State& state) {
  PhaseBenchmarkDataRandom2D data(SmallConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto row_keys_a = data.mesh_a.row_keys;
  const auto row_keys_b = data.mesh_b.row_keys;
  const auto num_rows_b = data.mesh_b.num_rows;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_row_mapping_random",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey2D<Coord> key = row_keys_a(i);
          const int idx_b = playground::subsetix::csr::intersection::detail::find_row_by_y(
              row_keys_b, num_rows_b, key.y);
          benchmark::DoNotOptimize(idx_b);
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase1_RowMapping_Random_2D_Small)->Unit(benchmark::kMicrosecond);

static void Phase1_RowMapping_Random_2D_Medium(benchmark::State& state) {
  PhaseBenchmarkDataRandom2D data(MediumConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto row_keys_a = data.mesh_a.row_keys;
  const auto row_keys_b = data.mesh_b.row_keys;
  const auto num_rows_b = data.mesh_b.num_rows;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_row_mapping_random",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey2D<Coord> key = row_keys_a(i);
          const int idx_b = playground::subsetix::csr::intersection::detail::find_row_by_y(
              row_keys_b, num_rows_b, key.y);
          benchmark::DoNotOptimize(idx_b);
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase1_RowMapping_Random_2D_Medium)->Unit(benchmark::kMicrosecond);

static void Phase1_RowMapping_Random_2D_Large(benchmark::State& state) {
  PhaseBenchmarkDataRandom2D data(LargeConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto row_keys_a = data.mesh_a.row_keys;
  const auto row_keys_b = data.mesh_b.row_keys;
  const auto num_rows_b = data.mesh_b.num_rows;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_row_mapping_random",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey2D<Coord> key = row_keys_a(i);
          const int idx_b = playground::subsetix::csr::intersection::detail::find_row_by_y(
              row_keys_b, num_rows_b, key.y);
          benchmark::DoNotOptimize(idx_b);
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase1_RowMapping_Random_2D_Large)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Random Mesh Benchmarks - Phase 2: Row Scan (2D)
// ============================================================================

static void Phase2_RowScan_Random_2D_Small(benchmark::State& state) {
  PhaseBenchmarkDataRandom2D data(SmallConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto flags = data.workspace.flags;
  const auto positions = data.workspace.positions;

  for (auto _ : state) {
    Kokkos::parallel_scan(
        "phase_row_scan_random",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = flags(i);
          if (final_pass) {
            positions(i) = update;
          }
          update += count;
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase2_RowScan_Random_2D_Small)->Unit(benchmark::kMicrosecond);

static void Phase2_RowScan_Random_2D_Medium(benchmark::State& state) {
  PhaseBenchmarkDataRandom2D data(MediumConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto flags = data.workspace.flags;
  const auto positions = data.workspace.positions;

  for (auto _ : state) {
    Kokkos::parallel_scan(
        "phase_row_scan_random",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = flags(i);
          if (final_pass) {
            positions(i) = update;
          }
          update += count;
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase2_RowScan_Random_2D_Medium)->Unit(benchmark::kMicrosecond);

static void Phase2_RowScan_Random_2D_Large(benchmark::State& state) {
  PhaseBenchmarkDataRandom2D data(LargeConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto flags = data.workspace.flags;
  const auto positions = data.workspace.positions;

  for (auto _ : state) {
    Kokkos::parallel_scan(
        "phase_row_scan_random",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = flags(i);
          if (final_pass) {
            positions(i) = update;
          }
          update += count;
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase2_RowScan_Random_2D_Large)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Random Mesh Benchmarks - Phase 1: Row Mapping (3D)
// ============================================================================

static void Phase1_RowMapping_Random_3D_Small(benchmark::State& state) {
  PhaseBenchmarkDataRandom3D data(SmallConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto row_keys_a = data.mesh_a.row_keys;
  const auto row_keys_b = data.mesh_b.row_keys;
  const auto num_rows_b = data.mesh_b.num_rows;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_row_mapping_random_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey3D<Coord> key = row_keys_a(i);
          const int idx_b = playground::subsetix::csr::intersection::detail::find_row_by_yz(
              row_keys_b, num_rows_b, key.y, key.z);
          benchmark::DoNotOptimize(idx_b);
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase1_RowMapping_Random_3D_Small)->Unit(benchmark::kMicrosecond);

static void Phase1_RowMapping_Random_3D_Medium(benchmark::State& state) {
  PhaseBenchmarkDataRandom3D data(MediumConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto row_keys_a = data.mesh_a.row_keys;
  const auto row_keys_b = data.mesh_b.row_keys;
  const auto num_rows_b = data.mesh_b.num_rows;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_row_mapping_random_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey3D<Coord> key = row_keys_a(i);
          const int idx_b = playground::subsetix::csr::intersection::detail::find_row_by_yz(
              row_keys_b, num_rows_b, key.y, key.z);
          benchmark::DoNotOptimize(idx_b);
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase1_RowMapping_Random_3D_Medium)->Unit(benchmark::kMicrosecond);

static void Phase1_RowMapping_Random_3D_Large(benchmark::State& state) {
  PhaseBenchmarkDataRandom3D data(LargeConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto row_keys_a = data.mesh_a.row_keys;
  const auto row_keys_b = data.mesh_b.row_keys;
  const auto num_rows_b = data.mesh_b.num_rows;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "phase_row_mapping_random_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey3D<Coord> key = row_keys_a(i);
          const int idx_b = playground::subsetix::csr::intersection::detail::find_row_by_yz(
              row_keys_b, num_rows_b, key.y, key.z);
          benchmark::DoNotOptimize(idx_b);
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase1_RowMapping_Random_3D_Large)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Random Mesh Benchmarks - Phase 2: Row Scan (3D)
// ============================================================================

static void Phase2_RowScan_Random_3D_Small(benchmark::State& state) {
  PhaseBenchmarkDataRandom3D data(SmallConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto flags = data.workspace.flags;
  const auto positions = data.workspace.positions;

  for (auto _ : state) {
    Kokkos::parallel_scan(
        "phase_row_scan_random_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = flags(i);
          if (final_pass) {
            positions(i) = update;
          }
          update += count;
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase2_RowScan_Random_3D_Small)->Unit(benchmark::kMicrosecond);

static void Phase2_RowScan_Random_3D_Medium(benchmark::State& state) {
  PhaseBenchmarkDataRandom3D data(MediumConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto flags = data.workspace.flags;
  const auto positions = data.workspace.positions;

  for (auto _ : state) {
    Kokkos::parallel_scan(
        "phase_row_scan_random_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = flags(i);
          if (final_pass) {
            positions(i) = update;
          }
          update += count;
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase2_RowScan_Random_3D_Medium)->Unit(benchmark::kMicrosecond);

static void Phase2_RowScan_Random_3D_Large(benchmark::State& state) {
  PhaseBenchmarkDataRandom3D data(LargeConfig());
  const auto num_rows_a = data.mesh_a.num_rows;
  const auto flags = data.workspace.flags;
  const auto positions = data.workspace.positions;

  for (auto _ : state) {
    Kokkos::parallel_scan(
        "phase_row_scan_random_3d",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = flags(i);
          if (final_pass) {
            positions(i) = update;
          }
          update += count;
        });
    Kokkos::fence();
  }

  state.SetItemsProcessed(state.iterations() * num_rows_a);
}

BENCHMARK(Phase2_RowScan_Random_3D_Large)->Unit(benchmark::kMicrosecond);

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
