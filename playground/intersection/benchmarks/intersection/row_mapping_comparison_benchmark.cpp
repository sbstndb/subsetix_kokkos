// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#include <benchmark/benchmark.h>
#include <playground/subsetix/csr/intersection/algorithm/optimized.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v4_hash.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v5_parallel_merge.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v6_direct_index.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v7_soa_optimized.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v8_hybrid_cpu_gpu.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v9_adaptive.hpp>
#include <playground/subsetix/csr/intersection/set_algebra.hpp>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

// Bring version namespaces into scope
using namespace playground::subsetix::csr::intersection;

// Type aliases for convenience
using Coord = int32_t;
using IntervalType = playground::subsetix::csr::intersection::Interval<Coord>;

// Type aliases for each version
using OptimizedMesh2D = optimized::Mesh2D;
using OptimizedMesh3D = optimized::Mesh3D;

using V4Mesh2D = v4_hash::Mesh2D;
using V4Mesh3D = v4_hash::Mesh3D;

using V5Mesh2D = v5_parallel_merge::Mesh2D;
using V5Mesh3D = v5_parallel_merge::Mesh3D;

using V6Mesh2D = v6_direct_index::Mesh2D;
using V6Mesh3D = v6_direct_index::Mesh3D;

using V7Mesh2D = v7_soa_optimized::Mesh2D;
using V7Mesh3D = v7_soa_optimized::Mesh3D;

using V8Mesh2D = v8_hybrid_cpu_gpu::Mesh2D;
using V8Mesh3D = v8_hybrid_cpu_gpu::Mesh3D;

using V9Mesh2D = v9_adaptive::Mesh2D;
using V9Mesh3D = v9_adaptive::Mesh3D;

// ============================================================================
// Mesh Generation Helpers
// ============================================================================

namespace mesh_generator {

/**
 * @brief Generate dense mesh (consecutive coordinates)
 *
 * Creates a mesh where all rows are present with consecutive Y coordinates:
 * - y = y_start, y_start+1, ..., y_start+num_rows-1
 * - Each row has one interval [0, 10)
 */
template <typename MeshType>
MeshType generate_dense_mesh_2d(std::size_t num_rows, int32_t y_start = 0) {
  using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
  using RowKey = playground::subsetix::csr::intersection::RowKey2D<Coord>;

  MeshType mesh;
  mesh.num_rows = num_rows;
  mesh.num_intervals = num_rows;

  mesh.row_keys = Kokkos::View<RowKey*, DeviceMemorySpace>("row_keys", num_rows);
  mesh.row_ptr = Kokkos::View<std::size_t*, DeviceMemorySpace>("row_ptr", num_rows + 1);
  mesh.intervals = Kokkos::View<IntervalType*, DeviceMemorySpace>("intervals", num_rows);

  auto host_keys = Kokkos::create_mirror_view(mesh.row_keys);
  auto host_ptr = Kokkos::create_mirror_view(mesh.row_ptr);
  auto host_intervals = Kokkos::create_mirror_view(mesh.intervals);

  std::size_t offset = 0;
  for (std::size_t i = 0; i < num_rows; ++i) {
    host_keys(i).y = y_start + static_cast<int32_t>(i);
    host_ptr(i) = offset;
    host_intervals(i) = {0, 10};
    offset += 1;
  }
  host_ptr(num_rows) = offset;

  Kokkos::deep_copy(mesh.row_keys, host_keys);
  Kokkos::deep_copy(mesh.row_ptr, host_ptr);
  Kokkos::deep_copy(mesh.intervals, host_intervals);

  return mesh;
}

/**
 * @brief Generate mesh with uniform stride
 *
 * Creates a mesh where rows are evenly spaced:
 * - y = y_start, y_start+stride, y_start+2*stride, ...
 * - Each row has one interval [0, 10)
 */
template <typename MeshType>
MeshType generate_stride_mesh_2d(std::size_t num_rows, int32_t stride, int32_t y_start = 0) {
  using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
  using RowKey = playground::subsetix::csr::intersection::RowKey2D<Coord>;

  MeshType mesh;
  mesh.num_rows = num_rows;
  mesh.num_intervals = num_rows;

  mesh.row_keys = Kokkos::View<RowKey*, DeviceMemorySpace>("row_keys", num_rows);
  mesh.row_ptr = Kokkos::View<std::size_t*, DeviceMemorySpace>("row_ptr", num_rows + 1);
  mesh.intervals = Kokkos::View<IntervalType*, DeviceMemorySpace>("intervals", num_rows);

  auto host_keys = Kokkos::create_mirror_view(mesh.row_keys);
  auto host_ptr = Kokkos::create_mirror_view(mesh.row_ptr);
  auto host_intervals = Kokkos::create_mirror_view(mesh.intervals);

  std::size_t offset = 0;
  for (std::size_t i = 0; i < num_rows; ++i) {
    host_keys(i).y = y_start + static_cast<int32_t>(i) * stride;
    host_ptr(i) = offset;
    host_intervals(i) = {0, 10};
    offset += 1;
  }
  host_ptr(num_rows) = offset;

  Kokkos::deep_copy(mesh.row_keys, host_keys);
  Kokkos::deep_copy(mesh.row_ptr, host_ptr);
  Kokkos::deep_copy(mesh.intervals, host_intervals);

  return mesh;
}

/**
 * @brief Generate random sparse mesh
 *
 * Creates a mesh with random Y coordinates sampled without replacement.
 * This simulates realistic sparse geometries.
 */
template <typename MeshType>
MeshType generate_random_mesh_2d(std::size_t num_rows, std::size_t coord_range, uint64_t seed = 42) {
  using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
  using RowKey = playground::subsetix::csr::intersection::RowKey2D<Coord>;

  std::mt19937 gen(seed);
  std::uniform_int_distribution<int32_t> dist(0, coord_range - 1);

  // Generate unique Y coordinates using Fisher-Yates shuffle
  std::vector<int32_t> all_coords(coord_range);
  std::iota(all_coords.begin(), all_coords.end(), 0);

  std::size_t actual_rows = std::min(num_rows, coord_range);
  for (std::size_t i = 0; i < actual_rows; ++i) {
    std::uniform_int_distribution<std::size_t> swap_dist(i, coord_range - 1);
    std::size_t j = swap_dist(gen);
    std::swap(all_coords[i], all_coords[j]);
  }

  // Keep only selected coordinates and sort them
  all_coords.resize(actual_rows);
  std::sort(all_coords.begin(), all_coords.end());

  MeshType mesh;
  mesh.num_rows = actual_rows;
  mesh.num_intervals = actual_rows;

  mesh.row_keys = Kokkos::View<RowKey*, DeviceMemorySpace>("row_keys", actual_rows);
  mesh.row_ptr = Kokkos::View<std::size_t*, DeviceMemorySpace>("row_ptr", actual_rows + 1);
  mesh.intervals = Kokkos::View<IntervalType*, DeviceMemorySpace>("intervals", actual_rows);

  auto host_keys = Kokkos::create_mirror_view(mesh.row_keys);
  auto host_ptr = Kokkos::create_mirror_view(mesh.row_ptr);
  auto host_intervals = Kokkos::create_mirror_view(mesh.intervals);

  std::mt19937 gen_interval(seed + 1);
  std::uniform_int_distribution<int32_t> length_dist(5, 20);

  std::size_t offset = 0;
  for (std::size_t i = 0; i < actual_rows; ++i) {
    host_keys(i).y = all_coords[i];
    host_ptr(i) = offset;
    int32_t length = length_dist(gen_interval);
    host_intervals(i) = {0, length};
    offset += 1;
  }
  host_ptr(actual_rows) = offset;

  Kokkos::deep_copy(mesh.row_keys, host_keys);
  Kokkos::deep_copy(mesh.row_ptr, host_ptr);
  Kokkos::deep_copy(mesh.intervals, host_intervals);

  return mesh;
}

/**
 * @brief Generate mesh with overlapping region
 *
 * Creates two meshes with a controlled overlap:
 * - Mesh A: y in [0, overlap_end)
 * - Mesh B: y in [overlap_start, overlap_end + non_overlap_size)
 */
template <typename MeshType>
struct OverlappingMeshPair {
  MeshType mesh_a;
  MeshType mesh_b;
};

template <typename MeshType>
OverlappingMeshPair<MeshType> generate_overlapping_meshes_2d(
    std::size_t overlap_size,
    std::size_t non_overlap_size,
    uint64_t seed = 42) {

  OverlappingMeshPair<MeshType> result;

  // Mesh A: [0, overlap_size)
  result.mesh_a = generate_dense_mesh_2d<MeshType>(overlap_size, 0);

  // Mesh B: [0, overlap_size + non_overlap_size)
  result.mesh_b = generate_dense_mesh_2d<MeshType>(overlap_size + non_overlap_size, 0);

  return result;
}

/**
 * @brief Generate mesh with no overlap
 *
 * Creates two meshes with disjoint Y coordinates:
 * - Mesh A: y in [0, size_a)
 * - Mesh B: y in [size_a, size_a + size_b)
 */
template <typename MeshType>
OverlappingMeshPair<MeshType> generate_disjoint_meshes_2d(
    std::size_t size_a,
    std::size_t size_b) {

  OverlappingMeshPair<MeshType> result;

  result.mesh_a = generate_dense_mesh_2d<MeshType>(size_a, 0);
  result.mesh_b = generate_dense_mesh_2d<MeshType>(size_b, size_a);

  return result;
}

} // namespace mesh_generator

// ============================================================================
// Benchmark Template
// ============================================================================

/**
 * @brief Generic benchmark for row mapping performance
 *
 * Measures the time to intersect two meshes and counts:
 * - Total intervals processed
 * - Bytes processed (row_keys + intervals)
 * - Average intervals per iteration
 */
template <typename MeshType, typename IntersectFunc>
void BM_RowMapping_Generic(benchmark::State& state,
                           const std::string& version,
                           IntersectFunc intersect_func,
                           MeshType mesh_a,
                           MeshType mesh_b) {
  std::size_t total_intervals = 0;
  std::size_t total_rows_out = 0;

  for (auto _ : state) {
    auto result = intersect_func(mesh_a, mesh_b);
    benchmark::DoNotOptimize(result);
    Kokkos::fence();
    total_intervals += result.num_intervals;
    total_rows_out += result.num_rows;
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(
      state.iterations() * (
        mesh_a.num_rows * sizeof(typename MeshType::RowKey) +
        mesh_b.num_rows * sizeof(typename MeshType::RowKey) +
        mesh_a.num_intervals * sizeof(IntervalType) +
        mesh_b.num_intervals * sizeof(IntervalType)
      )
  );

  // Custom counters
  state.counters["rows_a"] = benchmark::Counter(mesh_a.num_rows, benchmark::Counter::kAvgIterations);
  state.counters["rows_b"] = benchmark::Counter(mesh_b.num_rows, benchmark::Counter::kAvgIterations);
  state.counters["avg_intervals_out"] = benchmark::Counter(
      static_cast<double>(total_intervals) / state.iterations(),
      benchmark::Counter::kAvgIterations);
  state.counters["avg_rows_out"] = benchmark::Counter(
      static_cast<double>(total_rows_out) / state.iterations(),
      benchmark::Counter::kAvgIterations);
}

// ============================================================================
// Size-Based Benchmarks (2D)
// ============================================================================

void register_size_benchmarks_2d() {
  // Small: 100 x 100
  auto small_100_a = mesh_generator::generate_dense_mesh_2d<OptimizedMesh2D>(100, 0);
  auto small_100_b = mesh_generator::generate_dense_mesh_2d<OptimizedMesh2D>(100, 0);

  // v2 baseline (optimized)
  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_small_100x100_v2_baseline,
      "v2_baseline",
      [](const auto& a, const auto& b) { return optimized::intersect_meshes_2d(a, b); },
      small_100_a, small_100_b)
      ->Unit(benchmark::kMicrosecond);

  // v4 hash
  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_small_100x100_v4_hash,
      "v4_hash",
      [](const auto& a, const auto& b) { return v4_hash::intersect_meshes_2d(a, b); },
      small_100_a, small_100_b)
      ->Unit(benchmark::kMicrosecond);

  // v5 parallel merge
  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_small_100x100_v5_parallel_merge,
      "v5_parallel_merge",
      [](const auto& a, const auto& b) { return v5_parallel_merge::intersect_meshes_2d(a, b); },
      small_100_a, small_100_b)
      ->Unit(benchmark::kMicrosecond);

  // v6 direct index
  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_small_100x100_v6_direct_index,
      "v6_direct_index",
      [](const auto& a, const auto& b) { return v6_direct_index::intersect_meshes_2d(a, b); },
      small_100_a, small_100_b)
      ->Unit(benchmark::kMicrosecond);

  // v7 SoA optimized
  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_small_100x100_v7_soa_optimized,
      "v7_soa_optimized",
      [](const auto& a, const auto& b) { return v7_soa_optimized::intersect_meshes_2d(a, b); },
      small_100_a, small_100_b)
      ->Unit(benchmark::kMicrosecond);

  // v8 hybrid CPU-GPU
  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_small_100x100_v8_hybrid,
      "v8_hybrid_cpu_gpu",
      [](const auto& a, const auto& b) { return v8_hybrid_cpu_gpu::intersect_meshes_2d(a, b); },
      small_100_a, small_100_b)
      ->Unit(benchmark::kMicrosecond);

  // v9 adaptive
  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_small_100x100_v9_adaptive,
      "v9_adaptive",
      [](const auto& a, const auto& b) { return v9_adaptive::intersect_meshes_2d(a, b); },
      small_100_a, small_100_b)
      ->Unit(benchmark::kMicrosecond);

  // Medium: 1K x 1K
  auto medium_1k_a = mesh_generator::generate_dense_mesh_2d<OptimizedMesh2D>(1000, 0);
  auto medium_1k_b = mesh_generator::generate_dense_mesh_2d<OptimizedMesh2D>(1000, 0);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_medium_1Kx1K_v2_baseline,
      "v2_baseline",
      [](const auto& a, const auto& b) { return optimized::intersect_meshes_2d(a, b); },
      medium_1k_a, medium_1k_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_medium_1Kx1K_v4_hash,
      "v4_hash",
      [](const auto& a, const auto& b) { return v4_hash::intersect_meshes_2d(a, b); },
      medium_1k_a, medium_1k_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_medium_1Kx1K_v5_parallel_merge,
      "v5_parallel_merge",
      [](const auto& a, const auto& b) { return v5_parallel_merge::intersect_meshes_2d(a, b); },
      medium_1k_a, medium_1k_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_medium_1Kx1K_v6_direct_index,
      "v6_direct_index",
      [](const auto& a, const auto& b) { return v6_direct_index::intersect_meshes_2d(a, b); },
      medium_1k_a, medium_1k_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_medium_1Kx1K_v7_soa_optimized,
      "v7_soa_optimized",
      [](const auto& a, const auto& b) { return v7_soa_optimized::intersect_meshes_2d(a, b); },
      medium_1k_a, medium_1k_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_medium_1Kx1K_v8_hybrid,
      "v8_hybrid_cpu_gpu",
      [](const auto& a, const auto& b) { return v8_hybrid_cpu_gpu::intersect_meshes_2d(a, b); },
      medium_1k_a, medium_1k_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_medium_1Kx1K_v9_adaptive,
      "v9_adaptive",
      [](const auto& a, const auto& b) { return v9_adaptive::intersect_meshes_2d(a, b); },
      medium_1k_a, medium_1k_b)
      ->Unit(benchmark::kMicrosecond);

  // Large: 10K x 10K
  auto large_10k_a = mesh_generator::generate_dense_mesh_2d<OptimizedMesh2D>(10000, 0);
  auto large_10k_b = mesh_generator::generate_dense_mesh_2d<OptimizedMesh2D>(10000, 0);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_large_10Kx10K_v2_baseline,
      "v2_baseline",
      [](const auto& a, const auto& b) { return optimized::intersect_meshes_2d(a, b); },
      large_10k_a, large_10k_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_large_10Kx10K_v4_hash,
      "v4_hash",
      [](const auto& a, const auto& b) { return v4_hash::intersect_meshes_2d(a, b); },
      large_10k_a, large_10k_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_large_10Kx10K_v5_parallel_merge,
      "v5_parallel_merge",
      [](const auto& a, const auto& b) { return v5_parallel_merge::intersect_meshes_2d(a, b); },
      large_10k_a, large_10k_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_large_10Kx10K_v6_direct_index,
      "v6_direct_index",
      [](const auto& a, const auto& b) { return v6_direct_index::intersect_meshes_2d(a, b); },
      large_10k_a, large_10k_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_large_10Kx10K_v7_soa_optimized,
      "v7_soa_optimized",
      [](const auto& a, const auto& b) { return v7_soa_optimized::intersect_meshes_2d(a, b); },
      large_10k_a, large_10k_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_large_10Kx10K_v8_hybrid,
      "v8_hybrid_cpu_gpu",
      [](const auto& a, const auto& b) { return v8_hybrid_cpu_gpu::intersect_meshes_2d(a, b); },
      large_10k_a, large_10k_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_large_10Kx10K_v9_adaptive,
      "v9_adaptive",
      [](const auto& a, const auto& b) { return v9_adaptive::intersect_meshes_2d(a, b); },
      large_10k_a, large_10k_b)
      ->Unit(benchmark::kMillisecond);
}

// ============================================================================
// Pattern-Based Benchmarks (2D)
// ============================================================================

void register_pattern_benchmarks_2d() {
  // Dense: 1K x 2K (full overlap)
  auto dense_1k_a = mesh_generator::generate_dense_mesh_2d<OptimizedMesh2D>(1000, 0);
  auto dense_1k_b = mesh_generator::generate_dense_mesh_2d<OptimizedMesh2D>(2000, 0);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_dense_1Kx2K_v2_baseline,
      "v2_baseline",
      [](const auto& a, const auto& b) { return optimized::intersect_meshes_2d(a, b); },
      dense_1k_a, dense_1k_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_dense_1Kx2K_v4_hash,
      "v4_hash",
      [](const auto& a, const auto& b) { return v4_hash::intersect_meshes_2d(a, b); },
      dense_1k_a, dense_1k_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_dense_1Kx2K_v5_parallel_merge,
      "v5_parallel_merge",
      [](const auto& a, const auto& b) { return v5_parallel_merge::intersect_meshes_2d(a, b); },
      dense_1k_a, dense_1k_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_dense_1Kx2K_v6_direct_index,
      "v6_direct_index",
      [](const auto& a, const auto& b) { return v6_direct_index::intersect_meshes_2d(a, b); },
      dense_1k_a, dense_1k_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_dense_1Kx2K_v7_soa_optimized,
      "v7_soa_optimized",
      [](const auto& a, const auto& b) { return v7_soa_optimized::intersect_meshes_2d(a, b); },
      dense_1k_a, dense_1k_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_dense_1Kx2K_v8_hybrid,
      "v8_hybrid_cpu_gpu",
      [](const auto& a, const auto& b) { return v8_hybrid_cpu_gpu::intersect_meshes_2d(a, b); },
      dense_1k_a, dense_1k_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_dense_1Kx2K_v9_adaptive,
      "v9_adaptive",
      [](const auto& a, const auto& b) { return v9_adaptive::intersect_meshes_2d(a, b); },
      dense_1k_a, dense_1k_b)
      ->Unit(benchmark::kMicrosecond);

  // Uniform stride: stride=5
  auto stride_5_a = mesh_generator::generate_stride_mesh_2d<OptimizedMesh2D>(200, 5, 0);
  auto stride_5_b = mesh_generator::generate_stride_mesh_2d<OptimizedMesh2D>(400, 5, 0);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_stride5_200x400_v2_baseline,
      "v2_baseline",
      [](const auto& a, const auto& b) { return optimized::intersect_meshes_2d(a, b); },
      stride_5_a, stride_5_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_stride5_200x400_v4_hash,
      "v4_hash",
      [](const auto& a, const auto& b) { return v4_hash::intersect_meshes_2d(a, b); },
      stride_5_a, stride_5_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_stride5_200x400_v5_parallel_merge,
      "v5_parallel_merge",
      [](const auto& a, const auto& b) { return v5_parallel_merge::intersect_meshes_2d(a, b); },
      stride_5_a, stride_5_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_stride5_200x400_v6_direct_index,
      "v6_direct_index",
      [](const auto& a, const auto& b) { return v6_direct_index::intersect_meshes_2d(a, b); },
      stride_5_a, stride_5_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_stride5_200x400_v7_soa_optimized,
      "v7_soa_optimized",
      [](const auto& a, const auto& b) { return v7_soa_optimized::intersect_meshes_2d(a, b); },
      stride_5_a, stride_5_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_stride5_200x400_v8_hybrid,
      "v8_hybrid_cpu_gpu",
      [](const auto& a, const auto& b) { return v8_hybrid_cpu_gpu::intersect_meshes_2d(a, b); },
      stride_5_a, stride_5_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_stride5_200x400_v9_adaptive,
      "v9_adaptive",
      [](const auto& a, const auto& b) { return v9_adaptive::intersect_meshes_2d(a, b); },
      stride_5_a, stride_5_b)
      ->Unit(benchmark::kMicrosecond);

  // Random sparse: 1K x 1K from range 10K (10% density)
  auto sparse_1k_a = mesh_generator::generate_random_mesh_2d<OptimizedMesh2D>(1000, 10000, 42);
  auto sparse_1k_b = mesh_generator::generate_random_mesh_2d<OptimizedMesh2D>(1000, 10000, 43);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_sparse_1Kx1K_v2_baseline,
      "v2_baseline",
      [](const auto& a, const auto& b) { return optimized::intersect_meshes_2d(a, b); },
      sparse_1k_a, sparse_1k_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_sparse_1Kx1K_v4_hash,
      "v4_hash",
      [](const auto& a, const auto& b) { return v4_hash::intersect_meshes_2d(a, b); },
      sparse_1k_a, sparse_1k_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_sparse_1Kx1K_v5_parallel_merge,
      "v5_parallel_merge",
      [](const auto& a, const auto& b) { return v5_parallel_merge::intersect_meshes_2d(a, b); },
      sparse_1k_a, sparse_1k_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_sparse_1Kx1K_v6_direct_index,
      "v6_direct_index",
      [](const auto& a, const auto& b) { return v6_direct_index::intersect_meshes_2d(a, b); },
      sparse_1k_a, sparse_1k_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_sparse_1Kx1K_v7_soa_optimized,
      "v7_soa_optimized",
      [](const auto& a, const auto& b) { return v7_soa_optimized::intersect_meshes_2d(a, b); },
      sparse_1k_a, sparse_1k_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_sparse_1Kx1K_v8_hybrid,
      "v8_hybrid_cpu_gpu",
      [](const auto& a, const auto& b) { return v8_hybrid_cpu_gpu::intersect_meshes_2d(a, b); },
      sparse_1k_a, sparse_1k_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_sparse_1Kx1K_v9_adaptive,
      "v9_adaptive",
      [](const auto& a, const auto& b) { return v9_adaptive::intersect_meshes_2d(a, b); },
      sparse_1k_a, sparse_1k_b)
      ->Unit(benchmark::kMicrosecond);
}

// ============================================================================
// Ratio-Based Benchmarks (2D)
// ============================================================================

void register_ratio_benchmarks_2d() {
  // Balanced: 10K x 10K
  auto balanced_10k_a = mesh_generator::generate_dense_mesh_2d<OptimizedMesh2D>(10000, 0);
  auto balanced_10k_b = mesh_generator::generate_dense_mesh_2d<OptimizedMesh2D>(10000, 0);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_balanced_10Kx10K_v2_baseline,
      "v2_baseline",
      [](const auto& a, const auto& b) { return optimized::intersect_meshes_2d(a, b); },
      balanced_10k_a, balanced_10k_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_balanced_10Kx10K_v4_hash,
      "v4_hash",
      [](const auto& a, const auto& b) { return v4_hash::intersect_meshes_2d(a, b); },
      balanced_10k_a, balanced_10k_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_balanced_10Kx10K_v5_parallel_merge,
      "v5_parallel_merge",
      [](const auto& a, const auto& b) { return v5_parallel_merge::intersect_meshes_2d(a, b); },
      balanced_10k_a, balanced_10k_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_balanced_10Kx10K_v6_direct_index,
      "v6_direct_index",
      [](const auto& a, const auto& b) { return v6_direct_index::intersect_meshes_2d(a, b); },
      balanced_10k_a, balanced_10k_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_balanced_10Kx10K_v7_soa_optimized,
      "v7_soa_optimized",
      [](const auto& a, const auto& b) { return v7_soa_optimized::intersect_meshes_2d(a, b); },
      balanced_10k_a, balanced_10k_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_balanced_10Kx10K_v8_hybrid,
      "v8_hybrid_cpu_gpu",
      [](const auto& a, const auto& b) { return v8_hybrid_cpu_gpu::intersect_meshes_2d(a, b); },
      balanced_10k_a, balanced_10k_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_balanced_10Kx10K_v9_adaptive,
      "v9_adaptive",
      [](const auto& a, const auto& b) { return v9_adaptive::intersect_meshes_2d(a, b); },
      balanced_10k_a, balanced_10k_b)
      ->Unit(benchmark::kMillisecond);

  // Unbalanced: 10K x 1K
  auto unbalanced_10k_a = mesh_generator::generate_dense_mesh_2d<OptimizedMesh2D>(10000, 0);
  auto unbalanced_1k_b = mesh_generator::generate_dense_mesh_2d<OptimizedMesh2D>(1000, 0);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_unbalanced_10Kx1K_v2_baseline,
      "v2_baseline",
      [](const auto& a, const auto& b) { return optimized::intersect_meshes_2d(a, b); },
      unbalanced_10k_a, unbalanced_1k_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_unbalanced_10Kx1K_v4_hash,
      "v4_hash",
      [](const auto& a, const auto& b) { return v4_hash::intersect_meshes_2d(a, b); },
      unbalanced_10k_a, unbalanced_1k_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_unbalanced_10Kx1K_v5_parallel_merge,
      "v5_parallel_merge",
      [](const auto& a, const auto& b) { return v5_parallel_merge::intersect_meshes_2d(a, b); },
      unbalanced_10k_a, unbalanced_1k_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_unbalanced_10Kx1K_v6_direct_index,
      "v6_direct_index",
      [](const auto& a, const auto& b) { return v6_direct_index::intersect_meshes_2d(a, b); },
      unbalanced_10k_a, unbalanced_1k_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_unbalanced_10Kx1K_v7_soa_optimized,
      "v7_soa_optimized",
      [](const auto& a, const auto& b) { return v7_soa_optimized::intersect_meshes_2d(a, b); },
      unbalanced_10k_a, unbalanced_1k_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_unbalanced_10Kx1K_v8_hybrid,
      "v8_hybrid_cpu_gpu",
      [](const auto& a, const auto& b) { return v8_hybrid_cpu_gpu::intersect_meshes_2d(a, b); },
      unbalanced_10k_a, unbalanced_1k_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_unbalanced_10Kx1K_v9_adaptive,
      "v9_adaptive",
      [](const auto& a, const auto& b) { return v9_adaptive::intersect_meshes_2d(a, b); },
      unbalanced_10k_a, unbalanced_1k_b)
      ->Unit(benchmark::kMillisecond);
}

// ============================================================================
// 3D Benchmarks (Small)
// ============================================================================

void register_benchmarks_3d() {
  // Small: 64 x 64 (using 3D grid where grid_size=64)
  auto small_64_a = mesh_generator::generate_dense_mesh_2d<OptimizedMesh3D>(64 * 64, 0);
  auto small_64_b = mesh_generator::generate_dense_mesh_2d<OptimizedMesh3D>(64 * 64, 0);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 3D_small_64x64_v2_baseline,
      "v2_baseline",
      [](const auto& a, const auto& b) { return optimized::intersect_meshes_3d(a, b); },
      small_64_a, small_64_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 3D_small_64x64_v4_hash,
      "v4_hash",
      [](const auto& a, const auto& b) { return v4_hash::intersect_meshes_3d(a, b); },
      small_64_a, small_64_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 3D_small_64x64_v5_parallel_merge,
      "v5_parallel_merge",
      [](const auto& a, const auto& b) { return v5_parallel_merge::intersect_meshes_3d(a, b); },
      small_64_a, small_64_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 3D_small_64x64_v6_direct_index,
      "v6_direct_index",
      [](const auto& a, const auto& b) { return v6_direct_index::intersect_meshes_3d(a, b); },
      small_64_a, small_64_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 3D_small_64x64_v7_soa_optimized,
      "v7_soa_optimized",
      [](const auto& a, const auto& b) { return v7_soa_optimized::intersect_meshes_3d(a, b); },
      small_64_a, small_64_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 3D_small_64x64_v8_hybrid,
      "v8_hybrid_cpu_gpu",
      [](const auto& a, const auto& b) { return v8_hybrid_cpu_gpu::intersect_meshes_3d(a, b); },
      small_64_a, small_64_b)
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 3D_small_64x64_v9_adaptive,
      "v9_adaptive",
      [](const auto& a, const auto& b) { return v9_adaptive::intersect_meshes_3d(a, b); },
      small_64_a, small_64_b)
      ->Unit(benchmark::kMicrosecond);

  // Medium: 512 x 512
  auto medium_512_a = mesh_generator::generate_dense_mesh_2d<OptimizedMesh3D>(512 * 512, 0);
  auto medium_512_b = mesh_generator::generate_dense_mesh_2d<OptimizedMesh3D>(512 * 512, 0);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 3D_medium_512x512_v2_baseline,
      "v2_baseline",
      [](const auto& a, const auto& b) { return optimized::intersect_meshes_3d(a, b); },
      medium_512_a, medium_512_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 3D_medium_512x512_v4_hash,
      "v4_hash",
      [](const auto& a, const auto& b) { return v4_hash::intersect_meshes_3d(a, b); },
      medium_512_a, medium_512_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 3D_medium_512x512_v5_parallel_merge,
      "v5_parallel_merge",
      [](const auto& a, const auto& b) { return v5_parallel_merge::intersect_meshes_3d(a, b); },
      medium_512_a, medium_512_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 3D_medium_512x512_v6_direct_index,
      "v6_direct_index",
      [](const auto& a, const auto& b) { return v6_direct_index::intersect_meshes_3d(a, b); },
      medium_512_a, medium_512_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 3D_medium_512x512_v7_soa_optimized,
      "v7_soa_optimized",
      [](const auto& a, const auto& b) { return v7_soa_optimized::intersect_meshes_3d(a, b); },
      medium_512_a, medium_512_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 3D_medium_512x512_v8_hybrid,
      "v8_hybrid_cpu_gpu",
      [](const auto& a, const auto& b) { return v8_hybrid_cpu_gpu::intersect_meshes_3d(a, b); },
      medium_512_a, medium_512_b)
      ->Unit(benchmark::kMillisecond);

  BENCHMARK_CAPTURE(BM_RowMapping_Generic, 3D_medium_512x512_v9_adaptive,
      "v9_adaptive",
      [](const auto& a, const auto& b) { return v9_adaptive::intersect_meshes_3d(a, b); },
      medium_512_a, medium_512_b)
      ->Unit(benchmark::kMillisecond);
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

  // Register all benchmarks
  register_size_benchmarks_2d();
  register_pattern_benchmarks_2d();
  register_ratio_benchmarks_2d();
  register_benchmarks_3d();

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  Kokkos::finalize();
  return 0;
}
