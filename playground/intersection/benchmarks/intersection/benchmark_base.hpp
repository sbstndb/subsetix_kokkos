// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace playground::subsetix::csr::intersection::benchmarks {

// Type aliases for convenience
using Coord = int32_t;
using IntervalType = playground::subsetix::csr::intersection::Interval<Coord>;

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
    
    // Count output size
    Kokkos::fence();
    total_intervals += result.num_intervals;
    total_rows_out += result.num_rows;
  }

  // Calculate statistics
  std::size_t bytes_a = mesh_a.num_rows * sizeof(typename MeshType::RowKey) +
                       mesh_a.num_intervals * sizeof(IntervalType);
  std::size_t bytes_b = mesh_b.num_rows * sizeof(typename MeshType::RowKey) +
                       mesh_b.num_intervals * sizeof(IntervalType);
  std::size_t total_bytes = bytes_a + bytes_b;

  // Set counters
  state.counters["Intervals"] = benchmark::Counter(
      total_intervals, benchmark::Counter::kAvgIterations);

  state.counters["Rows"] = benchmark::Counter(
      total_rows_out, benchmark::Counter::kAvgIterations);

  state.counters["Bytes"] = benchmark::Counter(
      total_bytes, benchmark::Counter::kAvgIterations);

  state.counters["BytesPerInterval"] = benchmark::Counter(
      total_bytes / static_cast<double>(total_intervals),
      benchmark::Counter::kAvgIterations);
}

} // namespace playground::subsetix::csr::intersection::benchmarks
