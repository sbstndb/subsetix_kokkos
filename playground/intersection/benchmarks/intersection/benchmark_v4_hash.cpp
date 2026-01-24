// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#include <benchmark/benchmark.h>
#include <playground/subsetix/csr/intersection/algorithm/v4_hash.hpp>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <type_traits>

using namespace playground::subsetix::csr::intersection;

using V4Mesh2D = hash_based::Mesh2D<>;
using V4Mesh3D = hash_based::Mesh3D<>;

using Coord = int32_t;
using IntervalType = playground::subsetix::csr::intersection::Interval<Coord>;

namespace mesh_generator {

template <typename MeshType>
MeshType generate_dense_mesh(std::size_t num_rows, int32_t y_start = 0) {
  using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
  using RowKey = typename MeshType::RowKey;

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
    if constexpr (std::is_same_v<RowKey, RowKey2D<Coord>>) {
      host_keys(i).y = y_start + static_cast<int32_t>(i);
    } else {
      int32_t z = (i / 512);
      int32_t y = (i % 512) + y_start;
      host_keys(i).y = y;
      host_keys(i).z = z;
    }
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

}

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
    
    Kokkos::fence();
    total_intervals += result.num_intervals;
    total_rows_out += result.num_rows;
  }

  std::size_t bytes_a = mesh_a.num_rows * sizeof(typename MeshType::RowKey) +
                       mesh_a.num_intervals * sizeof(IntervalType);
  std::size_t bytes_b = mesh_b.num_rows * sizeof(typename MeshType::RowKey) +
                       mesh_b.num_intervals * sizeof(IntervalType);
  std::size_t total_bytes = bytes_a + bytes_b;

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

void register_large_benchmarks_2d() {
  auto large_10k_a = mesh_generator::generate_dense_mesh<V4Mesh2D>(10000, 0);
  auto large_10k_b = mesh_generator::generate_dense_mesh<V4Mesh2D>(10000, 0);

  benchmark::RegisterBenchmark("2D_LARGE_10Kx10K_v4_hash_balanced",
      [&, mesh_a=std::move(large_10k_a), mesh_b=std::move(large_10k_b)](benchmark::State& state) {
    BM_RowMapping_Generic(state, "v4_hash_balanced",
        [](const auto& a, const auto& b) {
          return hash_based::intersect_meshes_2d(a, b);
        },
        mesh_a, mesh_b);
  })->Unit(benchmark::kMillisecond);

  auto large_10k_unbal_a = mesh_generator::generate_dense_mesh<V4Mesh2D>(10000, 0);
  auto large_1k_unbal_b = mesh_generator::generate_dense_mesh<V4Mesh2D>(1000, 0);

  benchmark::RegisterBenchmark("2D_LARGE_10Kx1K_v4_hash_unbalanced",
      [&, mesh_a=std::move(large_10k_unbal_a), mesh_b=std::move(large_1k_unbal_b)](benchmark::State& state) {
    BM_RowMapping_Generic(state, "v4_hash_unbalanced",
        [](const auto& a, const auto& b) {
          return hash_based::intersect_meshes_2d(a, b);
        },
        mesh_a, mesh_b);
  })->Unit(benchmark::kMillisecond);
}

void register_large_benchmarks_3d() {
  auto large_512_a = mesh_generator::generate_dense_mesh<V4Mesh3D>(512 * 512, 0);
  auto large_512_b = mesh_generator::generate_dense_mesh<V4Mesh3D>(512 * 512, 0);

  benchmark::RegisterBenchmark("3D_LARGE_512x512_v4_hash_balanced",
      [&, mesh_a=std::move(large_512_a), mesh_b=std::move(large_512_b)](benchmark::State& state) {
    BM_RowMapping_Generic(state, "v4_hash_3d_balanced",
        [](const auto& a, const auto& b) {
          return hash_based::intersect_meshes_3d(a, b);
        },
        mesh_a, mesh_b);
  })->Unit(benchmark::kMillisecond);

  auto large_1024_unbal_a = mesh_generator::generate_dense_mesh<V4Mesh3D>(1024 * 512, 0);
  auto large_512_unbal_b = mesh_generator::generate_dense_mesh<V4Mesh3D>(512 * 512, 0);

  benchmark::RegisterBenchmark("3D_LARGE_1024x512_v4_hash_unbalanced",
      [&, mesh_a=std::move(large_1024_unbal_a), mesh_b=std::move(large_512_unbal_b)](benchmark::State& state) {
    BM_RowMapping_Generic(state, "v4_hash_3d_unbalanced",
        [](const auto& a, const auto& b) {
          return hash_based::intersect_meshes_3d(a, b);
        },
        mesh_a, mesh_b);
  })->Unit(benchmark::kMillisecond);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  benchmark::Initialize(&argc, argv);

  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    Kokkos::finalize();
    return 1;
  }

  register_large_benchmarks_2d();
  register_large_benchmarks_3d();

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  Kokkos::finalize();
  return 0;
}
