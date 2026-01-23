// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <benchmark/benchmark.h>
#include <experimental/subsetix/csr/set_algebra/v3.hpp>
#include <experimental/subsetix/csr/set_algebra/successive_intersection.hpp>
#include <set_algebra/test_random_mesh_generator.hpp>
#include <Kokkos_Core.hpp>
#include <vector>
#include <numeric>

// Bring namespaces into scope
using namespace experimental::subsetix::csr;
using namespace experimental::subsetix::csr::v3;
using namespace experimental::subsetix::csr::test;
using namespace experimental::subsetix::csr::successive;

// Type aliases for convenience
using Coord = int32_t;
using IntervalType = experimental::subsetix::csr::Interval<Coord>;

// ============================================================================
// Conversion helpers
// ============================================================================

namespace benchmark_helpers {

inline v3::Mesh2DDevice from_common_2d_v3(const DefaultCommonMesh2D& mesh) {
  return MeshConverter2D<v3::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(mesh);
}

} // namespace benchmark_helpers

// ============================================================================
// Benchmark Fixture for Successive Intersection
// ============================================================================

/**
 * @brief Benchmark fixture for successive intersection
 *
 * Tests intersection of N meshes in sequence:
 * result = (((M1 ∩ M2) ∩ M3) ∩ ... ∩ Mn)
 *
 * This pattern is common in AMR where coarse grids are successively
 * refined by intersection with level set constraints.
 */
template <typename GetConfigFunc, int ChainLength>
class SuccessiveIntersectionBenchmark2D : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State&) override {
    auto cfg = GetConfigFunc()();

    // Generate chain of meshes
    meshes_.reserve(ChainLength);
    for (int i = 0; i < ChainLength; ++i) {
      cfg.seed = 42 + i;  // Different seed for each mesh
      auto common = RandomMeshGenerator::generate_2d(cfg);
      meshes_.push_back(benchmark_helpers::from_common_2d_v3(common));
    }

    // Calculate total intervals for metrics
    total_intervals_ = 0;
    for (const auto& mesh : meshes_) {
      total_intervals_ += mesh.num_intervals;
    }
  }

  void TearDown(const benchmark::State&) override {
    meshes_.clear();
  }

protected:
  std::vector<v3::Mesh2DDevice> meshes_;
  std::size_t total_intervals_;
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
// Naive Approach Benchmarks
// ============================================================================

// 2 Meshes - Small
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Naive_2Meshes_Small, GetSmallConfig, 2)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = naive::intersect(meshes_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Naive_2Meshes_Small)->MinTime(3.0);

// 2 Meshes - Medium
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Naive_2Meshes_Medium, GetMediumConfig, 2)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = naive::intersect(meshes_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Naive_2Meshes_Medium)->MinTime(3.0);

// 2 Meshes - Large
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Naive_2Meshes_Large, GetLargeConfig, 2)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = naive::intersect(meshes_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Naive_2Meshes_Large)->MinTime(3.0);

// 4 Meshes - Small
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Naive_4Meshes_Small, GetSmallConfig, 4)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = naive::intersect(meshes_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Naive_4Meshes_Small)->MinTime(3.0);

// 4 Meshes - Medium
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Naive_4Meshes_Medium, GetMediumConfig, 4)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = naive::intersect(meshes_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Naive_4Meshes_Medium)->MinTime(3.0);

// 4 Meshes - Large
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Naive_4Meshes_Large, GetLargeConfig, 4)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = naive::intersect(meshes_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Naive_4Meshes_Large)->MinTime(3.0);

// 8 Meshes - Small
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Naive_8Meshes_Small, GetSmallConfig, 8)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = naive::intersect(meshes_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Naive_8Meshes_Small)->MinTime(3.0);

// 8 Meshes - Medium
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Naive_8Meshes_Medium, GetMediumConfig, 8)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = naive::intersect(meshes_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Naive_8Meshes_Medium)->MinTime(3.0);

// 8 Meshes - Large
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Naive_8Meshes_Large, GetLargeConfig, 8)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = naive::intersect(meshes_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Naive_8Meshes_Large)->MinTime(3.0);

// ============================================================================
// Workspace Approach Benchmarks
// ============================================================================

// 2 Meshes - Small
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Workspace_2Meshes_Small, GetSmallConfig, 2)
(benchmark::State& state) {
  workspace::IntersectionWorkspace<2, Kokkos::DefaultExecutionSpace::memory_space> ws;
  for (auto _ : state) {
    auto result = workspace::intersect_successive<2>(meshes_, ws);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Workspace_2Meshes_Small)->MinTime(3.0);

// 2 Meshes - Medium
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Workspace_2Meshes_Medium, GetMediumConfig, 2)
(benchmark::State& state) {
  workspace::IntersectionWorkspace<2, Kokkos::DefaultExecutionSpace::memory_space> ws;
  for (auto _ : state) {
    auto result = workspace::intersect_successive<2>(meshes_, ws);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Workspace_2Meshes_Medium)->MinTime(3.0);

// 2 Meshes - Large
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Workspace_2Meshes_Large, GetLargeConfig, 2)
(benchmark::State& state) {
  workspace::IntersectionWorkspace<2, Kokkos::DefaultExecutionSpace::memory_space> ws;
  for (auto _ : state) {
    auto result = workspace::intersect_successive<2>(meshes_, ws);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Workspace_2Meshes_Large)->MinTime(3.0);

// 4 Meshes - Small
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Workspace_4Meshes_Small, GetSmallConfig, 4)
(benchmark::State& state) {
  workspace::IntersectionWorkspace<2, Kokkos::DefaultExecutionSpace::memory_space> ws;
  for (auto _ : state) {
    auto result = workspace::intersect_successive<2>(meshes_, ws);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Workspace_4Meshes_Small)->MinTime(3.0);

// 4 Meshes - Medium
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Workspace_4Meshes_Medium, GetMediumConfig, 4)
(benchmark::State& state) {
  workspace::IntersectionWorkspace<2, Kokkos::DefaultExecutionSpace::memory_space> ws;
  for (auto _ : state) {
    auto result = workspace::intersect_successive<2>(meshes_, ws);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Workspace_4Meshes_Medium)->MinTime(3.0);

// 4 Meshes - Large
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Workspace_4Meshes_Large, GetLargeConfig, 4)
(benchmark::State& state) {
  workspace::IntersectionWorkspace<2, Kokkos::DefaultExecutionSpace::memory_space> ws;
  for (auto _ : state) {
    auto result = workspace::intersect_successive<2>(meshes_, ws);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Workspace_4Meshes_Large)->MinTime(3.0);

// 8 Meshes - Small
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Workspace_8Meshes_Small, GetSmallConfig, 8)
(benchmark::State& state) {
  workspace::IntersectionWorkspace<2, Kokkos::DefaultExecutionSpace::memory_space> ws;
  for (auto _ : state) {
    auto result = workspace::intersect_successive<2>(meshes_, ws);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Workspace_8Meshes_Small)->MinTime(3.0);

// 8 Meshes - Medium
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Workspace_8Meshes_Medium, GetMediumConfig, 8)
(benchmark::State& state) {
  workspace::IntersectionWorkspace<2, Kokkos::DefaultExecutionSpace::memory_space> ws;
  for (auto _ : state) {
    auto result = workspace::intersect_successive<2>(meshes_, ws);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Workspace_8Meshes_Medium)->MinTime(3.0);

// 8 Meshes - Large
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Workspace_8Meshes_Large, GetLargeConfig, 8)
(benchmark::State& state) {
  workspace::IntersectionWorkspace<2, Kokkos::DefaultExecutionSpace::memory_space> ws;
  for (auto _ : state) {
    auto result = workspace::intersect_successive<2>(meshes_, ws);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Workspace_8Meshes_Large)->MinTime(3.0);

// ============================================================================
// Graph DAG Approach Benchmarks
// ============================================================================

// 2 Meshes - Small
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Graph_2Meshes_Small, GetSmallConfig, 2)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = graph::successive_intersection<2>(meshes_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Graph_2Meshes_Small)->MinTime(3.0);

// 2 Meshes - Medium
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Graph_2Meshes_Medium, GetMediumConfig, 2)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = graph::successive_intersection<2>(meshes_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Graph_2Meshes_Medium)->MinTime(3.0);

// 2 Meshes - Large
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Graph_2Meshes_Large, GetLargeConfig, 2)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = graph::successive_intersection<2>(meshes_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Graph_2Meshes_Large)->MinTime(3.0);

// 4 Meshes - Small
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Graph_4Meshes_Small, GetSmallConfig, 4)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = graph::successive_intersection<2>(meshes_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Graph_4Meshes_Small)->MinTime(3.0);

// 4 Meshes - Medium
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Graph_4Meshes_Medium, GetMediumConfig, 4)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = graph::successive_intersection<2>(meshes_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Graph_4Meshes_Medium)->MinTime(3.0);

// 4 Meshes - Large
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Graph_4Meshes_Large, GetLargeConfig, 4)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = graph::successive_intersection<2>(meshes_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Graph_4Meshes_Large)->MinTime(3.0);

// 8 Meshes - Small
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Graph_8Meshes_Small, GetSmallConfig, 8)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = graph::successive_intersection<2>(meshes_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Graph_8Meshes_Small)->MinTime(3.0);

// 8 Meshes - Medium
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Graph_8Meshes_Medium, GetMediumConfig, 8)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = graph::successive_intersection<2>(meshes_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Graph_8Meshes_Medium)->MinTime(3.0);

// 8 Meshes - Large
BENCHMARK_TEMPLATE_F(SuccessiveIntersectionBenchmark2D, Graph_8Meshes_Large, GetLargeConfig, 8)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = graph::successive_intersection<2>(meshes_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();
  }
  state.SetItemsProcessed(state.iterations() * total_intervals_);
  state.SetBytesProcessed(state.iterations() * total_intervals_ * sizeof(IntervalType));
}
BENCHMARK_REGISTER_F(SuccessiveIntersectionBenchmark2D, Graph_8Meshes_Large)->MinTime(3.0);

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
