// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#ifdef SUBSETIX_ENABLE_PLAYGROUND

#include <benchmark/benchmark.h>

#include <cstdint>
#include <cstdlib>
#include <random>

#include <Kokkos_Core.hpp>

#include <playground/subsetix/field/csr_field.hpp>
#include <playground/subsetix/field/csr_stencil.hpp>

using namespace playground::subsetix::csr;

namespace {

using Coord = int32_t;

inline void set_stencil_counters(benchmark::State& state,
                                 std::size_t cells_per_iter,
                                 std::size_t bytes_per_cell,
                                 std::size_t flops_per_cell) {
  state.SetItemsProcessed(state.iterations() * cells_per_iter);
  state.SetBytesProcessed(state.iterations() * cells_per_iter * bytes_per_cell);

  state.counters["cells/s"] = benchmark::Counter(
      static_cast<double>(cells_per_iter),
      benchmark::Counter::kIsIterationInvariantRate);

  state.counters["ns/cell"] = benchmark::Counter(
      static_cast<double>(cells_per_iter),
      benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);

  state.counters["GFLOP/s"] = benchmark::Counter(
      (static_cast<double>(cells_per_iter) * static_cast<double>(flops_per_cell)) / 1e9,
      benchmark::Counter::kIsIterationInvariantRate);

  state.counters["FLOP/Byte"] =
      benchmark::Counter(static_cast<double>(flops_per_cell) /
                         static_cast<double>(bytes_per_cell));
}

struct FivePointAverage {
  KOKKOS_INLINE_FUNCTION
  float operator()(Coord x, Coord y, const CsrStencilPoint5<float>& p) const {
    (void)x;
    (void)y;
    return (p.center() + p.east() + p.west() + p.north() + p.south()) / 5.0f;
  }
};

struct SevenPointAverage {
  KOKKOS_INLINE_FUNCTION
  float operator()(Coord x, Coord y, Coord z, const CsrStencilPoint7<float>& p) const {
    (void)x;
    (void)y;
    (void)z;
    return (p.center() + p.east() + p.west() +
            p.north() + p.south() + p.up() + p.down()) /
           7.0f;
  }
};

CsrMeshHost<2, Coord> make_regular_mesh_2d(Coord nx, Coord ny) {
  CsrMeshHost<2, Coord> host;
  for (Coord y = 0; y < ny; ++y) {
    host.append_interval(y, 0, nx);
  }
  return host;
}

CsrMeshHost<2, Coord> make_interior_mask_2d(Coord nx, Coord ny) {
  CsrMeshHost<2, Coord> host;
  for (Coord y = 1; y < ny - 1; ++y) {
    host.append_interval(y, 1, static_cast<Coord>(nx - 1));
  }
  return host;
}

CsrMeshHost<2, Coord> make_fragmented_interior_mask_2d(Coord nx, Coord ny) {
  CsrMeshHost<2, Coord> host;
  const Coord mid = nx / 2;
  for (Coord y = 1; y < ny - 1; ++y) {
    host.append_interval(y, 1, mid);
    host.append_interval(y, static_cast<Coord>(mid + 1), static_cast<Coord>(nx - 1));
  }
  return host;
}

struct MaskWithCells2D {
  CsrMeshHost<2, Coord> mask;
  std::size_t cell_count = 0;
};

MaskWithCells2D make_random_interior_mask_2d(Coord nx,
                                            Coord ny,
                                            double density,
                                            std::uint32_t seed) {
  MaskWithCells2D out;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (Coord y = 1; y < ny - 1; ++y) {
    bool active_run = false;
    Coord run_begin = 0;

    for (Coord x = 1; x < nx - 1; ++x) {
      const bool active = dist(rng) < density;
      if (active && !active_run) {
        active_run = true;
        run_begin = x;
      } else if (!active && active_run) {
        active_run = false;
        out.mask.append_interval(y, run_begin, x);
        out.cell_count += static_cast<std::size_t>(x - run_begin);
      }
    }

    if (active_run) {
      out.mask.append_interval(y, run_begin, static_cast<Coord>(nx - 1));
      out.cell_count += static_cast<std::size_t>((nx - 1) - run_begin);
    }
  }

  return out;
}

CsrMeshHost<3, Coord> make_regular_mesh_3d(Coord n) {
  CsrMeshHost<3, Coord> host;
  for (Coord y = 0; y < n; ++y) {
    for (Coord z = 0; z < n; ++z) {
      host.append_interval(y, z, 0, n);
    }
  }
  return host;
}

CsrMeshHost<3, Coord> make_interior_mask_3d(Coord n) {
  CsrMeshHost<3, Coord> host;
  for (Coord y = 1; y < n - 1; ++y) {
    for (Coord z = 1; z < n - 1; ++z) {
      host.append_interval(y, z, 1, static_cast<Coord>(n - 1));
    }
  }
  return host;
}

struct MaskWithCells3D {
  CsrMeshHost<3, Coord> mask;
  std::size_t cell_count = 0;
};

MaskWithCells3D make_random_interior_mask_3d(Coord n,
                                            double density,
                                            std::uint32_t seed) {
  MaskWithCells3D out;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (Coord y = 1; y < n - 1; ++y) {
    for (Coord z = 1; z < n - 1; ++z) {
      bool active_run = false;
      Coord run_begin = 0;

      for (Coord x = 1; x < n - 1; ++x) {
        const bool active = dist(rng) < density;
        if (active && !active_run) {
          active_run = true;
          run_begin = x;
        } else if (!active && active_run) {
          active_run = false;
          out.mask.append_interval(y, z, run_begin, x);
          out.cell_count += static_cast<std::size_t>(x - run_begin);
        }
      }

      if (active_run) {
        out.mask.append_interval(y, z, run_begin, static_cast<Coord>(n - 1));
        out.cell_count += static_cast<std::size_t>((n - 1) - run_begin);
      }
    }
  }

  return out;
}

static void BM_Stencil5Pt_Uniform2D(benchmark::State& state) {
  const Coord n = static_cast<Coord>(state.range(0));
  const auto mesh_host = make_regular_mesh_2d(n, n);
  const auto mask_host = make_interior_mask_2d(n, n);

  const auto mesh_dev = build_device_mesh_from_host(mesh_host, "mesh2d");
  const auto mask_dev = build_device_mesh_from_host(mask_host, "mask2d");

  auto in = make_field_from_mesh<2, float>(mesh_dev, 1.0f, "in2d");
  auto out = make_field_from_mesh<2, float>(mesh_dev, 0.0f, "out2d");

  const auto mapping = detail::build_mask_field_mapping<2>(in.mesh, mask_dev);
  const auto neighbours = build_stencil_neighbour_mapping_2d(in.mesh, mask_dev, mapping);

  const std::size_t cells_per_iter = static_cast<std::size_t>(mask_dev.num_intervals) *
                                     static_cast<std::size_t>(n - 2);
  const std::size_t bytes_per_cell = (5 + 1) * sizeof(float);
  const std::size_t flops_per_cell = 5;

  for (auto _ : state) {
    apply_csr_stencil_5pt_on_mask_device(out, in, mask_dev, mapping, neighbours,
                                         FivePointAverage{});
    benchmark::DoNotOptimize(out.values.data());
  }

  set_stencil_counters(state, cells_per_iter, bytes_per_cell, flops_per_cell);
}

static void BM_Stencil5Pt_Subfield2D(benchmark::State& state) {
  const Coord n = static_cast<Coord>(state.range(0));
  const auto mesh_host = make_regular_mesh_2d(n, n);
  const auto mask_host = make_fragmented_interior_mask_2d(n, n);

  const auto mesh_dev = build_device_mesh_from_host(mesh_host, "mesh2d");
  const auto mask_dev = build_device_mesh_from_host(mask_host, "mask2d");

  auto in = make_field_from_mesh<2, float>(mesh_dev, 1.0f, "in2d");
  auto out = make_field_from_mesh<2, float>(mesh_dev, 0.0f, "out2d");

  const auto mapping = detail::build_mask_field_mapping<2>(in.mesh, mask_dev);
  const auto neighbours = build_stencil_neighbour_mapping_2d(in.mesh, mask_dev, mapping);

  const std::size_t cells_per_iter =
      static_cast<std::size_t>(n - 2) * static_cast<std::size_t>(n - 3);
  const std::size_t bytes_per_cell = (5 + 1) * sizeof(float);
  const std::size_t flops_per_cell = 5;

  for (auto _ : state) {
    apply_csr_stencil_5pt_on_mask_device(out, in, mask_dev, mapping, neighbours,
                                         FivePointAverage{});
    benchmark::DoNotOptimize(out.values.data());
  }

  set_stencil_counters(state, cells_per_iter, bytes_per_cell, flops_per_cell);
}

static void BM_Stencil5Pt_RandomMask2D(benchmark::State& state) {
  const Coord n = static_cast<Coord>(state.range(0));
  const int density_percent = static_cast<int>(state.range(1));
  const double density = static_cast<double>(density_percent) / 100.0;

  const auto mesh_host = make_regular_mesh_2d(n, n);
  const auto random_mask = make_random_interior_mask_2d(
      n, n, density, /*seed=*/static_cast<std::uint32_t>(1234 + density_percent));

  const auto mesh_dev = build_device_mesh_from_host(mesh_host, "mesh2d");
  const auto mask_dev = build_device_mesh_from_host(random_mask.mask, "mask2d");

  auto in = make_field_from_mesh<2, float>(mesh_dev, 1.0f, "in2d");
  auto out = make_field_from_mesh<2, float>(mesh_dev, 0.0f, "out2d");

  const auto mapping = detail::build_mask_field_mapping<2>(in.mesh, mask_dev);
  const auto neighbours = build_stencil_neighbour_mapping_2d(in.mesh, mask_dev, mapping);

  const std::size_t cells_per_iter = random_mask.cell_count;
  const std::size_t bytes_per_cell = (5 + 1) * sizeof(float);
  const std::size_t flops_per_cell = 5;

  for (auto _ : state) {
    apply_csr_stencil_5pt_on_mask_device(out, in, mask_dev, mapping, neighbours,
                                         FivePointAverage{});
    benchmark::DoNotOptimize(out.values.data());
  }

  set_stencil_counters(state, cells_per_iter, bytes_per_cell, flops_per_cell);
}

static void BM_Stencil7Pt_Uniform3D(benchmark::State& state) {
  const Coord n = static_cast<Coord>(state.range(0));
  const auto mesh_host = make_regular_mesh_3d(n);
  const auto mask_host = make_interior_mask_3d(n);

  const auto mesh_dev = build_device_mesh_from_host(mesh_host, "mesh3d");
  const auto mask_dev = build_device_mesh_from_host(mask_host, "mask3d");

  auto in = make_field_from_mesh<3, float>(mesh_dev, 1.0f, "in3d");
  auto out = make_field_from_mesh<3, float>(mesh_dev, 0.0f, "out3d");

  const auto mapping = detail::build_mask_field_mapping<3>(in.mesh, mask_dev);
  const auto neighbours = build_stencil_neighbour_mapping_3d(in.mesh, mask_dev, mapping);

  const std::size_t cells_per_iter =
      static_cast<std::size_t>(n - 2) * static_cast<std::size_t>(n - 2) *
      static_cast<std::size_t>(n - 2);
  const std::size_t bytes_per_cell = (7 + 1) * sizeof(float);
  const std::size_t flops_per_cell = 7;

  for (auto _ : state) {
    apply_csr_stencil_7pt_on_mask_device(out, in, mask_dev, mapping, neighbours,
                                         SevenPointAverage{});
    benchmark::DoNotOptimize(out.values.data());
  }

  set_stencil_counters(state, cells_per_iter, bytes_per_cell, flops_per_cell);
}

static void BM_Stencil7Pt_RandomMask3D(benchmark::State& state) {
  const Coord n = static_cast<Coord>(state.range(0));
  const int density_percent = static_cast<int>(state.range(1));
  const double density = static_cast<double>(density_percent) / 100.0;

  const auto mesh_host = make_regular_mesh_3d(n);
  const auto random_mask = make_random_interior_mask_3d(
      n, density, /*seed=*/static_cast<std::uint32_t>(5678 + density_percent));

  const auto mesh_dev = build_device_mesh_from_host(mesh_host, "mesh3d");
  const auto mask_dev = build_device_mesh_from_host(random_mask.mask, "mask3d");

  auto in = make_field_from_mesh<3, float>(mesh_dev, 1.0f, "in3d");
  auto out = make_field_from_mesh<3, float>(mesh_dev, 0.0f, "out3d");

  const auto mapping = detail::build_mask_field_mapping<3>(in.mesh, mask_dev);
  const auto neighbours = build_stencil_neighbour_mapping_3d(in.mesh, mask_dev, mapping);

  const std::size_t cells_per_iter = random_mask.cell_count;
  const std::size_t bytes_per_cell = (7 + 1) * sizeof(float);
  const std::size_t flops_per_cell = 7;

  for (auto _ : state) {
    apply_csr_stencil_7pt_on_mask_device(out, in, mask_dev, mapping, neighbours,
                                         SevenPointAverage{});
    benchmark::DoNotOptimize(out.values.data());
  }

  set_stencil_counters(state, cells_per_iter, bytes_per_cell, flops_per_cell);
}

static void BM_DenseStencil5Pt_Uniform2D(benchmark::State& state) {
  const int n = static_cast<int>(state.range(0));

  using View2D =
      Kokkos::View<float**, Kokkos::LayoutRight, DeviceMemorySpace>;
  View2D in("dense_in_2d", n, n);
  View2D out("dense_out_2d", n, n);
  Kokkos::deep_copy(in, 1.0f);
  Kokkos::deep_copy(out, 0.0f);

  using Policy = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>;
  const Policy policy({1, 1}, {n - 1, n - 1}, {8, 32});

  const std::size_t cells_per_iter =
      static_cast<std::size_t>(n - 2) * static_cast<std::size_t>(n - 2);
  const std::size_t bytes_per_cell = (5 + 1) * sizeof(float);
  const std::size_t flops_per_cell = 5;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "dense_5pt_stencil_2d",
        policy,
        KOKKOS_LAMBDA(const int y, const int x) {
          out(y, x) =
              (in(y, x) + in(y, x - 1) + in(y, x + 1) +
               in(y - 1, x) + in(y + 1, x)) /
              5.0f;
        });
    ExecSpace().fence();
    benchmark::DoNotOptimize(out.data());
  }

  set_stencil_counters(state, cells_per_iter, bytes_per_cell, flops_per_cell);
}

static void BM_DenseStencil7Pt_Uniform3D(benchmark::State& state) {
  const int n = static_cast<int>(state.range(0));

  using View3D =
      Kokkos::View<float***, Kokkos::LayoutRight, DeviceMemorySpace>;
  View3D in("dense_in_3d", n, n, n);
  View3D out("dense_out_3d", n, n, n);
  Kokkos::deep_copy(in, 1.0f);
  Kokkos::deep_copy(out, 0.0f);

  using Policy = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>;
  const Policy policy({1, 1, 1}, {n - 1, n - 1, n - 1}, {2, 4, 32});

  const std::size_t cells_per_iter =
      static_cast<std::size_t>(n - 2) * static_cast<std::size_t>(n - 2) *
      static_cast<std::size_t>(n - 2);
  const std::size_t bytes_per_cell = (7 + 1) * sizeof(float);
  const std::size_t flops_per_cell = 7;

  for (auto _ : state) {
    Kokkos::parallel_for(
        "dense_7pt_stencil_3d",
        policy,
        KOKKOS_LAMBDA(const int y, const int z, const int x) {
          out(y, z, x) =
              (in(y, z, x) + in(y, z, x - 1) + in(y, z, x + 1) +
               in(y - 1, z, x) + in(y + 1, z, x) +
               in(y, z - 1, x) + in(y, z + 1, x)) /
              7.0f;
        });
    ExecSpace().fence();
    benchmark::DoNotOptimize(out.data());
  }

  set_stencil_counters(state, cells_per_iter, bytes_per_cell, flops_per_cell);
}

}  // namespace

BENCHMARK(BM_Stencil5Pt_Uniform2D)
    ->Arg(32000)
    ->UseRealTime();
BENCHMARK(BM_Stencil5Pt_Subfield2D)
    ->Arg(32000)
    ->UseRealTime();
BENCHMARK(BM_Stencil5Pt_RandomMask2D)
    ->Args({32000, 10})->Args({32000, 25})->Args({32000, 50})
    ->UseRealTime();
BENCHMARK(BM_DenseStencil5Pt_Uniform2D)
    ->Arg(32000)
    ->UseRealTime();
BENCHMARK(BM_Stencil7Pt_Uniform3D)
    ->Arg(256)
    ->UseRealTime();
BENCHMARK(BM_Stencil7Pt_RandomMask3D)
    ->Args({256, 10})->Args({256, 25})->Args({256, 50})
    ->UseRealTime();
BENCHMARK(BM_DenseStencil7Pt_Uniform3D)
    ->Arg(256)
    ->UseRealTime();

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

#endif  // SUBSETIX_ENABLE_PLAYGROUND
