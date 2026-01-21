/**
 * @file fvd_benchmark.cpp
 * @brief Phase 6: FVD Layer Performance Benchmarks
 *
 * Benchmarks for the FVD layer measuring time step performance.
 * Note: AMR/refinement is disabled to avoid code path issues.
 */

#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>

#include <subsetix/fvd/solver/adaptive_solver.hpp>
#include <subsetix/fvd/system/euler2d.hpp>
#include <subsetix/fvd/system/advection2d.hpp>
#include <subsetix/fvd/reconstruction/reconstruction.hpp>
#include <subsetix/fvd/flux/flux_schemes.hpp>
#include <subsetix/fvd/geometry/geometry_builder.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>

using namespace subsetix::fvd;
using namespace subsetix::csr;

// ============================================================================
// EULER2D BENCHMARKS
// ============================================================================

static void BM_Euler2D_Step_32x32(benchmark::State& state) {
    using Real = float;
    using System = Euler2D<Real>;
    using Solver = AdaptiveSolver<
        System,
        reconstruction::NoReconstruction,
        flux::RusanovFlux,
        time::ForwardEuler<Real>
    >;

    Box2D domain{0, 32, 0, 32};
    Geometry2D<Real> geom = Geometry2D<Real>::build_box(32, 32, Real(1), Real(1));
    IntervalSet2DDevice fluid = geom.build();

    typename Solver::Config config;
    config.gamma = Real(1.4);
    config.cfl = Real(0.4);
    config.refine_fraction = Real(0);  // Disable AMR
    config.remesh_stride = 0;  // Disable AMR

    Solver solver(fluid, domain, config);
    typename System::Primitive initial{1.0f, 0.5f, 0.0f, 1.0f};
    solver.initialize(initial);

    for (auto _ : state) {
        solver.step();
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(
        state.iterations() * 32 * 32 * sizeof(typename System::Conserved)
    );
}
BENCHMARK(BM_Euler2D_Step_32x32)->Unit(benchmark::kMicrosecond);

static void BM_Euler2D_Step_64x64(benchmark::State& state) {
    using Real = float;
    using System = Euler2D<Real>;
    using Solver = AdaptiveSolver<
        System,
        reconstruction::NoReconstruction,
        flux::RusanovFlux,
        time::ForwardEuler<Real>
    >;

    Box2D domain{0, 64, 0, 64};
    Geometry2D<Real> geom = Geometry2D<Real>::build_box(64, 64, Real(1), Real(1));
    IntervalSet2DDevice fluid = geom.build();

    typename Solver::Config config;
    config.gamma = Real(1.4);
    config.cfl = Real(0.4);
    config.refine_fraction = Real(0);  // Disable AMR
    config.remesh_stride = 0;  // Disable AMR

    Solver solver(fluid, domain, config);
    typename System::Primitive initial{1.0f, 0.5f, 0.0f, 1.0f};
    solver.initialize(initial);

    for (auto _ : state) {
        solver.step();
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(
        state.iterations() * 64 * 64 * sizeof(typename System::Conserved)
    );
}
BENCHMARK(BM_Euler2D_Step_64x64)->Unit(benchmark::kMicrosecond);

static void BM_Euler2D_Step_128x128(benchmark::State& state) {
    using Real = float;
    using System = Euler2D<Real>;
    using Solver = AdaptiveSolver<
        System,
        reconstruction::NoReconstruction,
        flux::RusanovFlux,
        time::ForwardEuler<Real>
    >;

    Box2D domain{0, 128, 0, 128};
    Geometry2D<Real> geom = Geometry2D<Real>::build_box(128, 128, Real(1), Real(1));
    IntervalSet2DDevice fluid = geom.build();

    typename Solver::Config config;
    config.gamma = Real(1.4);
    config.cfl = Real(0.4);
    config.refine_fraction = Real(0);  // Disable AMR
    config.remesh_stride = 0;  // Disable AMR

    Solver solver(fluid, domain, config);
    typename System::Primitive initial{1.0f, 0.5f, 0.0f, 1.0f};
    solver.initialize(initial);

    for (auto _ : state) {
        solver.step();
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(
        state.iterations() * 128 * 128 * sizeof(typename System::Conserved)
    );
}
BENCHMARK(BM_Euler2D_Step_128x128)->Unit(benchmark::kMicrosecond);

// ============================================================================
// ADVECTION2D BENCHMARKS
// ============================================================================

static void BM_Advection2D_Step_32x32(benchmark::State& state) {
    using Real = float;
    using System = Advection2D<Real>;
    using Solver = AdaptiveSolver<
        System,
        reconstruction::NoReconstruction,
        flux::RusanovFlux,
        time::ForwardEuler<Real>
    >;

    Box2D domain{0, 32, 0, 32};
    Geometry2D<Real> geom = Geometry2D<Real>::build_box(32, 32, Real(1), Real(1));
    IntervalSet2DDevice fluid = geom.build();

    typename Solver::Config config;
    config.gamma = Real(1.4);
    config.cfl = Real(0.4);
    config.refine_fraction = Real(0);  // Disable AMR
    config.remesh_stride = 0;  // Disable AMR

    System sys_instance(Real(1), Real(0));
    Solver solver(fluid, domain, config, sys_instance);
    typename System::Primitive initial{1.0f};
    solver.initialize(initial);

    for (auto _ : state) {
        solver.step();
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(
        state.iterations() * 32 * 32 * sizeof(typename System::Conserved)
    );
}
BENCHMARK(BM_Advection2D_Step_32x32)->Unit(benchmark::kMicrosecond);

static void BM_Advection2D_Step_64x64(benchmark::State& state) {
    using Real = float;
    using System = Advection2D<Real>;
    using Solver = AdaptiveSolver<
        System,
        reconstruction::NoReconstruction,
        flux::RusanovFlux,
        time::ForwardEuler<Real>
    >;

    Box2D domain{0, 64, 0, 64};
    Geometry2D<Real> geom = Geometry2D<Real>::build_box(64, 64, Real(1), Real(1));
    IntervalSet2DDevice fluid = geom.build();

    typename Solver::Config config;
    config.gamma = Real(1.4);
    config.cfl = Real(0.4);
    config.refine_fraction = Real(0);  // Disable AMR
    config.remesh_stride = 0;  // Disable AMR

    System sys_instance(Real(1), Real(0));
    Solver solver(fluid, domain, config, sys_instance);
    typename System::Primitive initial{1.0f};
    solver.initialize(initial);

    for (auto _ : state) {
        solver.step();
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(
        state.iterations() * 64 * 64 * sizeof(typename System::Conserved)
    );
}
BENCHMARK(BM_Advection2D_Step_64x64)->Unit(benchmark::kMicrosecond);

static void BM_Advection2D_Step_128x128(benchmark::State& state) {
    using Real = float;
    using System = Advection2D<Real>;
    using Solver = AdaptiveSolver<
        System,
        reconstruction::NoReconstruction,
        flux::RusanovFlux,
        time::ForwardEuler<Real>
    >;

    Box2D domain{0, 128, 0, 128};
    Geometry2D<Real> geom = Geometry2D<Real>::build_box(128, 128, Real(1), Real(1));
    IntervalSet2DDevice fluid = geom.build();

    typename Solver::Config config;
    config.gamma = Real(1.4);
    config.cfl = Real(0.4);
    config.refine_fraction = Real(0);  // Disable AMR
    config.remesh_stride = 0;  // Disable AMR

    System sys_instance(Real(1), Real(0));
    Solver solver(fluid, domain, config, sys_instance);
    typename System::Primitive initial{1.0f};
    solver.initialize(initial);

    for (auto _ : state) {
        solver.step();
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(
        state.iterations() * 128 * 128 * sizeof(typename System::Conserved)
    );
}
BENCHMARK(BM_Advection2D_Step_128x128)->Unit(benchmark::kMicrosecond);

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    Kokkos::finalize();
    return 0;
}
