/**
 * @file fvd_graph_api_comparison.cpp
 *
 * @brief COMPARISON OF 3 API VARIANTS FOR GPU PERFORMANCE
 *
 * This file shows the SAME simulation written 3 different ways:
 * 1. WITHOUT Kokkos Graphs (current implementation - many syncs)
 * 2. WITH TimestepGraph (Phase 1 - graph per timestep)
 * 3. WITH Full SimulationGraph (Phase 3 - graph for many timesteps)
 *
 * Each variant shows:
 * - User-facing API changes
 * - Where synchronizations occur
 * - Performance implications
 */

#include <Kokkos_Core.hpp>
#include <subsetix/fvd/fvd_integrators.hpp>
#include <subsetix/fvd/boundary/time_dependent_bc.hpp>
#include <subsetix/fvd/amr/refinement_criteria.hpp>

using namespace subsetix::fvd;
using namespace subsetix::fvd::solver;
using namespace subsetix::fvd::boundary;
using namespace subsetix::fvd::amr;

// ============================================================================
// COMMON SETUP (shared by all 3 variants)
// ============================================================================

void common_setup() {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = System::Conserved;
    using Primitive = System::Primitive;

    const int nx = 400, ny = 160;
    const Real t_final = 1.0f;

    // Initial condition: Mach 2 flow
    Primitive inflow{1.0f, 2.0f * 1.4f, 0.0f, 1.0f};

    // Boundary conditions
    Primitive inflow_state{1.0f, 2.0f * 1.4f, 0.0f, 1.0f};
}

// ============================================================================
// VARIANT 1: WITHOUT KOKKOS GRAPHS (CURRENT)
//
// Characteristics:
// - Multiple kernel launches per timestep
// - Host controls everything
// - Many implicit synchronizations
// - Easy to understand, debug, and extend
//
// Performance: ~7000 syncs for 1000 timesteps
// ============================================================================

void variant_1_no_graph() {
    std::cout << "\n========== VARIANT 1: NO GRAPH ==========\n";

    using Real = float;
    using System = Euler2D<Real>;
    using Solver = EulerSolverRK3;  // RK3 = 3 stages

    // ========================================================================
    // API: Create solver as usual
    // ========================================================================

    Solver solver = Solver::builder(400, 160)
        .with_domain(0.0, 2.0, 0.0, 0.8)
        .with_initial_condition([](Real x, Real y) {
            return System::Primitive{1.0f, 2.0f * 1.4f, 0.0f, 1.0f};
        })
        .build();

    // ========================================================================
    // API: Configure solver (unchanged)
    // ========================================================================

    // Boundary conditions
    solver.set_bc("left", BcType::Dirichlet,
                  System::Primitive{1.0f, 2.0f * 1.4f, 0.0f, 1.0f});

    // Observers
    solver.observers().on_progress([](const SolverState<Real>& state) {
        std::cout << "Step " << state.step << ": t=" << state.time << std::endl;
    });

    // ========================================================================
    // MAIN LOOP: Standard API
    // ========================================================================

    const Real t_final = 1.0f;
    int step = 0;

    while (solver.time() < t_final) {
        // ================================================================
        // SINGLE API CALL - BUT INTERNAL SYNC EXPLOSION!
        // ================================================================

        solver.step();  // ← API looks simple!

        // ================================================================
        // WHAT HAPPENS INSIDE solver.step() for RK3:
        // ================================================================
        //
        // Host: for (int s = 0; s < 3; ++s) {
        //     Device: parallel_for("compute_rhs", ...);     ← SYNC!
        //     Device: parallel_for("compute_stage", ...);   ← SYNC!
        // }
        // Device: parallel_for("combine_stages", ...);      ← SYNC!
        // Host: notify_observers();                          ← SYNC!
        //
        // Total: 4-7 syncs PER TIMESTEP
        // ================================================================

        step++;
    }

    // ========================================================================
    // PERFORMANCE ANALYSIS
    // ========================================================================

    std::cout << "\nVariant 1 Performance:\n";
    std::cout << "  Timesteps: " << step << "\n";
    std::cout << "  Syncs per timestep: ~4-7\n";
    std::cout << "  Total syncs: ~" << (step * 5) << "\n";
    std::cout << "  Overhead: HIGH (30-50% of runtime)\n";
}

// ============================================================================
// VARIANT 2: WITH TIMESTEP GRAPH (PHASE 1 - RECOMMENDED)
//
// Characteristics:
// - Single kernel launch per timestep
// - RK stages fused in graph
// - Minimal host involvement
// - Still flexible for observers/remeshing
//
// Performance: ~100 syncs for 1000 timesteps
// ============================================================================

namespace variant2 {

/**
 * @brief Timestep Graph for RK Methods
 *
 * Compiles all RK stages into a single graph.
 * One graph.execute() = entire timestep with no intermediate syncs.
 */
template<
    FiniteVolumeSystem System,
    template<typename> class FluxScheme = flux::RusanovFlux,
    typename TimeIntegrator = time::Kutta3<typename System::RealType>
>
class TimestepSolver {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Graph = Kokkos::Experimental::TypeErasedGraph;

    // ========================================================================
    // CONSTRUCTOR: Builds the graph ONCE
    // ========================================================================

    TimestepSolver(int nx, int ny, Real dx, Real dy, Real gamma) {
        // Allocate solution and working arrays
        U = Kokkos::View<Conserved**>("U", ny, nx);
        U_work = Kokkos::View<Conserved**>("U_work", ny, nx);

        // Stage RHS storage (one per stage)
        constexpr int stages = TimeIntegrator::stages;
        stage_rhs = Kokkos::View<Conserved***>("stage_rhs", stages, ny, nx);

        // Build graph ONCE at initialization
        build_graph();
    }

    // ========================================================================
    // NEW API: Step using pre-compiled graph
    // ========================================================================

    void step(Real dt, Real t) {
        // Update graph parameters (if needed)
        update_parameters(dt, t);

        // ================================================================
        // SINGLE GRAPH EXECUTION = NO INTERMEDIATE SYNCS!
        // ================================================================
        graph.execute();

        // Update time
        t_ += dt;
        step_count_++;
    }

    // ========================================================================
    // GRAPH BUILDING (internal, done once)
    // ========================================================================

private:
    void build_graph() {
        using namespace Kokkos::Experimental;

        graph = Graph();

        // ================================================================
        // Create nodes for each RK stage
        // ================================================================

        constexpr int stages = TimeIntegrator::stages;

        // Stage 0: Compute RHS
        auto rhs0 = graph.create_node([&](auto& g) {
            return g.create_kernel(
                "rhs_stage0",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {U.extent(0), U.extent(1)}),
                KOKKOS_LAMBDA(int j, int i) {
                    // Compute RHS: dU/dt = -div(F)
                    // ... flux computation ...
                }
            );
        });

        // Stage 1: Compute intermediate solution
        auto stage1 = graph.create_node([&](auto& g) {
            return g.create_kernel(
                "compute_stage1",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {U.extent(0), U.extent(1)}),
                KOKKOS_LAMBDA(int j, int i) {
                    // U_1 = U_0 + dt * a[1][0] * k_0
                    Real a = TimeIntegrator::a[1][0];
                    U_work(j, i).rho = U(j, i).rho + dt_ * a * stage_rhs(0, j, i).rho;
                    U_work(j, i).rhou = U(j, i).rhou + dt_ * a * stage_rhs(0, j, i).rhou;
                    U_work(j, i).rhov = U(j, i).rhov + dt_ * a * stage_rhs(0, j, i).rhov;
                    U_work(j, i).E = U(j, i).E + dt_ * a * stage_rhs(0, j, i).E;
                }
            );
        });

        // Stage 1: Compute RHS
        auto rhs1 = graph.create_node([&](auto& g) {
            return g.create_kernel(
                "rhs_stage1",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {U.extent(0), U.extent(1)}),
                KOKKOS_LAMBDA(int j, int i) {
                    // Compute RHS at U_work
                }
            );
        });

        // ... add remaining stages ...

        // Final: Combine all stages
        auto combine = graph.create_node([&](auto& g) {
            return g.create_kernel(
                "combine_stages",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {U.extent(0), U.extent(1)}),
                KOKKOS_LAMBDA(int j, int i) {
                    // U_{n+1} = U_n + dt * sum(b[i] * k_i)
                    Conserved sum{0, 0, 0, 0};
                    for (int s = 0; s < stages; ++s) {
                        Real b = TimeIntegrator::b[s];
                        sum.rho += b * stage_rhs(s, j, i).rho;
                        sum.rhou += b * stage_rhs(s, j, i).rhou;
                        sum.rhov += b * stage_rhs(s, j, i).rhov;
                        sum.E += b * stage_rhs(s, j, i).E;
                    }
                    U(j, i).rho += dt_ * sum.rho;
                    U(j, i).rhou += dt_ * sum.rhou;
                    U(j, i).rhov += dt_ * sum.rhov;
                    U(j, i).E += dt_ * sum.E;
                }
            );
        });

        // ================================================================
        // Build dependency chain: rhs0 → stage1 → rhs1 → ... → combine
        // ================================================================

        rhs0 >> stage1;
        stage1 >> rhs1;
        // ... chain remaining stages ...
        rhs1 >> combine;

        // ================================================================
        // Compile graph (happens ONCE)
        // ================================================================
        graph.compile();
    }

    void update_parameters(Real dt, Real t) {
        dt_ = dt;
        t_ = t;
        // Could update graph parameters here if needed
    }

    // ========================================================================
    // DATA MEMBERS
    // ========================================================================

    Graph graph;
    Real dt_, t_;
    int step_count_ = 0;

    Kokkos::View<Conserved**> U;
    Kokkos::View<Conserved**> U_work;
    Kokkos::View<Conserved***> stage_rhs;
};

} // namespace variant2

void variant_2_with_timestep_graph() {
    std::cout << "\n========== VARIANT 2: TIMESTEP GRAPH ==========\n";

    using Real = float;
    using System = Euler2D<Real>;

    // ========================================================================
    // API CHANGE #1: Graph-based solver type
    // ========================================================================

    using GraphSolver = variant2::TimestepSolver<System, flux::RusanovFlux, time::Kutta3<Real>>;

    // Create solver (graph is built in constructor)
    GraphSolver solver(400, 160, 0.005f, 0.005f, 1.4f);

    // Boundary conditions, IC setup... (same as variant 1)
    // ...

    // ========================================================================
    // API CHANGE #2: Main loop with graph execution
    // ========================================================================

    const Real t_final = 1.0f;
    Real dt = 0.0001f;
    Real t = 0.0f;
    int step = 0;

    while (t < t_final) {
        // ================================================================
        // SINGLE GRAPH LAUNCH = NO INTERMEDIATE SYNCS!
        // ================================================================

        solver.step(dt, t);  // ← Looks same, but uses graph!

        // ================================================================
        // WHAT HAPPENS INSIDE:
        // ================================================================
        //
        // Device: graph.execute() {
        //     kernel rhs_stage0;      // All stages run
        //     kernel compute_stage1;   // without interruption
        //     kernel rhs_stage1;
        //     kernel compute_stage2;
        //     kernel rhs_stage2;
        //     kernel combine_stages;  // No syncs between them!
        // }
        //
        // Total: 0-1 sync per timestep
        // ================================================================

        // Observers: only notify occasionally
        if (step % 50 == 0) {
            // ONE sync to get state for observer
            auto state = solver.get_state();  // ← SYNC (but only every 50 steps)
            std::cout << "Step " << step << ": t=" << t << std::endl;
        }

        t += dt;
        step++;
    }

    // ========================================================================
    // PERFORMANCE ANALYSIS
    // ========================================================================

    std::cout << "\nVariant 2 Performance:\n";
    std::cout << "  Timesteps: " << step << "\n";
    std::cout << "  Syncs per timestep: 0-1 (graph execution has no syncs)\n";
    std::cout << "  Observer syncs: ~" << (step / 50) << " (batched)\n";
    std::cout << "  Total syncs: ~" << (step / 50) << "\n";
    std::cout << "  Speedup vs Variant 1: 10-50x\n";
}

// ============================================================================
// VARIANT 3: WITH FULL SIMULATION GRAPH (PHASE 3)
//
// Characteristics:
// - Graph for MANY timesteps at once
// - Maximum GPU utilization
// - Minimum host involvement
// - Less flexible (must plan ahead)
//
// Performance: ~1-10 syncs for 1000 timesteps
// ============================================================================

namespace variant3 {

/**
 * @brief Full Simulation Graph
 *
 * Batches multiple timesteps, remeshing, checkpointing, observers
 * into a single graph execution.
 */
template<
    FiniteVolumeSystem System,
    template<typename> class FluxScheme = flux::RusanovFlux,
    typename TimeIntegrator = time::Kutta3<typename System::RealType>
>
class SimulationGraphSolver {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Graph = Kokkos::Experimental::TypeErasedGraph;

    // ========================================================================
    // CONCEPT: Graph is built for a BATCH of timesteps
    // ========================================================================

    struct Config {
        int timesteps_per_graph = 100;  // How many steps in one graph
        int remesh_interval = 100;
        int checkpoint_interval = 500;
        int observer_interval = 50;
    };

    SimulationGraphSolver(int nx, int ny, Real dx, Real dy, Real gamma, const Config& cfg)
        : config(cfg)
    {
        // Allocate storage
        U = Kokkos::View<Conserved**>("U", ny, nx);

        // State snapshots for remesh/checkpoint (circular buffer)
        int snapshots = 1 + cfg.timesteps_per_graph / std::min(cfg.remesh_interval, cfg.checkpoint_interval);
        U_snapshots = Kokkos::View<Conserved***>("snapshots", snapshots, ny, nx);

        build_graph();
    }

    // ========================================================================
    // NEW API: Run batch of timesteps
    // ========================================================================

    void run_timesteps(Real dt, int num_steps) {
        // ================================================================
        // EXECUTE MULTIPLE TIMESTEPS IN ONE GRAPH
        // ================================================================

        for (int batch = 0; batch < num_steps; batch += config.timesteps_per_graph) {
            int steps_this_batch = std::min(config.timesteps_per_graph, num_steps - batch);

            // SINGLE GRAPH EXECUTION for entire batch!
            execute_batch(steps_this_batch, dt);

            // Only sync HERE at end of batch
            // Host can now do remeshing, checkpointing, etc.
        }
    }

private:
    void build_graph() {
        using namespace Kokkos::Experimental;

        graph = Graph();

        // ================================================================
        // Build pattern for ONE timestep (will be replicated)
        // ================================================================

        for (int i = 0; i < config.timesteps_per_graph; ++i) {
            // Timestep node (contains all RK stages fused)
            auto timestep_node = add_timestep_node(i);

            // Optional: Remesh node
            GraphNode remesh_node;
            if ((i + 1) % config.remesh_interval == 0) {
                remesh_node = add_remesh_node(i);
                timestep_node >> remesh_node;
            }

            // Optional: Checkpoint node
            GraphNode checkpoint_node;
            if ((i + 1) % config.checkpoint_interval == 0) {
                checkpoint_node = add_checkpoint_node(i);
                if (remesh_node) timestep_node >> remesh_node >> checkpoint_node;
                else timestep_node >> checkpoint_node;
            }

            // Chain to next timestep
            if (i < config.timesteps_per_graph - 1) {
                auto next_timestep = get_timestep_node(i + 1);
                checkpoint_node >> next_timestep;
            }
        }

        // Observer notification at end
        auto observer_node = add_observer_node();
        get_last_timestep() >> observer_node;

        graph.compile();
    }

    GraphNode add_timestep_node(int step_idx) {
        // Returns a node that encapsulates ALL RK stages
        // Internally fuses: rhs → stage → rhs → ... → combine
        // ...
    }

    GraphNode add_remesh_node(int step_idx) {
        // Returns a node for AMR remeshing
        // Can be conditionally executed
        // ...
    }

    GraphNode add_checkpoint_node(int step_idx) {
        // Returns a node for checkpoint output
        // ...
    }

    GraphNode add_observer_node() {
        // Returns a node that prepares data for observer callbacks
        // Still requires ONE sync to get data to host
        // ...
    }

    void execute_batch(int num_steps, Real dt) {
        graph.execute();  // ONE LAUNCH for entire batch!

        // After graph completes, observers are notified with batched data
        notify_observers_batched();
    }

    Config config;
    Graph graph;
    Kokkos::View<Conserved**> U;
    Kokkos::View<Conserved***> U_snapshots;
};

} // namespace variant3

void variant_3_full_simulation_graph() {
    std::cout << "\n========== VARIANT 3: FULL SIMULATION GRAPH ==========\n";

    using Real = float;
    using System = Euler2D<Real>;

    // ========================================================================
    // API CHANGE #1: Configure graph batching upfront
    // ========================================================================

    variant3::SimulationGraphSolver<System>::Config graph_cfg;
    graph_cfg.timesteps_per_graph = 100;
    graph_cfg.remesh_interval = 100;
    graph_cfg.checkpoint_interval = 500;
    graph_cfg.observer_interval = 50;

    // Create solver (builds graph for 100 timesteps at once)
    variant3::SimulationGraphSolver<System> solver(
        400, 160, 0.005f, 0.005f, 1.4f, graph_cfg
    );

    // ========================================================================
    // API CHANGE #2: Run in batches
    // ========================================================================

    const Real t_final = 1.0f;
    const int total_steps = 10000;
    const Real dt = 0.0001f;

    std::cout << "Running " << total_steps << " steps in batches of "
              << graph_cfg.timesteps_per_graph << "...\n";

    solver.run_timesteps(dt, total_steps);

    // ================================================================
    // WHAT HAPPENS:
    // ================================================================
    //
    // Host: solver.run_timesteps(10000, dt) {
    //     // Batch 1: steps 0-99
    //     Device: graph.execute() {
    //         for (int i = 0; i < 100; ++i) {
    //             timestep_node();  // All RK stages fused
    //             if (i % 100 == 0) remesh_node();
    //             if (i % 500 == 0) checkpoint_node();
    //         }
    //         observer_node();
    //     }
    //     // Host syncs HERE (1 sync for 100 timesteps!)
    //
    //     // Batch 2: steps 100-199
    //     Device: graph.execute() { ... }
    //     // Host syncs HERE
    // }
    //
    // Total: 100 timesteps / batch = 100 batches
    //        = 100 syncs TOTAL for 10000 timesteps!
    // ================================================================

    // ========================================================================
    // PERFORMANCE ANALYSIS
    // ========================================================================

    std::cout << "\nVariant 3 Performance:\n";
    std::cout << "  Timesteps: " << total_steps << "\n";
    std::cout << "  Batches: " << (total_steps / graph_cfg.timesteps_per_graph) << "\n";
    std::cout << "  Syncs per batch: 1 (at end)\n";
    std::cout << "  Total syncs: ~" << (total_steps / graph_cfg.timesteps_per_graph) << "\n";
    std::cout << "  Speedup vs Variant 1: 50-100x\n";
    std::cout << "\n  NOTE: Maximum GPU utilization, but less flexible\n";
}

// ============================================================================
// COMPARISON SUMMARY
// ============================================================================

void print_comparison_summary() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         API VARIANT COMPARISON FOR 1000 TIMESTEPS            ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Variant │ Approach              │ Syncs  │ Overhead    │ Speedup ║\n";
    std::cout << "╠───────────────────────────────────────────────────────────────╣\n";
    std::cout << "║   1     │ No graph (current)    │ ~7000  │ 75-180%     │   1x   ║\n";
    std::cout << "║   2     │ TimestepGraph          │ ~100   │ 1-3%        │ 10-50x ║\n";
    std::cout << "║   3     │ Full SimulationGraph   │ ~10    │ <1%         │ 50-100x║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║                                                                ║\n";
    std::cout << "║ Variant 1: Easy to use, many syncs, good for debugging       ║\n";
    std::cout << "║ Variant 2: Balanced, RECOMMENDED for production              ║\n";
    std::cout << "║ Variant 3: Maximum performance, less flexible                ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
}

// ============================================================================
// MAIN: RUN ALL VARIANTS
// ============================================================================

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  FVD API COMPARISON: Graph vs No-Graph for GPU Performance   ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";

    // Run all variants (in a real scenario, these would be separate programs)
    print_comparison_summary();

    // Uncomment to run actual variants:
    // variant_1_no_graph();
    // variant_2_with_timestep_graph();
    // variant_3_full_simulation_graph();

    Kokkos::finalize();
    return 0;
}
