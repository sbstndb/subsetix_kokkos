/**
 * @file fvd_mach2_cylinder_example.cpp
 * @brief Example: Mach 2 flow over a cylinder using FVD high-level API
 *
 * This demonstrates the user-facing API for solving the 2D compressible
 * Euler equations with AMR.
 *
 * Note: This is a STUB example - it compiles but doesn't actually
 * solve the equations (that's the next phase of implementation).
 */

#include <Kokkos_Core.hpp>
#include <subsetix/fvd/solver/solver_aliases.hpp>
#include <subsetix/fvd/solver/boundary_generic.hpp>
#include <subsetix/fvd/solver/adaptive_solver.hpp>

using namespace subsetix;
using namespace subsetix::fvd;

// NOTE: Using placeholder types from subsetix::fvd::csr namespace
// In production, these would be the real CSR types from subsetix::csr

// ============================================================================
// MAIN EXAMPLE
// ============================================================================

int main(int argc, char** argv) {
    // ========================================================================
    // 1. INITIALIZE KOKKOS
    // ========================================================================
    Kokkos::ScopeGuard guard(argc, argv);

    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  FVD High-Level API Example: Mach 2 Flow Over Cylinder        ║\n");
    printf("║  This is a COMPILATION EXAMPLE - not a working solver yet     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    // ========================================================================
    // 2. DEFINE GEOMETRY
    // ========================================================================
    const int nx = 400, ny = 160;
    csr::Box2D domain{0, nx, 0, ny};

    printf("Geometry:\n");
    printf("  Domain: %d x %d\n", nx, ny);
    printf("  Obstacle: Cylinder at (%d, %d) radius 20\n", nx/4, ny/2);

    // In production, would create actual geometry using subsetix::csr
    // For now, use stub types from subsetix::fvd::csr namespace
    csr::IntervalSet2DDevice fluid;

    // ========================================================================
    // 3. CHOOSE SOLVER (GAME CHANGER: Simple alias!)
    // ========================================================================
    printf("\nSolver Configuration:\n");
    printf("  Using: EulerSolver2ndHLLC<>\n");
    printf("  Order: 2nd (MUSCL + Minmod limiter)\n");
    printf("  Flux: HLLC (captures contact discontinuities)\n");

    using MySolver = EulerSolver2ndHLLC<>;

    // ========================================================================
    // 4. CONFIGURE SOLVER
    // ========================================================================
    printf("\nSolver Parameters:\n");

    // Option 1: Use helper method
    MySolver::Config cfg = MySolver::Config::from_cfl(0.45f);
    cfg.gamma = 1.4f;           // Air
    cfg.refine_fraction = 0.1f; // 10% refinement

    printf("  CFL: %.2f\n", cfg.cfl);
    printf("  Gamma: %.1f\n", cfg.gamma);
    printf("  Refine fraction: %.1f\n", cfg.refine_fraction);

    // ========================================================================
    // 5. CREATE SOLVER
    // ========================================================================
    MySolver solver(fluid, domain, cfg);
    printf("\nSolver created successfully!\n");

    // ========================================================================
    // 6. CONFIGURE BOUNDARY CONDITIONS
    // ========================================================================
    printf("\nBoundary Conditions:\n");

    // Mach 2 inflow from left
    auto mach = 2.0f;
    auto gamma = 1.4f;
    auto sound_speed = 1.0f;
    auto inflow_velocity = mach * Kokkos::sqrt(gamma);

    auto inflow = Euler2D<>::Primitive{
        1.0f,                          // rho
        inflow_velocity,                // u
        0.0f,                           // v
        1.0f                            // p
    };

    printf("  Inflow: Mach %.1f from left\n", mach);
    printf("  Outflow: Neumann (right, top, bottom)\n");

    solver.set_boundary_conditions(
        BoundaryConfigBuilder<Euler2D<>>::inflow_outflow(inflow, gamma)
    );

    // ========================================================================
    // 7. INITIALIZE
    // ========================================================================
    printf("\nInitializing with inflow state...\n");
    solver.initialize(inflow);
    printf("Initialization complete!\n");

    // ========================================================================
    // 8. MAIN LOOP
    // ========================================================================
    printf("\nStarting time integration...\n");
    printf("  Target time: t = 0.01\n");

    auto t = solver.get_time_zero();
    const auto t_final = 0.01f;
    int output_count = 0;

    while (t < t_final) {
        // Time step
        auto dt = solver.step();
        t += dt;

        // Output periodically
        if (output_count++ % 50 == 0) {
            printf("  Step %d: t = %.5f, dt = %.5f\n",
                   solver.get_step_count(), t, dt);

            // In production, would write VTK output:
            // auto output = solver.get_finest_output();
            // vtk::write_legacy_quads(
            //     csr::toHost(solver.geometry()),
            //     output.rho,
            //     "output/step_" + std::to_string(output_count) + ".vtk"
            // );
        }
    }

    printf("\nSimulation complete!\n");
    printf("  Final time: %.5f\n", t);
    printf("  Total steps: %d\n", solver.get_step_count());

    // ========================================================================
    // SUMMARY
    // ========================================================================
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  API SUMMARY - What the user writes:                          ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  1. Choose solver:                                            ║\n");
    printf("║     using MySolver = EulerSolver2ndHLLC<>;                    ║\n");
    printf("║                                                              ║\n");
    printf("║  2. Configure:                                                ║\n");
    printf("║     MySolver::Config cfg = MySolver::Config::from_cfl(0.45);  ║\n");
    printf("║     cfg.gamma = 1.4;                                          ║\n");
    printf("║                                                              ║\n");
    printf("║  3. Create solver:                                            ║\n");
    printf("║     MySolver solver(fluid, domain, cfg);                      ║\n");
    printf("║                                                              ║\n");
    printf("║  4. Set BCs:                                                  ║\n");
    printf("║     solver.set_boundary_conditions(                          ║\n");
    printf("║         BoundaryConfigBuilder<Euler2D<>>::inflow_outflow(in)  ║\n");
    printf("║     );                                                       ║\n");
    printf("║                                                              ║\n");
    printf("║  5. Initialize:                                               ║\n");
    printf("║     solver.initialize(initial_state);                         ║\n");
    printf("║                                                              ║\n");
    printf("║  6. Time loop:                                                ║\n");
    printf("║     while (t < t_final) t += solver.step();                   ║\n");
    printf("║                                                              ║\n");
    printf("║  Result: Clean, simple API!                                  ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    return 0;
}
