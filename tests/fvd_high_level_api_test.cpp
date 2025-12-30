/**
 * @file fvd_high_level_api_test.cpp
 * @brief Compilation test for FVD high-level API
 *
 * This test verifies that the high-level FVD API compiles correctly
 * and that all the pieces work together.
 *
 * Note: This is a COMPILATION test - we're not actually running solvers,
 * just verifying the API is well-formed.
 */

#include <subsetix/fvd/solver/solver_aliases.hpp>
#include <subsetix/fvd/solver/boundary_generic.hpp>

using namespace subsetix::fvd;

// ============================================================================
// TEST 1: Basic Type Compilation
// ============================================================================

static_assert(sizeof(Euler2D<>) > 0, "Euler2D should be compilable");
static_assert(sizeof(Euler2D<float>) > 0, "Euler2D<float> should be compilable");
static_assert(sizeof(Euler2D<double>) > 0, "Euler2D<double> should be compilable");

// ============================================================================
// TEST 2: System Types
// ============================================================================

void test_system_types() {
    // Conserved variables
    Euler2D<float>::Conserved U{1.0f, 2.0f, 0.0f, 2.5f};

    // Primitive variables
    Euler2D<float>::Primitive q{1.0f, 2.0f, 0.0f, 1.0f};

    // Conversion
    auto q2 = Euler2D<float>::to_primitive(U, 1.4f);
    auto U2 = Euler2D<float>::from_primitive(q, 1.4f);

    // Sound speed
    float a = Euler2D<float>::sound_speed(q, 1.4f);

    // Fluxes
    auto Fx = Euler2D<float>::flux_phys_x(U, q);
    auto Fy = Euler2D<float>::flux_phys_y(U, q);

    // Prevent unused warnings
    (void)q2; (void)U2; (void)a; (void)Fx; (void)Fy;
}

// ============================================================================
// TEST 3: Flux Schemes
// ============================================================================

void test_flux_schemes() {
    // Rusanov flux
    flux::RusanovFlux<Euler2D<float>> rusanov(1.4f);
    flux::RusanovFlux<Euler2D<float>> rusanov_with_sys(1.4f, Euler2D<float>{});

    // HLLC flux
    flux::HLLCFlux<Euler2D<float>> hllc(1.4f);
    flux::HLLCFlux<Euler2D<float>> hllc_with_sys(1.4f, Euler2D<float>{});

    // Roe flux
    flux::RoeFlux<Euler2D<float>> roe(1.4f);
    flux::RoeFlux<Euler2D<float>> roe_with_sys(1.4f, Euler2D<float>{});
}

// ============================================================================
// TEST 4: Reconstruction
// ============================================================================

void test_reconstruction() {
    // No reconstruction
    reconstruction::NoReconstruction no_recon;

    // MUSCL with limiters (template template parameter)
    reconstruction::MUSCL_Reconstruction<reconstruction::MinmodLimiter> muscl_minmod;
    reconstruction::MUSCL_Reconstruction<reconstruction::MCLimiter> muscl_mc;
    reconstruction::MUSCL_Reconstruction<reconstruction::SuperbeeLimiter> muscl_sb;
    reconstruction::MUSCL_Reconstruction<reconstruction::VanLeerLimiter> muscl_vl;
}

// ============================================================================
// TEST 5: Boundary Conditions
// ============================================================================

void test_boundary_conditions() {
    // Primitive state for BCs
    Euler2D<float>::Primitive inflow{1.0f, 2.0f, 0.0f, 1.0f};

    // Inflow-outflow
    auto bc1 = BoundaryConfigBuilder<Euler2D<float>>::inflow_outflow(inflow, 1.4f);

    // Dirichlet all
    auto bc2 = BoundaryConfigBuilder<Euler2D<float>>::dirichlet_all(inflow, 1.4f);

    // Neumann all
    auto bc3 = BoundaryConfigBuilder<Euler2D<float>>::neumann_all();

    // Custom
    Euler2D<float>::Primitive walls{1.0f, 0.0f, 0.0f, 1.0f};
    auto bc4 = BoundaryConfigBuilder<Euler2D<float>>::custom(
        inflow, inflow, walls, walls, 1.4f
    );

    // Prevent unused warnings
    (void)bc1; (void)bc2; (void)bc3; (void)bc4;
}

// ============================================================================
// TEST 6: Config API
// ============================================================================

void test_config_api() {
    // Default config
    EulerSolver2ndHLLC<float>::Config cfg_default;

    // From CFL
    auto cfg_cfl = EulerSolver2ndHLLC<float>::Config::from_cfl(0.5f);

    // From resolution
    auto cfg_res = EulerSolver2ndHLLC<float>::Config::from_resolution(0.01f, 0.01f);

    // With refinement
    auto cfg_ref = EulerSolver2ndHLLC<float>::Config::with_refinement(0.1f, 20);

    // For gamma
    auto cfg_gamma = EulerSolver2ndHLLC<float>::Config::for_gamma(1.67f);

    // Manual configuration
    EulerSolver2ndHLLC<float>::Config cfg;
    cfg.cfl = 0.5f;
    cfg.gamma = 1.4f;
    cfg.dx = 0.01f;
    cfg.dy = 0.01f;
    cfg.refine_fraction = 0.1f;
    cfg.ghost_layers = 1;
    cfg.remesh_stride = 20;

    // CTAD (deduced types)
    EulerSolver2ndHLLC<float>::Config cfg_ctad{0.01, 0.01, 0.45, 1.4, 0.1, 1, 20};

    // Prevent unused warnings
    (void)cfg_default; (void)cfg_cfl; (void)cfg_res; (void)cfg_ref;
    (void)cfg_gamma; (void)cfg; (void)cfg_ctad;
}

// ============================================================================
// TEST 7: Solver Aliases
// ============================================================================

void test_solver_aliases() {
    // All these should compile
    using S1 = EulerSolver1st<>;
    using S2 = EulerSolver1stHLLC<>;
    using S3 = EulerSolver1stRoe<>;
    using S4 = EulerSolver2nd<>;
    using S5 = EulerSolver2ndHLLC<>;
    using S6 = EulerSolver2ndRoe<>;
    using S7 = EulerSolver2ndMC<>;
    using S8 = EulerSolver2ndSuperbee<>;
    using S9 = EulerSolver2ndVanLeer<>;

    // Double precision
    using S10 = EulerSolver2ndHLLC<double>;
    using S11 = EulerSolver2ndHLLC<double>;

    // Custom limiter
    using S12 = EulerSolver2ndHLLC<float, reconstruction::MCLimiter>;
    using S13 = EulerSolver2nd<float, reconstruction::SuperbeeLimiter>;
    using S14 = EulerSolver2ndRoe<float, reconstruction::VanLeerLimiter>;
}

// ============================================================================
// TEST 8: Full API Compilation
// ============================================================================

void test_full_api() {
    // This test mimics what a user would write

    // 1. Choose solver
    using MySolver = EulerSolver2ndHLLC<>;

    // 2. Configure
    MySolver::Config cfg = MySolver::Config::from_cfl(0.45f);
    cfg.gamma = 1.4f;
    cfg.refine_fraction = 0.1f;

    // 3. Create solver (with stub geometry)
    csr::IntervalSet2DDevice fluid;  // Stub
    csr::Box2D domain{0, 100, 0, 100};
    MySolver solver(fluid, domain, cfg);

    // 4. Set BCs
    auto inflow = Euler2D<float>::Primitive{1.0f, 2.0f, 0.0f, 1.0f};
    solver.set_boundary_conditions(
        BoundaryConfigBuilder<Euler2D<float>>::inflow_outflow(inflow, 1.4f)
    );

    // 5. Initialize
    solver.initialize(inflow);

    // 6. Time step
    auto dt = solver.step();

    // 7. Output
    auto output = solver.get_output();
    auto geom = solver.geometry();

    // Prevent unused warnings
    (void)dt; (void)output; (void)geom;
}

// ============================================================================
// TEST 9: Double Precision
// ============================================================================

void test_double_precision() {
    using MySolverD = EulerSolver2ndHLLC<double>;

    MySolverD::Config cfg = MySolverD::Config::from_cfl(0.45);
    cfg.gamma = 1.67;  // Monatomic gas

    csr::IntervalSet2DDevice fluid;
    csr::Box2D domain{0, 100, 0, 100};
    MySolverD solver(fluid, domain, cfg);

    auto inflow = Euler2D<double>::Primitive{1.0, 2.0, 0.0, 1.0};
    solver.set_boundary_conditions(
        BoundaryConfigBuilder<Euler2D<double>>::inflow_outflow(inflow, 1.67)
    );

    solver.initialize(inflow);
    auto dt = solver.step();
    (void)dt;
}

// ============================================================================
// TEST 10: Custom Limiter
// ============================================================================

void test_custom_limiter() {
    // MC limiter (less dissipative)
    using SolverMC = EulerSolver2ndMC<>;
    SolverMC::Config cfg_mc;

    // Superbee limiter (least dissipative)
    using SolverSB = EulerSolver2ndSuperbee<>;
    SolverSB::Config cfg_sb;

    // Van Leer limiter (smooth)
    using SolverVL = EulerSolver2ndVanLeer<>;
    SolverVL::Config cfg_vl;

    // Prevent unused warnings
    (void)cfg_mc; (void)cfg_sb; (void)cfg_vl;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);

    // Run all tests (compile-time verification)
    test_system_types();
    test_flux_schemes();
    test_reconstruction();
    test_boundary_conditions();
    test_config_api();
    test_solver_aliases();
    test_full_api();
    test_double_precision();
    test_custom_limiter();

    printf("✅ All FVD high-level API compilation tests passed!\n");
    printf("   - System types: OK\n");
    printf("   - Flux schemes: OK\n");
    printf("   - Reconstruction: OK\n");
    printf("   - Boundary conditions: OK\n");
    printf("   - Config API: OK\n");
    printf("   - Solver aliases: OK\n");
    printf("   - Full API: OK\n");
    printf("   - Double precision: OK\n");
    printf("   - Custom limiters: OK\n");

    return 0;
}
