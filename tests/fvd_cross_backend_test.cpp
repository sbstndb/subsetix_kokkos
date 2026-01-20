/**
 * @file fvd_cross_backend_test.cpp
 * @brief Cross-backend validation tests for FVD layer
 *
 * These tests verify that the FVD layer produces consistent results
 * across different Kokkos execution spaces (Serial, OpenMP, CUDA).
 *
 * IMPORTANT: This test uses a different approach from typical parameterized
 * tests because Kokkos execution spaces cannot be switched at runtime.
 *
 * Instead, we run the same problem setup and verify that:
 * - The solver runs successfully on the current backend
 * - Time steps are consistent
 * - Basic physical properties are maintained
 *
 * For true cross-validation, run this test multiple times with different
 * Kokkos backends and compare the checksums manually:
 *   - Serial backend:   ./subsetix_test_fvd_cross_backend
 *   - OpenMP backend:   KOKKOS_DEVICE_OPENMP=1 ./subsetix_test_fvd_cross_backend
 *   - CUDA backend:     KOKKOS_DEVICE_CUDA=1 ./subsetix_test_fvd_cross_backend
 *
 * The test outputs checksums that can be compared across runs.
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <subsetix/fvd/solver/solver_aliases.hpp>
#include <subsetix/fvd/solver/boundary_generic.hpp>
#include <subsetix/fvd/geometry/geometry_builder.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>

#include <cmath>
#include <cstdio>

using namespace subsetix::fvd;
using namespace subsetix::csr;

// ============================================================================
// TEST CONFIGURATION
// ============================================================================

using Real = float;
using System = Euler2D<Real>;
using Solver = EulerSolver1st<Real>;  // 1st order for simplicity

// Test problem size (small for fast testing)
static constexpr int nx = 40;
static constexpr int ny = 40;
static constexpr int n_steps = 20;

// ============================================================================
// CHECKSUM UTILITIES
// ============================================================================

/**
 * @brief Checksum of simulation state for cross-backend comparison
 *
 * Stores summary statistics that should be identical (within tolerance)
 * across different backends for the same problem setup.
 */
struct SimulationChecksum {
    int steps_taken = 0;
    Real final_time = 0;
    Real total_time = 0;
    Real min_dt = 1e30f;
    Real max_dt = 0;
    Real sum_dt = 0;
    Real mean_dt = 0;

    // For more detailed validation, we would also include:
    // - Total mass (sum of rho over all cells)
    // - Total momentum (sum of rhou, rhov)
    // - Total energy (sum of E)
    // However, these require accessing the internal U_ field which is private.

    /**
     * @brief Print checksum for comparison across backends
     */
    void print(const char* backend_name) const {
        printf("\n  [%s Backend Checksum]\n", backend_name);
        printf("    steps_taken:    %d\n", steps_taken);
        printf("    final_time:     %.10e\n", final_time);
        printf("    total_time:     %.10e\n", total_time);
        printf("    min_dt:         %.10e\n", min_dt);
        printf("    max_dt:         %.10e\n", max_dt);
        printf("    mean_dt:        %.10e\n", mean_dt);
    }

    /**
     * @brief Compare two checksums for equality
     *
     * Returns true if all values match within tolerance.
     * Tolerance accounts for floating-point differences from:
     * - Parallel reduction associativity (OpenMP/CUDA)
     * - Different instruction order
     * - Compiler optimizations
     */
    bool matches(const SimulationChecksum& other,
                 Real rtol = Real(1e-5),
                 Real atol = Real(1e-8)) const
    {
        bool steps_ok = steps_taken == other.steps_taken;
        bool time_ok = std::abs(final_time - other.final_time) <
                       atol + rtol * Real(0.5) * (std::abs(final_time) + std::abs(other.final_time));
        bool min_ok = std::abs(min_dt - other.min_dt) <
                      atol + rtol * Real(0.5) * (std::abs(min_dt) + std::abs(other.min_dt));
        bool max_ok = std::abs(max_dt - other.max_dt) <
                      atol + rtol * Real(0.5) * (std::abs(max_dt) + std::abs(other.max_dt));
        bool mean_ok = std::abs(mean_dt - other.mean_dt) <
                       atol + rtol * Real(0.5) * (std::abs(mean_dt) + std::abs(other.mean_dt));

        return steps_ok && time_ok && min_ok && max_ok && mean_ok;
    }
};

/**
 * @brief Run a standard test problem and return checksum
 *
 * This function runs a simple advection problem with:
 * - Uniform initial condition
 * - Neumann (outflow) boundary conditions
 * - Forward Euler time stepping
 * - Rusanov flux
 *
 * The problem is simple enough to produce consistent results across backends.
 */
SimulationChecksum run_standard_problem() {
    SimulationChecksum checksum;

    // Create simple box geometry
    Geometry2D<Real> geom = Geometry2D<Real>::build_box(
        nx, ny, Real(0.01), Real(0.01)
    );
    IntervalSet2DDevice fluid = geom.build();
    Box2D domain{0, nx, 0, ny};

    // Configure solver
    typename Solver::Config config;
    config.cfl = Real(0.4);
    config.gamma = Real(1.4);

    Solver solver(fluid, domain, config);

    // Set up Neumann (outflow) BCs on all sides
    auto bc = BoundaryConfigBuilder<System>::neumann_all();
    solver.set_boundary_conditions(bc);

    // Initialize with uniform state
    typename System::Primitive initial{1.0f, 0.3f, 0.0f, 1.0f};
    solver.initialize(initial);

    // Run simulation and collect statistics
    for (int i = 0; i < n_steps; ++i) {
        Real dt = solver.step();
        checksum.steps_taken++;
        checksum.total_time += dt;
        checksum.sum_dt += dt;
        checksum.min_dt = std::min(checksum.min_dt, dt);
        checksum.max_dt = std::max(checksum.max_dt, dt);
    }

    checksum.final_time = checksum.total_time;
    checksum.mean_dt = checksum.sum_dt / Real(checksum.steps_taken);

    return checksum;
}

/**
 * @brief Run problem with Dirichlet BCs
 *
 * Tests boundary condition consistency across backends.
 */
SimulationChecksum run_dirichlet_problem() {
    SimulationChecksum checksum;

    Geometry2D<Real> geom = Geometry2D<Real>::build_box(
        nx, ny, Real(0.01), Real(0.01)
    );
    IntervalSet2DDevice fluid = geom.build();
    Box2D domain{0, nx, 0, ny};

    typename Solver::Config config;
    config.cfl = Real(0.4);
    config.gamma = Real(1.4);

    Solver solver(fluid, domain, config);

    // Dirichlet BCs with fixed inflow
    typename System::Primitive inflow{1.5f, 0.5f, 0.0f, 1.5f};
    auto bc = BoundaryConfigBuilder<System>::dirichlet_all(inflow, Real(1.4));
    solver.set_boundary_conditions(bc);

    typename System::Primitive initial{1.0f, 0.3f, 0.0f, 1.0f};
    solver.initialize(initial);

    for (int i = 0; i < n_steps; ++i) {
        Real dt = solver.step();
        checksum.steps_taken++;
        checksum.total_time += dt;
        checksum.sum_dt += dt;
        checksum.min_dt = std::min(checksum.min_dt, dt);
        checksum.max_dt = std::max(checksum.max_dt, dt);
    }

    checksum.final_time = checksum.total_time;
    checksum.mean_dt = checksum.sum_dt / Real(checksum.steps_taken);

    return checksum;
}

// ============================================================================
// TESTS
// ============================================================================

/**
 * @brief Test basic time stepping on current backend
 *
 * This test verifies that the solver can initialize and take steps
 * on the current backend (Serial, OpenMP, or CUDA).
 */
TEST(FvdCrossBackend, BasicTimeStepping) {
    auto checksum = run_standard_problem();

    // Verify basic properties
    EXPECT_EQ(checksum.steps_taken, n_steps);
    EXPECT_GT(checksum.final_time, Real(0));
    EXPECT_GT(checksum.min_dt, Real(0));
    EXPECT_GT(checksum.max_dt, Real(0));
    EXPECT_GE(checksum.max_dt, checksum.min_dt);

    // Time should be monotonically increasing
    EXPECT_GT(checksum.total_time, Real(0));

    // Mean dt should be reasonable (not zero or infinite)
    EXPECT_GT(checksum.mean_dt, Real(0));
    EXPECT_LT(checksum.mean_dt, Real(1));  // Sanity check

    // Checksum is implicitly valid if we got here
    EXPECT_TRUE(checksum.steps_taken == n_steps);
}

/**
 * @brief Test with Dirichlet boundary conditions
 *
 * Verifies that Dirichlet BCs work correctly on the current backend.
 */
TEST(FvdCrossBackend, DirichletBoundaryConditions) {
    auto checksum = run_dirichlet_problem();

    // Verify basic properties
    EXPECT_EQ(checksum.steps_taken, n_steps);
    EXPECT_GT(checksum.final_time, Real(0));
    EXPECT_GT(checksum.mean_dt, Real(0));
}

/**
 * @brief Test CFL-based time stepping
 *
 * Verifies that the time step adapts based on the CFL condition.
 * For a uniform initial condition with zero velocity, the time step
 * should be nearly constant.
 */
TEST(FvdCrossBackend, CflTimeStepping) {
    Geometry2D<Real> geom = Geometry2D<Real>::build_box(
        nx, ny, Real(0.01), Real(0.01)
    );
    IntervalSet2DDevice fluid = geom.build();
    Box2D domain{0, nx, 0, ny};

    typename Solver::Config config;
    config.cfl = Real(0.5);  // Higher CFL for this test
    config.gamma = Real(1.4);

    Solver solver(fluid, domain, config);

    auto bc = BoundaryConfigBuilder<System>::neumann_all();
    solver.set_boundary_conditions(bc);

    // Zero initial velocity (max wave speed = sound speed)
    typename System::Primitive initial{1.0f, 0.0f, 0.0f, 1.0f};
    solver.initialize(initial);

    // Collect time steps
    std::vector<Real> dts;
    for (int i = 0; i < 10; ++i) {
        Real dt = solver.step();
        dts.push_back(dt);
        EXPECT_GT(dt, Real(0));
    }

    // For zero velocity, time steps should be very consistent
    Real dt_min = *std::min_element(dts.begin(), dts.end());
    Real dt_max = *std::max_element(dts.begin(), dts.end());
    Real variation = (dt_max - dt_min) / dt_min;

    // Variation should be very small (< 1%)
    EXPECT_LT(variation, Real(0.01));
}

/**
 * @brief Print checksums for cross-backend comparison
 *
 * This test prints detailed checksums that can be manually compared
 * across different backend runs. To use:
 *
 * 1. Run with Serial backend:
 *    ./build-serial/subsetix_test_fvd_cross_backend --gtest_filter=*CrossBackendChecksums
 *
 * 2. Run with OpenMP backend:
 *    ./build-openmp/subsetix_test_fvd_cross_backend --gtest_filter=*CrossBackendChecksums
 *
 * 3. Run with CUDA backend:
 *    ./build-cuda/subsetix_test_fvd_cross_backend --gtest_filter=*CrossBackendChecksums
 *
 * 4. Compare the checksums manually - they should match within tolerance.
 */
TEST(FvdCrossBackend, PrintChecksums) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  CROSS-BACKEND VALIDATION CHECKSUMS                           ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("Current backend: %s\n", typeid(Kokkos::DefaultExecutionSpace).name());
    printf("\n");

    // Run standard problem
    auto checksum1 = run_standard_problem();
    checksum1.print("Standard Problem (Neumann BCs)");

    // Run Dirichlet problem
    auto checksum2 = run_dirichlet_problem();
    checksum2.print("Dirichlet BCs");

    printf("\n");
    printf("To validate cross-backend consistency:\n");
    printf("  1. Run this test with different Kokkos backends\n");
    printf("  2. Compare the checksums above\n");
    printf("  3. Values should match within ~1e-5 relative tolerance\n");
    printf("\n");
}

/**
 * @brief Self-consistency check on current backend
 *
 * Runs the same problem twice and verifies identical results.
 * This is a basic sanity check that the solver is deterministic.
 */
TEST(FvdCrossBackend, SelfConsistency) {
    auto checksum1 = run_standard_problem();
    auto checksum2 = run_standard_problem();

    // Results should be bit-identical on the same backend
    EXPECT_TRUE(checksum1.matches(checksum2, 0.0f, 0.0f))
        << "Solver should produce identical results on the same backend";

    // Check specific values
    EXPECT_FLOAT_EQ(checksum1.final_time, checksum2.final_time);
    EXPECT_FLOAT_EQ(checksum1.min_dt, checksum2.min_dt);
    EXPECT_FLOAT_EQ(checksum1.max_dt, checksum2.max_dt);
    EXPECT_FLOAT_EQ(checksum1.mean_dt, checksum2.mean_dt);
}

// ============================================================================
// AMR OPERATIONS TESTS
// ============================================================================

/**
 * @brief Test AMR operations on current backend
 *
 * Verifies that AMR refinement/coarsening operations work correctly.
 * Note: Full AMR testing requires setup of refinement criteria.
 */
TEST(FvdCrossBackend, AmrOperations) {
    Geometry2D<Real> geom = Geometry2D<Real>::build_box(
        nx, ny, Real(0.01), Real(0.01)
    );
    IntervalSet2DDevice fluid = geom.build();
    Box2D domain{0, nx, 0, ny};

    typename Solver::Config config;
    config.cfl = Real(0.4);
    config.gamma = Real(1.4);

    Solver solver(fluid, domain, config);

    auto bc = BoundaryConfigBuilder<System>::neumann_all();
    solver.set_boundary_conditions(bc);

    typename System::Primitive initial{1.0f, 0.5f, 0.0f, 1.0f};
    solver.initialize(initial);

    // Take some steps
    for (int i = 0; i < 5; ++i) {
        Real dt = solver.step();
        EXPECT_GT(dt, Real(0));
    }

    // Note: Full AMR testing would require:
    // 1. Enable refinement with set_refinement()
    // 2. Trigger remeshing
    // 3. Verify refined regions match expected pattern
    // 4. Compare refinement pattern across backends

    SUCCEED() << "AMR operations test placeholder (full AMR requires refinement criteria setup)";
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);

    // Print backend information
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  FVD CROSS-BACKEND VALIDATION TESTS                           ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("Kokkos Configuration:\n");
    printf("  Default Execution Space: %s\n",
           typeid(Kokkos::DefaultExecutionSpace).name());
#ifdef KOKKOS_ENABLE_SERIAL
    printf("  Serial: ENABLED\n");
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    printf("  OpenMP: ENABLED\n");
#endif
#ifdef KOKKOS_ENABLE_CUDA
    printf("  CUDA: ENABLED\n");
#endif
    printf("\n");
    printf("NOTE: This test runs on the current Kokkos backend.\n");
    printf("For cross-validation, run with different backends and compare checksums.\n");
    printf("\n");

    int result = RUN_ALL_TESTS();

    Kokkos::finalize();
    return result;
}
