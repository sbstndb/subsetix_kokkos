/**
 * @file fvd_integrators_test.cpp
 *
 * @brief Tests for time integrators, AMR with coarsening, and time-dependent BCs
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <subsetix/fvd/fvd_integrators.hpp>

using namespace subsetix::fvd;
using namespace subsetix::fvd::solver;
using namespace subsetix::fvd::time;
using namespace subsetix::fvd::boundary;
using namespace subsetix::fvd::amr;

// ============================================================================
// TEST FIXTURE
// ============================================================================

class FvdIntegratorsTest : public ::testing::Test {
protected:
    static constexpr int nx = 50;
    static constexpr int ny = 50;
    using Real = float;
    using System = Euler2D<Real>;
};

// ============================================================================
// TIME INTEGRATOR TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, TimeIntegratorConcepts) {
    // Test that time integrators satisfy the concept
    static_assert(TimeIntegrator<ForwardEuler<Real>>);
    static_assert(TimeIntegrator<Heun2<Real>>);
    static_assert(TimeIntegrator<Kutta3<Real>>);
    static_assert(TimeIntegrator<ClassicRK4<Real>>);
    static_assert(TimeIntegrator<SSPRK3<Real>>);
    static_assert(TimeIntegrator<Ralston3<Real>>);

    EXPECT_TRUE(true);
}

TEST_F(FvdIntegratorsTest, TimeIntegratorOrder) {
    EXPECT_EQ(ForwardEuler<Real>::order, 1);
    EXPECT_EQ(ForwardEuler<Real>::stages, 1);

    EXPECT_EQ(Heun2<Real>::order, 2);
    EXPECT_EQ(Heun2<Real>::stages, 2);

    EXPECT_EQ(Kutta3<Real>::order, 3);
    EXPECT_EQ(Kutta3<Real>::stages, 3);

    EXPECT_EQ(ClassicRK4<Real>::order, 4);
    EXPECT_EQ(ClassicRK4<Real>::stages, 4);
}

TEST_F(FvdIntegratorsTest, ButcherTableauCoefficients) {
    // Check that coefficients are properly defined

    // Forward Euler: single stage, coefficient b[0] = 1
    EXPECT_FLOAT_EQ(ForwardEuler<Real>::b[0], 1.0f);

    // Heun2: b = [0.5, 0.5]
    EXPECT_FLOAT_EQ(Heun2<Real>::b[0], 0.5f);
    EXPECT_FLOAT_EQ(Heun2<Real>::b[1], 0.5f);

    // Kutta3: b = [1/6, 2/3, 1/6]
    EXPECT_FLOAT_EQ(Kutta3<Real>::b[0], 1.0f/6.0f);
    EXPECT_FLOAT_EQ(Kutta3<Real>::b[1], 2.0f/3.0f);
    EXPECT_FLOAT_EQ(Kutta3<Real>::b[2], 1.0f/6.0f);

    // RK4: b = [1/6, 1/3, 1/3, 1/6]
    EXPECT_FLOAT_EQ(ClassicRK4<Real>::b[0], 1.0f/6.0f);
    EXPECT_FLOAT_EQ(ClassicRK4<Real>::b[1], 1.0f/3.0f);
    EXPECT_FLOAT_EQ(ClassicRK4<Real>::b[2], 1.0f/3.0f);
    EXPECT_FLOAT_EQ(ClassicRK4<Real>::b[3], 1.0f/6.0f);
}

TEST_F(FvdIntegratorsTest, TimeDependentBC_POD) {
    // Verify TimeDependentBC is POD (GPU-compatible)
    using TDB = TimeDependentBC<Real>;
    EXPECT_TRUE(std::is_trivially_copyable_v<TDB>);
    EXPECT_TRUE(std::is_standard_layout_v<TDB>);
}

TEST_F(FvdIntegratorsTest, TimeDependentBC_Sinusoidal) {
    TimeDependentBC<Real> bc;
    bc.rho0 = 1.0f;
    bc.u0 = 100.0f;
    bc.frequency = 2.0f * 3.14159f;  // 1 Hz
    bc.amplitude = 0.1f;
    bc.rho_mod = TimeDependentBC<Real>::Sinusoidal;

    // At t=0: rho = 1.0 * (1 + 0.1 * sin(0)) = 1.0
    EXPECT_FLOAT_EQ(bc.rho(0.0f), 1.0f);

    // At t=0.25: rho = 1.0 * (1 + 0.1 * sin(pi/2)) = 1.1
    float rho_quarter = bc.rho(0.25f);
    EXPECT_NEAR(rho_quarter, 1.1f, 0.01f);

    // At t=0.5: rho = 1.0 * (1 + 0.1 * sin(pi)) = 1.0
    EXPECT_FLOAT_EQ(bc.rho(0.5f), 1.0f);
}

TEST_F(FvdIntegratorsTest, TimeDependentBC_SquareWave) {
    TimeDependentBC<Real> bc;
    bc.rho0 = 1.0f;
    bc.frequency = 2.0f * 3.14159f;
    bc.amplitude = 0.2f;
    bc.rho_mod = TimeDependentBC<Real>::SquareWave;

    // At t=0: rho = 1.0 * (1 + 0.2) = 1.2
    EXPECT_NEAR(bc.rho(0.0f), 1.2f, 0.01f);

    // At t=0.5: rho = 1.0 * (1 - 0.2) = 0.8
    EXPECT_NEAR(bc.rho(0.5f), 0.8f, 0.01f);
}

// ============================================================================
// ZONE PREDICATE TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, ZonePredicate_IntervalX) {
    auto zone = ZonePredicate<Real>::interval_x(0.2f, 0.4f);

    EXPECT_TRUE(zone.contains(0.3f, 0.5f));  // x in range
    EXPECT_FALSE(zone.contains(0.1f, 0.5f)); // x too low
    EXPECT_FALSE(zone.contains(0.5f, 0.5f)); // x too high
}

TEST_F(FvdIntegratorsTest, ZonePredicate_Rectangle) {
    auto zone = ZonePredicate<Real>::rectangle(0.0f, 1.0f, 0.0f, 0.5f);

    EXPECT_TRUE(zone.contains(0.5f, 0.25f));  // Inside
    EXPECT_FALSE(zone.contains(1.5f, 0.25f)); // x outside
    EXPECT_FALSE(zone.contains(0.5f, 0.75f)); // y outside
}

TEST_F(FvdIntegratorsTest, ZonePredicate_Circle) {
    auto zone = ZonePredicate<Real>::circle(0.5f, 0.5f, 0.25f);

    EXPECT_TRUE(zone.contains(0.5f, 0.5f));   // Center
    EXPECT_TRUE(zone.contains(0.6f, 0.5f));   // Inside (radius 0.1 from center)
    EXPECT_FALSE(zone.contains(0.9f, 0.5f));  // Outside
}

TEST_F(FvdIntegratorsTest, ZonePredicate_POD) {
    using ZP = ZonePredicate<Real>;
    EXPECT_TRUE(std::is_trivially_copyable_v<ZP>);
    EXPECT_TRUE(std::is_standard_layout_v<ZP>);
}

// ============================================================================
// BC DESCRIPTOR TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, BcDescriptor_POD) {
    using BD = BcDescriptor<System>;
    EXPECT_TRUE(std::is_trivially_copyable_v<BD>);
    EXPECT_TRUE(std::is_standard_layout_v<BD>);
}

TEST_F(FvdIntegratorsTest, BcDescriptor_Static) {
    BcDescriptor<System> bc;
    bc.type = BcDescriptor<System>::StaticDirichlet;

    typename System::Primitive q{1.0f, 100.0f, 0.0f, 100000.0f};
    bc.static_value = System::from_primitive(q, System::default_gamma);

    auto value = bc.get_value(0.0f);
    EXPECT_FLOAT_EQ(value.rho, 1.0f);
}

TEST_F(FvdIntegratorsTest, BcDescriptor_TimeDependent) {
    BcDescriptor<System> bc;
    bc.type = BcDescriptor<System>::TimeDependentDirichlet;

    bc.time_policy.rho0 = 1.0f;
    bc.time_policy.frequency = 2.0f * 3.14159f;
    bc.time_policy.amplitude = 0.1f;
    bc.time_policy.rho_mod = TimeDependentBC<Real>::Sinusoidal;

    auto value = bc.get_value(0.0f);
    EXPECT_FLOAT_EQ(value.rho, 1.0f);

    // At t=0.25, should be 1.1
    value = bc.get_value(0.25f);
    EXPECT_NEAR(value.rho, 1.1f, 0.01f);
}

// ============================================================================
// AMR CRITERION TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, RefinementCriterion_Gradient_POD) {
    using GC = GradientCriterion<System>;
    EXPECT_TRUE(std::is_trivially_copyable_v<GC>);
}

TEST_F(FvdIntegratorsTest, RefinementCriterion_ShockSensor_POD) {
    using SSC = ShockSensorCriterion<System>;
    EXPECT_TRUE(std::is_trivially_copyable_v<SSC>);
}

TEST_F(FvdIntegratorsTest, RefinementCriterion_Vorticity_POD) {
    using VC = VorticityCriterion<System>;
    EXPECT_TRUE(std::is_trivially_copyable_v<VC>);
}

TEST_F(FvdIntegratorsTest, RefinementCriterion_ValueRange_POD) {
    using VRC = ValueRangeCriterion<System>;
    EXPECT_TRUE(std::is_trivially_copyable_v<VRC>);
}

TEST_F(FvdIntegratorsTest, ValueRangeCriterion_Inside) {
    ValueRangeCriterion<System> crit;
    crit.variable = ValueRangeCriterion<System>::Density;
    crit.min_val = 0.5f;
    crit.max_val = 1.5f;
    crit.invert = false;

    typename System::Conserved U{1.0f, 100.0f, 0.0f, 200000.0f};
    typename System::Primitive q{1.0f, 100.0f, 0.0f, 100000.0f};

    auto action = crit.evaluate(U, q, 0.01f);
    EXPECT_EQ(action, RefinementAction::Refine);
}

TEST_F(FvdIntegratorsTest, ValueRangeCriterion_Outside) {
    ValueRangeCriterion<System> crit;
    crit.variable = ValueRangeCriterion<System>::Density;
    crit.min_val = 0.5f;
    crit.max_val = 1.5f;
    crit.invert = false;

    typename System::Conserved U{2.0f, 200.0f, 0.0f, 400000.0f};
    typename System::Primitive q{2.0f, 100.0f, 0.0f, 100000.0f};

    auto action = crit.evaluate(U, q, 0.01f);
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST_F(FvdIntegratorsTest, ValueRangeCriterion_Inverted) {
    ValueRangeCriterion<System> crit;
    crit.variable = ValueRangeCriterion<System>::Density;
    crit.min_val = 0.5f;
    crit.max_val = 1.5f;
    crit.invert = true;  // Refine OUTSIDE range

    typename System::Conserved U{2.0f, 200.0f, 0.0f, 400000.0f};
    typename System::Primitive q{2.0f, 100.0f, 0.0f, 100000.0f};

    auto action = crit.evaluate(U, q, 0.01f);
    EXPECT_EQ(action, RefinementAction::Refine);
}

// ============================================================================
// COMPOSITE CRITERION TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, CompositeCriterion_OR) {
    CompositeCriterion<System, 8> comp;
    comp.logic_op = CompositeCriterion<System, 8>::Or;

    // Add two criteria
    GradientCriterion<System> grad1;
    grad1.threshold = 0.5f;
    comp.add_gradient(grad1);

    ValueRangeCriterion<System> range;
    range.variable = ValueRangeCriterion<System>::Density;
    range.min_val = 0.5f;
    range.max_val = 1.5f;
    comp.add_value_range(range);

    EXPECT_EQ(comp.num_criteria, 2);
    EXPECT_EQ(static_cast<int>(comp.logic_op),
              static_cast<int>(CompositeCriterion<System, 8>::Or));
}

TEST_F(FvdIntegratorsTest, CompositeCriterion_AND) {
    CompositeCriterion<System, 8> comp;
    comp.logic_op = CompositeCriterion<System, 8>::And;

    EXPECT_EQ(static_cast<int>(comp.logic_op),
              static_cast<int>(CompositeCriterion<System, 8>::And));
}

// ============================================================================
// EXCLUSION ZONE TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, ExclusionZone_Rectangle) {
    ExclusionZone<Real> zone;
    zone.predicate = ExclusionZone<Real>::Rectangle;
    zone.x_min = 0.0f;
    zone.x_max = 0.5f;
    zone.y_min = 0.0f;
    zone.y_max = 0.5f;
    zone.min_level = 2;

    EXPECT_TRUE(zone.contains(0.25f, 0.25f));
    EXPECT_FALSE(zone.contains(0.75f, 0.75f));
}

TEST_F(FvdIntegratorsTest, ExclusionZone_Circle) {
    ExclusionZone<Real> zone;
    zone.predicate = ExclusionZone<Real>::Circle;
    zone.center_x = 0.5f;
    zone.center_y = 0.5f;
    zone.radius = 0.25f;
    zone.min_level = 3;

    EXPECT_TRUE(zone.contains(0.5f, 0.5f));
    EXPECT_TRUE(zone.contains(0.6f, 0.5f));
    EXPECT_FALSE(zone.contains(0.9f, 0.5f));
}

TEST_F(FvdIntegratorsTest, ExclusionZone_POD) {
    using EZ = ExclusionZone<Real>;
    EXPECT_TRUE(std::is_trivially_copyable_v<EZ>);
}

// ============================================================================
// REFINEMENT MANAGER TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, RefinementManager_AddCriteria) {
    RefinementManager<System> mgr;

    mgr.add_gradient_criterion(0.1f);
    mgr.add_shock_sensor_criterion(
        ShockSensorCriterion<System>::Ducros,
        0.8f
    );
    mgr.add_vorticity_criterion(1.0f);
    mgr.add_value_range_criterion(
        ValueRangeCriterion<System>::Density,
        0.5f, 1.5f
    );

    EXPECT_EQ(mgr.config.criterion.num_criteria, 4);
}

TEST_F(FvdIntegratorsTest, RefinementManager_AddExclusions) {
    RefinementManager<System> mgr;

    mgr.add_exclusion_rectangle(0.0f, 0.5f, 0.0f, 0.5f, 2);
    mgr.add_exclusion_circle(0.75f, 0.75f, 0.1f, 3);

    EXPECT_EQ(mgr.config.num_exclusions, 2);
}

TEST_F(FvdIntegratorsTest, RefinementManager_Config) {
    RefinementManager<System> mgr;

    mgr.add_gradient_criterion(0.1f);
    mgr.set_level_limits(0, 5);
    mgr.set_remesh_frequency(100);
    mgr.set_coarsening(true);

    EXPECT_EQ(mgr.config.min_level, 0);
    EXPECT_EQ(mgr.config.max_level, 5);
    EXPECT_EQ(mgr.config.remesh_interval, 100);
    EXPECT_TRUE(mgr.config.enable_coarsening);
}

// ============================================================================
// BC MANAGER TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, BcManager_Initialize) {
    BcManager<System> mgr;

    mgr.initialize(nx, ny, 0.01f, 0.01f, 0.0f, 0.0f);

    EXPECT_EQ(mgr.needs_sync(), false);
}

TEST_F(FvdIntegratorsTest, BcManager_AddStaticBC) {
    BcManager<System> mgr;
    mgr.initialize(nx, ny, 0.01f, 0.01f);

    typename System::Primitive q{1.0f, 100.0f, 0.0f, 100000.0f};
    mgr.add_static_bc("left", BcDescriptor<System>::StaticDirichlet, q);

    EXPECT_TRUE(mgr.needs_sync());
}

TEST_F(FvdIntegratorsTest, BcManager_AddTimeDependentBC) {
    BcManager<System> mgr;
    mgr.initialize(nx, ny, 0.01f, 0.01f);

    auto sinusoidal = sinusoidal_inlet<System>(1.0f, 100.0f, 2.0f * 3.14159f);
    mgr.add_time_dependent_bc("left", sinusoidal);

    EXPECT_TRUE(mgr.needs_sync());
}

TEST_F(FvdIntegratorsTest, BcManager_AddZonalBC) {
    BcManager<System> mgr;
    mgr.initialize(nx, ny, 0.01f, 0.01f);

    auto zone = ZonePredicate<Real>::interval_x(0.2f, 0.4f);
    typename System::Primitive q{1.0f, 50.0f, 0.0f, 100000.0f};
    mgr.add_zonal_bc("bottom", zone, q, System::default_gamma, 1);

    EXPECT_TRUE(mgr.needs_sync());
}

TEST_F(FvdIntegratorsTest, ConvenienceFunctions) {
    // Test sinusoidal_inlet
    auto inlet = sinusoidal_inlet<System>(1.0f, 100.0f, 2.0f * 3.14159f);
    EXPECT_FLOAT_EQ(inlet.rho0, 1.0f);
    EXPECT_FLOAT_EQ(inlet.u0, 100.0f);

    // Test pulsating_inlet
    auto pulse = pulsating_inlet<System>(1.0f, 100.0f, 3.0f);
    EXPECT_FLOAT_EQ(pulse.frequency, 3.0f);

    // Test linear_ramp
    auto ramp = linear_ramp<System>(1.0f, 100.0f, 0.5f);
    EXPECT_FLOAT_EQ(ramp.amplitude, 0.5f);
}

// ============================================================================
// INTEGRATED API TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, StandardAMR) {
    auto mgr = standard_amr<System>();

    EXPECT_EQ(mgr.config.criterion.num_criteria, 2);  // shock + vorticity
    EXPECT_EQ(static_cast<int>(mgr.config.criterion.logic_op),
              static_cast<int>(CompositeCriterion<System, 8>::Or));
    EXPECT_EQ(mgr.config.min_level, 0);
    EXPECT_EQ(mgr.config.max_level, 5);
    EXPECT_TRUE(mgr.config.enable_coarsening);
}

TEST_F(FvdIntegratorsTest, StandardAdaptiveDT) {
    auto cfg = standard_adaptive_dt<Real>();

    EXPECT_FLOAT_EQ(cfg.cfl_target, 0.8f);
    EXPECT_FLOAT_EQ(cfg.cfl_max, 1.0f);
    EXPECT_FLOAT_EQ(cfg.dt_max, 0.01f);
    EXPECT_FLOAT_EQ(cfg.growth_factor, 1.2f);
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
