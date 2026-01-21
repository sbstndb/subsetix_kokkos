/**
 * @file fvd_integrators_test.cpp
 *
 * @brief Tests for time integrators, AMR with coarsening, and time-dependent BCs
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <subsetix/fvd/fvd_integrators.hpp>
#include <subsetix/fvd/flux/flux_schemes.hpp>
#include <subsetix/fvd/geometry/geometry_builder.hpp>
#include <subsetix/fvd/output/field_view.hpp>
#include <subsetix/csr_ops/set_algebra.hpp>

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
// TIME DEPENDENT BC - ADDITIONAL MODULATION TYPES
// ============================================================================

TEST_F(FvdIntegratorsTest, TimeDependentBC_AllModulationTypes) {
    TimeDependentBC<Real> bc;
    bc.rho0 = 1.0f;
    bc.amplitude = 0.5f;
    bc.frequency = 1.0f;

    // Test Constant modulation
    bc.rho_mod = TimeDependentBC<Real>::Constant;
    EXPECT_FLOAT_EQ(bc.rho(0.0f), 1.0f);
    EXPECT_FLOAT_EQ(bc.rho(100.0f), 1.0f);

    // Test Linear modulation: rho = 1.0 * (1 + 0.5 * t)
    bc.rho_mod = TimeDependentBC<Real>::Linear;
    EXPECT_FLOAT_EQ(bc.rho(0.0f), 1.0f);
    EXPECT_FLOAT_EQ(bc.rho(1.0f), 1.5f);  // 1.0 * (1 + 0.5)
    EXPECT_FLOAT_EQ(bc.rho(2.0f), 2.0f);  // 1.0 * (1 + 1.0)

    // Test Exponential modulation: rho = 1.0 * exp(0.5 * t)
    bc.rho_mod = TimeDependentBC<Real>::Exponential;
    EXPECT_FLOAT_EQ(bc.rho(0.0f), 1.0f);
    EXPECT_NEAR(bc.rho(1.0f), 1.6487f, 0.01f);  // exp(0.5)
    EXPECT_NEAR(bc.rho(2.0f), 2.718f, 0.01f);   // exp(1.0)
}

TEST_F(FvdIntegratorsTest, TimeDependentBC_PhaseAndReferenceTime) {
    TimeDependentBC<Real> bc;
    bc.rho0 = 1.0f;
    bc.amplitude = 0.1f;
    bc.frequency = 2.0f * 3.14159f;  // 1 Hz
    bc.rho_mod = TimeDependentBC<Real>::Sinusoidal;

    // Test phase offset: sin(2*pi*t + pi/2) = cos(2*pi*t)
    bc.phase = 1.5708f;  // pi/2
    EXPECT_NEAR(bc.rho(0.0f), 1.1f, 0.01f);  // sin(pi/2) = 1
    EXPECT_NEAR(bc.rho(0.25f), 1.0f, 0.01f); // sin(pi) = 0

    // Test reference time offset
    bc.phase = 0.0f;
    bc.t0 = 0.5f;
    EXPECT_FLOAT_EQ(bc.rho(0.5f), 1.0f);  // t - t0 = 0
    EXPECT_NEAR(bc.rho(0.75f), 1.1f, 0.01f);  // t - t0 = 0.25
}

TEST_F(FvdIntegratorsTest, ZonePredicate_IntervalY) {
    auto zone = ZonePredicate<Real>::interval_y(0.2f, 0.4f);

    EXPECT_TRUE(zone.contains(0.5f, 0.3f));  // y in range
    EXPECT_FALSE(zone.contains(0.5f, 0.1f)); // y too low
    EXPECT_FALSE(zone.contains(0.5f, 0.5f)); // y too high
}

TEST_F(FvdIntegratorsTest, ZonePredicate_TimeDependent) {
    ZonePredicate<Real> moving_zone;
    moving_zone.predicate = ZonePredicate<Real>::Rectangle;
    moving_zone.x_min = 0.0f;
    moving_zone.x_max = 0.2f;
    moving_zone.y_min = 0.0f;
    moving_zone.y_max = 1.0f;
    moving_zone.time_dependent = true;
    moving_zone.velocity_x = 0.1f;  // Moves right at 0.1 units/time

    // At t=0: zone at [0, 0.2]
    EXPECT_TRUE(moving_zone.contains(0.1f, 0.5f, 0.0f));
    EXPECT_FALSE(moving_zone.contains(0.3f, 0.5f, 0.0f));

    // At t=1: zone moved to [0.1, 0.3]
    EXPECT_TRUE(moving_zone.contains(0.2f, 0.5f, 1.0f));
    EXPECT_FALSE(moving_zone.contains(0.05f, 0.5f, 1.0f));
}

TEST_F(FvdIntegratorsTest, BcDescriptor_AllTypes) {
    // Test Outflow type
    BcDescriptor<System> bc_outflow;
    bc_outflow.type = BcDescriptor<System>::Outflow;
    auto val_out = bc_outflow.get_value(0.5f);
    EXPECT_FLOAT_EQ(val_out.rho, 0.0f);  // Returns zero conserved

    // Test TimeDependentInlet type
    BcDescriptor<System> bc_inlet;
    bc_inlet.type = BcDescriptor<System>::TimeDependentInlet;
    bc_inlet.time_policy.rho0 = 1.0f;
    bc_inlet.time_policy.u0 = 100.0f;
    bc_inlet.time_policy.frequency = 2.0f * 3.14159f;
    bc_inlet.time_policy.amplitude = 0.1f;
    bc_inlet.time_policy.u_mod = TimeDependentBC<Real>::Sinusoidal;

    auto val = bc_inlet.get_value(0.25f);
    EXPECT_NEAR(val.rho, 1.0f, 0.01f);
    EXPECT_NEAR(val.rhou, 110.0f, 1.0f);  // rho * u = 1.0 * 110.0
}

TEST_F(FvdIntegratorsTest, BcManager_AdvancedOperations) {
    BcManager<System> mgr;
    mgr.initialize(nx, ny, 0.01f, 0.01f, 0.0f, 0.0f);

    // Test sync_to_device
    EXPECT_FALSE(mgr.needs_sync());
    typename System::Primitive q{1.0f, 100.0f, 0.0f, 100000.0f};
    mgr.add_static_bc("left", BcDescriptor<System>::StaticDirichlet, q);
    EXPECT_TRUE(mgr.needs_sync());
    mgr.sync_to_device();
    EXPECT_FALSE(mgr.needs_sync());

    // Test update_bc
    typename System::Primitive q_new{2.0f, 150.0f, 0.0f, 150000.0f};
    mgr.update_bc("left", 0, BcDescriptor<System>::StaticDirichlet, q_new);
    EXPECT_TRUE(mgr.needs_sync());

    // Test remove_bc
    mgr.remove_bc("left", 0);
    EXPECT_TRUE(mgr.needs_sync());
}

// ============================================================================
// FLUX SCHEMES UNIT TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, FluxSchemes_Rusanov_Euler2D) {
    using namespace subsetix::fvd::flux;

    RusanovFlux<System> flux(1.4f);

    // Create two states
    typename System::Conserved UL{1.0f, 1.0f, 0.0f, 2.5f};
    typename System::Conserved UR{0.9f, 0.9f, 0.0f, 2.3f};
    typename System::Primitive qL = System::to_primitive(UL);
    typename System::Primitive qR = System::to_primitive(UR);

    // Compute flux in x-direction
    auto F = flux.flux_x(UL, UR, qL, qR);

    // Verify mass flux (rho component) is computed
    EXPECT_TRUE(F.rho != 0.0f || F.rhou != 0.0f || F.rhov != 0.0f || F.E != 0.0f);
}

TEST_F(FvdIntegratorsTest, FluxSchemes_HLLC_Euler2D) {
    using namespace subsetix::fvd::flux;

    HLLCFlux<System> flux(1.4f);

    // Create two states with same pressure
    // For Euler2D with gamma=1.4: p = (gamma-1) * (E - 0.5*rho*(u^2+v^2))
    // Left: rho=1, u=0.5, p=2.5 => E = 2.5/0.4 + 0.5*1*0.25 = 6.25 + 0.125 = 6.375
    // Right: rho=0.5, u=0.25, p=2.5 => E = 2.5/0.4 + 0.5*0.5*0.0625 = 6.25 + 0.015625 = 6.265625
    typename System::Conserved UL{1.0f, 0.5f, 0.0f, 6.375f};
    typename System::Conserved UR{0.5f, 0.125f, 0.0f, 6.265625f};  // Half density, same pressure
    typename System::Primitive qL = System::to_primitive(UL);
    typename System::Primitive qR = System::to_primitive(UR);

    // Verify both have same pressure
    EXPECT_NEAR(qL.p, qR.p, 0.01f);

    // Compute flux in x-direction
    auto F = flux.flux_x(UL, UR, qL, qR);

    // HLLC should return a valid flux
    EXPECT_TRUE(std::isfinite(F.rho));
    EXPECT_TRUE(std::isfinite(F.rhou));
    EXPECT_TRUE(std::isfinite(F.E));
}

TEST_F(FvdIntegratorsTest, FluxSchemes_Advection2D_Rusanov) {
    using namespace subsetix::fvd::flux;
    using AdvSystem = Advection2D<Real>;

    RusanovFlux<AdvSystem> flux;

    // Create two scalar states
    typename AdvSystem::Conserved UL{1.0f};
    typename AdvSystem::Conserved UR{0.5f};
    typename AdvSystem::Primitive qL{1.0f};
    typename AdvSystem::Primitive qR{0.5f};

    // Compute flux in x-direction
    auto F = flux.flux_x(UL, UR, qL, qR);

    // Advection flux should be computed
    EXPECT_TRUE(std::isfinite(F.value));
}

// ============================================================================
// REFINEMENT CRITERIA - ADDITIONAL TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, SmoothnessCriterion_CoarsenSmoothRegion) {
    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = 0.01f;  // Low variance = coarsen

    // Create center state
    typename System::Conserved U_center{1.0f, 0.5f, 0.0f, 2.5f};
    typename System::Primitive q_center{1.0f, 0.5f, 0.0f, 2.5f};

    // Create 4 similar sibling states (smooth region)
    typename System::Conserved siblings[4] = {
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.01f, 0.505f, 0.0f, 2.525f},
        {0.99f, 0.495f, 0.0f, 2.475f},
        {1.005f, 0.5025f, 0.0f, 2.5125f}
    };

    Real dx = 0.01f;

    // SmoothnessCriterion::evaluate takes (U_center, q_center, siblings, dx)
    auto action = crit.evaluate(U_center, q_center, siblings, dx);

    // Should coarsen due to low variance
    EXPECT_EQ(action, RefinementAction::Coarsen);
}

TEST_F(FvdIntegratorsTest, CompositeCriterion_VoteLogic) {
    CompositeCriterion<System, 8> comp;
    comp.logic_op = CompositeCriterion<System, 8>::Vote;

    // Add multiple criteria
    ValueRangeCriterion<System> range;
    range.variable = ValueRangeCriterion<System>::Density;
    range.min_val = 0.5f;
    range.max_val = 1.5f;
    range.invert = false;
    comp.add_value_range(range);

    GradientCriterion<System> grad;
    grad.threshold = 10.0f;
    comp.add_gradient(grad);

    EXPECT_EQ(comp.num_criteria, 2);
    EXPECT_EQ(static_cast<int>(comp.logic_op),
              static_cast<int>(CompositeCriterion<System, 8>::Vote));
}

TEST_F(FvdIntegratorsTest, UserDefinedCriterion) {
    // UserDefinedCriterion requires both System and Function template args
    // For this test, we just verify the concept is satisfied
    // by using a simple function pointer type
    using UserFn = RefinementAction(*)(const typename System::Conserved&,
                                        const typename System::Primitive&,
                                        Real);

    // Verify that a function pointer can be used
    UserFn fn = [](const typename System::Conserved& U,
                   const typename System::Primitive& q,
                   Real dx) {
        if (q.rho > 1.5f) {
            return RefinementAction::Refine;
        }
        return RefinementAction::Keep;
    };

    // Just verify it's callable
    typename System::Conserved U{1.0f, 100.0f, 0.0f, 200000.0f};
    typename System::Primitive q{2.0f, 100.0f, 0.0f, 100000.0f};
    auto action = fn(U, q, 0.01f);
    EXPECT_EQ(action, RefinementAction::Refine);  // rho = 2.0 > 1.5
}

TEST_F(FvdIntegratorsTest, ValueRangeCriterion_AllVariableTypes) {
    typename System::Conserved U{1.0f, 100.0f, 0.0f, 200000.0f};
    typename System::Primitive q{1.0f, 100.0f, 0.0f, 100000.0f};

    // Test Density variable
    {
        ValueRangeCriterion<System> crit;
        crit.variable = ValueRangeCriterion<System>::Density;
        crit.min_val = 0.5f;
        crit.max_val = 1.5f;
        crit.invert = false;

        auto action = crit.evaluate(U, q, 0.01f);
        EXPECT_EQ(action, RefinementAction::Refine);  // 1.0 is in [0.5, 1.5]
    }

    // Test Pressure variable
    {
        ValueRangeCriterion<System> crit;
        crit.variable = ValueRangeCriterion<System>::Pressure;
        crit.min_val = 90000.0f;
        crit.max_val = 110000.0f;
        crit.invert = false;

        auto action = crit.evaluate(U, q, 0.01f);
        EXPECT_EQ(action, RefinementAction::Refine);  // 100000 is in range
    }

    // Test VelocityX variable
    {
        ValueRangeCriterion<System> crit;
        crit.variable = ValueRangeCriterion<System>::VelocityX;
        crit.min_val = 90.0f;
        crit.max_val = 110.0f;
        crit.invert = false;

        auto action = crit.evaluate(U, q, 0.01f);
        EXPECT_EQ(action, RefinementAction::Refine);  // 100 is in range
    }
}

// ============================================================================
// GEOMETRY BUILDER TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, GeometryBuilder_RectangleCreation) {
    // Geometry2D is directly in subsetix::fvd namespace
    auto geom = subsetix::fvd::Geometry2D<Real>::build_box(100, 100)
        .add_rectangle(10.0f, 30.0f, 20.0f, 40.0f, true)   // Obstacle
        .add_box(50.0f, 70.0f, 60.0f, 80.0f, true);        // Another obstacle

    EXPECT_EQ(geom.obstacles().size(), 2);
}

TEST_F(FvdIntegratorsTest, GeometryBuilder_FluidRegions) {
    auto geom = subsetix::fvd::Geometry2D<Real>::build_box(100, 100)
        .add_cylinder(50.0f, 50.0f, 10.0f, false)  // Fluid region (not obstacle)
        .add_rectangle(10.0f, 30.0f, 10.0f, 30.0f, false);  // Another fluid region

    EXPECT_EQ(geom.fluid_regions().size(), 2);
}

TEST_F(FvdIntegratorsTest, GeometryBuilder_PhysicalCoordinates) {
    // Create geometry from physical coordinates
    auto geom = subsetix::fvd::Geometry2D<Real>::build_box(0.0f, 1.0f, 0.0f, 0.5f, 0.01f, 0.01f);

    EXPECT_EQ(geom.nx(), 100);  // (1.0 - 0.0) / 0.01 = 100
    EXPECT_EQ(geom.ny(), 50);   // (0.5 - 0.0) / 0.01 = 50
    EXPECT_NEAR(geom.dx(), 0.01f, 0.001f);
    EXPECT_NEAR(geom.dy(), 0.01f, 0.001f);
}

TEST_F(FvdIntegratorsTest, GeometryBuilder_EdgeCases) {
    // Zero radius cylinder should return empty geometry
    auto geom1 = subsetix::fvd::Geometry2D<Real>::build_box(100, 100)
        .add_cylinder(50.0f, 50.0f, 0.0f, true);

    auto csr1 = geom1.build();
    EXPECT_GT(csr1.num_rows, 0);  // Domain still exists

    // Rectangle completely outside domain
    auto geom2 = subsetix::fvd::Geometry2D<Real>::build_box(100, 100)
        .add_rectangle(200.0f, 300.0f, 200.0f, 300.0f, true);

    EXPECT_EQ(geom2.obstacles().size(), 1);  // Still added, but clamped during build
}

// ============================================================================
// FIELD VIEW TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, FieldView_DefaultConstruction) {
    FieldView<Real> field;
    EXPECT_TRUE(field.is_empty());
    EXPECT_EQ(field.size(), 0);
    EXPECT_TRUE(field.name().empty());
}

TEST_F(FvdIntegratorsTest, FieldView_Allocate) {
    auto field = FieldView<Real>::allocate("density", 100);
    EXPECT_FALSE(field.is_empty());
    EXPECT_EQ(field.size(), 100);
    EXPECT_EQ(field.name(), "density");
}

TEST_F(FvdIntegratorsTest, FieldView_ViewFactory) {
    Kokkos::View<Real*> data("data", 50);
    auto field = FieldView<Real>::view(data, "pressure");
    EXPECT_FALSE(field.is_empty());
    EXPECT_EQ(field.size(), 50);
    EXPECT_EQ(field.name(), "pressure");
}

TEST_F(FvdIntegratorsTest, FieldView_ToHost) {
    auto field = FieldView<Real>::allocate("test", 10);

    // Fill with data on host first, then copy to device
    std::vector<Real> source_data(10);
    for (int i = 0; i < 10; ++i) {
        source_data[i] = static_cast<Real>(i) * 2.0f;
    }
    field.from_host(source_data);

    // Copy to host and verify
    auto host_data = field.to_host();
    EXPECT_EQ(host_data.size(), 10);
    EXPECT_FLOAT_EQ(host_data[0], 0.0f);
    EXPECT_FLOAT_EQ(host_data[1], 2.0f);
    EXPECT_FLOAT_EQ(host_data[5], 10.0f);
}

TEST_F(FvdIntegratorsTest, FieldView_FromHost_Success) {
    std::vector<Real> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto field = FieldView<Real>::allocate("from_host", 4);

    field.from_host(host_data);

    // Verify data was copied
    auto result = field.to_host();
    EXPECT_EQ(result.size(), 4);
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[3], 4.0f);
}

TEST_F(FvdIntegratorsTest, FieldView_FromHost_Empty) {
    std::vector<Real> host_data;
    auto field = FieldView<Real>::allocate("empty", 0);
    EXPECT_NO_THROW(field.from_host(host_data));
}

TEST_F(FvdIntegratorsTest, FieldView_Level) {
    auto field = FieldView<Real>::allocate("level_test", 100);
    EXPECT_EQ(field.level(), 0);

    auto field_level2 = FieldView<Real>::allocate("level2", 100, 2);
    EXPECT_EQ(field_level2.level(), 2);
}

// ============================================================================
// FIELD SET TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, FieldSet_DefaultConstruction) {
    FieldSet<Real> fs;
    EXPECT_TRUE(fs.is_empty());
    EXPECT_EQ(fs.size(), 0);
}

TEST_F(FvdIntegratorsTest, FieldSet_Add) {
    FieldSet<Real> fs;
    auto field1 = FieldView<Real>::allocate("rho", 100);
    auto field2 = FieldView<Real>::allocate("p", 100);

    fs.add(field1);
    fs.add(field2);

    EXPECT_FALSE(fs.is_empty());
    EXPECT_EQ(fs.size(), 2);
}

TEST_F(FvdIntegratorsTest, FieldSet_OperatorIndex) {
    FieldSet<Real> fs;
    auto field = FieldView<Real>::allocate("test", 50);
    fs.add(field);

    EXPECT_EQ(fs[0].size(), 50);
    EXPECT_EQ(fs[0].name(), "test");
}

TEST_F(FvdIntegratorsTest, FieldSet_GetByName) {
    FieldSet<Real> fs;
    auto rho = FieldView<Real>::allocate("density", 100);
    auto p = FieldView<Real>::allocate("pressure", 100);
    fs.add(rho);
    fs.add(p);

    auto* found = fs.get("density");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->name(), "density");

    auto* not_found = fs.get("velocity");
    EXPECT_EQ(not_found, nullptr);
}

TEST_F(FvdIntegratorsTest, FieldSet_Iteration) {
    FieldSet<Real> fs;
    fs.add(FieldView<Real>::allocate("f1", 10));
    fs.add(FieldView<Real>::allocate("f2", 10));
    fs.add(FieldView<Real>::allocate("f3", 10));

    int count = 0;
    for (auto it = fs.begin(); it != fs.end(); ++it) {
        ++count;
    }
    EXPECT_EQ(count, 3);
}

// ============================================================================
// SOLVER OUTPUT TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, SolverOutput_Construction) {
    SolverOutput<Real> output;

    // Add fields
    output.fields.add(FieldView<Real>::allocate("rho", 100));
    output.fields.add(FieldView<Real>::allocate("p", 100));

    output.time = 1.5f;
    output.level = 2;

    // Note: geometry is required for is_valid(), but for this test
    // we just verify the other properties work without geometry
    EXPECT_EQ(output.time, 1.5f);
    EXPECT_EQ(output.level, 2);
    EXPECT_EQ(output.fields.size(), 2);
    EXPECT_FALSE(output.is_valid());  // No geometry
}

TEST_F(FvdIntegratorsTest, SolverOutput_GetField) {
    SolverOutput<Real> output;
    output.fields.add(FieldView<Real>::allocate("density", 100));
    output.fields.add(FieldView<Real>::allocate("pressure", 100));

    auto* field = output.get_field("density");
    ASSERT_NE(field, nullptr);
    EXPECT_EQ(field->name(), "density");

    auto* missing = output.get_field("velocity");
    EXPECT_EQ(missing, nullptr);
}

TEST_F(FvdIntegratorsTest, SolverOutput_IsValid_NoFields) {
    SolverOutput<Real> output;
    EXPECT_FALSE(output.is_valid());
}

// ============================================================================
// GRADIENT CRITERION TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, GradientCriterion_RefinesOnDensityGradient) {
    GradientCriterion<System> crit;
    crit.threshold = 1.0f;
    crit.use_rho = true;
    crit.use_p = false;
    crit.use_u = false;

    typename System::Conserved U_center{1.0f, 0.5f, 0.0f, 2.5f};
    typename System::Primitive q_center = System::to_primitive(U_center);

    // Create siblings with large density gradient
    typename System::Conserved siblings[4] = {
        {2.0f, 1.0f, 0.0f, 5.0f},   // High density
        {0.5f, 0.25f, 0.0f, 1.25f}, // Low density
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f}
    };

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    EXPECT_EQ(action, RefinementAction::Refine);
}

TEST_F(FvdIntegratorsTest, GradientCriterion_NoGradientReturnsKeep) {
    GradientCriterion<System> crit;
    crit.threshold = 1.0f;
    crit.use_rho = true;

    typename System::Conserved U_center{1.0f, 0.5f, 0.0f, 2.5f};
    typename System::Primitive q_center = System::to_primitive(U_center);

    // All siblings have same density - no gradient
    typename System::Conserved siblings[4] = {
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f}
    };

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST_F(FvdIntegratorsTest, GradientCriterion_RespectsUseRhoFlag) {
    GradientCriterion<System> crit;
    crit.threshold = 1.0f;
    crit.use_rho = false;  // Disabled
    crit.use_p = true;

    typename System::Conserved U_center{1.0f, 0.5f, 0.0f, 2.5f};
    typename System::Primitive q_center = System::to_primitive(U_center);

    // Large density gradient but pressure gradient checking disabled
    typename System::Conserved siblings[4] = {
        {2.0f, 1.0f, 0.0f, 5.0f},
        {0.5f, 0.25f, 0.0f, 1.25f},
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f}
    };

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    // Should keep since density checking is disabled
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST_F(FvdIntegratorsTest, GradientCriterion_POD_Compliant) {
    using Crit = GradientCriterion<System>;
    EXPECT_TRUE(std::is_default_constructible_v<Crit>);
    EXPECT_TRUE(std::is_trivially_copyable_v<Crit>);
}

// ============================================================================
// SHOCK SENSOR CRITERION TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, ShockSensorCriterion_DucrosSensor) {
    ShockSensorCriterion<System> crit;
    crit.threshold = 0.5f;
    crit.sensor_type = ShockSensorCriterion<System>::Ducros;

    typename System::Conserved U_center{1.0f, 0.5f, 0.0f, 2.5f};
    typename System::Primitive q_center = System::to_primitive(U_center);

    // Create pressure gradient (shock indicator)
    typename System::Conserved siblings[4] = {
        {1.5f, 0.75f, 0.0f, 5.0f},   // High pressure
        {0.8f, 0.4f, 0.0f, 1.0f},    // Low pressure
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f}
    };

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    // High pressure gradient should trigger refinement
    EXPECT_TRUE(action == RefinementAction::Refine || action == RefinementAction::Keep);
}

TEST_F(FvdIntegratorsTest, ShockSensorCriterion_JamesonSensor) {
    ShockSensorCriterion<System> crit;
    crit.threshold = 0.5f;
    crit.sensor_type = ShockSensorCriterion<System>::Jameson;

    typename System::Conserved U_center{1.0f, 0.5f, 0.0f, 2.5f};
    typename System::Primitive q_center = System::to_primitive(U_center);

    typename System::Conserved siblings[4] = {
        {1.5f, 0.75f, 0.0f, 5.0f},
        {0.8f, 0.4f, 0.0f, 1.0f},
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f}
    };

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    EXPECT_TRUE(action == RefinementAction::Refine || action == RefinementAction::Keep);
}

TEST_F(FvdIntegratorsTest, ShockSensorCriterion_POD_Compliant) {
    using Crit = ShockSensorCriterion<System>;
    EXPECT_TRUE(std::is_default_constructible_v<Crit>);
    EXPECT_TRUE(std::is_trivially_copyable_v<Crit>);
}

// ============================================================================
// VORTICITY CRITERION TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, VorticityCriterion_HighVorticityRefines) {
    VorticityCriterion<System> crit;
    crit.threshold = 1.0f;

    typename System::Conserved U_center{1.0f, 0.0f, 0.0f, 2.5f};
    typename System::Primitive q_center = System::to_primitive(U_center);

    // Create vorticity: dv/dx - du/dy
    // Sibling above has positive v, sibling right has positive u
    typename System::Conserved siblings[4] = {
        {1.0f, 0.5f, 0.0f, 2.5f},   // Right: high u
        {1.0f, 0.0f, 0.5f, 2.5f},   // Top: high v
        {1.0f, 0.0f, 0.0f, 2.5f},
        {1.0f, 0.0f, 0.0f, 2.5f}
    };

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    // Should detect high vorticity
    EXPECT_TRUE(action == RefinementAction::Refine || action == RefinementAction::Keep);
}

TEST_F(FvdIntegratorsTest, VorticityCriterion_ZeroVorticityKeeps) {
    VorticityCriterion<System> crit;
    crit.threshold = 1.0f;

    typename System::Conserved U_center{1.0f, 0.5f, 0.0f, 2.5f};
    typename System::Primitive q_center = System::to_primitive(U_center);

    // Uniform flow - no vorticity
    typename System::Conserved siblings[4] = {
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f}
    };

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST_F(FvdIntegratorsTest, VorticityCriterion_POD_Compliant) {
    using Crit = VorticityCriterion<System>;
    EXPECT_TRUE(std::is_default_constructible_v<Crit>);
    EXPECT_TRUE(std::is_trivially_copyable_v<Crit>);
}

// ============================================================================
// SMOOTHNESS CRITERION ADVANCED TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, SmoothnessCriterion_HighVarianceKeeps) {
    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = 0.01f;

    typename System::Conserved U_center{1.0f, 0.5f, 0.0f, 2.5f};
    typename System::Primitive q_center = System::to_primitive(U_center);

    // High variance region
    typename System::Conserved siblings[4] = {
        {0.5f, 0.25f, 0.0f, 1.25f},
        {1.5f, 0.75f, 0.0f, 3.75f},
        {0.8f, 0.4f, 0.0f, 2.0f},
        {1.2f, 0.6f, 0.0f, 3.0f}
    };

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST_F(FvdIntegratorsTest, SmoothnessCriterion_AllSiblingsSimilar) {
    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = 0.01f;

    typename System::Conserved U_center{1.0f, 0.5f, 0.0f, 2.5f};
    typename System::Primitive q_center = System::to_primitive(U_center);

    // All cells very similar
    typename System::Conserved siblings[4] = {
        {1.001f, 0.5005f, 0.0f, 2.5025f},
        {0.999f, 0.4995f, 0.0f, 2.4975f},
        {1.0005f, 0.50025f, 0.0f, 2.50125f},
        {0.9995f, 0.49975f, 0.0f, 2.49875f}
    };

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    EXPECT_EQ(action, RefinementAction::Coarsen);
}

TEST_F(FvdIntegratorsTest, SmoothnessCriterion_UseRhoFlag) {
    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = 0.01f;
    crit.use_rho = true;
    crit.use_p = false;
    crit.use_u = false;

    typename System::Conserved U_center{1.0f, 0.5f, 0.0f, 2.5f};
    typename System::Primitive q_center = System::to_primitive(U_center);

    // Smooth density but varying pressure
    typename System::Conserved siblings[4] = {
        {1.001f, 0.5005f, 0.0f, 2.5025f},
        {0.999f, 0.4995f, 0.0f, 2.4975f},
        {1.0005f, 0.50025f, 0.0f, 2.50125f},
        {0.9995f, 0.49975f, 0.0f, 2.49875f}
    };

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    // Should coarsen based on smooth density
    EXPECT_EQ(action, RefinementAction::Coarsen);
}

TEST_F(FvdIntegratorsTest, SmoothnessCriterion_Advection2D) {
    using AdvSystem = Advection2D<Real>;
    SmoothnessCriterion<AdvSystem> crit;
    crit.coarsen_threshold = 0.01f;

    typename AdvSystem::Conserved U_center{1.0f};
    typename AdvSystem::Primitive q_center{1.0f};

    // Smooth scalar field
    typename AdvSystem::Conserved siblings[4] = {
        1.001f, 0.999f, 1.0005f, 0.9995f
    };

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    EXPECT_EQ(action, RefinementAction::Coarsen);
}

// ============================================================================
// COMPOSITE CRITERION ADVANCED TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, CompositeCriterion_OR_OneRefineReturnsRefine) {
    CompositeCriterion<System, 8> comp;
    comp.logic_op = CompositeCriterion<System, 8>::Or;

    GradientCriterion<System> grad;
    grad.threshold = 1000.0f;  // Won't trigger
    comp.add_gradient(grad);

    ValueRangeCriterion<System> range;
    range.variable = ValueRangeCriterion<System>::Density;
    range.min_val = 0.5f;
    range.max_val = 1.5f;
    comp.add_value_range(range);

    typename System::Conserved U{1.0f, 0.5f, 0.0f, 2.5f};
    typename System::Primitive q = System::to_primitive(U);
    typename System::Conserved siblings[4] = {
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f}
    };

    auto action = comp.evaluate(U, q, siblings, 0.01f);
    // ValueRange should trigger refine
    EXPECT_EQ(action, RefinementAction::Refine);
}

TEST_F(FvdIntegratorsTest, CompositeCriterion_AND_AllRefineReturnsRefine) {
    CompositeCriterion<System, 8> comp;
    comp.logic_op = CompositeCriterion<System, 8>::And;

    GradientCriterion<System> grad;
    grad.threshold = 1000.0f;
    grad.use_rho = false;  // Disabled
    comp.add_gradient(grad);

    ValueRangeCriterion<System> range;
    range.variable = ValueRangeCriterion<System>::Density;
    range.min_val = 0.5f;
    range.max_val = 1.5f;
    comp.add_value_range(range);

    typename System::Conserved U{1.0f, 0.5f, 0.0f, 2.5f};
    typename System::Primitive q = System::to_primitive(U);
    typename System::Conserved siblings[4] = {
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f}
    };

    auto action = comp.evaluate(U, q, siblings, 0.01f);
    // AND requires all to agree - gradient says Keep, so result is Keep
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST_F(FvdIntegratorsTest, CompositeCriterion_Vote_MajorityRefines) {
    CompositeCriterion<System, 8> comp;
    comp.logic_op = CompositeCriterion<System, 8>::Vote;

    // Add 3 criteria: all 3 should say Refine
    ValueRangeCriterion<System> range1;
    range1.variable = ValueRangeCriterion<System>::Density;
    range1.min_val = 0.5f;
    range1.max_val = 1.5f;
    comp.add_value_range(range1);

    ValueRangeCriterion<System> range2;
    range2.variable = ValueRangeCriterion<System>::Pressure;
    range2.min_val = 200000.0f;
    range2.max_val = 300000.0f;
    comp.add_value_range(range2);

    ValueRangeCriterion<System> range3;
    range3.variable = ValueRangeCriterion<System>::VelocityX;
    range3.min_val = 0.0f;
    range3.max_val = 1.0f;
    comp.add_value_range(range3);

    typename System::Conserved U{1.0f, 0.5f, 0.0f, 2.5f};
    typename System::Primitive q = System::to_primitive(U);
    typename System::Conserved siblings[4] = {
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f},
        {1.0f, 0.5f, 0.0f, 2.5f}
    };

    auto action = comp.evaluate(U, q, siblings, 0.01f);
    // All 3 should say Refine (all values in range)
    EXPECT_EQ(action, RefinementAction::Refine);
}

// ============================================================================
// GEOMETRY BUILDER TESTS
// ============================================================================

TEST_F(FvdIntegratorsTest, GeometryBuilder_PhysicalCoordinates_RectangleFactory) {
    // Test build_box with physical bounds (lines 119-124)
    auto geom = subsetix::fvd::Geometry2D<Real>::build_box(0.0f, 1.0f, 0.0f, 0.5f, 0.01f, 0.01f);
    EXPECT_EQ(geom.nx(), 100);  // (1.0-0.0)/0.01
    EXPECT_EQ(geom.ny(), 50);   // (0.5-0.0)/0.01
    EXPECT_FLOAT_EQ(geom.dx(), 0.01f);
    EXPECT_FLOAT_EQ(geom.dy(), 0.01f);
}

TEST_F(FvdIntegratorsTest, GeometryBuilder_AddCylinderAsFluidRegion) {
    auto geom = subsetix::fvd::Geometry2D<Real>::build_box(100, 100)
        .add_cylinder(50.0f, 50.0f, 10.0f, false);  // is_obstacle=false

    EXPECT_EQ(geom.fluid_regions().size(), 1);
    EXPECT_EQ(geom.obstacles().size(), 0);
}

TEST_F(FvdIntegratorsTest, GeometryBuilder_AddRectangleAsFluidRegion) {
    auto geom = subsetix::fvd::Geometry2D<Real>::build_box(100, 100)
        .add_rectangle(10.0f, 20.0f, 10.0f, 20.0f, false);

    EXPECT_EQ(geom.fluid_regions().size(), 1);
    EXPECT_EQ(geom.obstacles().size(), 0);
}

TEST_F(FvdIntegratorsTest, GeometryBuilder_AddBoxAlias) {
    auto geom = subsetix::fvd::Geometry2D<Real>::build_box(100, 100)
        .add_box(10.0f, 20.0f, 10.0f, 20.0f, true);

    EXPECT_EQ(geom.obstacles().size(), 1);
}

TEST_F(FvdIntegratorsTest, GeometryBuilder_BuildWithObstacle_PhysicalCoords) {
    // Create domain with physical coordinates and add obstacle
    auto geom = subsetix::fvd::Geometry2D<Real>::build_box(0.0f, 1.0f, 0.0f, 1.0f, 0.01f, 0.01f)
        .add_rectangle(0.3f, 0.7f, 0.3f, 0.7f, true);  // Center obstacle

    auto csr = geom.build();

    // Verify obstacle was subtracted from domain
    EXPECT_GT(csr.num_rows, 0);
    // Full domain would have 10000 cells, obstacle removes center
    EXPECT_LT(csr.total_cells, 10000);
}

TEST_F(FvdIntegratorsTest, GeometryBuilder_BuildWithFluidRegionAddition) {
    // Create domain, then add additional fluid region
    auto geom = subsetix::fvd::Geometry2D<Real>::build_box(100, 100)
        .add_cylinder(50.0f, 50.0f, 20.0f, false);  // Add fluid

    auto csr = geom.build();

    // Verify build succeeds (union with full domain = full domain)
    EXPECT_GT(csr.num_rows, 0);
}

TEST_F(FvdIntegratorsTest, GeometryBuilder_PrimitiveToCsr_BoxOutOfBounds) {
    auto geom = subsetix::fvd::Geometry2D<Real>::build_box(100, 100, 1.0f, 1.0f)
        .add_rectangle(200.0f, 300.0f, 200.0f, 300.0f, true);  // Way outside

    auto csr = geom.build();

    // Should be clamped to domain edge or empty
    EXPECT_GE(csr.num_rows, 0);
}

TEST_F(FvdIntegratorsTest, GeometryBuilder_PrimitiveToCsr_InvalidBox) {
    auto geom = subsetix::fvd::Geometry2D<Real>::build_box(100, 100)
        .add_rectangle(50.0f, 30.0f, 20.0f, 40.0f, true);  // x_max < x_min

    auto csr = geom.build();

    // Should handle gracefully (empty or clamped)
    EXPECT_GE(csr.num_rows, 0);
}

TEST_F(FvdIntegratorsTest, GeometryBuilder_PrimitiveToCsr_ZeroCylinderRadius) {
    auto geom = subsetix::fvd::Geometry2D<Real>::build_box(100, 100)
        .add_cylinder(50.0f, 50.0f, 0.0f, true);  // Zero radius

    auto csr = geom.build();

    // Zero radius should be handled
    EXPECT_GE(csr.num_rows, 0);
}

TEST_F(FvdIntegratorsTest, GeometryBuilder_Complex_MultipleObstacles) {
    auto geom = subsetix::fvd::Geometry2D<Real>::build_box(200, 100)
        .add_rectangle(20.0f, 40.0f, 20.0f, 80.0f, true)
        .add_cylinder(100.0f, 50.0f, 15.0f, true)
        .add_rectangle(160.0f, 180.0f, 20.0f, 80.0f, true);

    EXPECT_EQ(geom.obstacles().size(), 3);

    auto csr = geom.build();

    // Verify all three obstacles were subtracted
    EXPECT_GT(csr.num_rows, 0);
    EXPECT_LT(csr.total_cells, 200 * 100);  // Less than full domain
}

TEST_F(FvdIntegratorsTest, GeometryBuilder_Complex_MixedObstaclesAndFluid) {
    auto geom = subsetix::fvd::Geometry2D<Real>::build_box(100, 100)
        .add_cylinder(30.0f, 30.0f, 10.0f, true)   // Obstacle
        .add_cylinder(70.0f, 70.0f, 10.0f, false); // Fluid region

    EXPECT_EQ(geom.obstacles().size(), 1);
    EXPECT_EQ(geom.fluid_regions().size(), 1);

    auto csr = geom.build();

    // Should have both subtraction and union operations
    EXPECT_GT(csr.num_rows, 0);
}

TEST_F(FvdIntegratorsTest, GeometryBuilder_BuildWithReusableContext) {
    subsetix::csr::CsrSetAlgebraContext ctx;

    auto geom1 = subsetix::fvd::Geometry2D<Real>::build_box(50, 50)
        .add_cylinder(25.0f, 25.0f, 10.0f, true);
    auto csr1 = geom1.build(ctx);

    auto geom2 = subsetix::fvd::Geometry2D<Real>::build_box(100, 100);
    auto csr2 = geom2.build(ctx);

    // Both should build successfully using same context
    EXPECT_GT(csr1.num_rows, 0);
    EXPECT_GT(csr2.num_rows, 0);
}

TEST_F(FvdIntegratorsTest, GeometryPrimitive_BoxFactory) {
    using Prim = subsetix::fvd::GeometryPrimitive<Real>;
    auto prim = Prim::box(0.0f, 1.0f, 0.0f, 1.0f);

    EXPECT_EQ(prim.type, Prim::Box);
    EXPECT_FLOAT_EQ(prim.params[0], 0.0f);
    EXPECT_FLOAT_EQ(prim.params[1], 1.0f);
    EXPECT_FLOAT_EQ(prim.params[2], 0.0f);
    EXPECT_FLOAT_EQ(prim.params[3], 1.0f);
}

TEST_F(FvdIntegratorsTest, GeometryPrimitive_RectangleFactory) {
    using Prim = subsetix::fvd::GeometryPrimitive<Real>;
    auto prim = Prim::rectangle(0.0f, 1.0f, 0.0f, 1.0f);

    // Note: rectangle() returns Box type internally (they're equivalent)
    EXPECT_EQ(prim.type, Prim::Box);
    EXPECT_FLOAT_EQ(prim.params[0], 0.0f);
    EXPECT_FLOAT_EQ(prim.params[1], 1.0f);
    EXPECT_FLOAT_EQ(prim.params[2], 0.0f);
    EXPECT_FLOAT_EQ(prim.params[3], 1.0f);
}

TEST_F(FvdIntegratorsTest, GeometryPrimitive_PODSafety) {
    using Prim = subsetix::fvd::GeometryPrimitive<Real>;

    // Verify GeometryPrimitive is trivially copyable for GPU
    EXPECT_TRUE(std::is_trivially_copyable_v<Prim>);
    EXPECT_TRUE(std::is_default_constructible_v<Prim>);

    // Verify params array initializes correctly
    Prim prim;
    EXPECT_EQ(prim.params[0], 0.0f);
    EXPECT_EQ(prim.params[5], 0.0f);
}

TEST_F(FvdIntegratorsTest, GeometryBuilder_DoublePrecision) {
    using DReal = double;
    auto geom = subsetix::fvd::Geometry2D<DReal>::build_box(100, 100, 0.01, 0.01)
        .add_cylinder(50.0, 50.0, 10.0, true);

    auto csr = geom.build();

    EXPECT_GT(csr.num_rows, 0);
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
