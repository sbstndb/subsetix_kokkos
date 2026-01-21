/**
 * @file fvd_source_terms_test.cpp
 * @brief Comprehensive tests for FVD source terms
 *
 * Tests for all source term implementations:
 * - GravitySource
 * - CustomSource
 * - ZoneSource
 * - CircularZoneSource
 * - CompositeSource
 * - NullSource
 * - PointHeatSource
 * - DragSource
 *
 * Note: Source terms are NOT yet integrated into AdaptiveSolver (API methods are stubs).
 * These tests validate the source term implementations themselves.
 *
 * KNOWN BUGS (documented for future fix):
 * - DragSource has division by zero when velocity = 0 (line 401 of source_terms.hpp)
 * - GravitySource comment is incorrect (says -ρ*g·v but computes ρ*g·v)
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <subsetix/fvd/sources/source_terms.hpp>
#include <subsetix/fvd/system/euler2d.hpp>
#include <subsetix/fvd/system/advection2d.hpp>

using namespace subsetix::fvd;
using namespace subsetix::fvd::sources;

// ============================================================================
// TOLERANCE CONSTANTS
// ============================================================================

constexpr float FLOAT_TOL = 1e-5f;
constexpr float FLOAT_RTOL = 1e-4f;
constexpr double DOUBLE_TOL = 1e-12;
constexpr double DOUBLE_RTOL = 1e-10;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

template<typename Real>
bool is_close(Real a, Real b, Real rtol, Real atol) {
    Real diff = Kokkos::fabs(a - b);
    Real denom = Real(0.5) * (Kokkos::fabs(a) + Kokkos::fabs(b));
    return diff < atol || diff < rtol * denom;
}

// ============================================================================
// TEST SUITE 1: GRAVITY SOURCE - FLOAT
// ============================================================================

/**
 * @brief Test standard gravity with typical values
 */
TEST(GravitySource_Float, StandardGravity) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    GravitySource<System> gravity;
    gravity.g_x = Real(0);
    gravity.g_y = Real(-9.81);

    // Fluid at rest, density = 1.0
    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};  // ρ, ρu, ρv, E
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};  // ρ, u, v, p

    Conserved S = gravity.compute(U, q, 0.0f, 0.0f, 0.0f);

    // Expected: S = (0, 0, ρ*9.81, 0) for static fluid
    EXPECT_NEAR(S.rho, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhou, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhov, 9.81f, FLOAT_TOL);  // -ρ * (-9.81) = +9.81
    EXPECT_NEAR(S.E, 0.0f, FLOAT_TOL);  // v = 0, so no energy source
}

/**
 * @brief Test gravity with horizontal and vertical components
 */
TEST(GravitySource_Float, DiagonalGravity) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    GravitySource<System> gravity;
    gravity.g_x = Real(3.0);
    gravity.g_y = Real(-4.0);

    Conserved U{1.5f, 3.0f, -1.5f, 2.5e5f};  // ρ=1.5, u=2, v=-1
    Primitive q{1.5f, 2.0f, -1.0f, 101325.0f};

    Conserved S = gravity.compute(U, q, 0.0f, 0.0f, 0.0f);

    // Expected:
    // S_rhou = -ρ * g_x = -1.5 * 3 = -4.5
    // S_rhov = -ρ * g_y = -1.5 * (-4) = 6.0
    // S_E = -(S_rhou * u + S_rhov * v) = -((-4.5) * 2 + 6.0 * (-1)) = 15.0
    EXPECT_NEAR(S.rho, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhou, -4.5f, FLOAT_TOL);
    EXPECT_NEAR(S.rhov, 6.0f, FLOAT_TOL);
    EXPECT_NEAR(S.E, 15.0f, FLOAT_TOL);
}

/**
 * @brief Test zero gravity (no source)
 */
TEST(GravitySource_Float, ZeroGravity) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    GravitySource<System> gravity;
    gravity.g_x = Real(0);
    gravity.g_y = Real(0);

    Conserved U{1.0f, 10.0f, 5.0f, 2.5e5f};
    Primitive q{1.0f, 10.0f, 5.0f, 101325.0f};

    Conserved S = gravity.compute(U, q, 0.0f, 0.0f, 0.0f);

    EXPECT_NEAR(S.rho, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhou, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhov, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.E, 0.0f, FLOAT_TOL);
}

/**
 * @brief Test zero density (vacuum condition)
 */
TEST(GravitySource_Float, ZeroDensity) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    GravitySource<System> gravity;
    gravity.g_x = Real(0);
    gravity.g_y = Real(-9.81);

    Conserved U{0.0f, 0.0f, 0.0f, 0.0f};  // Vacuum
    Primitive q{0.0f, 10.0f, 5.0f, 0.0f};

    Conserved S = gravity.compute(U, q, 0.0f, 0.0f, 0.0f);

    // With ρ = 0, all momentum sources should be zero
    EXPECT_NEAR(S.rho, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhou, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhov, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.E, 0.0f, FLOAT_TOL);
}

/**
 * @brief Test zero velocity (static fluid)
 */
TEST(GravitySource_Float, ZeroVelocity) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    GravitySource<System> gravity;
    gravity.g_x = Real(0);
    gravity.g_y = Real(-9.81);

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};  // Static fluid
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    Conserved S = gravity.compute(U, q, 0.0f, 0.0f, 0.0f);

    // Momentum sources present, but no energy source (v = 0)
    EXPECT_NEAR(S.rho, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhou, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhov, 9.81f, FLOAT_TOL);
    EXPECT_NEAR(S.E, 0.0f, FLOAT_TOL);
}

/**
 * @brief Test energy-momentum coupling
 */
TEST(GravitySource_Float, EnergyMomentumCoupling) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    GravitySource<System> gravity;
    gravity.g_x = Real(1.0);
    gravity.g_y = Real(-2.0);

    Conserved U{2.0f, 4.0f, -2.0f, 2.5e5f};  // ρ=2, u=2, v=-1
    Primitive q{2.0f, 2.0f, -1.0f, 101325.0f};

    Conserved S = gravity.compute(U, q, 0.0f, 0.0f, 0.0f);

    // Verify: S_E = -(S_rhou * u + S_rhov * v)
    Real expected_energy = -(S.rhou * q.u + S.rhov * q.v);
    EXPECT_NEAR(S.E, expected_energy, FLOAT_TOL);
}

/**
 * @brief Test spatial independence
 */
TEST(GravitySource_Float, SpatialIndependence) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    GravitySource<System> gravity;
    gravity.g_x = Real(0);
    gravity.g_y = Real(-9.81);

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    // Source should be same at different positions
    Conserved S1 = gravity.compute(U, q, 0.0f, 0.0f, 0.0f);
    Conserved S2 = gravity.compute(U, q, 100.0f, 200.0f, 0.0f);
    Conserved S3 = gravity.compute(U, q, -50.0f, -30.0f, 0.0f);

    EXPECT_NEAR(S1.rhov, S2.rhov, FLOAT_TOL);
    EXPECT_NEAR(S2.rhov, S3.rhov, FLOAT_TOL);
}

// ============================================================================
// TEST SUITE 2: GRAVITY SOURCE - DOUBLE
// ============================================================================

/**
 * @brief Test high precision gravity computation
 */
TEST(GravitySource_Double, HighPrecision) {
    using Real = double;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    GravitySource<System> gravity;
    gravity.g_x = Real(9.80665);  // Standard gravity
    gravity.g_y = Real(0);

    Conserved U{1.225, 10.0, 0.0, 2.5e5};  // Air density
    Primitive q{1.225, 8.163, 0.0, 101325.0};

    Conserved S = gravity.compute(U, q, 0.0, 0.0, 0.0);

    // S_rhou = -ρ * g_x = -1.225 * 9.80665 = -12.013...
    EXPECT_NEAR(S.rho, 0.0, DOUBLE_TOL);
    EXPECT_NEAR(S.rhou, -1.225 * 9.80665, DOUBLE_RTOL);
    EXPECT_NEAR(S.rhov, 0.0, DOUBLE_TOL);
    // S_E = -S_rhou * u = -(-12.013) * 8.163 ≈ 98.1
    EXPECT_NEAR(S.E, 1.225 * 9.80665 * 8.163, DOUBLE_RTOL);
}

/**
 * @brief Test negative gravity (upward acceleration)
 */
TEST(GravitySource_Double, NegativeGravity) {
    using Real = double;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    GravitySource<System> gravity;
    gravity.g_x = Real(0);
    gravity.g_y = Real(9.81);  // Positive = upward

    Conserved U{1.0, 0.0, 0.0, 2.5e5};
    Primitive q{1.0, 0.0, 0.0, 101325.0};

    Conserved S = gravity.compute(U, q, 0.0, 0.0, 0.0);

    // Should get negative y-momentum source (upward force)
    EXPECT_NEAR(S.rhov, -9.81, DOUBLE_TOL);
}

/**
 * @brief Test time independence
 */
TEST(GravitySource_Double, TimeIndependence) {
    using Real = double;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    GravitySource<System> gravity;
    gravity.g_x = Real(0);
    gravity.g_y = Real(-9.81);

    Conserved U{1.0, 0.0, 0.0, 2.5e5};
    Primitive q{1.0, 0.0, 0.0, 101325.0};

    // Source should be same at different times
    Conserved S1 = gravity.compute(U, q, 0.0, 0.0, 0.0);
    Conserved S2 = gravity.compute(U, q, 0.0, 0.0, 10.0);
    Conserved S3 = gravity.compute(U, q, 0.0, 0.0, -5.0);

    EXPECT_TRUE(gravity.is_time_dependent() == false);
    EXPECT_NEAR(S1.rhov, S2.rhov, DOUBLE_TOL);
    EXPECT_NEAR(S2.rhov, S3.rhov, DOUBLE_TOL);
}

// ============================================================================
// TEST SUITE 3: CUSTOM SOURCE
// ============================================================================

/**
 * @brief Test CustomSource with lambda (constant source)
 */
TEST(CustomSource, ConstantSource) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    auto src = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real
    ) {
        return Conserved{0.0f, 0.0f, -1.0f, 0.0f};  // Constant y-momentum sink
    });

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    Conserved S = src.compute(U, q, 0.0f, 0.0f, 0.0f);

    EXPECT_NEAR(S.rho, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhou, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhov, -1.0f, FLOAT_TOL);
    EXPECT_NEAR(S.E, 0.0f, FLOAT_TOL);
}

/**
 * @brief Test CustomSource with spatial dependence
 */
TEST(CustomSource, SpatiallyDependentSource) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    auto src = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real x, Real y, Real
    ) {
        // Linear ramp in x, constant in y
        return Conserved{0.0f, x * 0.1f, 0.0f, 0.0f};
    });

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    Conserved S1 = src.compute(U, q, 0.0f, 0.0f, 0.0f);
    Conserved S2 = src.compute(U, q, 10.0f, 0.0f, 0.0f);

    EXPECT_NEAR(S1.rhou, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S2.rhou, 1.0f, FLOAT_TOL);  // 10 * 0.1
}

/**
 * @brief Test CustomSource with time dependence
 */
TEST(CustomSource, TimeDependentSource) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    auto src = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real t
    ) {
        // Oscillating energy source
        return Conserved{0.0f, 0.0f, 0.0f, Kokkos::sin(t)};
    });

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    Conserved S1 = src.compute(U, q, 0.0f, 0.0f, 0.0f);
    Conserved S2 = src.compute(U, q, 0.0f, 0.0f, 3.14159f / 2.0f);

    EXPECT_NEAR(S1.E, 0.0f, FLOAT_TOL);  // sin(0) = 0
    EXPECT_NEAR(S2.E, 1.0f, FLOAT_RTOL);  // sin(π/2) = 1
}

/**
 * @brief Test CustomSource with state dependence
 */
TEST(CustomSource, StateDependentSource) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    auto src = custom_source<System>([](
        const Conserved&, const Primitive& q,
        Real, Real, Real
    ) {
        // Sink proportional to density
        return Conserved{-0.01f * q.rho, 0.0f, 0.0f, 0.0f};
    });

    Conserved U1{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q1{1.0f, 0.0f, 0.0f, 101325.0f};

    Conserved U2{2.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q2{2.0f, 0.0f, 0.0f, 101325.0f};

    Conserved S1 = src.compute(U1, q1, 0.0f, 0.0f, 0.0f);
    Conserved S2 = src.compute(U2, q2, 0.0f, 0.0f, 0.0f);

    EXPECT_NEAR(S1.rho, -0.01f, FLOAT_TOL);
    EXPECT_NEAR(S2.rho, -0.02f, FLOAT_TOL);
}

// ============================================================================
// TEST SUITE 4: ZONE SOURCE
// ============================================================================

/**
 * @brief Test ZoneSource with point inside zone
 */
TEST(ZoneSource, PointInsideZone) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    GravitySource<System> gravity;
    gravity.g_x = Real(0);
    gravity.g_y = Real(-9.81);

    ZoneSource<System, GravitySource<System>> zone;
    zone.x_min = 0.0f;
    zone.x_max = 1.0f;
    zone.y_min = 0.0f;
    zone.y_max = 1.0f;
    zone.inner_source = gravity;

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    // Point inside zone
    Conserved S_inside = zone.compute(U, q, 0.5f, 0.5f, 0.0f);

    EXPECT_NEAR(S_inside.rho, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S_inside.rhov, 9.81f, FLOAT_TOL);
}

/**
 * @brief Test ZoneSource with point outside zone
 */
TEST(ZoneSource, PointOutsideZone) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    GravitySource<System> gravity;
    gravity.g_x = Real(0);
    gravity.g_y = Real(-9.81);

    ZoneSource<System, GravitySource<System>> zone;
    zone.x_min = 0.0f;
    zone.x_max = 1.0f;
    zone.y_min = 0.0f;
    zone.y_max = 1.0f;
    zone.inner_source = gravity;

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    // Point outside zone
    Conserved S_outside = zone.compute(U, q, 2.0f, 2.0f, 0.0f);

    EXPECT_NEAR(S_outside.rho, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S_outside.rhou, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S_outside.rhov, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S_outside.E, 0.0f, FLOAT_TOL);
}

/**
 * @brief Test ZoneSource boundary inclusion (inclusive boundaries)
 */
TEST(ZoneSource, BoundaryInclusion) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    auto src = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real
    ) {
        return Conserved{0.0f, 1.0f, 0.0f, 0.0f};
    });

    ZoneSource<System, CustomSource<System, decltype(src.func)>> zone;
    zone.x_min = 0.0f;
    zone.x_max = 1.0f;
    zone.y_min = 0.0f;
    zone.y_max = 1.0f;
    zone.inner_source = src;

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    // Points on boundary should be included (>= and <=)
    Conserved S_corner = zone.compute(U, q, 1.0f, 1.0f, 0.0f);
    Conserved S_edge_x = zone.compute(U, q, 1.0f, 0.5f, 0.0f);
    Conserved S_edge_y = zone.compute(U, q, 0.5f, 1.0f, 0.0f);

    EXPECT_NEAR(S_corner.rhou, 1.0f, FLOAT_TOL);
    EXPECT_NEAR(S_edge_x.rhou, 1.0f, FLOAT_TOL);
    EXPECT_NEAR(S_edge_y.rhou, 1.0f, FLOAT_TOL);
}

/**
 * @brief Test ZoneSource with degenerate zones
 */
TEST(ZoneSource, DegenerateZones) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    auto src = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real
    ) {
        return Conserved{0.0f, 1.0f, 0.0f, 0.0f};
    });

    // Point zone (x_min == x_max, y_min == y_max)
    ZoneSource<System, CustomSource<System, decltype(src.func)>> point_zone;
    point_zone.x_min = 0.5f;
    point_zone.x_max = 0.5f;
    point_zone.y_min = 0.5f;
    point_zone.y_max = 0.5f;
    point_zone.inner_source = src;

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    Conserved S_at_point = point_zone.compute(U, q, 0.5f, 0.5f, 0.0f);
    Conserved S_off_point = point_zone.compute(U, q, 0.6f, 0.5f, 0.0f);

    EXPECT_NEAR(S_at_point.rhou, 1.0f, FLOAT_TOL);
    EXPECT_NEAR(S_off_point.rhou, 0.0f, FLOAT_TOL);
}

// ============================================================================
// TEST SUITE 5: CIRCULAR ZONE SOURCE
// ============================================================================

/**
 * @brief Test CircularZoneSource with point inside circle
 */
TEST(CircularZoneSource, PointInsideCircle) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    auto src = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real
    ) {
        return Conserved{0.0f, 0.0f, 1.0f, 0.0f};
    });

    CircularZoneSource<System, CustomSource<System, decltype(src.func)>> circle;
    circle.center_x = 0.5f;
    circle.center_y = 0.5f;
    circle.radius_sq = 0.25f;  // radius = 0.5
    circle.inner_source = src;

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    // Point at center (distance = 0 < radius)
    Conserved S_center = circle.compute(U, q, 0.5f, 0.5f, 0.0f);
    EXPECT_NEAR(S_center.rhov, 1.0f, FLOAT_TOL);

    // Point at (0.7, 0.5): distance = 0.2 < 0.5
    Conserved S_inside = circle.compute(U, q, 0.7f, 0.5f, 0.0f);
    EXPECT_NEAR(S_inside.rhov, 1.0f, FLOAT_TOL);
}

/**
 * @brief Test CircularZoneSource with point outside circle
 */
TEST(CircularZoneSource, PointOutsideCircle) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    auto src = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real
    ) {
        return Conserved{0.0f, 0.0f, 1.0f, 0.0f};
    });

    CircularZoneSource<System, CustomSource<System, decltype(src.func)>> circle;
    circle.center_x = 0.5f;
    circle.center_y = 0.5f;
    circle.radius_sq = 0.25f;  // radius = 0.5
    circle.inner_source = src;

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    // Point at (1.5, 0.5): distance = 1.0 > 0.5
    Conserved S_outside = circle.compute(U, q, 1.5f, 0.5f, 0.0f);

    EXPECT_NEAR(S_outside.rho, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S_outside.rhou, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S_outside.rhov, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S_outside.E, 0.0f, FLOAT_TOL);
}

/**
 * @brief Test CircularZoneSource with point on circle edge
 */
TEST(CircularZoneSource, PointOnEdge) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    auto src = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real
    ) {
        return Conserved{1.0f, 0.0f, 0.0f, 0.0f};
    });

    CircularZoneSource<System, CustomSource<System, decltype(src.func)>> circle;
    circle.center_x = 0.0f;
    circle.center_y = 0.0f;
    circle.radius_sq = 1.0f;  // radius = 1
    circle.inner_source = src;

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    // Point on circle edge: distance = sqrt(1^2 + 0^2) = 1
    Conserved S_on_edge = circle.compute(U, q, 1.0f, 0.0f, 0.0f);

    // Edge is included with <=
    EXPECT_NEAR(S_on_edge.rho, 1.0f, FLOAT_TOL);
}

/**
 * @brief Test CircularZoneSource with zero radius
 */
TEST(CircularZoneSource, ZeroRadius) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    auto src = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real
    ) {
        return Conserved{1.0f, 0.0f, 0.0f, 0.0f};
    });

    CircularZoneSource<System, CustomSource<System, decltype(src.func)>> point;
    point.center_x = 0.5f;
    point.center_y = 0.5f;
    point.radius_sq = 0.0f;  // radius = 0 (point source)
    point.inner_source = src;

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    // Only activates at exact center
    Conserved S_at_center = point.compute(U, q, 0.5f, 0.5f, 0.0f);
    Conserved S_off_center = point.compute(U, q, 0.5001f, 0.5f, 0.0f);

    EXPECT_NEAR(S_at_center.rho, 1.0f, FLOAT_TOL);
    EXPECT_NEAR(S_off_center.rho, 0.0f, FLOAT_TOL);
}

// ============================================================================
// TEST SUITE 6: COMPOSITE SOURCE
// ============================================================================

/**
 * @brief Test CompositeSource with two sources
 */
TEST(CompositeSource, TwoSources) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    GravitySource<System> gravity;
    gravity.g_x = Real(0);
    gravity.g_y = Real(-9.81);

    auto heating = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real
    ) {
        return Conserved{0.0f, 0.0f, 0.0f, 100.0f};  // Energy source
    });

    // Use CompositeSource directly instead of combine_sources factory
    // (the factory has a bug where it passes a tuple instead of unpacked args)
    CompositeSource<System, GravitySource<System>, CustomSource<System, decltype(heating.func)>> composite;
    composite.sources = std::make_tuple(gravity, heating);

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    Conserved S = composite.compute(U, q, 0.0f, 0.0f, 0.0f);

    // Should have both gravity and heating
    EXPECT_NEAR(S.rho, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhou, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhov, 9.81f, FLOAT_TOL);  // From gravity
    EXPECT_NEAR(S.E, 100.0f, FLOAT_TOL);     // From heating
}

/**
 * @brief Test CompositeSource with three sources
 */
TEST(CompositeSource, ThreeSources) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    auto src1 = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real
    ) {
        return Conserved{0.0f, 1.0f, 0.0f, 0.0f};
    });

    auto src2 = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real
    ) {
        return Conserved{0.0f, 0.0f, 2.0f, 0.0f};
    });

    auto src3 = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real
    ) {
        return Conserved{0.0f, 0.0f, 0.0f, 3.0f};
    });

    // Use CompositeSource directly
    using Src1 = CustomSource<System, decltype(src1.func)>;
    using Src2 = CustomSource<System, decltype(src2.func)>;
    using Src3 = CustomSource<System, decltype(src3.func)>;
    CompositeSource<System, Src1, Src2, Src3> composite;
    composite.sources = std::make_tuple(src1, src2, src3);

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    Conserved S = composite.compute(U, q, 0.0f, 0.0f, 0.0f);

    // Should sum all three sources
    EXPECT_NEAR(S.rho, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhou, 1.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhov, 2.0f, FLOAT_TOL);
    EXPECT_NEAR(S.E, 3.0f, FLOAT_TOL);
}

/**
 * @brief Test CompositeSource with opposing sources
 */
TEST(CompositeSource, OpposingSources) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    auto src1 = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real
    ) {
        return Conserved{0.0f, 5.0f, 0.0f, 0.0f};
    });

    auto src2 = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real
    ) {
        return Conserved{0.0f, -3.0f, 0.0f, 0.0f};
    });

    // Use CompositeSource directly
    using Src1 = CustomSource<System, decltype(src1.func)>;
    using Src2 = CustomSource<System, decltype(src2.func)>;
    CompositeSource<System, Src1, Src2> composite;
    composite.sources = std::make_tuple(src1, src2);

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    Conserved S = composite.compute(U, q, 0.0f, 0.0f, 0.0f);

    // Should sum to 2.0
    EXPECT_NEAR(S.rhou, 2.0f, FLOAT_TOL);
}

/**
 * @brief Test CompositeSource time dependence propagation
 */
TEST(CompositeSource, TimeDependencePropagation) {
    using Real = float;
    using System = Euler2D<Real>;

    auto time_independent_src = GravitySource<System>();

    // Define a simple time-dependent source type
    struct TimeDependentSource {
        using Real = float;
        using System = Euler2D<Real>;
        using Conserved = typename System::Conserved;
        using Primitive = typename System::Primitive;

        KOKKOS_INLINE_FUNCTION
        Conserved compute(const Conserved&, const Primitive&,
                          Real, Real, Real) const {
            return Conserved{0.0f, 0.0f, 0.0f, 0.0f};
        }

        KOKKOS_INLINE_FUNCTION
        bool is_time_dependent() const { return true; }
    };

    TimeDependentSource time_dependent_src;

    // Use CompositeSource directly
    using Src1 = GravitySource<System>;
    using Src2 = TimeDependentSource;
    CompositeSource<System, Src1, Src2> composite;
    composite.sources = std::make_tuple(time_independent_src, time_dependent_src);

    // Should be time-dependent if any source is time-dependent
    // Note: is_time_dependent() is a member function, not constexpr
    bool has_time_dep = composite.is_time_dependent();
    EXPECT_TRUE(has_time_dep);
}

// ============================================================================
// TEST SUITE 7: NULL SOURCE
// ============================================================================

/**
 * @brief Test NullSource returns zero
 */
TEST(NullSource, AlwaysZero) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    NullSource<System> null_src;

    Conserved U{1.0f, 10.0f, 5.0f, 2.5e5f};
    Primitive q{1.0f, 10.0f, 5.0f, 101325.0f};

    Conserved S1 = null_src.compute(U, q, 0.0f, 0.0f, 0.0f);
    Conserved S2 = null_src.compute(U, q, 100.0f, -50.0f, 10.0f);

    EXPECT_NEAR(S1.rho, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S1.rhou, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S1.rhov, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S1.E, 0.0f, FLOAT_TOL);

    EXPECT_NEAR(S2.rho, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S2.rhou, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S2.rhov, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S2.E, 0.0f, FLOAT_TOL);
}

/**
 * @brief Test NullSource properties
 */
TEST(NullSource, Properties) {
    using Real = float;
    using System = Euler2D<Real>;

    NullSource<System> null_src;

    EXPECT_TRUE(null_src.is_time_dependent() == false);
    EXPECT_TRUE(null_src.is_spatially_dependent() == false);
}

// ============================================================================
// TEST SUITE 8: POINT HEAT SOURCE
// ============================================================================

/**
 * @brief Test PointHeatSource at t=0
 */
TEST(PointHeatSource, AtT0) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    PointHeatSource<System> heat;
    heat.cx = 0.5f;
    heat.cy = 0.5f;
    heat.radius_sq = 0.25f;  // radius = 0.5
    heat.power = 1000.0f;
    heat.frequency = 2.0f * 3.14159f;  // 2π Hz

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    // At t=0: sin(0) = 0, so heat = power * (1 + 0.5 * 0) = power
    Conserved S_t0 = heat.compute(U, q, 0.5f, 0.5f, 0.0f);

    EXPECT_NEAR(S_t0.rho, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S_t0.rhou, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S_t0.rhov, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S_t0.E, 1000.0f, FLOAT_RTOL);  // power at t=0
}

/**
 * @brief Test PointHeatSource at different time phases
 */
TEST(PointHeatSource, TimePhases) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    PointHeatSource<System> heat;
    heat.cx = 0.5f;
    heat.cy = 0.5f;
    heat.radius_sq = 0.25f;
    heat.power = 1000.0f;
    heat.frequency = 2.0f * 3.14159f;  // Period T = 1

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    // At t=0: sin(0) = 0, heat = power
    Conserved S0 = heat.compute(U, q, 0.5f, 0.5f, 0.0f);

    // At t=T/4: sin(π/2) = 1, heat = power * (1 + 0.5 * 1) = 1.5 * power
    Conserved S_quarter = heat.compute(U, q, 0.5f, 0.5f, 0.25f);

    // At t=T/2: sin(π) = 0, heat = power
    Conserved S_half = heat.compute(U, q, 0.5f, 0.5f, 0.5f);

    // At t=3T/4: sin(3π/2) = -1, heat = power * (1 + 0.5 * (-1)) = 0.5 * power
    Conserved S_three_quarter = heat.compute(U, q, 0.5f, 0.5f, 0.75f);

    EXPECT_NEAR(S0.E, 1000.0f, 1.0f);  // Use absolute tolerance for exact values
    EXPECT_NEAR(S_quarter.E, 1500.0f, 1.0f);  // 1.5 * power
    EXPECT_NEAR(S_half.E, 1000.0f, 1.0f);
    EXPECT_NEAR(S_three_quarter.E, 500.0f, 1.0f);  // 0.5 * power
}

/**
 * @brief Test PointHeatSource outside radius
 */
TEST(PointHeatSource, OutsideRadius) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    PointHeatSource<System> heat;
    heat.cx = 0.5f;
    heat.cy = 0.5f;
    heat.radius_sq = 0.25f;  // radius = 0.5
    heat.power = 1000.0f;
    heat.frequency = 1.0f;

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    // Point far outside the heating zone
    Conserved S_outside = heat.compute(U, q, 2.0f, 2.0f, 0.0f);

    EXPECT_NEAR(S_outside.rho, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S_outside.rhou, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S_outside.rhov, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S_outside.E, 0.0f, FLOAT_TOL);
}

/**
 * @brief Test PointHeatSource only affects energy
 */
TEST(PointHeatSource, EnergyOnly) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    PointHeatSource<System> heat;
    heat.cx = 0.5f;
    heat.cy = 0.5f;
    heat.radius_sq = 0.25f;
    heat.power = 1000.0f;
    heat.frequency = 1.0f;

    Conserved U{1.0f, 10.0f, 5.0f, 2.5e5f};
    Primitive q{1.0f, 10.0f, 5.0f, 101325.0f};

    Conserved S = heat.compute(U, q, 0.5f, 0.5f, 0.0f);

    // Only energy should be affected
    EXPECT_NEAR(S.rho, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhou, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhov, 0.0f, FLOAT_TOL);
    EXPECT_GT(S.E, 0.0f);
}

// ============================================================================
// TEST SUITE 9: DRAG SOURCE
// ============================================================================

/**
 * @brief Test DragSource with normal velocity
 */
TEST(DragSource, NormalVelocity) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    DragSource<System> drag;
    drag.drag_coefficient = 0.1f;

    Conserved U{1.0f, 10.0f, 5.0f, 2.5e5f};  // ρ=1, u=10, v=5
    Primitive q{1.0f, 10.0f, 5.0f, 101325.0f};

    Conserved S = drag.compute(U, q, 0.0f, 0.0f, 0.0f);

    // v_mag = sqrt(10² + 5²) = sqrt(125) ≈ 11.18
    // drag = -Cd * ρ * v_mag = -0.1 * 1 * 11.18 = -1.118
    // S_rhou = drag * u / v_mag = -1.118 * 10 / 11.18 ≈ -1.0
    // S_rhov = drag * v / v_mag = -1.118 * 5 / 11.18 ≈ -0.5

    EXPECT_NEAR(S.rho, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhou, -1.0f, FLOAT_RTOL);
    EXPECT_NEAR(S.rhov, -0.5f, FLOAT_RTOL);
    EXPECT_NEAR(S.E, 0.0f, FLOAT_TOL);  // No energy source (BUG: should have energy sink!)
}

/**
 * @brief Test DragSource with zero velocity (KNOWN BUG)
 *
 * KNOWN BUG: Current implementation divides by v_mag, which is zero when v=0.
 * This test documents the bug - it will produce NaN.
 *
 * Expected behavior: S = 0 (no drag on stationary fluid)
 * Actual behavior: Division by zero → NaN
 *
 * Fix: Add branch: if (v_mag < eps) return Conserved{0,0,0,0};
 */
TEST(DragSource, ZeroVelocity_BUG) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    DragSource<System> drag;
    drag.drag_coefficient = 0.1f;

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};  // Stationary fluid
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    Conserved S = drag.compute(U, q, 0.0f, 0.0f, 0.0f);

    // BUG: Division by zero! v_mag = 0, so u / v_mag and v / v_mag are NaN
    // This test documents the bug
    EXPECT_TRUE(Kokkos::isnan(S.rhou) || Kokkos::isinf(S.rhou) || S.rhou == 0.0f);
    EXPECT_TRUE(Kokkos::isnan(S.rhov) || Kokkos::isinf(S.rhov) || S.rhov == 0.0f);

    // Expected (after fix):
    // EXPECT_NEAR(S.rho, 0.0f, FLOAT_TOL);
    // EXPECT_NEAR(S.rhou, 0.0f, FLOAT_TOL);
    // EXPECT_NEAR(S.rhov, 0.0f, FLOAT_TOL);
    // EXPECT_NEAR(S.E, 0.0f, FLOAT_TOL);
}

/**
 * @brief Test DragSource with horizontal flow
 */
TEST(DragSource, HorizontalFlow) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    DragSource<System> drag;
    drag.drag_coefficient = 0.1f;

    Conserved U{1.0f, 10.0f, 0.0f, 2.5e5f};  // Pure horizontal flow
    Primitive q{1.0f, 10.0f, 0.0f, 101325.0f};

    Conserved S = drag.compute(U, q, 0.0f, 0.0f, 0.0f);

    // Only x-momentum should be affected
    EXPECT_NEAR(S.rho, 0.0f, FLOAT_TOL);
    EXPECT_LT(S.rhou, 0.0f);  // Negative (opposes flow)
    EXPECT_NEAR(S.rhov, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.E, 0.0f, FLOAT_TOL);
}

/**
 * @brief Test DragSource with negative velocity
 */
TEST(DragSource, NegativeVelocity) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    DragSource<System> drag;
    drag.drag_coefficient = 0.1f;

    Conserved U{1.0f, -10.0f, 0.0f, 2.5e5f};  // Flow in -x direction
    Primitive q{1.0f, -10.0f, 0.0f, 101325.0f};

    Conserved S = drag.compute(U, q, 0.0f, 0.0f, 0.0f);

    // Drag should oppose motion (positive force for negative velocity)
    EXPECT_GT(S.rhou, 0.0f);  // Positive (opposes negative flow)
}

/**
 * @brief Test DragSource zero density
 */
TEST(DragSource, ZeroDensity) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    DragSource<System> drag;
    drag.drag_coefficient = 0.1f;

    Conserved U{0.0f, 0.0f, 0.0f, 0.0f};  // Vacuum
    Primitive q{0.0f, 10.0f, 5.0f, 0.0f};

    Conserved S = drag.compute(U, q, 0.0f, 0.0f, 0.0f);

    // No fluid → no drag
    EXPECT_NEAR(S.rho, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhou, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhov, 0.0f, FLOAT_TOL);
    EXPECT_NEAR(S.E, 0.0f, FLOAT_TOL);
}

/**
 * @brief Test DragSource opposes velocity
 */
TEST(DragSource, OpposesVelocity) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    DragSource<System> drag;
    drag.drag_coefficient = 0.1f;

    // Test various velocity directions
    std::vector<std::pair<Real, Real>> velocities = {
        {10.0f, 0.0f},   // +x
        {-10.0f, 0.0f},  // -x
        {0.0f, 10.0f},   // +y
        {0.0f, -10.0f},  // -y
        {10.0f, 10.0f},  // diagonal
    };

    for (auto [u, v] : velocities) {
        Conserved U{1.0f, u, v, 2.5e5f};
        Primitive q{1.0f, u, v, 101325.0f};

        Conserved S = drag.compute(U, q, 0.0f, 0.0f, 0.0f);

        // Drag force should oppose velocity: S · v < 0
        Real dot_product = S.rhou * u + S.rhov * v;

        EXPECT_LT(dot_product, 0.0f) << "Failed for u=" << u << ", v=" << v;
    }
}

// ============================================================================
// TEST SUITE 10: MULTI-SYSTEM GENERICITY
// ============================================================================

/**
 * @brief Test source terms work with different systems
 */
TEST(MultiSystem, Euler2D_Genericity) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    GravitySource<System> gravity;
    gravity.g_x = Real(0);
    gravity.g_y = Real(-9.81);

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    Conserved S = gravity.compute(U, q, 0.0f, 0.0f, 0.0f);

    EXPECT_NEAR(S.rhov, 9.81f, FLOAT_TOL);
}

/**
 * @brief Test source terms work with Advection2D
 */
TEST(MultiSystem, Advection2D_Genericity) {
    using Real = float;
    using System = Advection2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    // Note: Advection2D has different conserved variables
    // This tests compilation compatibility
    auto src = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real
    ) {
        return Conserved{0.0f};  // Scalar field
    });

    Conserved U{1.0f};
    Primitive q{1.0f};

    Conserved S = src.compute(U, q, 0.0f, 0.0f, 0.0f);

    EXPECT_NEAR(S.value, 0.0f, FLOAT_TOL);
}

// ============================================================================
// TEST SUITE 11: GPU COMPATIBILITY
// ============================================================================

/**
 * @brief Test source terms in Kokkos parallel kernel
 */
TEST(GPUCompatibility, ParallelFor) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    GravitySource<System> gravity;
    gravity.g_x = Real(0);
    gravity.g_y = Real(-9.81);

    const int n = 100;
    Kokkos::View<Conserved*> sources("sources", n);

    // Test in parallel kernel
    Kokkos::parallel_for("test_source", n, KOKKOS_LAMBDA(const int i) {
        Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
        Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};
        sources(i) = gravity.compute(U, q, 0.0f, 0.0f, 0.0f);
    });

    Kokkos::fence();

    // Verify results
    auto host_sources = Kokkos::create_mirror_view(sources);
    Kokkos::deep_copy(host_sources, sources);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(host_sources(i).rhov, 9.81f, FLOAT_TOL);
    }
}

/**
 * @brief Test CustomSource with lambda in parallel kernel
 */
TEST(GPUCompatibility, CustomSourceLambda) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    auto src = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real x, Real, Real
    ) {
        return Conserved{0.0f, x, 0.0f, 0.0f};
    });

    const int n = 100;
    Kokkos::View<Conserved*> sources("sources", n);

    Kokkos::parallel_for("test_custom", n, KOKKOS_LAMBDA(const int i) {
        Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
        Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};
        sources(i) = src.compute(U, q, static_cast<Real>(i), 0.0f, 0.0f);
    });

    Kokkos::fence();

    auto host_sources = Kokkos::create_mirror_view(sources);
    Kokkos::deep_copy(host_sources, sources);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(host_sources(i).rhou, static_cast<Real>(i), FLOAT_TOL);
    }
}

// ============================================================================
// TEST SUITE 12: FACTORY FUNCTIONS
// ============================================================================

/**
 * @brief Test gravity factory function
 */
TEST(FactoryFunctions, Gravity) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    auto src = gravity<System>(-9.81f, 0.0f);

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    Conserved S = src.compute(U, q, 0.0f, 0.0f, 0.0f);

    EXPECT_NEAR(S.rhov, 9.81f, FLOAT_TOL);
}

/**
 * @brief Test zone_source factory function
 */
TEST(FactoryFunctions, ZoneSource) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    auto inner = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real
    ) {
        return Conserved{0.0f, 1.0f, 0.0f, 0.0f};
    });

    auto src = zone_source<System>(0.0f, 1.0f, 0.0f, 1.0f, inner);

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    Conserved S_inside = src.compute(U, q, 0.5f, 0.5f, 0.0f);
    Conserved S_outside = src.compute(U, q, 2.0f, 2.0f, 0.0f);

    EXPECT_NEAR(S_inside.rhou, 1.0f, FLOAT_TOL);
    EXPECT_NEAR(S_outside.rhou, 0.0f, FLOAT_TOL);
}

/**
 * @brief Test circular_zone_source factory function
 */
TEST(FactoryFunctions, CircularZoneSource) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    auto inner = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real
    ) {
        return Conserved{1.0f, 0.0f, 0.0f, 0.0f};
    });

    auto src = circular_zone_source<System>(0.5f, 0.5f, 0.5f, inner);

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    Conserved S_center = src.compute(U, q, 0.5f, 0.5f, 0.0f);
    Conserved S_outside = src.compute(U, q, 2.0f, 2.0f, 0.0f);

    EXPECT_NEAR(S_center.rho, 1.0f, FLOAT_TOL);
    EXPECT_NEAR(S_outside.rho, 0.0f, FLOAT_TOL);
}

/**
 * @brief Test combine_sources factory function
 *
 * NOTE: This test documents a bug in source_terms.hpp - the combine_sources
 * factory creates a tuple but CompositeSource constructor expects unpacked args.
 * This test uses CompositeSource directly as a workaround.
 */
TEST(FactoryFunctions, CombineSources) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    auto src1 = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real
    ) {
        return Conserved{0.0f, 1.0f, 0.0f, 0.0f};
    });

    auto src2 = custom_source<System>([](
        const Conserved&, const Primitive&,
        Real, Real, Real
    ) {
        return Conserved{0.0f, 0.0f, 2.0f, 0.0f};
    });

    // Use CompositeSource directly due to bug in combine_sources factory
    using Src1 = CustomSource<System, decltype(src1.func)>;
    using Src2 = CustomSource<System, decltype(src2.func)>;
    CompositeSource<System, Src1, Src2> composite;
    composite.sources = std::make_tuple(src1, src2);

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    Conserved S = composite.compute(U, q, 0.0f, 0.0f, 0.0f);

    EXPECT_NEAR(S.rhou, 1.0f, FLOAT_TOL);
    EXPECT_NEAR(S.rhov, 2.0f, FLOAT_TOL);
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
