/**
 * @file fvd_refinement_criteria_test.cpp
 * @brief Tests for refinement criteria, focusing on SmoothnessCriterion
 *
 * Tests coverage for:
 * - SmoothnessCriterion (lines 489-607 in refinement_criteria.hpp)
 * - GradientCriterion
 * - VorticityCriterion
 * - ValueRangeCriterion
 * - CompositeCriterion
 *
 * Test patterns:
 * - Euler2D and Advection2D system genericity
 * - Float and double precision
 * - GPU parallel execution
 * - Edge cases and numerical stability
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "subsetix/fvd/system/euler2d.hpp"
#include "subsetix/fvd/system/advection2d.hpp"
#include "subsetix/fvd/amr/refinement_criteria.hpp"

using namespace subsetix::fvd;
using namespace subsetix::fvd::amr;

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
// SMOOTHNESS CRITERION TESTS - EULER2D FLOAT
// ============================================================================

TEST(SmoothnessCriterion_Euler2D_Float, DefaultConstruction) {
    using Real = float;
    using System = Euler2D<Real>;

    SmoothnessCriterion<System> crit;

    EXPECT_FLOAT_EQ(crit.coarsen_threshold, 0.05f);
    EXPECT_TRUE(crit.use_rho);
    EXPECT_TRUE(crit.use_p);
    EXPECT_FALSE(crit.use_u);
    EXPECT_FLOAT_EQ(crit.min_value, 0.0f);
}

TEST(SmoothnessCriterion_Euler2D_Float, AllSiblingsIdentical_Coarsens) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.01);

    // All 4 siblings have identical density
    Conserved siblings[4] = {
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f}
    };

    Conserved U_center{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q_center{1.0f, 0.0f, 0.0f, 101325.0f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    EXPECT_EQ(action, RefinementAction::Coarsen);
}

TEST(SmoothnessCriterion_Euler2D_Float, LowVariance_Coarsens) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.05);  // 5% variation

    // Small variance: rho from 0.98 to 1.02 (2% variation)
    Conserved siblings[4] = {
        Conserved{0.98f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.02f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.00f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.00f, 0.0f, 0.0f, 2.5e5f}
    };

    Conserved U_center{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q_center{1.0f, 0.0f, 0.0f, 101325.0f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    EXPECT_EQ(action, RefinementAction::Coarsen);
}

TEST(SmoothnessCriterion_Euler2D_Float, HighVariance_Keeps) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.05);  // 5% variation

    // High variance: rho from 0.5 to 1.5 (50% variation)
    Conserved siblings[4] = {
        Conserved{0.5f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.5f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{0.8f, 0.0f, 0.0f, 2.5e5f}
    };

    Conserved U_center{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q_center{1.0f, 0.0f, 0.0f, 101325.0f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST(SmoothnessCriterion_Euler2D_Float, ZeroDensitySibling_NoCoarsen) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;

    // One sibling has zero density (invalid)
    Conserved siblings[4] = {
        Conserved{0.0f, 0.0f, 0.0f, 0.0f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f}
    };

    Conserved U_center{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q_center{1.0f, 0.0f, 0.0f, 101325.0f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST(SmoothnessCriterion_Euler2D_Float, NegativeDensitySibling_NoCoarsen) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;

    // One sibling has negative density (unphysical)
    Conserved siblings[4] = {
        Conserved{-0.1f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f}
    };

    Conserved U_center{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q_center{1.0f, 0.0f, 0.0f, 101325.0f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST(SmoothnessCriterion_Euler2D_Float, VerySmallDensity_HandlesEpsilon) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.05);

    // Very small density values (near-vacuum conditions)
    Conserved siblings[4] = {
        Conserved{1e-8f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.01e-8f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1e-8f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1e-8f, 0.0f, 0.0f, 2.5e5f}
    };

    Conserved U_center{1e-8f, 0.0f, 0.0f, 2.5e5f};
    Primitive q_center{1e-8f, 0.0f, 0.0f, 100.0f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    // Should handle without division by zero - small variance so should coarsen
    EXPECT_EQ(action, RefinementAction::Coarsen);
}

TEST(SmoothnessCriterion_Euler2D_Float, AllSiblingsZero_Coarsens) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;

    // All siblings have zero density (edge case - but valid for initialization)
    // With epsilon protection, variance = 0, so rel_variation = 0
    Conserved siblings[4] = {
        Conserved{0.0f, 0.0f, 0.0f, 0.0f},
        Conserved{0.0f, 0.0f, 0.0f, 0.0f},
        Conserved{0.0f, 0.0f, 0.0f, 0.0f},
        Conserved{0.0f, 0.0f, 0.0f, 0.0f}
    };

    Conserved U_center{0.0f, 0.0f, 0.0f, 0.0f};
    Primitive q_center{0.0f, 0.0f, 0.0f, 0.0f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    // With invalid check (rho <= 0), this returns Keep
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST(SmoothnessCriterion_Euler2D_Float, ThresholdBoundary_CoarsensAtThreshold) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.05);

    // Create variance just below threshold (~4.9%)
    // Values: 0.951, 1.0, 1.0, 1.049
    // Mean = 1.0, variance ~ 0.0006, std ~ 0.0245, rel_var ~ 0.0245
    Conserved siblings[4] = {
        Conserved{0.951f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.049f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f}
    };

    Conserved U_center{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q_center{1.0f, 0.0f, 0.0f, 101325.0f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    EXPECT_EQ(action, RefinementAction::Coarsen);
}

TEST(SmoothnessCriterion_Euler2D_Float, ThresholdBoundary_KeepsAboveThreshold) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.02);  // 2% threshold

    // Create variance above threshold (~4.9%)
    Conserved siblings[4] = {
        Conserved{0.951f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.049f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f}
    };

    Conserved U_center{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q_center{1.0f, 0.0f, 0.0f, 101325.0f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST(SmoothnessCriterion_Euler2D_Float, Discontinuity_KeepsRefined) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.1);  // Very permissive

    // Simulate a shock: two distinct values
    Conserved siblings[4] = {
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{2.0f, 0.0f, 0.0f, 3.5e5f},
        Conserved{2.0f, 0.0f, 0.0f, 3.5e5f}
    };

    Conserved U_center{1.5f, 0.0f, 0.0f, 3.0e5f};
    Primitive q_center{1.5f, 0.0f, 0.0f, 150000.0f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    // High variance due to discontinuity
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST(SmoothnessCriterion_Euler2D_Float, LinearGradient_SmallGradient_Coarsens) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.03);

    // Small linear gradient: 0.99, 1.0, 1.01, 1.02
    Conserved siblings[4] = {
        Conserved{0.99f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.00f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.01f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.02f, 0.0f, 0.0f, 2.5e5f}
    };

    Conserved U_center{1.005f, 0.0f, 0.0f, 2.5e5f};
    Primitive q_center{1.005f, 0.0f, 0.0f, 101325.0f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    EXPECT_EQ(action, RefinementAction::Coarsen);
}

TEST(SmoothnessCriterion_Euler2D_Float, UseRhoFalse_NoCoarsen) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.05);
    crit.use_rho = false;  // Disable density check

    // Smooth density variation
    Conserved siblings[4] = {
        Conserved{0.99f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.01f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f}
    };

    Conserved U_center{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q_center{1.0f, 0.0f, 0.0f, 101325.0f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    // With use_rho=false and no other variables implemented, returns Keep
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST(SmoothnessCriterion_Euler2D_Float, LinearlyIncreasingValues_Coarsens) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.1);

    // Linear ramp: 0.9, 1.0, 1.1, 1.2
    Conserved siblings[4] = {
        Conserved{0.9f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.1f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.2f, 0.0f, 0.0f, 2.5e5f}
    };

    Conserved U_center{1.05f, 0.0f, 0.0f, 2.5e5f};
    Primitive q_center{1.05f, 0.0f, 0.0f, 101325.0f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    // Mean=1.05, variance≈0.0125, std≈0.112, rel_var≈0.107 > 0.1
    EXPECT_EQ(action, RefinementAction::Keep);
}

// ============================================================================
// SMOOTHNESS CRITERION TESTS - EULER2D DOUBLE
// ============================================================================

TEST(SmoothnessCriterion_Euler2D_Double, HighPrecision_Coarsens) {
    using Real = double;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.05);

    // Very small variance in double precision
    Conserved siblings[4] = {
        Conserved{0.9999, 0.0, 0.0, 2.5e5},
        Conserved{1.0001, 0.0, 0.0, 2.5e5},
        Conserved{1.0, 0.0, 0.0, 2.5e5},
        Conserved{1.0, 0.0, 0.0, 2.5e5}
    };

    Conserved U_center{1.0, 0.0, 0.0, 2.5e5};
    Primitive q_center{1.0, 0.0, 0.0, 101325.0};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01);
    EXPECT_EQ(action, RefinementAction::Coarsen);
}

TEST(SmoothnessCriterion_Euler2D_Double, VeryHighDensity_Coarsens) {
    using Real = double;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.05);

    // High density values (e.g., high-pressure conditions)
    Conserved siblings[4] = {
        Conserved{999.9, 0.0, 0.0, 2.5e8},
        Conserved{1000.1, 0.0, 0.0, 2.5e8},
        Conserved{1000.0, 0.0, 0.0, 2.5e8},
        Conserved{1000.0, 0.0, 0.0, 2.5e8}
    };

    Conserved U_center{1000.0, 0.0, 0.0, 2.5e8};
    Primitive q_center{1000.0, 0.0, 0.0, 1.01325e8};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01);
    EXPECT_EQ(action, RefinementAction::Coarsen);
}

TEST(SmoothnessCriterion_Euler2D_Double, EpsilonScaling_HandlesTinyValues) {
    using Real = double;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.05);

    // Astrophysical-scale tiny densities
    Conserved siblings[4] = {
        Conserved{1e-20, 0.0, 0.0, 1e-10},
        Conserved{1.01e-20, 0.0, 0.0, 1.01e-10},
        Conserved{1e-20, 0.0, 0.0, 1e-10},
        Conserved{1e-20, 0.0, 0.0, 1e-10}
    };

    Conserved U_center{1e-20, 0.0, 0.0, 1e-10};
    Primitive q_center{1e-20, 0.0, 0.0, 1e-5};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01);
    // Epsilon (1e-10) is much larger than these values, so rel_variation will be tiny
    EXPECT_EQ(action, RefinementAction::Coarsen);
}

// ============================================================================
// SMOOTHNESS CRITERION TESTS - ADVECTION2D
// ============================================================================

TEST(SmoothnessCriterion_Advection2D, AllSiblingsIdentical_Coarsens) {
    using Real = float;
    using System = Advection2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.01);

    // All siblings have same scalar value
    Conserved siblings[4] = {
        Conserved{1.0f},
        Conserved{1.0f},
        Conserved{1.0f},
        Conserved{1.0f}
    };

    Conserved U_center{1.0f};
    Primitive q_center{1.0f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    EXPECT_EQ(action, RefinementAction::Coarsen);
}

TEST(SmoothnessCriterion_Advection2D, LowVariance_Coarsens) {
    using Real = float;
    using System = Advection2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.05);

    // Small variance: 0.98, 1.02, 1.0, 1.0
    Conserved siblings[4] = {
        Conserved{0.98f},
        Conserved{1.02f},
        Conserved{1.00f},
        Conserved{1.00f}
    };

    Conserved U_center{1.0f};
    Primitive q_center{1.0f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    EXPECT_EQ(action, RefinementAction::Coarsen);
}

TEST(SmoothnessCriterion_Advection2D, HighVariance_Keeps) {
    using Real = float;
    using System = Advection2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.05);

    // High variance
    Conserved siblings[4] = {
        Conserved{0.5f},
        Conserved{1.5f},
        Conserved{1.0f},
        Conserved{0.8f}
    };

    Conserved U_center{0.95f};
    Primitive q_center{0.95f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST(SmoothnessCriterion_Advection2D, AllZero_Coarsens) {
    using Real = float;
    using System = Advection2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;

    // All siblings zero
    Conserved siblings[4] = {
        Conserved{0.0f},
        Conserved{0.0f},
        Conserved{0.0f},
        Conserved{0.0f}
    };

    Conserved U_center{0.0f};
    Primitive q_center{0.0f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    // With abs(mean) + eps, variance = 0, rel_var = 0 < threshold
    EXPECT_EQ(action, RefinementAction::Coarsen);
}

TEST(SmoothnessCriterion_Advection2D, NegativeValues_HandlesCorrectly) {
    using Real = float;
    using System = Advection2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.05);

    // Negative scalar values (valid for advection)
    Conserved siblings[4] = {
        Conserved{-1.0f},
        Conserved{-1.01f},
        Conserved{-0.99f},
        Conserved{-1.0f}
    };

    Conserved U_center{-1.0f};
    Primitive q_center{-1.0f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    // Small variance around negative mean
    EXPECT_EQ(action, RefinementAction::Coarsen);
}

TEST(SmoothnessCriterion_Advection2D, Discontinuity_Keeps) {
    using Real = float;
    using System = Advection2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.1);

    // Step discontinuity
    Conserved siblings[4] = {
        Conserved{0.0f},
        Conserved{0.0f},
        Conserved{1.0f},
        Conserved{1.0f}
    };

    Conserved U_center{0.5f};
    Primitive q_center{0.5f};

    auto action = crit.evaluate(U_center, q_center, siblings, 0.01f);
    // High variance due to discontinuity
    EXPECT_EQ(action, RefinementAction::Keep);
}

// ============================================================================
// SMOOTHNESS CRITERION TESTS - GPU COMPATIBILITY
// ============================================================================

TEST(SmoothnessCriterion_GPU, ParallelFor_Coarsen) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.01);

    const int n = 100;
    Kokkos::View<RefinementAction*> actions("actions", n);
    Kokkos::View<Conserved*[4]> siblings_array("siblings", n);

    // Initialize siblings on host - all identical (should coarsen)
    auto host_siblings = Kokkos::create_mirror_view(siblings_array);
    for (int i = 0; i < n; ++i) {
        host_siblings(i, 0) = Conserved{1.0f, 0.0f, 0.0f, 2.5e5f};
        host_siblings(i, 1) = Conserved{1.0f, 0.0f, 0.0f, 2.5e5f};
        host_siblings(i, 2) = Conserved{1.0f, 0.0f, 0.0f, 2.5e5f};
        host_siblings(i, 3) = Conserved{1.0f, 0.0f, 0.0f, 2.5e5f};
    }
    Kokkos::deep_copy(siblings_array, host_siblings);

    // Evaluate in parallel kernel
    Kokkos::parallel_for("test_smoothness_coarsen", n,
        KOKKOS_LAMBDA(const int i) {
            Conserved siblings[4];
            for (int j = 0; j < 4; ++j) {
                siblings[j] = siblings_array(i, j);
            }
            Conserved U_center{1.0f, 0.0f, 0.0f, 2.5e5f};
            Primitive q_center{1.0f, 0.0f, 0.0f, 101325.0f};
            actions(i) = crit.evaluate(U_center, q_center, siblings, 0.01f);
        }
    );

    Kokkos::fence();

    // Verify all coarsen
    auto host_actions = Kokkos::create_mirror_view(actions);
    Kokkos::deep_copy(host_actions, actions);

    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(host_actions(i), RefinementAction::Coarsen)
            << "Failed at index " << i;
    }
}

TEST(SmoothnessCriterion_GPU, ParallelFor_Keep) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.05);

    const int n = 100;
    Kokkos::View<RefinementAction*> actions("actions", n);
    Kokkos::View<Conserved*[4]> siblings_array("siblings", n);

    // Initialize siblings on host - high variance (should keep)
    auto host_siblings = Kokkos::create_mirror_view(siblings_array);
    for (int i = 0; i < n; ++i) {
        host_siblings(i, 0) = Conserved{0.5f, 0.0f, 0.0f, 2.5e5f};
        host_siblings(i, 1) = Conserved{1.5f, 0.0f, 0.0f, 2.5e5f};
        host_siblings(i, 2) = Conserved{1.0f, 0.0f, 0.0f, 2.5e5f};
        host_siblings(i, 3) = Conserved{0.8f, 0.0f, 0.0f, 2.5e5f};
    }
    Kokkos::deep_copy(siblings_array, host_siblings);

    // Evaluate in parallel kernel
    Kokkos::parallel_for("test_smoothness_keep", n,
        KOKKOS_LAMBDA(const int i) {
            Conserved siblings[4];
            for (int j = 0; j < 4; ++j) {
                siblings[j] = siblings_array(i, j);
            }
            Conserved U_center{1.0f, 0.0f, 0.0f, 2.5e5f};
            Primitive q_center{1.0f, 0.0f, 0.0f, 101325.0f};
            actions(i) = crit.evaluate(U_center, q_center, siblings, 0.01f);
        }
    );

    Kokkos::fence();

    // Verify all keep
    auto host_actions = Kokkos::create_mirror_view(actions);
    Kokkos::deep_copy(host_actions, actions);

    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(host_actions(i), RefinementAction::Keep)
            << "Failed at index " << i;
    }
}

TEST(SmoothnessCriterion_GPU, MixedDecisions) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SmoothnessCriterion<System> crit;
    crit.coarsen_threshold = Real(0.05);

    const int n = 100;
    Kokkos::View<RefinementAction*> actions("actions", n);
    Kokkos::View<Conserved*[4]> siblings_array("siblings", n);

    // Initialize siblings on host - mixed
    auto host_siblings = Kokkos::create_mirror_view(siblings_array);
    for (int i = 0; i < n; ++i) {
        if (i % 2 == 0) {
            // Even: smooth (coarsen)
            host_siblings(i, 0) = Conserved{1.0f, 0.0f, 0.0f, 2.5e5f};
            host_siblings(i, 1) = Conserved{1.0f, 0.0f, 0.0f, 2.5e5f};
            host_siblings(i, 2) = Conserved{1.0f, 0.0f, 0.0f, 2.5e5f};
            host_siblings(i, 3) = Conserved{1.0f, 0.0f, 0.0f, 2.5e5f};
        } else {
            // Odd: rough (keep)
            host_siblings(i, 0) = Conserved{0.5f, 0.0f, 0.0f, 2.5e5f};
            host_siblings(i, 1) = Conserved{1.5f, 0.0f, 0.0f, 2.5e5f};
            host_siblings(i, 2) = Conserved{1.0f, 0.0f, 0.0f, 2.5e5f};
            host_siblings(i, 3) = Conserved{0.8f, 0.0f, 0.0f, 2.5e5f};
        }
    }
    Kokkos::deep_copy(siblings_array, host_siblings);

    // Evaluate in parallel kernel
    Kokkos::parallel_for("test_smoothness_mixed", n,
        KOKKOS_LAMBDA(const int i) {
            Conserved siblings[4];
            for (int j = 0; j < 4; ++j) {
                siblings[j] = siblings_array(i, j);
            }
            Conserved U_center{1.0f, 0.0f, 0.0f, 2.5e5f};
            Primitive q_center{1.0f, 0.0f, 0.0f, 101325.0f};
            actions(i) = crit.evaluate(U_center, q_center, siblings, 0.01f);
        }
    );

    Kokkos::fence();

    // Verify mixed results
    auto host_actions = Kokkos::create_mirror_view(actions);
    Kokkos::deep_copy(host_actions, actions);

    for (int i = 0; i < n; ++i) {
        if (i % 2 == 0) {
            EXPECT_EQ(host_actions(i), RefinementAction::Coarsen)
                << "Even index " << i << " should coarsen";
        } else {
            EXPECT_EQ(host_actions(i), RefinementAction::Keep)
                << "Odd index " << i << " should keep";
        }
    }
}

// ============================================================================
// GRADIENT CRITERION TESTS
// ============================================================================

TEST(GradientCriterion_Euler2D, ZeroGradient_Keeps) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    GradientCriterion<System> crit;
    crit.threshold = Real(0.1);
    crit.use_rho = true;

    // All neighbors have same density
    Conserved neighbors[4] = {
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f}
    };

    Conserved U_center{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q_center{1.0f, 0.0f, 0.0f, 101325.0f};

    auto action = crit.evaluate(U_center, q_center, neighbors, 0.01f);
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST(GradientCriterion_Euler2D, HighGradient_Refines) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    GradientCriterion<System> crit;
    crit.threshold = Real(0.1);
    crit.use_rho = true;

    // Large gradient: 1.0 to 2.0 over 2*dx = 0.02
    // gradient = 1.0 / 0.02 = 50 >> 0.1
    Conserved neighbors[4] = {
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{2.0f, 0.0f, 0.0f, 3.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{2.0f, 0.0f, 0.0f, 3.5e5f}
    };

    Conserved U_center{1.5f, 0.0f, 0.0f, 3.0e5f};
    Primitive q_center{1.5f, 0.0f, 0.0f, 150000.0f};

    auto action = crit.evaluate(U_center, q_center, neighbors, 0.01f);
    EXPECT_EQ(action, RefinementAction::Refine);
}

TEST(GradientCriterion_Advection2D, ScalarGradient) {
    using Real = float;
    using System = Advection2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    GradientCriterion<System> crit;
    crit.threshold = Real(1.0);

    Conserved neighbors[4] = {
        Conserved{0.0f},
        Conserved{1.0f},
        Conserved{0.0f},
        Conserved{1.0f}
    };

    Conserved U_center{0.5f};
    Primitive q_center{0.5f};

    auto action = crit.evaluate(U_center, q_center, neighbors, 0.01f);
    // gradient = 1.0 / 0.02 = 50 >> 1.0
    EXPECT_EQ(action, RefinementAction::Refine);
}

// ============================================================================
// VORTICITY CRITERION TESTS
// ============================================================================

TEST(VorticityCriterion_Euler2D, SolidBodyRotation_Refines) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    VorticityCriterion<System> crit;
    crit.threshold = Real(1.0);

    // Solid body rotation velocity field
    // vorticity = dv/dx - du/dy
    // Create high vorticity: v varies with x, u constant (no variation in y)
    Real rho = 1.0f;
    Real dx = 0.01f;
    Real v_left = -10.0f;
    Real v_right = 10.0f;

    Conserved neighbors[4] = {
        Conserved{rho, rho*0, rho*v_left, 2.5e5f},      // left: v = -10
        Conserved{rho, rho*0, rho*v_right, 2.5e5f},     // right: v = +10
        Conserved{rho, rho*0, rho*0, 2.5e5f},           // bottom: u = 0 (same as top)
        Conserved{rho, rho*0, rho*0, 2.5e5f}            // top: u = 0 (same as bottom)
    };

    Conserved U_center{rho, 0.0f, 0.0f, 2.5e5f};
    Primitive q_center{rho, 0.0f, 0.0f, 101325.0f};

    auto action = crit.evaluate(U_center, q_center, neighbors, dx);
    // dv/dx = (10 - (-10)) / (2*dx) = 20 / 0.02 = 1000
    // du/dy = 0 (no variation)
    // vorticity = 1000 > 1.0
    EXPECT_EQ(action, RefinementAction::Refine);
}

TEST(VorticityCriterion_Euler2D, UniformFlow_NoRefine) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    VorticityCriterion<System> crit;
    crit.threshold = Real(0.1);

    // Uniform flow
    Conserved neighbors[4] = {
        Conserved{1.0f, 10.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 10.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 10.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 10.0f, 0.0f, 2.5e5f}
    };

    Conserved U_center{1.0f, 10.0f, 0.0f, 2.5e5f};
    Primitive q_center{1.0f, 10.0f, 0.0f, 101325.0f};

    auto action = crit.evaluate(U_center, q_center, neighbors, 0.01f);
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST(VorticityCriterion_Advection2D, NoOp) {
    using Real = float;
    using System = Advection2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    VorticityCriterion<System> crit;
    crit.threshold = Real(0.1);

    Conserved neighbors[4] = {
        Conserved{1.0f},
        Conserved{2.0f},
        Conserved{1.0f},
        Conserved{2.0f}
    };

    Conserved U_center{1.5f};
    Primitive q_center{1.5f};

    auto action = crit.evaluate(U_center, q_center, neighbors, 0.01f);
    // Vorticity doesn't apply to scalar systems
    EXPECT_EQ(action, RefinementAction::Keep);
}

// ============================================================================
// VALUE RANGE CRITERION TESTS
// ============================================================================

TEST(ValueRangeCriterion_Euler2D, InsideRange_Refines) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    ValueRangeCriterion<System> crit;
    crit.variable = ValueRangeCriterion<System>::Density;
    crit.min_val = Real(0.5);
    crit.max_val = Real(1.5);
    crit.invert = false;

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 101325.0f};

    auto action = crit.evaluate(U, q, 0.01f);
    EXPECT_EQ(action, RefinementAction::Refine);
}

TEST(ValueRangeCriterion_Euler2D, OutsideRange_Keeps) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    ValueRangeCriterion<System> crit;
    crit.variable = ValueRangeCriterion<System>::Density;
    crit.min_val = Real(0.5);
    crit.max_val = Real(1.5);
    crit.invert = false;

    Conserved U{2.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{2.0f, 0.0f, 0.0f, 101325.0f};

    auto action = crit.evaluate(U, q, 0.01f);
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST(ValueRangeCriterion_Euler2D, InvertedLogic_RefinesOutside) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    ValueRangeCriterion<System> crit;
    crit.variable = ValueRangeCriterion<System>::Density;
    crit.min_val = Real(0.5);
    crit.max_val = Real(1.5);
    crit.invert = true;  // Refine OUTSIDE range

    Conserved U{2.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{2.0f, 0.0f, 0.0f, 101325.0f};

    auto action = crit.evaluate(U, q, 0.01f);
    EXPECT_EQ(action, RefinementAction::Refine);
}

TEST(ValueRangeCriterion_Euler2D, PressureVariable) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    ValueRangeCriterion<System> crit;
    crit.variable = ValueRangeCriterion<System>::Pressure;
    crit.min_val = Real(100000.0);
    crit.max_val = Real(200000.0);

    Conserved U{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 0.0f, 0.0f, 150000.0f};

    auto action = crit.evaluate(U, q, 0.01f);
    EXPECT_EQ(action, RefinementAction::Refine);
}

TEST(ValueRangeCriterion_Euler2D, VelocityMagVariable) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    ValueRangeCriterion<System> crit;
    crit.variable = ValueRangeCriterion<System>::VelocityMag;
    crit.min_val = Real(100.0);
    crit.max_val = Real(300.0);

    Conserved U{1.0f, 150.0f, 0.0f, 2.5e5f};
    Primitive q{1.0f, 150.0f, 0.0f, 101325.0f};

    auto action = crit.evaluate(U, q, 0.01f);
    EXPECT_EQ(action, RefinementAction::Refine);
}

TEST(ValueRangeCriterion_Advection2D, ScalarRange) {
    using Real = float;
    using System = Advection2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    ValueRangeCriterion<System> crit;
    crit.variable = ValueRangeCriterion<System>::Density;
    crit.min_val = Real(0.0);
    crit.max_val = Real(1.0);

    Conserved U{0.5f};
    Primitive q{0.5f};

    auto action = crit.evaluate(U, q, 0.01f);
    EXPECT_EQ(action, RefinementAction::Refine);
}

// ============================================================================
// CURL CRITERION TESTS
// ============================================================================

TEST(CurlCriterion_Euler2D, UniformFlow_NoRefine) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    CurlCriterion<System> crit;
    crit.threshold = Real(1.0);

    Conserved neighbors[4] = {
        Conserved{1.0f, 10.0f, 5.0f, 2.5e5f},
        Conserved{1.0f, 10.0f, 5.0f, 2.5e5f},
        Conserved{1.0f, 10.0f, 5.0f, 2.5e5f},
        Conserved{1.0f, 10.0f, 5.0f, 2.5e5f}
    };

    Conserved U_center{1.0f, 10.0f, 5.0f, 2.5e5f};
    Primitive q_center{1.0f, 10.0f, 5.0f, 101325.0f};

    auto action = crit.evaluate(U_center, q_center, neighbors, 0.01f);
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST(CurlCriterion_Euler2D, HighCurl_Refines) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    CurlCriterion<System> crit;
    crit.threshold = Real(1.0);

    Real rho = 1.0f;
    Real dx = 0.01f;
    Real v_left = -10.0f;
    Real v_right = 10.0f;

    // Create high vorticity: v varies with x, u constant (no variation in y)
    Conserved neighbors[4] = {
        Conserved{rho, rho*0, rho*v_left, 2.5e5f},      // left: v = -10
        Conserved{rho, rho*0, rho*v_right, 2.5e5f},     // right: v = +10
        Conserved{rho, rho*0, rho*0, 2.5e5f},           // bottom: u = 0 (same as top)
        Conserved{rho, rho*0, rho*0, 2.5e5f}            // top: u = 0 (same as bottom)
    };

    Conserved U_center{rho, 0.0f, 0.0f, 2.5e5f};
    Primitive q_center{rho, 0.0f, 0.0f, 101325.0f};

    auto action = crit.evaluate(U_center, q_center, neighbors, dx);
    // dv/dx = (10 - (-10)) / (2*dx) = 20 / 0.02 = 1000
    // du/dy = 0 (no variation)
    // curl = 1000 > 1.0
    EXPECT_EQ(action, RefinementAction::Refine);
}

TEST(CurlCriterion_Advection2D, NoOp) {
    using Real = float;
    using System = Advection2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    CurlCriterion<System> crit;
    crit.threshold = Real(1.0);

    Conserved neighbors[4] = {
        Conserved{1.0f},
        Conserved{2.0f},
        Conserved{1.0f},
        Conserved{2.0f}
    };

    Conserved U_center{1.5f};
    Primitive q_center{1.5f};

    auto action = crit.evaluate(U_center, q_center, neighbors, 0.01f);
    EXPECT_EQ(action, RefinementAction::Keep);
}

// ============================================================================
// COMPOSITE CRITERION TESTS
// ============================================================================

TEST(CompositeCriterion, OrLogic_AnyRefines) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    CompositeCriterion<System, 4> comp;
    comp.logic_op = CompositeCriterion<System, 4>::Or;

    // Add gradient criterion (will refine)
    GradientCriterion<System> grad_crit;
    grad_crit.threshold = Real(0.1);
    comp.add_gradient(grad_crit);

    // Add vorticity criterion (will keep)
    VorticityCriterion<System> vort_crit;
    vort_crit.threshold = Real(1000.0);
    comp.add_vorticity(vort_crit);

    // Gradient should trigger refine
    Conserved neighbors[4] = {
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{2.0f, 0.0f, 0.0f, 3.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{2.0f, 0.0f, 0.0f, 3.5e5f}
    };

    Conserved U_center{1.5f, 0.0f, 0.0f, 3.0e5f};
    Primitive q_center{1.5f, 0.0f, 0.0f, 150000.0f};

    auto action = comp.evaluate(U_center, q_center, neighbors, 0.01f);
    EXPECT_EQ(action, RefinementAction::Refine);
}

TEST(CompositeCriterion, AndLogic_AllMustRefine) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    CompositeCriterion<System, 4> comp;
    comp.logic_op = CompositeCriterion<System, 4>::And;

    // Add gradient criterion (will refine)
    GradientCriterion<System> grad_crit;
    grad_crit.threshold = Real(0.1);
    comp.add_gradient(grad_crit);

    // Add vorticity criterion (will keep - low vorticity)
    VorticityCriterion<System> vort_crit;
    vort_crit.threshold = Real(1.0);
    comp.add_vorticity(vort_crit);

    // High gradient but low vorticity - should keep
    Conserved neighbors[4] = {
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{2.0f, 0.0f, 0.0f, 3.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{2.0f, 0.0f, 0.0f, 3.5e5f}
    };

    Conserved U_center{1.5f, 0.0f, 0.0f, 3.0e5f};
    Primitive q_center{1.5f, 0.0f, 0.0f, 150000.0f};

    auto action = comp.evaluate(U_center, q_center, neighbors, 0.01f);
    // AND requires both to agree - vorticity says keep
    EXPECT_EQ(action, RefinementAction::Keep);
}

TEST(CompositeCriterion, VoteLogic_MajorityRefines) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    CompositeCriterion<System, 8> comp;
    comp.logic_op = CompositeCriterion<System, 8>::Vote;

    // Add 3 criteria that will refine, 2 that will keep
    GradientCriterion<System> grad_crit;
    grad_crit.threshold = Real(0.1);
    comp.add_gradient(grad_crit);

    VorticityCriterion<System> vort_crit;
    vort_crit.threshold = Real(0.1);
    comp.add_vorticity(vort_crit);

    ValueRangeCriterion<System> range_crit;
    range_crit.variable = ValueRangeCriterion<System>::Density;
    range_crit.min_val = Real(0.0);
    range_crit.max_val = Real(10.0);
    comp.add_value_range(range_crit);

    // High gradient + vorticity to trigger refine
    Conserved neighbors[4] = {
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{2.0f, 0.0f, 0.0f, 3.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{2.0f, 0.0f, 0.0f, 3.5e5f}
    };

    Conserved U_center{1.5f, 0.0f, 0.0f, 3.0e5f};
    Primitive q_center{1.5f, 0.0f, 0.0f, 150000.0f};

    auto action = comp.evaluate(U_center, q_center, neighbors, 0.01f);
    // 3/5 = 60% > 50%, should refine
    EXPECT_EQ(action, RefinementAction::Refine);
}

TEST(CompositeCriterion, UnanimousCoarsen_AllCoarsen) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    CompositeCriterion<System, 4> comp;
    comp.logic_op = CompositeCriterion<System, 4>::Or;

    // Add two smoothness criteria (both will coarsen)
    SmoothnessCriterion<System> smooth1;
    smooth1.coarsen_threshold = Real(0.05);
    comp.add_smoothness(smooth1);

    SmoothnessCriterion<System> smooth2;
    smooth2.coarsen_threshold = Real(0.1);
    comp.add_smoothness(smooth2);

    // Smooth siblings
    Conserved siblings[4] = {
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f}
    };

    Conserved U_center{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q_center{1.0f, 0.0f, 0.0f, 101325.0f};

    auto action = comp.evaluate(U_center, q_center, siblings, 0.01f);
    // Both smoothness criteria agree to coarsen
    EXPECT_EQ(action, RefinementAction::Coarsen);
}

TEST(CompositeCriterion, MixedActions_Keeps) {
    using Real = float;
    using System = Euler2D<Real>;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    CompositeCriterion<System, 4> comp;
    comp.logic_op = CompositeCriterion<System, 4>::Or;

    // One says refine, one says coarsen
    GradientCriterion<System> grad_crit;
    grad_crit.threshold = Real(10.0);  // Won't trigger
    comp.add_gradient(grad_crit);

    SmoothnessCriterion<System> smooth_crit;
    smooth_crit.coarsen_threshold = Real(0.05);
    comp.add_smoothness(smooth_crit);

    // Smooth siblings (coarsen) but no gradient (keep)
    Conserved siblings[4] = {
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f},
        Conserved{1.0f, 0.0f, 0.0f, 2.5e5f}
    };

    Conserved U_center{1.0f, 0.0f, 0.0f, 2.5e5f};
    Primitive q_center{1.0f, 0.0f, 0.0f, 101325.0f};

    auto action = comp.evaluate(U_center, q_center, siblings, 0.01f);
    // Smoothness says coarsen, but gradient says keep
    // With OR logic and no unanimous coarsening (grad says keep), result is keep
    EXPECT_EQ(action, RefinementAction::Keep);
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
