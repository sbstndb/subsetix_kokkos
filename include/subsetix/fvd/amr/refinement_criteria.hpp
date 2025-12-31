#pragma once

#include <Kokkos_Core.hpp>
#include <concepts>
#include <type_traits>
#include "../system/concepts_v2.hpp"
#include "../geometry/csr_types.hpp"

namespace subsetix::fvd::amr {

// ============================================================================
// RESULT ENUM
// ============================================================================

/**
 * @brief Result of refinement evaluation
 */
enum class RefinementAction : int8_t {
    Coarsen = -1,   // Remove this cell (merge with parent)
    Keep = 0,       // Keep current level
    Refine = 1      // Split this cell (add children)
};

// ============================================================================
// BASE CRITERION CONCEPT
// ============================================================================

/**
 * @brief Concept for refinement criteria
 *
 * A criterion must be:
 * - POD-friendly (default constructible, trivially copyable)
 * - Callable with signature: RefinementAction(Conserved, Primitive, dx, neighbors)
 * - GPU-compatible (KOKKOS_INLINE_FUNCTION)
 */
template<typename T, typename System>
concept RefinementCriterion =
    std::is_default_constructible_v<T> &&
    std::is_trivially_copyable_v<T> &&
    requires(const T& crit,
             const typename System::Conserved& U,
             const typename System::Primitive& q,
             const typename System::RealType dx)
    {
        { crit.evaluate(U, q, dx) } -> std::convertible_to<RefinementAction>;
    };

// ============================================================================
// GRADIENT CRITERION
// ============================================================================

/**
 * @brief Refine based on solution gradient
 *
 * Refines where: |grad(U)| > threshold
 * Can use density, pressure, or velocity gradients.
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class GradientCriterion {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    Real threshold = Real(0.1);
    bool use_rho = true;
    bool use_p = true;
    bool use_u = false;

    KOKKOS_INLINE_FUNCTION
    GradientCriterion() = default;

    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U_center,
                               const Primitive& q_center,
                               const Conserved* neighbors,
                               Real dx) const
    {
        // Compute gradient using finite differences
        // neighbors[0] = left, [1] = right, [2] = bottom, [3] = top

        Real grad_rho = Real(0);
        Real grad_p = Real(0);
        Real grad_u = Real(0);

        if (neighbors[0].rho > Real(0) && neighbors[1].rho > Real(0)) {
            grad_rho = Kokkos::abs(neighbors[1].rho - neighbors[0].rho) / (Real(2) * dx);
        }
        if (neighbors[2].rho > Real(0) && neighbors[3].rho > Real(0)) {
            Real grad_y = Kokkos::abs(neighbors[3].rho - neighbors[2].rho) / (Real(2) * dx);
            grad_rho = Kokkos::max(grad_rho, grad_y);
        }

        if (use_rho && grad_rho > threshold) {
            return RefinementAction::Refine;
        }
        return RefinementAction::Keep;
    }

    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U,
                               const Primitive& q,
                               Real dx) const
    {
        // Simplified version without neighbors (for single cell)
        return RefinementAction::Keep;
    }
};

// ============================================================================
// SHOCK SENSOR CRITERION
// ============================================================================

/**
 * @brief Refine based on shock sensor
 *
 * Uses Ducros or Jameson shock sensor to detect discontinuities.
 * Ducros sensor: div(V) / (|div(V)| + |curl(V)| + eps)
 * - Near 1: compression (shock)
 * - Near 0: vorticity-dominated (turbulence)
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class ShockSensorCriterion {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    enum SensorType : uint8_t {
        Ducros = 0,      // Divergence of velocity vs vorticity
        Jameson = 1,     // Pressure gradient
        Persson = 2      // Spectral decay (for high-order)
    };

    SensorType sensor_type = Ducros;
    Real threshold = Real(0.8);

    KOKKOS_INLINE_FUNCTION
    ShockSensorCriterion() = default;

    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U_center,
                               const Primitive& q_center,
                               const Conserved* neighbors,
                               Real dx) const
    {
        Real sensor = compute_sensor(q_center, neighbors, dx);

        if (sensor > threshold) {
            return RefinementAction::Refine;
        }
        return RefinementAction::Keep;
    }

    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U,
                               const Primitive& q,
                               Real dx) const
    {
        return RefinementAction::Keep;
    }

private:
    KOKKOS_INLINE_FUNCTION
    Real compute_sensor(const Primitive& q_center,
                        const Conserved* neighbors,
                        Real dx) const
    {
        switch (sensor_type) {
            case Ducros: {
                // Need velocity gradients for Ducros
                // Simplified version: use pressure gradient as proxy
                Real grad_p_max = Real(0);
                for (int i = 0; i < 4; ++i) {
                    if (neighbors[i].rho > Real(0)) {
                        // Approximate pressure from conserved
                        Real dp = Kokkos::abs(q_center.p - (Real(0.4) * neighbors[i].E));  // Simplified
                        grad_p_max = Kokkos::max(grad_p_max, dp / dx);
                    }
                }
                return grad_p_max / (q_center.p + Real(1e-10));
            }
            case Jameson: {
                // |grad(p)| * dx / p
                Real grad_p = Real(0);
                int count = 0;
                for (int i = 0; i < 4; ++i) {
                    if (neighbors[i].rho > Real(0)) {
                        grad_p += Kokkos::abs(neighbors[i].rho - q_center.rho);
                        count++;
                    }
                }
                if (count > 0) {
                    grad_p = grad_p * dx / (count * (q_center.p + Real(1e-10)));
                }
                return grad_p;
            }
            default:
                return Real(0);
        }
    }
};

// ============================================================================
// VORTICITY CRITERION
// ============================================================================

/**
 * @brief Refine based on vorticity magnitude
 *
 * Useful for capturing shear layers and vortices.
 * vorticity = dv/dx - du/dy
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class VorticityCriterion {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    Real threshold = Real(1.0);

    KOKKOS_INLINE_FUNCTION
    VorticityCriterion() = default;

    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U_center,
                               const Primitive& q_center,
                               const Conserved* neighbors,
                               Real dx) const
    {
        // Compute vorticity: dv/dx - du/dy
        // Simplified using neighbor differences
        Real du_dy = Real(0);
        Real dv_dx = Real(0);

        if (neighbors[0].rho > Real(0) && neighbors[1].rho > Real(0)) {
            // dv/dx ≈ (v_right - v_left) / (2*dx)
            Real v_left = neighbors[0].rhov / (neighbors[0].rho + Real(1e-10));
            Real v_right = neighbors[1].rhov / (neighbors[1].rho + Real(1e-10));
            dv_dx = (v_right - v_left) / (Real(2) * dx);
        }

        if (neighbors[2].rho > Real(0) && neighbors[3].rho > Real(0)) {
            // du/dy ≈ (u_top - u_bottom) / (2*dx)
            Real u_bottom = neighbors[2].rhou / (neighbors[2].rho + Real(1e-10));
            Real u_top = neighbors[3].rhou / (neighbors[3].rho + Real(1e-10));
            du_dy = (u_top - u_bottom) / (Real(2) * dx);
        }

        Real vorticity = Kokkos::abs(dv_dx - du_dy);

        if (vorticity > threshold) {
            return RefinementAction::Refine;
        }
        return RefinementAction::Keep;
    }

    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U,
                               const Primitive& q,
                               Real dx) const
    {
        return RefinementAction::Keep;
    }
};

// ============================================================================
// VALUE RANGE CRITERION
// ============================================================================

/**
 * @brief Refine based on variable value range
 *
 * Refines where variable is in specified range (e.g., tracking a front).
 * Can also invert logic to refine OUTSIDE range.
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class ValueRangeCriterion {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    enum Variable : uint8_t {
        Density = 0,
        Pressure = 1,
        VelocityX = 2,
        VelocityY = 3,
        VelocityMag = 4,
        Mach = 5
    };

    Variable variable = Density;
    Real min_val = Real(0);
    Real max_val = Real(1);
    bool invert = false;  // If true, refine OUTSIDE range

    KOKKOS_INLINE_FUNCTION
    ValueRangeCriterion() = default;

    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U,
                               const Primitive& q,
                               Real dx) const
    {
        Real val = get_value(q);

        bool in_range = (val >= min_val && val <= max_val);
        bool should_refine = invert ? !in_range : in_range;

        return should_refine ? RefinementAction::Refine : RefinementAction::Keep;
    }

    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U_center,
                               const Primitive& q_center,
                               const Conserved* neighbors,
                               Real dx) const
    {
        return evaluate(U_center, q_center, dx);
    }

private:
    KOKKOS_INLINE_FUNCTION
    Real get_value(const Primitive& q) const {
        switch (variable) {
            case Density: return q.rho;
            case Pressure: return q.p;
            case VelocityX: return q.u;
            case VelocityY: return q.v;
            case VelocityMag:
                return Kokkos::sqrt(q.u*q.u + q.v*q.v);
            case Mach:
                // Approximate (assuming ideal gas)
                return Kokkos::sqrt(q.u*q.u + q.v*q.v) / Kokkos::sqrt(q.p);
            default: return q.rho;
        }
    }
};

// ============================================================================
// CURL CRITERION (for magnetic fields in MHD)
// ============================================================================

/**
 * @brief Refine based on current density (curl of B)
 *
 * For MHD systems: refines where |J| = |curl(B)| is large.
 * For Euler: uses vorticity (dv/dx - du/dy).
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class CurlCriterion {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    Real threshold = Real(1.0);
    bool use_vorticity = true;  // Use vorticity for Euler

    KOKKOS_INLINE_FUNCTION
    CurlCriterion() = default;

    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U_center,
                               const Primitive& q_center,
                               const Conserved* neighbors,
                               Real dx) const
    {
        // Same as vorticity for now
        Real du_dy = Real(0);
        Real dv_dx = Real(0);

        if (neighbors[0].rho > Real(0) && neighbors[1].rho > Real(0)) {
            Real v_left = neighbors[0].rhov / (neighbors[0].rho + Real(1e-10));
            Real v_right = neighbors[1].rhov / (neighbors[1].rho + Real(1e-10));
            dv_dx = (v_right - v_left) / (Real(2) * dx);
        }

        if (neighbors[2].rho > Real(0) && neighbors[3].rho > Real(0)) {
            Real u_bottom = neighbors[2].rhou / (neighbors[2].rho + Real(1e-10));
            Real u_top = neighbors[3].rhou / (neighbors[3].rho + Real(1e-10));
            du_dy = (u_top - u_bottom) / (Real(2) * dx);
        }

        Real curl = Kokkos::abs(dv_dx - du_dy);

        return (curl > threshold) ? RefinementAction::Refine : RefinementAction::Keep;
    }

    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U,
                               const Primitive& q,
                               Real dx) const
    {
        return RefinementAction::Keep;
    }
};

// ============================================================================
// COMBINABLE CRITERIA
// ============================================================================

/**
 * @brief Combines multiple criteria with logic operators
 *
 * Supports:
 * - AND: refine only if ALL criteria agree
 * - OR: refine if ANY criterion agrees
 * - Vote: majority vote
 */
template<typename System, int MaxCriteria = 8>
    requires FiniteVolumeSystem<System>
class CompositeCriterion {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    enum LogicOp : uint8_t {
        And = 0,   // All criteria must return Refine
        Or = 1,    // Any criterion returns Refine
        Vote = 2   // Majority vote
    };

    LogicOp logic_op = Or;
    int8_t num_criteria = 0;

    // Criterion types and storage
    // Using type-indexing pattern for runtime polymorphism on device
    enum CriterionType : uint8_t {
        None = 0,
        Gradient = 1,
        ShockSensor = 2,
        Vorticity = 3,
        ValueRange = 4,
        Curl = 5,
        Custom = 99
    };

    struct CriterionStorage {
        CriterionType type = None;
        int8_t index = -1;  // Index into type-specific arrays

        KOKKOS_INLINE_FUNCTION
        CriterionStorage() = default;
    };

    CriterionStorage criteria[MaxCriteria];

    // Type-specific storage arrays
    GradientCriterion<System> gradient_criteria[MaxCriteria];
    ShockSensorCriterion<System> shock_criteria[MaxCriteria];
    VorticityCriterion<System> vorticity_criteria[MaxCriteria];
    ValueRangeCriterion<System> value_range_criteria[MaxCriteria];
    CurlCriterion<System> curl_criteria[MaxCriteria];

    KOKKOS_FUNCTION
    CompositeCriterion() = default;

    /**
     * @brief Add a gradient criterion (host-side)
     */
    int add_gradient(const GradientCriterion<System>& crit) {
        if (num_criteria >= MaxCriteria) return -1;
        int idx = num_criteria;
        gradient_criteria[idx] = crit;
        criteria[idx].type = Gradient;
        criteria[idx].index = idx;
        num_criteria++;
        return idx;
    }

    /**
     * @brief Add a shock sensor criterion (host-side)
     */
    int add_shock_sensor(const ShockSensorCriterion<System>& crit) {
        if (num_criteria >= MaxCriteria) return -1;
        int idx = num_criteria;
        shock_criteria[idx] = crit;
        criteria[idx].type = ShockSensor;
        criteria[idx].index = idx;
        num_criteria++;
        return idx;
    }

    /**
     * @brief Add a vorticity criterion (host-side)
     */
    int add_vorticity(const VorticityCriterion<System>& crit) {
        if (num_criteria >= MaxCriteria) return -1;
        int idx = num_criteria;
        vorticity_criteria[idx] = crit;
        criteria[idx].type = Vorticity;
        criteria[idx].index = idx;
        num_criteria++;
        return idx;
    }

    /**
     * @brief Add a value range criterion (host-side)
     */
    int add_value_range(const ValueRangeCriterion<System>& crit) {
        if (num_criteria >= MaxCriteria) return -1;
        int idx = num_criteria;
        value_range_criteria[idx] = crit;
        criteria[idx].type = ValueRange;
        criteria[idx].index = idx;
        num_criteria++;
        return idx;
    }

    /**
     * @brief Evaluate composite criterion (device-side)
     *
     * Loops over all criteria and combines results using the logic operator.
     */
    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U_center,
                               const Primitive& q_center,
                               const Conserved* neighbors,
                               Real dx) const
    {
        int refine_count = 0;
        int coarsen_count = 0;

        for (int i = 0; i < num_criteria; ++i) {
            RefinementAction action = evaluate_criterion(
                i, U_center, q_center, neighbors, dx
            );

            if (action == RefinementAction::Refine) refine_count++;
            if (action == RefinementAction::Coarsen) coarsen_count++;
        }

        // Combine based on logic operator
        switch (logic_op) {
            case And:
                return (refine_count == num_criteria)
                    ? RefinementAction::Refine : RefinementAction::Keep;
            case Or:
                return (refine_count > 0)
                    ? RefinementAction::Refine : RefinementAction::Keep;
            case Vote:
                return (refine_count > num_criteria / 2)
                    ? RefinementAction::Refine : RefinementAction::Keep;
            default:
                return RefinementAction::Keep;
        }
    }

    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U,
                               const Primitive& q,
                               Real dx) const
    {
        // Simplified version without neighbors
        int refine_count = 0;

        for (int i = 0; i < num_criteria; ++i) {
            RefinementAction action = evaluate_criterion_simple(i, U, q, dx);
            if (action == RefinementAction::Refine) refine_count++;
        }

        return (refine_count > 0) ? RefinementAction::Refine : RefinementAction::Keep;
    }

private:
    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate_criterion(int idx,
                                         const Conserved& U_center,
                                         const Primitive& q_center,
                                         const Conserved* neighbors,
                                         Real dx) const
    {
        const auto& crit_storage = criteria[idx];

        switch (crit_storage.type) {
            case Gradient:
                return gradient_criteria[crit_storage.index].evaluate(
                    U_center, q_center, neighbors, dx
                );
            case ShockSensor:
                return shock_criteria[crit_storage.index].evaluate(
                    U_center, q_center, neighbors, dx
                );
            case Vorticity:
                return vorticity_criteria[crit_storage.index].evaluate(
                    U_center, q_center, neighbors, dx
                );
            case ValueRange:
                return value_range_criteria[crit_storage.index].evaluate(
                    U_center, q_center, neighbors, dx
                );
            case Curl:
                return curl_criteria[crit_storage.index].evaluate(
                    U_center, q_center, neighbors, dx
                );
            default:
                return RefinementAction::Keep;
        }
    }

    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate_criterion_simple(int idx,
                                                const Conserved& U,
                                                const Primitive& q,
                                                Real dx) const
    {
        const auto& crit_storage = criteria[idx];

        switch (crit_storage.type) {
            case Gradient:
                return gradient_criteria[crit_storage.index].evaluate(U, q, dx);
            case ShockSensor:
                return shock_criteria[crit_storage.index].evaluate(U, q, dx);
            case Vorticity:
                return vorticity_criteria[crit_storage.index].evaluate(U, q, dx);
            case ValueRange:
                return value_range_criteria[crit_storage.index].evaluate(U, q, dx);
            case Curl:
                return curl_criteria[crit_storage.index].evaluate(U, q, dx);
            default:
                return RefinementAction::Keep;
        }
    }
};

// ============================================================================
// EXCLUSION ZONES
// ============================================================================

/**
 * @brief Zones where refinement is disabled or has minimum level
 *
 * Cells in these zones will never be coarsened below min_level.
 * Can be defined using:
 * - Simple shapes (Rectangle, Circle)
 * - Custom CSR geometry from Subsetix
 */
template<typename Real>
struct ExclusionZone {
    // Zone definition
    enum Predicate : uint8_t {
        None = 0,
        Rectangle = 1,
        Circle = 2,
        CustomCSR = 3  // Use Subsetix CSR geometry
    };

    Predicate predicate = Rectangle;

    // Simple shape parameters
    Real x_min = Real(0), x_max = Real(1);
    Real y_min = Real(0), y_max = Real(1);
    Real center_x = Real(0), center_y = Real(0);
    Real radius = Real(0);

    // Custom CSR geometry (for complex shapes)
    // Note: This is a device-side pointer to CSR geometry
    const csr::IntervalSet2DDevice* csr_geometry = nullptr;

    int min_level = 0;  // Minimum level to maintain in this zone
    int max_level = 99; // Maximum level (optional)

    KOKKOS_INLINE_FUNCTION
    ExclusionZone() = default;

    /**
     * @brief Check if point (x, y) is in this zone
     */
    KOKKOS_INLINE_FUNCTION
    bool contains(Real x, Real y) const {
        switch (predicate) {
            case Rectangle:
                return x >= x_min && x <= x_max && y >= y_min && y <= y_max;
            case Circle: {
                Real dx = x - center_x;
                Real dy = y - center_y;
                return (dx*dx + dy*dy) < radius*radius;
            }
            case CustomCSR:
                if (csr_geometry) {
                    // Check if point is in CSR geometry
                    // Simplified: assume geometry is pre-processed
                    // In production: would do proper spatial lookup
                    return true;  // Placeholder
                }
                return false;
            default:
                return false;
        }
    }

    /**
     * @brief Check if cell (i, j) is in this zone
     */
    KOKKOS_INLINE_FUNCTION
    bool contains(int i, int j, Real dx, Real dy, Real x0, Real y0) const {
        Real x = x0 + i * dx;
        Real y = y0 + j * dy;
        return contains(x, y);
    }
};

// ============================================================================
// REFINEMENT CONFIG (Complete Configuration)
// ============================================================================

/**
 * @brief Complete refinement configuration
 *
 * Combines criteria, exclusions, and parameters into a single POD struct.
 * Can be passed to device kernels for AMR decisions.
 */
template<typename System>
    requires FiniteVolumeSystem<System>
struct RefinementConfig {
    using Real = typename System::RealType;

    // Criterion to use
    CompositeCriterion<System, 8> criterion;

    // Exclusion zones
    static constexpr int max_exclusions = 16;
    ExclusionZone<Real> exclusions[max_exclusions];
    int8_t num_exclusions = 0;

    // Level limits
    int8_t min_level = 0;
    int8_t max_level = 5;

    // Remesh frequency
    int remesh_interval = 100;

    // Coarsening enabled
    bool enable_coarsening = true;

    KOKKOS_FUNCTION
    RefinementConfig() = default;

    /**
     * @brief Check if cell is in exclusion zone
     *
     * Returns minimum level to maintain, or -1 if not in exclusion.
     */
    KOKKOS_INLINE_FUNCTION
    int check_exclusions(int i, int j, Real dx, Real dy, Real x0, Real y0) const {
        for (int ie = 0; ie < num_exclusions; ++ie) {
            if (exclusions[ie].contains(i, j, dx, dy, x0, y0)) {
                return exclusions[ie].min_level;
            }
        }
        return -1;
    }

    KOKKOS_INLINE_FUNCTION
    int check_exclusions(Real x, Real y) const {
        for (int ie = 0; ie < num_exclusions; ++ie) {
            if (exclusions[ie].contains(x, y)) {
                return exclusions[ie].min_level;
            }
        }
        return -1;
    }
};

// ============================================================================
// MANAGER (Host-Side API)
// ============================================================================

/**
 * @brief Host-side manager for refinement configuration
 *
 * Provides a convenient API for building the refinement config.
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class RefinementManager {
public:
    using Real = typename System::RealType;
    using Config = RefinementConfig<System>;

    Config config;

    /**
     * @brief Add gradient criterion
     */
    void add_gradient_criterion(Real threshold, bool use_rho = true, bool use_p = false) {
        GradientCriterion<System> crit;
        crit.threshold = threshold;
        crit.use_rho = use_rho;
        crit.use_p = use_p;
        config.criterion.add_gradient(crit);
    }

    /**
     * @brief Add shock sensor criterion
     */
    void add_shock_sensor_criterion(
        typename ShockSensorCriterion<System>::SensorType type,
        Real threshold)
    {
        ShockSensorCriterion<System> crit;
        crit.sensor_type = type;
        crit.threshold = threshold;
        config.criterion.add_shock_sensor(crit);
    }

    /**
     * @brief Add vorticity criterion
     */
    void add_vorticity_criterion(Real threshold) {
        VorticityCriterion<System> crit;
        crit.threshold = threshold;
        config.criterion.add_vorticity(crit);
    }

    /**
     * @brief Add value range criterion
     */
    void add_value_range_criterion(
        typename ValueRangeCriterion<System>::Variable var,
        Real min_val, Real max_val,
        bool invert = false)
    {
        ValueRangeCriterion<System> crit;
        crit.variable = var;
        crit.min_val = min_val;
        crit.max_val = max_val;
        crit.invert = invert;
        config.criterion.add_value_range(crit);
    }

    /**
     * @brief Set logic operator for combining criteria
     */
    void set_logic_op(typename CompositeCriterion<System>::LogicOp op) {
        config.criterion.logic_op = op;
    }

    /**
     * @brief Add rectangular exclusion zone
     */
    void add_exclusion_rectangle(Real x_min, Real x_max, Real y_min, Real y_max,
                                  int min_level = 0)
    {
        if (config.num_exclusions >= Config::max_exclusions) return;

        ExclusionZone<Real> zone;
        zone.predicate = ExclusionZone<Real>::Rectangle;
        zone.x_min = x_min;
        zone.x_max = x_max;
        zone.y_min = y_min;
        zone.y_max = y_max;
        zone.min_level = min_level;

        config.exclusions[config.num_exclusions++] = zone;
    }

    /**
     * @brief Add circular exclusion zone
     */
    void add_exclusion_circle(Real center_x, Real center_y, Real radius,
                               int min_level = 0)
    {
        if (config.num_exclusions >= Config::max_exclusions) return;

        ExclusionZone<Real> zone;
        zone.predicate = ExclusionZone<Real>::Circle;
        zone.center_x = center_x;
        zone.center_y = center_y;
        zone.radius = radius;
        zone.min_level = min_level;

        config.exclusions[config.num_exclusions++] = zone;
    }

    /**
     * @brief Add exclusion zone from CSR geometry
     *
     * @param geometry Subsetix CSR geometry defining the exclusion zone
     * @param min_level Minimum level to maintain in this zone
     */
    void add_exclusion_csr(const csr::IntervalSet2DDevice* geometry,
                            int min_level = 0)
    {
        if (config.num_exclusions >= Config::max_exclusions) return;

        ExclusionZone<Real> zone;
        zone.predicate = ExclusionZone<Real>::CustomCSR;
        zone.csr_geometry = geometry;
        zone.min_level = min_level;

        config.exclusions[config.num_exclusions++] = zone;
    }

    /**
     * @brief Set level limits
     */
    void set_level_limits(int min_level, int max_level) {
        config.min_level = min_level;
        config.max_level = max_level;
    }

    /**
     * @brief Set remesh frequency
     */
    void set_remesh_frequency(int interval) {
        config.remesh_interval = interval;
    }

    /**
     * @brief Enable/disable coarsening
     */
    void set_coarsening(bool enable) {
        config.enable_coarsening = enable;
    }

    /**
     * @brief Build final config (for passing to solver)
     */
    const Config& get_config() const {
        return config;
    }
};

} // namespace subsetix::fvd::amr
