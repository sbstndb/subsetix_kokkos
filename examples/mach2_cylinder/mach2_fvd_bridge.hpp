#pragma once

/// @file mach2_fvd_bridge.hpp
/// @brief Bridge adapter between CSR fields and FVD abstraction layer
///
/// This file provides the critical adapter layer that enables the gradual migration
/// from CSR-based storage to the FVD (Finite Volume Dynamics) abstraction layer.
///
/// Key Design Decisions:
/// 1. CSR fields remain the primary storage (sparse, efficient)
/// 2. FVD types (Euler2D) are used for type safety and function dispatch
/// 3. Adapter provides bidirectional conversion between CSR and dense representations
/// 4. All functions are GPU-safe (KOKKOS_INLINE_FUNCTION)
///
/// PHASE 0: This is the FIRST file to create - it's the foundation for all phases.

#include <Kokkos_Core.hpp>
#include <subsetix/field/csr_field.hpp>
#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/csr_ops/field_subset.hpp>

// Forward declaration - FVD types will be included later
namespace subsetix::fvd {
    template<typename Real> class Euler2D;
}

namespace mach2::bridge {

using Real = float;
using subsetix::csr::Coord;
using subsetix::csr::Box2D;
using subsetix::csr::IntervalSet2DDevice;
using subsetix::csr::Field2DDevice;
using subsetix::csr::DeviceMemorySpace;
using subsetix::csr::ExecSpace;

// ============================================================================
// ORIGINAL MACH2 TYPES (for compatibility and gradual migration)
// ============================================================================

/// Conserved variables: U = (rho, rhou, rhov, E)
struct Conserved {
    Real rho;
    Real rhou;
    Real rhov;
    Real E;

    KOKKOS_INLINE_FUNCTION
    Conserved() = default;

    KOKKOS_INLINE_FUNCTION
    Conserved(Real r_, Real rhox_, Real rhoy_, Real E_)
        : rho(r_), rhou(rhox_), rhov(rhoy_), E(E_) {}
};

/// Primitive variables: Q = (rho, u, v, p)
struct Primitive {
    Real rho;
    Real u;
    Real v;
    Real p;

    KOKKOS_INLINE_FUNCTION
    Primitive() = default;

    KOKKOS_INLINE_FUNCTION
    Primitive(Real r_, Real u_, Real v_, Real p_)
        : rho(r_), u(u_), v(v_), p(p_) {}
};

/// Views wrapper for SoA (Structure of Arrays) access
struct ConservedViews {
    Kokkos::View<Real*, DeviceMemorySpace> rho;
    Kokkos::View<Real*, DeviceMemorySpace> rhou;
    Kokkos::View<Real*, DeviceMemorySpace> rhov;
    Kokkos::View<Real*, DeviceMemorySpace> E;
};

/// Fields container with CSR geometry
struct ConservedFields {
    Field2DDevice<Real> rho;
    Field2DDevice<Real> rhou;
    Field2DDevice<Real> rhov;
    Field2DDevice<Real> E;

    KOKKOS_INLINE_FUNCTION
    std::size_t size() const { return rho.size(); }

    KOKKOS_INLINE_FUNCTION
    const IntervalSet2DDevice& geometry() const {
        return rho.geometry;
    }
};

// ============================================================================
// TYPE SAFETY VALIDATION (Phase 0.5)
// ============================================================================

namespace type_safety {

/// Compile-time validation: Ensure FVD types are binary compatible
/// This prevents subtle bugs when using FVD functions with CSR data
#ifdef SUBSETIX_FVD_ENABLED
    #include <subsetix/fvd/system/euler2d.hpp>

    using FVDSystem = subsetix::fvd::Euler2D<Real>;
    using FVDConserved = typename FVDSystem::Conserved;
    using FVDPrimitive = typename FVDSystem::Primitive;

    // Binary compatibility checks (compile-time)
    static_assert(sizeof(FVDConserved) == sizeof(Conserved),
                  "FVD::Conserved must be binary compatible with mach2::Conserved");
    static_assert(alignof(FVDConserved) == alignof(Conserved),
                  "FVD::Conserved must have same alignment as mach2::Conserved");

    static_assert(sizeof(FVDPrimitive) == sizeof(Primitive),
                  "FVD::Primitive must be binary compatible with mach2::Primitive");
    static_assert(alignof(FVDPrimitive) == alignof(Primitive),
                  "FVD::Primitive must have same alignment as mach2::Primitive");

    // GPU safety checks
    static_assert(std::is_trivially_copyable_v<Conserved>,
                  "Conserved must be trivially copyable for GPU use");
    static_assert(std::is_trivially_copyable_v<Primitive>,
                  "Primitive must be trivially copyable for GPU use");

    /// Bitwise equivalence test (can be used in device code)
    KOKKOS_INLINE_FUNCTION
    bool binary_equals_conserved(const Conserved& a, const Conserved& b) {
        // Use memcmp-like comparison for exact bitwise equivalence
        return (a.rho == b.rho) &&
               (a.rhou == b.rhou) &&
               (a.rhov == b.rhov) &&
               (a.E == b.E);
    }

#endif // SUBSETIX_FVD_ENABLED

} // namespace type_safety

// ============================================================================
// CSR DENSE ADAPTER (Phase 0 - CRITICAL)
// ============================================================================

/// @brief Adapter for converting between CSR sparse fields and dense views
///
/// This is the CRITICAL component for Phase 0. It enables the use of
/// FVD time integrators (which expect dense arrays) with CSR field storage.
///
/// Design notes:
/// - Temporary dense allocation is acceptable for MVP
/// - Future optimization: CSR-aware integrators (avoid conversion overhead)
/// - Conversion happens on device when possible
template<typename System>
class CSRFieldAdapter {
public:
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    /// @brief Convert CSR ConservedFields to dense Kokkos views
    ///
    /// This creates temporary dense arrays that can be passed to FVD integrators.
    /// The dense views contain only the active cells (not the full bounding box).
    ///
    /// Usage:
    ///   auto [U_dense, geometry] = adapter.to_dense(U_csr);
    ///   rk_step<System, Integrator>(U_dense, dt, t, rhs, ...);
    ///   U_csr = adapter.from_dense(U_dense, geometry);
    struct DenseRepresentation {
        Kokkos::View<Conserved*, DeviceMemorySpace> U;
        Kokkos::View<Primitive*, DeviceMemorySpace> Q;
        IntervalSet2DDevice geometry;
        std::size_t n;
    };

    /// Convert CSR fields to dense representation
    DenseRepresentation to_dense(const ConservedFields& csr,
                                  const std::string& label = "dense") {
        const std::size_t n = csr.size();

        DenseRepresentation dense;
        dense.n = n;
        dense.geometry = csr.geometry;

        // Allocate dense views
        dense.U = Kokkos::View<Conserved*, DeviceMemorySpace>(label + "_U", n);
        dense.Q = Kokkos::View<Primitive*, DeviceMemorySpace>(label + "_Q", n);

        // Copy data from CSR to dense (parallel)
        auto rho = csr.rho.values;
        auto rhou = csr.rhou.values;
        auto rhov = csr.rhov.values;
        auto E = csr.E.values;
        auto U_dense = dense.U;
        auto gamma = System::default_gamma;

        Kokkos::parallel_for(
            "csr_to_dense_conversion",
            Kokkos::RangePolicy<ExecSpace>(0, n),
            KOKKOS_LAMBDA(const std::size_t idx) {
                // Convert from SoA to AoS
                U_dense(idx).rho = rho(idx);
                U_dense(idx).rhou = rhou(idx);
                U_dense(idx).rhov = rhov(idx);
                U_dense(idx).E = E(idx);
            });

        return dense;
    }

    /// Convert dense representation back to CSR fields
    void from_dense(const DenseRepresentation& dense,
                    ConservedFields& csr,
                    const IntervalSet2DDevice& target_geometry) {
        const std::size_t n = dense.n;

        // Sanity check
        if (csr.size() != n) {
            // TODO: Handle geometry changes (remeshing)
            Kokkos::abort("CSR field size mismatch in from_dense");
        }

        auto rho = csr.rho.values;
        auto rhou = csr.rhou.values;
        auto rhov = csr.rhov.values;
        auto E = csr.E.values;
        auto U_dense = dense.U;

        // Copy data from dense to CSR (parallel)
        Kokkos::parallel_for(
            "dense_to_csr_conversion",
            Kokkos::RangePolicy<ExecSpace>(0, n),
            KOKKOS_LAMBDA(const std::size_t idx) {
                // Convert from AoS to SoA
                rho(idx) = U_dense(idx).rho;
                rhou(idx) = U_dense(idx).rhou;
                rhov(idx) = U_dense(idx).rhov;
                E(idx) = U_dense(idx).E;
            });
    }

    /// Lightweight wrapper: Create ConservedViews from ConservedFields
    /// This avoids memory allocation - just wraps existing data
    ConservedViews wrap(const ConservedFields& csr) {
        ConservedViews views;
        views.rho = csr.rho.values;
        views.rhou = csr.rhou.values;
        views.rhov = csr.rhov.values;
        views.E = csr.E.values;
        return views;
    }
};

// ============================================================================
// UTILITY FUNCTIONS (extracted from mach2_cylinder.cpp)
// ============================================================================

KOKKOS_INLINE_FUNCTION
Primitive cons_to_prim(const Conserved& U, Real gamma) {
    constexpr Real eps = static_cast<Real>(1e-12);
    Primitive q;
    q.rho = U.rho;
    const Real inv_rho = static_cast<Real>(1.0) / (U.rho + eps);
    q.u = U.rhou * inv_rho;
    q.v = U.rhov * inv_rho;
    const Real kinetic = static_cast<Real>(0.5) * (q.u * q.u + q.v * q.v);
    const Real pressure = (gamma - static_cast<Real>(1.0)) * (U.E - U.rho * kinetic);
    q.p = (pressure > eps) ? pressure : eps;
    return q;
}

KOKKOS_INLINE_FUNCTION
Conserved prim_to_cons(const Primitive& q, Real gamma) {
    Conserved U;
    const Real kinetic = static_cast<Real>(0.5) * q.rho * (q.u * q.u + q.v * q.v);
    U.rho = q.rho;
    U.rhou = q.rho * q.u;
    U.rhov = q.rho * q.v;
    U.E = q.p / (gamma - static_cast<Real>(1.0)) + kinetic;
    return U;
}

KOKKOS_INLINE_FUNCTION
Real sound_speed(const Primitive& q, Real gamma) {
    constexpr Real eps = static_cast<Real>(1e-12);
    return Kokkos::sqrt(gamma * q.p / (q.rho + eps));
}

// ============================================================================
// UTILITY FUNCTIONS IMPLEMENTATIONS
// ============================================================================

/// @brief Make ConservedFields from geometry
/// @param geom CSR geometry defining the active cells
/// @param base_label Label for Kokkos views (can be empty)
/// @return Initialized ConservedFields with zero values
inline ConservedFields make_conserved_fields(const IntervalSet2DDevice& geom,
                                              const std::string& base_label = "") {
    ConservedFields out;
    const auto label = [&](const char* suffix) {
        return base_label.empty() ? std::string() : (base_label + suffix);
    };
    out.rho = Field2DDevice<Real>(geom, label("_rho"));
    out.rhou = Field2DDevice<Real>(geom, label("_rhou"));
    out.rhov = Field2DDevice<Real>(geom, label("_rhov"));
    out.E = Field2DDevice<Real>(geom, label("_E"));
    return out;
}

/// @brief Compute diagnostics (density, pressure, Mach number)
/// @param U Conserved variables
/// @param density Output density field (can be same as U.rho for in-place)
/// @param pressure Output pressure field
/// @param mach Output Mach number field
/// @param gamma Specific heat ratio
inline void compute_diagnostics(const ConservedFields& U,
                                Field2DDevice<Real>& density,
                                Field2DDevice<Real>& pressure,
                                Field2DDevice<Real>& mach,
                                Real gamma) {
    using namespace subsetix::csr;
    auto rho = U.rho.values;
    auto rhou = U.rhou.values;
    auto rhov = U.rhov.values;
    auto E = U.E.values;
    auto p_out = pressure.values;
    auto m_out = mach.values;

    apply_on_set_device(
        density, density.geometry,
        KOKKOS_LAMBDA(Coord /*x*/, Coord /*y*/,
                      Real& rho_out, std::size_t idx) {
            Conserved s;
            s.rho = rho(idx);
            s.rhou = rhou(idx);
            s.rhov = rhov(idx);
            s.E = E(idx);
            const Primitive q = cons_to_prim(s, gamma);
            const Real a = sound_speed(q, gamma);
            const Real vel = Kokkos::sqrt(q.u * q.u + q.v * q.v);
            rho_out = q.rho;
            p_out(idx) = q.p;
            m_out(idx) = (a > static_cast<Real>(1e-12)) ? (vel / a)
                                                         : static_cast<Real>(0.0);
        });
}

/// @brief Compute total mass in the system
/// @param U Conserved variables
/// @return Total mass (sum of rho over all cells)
inline Real compute_total_mass(const ConservedFields& U) {
    Real total = static_cast<Real>(0.0);
    auto rho = U.rho.values;

    Kokkos::parallel_reduce(
        "mach2_total_mass",
        Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(U.size())),
        KOKKOS_LAMBDA(const int idx, Real& sum) {
          sum += rho(idx);
        },
        total);

    return total;
}

} // namespace mach2::bridge

// ============================================================================
// FVD TYPE ALIASES (for gradual migration)
// ============================================================================

#ifdef SUBSETIX_FVD_ENABLED
    #include <subsetix/fvd/system/euler2d.hpp>

    namespace mach2::fvd_types {
        using System = subsetix::fvd::Euler2D<Real>;
        using Conserved = typename System::Conserved;
        using Primitive = typename System::Primitive;

        // Compatibility: bridge types can be converted to FVD types
        using BridgeConserved = bridge::Conserved;
        using BridgePrimitive = bridge::Primitive;

        // Type alias for the adapter
        template<typename Sys = System>
        using CSRAdapter = bridge::CSRFieldAdapter<Sys>;
    }
#endif
