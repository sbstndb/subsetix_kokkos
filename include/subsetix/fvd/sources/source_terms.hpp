#pragma once

#include <Kokkos_Core.hpp>
#include <type_traits>
#include "../system/concepts_v2.hpp"

namespace subsetix::fvd::sources {

// ============================================================================
// SOURCE TERM CONCEPT (Compile-time interface)
// ============================================================================

/**
 * @brief Concept for a source term (compile-time interface, no virtual functions)
 *
 * A SourceTerm must provide:
 * - compute(U, q, x, y, t) -> Conserved
 * - is_time_dependent() -> bool (optional, defaults to false)
 * - is_spatially_dependent() -> bool (optional, defaults to true)
 *
 * NO VIRTUAL FUNCTIONS - All resolved at compile-time for GPU compatibility
 */
template<typename T>
concept SourceTerm = requires {
    typename T::System;
    typename T::Real;
    typename T::Conserved;
    typename T::Primitive;
} && requires(const T& src,
              const typename T::Conserved& U,
              const typename T::Primitive& q,
              typename T::Real x, typename T::Real y, typename T::Real t) {
    { src.compute(U, q, x, y, t) } -> std::convertible_to<typename T::Conserved>;
};

// ============================================================================
// GRAVITY SOURCE (Compile-time functor)
// ============================================================================

/**
 * @brief Gravity source term - no virtual functions
 *
 * S_momentum_x = 0
 * S_momentum_y = -ρ * g
 * S_energy = -ρ * g · v
 */
template<FiniteVolumeSystem System>
struct GravitySource {
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    Real g_x = Real(0);
    Real g_y = Real(-9.81);

    KOKKOS_INLINE_FUNCTION
    Conserved compute(const Conserved& U, const Primitive& q,
                      Real /*x*/, Real /*y*/, Real /*t*/) const {
        // Source: S = (0, -ρ*g_x, -ρ*g_y, -ρ*g·v)
        Real rho = U.rho;
        Real momentum_x_src = -rho * g_x;
        Real momentum_y_src = -rho * g_y;
        Real energy_src = -(momentum_x_src * q.u + momentum_y_src * q.v);

        return Conserved{0, momentum_x_src, momentum_y_src, energy_src};
    }

    KOKKOS_INLINE_FUNCTION
    static constexpr bool is_time_dependent() { return false; }

    KOKKOS_INLINE_FUNCTION
    static constexpr bool is_spatially_dependent() { return false; }
};

// ============================================================================
// CUSTOM FUNCTION SOURCE (Template-based)
// ============================================================================

/**
 * @brief Custom source term from compile-time function object
 *
 * Usage:
 *   struct MySource {
 *       KOKKOS_INLINE_FUNCTION
 *       Euler2D<float>::Conserved compute(...) const { ... }
 *   };
 *
 *   CustomSource<Euler2D<float>, MySource> src;
 *
 * For lambdas: use the factory function custom_source() which wraps the lambda
 * in a functor type.
 */
template<FiniteVolumeSystem System, typename Func>
struct CustomSource {
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    Func func;

    KOKKOS_INLINE_FUNCTION
    Conserved compute(const Conserved& U, const Primitive& q,
                      Real x, Real y, Real t) const {
        return func(U, q, x, y, t);
    }

    KOKKOS_INLINE_FUNCTION
    static constexpr bool is_time_dependent() { return true; }

    KOKKOS_INLINE_FUNCTION
    static constexpr bool is_spatially_dependent() { return true; }
};

// ============================================================================
// ZONE SOURCE (Source active only in specific region)
// ============================================================================

/**
 * @brief Source that is only active within a rectangular zone
 *
 * Template parameters allow compile-time optimization
 */
template<FiniteVolumeSystem System, typename InnerSource>
struct ZoneSource {
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    Real x_min, x_max, y_min, y_max;
    InnerSource inner_source;

    KOKKOS_INLINE_FUNCTION
    Conserved compute(const Conserved& U, const Primitive& q,
                      Real x, Real y, Real t) const {
        // Check if (x, y) is inside zone
        if (x >= x_min && x <= x_max && y >= y_min && y <= y_max) {
            return inner_source.compute(U, q, x, y, t);
        }
        return Conserved{0, 0, 0, 0};
    }

    KOKKOS_INLINE_FUNCTION
    bool is_time_dependent() const {
        // Forward to inner source (need to call as member)
        InnerSource tmp = inner_source;
        return tmp.is_time_dependent();
    }
};

// ============================================================================
// CIRCULAR ZONE SOURCE
// ============================================================================

template<FiniteVolumeSystem System, typename InnerSource>
struct CircularZoneSource {
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    Real center_x, center_y, radius_sq;
    InnerSource inner_source;

    KOKKOS_INLINE_FUNCTION
    Conserved compute(const Conserved& U, const Primitive& q,
                      Real x, Real y, Real t) const {
        Real dx = x - center_x;
        Real dy = y - center_y;
        if (dx*dx + dy*dy <= radius_sq) {
            return inner_source.compute(U, q, x, y, t);
        }
        return Conserved{0, 0, 0, 0};
    }
};

// ============================================================================
// COMPOSITE SOURCE (Variadic template, compile-time composition)
// ============================================================================

/**
 * @brief Composite source combining multiple sources at compile-time
 *
 * Uses variadic templates to store sources by value (no pointers).
 * All function calls are inlined by the compiler.
 *
 * Usage:
 *   using MySource = CompositeSource<System,
 *       GravitySource<System>,
 *       CustomSource<System, MyFunc>,
 *       ZoneSource<System, SomeSource>
 *   >;
 */
template<FiniteVolumeSystem System, typename... Sources>
struct CompositeSource {
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    std::tuple<Sources...> sources;

    KOKKOS_INLINE_FUNCTION
    CompositeSource() = default;

    KOKKOS_INLINE_FUNCTION
    explicit CompositeSource(Sources... srcs) : sources(srcs...) {}

    KOKKOS_INLINE_FUNCTION
    Conserved compute(const Conserved& U, const Primitive& q,
                      Real x, Real y, Real t) const {
        Conserved total{0, 0, 0, 0};

        // Fold expression: sum all sources (C++17)
        auto add_source = [&](const auto&... srcs) {
            (([&](const auto& s) {
                Conserved contrib = s.compute(U, q, x, y, t);
                total.rho += contrib.rho;
                total.rhou += contrib.rhou;
                total.rhov += contrib.rhov;
                total.E += contrib.E;
            }(srcs)), ...);
        };

        std::apply(add_source, sources);

        return total;
    }

    KOKKOS_INLINE_FUNCTION
    bool is_time_dependent() const {
        // Check if any source is time-dependent
        bool time_dep = false;

        auto check_time = [&](const auto&... srcs) {
            ((time_dep = time_dep || srcs.is_time_dependent()), ...);
        };

        std::apply(check_time, sources);

        return time_dep;
    }

    KOKKOS_INLINE_FUNCTION
    static constexpr std::size_t num_sources() { return sizeof...(Sources); }
};

// ============================================================================
// NULL SOURCE (Empty source for compile-time defaults)
// ============================================================================

template<FiniteVolumeSystem System>
struct NullSource {
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    KOKKOS_INLINE_FUNCTION
    Conserved compute(const Conserved&, const Primitive&,
                      Real, Real, Real) const {
        return Conserved{0, 0, 0, 0};
    }

    KOKKOS_INLINE_FUNCTION
    static constexpr bool is_time_dependent() { return false; }

    KOKKOS_INLINE_FUNCTION
    static constexpr bool is_spatially_dependent() { return false; }
};

// ============================================================================
// CONVENIENCE TYPE ALIASES FOR COMMON COMBINATIONS
// ============================================================================

namespace detail {

// Helper to deduce system type from first source
template<typename First, typename... Rest>
struct get_system_type {
    using type = typename First::System;
};

} // namespace detail

// Single source (just wrap it)
template<SourceTerm S>
using SingleSource = S;

// Two sources
template<SourceTerm S1, SourceTerm S2>
using DualSource = CompositeSource<typename S1::System, S1, S2>;

// Three sources
template<SourceTerm S1, SourceTerm S2, SourceTerm S3>
using TripleSource = CompositeSource<typename S1::System, S1, S2, S3>;

// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================

/**
 * @brief Create a gravity source
 */
template<FiniteVolumeSystem System>
KOKKOS_INLINE_FUNCTION
auto gravity(typename System::RealType g_y = typename System::RealType(-9.81),
             typename System::RealType g_x = typename System::RealType(0)) {
    return GravitySource<System>{g_x, g_y};
}

/**
 * @brief Create a custom source from a functor
 */
template<FiniteVolumeSystem System, typename Func>
KOKKOS_INLINE_FUNCTION
auto custom_source(Func&& func) {
    return CustomSource<System, std::decay_t<Func>>{std::forward<Func>(func)};
}

/**
 * @brief Create a zone-restricted source
 */
template<FiniteVolumeSystem System, typename InnerSource>
KOKKOS_INLINE_FUNCTION
auto zone_source(typename System::RealType x_min, typename System::RealType x_max,
                 typename System::RealType y_min, typename System::RealType y_max,
                 InnerSource&& inner) {
    return ZoneSource<System, std::decay_t<InnerSource>>{
        x_min, x_max, y_min, y_max, std::forward<InnerSource>(inner)
    };
}

/**
 * @brief Create a circular zone source
 */
template<FiniteVolumeSystem System, typename InnerSource>
KOKKOS_INLINE_FUNCTION
auto circular_zone_source(typename System::RealType cx, typename System::RealType cy,
                         typename System::RealType radius, InnerSource&& inner) {
    return CircularZoneSource<System, std::decay_t<InnerSource>>{
        cx, cy, radius*radius, std::forward<InnerSource>(inner)
    };
}

/**
 * @brief Combine multiple sources (variadic, compile-time)
 *
 * Usage: combine_sources<System>(gravity_source, custom_func, ...)
 */
template<FiniteVolumeSystem System, typename... Sources>
KOKKOS_INLINE_FUNCTION
auto combine_sources(Sources&&... srcs) {
    return CompositeSource<System, std::decay_t<Sources>...>{
        std::make_tuple(std::forward<Sources>(srcs)...)
    };
}

// ============================================================================
// HEATING SOURCE EXAMPLE (Device-friendly)
// ============================================================================

/**
 * @brief Point heat source at (cx, cy) with given power
 */
template<FiniteVolumeSystem System>
struct PointHeatSource {
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    Real cx, cy, radius_sq, power, frequency;

    KOKKOS_INLINE_FUNCTION
    Conserved compute(const Conserved&, const Primitive& q,
                      Real x, Real y, Real t) const {
        Real dx = x - cx;
        Real dy = y - cy;
        if (dx*dx + dy*dy <= radius_sq) {
            // Pulsating heat source
            Real heat = power * (1.0f + 0.5f * Kokkos::sin(frequency * t));
            return Conserved{0, 0, 0, heat};
        }
        return Conserved{0, 0, 0, 0};
    }
};

/**
 * @brief Drag force source
 */
template<FiniteVolumeSystem System>
struct DragSource {
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    Real drag_coefficient;

    KOKKOS_INLINE_FUNCTION
    Conserved compute(const Conserved&, const Primitive& q,
                      Real, Real, Real) const {
        Real v_mag = Kokkos::sqrt(q.u * q.u + q.v * q.v);
        Real drag = -drag_coefficient * q.rho * v_mag;

        return Conserved{0, drag * q.u / v_mag, drag * q.v / v_mag, 0};
    }
};

} // namespace subsetix::fvd::sources
