#pragma once

#include <Kokkos_Core.hpp>
#include <concepts>
#include <type_traits>

namespace subsetix::fvd {

// ============================================================================
// C++20 CONCEPTS FOR FVD SYSTEM
//
// NOTE: Some concepts disabled due to GCC bugs with parameter names in
// requires clauses. Relying on compile-time errors instead.
// ============================================================================

/**
 * @brief Concept for a PDE system used in finite volume methods
 */
template<typename T>
concept FiniteVolumeSystem = std::is_floating_point_v<typename T::RealType>
    && requires {
        typename T::RealType;
        typename T::Conserved;
        typename T::Primitive;
    }
    && requires(T t, typename T::Conserved U, typename T::Primitive q) {
        { T::to_primitive(U) } -> std::same_as<typename T::Primitive>;
        { T::from_primitive(q) } -> std::same_as<typename T::Conserved>;
        { T::flux_phys_x(U, q) } -> std::same_as<typename T::Conserved>;
        { T::flux_phys_y(U, q) } -> std::same_as<typename T::Conserved>;
    };

/**
 * @brief Concept for reconstruction schemes
 */
template<typename T>
concept ReconstructionScheme = std::is_default_constructible_v<T>;

/**
 * @brief Concept for flux schemes
 *
 * DISABLED: Use compile-time errors instead
 */
template<typename T, typename System>
concept FluxScheme = true;  // Disabled: always satisfied

/**
 * @brief Concept for source terms
 *
 * DISABLED: Use compile-time errors instead
 */
template<typename T, typename System>
concept SourceTerm = true;  // Disabled: always satisfied

/**
 * @brief Concept for output writers
 */
template<typename T>
concept OutputWriter = requires(T writer, std::string filename) {
    { writer.write(filename) } -> std::convertible_to<bool>;
};

/**
 * @brief Helper concept for systems with default gamma
 */
template<typename T>
concept HasDefaultGamma = requires {
    { T::default_gamma } -> std::convertible_to<typename T::RealType>;
};

/**
 * @brief Helper concept for systems with runtime parameters
 */
template<typename T>
concept HasRuntimeParameters = requires {
    typename T::Config;
};

} // namespace subsetix::fvd
