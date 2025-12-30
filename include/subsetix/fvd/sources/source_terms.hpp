#pragma once

#include <Kokkos_Core.hpp>
#include <functional>
#include <vector>
#include "../system/concepts_v2.hpp"

namespace subsetix::fvd::sources {

// ============================================================================
// SOURCE TERM BASE CONCEPT
// ============================================================================

/**
 * @brief Base class for source terms
 *
 * Source terms add contributions to the right-hand side:
 * dU/dt = -∇·F + S(U, x, y, t)
 *
 * Common sources: gravity, chemistry, heat, custom
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class SourceTermBase {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    SourceTermBase() = default;
    virtual ~SourceTermBase() = default;

    /**
     * @brief Compute source term at given location and time
     *
     * @param U Conserved variables
     * @param q Primitive variables
     * @param x X coordinate
     * @param y Y coordinate
     * @param t Current time
     * @return Source contribution to dU/dt
     */
    KOKKOS_INLINE_FUNCTION
    virtual Conserved compute(const Conserved& U, const Primitive& q,
                              Real x, Real y, Real t) const = 0;

    /**
     * @brief Check if source is time-dependent
     */
    KOKKOS_INLINE_FUNCTION
    virtual bool is_time_dependent() const { return false; }

    /**
     * @brief Check if source is spatially dependent
     */
    KOKKOS_INLINE_FUNCTION
    virtual bool is_spatially_dependent() const { return false; }
};

// ============================================================================
// GRAVITY SOURCE
// ============================================================================

/**
 * @brief Gravity source term
 *
 * Adds gravitational acceleration to momentum equations:
 * S_momentum_x = 0
 * S_momentum_y = -ρ * g
 * S_energy = -ρ * g * v
 *
 * For Euler equations in conservative form.
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class GravitySource : public SourceTermBase<System> {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    /**
     * @brief Construct gravity source
     *
     * @param g_y Gravitational acceleration in y-direction (default: -9.81)
     * @param g_x Gravitational acceleration in x-direction (default: 0)
     */
    KOKKOS_INLINE_FUNCTION
    explicit GravitySource(Real g_y = Real(-9.81), Real g_x = Real(0))
        : g_x_(g_x), g_y_(g_y) {}

    KOKKOS_INLINE_FUNCTION
    Conserved compute(const Conserved& U, const Primitive& q,
                      Real /*x*/, Real /*y*/, Real /*t*/) const override {
        // Source: S = (0, -ρ*g_x, -ρ*g_y, -ρ*g·v)
        Real rho = U.rho;
        Real momentum_x_src = -rho * g_x_;
        Real momentum_y_src = -rho * g_y_;
        Real energy_src = -(momentum_x_src * q.u + momentum_y_src * q.v);

        return Conserved{0, momentum_x_src, momentum_y_src, energy_src};
    }

    KOKKOS_INLINE_FUNCTION
    bool is_spatially_dependent() const override { return false; }

    KOKKOS_INLINE_FUNCTION
    Real g_x() const { return g_x_; }
    KOKKOS_INLINE_FUNCTION
    Real g_y() const { return g_y_; }

private:
    Real g_x_ = Real(0);
    Real g_y_ = Real(-9.81);
};

// ============================================================================
// CUSTOM FUNCTION SOURCE
// ============================================================================

/**
 * @brief Custom source term from user-provided function
 *
 * For GPU: Use template parameter for compile-time function
 * For Host: Use std::function for runtime flexibility
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class CustomSource : public SourceTermBase<System> {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    // Function pointer type (host-side)
    using SourceFuncPtr = Conserved(*)(const Conserved&, const Primitive&, Real, Real, Real);

    /**
     * @brief Construct from function pointer (host-only)
     */
    explicit CustomSource(SourceFuncPtr func,
                          bool time_dep = false,
                          bool spatial_dep = true)
        : func_ptr_(func), time_dep_(time_dep), spatial_dep_(spatial_dep), has_func_ptr_(true) {}

    KOKKOS_INLINE_FUNCTION
    Conserved compute(const Conserved& U, const Primitive& q,
                      Real x, Real y, Real t) const override {
        if (has_func_ptr_ && func_ptr_) {
            return func_ptr_(U, q, x, y, t);
        }
        // Default: no source
        return Conserved{0, 0, 0, 0};
    }

    KOKKOS_INLINE_FUNCTION
    bool is_time_dependent() const override { return time_dep_; }

    KOKKOS_INLINE_FUNCTION
    bool is_spatially_dependent() const override { return spatial_dep_; }

private:
    SourceFuncPtr func_ptr_ = nullptr;
    bool time_dep_ = false;
    bool spatial_dep_ = false;
    bool has_func_ptr_ = false;
};

// ============================================================================
// COMPOSITE SOURCE (Multiple sources combined)
// ============================================================================

/**
 * @brief Combines multiple source terms
 *
 * Total source = sum of all sources: S_total = Σ S_i
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class CompositeSource : public SourceTermBase<System> {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    KOKKOS_FUNCTION
    CompositeSource() = default;

    KOKKOS_FUNCTION
    void add(const SourceTermBase<System>* source) {
        if (num_sources_ < max_sources_) {
            sources_[num_sources_++] = source;
        }
    }

    KOKKOS_INLINE_FUNCTION
    Conserved compute(const Conserved& U, const Primitive& q,
                      Real x, Real y, Real t) const override {
        Conserved total{0, 0, 0, 0};

        for (int i = 0; i < num_sources_; ++i) {
            if (sources_[i]) {
                Conserved s = sources_[i]->compute(U, q, x, y, t);
                total.rho += s.rho;
                total.rhou += s.rhou;
                total.rhov += s.rhov;
                total.E += s.E;
            }
        }

        return total;
    }

    KOKKOS_INLINE_FUNCTION
    bool is_time_dependent() const override {
        for (int i = 0; i < num_sources_; ++i) {
            if (sources_[i] && sources_[i]->is_time_dependent()) {
                return true;
            }
        }
        return false;
    }

    KOKKOS_INLINE_FUNCTION
    int num_sources() const { return num_sources_; }

private:
    static constexpr int max_sources_ = 8;
    const SourceTermBase<System>* sources_[max_sources_] = {nullptr};
    int num_sources_ = 0;
};

// ============================================================================
// SOURCE MANAGER (RAII wrapper for composite source)
// ============================================================================

/**
 * @brief RAII manager for source terms
 *
 * Handles ownership of source objects.
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class SourceManager {
public:
    using Real = typename System::RealType;

    SourceManager() = default;
    ~SourceManager() = default;

    // Disable copy
    SourceManager(const SourceManager&) = delete;
    SourceManager& operator=(const SourceManager&) = delete;

    // Enable move
    SourceManager(SourceManager&&) = default;
    SourceManager& operator=(SourceManager&&) = default;

    /**
     * @brief Add gravity source
     */
    void add_gravity(Real g_y = Real(-9.81), Real g_x = Real(0)) {
        auto grav = std::make_unique<GravitySource<System>>(g_y, g_x);
        sources_.push_back(std::move(grav));
    }

    /**
     * @brief Add custom source from lambda
     */
    template<typename Func>
    void add_custom(Func&& func, bool time_dep = false, bool spatial_dep = true) {
        auto custom = std::make_unique<CustomSource<System>>(
            std::forward<Func>(func), time_dep, spatial_dep);
        sources_.push_back(std::move(custom));
    }

    /**
     * @brief Build composite source for solver use
     */
    CompositeSource<System> build() const {
        CompositeSource<System> comp;
        for (const auto& src : sources_) {
            comp.add(src.get());
        }
        return comp;
    }

    /**
     * @brief Check if any source is time-dependent
     */
    bool has_time_dependent() const {
        for (const auto& src : sources_) {
            if (src->is_time_dependent()) return true;
        }
        return false;
    }

    /**
     * @brief Get number of sources
     */
    std::size_t size() const { return sources_.size(); }

private:
    std::vector<std::unique_ptr<SourceTermBase<System>>> sources_;
};

// ============================================================================
// CONVENIENCE FACTORIES
// ============================================================================

/**
 * @brief Create gravity source
 */
template<typename System>
    requires FiniteVolumeSystem<System>
inline auto gravity(typename System::RealType g_y = typename System::RealType(-9.81)) {
    return GravitySource<System>(g_y);
}

/**
 * @brief Create custom source from lambda
 */
template<typename System, typename Func>
    requires FiniteVolumeSystem<System>
inline auto custom_source(Func&& func) {
    return CustomSource<System>(std::forward<Func>(func));
}

} // namespace subsetix::fvd::sources
