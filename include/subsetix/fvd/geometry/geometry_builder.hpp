#pragma once

#include <Kokkos_Core.hpp>
#include <vector>
#include <string>
#include <functional>
#include "csr_types.hpp"

namespace subsetix::fvd {

// ============================================================================
// GEOMETRY PRIMITIVES
// ============================================================================

/**
 * @brief 2D geometry primitive for building shapes
 */
template<typename Real = float>
struct GeometryPrimitive {
    enum Type : int {
        Box = 0,
        Cylinder = 1,
        Rectangle = 2,
        Polygon = 3,
        Custom = 4
    };

    Type type = Box;
    Real params[6] = {Real(0)};  // Flexible parameter storage
    std::function<bool(Real, Real)> custom_func;  // For custom shapes

    // Factory methods
    static GeometryPrimitive box(Real x_min, Real x_max, Real y_min, Real y_max) {
        GeometryPrimitive g;
        g.type = Box;
        g.params[0] = x_min;
        g.params[1] = x_max;
        g.params[2] = y_min;
        g.params[3] = y_max;
        return g;
    }

    static GeometryPrimitive cylinder(Real center_x, Real center_y, Real radius) {
        GeometryPrimitive g;
        g.type = Cylinder;
        g.params[0] = center_x;
        g.params[1] = center_y;
        g.params[2] = radius;
        return g;
    }

    static GeometryPrimitive rectangle(Real x_min, Real x_max, Real y_min, Real y_max) {
        return box(x_min, x_max, y_min, y_max);
    }

    // Check if point (x, y) is inside this primitive (solid = obstacle)
    KOKKOS_INLINE_FUNCTION
    bool is_inside(Real x, Real y) const {
        switch (type) {
            case Box:
            case Rectangle:
                return x >= params[0] && x <= params[1] &&
                       y >= params[2] && y <= params[3];
            case Cylinder: {
                Real dx = x - params[0];
                Real dy = y - params[1];
                return dx*dx + dy*dy <= params[2]*params[2];
            }
            default:
                return false;
        }
    }
};

// ============================================================================
// GEOMETRY BUILDER
// ============================================================================

/**
 * @brief Fluent API for building 2D computational geometries
 *
 * Provides a clean, user-friendly interface for creating complex
 * geometries with obstacles and boundaries.
 *
 * Example:
 *   Geometry2D<float> geom = Geometry2D<float>::build_box(0, 400, 0, 160)
 *                                  .add_cylinder(100, 80, 20, true);  // Obstacle
 */
template<typename Real = float>
class Geometry2D {
public:
    using Primitive = GeometryPrimitive<Real>;

    // ========================================================================
    // FACTORY METHODS
    // ========================================================================

    /**
     * @brief Start building a rectangular domain
     */
    static Geometry2D build_box(int nx, int ny) {
        Geometry2D geom;
        geom.nx_ = nx;
        geom.ny_ = ny;
        geom.domain_ = csr::Box2D{0, nx, 0, ny};
        return geom;
    }

    static Geometry2D build_box(Real x_min, Real x_max, Real y_min, Real y_max) {
        return build_box(
            static_cast<int>(x_max - x_min),
            static_cast<int>(y_max - y_min)
        );
    }

    // ========================================================================
    // FLUENT API: ADD SHAPES
    // ========================================================================

    /**
     * @brief Add a cylinder obstacle
     *
     * @param center_x X coordinate of cylinder center
     * @param center_y Y coordinate of cylinder center
     * @param radius Cylinder radius
     * @param is_obstacle If true, this is a solid; if false, it's fluid
     */
    Geometry2D& add_cylinder(Real center_x, Real center_y, Real radius,
                              bool is_obstacle = true) {
        Primitive prim = Primitive::cylinder(center_x, center_y, radius);
        if (is_obstacle) {
            obstacles_.push_back(prim);
        } else {
            fluid_regions_.push_back(prim);
        }
        return *this;
    }

    /**
     * @brief Add a rectangular obstacle
     */
    Geometry2D& add_rectangle(Real x_min, Real x_max, Real y_min, Real y_max,
                               bool is_obstacle = true) {
        Primitive prim = Primitive::box(x_min, x_max, y_min, y_max);
        if (is_obstacle) {
            obstacles_.push_back(prim);
        } else {
            fluid_regions_.push_back(prim);
        }
        return *this;
    }

    /**
     * @brief Add a box (alias for rectangle)
     */
    Geometry2D& add_box(Real x_min, Real x_max, Real y_min, Real y_max,
                        bool is_obstacle = true) {
        return add_rectangle(x_min, x_max, y_min, y_max, is_obstacle);
    }

    /**
     * @brief Add custom geometry shape
     */
    Geometry2D& add_custom(std::function<bool(Real, Real)> inside_func,
                           bool is_obstacle = true) {
        Primitive prim;
        prim.type = Primitive::Custom;
        prim.custom_func = std::move(inside_func);
        if (is_obstacle) {
            obstacles_.push_back(prim);
        } else {
            fluid_regions_.push_back(prim);
        }
        return *this;
    }

    // ========================================================================
    // BUILD
    // ========================================================================

    /**
     * @brief Build the CSR geometry representation
     *
     * This converts the high-level geometry description into the
     * CSR interval set format used by the solver.
     *
     * STUB: In production, this would rasterize the geometry and
     * build the CSR structure.
     */
    csr::IntervalSet2DDevice build() const {
        csr::IntervalSet2DDevice fluid;

        // Stub: simple full domain (no obstacles)
        // In production:
        // 1. Rasterize each primitive onto a grid
        // 2. Mark obstacle cells
        // 3. Build CSR interval set from fluid cells
        // 4. Compress storage

        fluid.num_rows = ny_;
        fluid.num_intervals = nx_;  // One interval per row (full domain)
        fluid.row_offsets = Kokkos::View<int*>("row_offsets", ny_ + 1);
        fluid.intervals = Kokkos::View<int*>("intervals", 2 * nx_);

        // Initialize with stub data
        auto row_offsets_host = Kokkos::create_mirror_view(fluid.row_offsets);
        auto intervals_host = Kokkos::create_mirror_view(fluid.intervals);

        for (int j = 0; j <= ny_; ++j) {
            row_offsets_host(j) = j * 2;
        }
        for (int i = 0; i < nx_; ++i) {
            intervals_host(2*i) = 0;      // x_min
            intervals_host(2*i + 1) = nx_; // x_max
        }

        Kokkos::deep_copy(fluid.row_offsets, row_offsets_host);
        Kokkos::deep_copy(fluid.intervals, intervals_host);

        return fluid;
    }

    // ========================================================================
    // ACCESSORS
    // ========================================================================

    const csr::Box2D& domain() const { return domain_; }
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    const std::vector<Primitive>& obstacles() const { return obstacles_; }
    const std::vector<Primitive>& fluid_regions() const { return fluid_regions_; }

private:
    Geometry2D() = default;

    int nx_ = 0, ny_ = 0;
    csr::Box2D domain_{0, 0, 0, 0};
    std::vector<Primitive> obstacles_;
    std::vector<Primitive> fluid_regions_;
};

// ============================================================================
// CONVENIENCE TYPE ALIASES
// ============================================================================

using Geometry2Df = Geometry2D<float>;
using Geometry2Dd = Geometry2D<double>;

} // namespace subsetix::fvd
