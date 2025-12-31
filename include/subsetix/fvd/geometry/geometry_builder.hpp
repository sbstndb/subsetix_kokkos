#pragma once

#include <Kokkos_Core.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/csr_ops/set_algebra.hpp>
#include <subsetix/csr_ops/workspace.hpp>

namespace subsetix::fvd {

// ============================================================================
// FORWARD DECLARATIONS - Using subsetix::csr types
// ============================================================================

using IntervalSet2DDevice = subsetix::csr::IntervalSet2D<subsetix::csr::DeviceMemorySpace>;
using CsrSetAlgebraContext = subsetix::csr::CsrSetAlgebraContext;

// ============================================================================
// GEOMETRY PRIMITIVES
// ============================================================================

/**
 * @brief 2D geometry primitive for building shapes
 *
 * Stores geometry parameters in physical coordinates (meters).
 * Will be converted to grid indices during build().
 */
template<typename Real = float>
struct GeometryPrimitive {
    enum Type : int {
        Box = 0,
        Cylinder = 1,
        Rectangle = 2
        // Custom = 4  // TODO: Add later with bitmap-based approach
    };

    Type type = Box;
    Real params[6] = {Real(0)};  // Flexible parameter storage (physical coords)

    // Factory methods - physical coordinates (meters)
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
};

// ============================================================================
// GEOMETRY BUILDER
// ============================================================================

/**
 * @brief Fluent API for building 2D computational geometries
 *
 * Provides a clean, user-friendly interface for creating complex
 * geometries with obstacles and boundaries using physical coordinates.
 *
 * The builder stores primitives in physical coordinates (meters) and
 * converts them to grid indices during build().
 *
 * Example:
 *   Geometry2D<float> geom = Geometry2D<float>::build_box(400, 160, 0.005f, 0.005f)
 *                                  .add_cylinder(0.5f, 0.4f, 0.1f, true);  // 50cm radius cylinder
 *   auto fluid_geom = geom.build();
 */
template<typename Real = float>
class Geometry2D {
public:
    using Primitive = GeometryPrimitive<Real>;
    using CSRGeometry = IntervalSet2DDevice;

    // ========================================================================
    // FACTORY METHODS
    // ========================================================================

    /**
     * @brief Start building a rectangular domain
     *
     * @param nx Number of cells in x-direction
     * @param ny Number of cells in y-direction
     * @param dx Cell size in x-direction (meters)
     * @param dy Cell size in y-direction (meters)
     */
    static Geometry2D build_box(int nx, int ny, Real dx = Real(1), Real dy = Real(1)) {
        Geometry2D geom;
        geom.nx_ = nx;
        geom.ny_ = ny;
        geom.dx_ = dx;
        geom.dy_ = dy;
        return geom;
    }

    /**
     * @brief Start building from physical domain size
     *
     * @param x_min, x_max Physical domain bounds in x (meters)
     * @param y_min, y_max Physical domain bounds in y (meters)
     * @param dx, dy Cell sizes (meters)
     */
    static Geometry2D build_box(Real x_min, Real x_max, Real y_min, Real y_max,
                                Real dx, Real dy) {
        int nx = static_cast<int>((x_max - x_min) / dx);
        int ny = static_cast<int>((y_max - y_min) / dy);
        return build_box(nx, ny, dx, dy);
    }

    // ========================================================================
    // FLUENT API: ADD SHAPES
    // ========================================================================

    /**
     * @brief Add a cylinder obstacle (physical coordinates)
     *
     * @param center_x X coordinate of cylinder center (meters)
     * @param center_y Y coordinate of cylinder center (meters)
     * @param radius Cylinder radius (meters)
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
     * @brief Add a rectangular obstacle (physical coordinates)
     *
     * @param x_min, x_max X bounds (meters)
     * @param y_min, y_max Y bounds (meters)
     * @param is_obstacle If true, this is a solid; if false, it's fluid
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

    // ========================================================================
    // BUILD
    // ========================================================================

    /**
     * @brief Build the CSR geometry representation
     *
     * Converts physical primitives to grid indices and uses subsetix::csr
     * CSG operations to construct the final fluid geometry.
     *
     * @return IntervalSet2DDevice CSR geometry for fluid region
     */
    CSRGeometry build() const {
        CsrSetAlgebraContext ctx;  // Local workspace
        return build_impl(ctx);
    }

    /**
     * @brief Build with reusable workspace (optimization for repeated builds)
     *
     * @param ctx Reusable workspace for CSG operations
     * @return IntervalSet2DDevice CSR geometry for fluid region
     */
    CSRGeometry build(CsrSetAlgebraContext& ctx) const {
        return build_impl(ctx);
    }

    // ========================================================================
    // ACCESSORS
    // ========================================================================

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    Real dx() const { return dx_; }
    Real dy() const { return dy_; }
    const std::vector<Primitive>& obstacles() const { return obstacles_; }
    const std::vector<Primitive>& fluid_regions() const { return fluid_regions_; }

private:
    Geometry2D() = default;

    /**
     * @brief Common implementation for build()
     */
    CSRGeometry build_impl(CsrSetAlgebraContext& ctx) const {
        using namespace subsetix::csr;

        // 1. Start with full domain (all cells initially fluid)
        Box2D domain_box{0, nx_, 0, ny_};  // Grid indices
        auto fluid = make_box_device(domain_box);

        if (obstacles_.empty() && fluid_regions_.empty()) {
            compute_cell_offsets_device(fluid);
            return fluid;
        }

        // 2. Subtract all obstacles: fluid = fluid - obstacle
        for (const auto& prim : obstacles_) {
            auto obstacle_geom = primitive_to_csr(prim);

            if (obstacle_geom.num_rows == 0) continue;  // Skip empty

            // Allocate result with capacity for both sets
            auto result = allocate_interval_set_device(
                fluid.num_rows,
                fluid.num_intervals + obstacle_geom.num_intervals
            );

            set_difference_device(fluid, obstacle_geom, result, ctx);
            fluid = result;
        }

        // 3. Add additional fluid regions: fluid = fluid U region
        for (const auto& prim : fluid_regions_) {
            auto add_geom = primitive_to_csr(prim);

            if (add_geom.num_rows == 0) continue;  // Skip empty

            auto result = allocate_interval_set_device(
                fluid.num_rows,
                fluid.num_intervals + add_geom.num_intervals
            );

            set_union_device(fluid, add_geom, result, ctx);
            fluid = result;
        }

        compute_cell_offsets_device(fluid);
        return fluid;
    }

    /**
     * @brief Convert GeometryPrimitive to subsetix CSR geometry
     *
     * Converts physical coordinates (meters) to grid indices (cell units).
     * Clamps results to domain bounds [0, nx_) x [0, ny_).
     */
    subsetix::csr::IntervalSet2DDevice primitive_to_csr(const Primitive& prim) const {
        using namespace subsetix::csr;

        switch (prim.type) {
            case Primitive::Box:
            case Primitive::Rectangle: {
                // Convert physical coords to grid indices
                int ix_min = static_cast<int>(prim.params[0] / dx_);
                int ix_max = static_cast<int>(prim.params[1] / dx_);
                int iy_min = static_cast<int>(prim.params[2] / dy_);
                int iy_max = static_cast<int>(prim.params[3] / dy_);

                // Clamp to domain bounds
                ix_min = std::max(0, std::min(nx_, ix_min));
                ix_max = std::max(0, std::min(nx_, ix_max));
                iy_min = std::max(0, std::min(ny_, iy_min));
                iy_max = std::max(0, std::min(ny_, iy_max));

                if (ix_min >= ix_max || iy_min >= iy_max) {
                    return IntervalSet2DDevice{};  // Empty or out of bounds
                }

                Box2D box{ix_min, ix_max, iy_min, iy_max};
                return make_box_device(box);
            }

            case Primitive::Cylinder: {
                // Convert center and radius to grid indices
                int icx = static_cast<int>(prim.params[0] / dx_);
                int icy = static_cast<int>(prim.params[1] / dy_);
                int ir = static_cast<int>(prim.params[2] / dx_);  // Use dx for radius

                if (ir <= 0) {
                    return IntervalSet2DDevice{};  // Invalid radius
                }

                Disk2D disk{icx, icy, ir};
                return make_disk_device(disk);
            }

            default:
                return IntervalSet2DDevice{};  // Unsupported type
        }
    }

    int nx_ = 0, ny_ = 0;           // Number of cells in x, y
    Real dx_ = Real(1);              // Cell size in x (meters)
    Real dy_ = Real(1);              // Cell size in y (meters)

    std::vector<Primitive> obstacles_;
    std::vector<Primitive> fluid_regions_;
};

// ============================================================================
// CONVENIENCE TYPE ALIASES
// ============================================================================

using Geometry2Df = Geometry2D<float>;
using Geometry2Dd = Geometry2D<double>;

} // namespace subsetix::fvd
