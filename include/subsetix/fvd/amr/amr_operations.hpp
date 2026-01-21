#pragma once

#include <Kokkos_Core.hpp>
#include "../../geometry/csr_interval_set.hpp"
#include "../../field/csr_field.hpp"
#include "../../csr_ops/field_amr.hpp"
#include "../../csr_ops/amr.hpp"
#include "../../csr_ops/set_algebra.hpp"
#include "../system/concepts_v2.hpp"
#include "../system/euler2d.hpp"
#include "../system/advection2d.hpp"

namespace subsetix::fvd::amr {

// Import CSR types for convenience
using subsetix::csr::Field2DDevice;
using subsetix::csr::IntervalSet2DDevice;
using subsetix::csr::DeviceMemorySpace;
using subsetix::csr::HostMemorySpace;

// ============================================================================
// AMR OPERATIONS WRAPPER
// ============================================================================

/**
 * @brief High-level wrapper for AMR operations
 *
 * This class provides a clean, FVD-friendly interface to the low-level
 * CSR AMR operations in csr_ops/field_amr.hpp and csr_ops/amr.hpp.
 *
 * Supported operations:
 * - Prolongation: Coarse → Fine (injection or linear prediction)
 * - Restriction: Fine → Coarse (volume-weighted averaging)
 * - Geometry refinement: Level up (refine cells)
 * - Geometry coarsening: Level down (merge cells)
 *
 * @tparam System The PDE system (must satisfy FiniteVolumeSystem concept)
 */
template<FiniteVolumeSystem System>
class AmrOperations {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using ExecSpace = typename Kokkos::DefaultExecutionSpace;
    using DeviceMemorySpace = typename ExecSpace::memory_space;

    // ========================================================================
    // PROLONGATION (Coarse → Fine)
    // ========================================================================

    /**
     * @brief Prolong field from coarse level to fine level
     *
     * Transfers solution data from a coarse mesh to a refined fine mesh.
     *
     * @param fine_field Destination field on fine mesh (modified)
     * @param coarse_field Source field on coarse mesh
     * @param fine_subset Subset of fine mesh to prolong (use entire geometry for full prolong)
     * @param use_linear_prediction If true, use gradient-based reconstruction; otherwise use injection
     *
     * @note Injection: Direct copy (fast, 1st order accurate)
     *       Linear prediction: Gradient-based (slower, 2nd order accurate)
     *
     * Conservation: Injection conserves exactly, linear prediction is approximately conservative
     */
    static void prolong_level(
        Field2DDevice<Conserved>& fine_field,
        const Field2DDevice<Conserved>& coarse_field,
        bool use_linear_prediction = false
    ) {
        // Use fine_field geometry as the subset
        const IntervalSet2DDevice& fine_subset = fine_field.geometry;
        subsetix::csr::CsrSetAlgebraContext ctx;

        if (use_linear_prediction) {
            // Use gradient-based reconstruction (2nd order)
            subsetix::csr::prolong_field_prediction_on_subset_device(
                fine_field, coarse_field, fine_subset, &ctx
            );
        } else {
            // Use direct injection (1st order, conservative)
            subsetix::csr::prolong_field_on_subset_device(
                fine_field, coarse_field, fine_subset, &ctx
            );
        }
    }

    /**
     * @brief Convenience overload for Views (without geometry wrapper)
     */
    static void prolong_level(
        Kokkos::View<Conserved*>& fine_U,
        const IntervalSet2DDevice& fine_geom,
        const Kokkos::View<Conserved*>& coarse_U,
        const IntervalSet2DDevice& coarse_geom,
        bool use_linear_prediction = false
    ) {
        Field2DDevice<Conserved> fine_field;
        fine_field.geometry = fine_geom;
        fine_field.values = fine_U;

        Field2DDevice<Conserved> coarse_field;
        coarse_field.geometry = coarse_geom;
        coarse_field.values = coarse_U;

        prolong_level(fine_field, coarse_field, fine_geom, use_linear_prediction);
    }

    // ========================================================================
    // RESTRICTION (Fine → Coarse)
    // ========================================================================

    /**
     * @brief Restrict field from fine level to coarse level
     *
     * Transfers solution data from a refined fine mesh to a coarse mesh
     * using volume-weighted averaging (conservative).
     *
     * @param coarse_field Destination field on coarse mesh (modified)
     * @param fine_field Source field on fine mesh
     * @param coarse_subset Subset of coarse mesh to restrict (use entire geometry for full restriction)
     *
     * Conservation: Volume-weighted averaging is exactly conservative
     *
     * Algorithm: For each coarse cell, average the 4 fine children (2x2 in 2D)
     */
    static void restrict_level(
        Field2DDevice<Conserved>& coarse_field,
        const Field2DDevice<Conserved>& fine_field
    ) {
        // Use coarse_field geometry as the subset
        const IntervalSet2DDevice& coarse_subset = coarse_field.geometry;
        subsetix::csr::CsrSetAlgebraContext ctx;
        subsetix::csr::restrict_field_on_subset_device(
            coarse_field, fine_field, coarse_subset, &ctx
        );
    }

    /**
     * @brief Convenience overload for Views (without geometry wrapper)
     */
    static void restrict_level(
        Kokkos::View<Conserved*>& coarse_U,
        const IntervalSet2DDevice& coarse_geom,
        const Kokkos::View<Conserved*>& fine_U,
        const IntervalSet2DDevice& fine_geom
    ) {
        Field2DDevice<Conserved> coarse_field;
        coarse_field.geometry = coarse_geom;
        coarse_field.values = coarse_U;

        Field2DDevice<Conserved> fine_field;
        fine_field.geometry = fine_geom;
        fine_field.values = fine_U;

        restrict_level(coarse_field, fine_field, coarse_geom);
    }

    // ========================================================================
    // GEOMETRY REFINEMENT (Level Up)
    // ========================================================================

    /**
     * @brief Refine geometry to next level (2x refinement in each direction)
     *
     * Creates a refined geometry where each coarse cell becomes 4 fine cells (2x2 in 2D).
     *
     * @param coarse_geometry Input geometry at current level
     * @param fine_geometry Output geometry at next refined level (allocated)
     *
     * Algorithm:
     * - Each row splits into 2 rows: [y, y+1) → [2y, 2y+1], [2y+2, 2y+3)
     * - Each interval splits into 2 intervals: [x, x_end) → [2x, 2x_end), [2x_end, 2x_end*2)
     * - Result: 4x cell count increase
     */
    static void refine_geometry(
        const IntervalSet2DDevice& coarse_geometry,
        IntervalSet2DDevice& fine_geometry
    ) {
        subsetix::csr::CsrSetAlgebraContext ctx;
        subsetix::csr::refine_level_up_device(coarse_geometry, fine_geometry, ctx);
    }

    // ========================================================================
    // GEOMETRY COARSENING (Level Down)
    // ========================================================================

    /**
     * @brief Coarsen geometry to previous level (merge 2x2 cells into 1)
     *
     * Creates a coarse geometry from a refined fine geometry.
     *
     * @param fine_geometry Input geometry at current level
     * @param coarse_geometry Output geometry at previous coarser level (allocated)
     *
     * Algorithm:
     * - Merge pairs of fine rows with same coarse Y: floor(y_f/2)
     * - Merge pairs of fine intervals: floor(x_f/2)
     * - Union of overlapping intervals to maintain CSR invariants
     *
     * @note The fine geometry must be "refinement-compatible" (even number of cells in each block)
     */
    static void coarsen_geometry(
        const IntervalSet2DDevice& fine_geometry,
        IntervalSet2DDevice& coarse_geometry
    ) {
        subsetix::csr::CsrSetAlgebraContext ctx;
        subsetix::csr::project_level_down_device(fine_geometry, coarse_geometry, ctx);
    }

    // ========================================================================
    // UTILITIES
    // ========================================================================

    /**
     * @brief Compute expected cell count ratio between levels
     *
     * @param coarse_level Coarser level index
     * @param fine_level Finer level index
     * @return Expected ratio of fine cells to coarse cells (2^(2 * level_diff) in 2D)
     */
    static std::size_t cell_count_ratio(int coarse_level, int fine_level) {
        int level_diff = fine_level - coarse_level;
        return static_cast<std::size_t>(1) << (2 * level_diff);  // 2^(2*diff) for 2D
    }

    /**
     * @brief Compute expected cell count at a given level
     *
     * @param base_cells Cell count at level 0
     * @param level Target level
     * @return Expected cell count at target level
     */
    static std::size_t expected_cell_count(std::size_t base_cells, int level) {
        return base_cells * cell_count_ratio(0, level);
    }

    /**
     * @brief Check if refinement ratio between levels is compatible
     *
     * @param coarse_cells Cell count at coarse level
     * @param fine_cells Cell count at fine level
     * @return true if fine_cells ≈ 4 * coarse_cells (within 1% tolerance)
     */
    static bool is_refinement_compatible(std::size_t coarse_cells, std::size_t fine_cells) {
        double ratio = static_cast<double>(fine_cells) / static_cast<double>(coarse_cells);
        return (ratio > 3.96 && ratio < 4.04);  // 4.0 ± 1%
    }
};

// ============================================================================
// CONVENIENCE TYPE ALIASES
// ============================================================================

// Common systems
using AmrOpsEuler2Df = AmrOperations<Euler2D<float>>;
using AmrOpsEuler2Dd = AmrOperations<Euler2D<double>>;
using AmrOpsAdvection2Df = AmrOperations<Advection2D<float>>;
using AmrOpsAdvection2Dd = AmrOperations<Advection2D<double>>;

} // namespace subsetix::fvd::amr
