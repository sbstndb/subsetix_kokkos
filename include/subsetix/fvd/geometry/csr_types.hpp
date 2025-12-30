#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>

namespace subsetix::fvd::csr {

// ============================================================================
// CSR GEOMETRY TYPES - STUB IMPLEMENTATIONS
// ============================================================================

/**
 * @brief 2D bounding box
 *
 * Defines a rectangular region in 2D space.
 */
struct Box2D {
    int x_min, x_max, y_min, y_max;

    Box2D() = default;
    Box2D(int xmin, int xmax, int ymin, int ymax)
        : x_min(xmin), x_max(xmax), y_min(ymin), y_max(ymax) {}
};

/**
 * @brief 2D interval set on device (CSR compressed storage)
 *
 * This is a stub placeholder. In production, this would be the real
 * subsetix::csr::IntervalSet2DDevice type with full CSR functionality.
 *
 * CSR format: Each row (y-coordinate) has a set of [x_min, x_max] intervals
 * representing the fluid cells in that row.
 */
struct IntervalSet2DDevice {
    std::size_t num_rows = 0;
    std::size_t num_intervals = 0;

    // Kokkos views for CSR storage
    Kokkos::View<int*> row_offsets;  // Offset into intervals for each row
    Kokkos::View<int*> intervals;    // [x_min, x_max] pairs (2 per interval)

    IntervalSet2DDevice() = default;

    // Device-friendly constructor with allocation
    IntervalSet2DDevice(std::size_t rows, std::size_t num_intervals)
        : num_rows(rows)
        , num_intervals(num_intervals)
        , row_offsets("row_offsets", rows + 1)
        , intervals("intervals", 2 * num_intervals)
    {}
};

} // namespace subsetix::fvd::csr
