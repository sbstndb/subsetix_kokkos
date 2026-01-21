// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cstddef>

namespace experimental::subsetix::csr {

// ============================================================================
// Interval template
// ============================================================================

/**
 * @brief Half-open interval [begin, end) on the X axis.
 *
 * Invariant: begin < end for non-empty intervals
 *
 * @tparam CoordType The coordinate type (e.g., int16_t, int32_t, int64_t)
 */
template<class CoordType>
struct Interval {
  using coord_type = CoordType;

  CoordType begin = 0;  // Inclusive start
  CoordType end = 0;    // Exclusive end

  KOKKOS_INLINE_FUNCTION
  CoordType size() const { return end - begin; }

  KOKKOS_INLINE_FUNCTION
  bool empty() const { return begin >= end; }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const Interval& other) const {
    return begin == other.begin && end == other.end;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator!=(const Interval& other) const {
    return !(*this == other);
  }
};

// ============================================================================
// Row key types (2D and 3D)
// ============================================================================

/**
 * @brief Row key for 2D sparse structure.
 *
 * Rows are identified by their y coordinate.
 *
 * @tparam CoordType The coordinate type
 */
template<class CoordType>
struct RowKey2D {
  using coord_type = CoordType;
  CoordType y = 0;

  KOKKOS_INLINE_FUNCTION
  bool operator==(const RowKey2D& other) const {
    return y == other.y;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator!=(const RowKey2D& other) const {
    return !(*this == other);
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<(const RowKey2D& other) const {
    return y < other.y;
  }
};

/**
 * @brief Row key for 3D sparse structure (Y and Z axes).
 *
 * Rows are identified by their (y, z) coordinates with lexicographic ordering.
 *
 * @tparam CoordType The coordinate type
 */
template<class CoordType>
struct RowKey3D {
  using coord_type = CoordType;
  CoordType y = 0;
  CoordType z = 0;

  KOKKOS_INLINE_FUNCTION
  bool operator==(const RowKey3D& other) const {
    return y == other.y && z == other.z;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator!=(const RowKey3D& other) const {
    return !(*this == other);
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<(const RowKey3D& other) const {
    if (y != other.y) {
      return y < other.y;
    }
    return z < other.z;
  }
};

} // namespace experimental::subsetix::csr
