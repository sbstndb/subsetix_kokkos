// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cstddef>

namespace subsetix::experimental {

// Basic coordinate type for cell indices
using Coord = int32_t;

/**
 * @brief Half-open interval [begin, end) on the X axis.
 *
 * Invariant: begin < end
 */
struct Interval {
  Coord begin = 0;  // Inclusive
  Coord end = 0;    // Exclusive

  KOKKOS_INLINE_FUNCTION
  Coord size() const { return end - begin; }

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
 */
struct RowKey2D {
  Coord y = 0;

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
 * Rows are identified by their (y, z) coordinates.
 */
struct RowKey3D {
  Coord y = 0;
  Coord z = 0;

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

// ============================================================================
// Mesh templates (2D and 3D)
// ============================================================================

/**
 * @brief Base template for CSR-based mesh representation.
 *
 * This is a compressed sparse row (CSR) representation where:
 * - row_keys stores the row coordinates (sorted)
 * - row_ptr stores offsets into the intervals array for each row
 * - intervals stores [begin, end) X-intervals for each row
 *
 * Specializations for DIM=2 (2D) and DIM=3 (3D) are provided below.
 *
 * Invariants:
 * - row_keys.extent(0) == num_rows
 * - row_ptr.extent(0) == num_rows + 1
 * - intervals.extent(0) >= num_intervals
 * - For each row, intervals are sorted and non-overlapping
 * - row_keys are sorted
 */
template <int DIM, class MemorySpace>
class Mesh;

// ============================================================================
// 2D Mesh specialization
// ============================================================================

template <class MemorySpace>
class Mesh<2, MemorySpace> {
public:
  static constexpr int DIM = 2;
  using RowKey = RowKey2D;
  using RowKeyView = Kokkos::View<RowKey*, MemorySpace>;
  using IndexView = Kokkos::View<std::size_t*, MemorySpace>;
  using IntervalView = Kokkos::View<Interval*, MemorySpace>;

  RowKeyView row_keys;     // [num_rows] - y coordinates
  IndexView row_ptr;       // [num_rows + 1] - CSR offsets
  IntervalView intervals;  // [num_intervals] - X-intervals

  std::size_t num_rows = 0;
  std::size_t num_intervals = 0;

  KOKKOS_INLINE_FUNCTION
  Mesh() = default;

  KOKKOS_INLINE_FUNCTION
  Mesh(const Mesh&) = default;

  KOKKOS_INLINE_FUNCTION
  Mesh& operator=(const Mesh&) = default;
};

// ============================================================================
// 3D Mesh specialization
// ============================================================================

template <class MemorySpace>
class Mesh<3, MemorySpace> {
public:
  static constexpr int DIM = 3;
  using RowKey = RowKey3D;
  using RowKeyView = Kokkos::View<RowKey*, MemorySpace>;
  using IndexView = Kokkos::View<std::size_t*, MemorySpace>;
  using IntervalView = Kokkos::View<Interval*, MemorySpace>;

  RowKeyView row_keys;     // [num_rows] - (y,z) coordinates
  IndexView row_ptr;       // [num_rows + 1] - CSR offsets
  IntervalView intervals;  // [num_intervals] - X-intervals

  std::size_t num_rows = 0;
  std::size_t num_intervals = 0;

  KOKKOS_INLINE_FUNCTION
  Mesh() = default;

  KOKKOS_INLINE_FUNCTION
  Mesh(const Mesh&) = default;

  KOKKOS_INLINE_FUNCTION
  Mesh& operator=(const Mesh&) = default;
};

// ============================================================================
// Primary type aliases for common memory spaces
// ============================================================================

// 2D Mesh aliases
template <class MemorySpace>
using Mesh2D = Mesh<2, MemorySpace>;
using Mesh2DDevice = Mesh2D<Kokkos::DefaultExecutionSpace::memory_space>;
using Mesh2DHost = Mesh2D<Kokkos::HostSpace>;

// 3D Mesh aliases
template <class MemorySpace>
using Mesh3D = Mesh<3, MemorySpace>;
using Mesh3DDevice = Mesh3D<Kokkos::DefaultExecutionSpace::memory_space>;
using Mesh3DHost = Mesh3D<Kokkos::HostSpace>;

} // namespace subsetix::experimental
