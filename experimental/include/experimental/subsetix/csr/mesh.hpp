// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#include <experimental/subsetix/csr/types.hpp>
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cstddef>

namespace experimental::subsetix::csr {

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
 * @tparam DIM Dimension (2 for 2D, 3 for 3D)
 * @tparam MemorySpace Kokkos memory space
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
  using RowKey = RowKey2D<int32_t>;
  using RowKeyView = Kokkos::View<RowKey*, MemorySpace>;
  using IndexView = Kokkos::View<std::size_t*, MemorySpace>;
  using IntervalView = Kokkos::View<csr::Interval<int32_t>*, MemorySpace>;

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
  using RowKey = RowKey3D<int32_t>;
  using RowKeyView = Kokkos::View<RowKey*, MemorySpace>;
  using IndexView = Kokkos::View<std::size_t*, MemorySpace>;
  using IntervalView = Kokkos::View<csr::Interval<int32_t>*, MemorySpace>;

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

} // namespace experimental::subsetix::csr
