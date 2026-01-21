// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

/**
 * @file set_algebra.hpp
 * @brief Experimental set algebra algorithms for CSR meshes.
 *
 * This module provides alternative implementations of set algebra operations
 * (intersection, union, difference, etc.) for 2D and 3D CSR meshes.
 *
 * Design goals:
 * - Isolated from stable code in subsetix/
 * - Templated for 2D and 3D meshes
 * - Templated on coordinate types (int16_t, int32_t, int64_t, etc.)
 * - Templated on index types (uint32_t, uint64_t, etc.)
 * - Extensible framework for comparing different algorithms
 * - Disabled by default (SUBSETIX_ENABLE_EXPERIMENTAL=OFF)
 *
 * Architecture:
 * - types.hpp: Fundamental shared types (Interval, RowKey, CoordTraits)
 * - Each version (v1, v2, v3) has its own Mesh type templated on CoordType/IndexType
 * - Versioned Mesh types allow future versions to use different data structures
 *
 * Current status:
 * - v1, v2, v3 are currently identical (baseline algorithm)
 * - Framework ready for algorithm experimentation
 *
 * Usage:
 * @code
 *   #ifdef SUBSETIX_ENABLE_EXPERIMENTAL
 *   #include <experimental/subsetix/csr/set_algebra.hpp>
 *
 *   using namespace experimental::subsetix::csr::v1;
 *   auto result = intersect_meshes<2>(mesh_a, mesh_b);
 *
 *   // Or with custom coordinate/index types:
 *   using Mesh16 = v1::Mesh2D<int16_t, uint32_t>;
 *   auto mesh_a = ...;
 *   auto mesh_b = ...;
 *   auto result = v1::intersect_meshes<2, int16_t, uint32_t>(mesh_a, mesh_b);
 *   #endif
 * @endcode
 */

// Core types (shared across all versions)
#include <experimental/subsetix/csr/types.hpp>

// Utility functions
#include <experimental/subsetix/csr/detail/utils.hpp>

// v1: Baseline intersection algorithm (5-phase: map, count, scan, fill, compact)
// v1::Mesh<DIM, MemSpace, CoordType, IndexType> is version-specific
#include <experimental/subsetix/csr/set_algebra/v1.hpp>

// v2: Research slot (currently identical to v1)
// v2::Mesh<DIM, MemSpace, CoordType, IndexType> is version-specific
#include <experimental/subsetix/csr/set_algebra/v2.hpp>

// v3: Research slot (currently identical to v1)
// v3::Mesh<DIM, MemSpace, CoordType, IndexType> is version-specific
#include <experimental/subsetix/csr/set_algebra/v3.hpp>

namespace experimental::subsetix::csr {

/**
 * @brief Version 1 - Baseline intersection algorithm.
 *
 * Original subsetix_kokkos_2 algorithm:
 * - Row mapping via binary search (O(log n))
 * - Two-pointer merge for interval intersection
 * - 5-phase pipeline: map, count, scan, fill, compact
 *
 * Works for both 2D and 3D meshes via template parameter.
 * Templated on coordinate type (int16_t, int32_t, int64_t) and index type (uint32_t, uint64_t).
 *
 * Usage:
 *   using namespace v1;
 *   auto result = intersect_meshes<2>(mesh_a, mesh_b);  // 2D, default types
 *   auto result = intersect_meshes<3>(mesh_a, mesh_b);  // 3D, default types
 *
 *   // With custom types:
 *   auto result = intersect_meshes<2, int16_t, uint32_t>(mesh_a, mesh_b);
 *
 * Mesh type aliases:
 *   v1::Mesh2DDevice  - Default: Mesh<2, DeviceSpace, int32_t, std::size_t>
 *   v1::Mesh2DHost    - Default: Mesh<2, HostSpace, int32_t, std::size_t>
 *   v1::Mesh2D<int16_t, uint32_t>  - Custom coordinate/index types
 *   v1::Mesh3DDevice  - Default: Mesh<3, DeviceSpace, int32_t, std::size_t>
 */

/**
 * @brief Version 2 - Research slot for alternative algorithms.
 *
 * Currently identical to v1 (baseline).
 * Intended for algorithm experimentation and comparison.
 *
 * Usage:
 *   using namespace v2;
 *   auto result = intersect_meshes<2>(mesh_a, mesh_b);
 *
 *   // With custom types:
 *   auto result = intersect_meshes<2, int16_t, uint32_t>(mesh_a, mesh_b);
 */

/**
 * @brief Version 3 - Research slot for alternative algorithms.
 *
 * Currently identical to v1 (baseline).
 * Intended for algorithm experimentation and comparison.
 *
 * Usage:
 *   using namespace v3;
 *   auto result = intersect_meshes<2>(mesh_a, mesh_b);
 *
 *   // With custom types:
 *   auto result = intersect_meshes<2, int16_t, uint32_t>(mesh_a, mesh_b);
 */

} // namespace experimental::subsetix::csr
