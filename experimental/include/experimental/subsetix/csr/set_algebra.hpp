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
 * - Extensible framework for comparing different algorithms
 * - Disabled by default (SUBSETIX_ENABLE_EXPERIMENTAL=OFF)
 *
 * Usage:
 * @code
 *   #ifdef SUBSETIX_ENABLE_EXPERIMENTAL
 *   #include <experimental/subsetix/csr/set_algebra.hpp>
 *
 *   using namespace experimental::subsetix::csr::v1;
 *   auto result = intersect_meshes<2>(mesh_a, mesh_b);
 *   #endif
 * @endcode
 */

// Core geometry types (Mesh<2>, Mesh<3>)
#include <experimental/subsetix/csr/mesh.hpp>

// Utility functions
#include <experimental/subsetix/csr/detail/utils.hpp>

// v1: Port of subsetix_kokkos_2 intersection algorithm
#include <experimental/subsetix/csr/set_algebra/v1.hpp>

// v2: Optimized intersection with hash-based row mapping and workspace
#include <experimental/subsetix/csr/set_algebra/v2.hpp>

namespace experimental::subsetix::csr {

/**
 * @brief Version 1 of set intersection algorithm.
 *
 * This is the original subsetix_kokkos_2 algorithm:
 * - Row mapping via binary search
 * - Two-pointer merge for interval intersection
 * - 5-phase: map, count, scan, fill, compact
 *
 * Works for both 2D and 3D meshes via template parameter.
 *
 * Usage:
 *   using namespace v1;
 *   auto result = intersect_meshes<2>(mesh_a, mesh_b);  // 2D
 *   auto result = intersect_meshes<3>(mesh_a, mesh_b);  // 3D
 */

/**
 * @brief Version 2 of set intersection algorithm.
 *
 * Optimized implementation addressing v1 bottlenecks:
 * - Hash-based row mapping (O(1) instead of O(log n))
 * - Single-pass intersection (no separate count+fill)
 * - Reusable workspace (eliminates allocations)
 * - No host synchronization
 *
 * Usage:
 *   using namespace v2;
 *   MeshIntersectionWorkspace<MemorySpace> ws;
 *   auto result = intersect_meshes<2>(mesh_a, mesh_b, ws);
 *
 * For single operations, workspace is created automatically:
 *   auto result = intersect_meshes<2>(mesh_a, mesh_b);
 */

} // namespace experimental::subsetix::csr
