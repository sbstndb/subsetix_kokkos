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
 * - Isolated from stable code in csr_ops/
 * - Templated for 2D and 3D meshes
 * - Extensible framework for comparing different algorithms
 * - Disabled by default (SUBSETIX_ENABLE_EXPERIMENTAL=OFF)
 *
 * Usage:
 * @code
 *   #ifdef SUBSETIX_ENABLE_EXPERIMENTAL
 *   #include <subsetix/csr_ops_experimental/set_algebra.hpp>
 *
 *   using namespace subsetix::experimental;
 *   auto result = v1::intersect_meshes<2>(mesh_a, mesh_b);
 *   #endif
 * @endcode
 */

// Core geometry types (Mesh<2>, Mesh<3>)
#include <subsetix/csr_ops_experimental/geometry/mesh.hpp>

// Utility functions
#include <subsetix/csr_ops_experimental/detail/utils.hpp>

// Concepts for algorithm requirements
#include <subsetix/csr_ops_experimental/set_algebra/concepts.hpp>

// v1: Port of subsetix_kokkos_2 intersection algorithm
#include <subsetix/csr_ops_experimental/set_algebra/v1.hpp>

namespace subsetix::experimental {

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
 *   auto result = v1::intersect_meshes<2>(mesh_a, mesh_b);  // 2D
 *   auto result = v1::intersect_meshes<3>(mesh_a, mesh_b);  // 3D
 */
// v1 namespace contains the intersect_meshes function
// No type alias needed - use v1::intersect_meshes<DIM> directly

} // namespace subsetix::experimental
