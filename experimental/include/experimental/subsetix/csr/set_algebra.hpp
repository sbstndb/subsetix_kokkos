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
 * - types.hpp: Fundamental shared types (Interval, RowKey2D, RowKey3D)
 * - mesh.hpp: CSR mesh representation (Mesh<2>, Mesh<3>)
 * - Each version (v1, v2, v3) has its own intersect_meshes function
 * - Versioned algorithms allow independent evolution and comparison
 *
 * Version summary:
 * - v1: Baseline intersection algorithm (5-phase: map, count, scan, fill, compact)
 * - v2: Research slot (currently identical to v1)
 * - v3: Research slot (currently identical to v1)
 *
 * Usage:
 * @code
 *   #ifdef SUBSETIX_ENABLE_EXPERIMENTAL
 *   #include <experimental/subsetix/csr/set_algebra.hpp>
 *
 *   using namespace experimental::subsetix::csr::v1;
 *   auto result = intersect_meshes_2d(mesh_a, mesh_b);
 *
 *   // With custom coordinate/index types:
 *   using Mesh16 = v1::Mesh2D<int16_t, uint32_t>;
 *   auto result = v1::intersect_meshes<2, int16_t, uint32_t>(mesh_a, mesh_b);
 *   #endif
 * @endcode
 */

// Core types (shared across all versions)
#include <experimental/subsetix/csr/types.hpp>

// Utility functions
#include <experimental/subsetix/csr/detail/utils.hpp>

// v1: Baseline intersection algorithm
#include <experimental/subsetix/csr/set_algebra/v1.hpp>

// v2: Research slot
#include <experimental/subsetix/csr/set_algebra/v2.hpp>

// v3: Research slot
#include <experimental/subsetix/csr/set_algebra/v3.hpp>
