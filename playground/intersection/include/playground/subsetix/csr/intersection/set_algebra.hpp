// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

/**
 * @file set_algebra.hpp
 * @brief Playground intersection algorithms for CSR meshes.
 *
 * This module provides intersection algorithms for 2D and 3D CSR meshes.
 *
 * Design goals:
 * - Isolated from stable code in subsetix/
 * - Templated for 2D and 3D meshes
 * - Templated on coordinate types (int16_t, int32_t, int64_t, etc.)
 * - Templated on index types (uint32_t, uint64_t, etc.)
 * - Extensible framework for comparing different algorithms
 *
 * Version summary:
 * - v1: Baseline intersection algorithm (5-phase: map, count, scan, fill, compact)
 * - v2: Research slot (currently identical to v1)
 * - v3: Research slot (currently identical to v1)
 *
 * Usage:
 * @code
 *   #ifdef SUBSETIX_ENABLE_PLAYGROUND
 *   #include <playground/subsetix/csr/intersection/set_algebra.hpp>
 *
 *   using namespace playground::subsetix::csr::intersection::v1;
 *   auto result = intersect_meshes_2d(mesh_a, mesh_b);
 *
 *   // With custom coordinate/index types:
 *   using Mesh16 = v1::Mesh2D<int16_t, uint32_t>;
 *   auto result = v1::intersect_meshes<2, int16_t, uint32_t>(mesh_a, mesh_b);
 *   #endif
 * @endcode
 */

// Core types (shared across all versions)
#include <playground/subsetix/csr/intersection/types.hpp>

// Utility functions
#include <playground/subsetix/csr/intersection/detail/utils.hpp>

// v1: Baseline intersection algorithm
#include <playground/subsetix/csr/intersection/algorithm/v1.hpp>

// v2: Research slot
#include <playground/subsetix/csr/intersection/algorithm/v2.hpp>

// v3: Research slot
#include <playground/subsetix/csr/intersection/algorithm/v3.hpp>
