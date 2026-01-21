// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <experimental/subsetix/csr/set_algebra/v1.hpp>
#include <experimental/subsetix/csr/set_algebra/v2.hpp>
#include <experimental/subsetix/csr/set_algebra/v3.hpp>
#include <Kokkos_Core.hpp>

using namespace experimental::subsetix::csr;

// ============================================================================
// Version Wrappers for Unified Testing
// ============================================================================

/**
 * @brief Wrapper for v1 intersection algorithm
 */
struct V1Intersection {
  static constexpr char name[] = "v1";

  static Mesh2DDevice intersect_2d(const Mesh2DDevice& a, const Mesh2DDevice& b) {
    return v1::intersect_meshes_2d(a, b);
  }

  static Mesh3DDevice intersect_3d(const Mesh3DDevice& a, const Mesh3DDevice& b) {
    return v1::intersect_meshes_3d(a, b);
  }

  // No workspace needed
  struct WorkspaceType {
    // Empty for v1
  };

  static WorkspaceType create_workspace() {
    return WorkspaceType();
  }
};

/**
 * @brief Wrapper for v2 intersection algorithm
 */
struct V2Intersection {
  static constexpr char name[] = "v2";

  static Mesh2DDevice intersect_2d(const Mesh2DDevice& a, const Mesh2DDevice& b, v2::MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space>& ws) {
    return v2::intersect_meshes_2d(a, b, ws);
  }

  static Mesh3DDevice intersect_3d(const Mesh3DDevice& a, const Mesh3DDevice& b, v2::MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space>& ws) {
    return v2::intersect_meshes_3d(a, b, ws);
  }

  using WorkspaceType = v2::MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space>;

  static WorkspaceType create_workspace() {
    return WorkspaceType();
  }
};

/**
 * @brief Wrapper for v3 intersection algorithm
 */
struct V3Intersection {
  static constexpr char name[] = "v3";

  static Mesh2DDevice intersect_2d(const Mesh2DDevice& a, const Mesh2DDevice& b) {
    return v3::intersect_meshes_2d(a, b);
  }

  static Mesh3DDevice intersect_3d(const Mesh3DDevice& a, const Mesh3DDevice& b) {
    return v3::intersect_meshes_3d(a, b);
  }

  // No workspace needed
  struct WorkspaceType {
    // Empty for v3
  };

  static WorkspaceType create_workspace() {
    return WorkspaceType();
  }
};

// ============================================================================
// Version List for Typed Tests
// ============================================================================

using IntersectionVersions = ::testing::Types<V1Intersection, V2Intersection, V3Intersection>;

#endif
