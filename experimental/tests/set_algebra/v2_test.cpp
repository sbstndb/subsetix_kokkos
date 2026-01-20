// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <gtest/gtest.h>
#include <experimental/subsetix/csr/set_algebra.hpp>
#include <Kokkos_Core.hpp>

using namespace experimental::subsetix::csr;

// Note: Kokkos is already initialized by the test framework

// ============================================================================
// v2-Specific Tests
// ============================================================================

TEST(ExperimentalV2Test, EmptyMeshIntersection_2D) {
  Mesh2DDevice A, B;
  A.num_rows = 0;
  A.num_intervals = 0;
  B.num_rows = 0;
  B.num_intervals = 0;

  auto result1 = v2::intersect_meshes_2d(A, B);
  EXPECT_EQ(result1.num_rows, 0);
  EXPECT_EQ(result1.num_intervals, 0);
}

TEST(ExperimentalV2Test, EmptyMeshIntersection_3D) {
  Mesh3DDevice A, B;
  A.num_rows = 0;
  A.num_intervals = 0;
  B.num_rows = 0;
  B.num_intervals = 0;

  auto result = v2::intersect_meshes_3d(A, B);
  EXPECT_EQ(result.num_rows, 0);
  EXPECT_EQ(result.num_intervals, 0);
}

// ============================================================================
// Type Traits
// ============================================================================

TEST(ExperimentalV2Test, WorkspaceTypeTraits) {
  // Test that workspace compiles for both 2D and 3D
  using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

  v2::MeshIntersectionWorkspace<MemorySpace> ws_2d;
  v2::MeshIntersectionWorkspace<MemorySpace> ws_3d;

  // Workspaces should be default constructible
  EXPECT_TRUE(true);
}

// ============================================================================
// Basic API Compilation Tests
// ============================================================================

TEST(ExperimentalV2Test, API_Compiles_2D) {
  // Test that the v2 API compiles correctly for 2D
  Mesh2DDevice A, B;
  A.num_rows = 0;
  B.num_rows = 0;

  // Without workspace (auto-creates workspace)
  auto result1 = v2::intersect_meshes_2d(A, B);
  EXPECT_EQ(result1.num_rows, 0);

  // With workspace
  v2::MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space> ws;
  auto result2 = v2::intersect_meshes_2d(A, B, ws);
  EXPECT_EQ(result2.num_rows, 0);
}

TEST(ExperimentalV2Test, API_Compiles_3D) {
  // Test that the v2 API compiles correctly for 3D
  Mesh3DDevice A, B;
  A.num_rows = 0;
  B.num_rows = 0;

  auto result1 = v2::intersect_meshes_3d(A, B);
  EXPECT_EQ(result1.num_rows, 0);

  v2::MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space> ws;
  auto result2 = v2::intersect_meshes_3d(A, B, ws);
  EXPECT_EQ(result2.num_rows, 0);
}

#endif // SUBSETIX_ENABLE_EXPERIMENTAL
