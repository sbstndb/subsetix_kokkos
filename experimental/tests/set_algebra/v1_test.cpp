// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <gtest/gtest.h>
#include <experimental/subsetix/csr/set_algebra.hpp>
#include <Kokkos_Core.hpp>

using namespace experimental::subsetix::csr;

// Note: Kokkos is already initialized by the test framework
// No need to initialize/finalize in each test

// ============================================================================
// 2D Mesh Tests
// ============================================================================

TEST(ExperimentalV1Test, EmptyMeshIntersection_2D) {
  Mesh2DDevice A, B;
  A.num_rows = 0;
  A.num_intervals = 0;
  B.num_rows = 0;
  B.num_intervals = 0;

  auto result = v1::intersect_meshes_2d(A, B);

  EXPECT_EQ(result.num_rows, 0);
  EXPECT_EQ(result.num_intervals, 0);
}

TEST(ExperimentalV1Test, Mesh2D_TypeTraits) {
  // Test that 2D mesh types are correctly defined
  using MeshType = Mesh<2, Kokkos::DefaultExecutionSpace::memory_space>;

  EXPECT_EQ(MeshType::DIM, 2);

  // Test RowKey type
  constexpr bool has_row_key = std::same_as<typename MeshType::RowKey, RowKey2D>;
  EXPECT_TRUE(has_row_key);
}

// ============================================================================
// 3D Mesh Tests
// ============================================================================

TEST(ExperimentalV1Test, EmptyMeshIntersection_3D) {
  Mesh3DDevice A, B;
  A.num_rows = 0;
  A.num_intervals = 0;
  B.num_rows = 0;
  B.num_intervals = 0;

  auto result = v1::intersect_meshes_3d(A, B);

  EXPECT_EQ(result.num_rows, 0);
  EXPECT_EQ(result.num_intervals, 0);
}

TEST(ExperimentalV1Test, Mesh3D_TypeTraits) {
  // Test that 3D mesh types are correctly defined
  using MeshType = Mesh<3, Kokkos::DefaultExecutionSpace::memory_space>;

  EXPECT_EQ(MeshType::DIM, 3);

  // Test RowKey type
  constexpr bool has_row_key = std::same_as<typename MeshType::RowKey, RowKey3D>;
  EXPECT_TRUE(has_row_key);
}

// ============================================================================
// Memory space conversion
// ============================================================================

TEST(ExperimentalV1Test, Mesh2D_HostDeviceConversion) {
  // Test that mesh_to compiles for 2D
  Mesh2DDevice device_mesh;
  device_mesh.num_rows = 0;
  device_mesh.num_intervals = 0;

  auto host_mesh = v1::mesh_to<2, Kokkos::HostSpace>(device_mesh);

  EXPECT_EQ(host_mesh.num_rows, 0);
  EXPECT_EQ(host_mesh.num_intervals, 0);
}

TEST(ExperimentalV1Test, Mesh3D_HostDeviceConversion) {
  // Test that mesh_to compiles for 3D
  Mesh3DDevice device_mesh;
  device_mesh.num_rows = 0;
  device_mesh.num_intervals = 0;

  auto host_mesh = v1::mesh_to<3, Kokkos::HostSpace>(device_mesh);

  EXPECT_EQ(host_mesh.num_rows, 0);
  EXPECT_EQ(host_mesh.num_intervals, 0);
}

#endif // SUBSETIX_ENABLE_EXPERIMENTAL
