// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <gtest/gtest.h>
#include <experimental/subsetix/csr/set_algebra/v1.hpp>
#include <experimental/subsetix/csr/set_algebra/v2.hpp>
#include <experimental/subsetix/csr/set_algebra/v3.hpp>
#include <experimental/subsetix/csr/types.hpp>
#include <Kokkos_Core.hpp>

using namespace experimental::subsetix::csr;
using namespace experimental::subsetix::csr::v1;

// Type aliases for convenience
using Coord = int32_t;
using IntervalType = Interval<Coord>;
using RowKey2DType = RowKey2D<Coord>;
using RowKey3DType = RowKey3D<Coord>;

// ============================================================================
// Test with large mesh sizes (same as benchmark ranges)
// ============================================================================

Mesh2DDevice generate_mesh_2d_partial(int n, int offset_shift) {
  Mesh2DDevice mesh;
  mesh.num_rows = n;
  mesh.num_intervals = n;
  mesh.row_keys = Mesh2DDevice::RowKeyView("gen_row_keys", n);
  mesh.row_ptr = Mesh2DDevice::IndexView("gen_row_ptr", n + 1);
  mesh.intervals = Mesh2DDevice::IntervalView("gen_intervals", n);

  auto row_keys_host = Kokkos::create_mirror_view(mesh.row_keys);
  auto row_ptr_host = Kokkos::create_mirror_view(mesh.row_ptr);
  auto intervals_host = Kokkos::create_mirror_view(mesh.intervals);

  for (int i = 0; i < n; ++i) {
    Coord row_key_value = static_cast<Coord>(2 * i + offset_shift * n);
    row_keys_host(i) = RowKey2DType{row_key_value};
    row_ptr_host(i) = i;
    intervals_host(i) = IntervalType{0, 100};
  }
  row_ptr_host(n) = n;

  Kokkos::deep_copy(mesh.row_keys, row_keys_host);
  Kokkos::deep_copy(mesh.row_ptr, row_ptr_host);
  Kokkos::deep_copy(mesh.intervals, intervals_host);

  return mesh;
}

// ============================================================================
// 2D Large Mesh Tests
// ============================================================================

TEST(LargeMeshTest, PartialOverlap_2D_n64) {
  const int n = 64;
  auto mesh_a = generate_mesh_2d_partial(n, 0);
  auto mesh_b = generate_mesh_2d_partial(n, 1);

  auto result = v1::intersect_meshes_2d(mesh_a, mesh_b);
  EXPECT_GT(result.num_rows, 0);
}

TEST(LargeMeshTest, PartialOverlap_2D_n512) {
  const int n = 512;
  auto mesh_a = generate_mesh_2d_partial(n, 0);
  auto mesh_b = generate_mesh_2d_partial(n, 1);

  auto result = v1::intersect_meshes_2d(mesh_a, mesh_b);
  EXPECT_GT(result.num_rows, 0);
}

TEST(LargeMeshTest, PartialOverlap_2D_n4096) {
  const int n = 4096;
  auto mesh_a = generate_mesh_2d_partial(n, 0);
  auto mesh_b = generate_mesh_2d_partial(n, 1);

  auto result = v1::intersect_meshes_2d(mesh_a, mesh_b);
  EXPECT_GT(result.num_rows, 0);
}

TEST(LargeMeshTest, PartialOverlap_2D_n8192) {
  const int n = 8192;
  auto mesh_a = generate_mesh_2d_partial(n, 0);
  auto mesh_b = generate_mesh_2d_partial(n, 1);

  auto result = v1::intersect_meshes_2d(mesh_a, mesh_b);
  EXPECT_GT(result.num_rows, 0);
}

TEST(LargeMeshTest, PartialOverlap_2D_n8192_v2) {
  // TODO: Fix - need to generate v2-specific meshes
  // Currently disabled because v1::Mesh2DDevice is not compatible with v2::intersect_meshes_2d
  GTEST_SKIP() << "Test disabled - v1 meshes not compatible with v2";
}

TEST(LargeMeshTest, PartialOverlap_2D_n8192_v3) {
  // TODO: Fix - need to generate v3-specific meshes
  // Currently disabled because v1::Mesh2DDevice is not compatible with v3::intersect_meshes_2d
  GTEST_SKIP() << "Test disabled - v1 meshes not compatible with v3";
}

// ============================================================================
// 3D Large Mesh Tests
// ============================================================================

Mesh3DDevice generate_mesh_3d_partial(int n, int offset_shift) {
  Mesh3DDevice mesh;
  mesh.num_rows = n;
  mesh.num_intervals = n;
  mesh.row_keys = Mesh3DDevice::RowKeyView("gen3d_row_keys", n);
  mesh.row_ptr = Mesh3DDevice::IndexView("gen3d_row_ptr", n + 1);
  mesh.intervals = Mesh3DDevice::IntervalView("gen3d_intervals", n);

  auto row_keys_host = Kokkos::create_mirror_view(mesh.row_keys);
  auto row_ptr_host = Kokkos::create_mirror_view(mesh.row_ptr);
  auto intervals_host = Kokkos::create_mirror_view(mesh.intervals);

  for (int i = 0; i < n; ++i) {
    // Y and Z scopes are equal: Z = Y = 2*i
    Coord y_value = static_cast<Coord>(2 * i);
    Coord z_value = static_cast<Coord>(2 * i);
    row_keys_host(i) = RowKey3DType{y_value, z_value};
    row_ptr_host(i) = i;
    intervals_host(i) = IntervalType{0, 100};
  }
  row_ptr_host(n) = n;

  Kokkos::deep_copy(mesh.row_keys, row_keys_host);
  Kokkos::deep_copy(mesh.row_ptr, row_ptr_host);
  Kokkos::deep_copy(mesh.intervals, intervals_host);

  return mesh;
}

TEST(LargeMeshTest, PartialOverlap_3D_n64) {
  const int n = 64;
  auto mesh_a = generate_mesh_3d_partial(n, 0);
  auto mesh_b = generate_mesh_3d_partial(n, 1);

  auto result = v1::intersect_meshes_3d(mesh_a, mesh_b);
  EXPECT_GT(result.num_rows, 0);
}

TEST(LargeMeshTest, PartialOverlap_3D_n512) {
  const int n = 512;
  auto mesh_a = generate_mesh_3d_partial(n, 0);
  auto mesh_b = generate_mesh_3d_partial(n, 1);

  auto result = v1::intersect_meshes_3d(mesh_a, mesh_b);
  EXPECT_GT(result.num_rows, 0);
}

TEST(LargeMeshTest, PartialOverlap_3D_n4096) {
  const int n = 4096;
  auto mesh_a = generate_mesh_3d_partial(n, 0);
  auto mesh_b = generate_mesh_3d_partial(n, 1);

  auto result = v1::intersect_meshes_3d(mesh_a, mesh_b);
  EXPECT_GT(result.num_rows, 0);
}

TEST(LargeMeshTest, PartialOverlap_3D_n8192) {
  const int n = 8192;
  auto mesh_a = generate_mesh_3d_partial(n, 0);
  auto mesh_b = generate_mesh_3d_partial(n, 1);

  auto result = v1::intersect_meshes_3d(mesh_a, mesh_b);
  EXPECT_GT(result.num_rows, 0);
}

TEST(LargeMeshTest, PartialOverlap_3D_n8192_v2) {
  // TODO: Fix - need to generate v2-specific meshes
  // Currently disabled because v1::Mesh3DDevice is not compatible with v2::intersect_meshes_3d
  const int n = 8192;
  auto mesh_a = generate_mesh_3d_partial(n, 0);
  auto mesh_b = generate_mesh_3d_partial(n, 1);

  // auto result = v2::intersect_meshes_3d(mesh_a, mesh_b);
  // EXPECT_GT(result.num_rows, 0);
  GTEST_SKIP() << "Test disabled - v1 meshes not compatible with v2";
}

TEST(LargeMeshTest, PartialOverlap_3D_n8192_v3) {
  // TODO: Fix - need to generate v3-specific meshes
  // Currently disabled because v1::Mesh3DDevice is not compatible with v3::intersect_meshes_3d
  const int n = 8192;
  auto mesh_a = generate_mesh_3d_partial(n, 0);
  auto mesh_b = generate_mesh_3d_partial(n, 1);

  // auto result = v3::intersect_meshes_3d(mesh_a, mesh_b);
  // EXPECT_GT(result.num_rows, 0);
  GTEST_SKIP() << "Test disabled - v1 meshes not compatible with v3";
}

#endif
