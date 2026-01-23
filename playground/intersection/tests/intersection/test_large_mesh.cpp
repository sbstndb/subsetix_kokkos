// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#ifdef SUBSETIX_ENABLE_PLAYGROUND

#include <gtest/gtest.h>
#include <playground/subsetix/csr/intersection/algorithm/baseline.hpp>
#include <playground/subsetix/csr/intersection/algorithm/optimized.hpp>
#include <playground/subsetix/csr/intersection/types.hpp>
#include <Kokkos_Core.hpp>

using namespace playground::subsetix::csr::intersection;

// Type aliases for convenience
using Coord = int32_t;
using IntervalType = Interval<Coord>;
using RowKey2DType = RowKey2D<Coord>;
using RowKey3DType = RowKey3D<Coord>;

// ============================================================================
// Test with large mesh sizes (same as benchmark ranges)
// ============================================================================

// Template mesh generator for any version
template <typename MeshType>
MeshType generate_mesh_2d_partial_impl(int n, int offset_shift) {
  MeshType mesh;
  mesh.num_rows = n;
  mesh.num_intervals = n;
  mesh.row_keys = typename MeshType::RowKeyView("gen_row_keys", n);
  mesh.row_ptr = typename MeshType::IndexView("gen_row_ptr", n + 1);
  mesh.intervals = typename MeshType::IntervalView("gen_intervals", n);

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

// Version-specific wrappers
inline baseline::Mesh2DDevice generate_mesh_2d_partial_baseline(int n, int offset_shift) {
  return generate_mesh_2d_partial_impl<baseline::Mesh2DDevice>(n, offset_shift);
}

inline optimized::Mesh2DDevice generate_mesh_2d_partial_optimized(int n, int offset_shift) {
  return generate_mesh_2d_partial_impl<optimized::Mesh2DDevice>(n, offset_shift);
}

// For backward compatibility with existing baseline tests
inline baseline::Mesh2DDevice generate_mesh_2d_partial(int n, int offset_shift) {
  return generate_mesh_2d_partial_baseline(n, offset_shift);
}

// ============================================================================
// 2D Large Mesh Tests
// ============================================================================

TEST(LargeMeshTest, PartialOverlap_2D_n64) {
  const int n = 64;
  auto mesh_a = generate_mesh_2d_partial(n, 0);
  auto mesh_b = generate_mesh_2d_partial(n, 1);

  auto result = baseline::intersect_meshes_2d(mesh_a, mesh_b);
  EXPECT_GT(result.num_rows, 0);
}

TEST(LargeMeshTest, PartialOverlap_2D_n512) {
  const int n = 512;
  auto mesh_a = generate_mesh_2d_partial(n, 0);
  auto mesh_b = generate_mesh_2d_partial(n, 1);

  auto result = baseline::intersect_meshes_2d(mesh_a, mesh_b);
  EXPECT_GT(result.num_rows, 0);
}

TEST(LargeMeshTest, PartialOverlap_2D_n4096) {
  const int n = 4096;
  auto mesh_a = generate_mesh_2d_partial(n, 0);
  auto mesh_b = generate_mesh_2d_partial(n, 1);

  auto result = baseline::intersect_meshes_2d(mesh_a, mesh_b);
  EXPECT_GT(result.num_rows, 0);
}

TEST(LargeMeshTest, PartialOverlap_2D_n8192) {
  const int n = 8192;
  auto mesh_a = generate_mesh_2d_partial(n, 0);
  auto mesh_b = generate_mesh_2d_partial(n, 1);

  auto result = baseline::intersect_meshes_2d(mesh_a, mesh_b);
  EXPECT_GT(result.num_rows, 0);
}

TEST(LargeMeshTest, PartialOverlap_2D_n8192_optimized) {
  const int n = 8192;
  auto mesh_a = generate_mesh_2d_partial_optimized(n, 0);
  auto mesh_b = generate_mesh_2d_partial_optimized(n, 1);

  auto result = optimized::intersect_meshes_2d(mesh_a, mesh_b);
  EXPECT_GT(result.num_rows, 0);
}

// ============================================================================
// 3D Large Mesh Tests
// ============================================================================

// Template mesh generator for any version (3D)
template <typename MeshType>
MeshType generate_mesh_3d_partial_impl(int n, int offset_shift) {
  MeshType mesh;
  mesh.num_rows = n;
  mesh.num_intervals = n;
  mesh.row_keys = typename MeshType::RowKeyView("gen3d_row_keys", n);
  mesh.row_ptr = typename MeshType::IndexView("gen3d_row_ptr", n + 1);
  mesh.intervals = typename MeshType::IntervalView("gen3d_intervals", n);

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

// Version-specific wrappers
inline baseline::Mesh3DDevice generate_mesh_3d_partial_baseline(int n, int offset_shift) {
  return generate_mesh_3d_partial_impl<baseline::Mesh3DDevice>(n, offset_shift);
}

inline optimized::Mesh3DDevice generate_mesh_3d_partial_optimized(int n, int offset_shift) {
  return generate_mesh_3d_partial_impl<optimized::Mesh3DDevice>(n, offset_shift);
}

// For backward compatibility with existing baseline tests
inline baseline::Mesh3DDevice generate_mesh_3d_partial(int n, int offset_shift) {
  return generate_mesh_3d_partial_baseline(n, offset_shift);
}

TEST(LargeMeshTest, PartialOverlap_3D_n64) {
  const int n = 64;
  auto mesh_a = generate_mesh_3d_partial(n, 0);
  auto mesh_b = generate_mesh_3d_partial(n, 1);

  auto result = baseline::intersect_meshes_3d(mesh_a, mesh_b);
  EXPECT_GT(result.num_rows, 0);
}

TEST(LargeMeshTest, PartialOverlap_3D_n512) {
  const int n = 512;
  auto mesh_a = generate_mesh_3d_partial(n, 0);
  auto mesh_b = generate_mesh_3d_partial(n, 1);

  auto result = baseline::intersect_meshes_3d(mesh_a, mesh_b);
  EXPECT_GT(result.num_rows, 0);
}

TEST(LargeMeshTest, PartialOverlap_3D_n4096) {
  const int n = 4096;
  auto mesh_a = generate_mesh_3d_partial(n, 0);
  auto mesh_b = generate_mesh_3d_partial(n, 1);

  auto result = baseline::intersect_meshes_3d(mesh_a, mesh_b);
  EXPECT_GT(result.num_rows, 0);
}

TEST(LargeMeshTest, PartialOverlap_3D_n8192) {
  const int n = 8192;
  auto mesh_a = generate_mesh_3d_partial(n, 0);
  auto mesh_b = generate_mesh_3d_partial(n, 1);

  auto result = baseline::intersect_meshes_3d(mesh_a, mesh_b);
  EXPECT_GT(result.num_rows, 0);
}

TEST(LargeMeshTest, PartialOverlap_3D_n8192_optimized) {
  const int n = 8192;
  auto mesh_a = generate_mesh_3d_partial_optimized(n, 0);
  auto mesh_b = generate_mesh_3d_partial_optimized(n, 1);

  auto result = optimized::intersect_meshes_3d(mesh_a, mesh_b);
  EXPECT_GT(result.num_rows, 0);
}

#endif
