// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <gtest/gtest.h>
#include <experimental/subsetix/csr/set_algebra/v1.hpp>
#include <experimental/subsetix/csr/set_algebra/v2.hpp>
#include <Kokkos_Core.hpp>

using namespace experimental::subsetix::csr::v1;

// ============================================================================
// Test each overlap pattern individually
// ============================================================================

v1::Mesh2DDevice generate_mesh_2d_pattern(int n, int pattern, int offset_shift) {
  v1::Mesh2DDevice mesh;
  mesh.num_rows = n;
  mesh.num_intervals = n;
  mesh.row_keys = Kokkos::View<csr::RowKey2D<int32_t>*, Kokkos::DefaultExecutionSpace::memory_space>("gen_row_keys", n);
  mesh.row_ptr = Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space>("gen_row_ptr", n + 1);
  mesh.intervals = Kokkos::View<csr::Interval<int32_t>*, Kokkos::DefaultExecutionSpace::memory_space>("gen_intervals", n);

  auto row_keys_host = Kokkos::create_mirror_view(mesh.row_keys);
  auto row_ptr_host = Kokkos::create_mirror_view(mesh.row_ptr);
  auto intervals_host = Kokkos::create_mirror_view(mesh.intervals);

  for (int i = 0; i < n; ++i) {
    int32_t row_key_value;
    switch (pattern) {
      case 0: // FULL_OVERLAP
        row_key_value = static_cast<int32_t>(i);
        break;
      case 1: // PARTIAL_OVERLAP
        row_key_value = static_cast<int32_t>(2 * i + offset_shift * n);
        break;
      case 2: // MINIMAL_OVERLAP
        row_key_value = static_cast<int32_t>(10 * i + offset_shift * (9 * n / 10) * 10);
        break;
      case 3: // NO_OVERLAP
        row_key_value = static_cast<int32_t>(i + offset_shift * n);
        break;
      default:
        row_key_value = static_cast<int32_t>(i);
        break;
    }

    row_keys_host(i) = csr::RowKey2D<int32_t>{row_key_value};
    row_ptr_host(i) = i;
    intervals_host(i) = csr::Interval<int32_t>{0, 100};
  }
  row_ptr_host(n) = n;

  Kokkos::deep_copy(mesh.row_keys, row_keys_host);
  Kokkos::deep_copy(mesh.row_ptr, row_ptr_host);
  Kokkos::deep_copy(mesh.intervals, intervals_host);

  return mesh;
}

TEST(OverlapPatternTest, FullOverlapWorks) {
  const int n = 64;
  auto mesh_a = generate_mesh_2d_pattern(n, 0, 0);  // FULL_OVERLAP
  auto mesh_b = generate_mesh_2d_pattern(n, 0, 0);

  auto result = v1::intersect_meshes_2d(mesh_a, mesh_b);

  EXPECT_GT(result.num_rows, 0);
  EXPECT_GT(result.num_intervals, 0);
  EXPECT_EQ(result.num_rows, n);
  EXPECT_EQ(result.num_intervals, n);
}

TEST(OverlapPatternTest, PartialOverlapWorks) {
  const int n = 64;
  auto mesh_a = generate_mesh_2d_pattern(n, 1, 0);  // PARTIAL_OVERLAP
  auto mesh_b = generate_mesh_2d_pattern(n, 1, 1);

  auto result = v1::intersect_meshes_2d(mesh_a, mesh_b);

  EXPECT_GT(result.num_rows, 0);
  EXPECT_GT(result.num_intervals, 0);
  // Should be ~50% overlap
  EXPECT_GT(result.num_rows, n/4);
  EXPECT_LT(result.num_rows, n);
}

TEST(OverlapPatternTest, MinimalOverlapWorks) {
  const int n = 64;
  auto mesh_a = generate_mesh_2d_pattern(n, 2, 0);  // MINIMAL_OVERLAP
  auto mesh_b = generate_mesh_2d_pattern(n, 2, 1);

  auto result = v1::intersect_meshes_2d(mesh_a, mesh_b);

  EXPECT_GT(result.num_rows, 0);
  EXPECT_GT(result.num_intervals, 0);
  // Should be ~10% overlap
  EXPECT_GT(result.num_rows, n/20);
  EXPECT_LT(result.num_rows, n/3);
}

TEST(OverlapPatternTest, NoOverlapWorks) {
  const int n = 64;
  auto mesh_a = generate_mesh_2d_pattern(n, 3, 0);  // NO_OVERLAP
  auto mesh_b = generate_mesh_2d_pattern(n, 3, 1);

  auto result = v1::intersect_meshes_2d(mesh_a, mesh_b);

  EXPECT_EQ(result.num_rows, 0);
  EXPECT_EQ(result.num_intervals, 0);
}

#endif
