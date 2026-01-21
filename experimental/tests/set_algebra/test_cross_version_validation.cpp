// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <gtest/gtest.h>
#include <experimental/subsetix/csr/set_algebra/v1.hpp>
#include <experimental/subsetix/csr/set_algebra/v2.hpp>
#include <experimental/subsetix/csr/set_algebra/v3.hpp>
#include <Kokkos_Core.hpp>

using namespace experimental::subsetix::csr;

// ============================================================================
// Cross-Version Validation: Verify v1, v2, v3 produce identical results
// ============================================================================

/**
 * @brief Deep comparison of two meshes for equality
 *
 * Compares all fields including the actual contents of row_keys, row_ptr, and intervals.
 * This ensures that different algorithm implementations produce bitwise-identical results.
 */
template <int DIM, typename MemorySpace>
bool meshes_equal(const Mesh<DIM, MemorySpace>& a, const Mesh<DIM, MemorySpace>& b) {
  // Quick size checks
  if (a.num_rows != b.num_rows) return false;
  if (a.num_intervals != b.num_intervals) return false;

  if (a.num_rows == 0) return true;  // Empty meshes are equal

  // Create host mirrors to compare contents
  auto a_row_keys = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a.row_keys);
  auto a_row_ptr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a.row_ptr);
  auto a_intervals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a.intervals);

  auto b_row_keys = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b.row_keys);
  auto b_row_ptr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b.row_ptr);
  auto b_intervals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b.intervals);

  // Compare row_keys (order matters!)
  for (std::size_t i = 0; i < a.num_rows; ++i) {
    if (a_row_keys(i) != b_row_keys(i)) {
      std::cerr << "Row keys differ at index " << i << std::endl;
      return false;
    }
  }

  // Compare row_ptr (CSR structure)
  for (std::size_t i = 0; i <= a.num_rows; ++i) {
    if (a_row_ptr(i) != b_row_ptr(i)) {
      std::cerr << "Row ptr differs at index " << i << std::endl;
      return false;
    }
  }

  // Compare intervals (order matters!)
  for (std::size_t i = 0; i < a.num_intervals; ++i) {
    const auto& ia = a_intervals(i);
    const auto& ib = b_intervals(i);
    if (ia.begin != ib.begin || ia.end != ib.end) {
      std::cerr << "Intervals differ at index " << i << ": "
                << "[" << ia.begin << "," << ia.end << ") vs "
                << "[" << ib.begin << "," << ib.end << ")" << std::endl;
      return false;
    }
  }

  return true;
}

// ============================================================================
// Test Helpers
// ============================================================================

Mesh2DDevice create_test_mesh_2d(int n, int offset) {
  Mesh2DDevice mesh;
  mesh.num_rows = n;
  mesh.num_intervals = n;
  mesh.row_keys = Mesh2DDevice::RowKeyView("test_keys", n);
  mesh.row_ptr = Mesh2DDevice::IndexView("test_ptr", n + 1);
  mesh.intervals = Mesh2DDevice::IntervalView("test_intervals", n);

  auto keys_h = Kokkos::create_mirror_view(mesh.row_keys);
  auto ptr_h = Kokkos::create_mirror_view(mesh.row_ptr);
  auto ints_h = Kokkos::create_mirror_view(mesh.intervals);

  for (int i = 0; i < n; ++i) {
    keys_h(i) = RowKey2D{i + offset};
    ptr_h(i) = i;
    ints_h(i) = Interval{0, 100};
  }
  ptr_h(n) = n;

  Kokkos::deep_copy(mesh.row_keys, keys_h);
  Kokkos::deep_copy(mesh.row_ptr, ptr_h);
  Kokkos::deep_copy(mesh.intervals, ints_h);

  return mesh;
}

// ============================================================================
// 2D Cross-Version Tests
// ============================================================================

TEST(CrossVersionValidation2D, V1_vs_V2_FullOverlap) {
  const int n = 128;
  auto mesh_a = create_test_mesh_2d(n, 0);
  auto mesh_b = create_test_mesh_2d(n, 0);

  v2::MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space> ws;

  auto result_v1 = v1::intersect_meshes_2d(mesh_a, mesh_b);
  auto result_v2 = v2::intersect_meshes_2d(mesh_a, mesh_b, ws);

  EXPECT_TRUE(meshes_equal(result_v1, result_v2))
      << "v1 and v2 produced different 2D intersection results";
}

TEST(CrossVersionValidation2D, V1_vs_V3_FullOverlap) {
  const int n = 128;
  auto mesh_a = create_test_mesh_2d(n, 0);
  auto mesh_b = create_test_mesh_2d(n, 0);

  auto result_v1 = v1::intersect_meshes_2d(mesh_a, mesh_b);
  auto result_v3 = v3::intersect_meshes_2d(mesh_a, mesh_b);

  EXPECT_TRUE(meshes_equal(result_v1, result_v3))
      << "v1 and v3 produced different 2D intersection results";
}

TEST(CrossVersionValidation2D, AllVersions_PartialOverlap) {
  const int n = 128;
  // Create partial overlap: A has [0,2,4,...], B has [128,130,132,...]
  auto mesh_a = create_test_mesh_2d(n, 0);
  auto mesh_b = create_test_mesh_2d(n, 128);

  v2::MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space> ws;

  auto result_v1 = v1::intersect_meshes_2d(mesh_a, mesh_b);
  auto result_v2 = v2::intersect_meshes_2d(mesh_a, mesh_b, ws);
  auto result_v3 = v3::intersect_meshes_2d(mesh_a, mesh_b);

  EXPECT_TRUE(meshes_equal(result_v1, result_v2))
      << "v1 and v2 produced different 2D results (partial overlap)";
  EXPECT_TRUE(meshes_equal(result_v1, result_v3))
      << "v1 and v3 produced different 2D results (partial overlap)";
}

TEST(CrossVersionValidation2D, AllVersions_NoOverlap) {
  const int n = 128;
  auto mesh_a = create_test_mesh_2d(n, 0);
  auto mesh_b = create_test_mesh_2d(n, 1000);  // No overlap

  v2::MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space> ws;

  auto result_v1 = v1::intersect_meshes_2d(mesh_a, mesh_b);
  auto result_v2 = v2::intersect_meshes_2d(mesh_a, mesh_b, ws);
  auto result_v3 = v3::intersect_meshes_2d(mesh_a, mesh_b);

  EXPECT_TRUE(meshes_equal(result_v1, result_v2))
      << "v1 and v2 produced different 2D results (no overlap)";
  EXPECT_TRUE(meshes_equal(result_v1, result_v3))
      << "v1 and v3 produced different 2D results (no overlap)";
}

TEST(CrossVersionValidation2D, AllVersions_MultipleIntervalsPerRow) {
  const int n = 64;
  const int intervals_per_row = 5;

  Mesh2DDevice mesh_a, mesh_b;
  mesh_a.num_rows = n;
  mesh_a.num_intervals = n * intervals_per_row;
  mesh_a.row_keys = Mesh2DDevice::RowKeyView("keys_a", n);
  mesh_a.row_ptr = Mesh2DDevice::IndexView("ptr_a", n + 1);
  mesh_a.intervals = Mesh2DDevice::IntervalView("ints_a", n * intervals_per_row);

  mesh_b.num_rows = n;
  mesh_b.num_intervals = n * intervals_per_row;
  mesh_b.row_keys = Mesh2DDevice::RowKeyView("keys_b", n);
  mesh_b.row_ptr = Mesh2DDevice::IndexView("ptr_b", n + 1);
  mesh_b.intervals = Mesh2DDevice::IntervalView("ints_b", n * intervals_per_row);

  auto keys_a = Kokkos::create_mirror_view(mesh_a.row_keys);
  auto ptr_a = Kokkos::create_mirror_view(mesh_a.row_ptr);
  auto ints_a = Kokkos::create_mirror_view(mesh_a.intervals);
  auto keys_b = Kokkos::create_mirror_view(mesh_b.row_keys);
  auto ptr_b = Kokkos::create_mirror_view(mesh_b.row_ptr);
  auto ints_b = Kokkos::create_mirror_view(mesh_b.intervals);

  // Create staggered intervals
  for (int i = 0; i < n; ++i) {
    keys_a(i) = RowKey2D{i};
    keys_b(i) = RowKey2D{i};
    ptr_a(i) = i * intervals_per_row;
    ptr_b(i) = i * intervals_per_row;

    for (int j = 0; j < intervals_per_row; ++j) {
      ints_a(i * intervals_per_row + j) = Interval{j * 20, j * 20 + 15};
      ints_b(i * intervals_per_row + j) = Interval{j * 20 + 5, j * 20 + 20};
    }
  }
  ptr_a(n) = n * intervals_per_row;
  ptr_b(n) = n * intervals_per_row;

  Kokkos::deep_copy(mesh_a.row_keys, keys_a);
  Kokkos::deep_copy(mesh_a.row_ptr, ptr_a);
  Kokkos::deep_copy(mesh_a.intervals, ints_a);
  Kokkos::deep_copy(mesh_b.row_keys, keys_b);
  Kokkos::deep_copy(mesh_b.row_ptr, ptr_b);
  Kokkos::deep_copy(mesh_b.intervals, ints_b);

  v2::MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space> ws;

  auto result_v1 = v1::intersect_meshes_2d(mesh_a, mesh_b);
  auto result_v2 = v2::intersect_meshes_2d(mesh_a, mesh_b, ws);
  auto result_v3 = v3::intersect_meshes_2d(mesh_a, mesh_b);

  EXPECT_TRUE(meshes_equal(result_v1, result_v2))
      << "v1 and v2 produced different 2D results (multiple intervals)";
  EXPECT_TRUE(meshes_equal(result_v1, result_v3))
      << "v1 and v3 produced different 2D results (multiple intervals)";
}

TEST(CrossVersionValidation2D, AllVersions_DifferentRowCounts) {
  // Mesh A has 100 rows, Mesh B has 50 rows (subset of A)
  const int n_a = 100;
  const int n_b = 50;

  Mesh2DDevice mesh_a, mesh_b;
  mesh_a.num_rows = n_a;
  mesh_a.num_intervals = n_a;
  mesh_a.row_keys = Mesh2DDevice::RowKeyView("keys_a", n_a);
  mesh_a.row_ptr = Mesh2DDevice::IndexView("ptr_a", n_a + 1);
  mesh_a.intervals = Mesh2DDevice::IntervalView("ints_a", n_a);

  mesh_b.num_rows = n_b;
  mesh_b.num_intervals = n_b;
  mesh_b.row_keys = Mesh2DDevice::RowKeyView("keys_b", n_b);
  mesh_b.row_ptr = Mesh2DDevice::IndexView("ptr_b", n_b + 1);
  mesh_b.intervals = Mesh2DDevice::IntervalView("ints_b", n_b);

  auto keys_a = Kokkos::create_mirror_view(mesh_a.row_keys);
  auto ptr_a = Kokkos::create_mirror_view(mesh_a.row_ptr);
  auto ints_a = Kokkos::create_mirror_view(mesh_a.intervals);
  auto keys_b = Kokkos::create_mirror_view(mesh_b.row_keys);
  auto ptr_b = Kokkos::create_mirror_view(mesh_b.row_ptr);
  auto ints_b = Kokkos::create_mirror_view(mesh_b.intervals);

  // A: rows 0-99, B: rows 0-49 (subset)
  for (int i = 0; i < n_a; ++i) {
    keys_a(i) = RowKey2D{i};
    ptr_a(i) = i;
    ints_a(i) = Interval{0, 100};
  }
  ptr_a(n_a) = n_a;

  for (int i = 0; i < n_b; ++i) {
    keys_b(i) = RowKey2D{i};
    ptr_b(i) = i;
    ints_b(i) = Interval{0, 100};
  }
  ptr_b(n_b) = n_b;

  Kokkos::deep_copy(mesh_a.row_keys, keys_a);
  Kokkos::deep_copy(mesh_a.row_ptr, ptr_a);
  Kokkos::deep_copy(mesh_a.intervals, ints_a);
  Kokkos::deep_copy(mesh_b.row_keys, keys_b);
  Kokkos::deep_copy(mesh_b.row_ptr, ptr_b);
  Kokkos::deep_copy(mesh_b.intervals, ints_b);

  v2::MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space> ws;

  auto result_v1 = v1::intersect_meshes_2d(mesh_a, mesh_b);
  auto result_v2 = v2::intersect_meshes_2d(mesh_a, mesh_b, ws);
  auto result_v3 = v3::intersect_meshes_2d(mesh_a, mesh_b);

  EXPECT_TRUE(meshes_equal(result_v1, result_v2))
      << "v1 and v2 produced different 2D results (different row counts)";
  EXPECT_TRUE(meshes_equal(result_v1, result_v3))
      << "v1 and v3 produced different 2D results (different row counts)";
}

// ============================================================================
// 3D Cross-Version Tests
// ============================================================================

Mesh3DDevice create_test_mesh_3d(int n, int z_offset) {
  Mesh3DDevice mesh;
  mesh.num_rows = n;
  mesh.num_intervals = n;
  mesh.row_keys = Mesh3DDevice::RowKeyView("test_keys", n);
  mesh.row_ptr = Mesh3DDevice::IndexView("test_ptr", n + 1);
  mesh.intervals = Mesh3DDevice::IntervalView("test_intervals", n);

  auto keys_h = Kokkos::create_mirror_view(mesh.row_keys);
  auto ptr_h = Kokkos::create_mirror_view(mesh.row_ptr);
  auto ints_h = Kokkos::create_mirror_view(mesh.intervals);

  for (int i = 0; i < n; ++i) {
    keys_h(i) = RowKey3D{i, z_offset};  // Vary y, keep z constant
    ptr_h(i) = i;
    ints_h(i) = Interval{0, 100};
  }
  ptr_h(n) = n;

  Kokkos::deep_copy(mesh.row_keys, keys_h);
  Kokkos::deep_copy(mesh.row_ptr, ptr_h);
  Kokkos::deep_copy(mesh.intervals, ints_h);

  return mesh;
}

TEST(CrossVersionValidation3D, V1_vs_V2_FullOverlap) {
  const int n = 64;
  auto mesh_a = create_test_mesh_3d(n, 0);
  auto mesh_b = create_test_mesh_3d(n, 0);

  v2::MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space> ws;

  auto result_v1 = v1::intersect_meshes_3d(mesh_a, mesh_b);
  auto result_v2 = v2::intersect_meshes_3d(mesh_a, mesh_b, ws);

  EXPECT_TRUE(meshes_equal(result_v1, result_v2))
      << "v1 and v2 produced different 3D intersection results";
}

TEST(CrossVersionValidation3D, V1_vs_V3_FullOverlap) {
  const int n = 64;
  auto mesh_a = create_test_mesh_3d(n, 0);
  auto mesh_b = create_test_mesh_3d(n, 0);

  auto result_v1 = v1::intersect_meshes_3d(mesh_a, mesh_b);
  auto result_v3 = v3::intersect_meshes_3d(mesh_a, mesh_b);

  EXPECT_TRUE(meshes_equal(result_v1, result_v3))
      << "v1 and v3 produced different 3D intersection results";
}

TEST(CrossVersionValidation3D, AllVersions_PartialOverlap) {
  const int n = 64;
  auto mesh_a = create_test_mesh_3d(n, 0);    // z=0
  auto mesh_b = create_test_mesh_3d(n, 10);   // z=10 (no overlap in z)

  v2::MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space> ws;

  auto result_v1 = v1::intersect_meshes_3d(mesh_a, mesh_b);
  auto result_v2 = v2::intersect_meshes_3d(mesh_a, mesh_b, ws);
  auto result_v3 = v3::intersect_meshes_3d(mesh_a, mesh_b);

  // All should produce empty results
  EXPECT_EQ(result_v1.num_rows, 0);
  EXPECT_EQ(result_v2.num_rows, 0);
  EXPECT_EQ(result_v3.num_rows, 0);

  EXPECT_TRUE(meshes_equal(result_v1, result_v2))
      << "v1 and v2 produced different 3D results";
  EXPECT_TRUE(meshes_equal(result_v1, result_v3))
      << "v1 and v3 produced different 3D results";
}

#endif
