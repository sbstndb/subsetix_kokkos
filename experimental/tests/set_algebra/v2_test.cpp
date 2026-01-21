// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <gtest/gtest.h>
#include <experimental/subsetix/csr/set_algebra/v1.hpp>
#include <experimental/subsetix/csr/set_algebra/v2.hpp>
#include <experimental/subsetix/csr/mesh.hpp>
#include <Kokkos_Core.hpp>

using namespace experimental::subsetix::csr;

// Note: Kokkos is already initialized by the test framework

// ============================================================================
// Helper: Deep mesh comparison (bitwise equality)
// ============================================================================

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
    if (a_row_keys(i) != b_row_keys(i)) return false;
  }

  // Compare row_ptr (CSR structure)
  for (std::size_t i = 0; i <= a.num_rows; ++i) {
    if (a_row_ptr(i) != b_row_ptr(i)) return false;
  }

  // Compare intervals (order matters!)
  for (std::size_t i = 0; i < a.num_intervals; ++i) {
    const auto& ia = a_intervals(i);
    const auto& ib = b_intervals(i);
    if (ia.begin != ib.begin || ia.end != ib.end) return false;
  }

  return true;
}

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

TEST(ExperimentalV2Test, CanCreateNonEmptyMesh2D) {
  // Test that we can create non-empty 2D meshes
  const std::size_t n_rows = 1;
  const std::size_t n_intervals = 1;

  Mesh2DDevice A;
  A.num_rows = n_rows;
  A.num_intervals = n_intervals;
  A.row_keys = Mesh2DDevice::RowKeyView("A_row_keys", n_rows);
  A.row_ptr = Mesh2DDevice::IndexView("A_row_ptr", n_rows + 1);
  A.intervals = Mesh2DDevice::IntervalView("A_intervals", n_intervals);

  EXPECT_EQ(A.num_rows, 1);
  EXPECT_EQ(A.num_intervals, 1);
}

TEST(ExperimentalV2Test, CanRunV1Intersection) {
  // Test that v1 intersection works with non-empty meshes
  Mesh2DDevice A;
  A.num_rows = 1;
  A.num_intervals = 1;
  A.row_keys = Mesh2DDevice::RowKeyView("A_row_keys", 1);
  A.row_ptr = Mesh2DDevice::IndexView("A_row_ptr", 2);
  A.intervals = Mesh2DDevice::IntervalView("A_intervals", 1);

  Mesh2DDevice B;
  B.num_rows = 1;
  B.num_intervals = 1;
  B.row_keys = Mesh2DDevice::RowKeyView("B_row_keys", 1);
  B.row_ptr = Mesh2DDevice::IndexView("B_row_ptr", 2);
  B.intervals = Mesh2DDevice::IntervalView("B_intervals", 1);

  auto A_keys_h = Kokkos::create_mirror_view(A.row_keys);
  auto A_ptr_h = Kokkos::create_mirror_view(A.row_ptr);
  auto A_int_h = Kokkos::create_mirror_view(A.intervals);
  auto B_keys_h = Kokkos::create_mirror_view(B.row_keys);
  auto B_ptr_h = Kokkos::create_mirror_view(B.row_ptr);
  auto B_int_h = Kokkos::create_mirror_view(B.intervals);

  A_keys_h(0) = RowKey2D{0};
  A_ptr_h(0) = 0;
  A_ptr_h(1) = 1;
  A_int_h(0) = Interval{0, 10};

  B_keys_h(0) = RowKey2D{0};
  B_ptr_h(0) = 0;
  B_ptr_h(1) = 1;
  B_int_h(0) = Interval{5, 15};

  Kokkos::deep_copy(A.row_keys, A_keys_h);
  Kokkos::deep_copy(A.row_ptr, A_ptr_h);
  Kokkos::deep_copy(A.intervals, A_int_h);
  Kokkos::deep_copy(B.row_keys, B_keys_h);
  Kokkos::deep_copy(B.row_ptr, B_ptr_h);
  Kokkos::deep_copy(B.intervals, B_int_h);

  auto result = v1::intersect_meshes_2d(A, B);

  EXPECT_EQ(result.num_rows, 1);
  EXPECT_EQ(result.num_intervals, 1);
}

TEST(ExperimentalV2Test, CanRunV2Intersection) {
  // Test that v2 intersection works with non-empty meshes
  Mesh2DDevice A;
  A.num_rows = 1;
  A.num_intervals = 1;
  A.row_keys = Mesh2DDevice::RowKeyView("A_row_keys", 1);
  A.row_ptr = Mesh2DDevice::IndexView("A_row_ptr", 2);
  A.intervals = Mesh2DDevice::IntervalView("A_intervals", 1);

  Mesh2DDevice B;
  B.num_rows = 1;
  B.num_intervals = 1;
  B.row_keys = Mesh2DDevice::RowKeyView("B_row_keys", 1);
  B.row_ptr = Mesh2DDevice::IndexView("B_row_ptr", 2);
  B.intervals = Mesh2DDevice::IntervalView("B_intervals", 1);

  auto A_keys_h = Kokkos::create_mirror_view(A.row_keys);
  auto A_ptr_h = Kokkos::create_mirror_view(A.row_ptr);
  auto A_int_h = Kokkos::create_mirror_view(A.intervals);
  auto B_keys_h = Kokkos::create_mirror_view(B.row_keys);
  auto B_ptr_h = Kokkos::create_mirror_view(B.row_ptr);
  auto B_int_h = Kokkos::create_mirror_view(B.intervals);

  A_keys_h(0) = RowKey2D{0};
  A_ptr_h(0) = 0;
  A_ptr_h(1) = 1;
  A_int_h(0) = Interval{0, 10};

  B_keys_h(0) = RowKey2D{0};
  B_ptr_h(0) = 0;
  B_ptr_h(1) = 1;
  B_int_h(0) = Interval{5, 15};

  Kokkos::deep_copy(A.row_keys, A_keys_h);
  Kokkos::deep_copy(A.row_ptr, A_ptr_h);
  Kokkos::deep_copy(A.intervals, A_int_h);
  Kokkos::deep_copy(B.row_keys, B_keys_h);
  Kokkos::deep_copy(B.row_ptr, B_ptr_h);
  Kokkos::deep_copy(B.intervals, B_int_h);

  // Test v2 intersection without workspace (auto-created)
  auto result = v2::intersect_meshes_2d(A, B);

  EXPECT_EQ(result.num_rows, 1);
  EXPECT_EQ(result.num_intervals, 1);
}

// ============================================================================
// Correctness Tests: v1 vs v2 comparison
// ============================================================================

TEST(ExperimentalV2Test, V1_vs_V2_SimpleIntersection_2D) {
  // Test: Two meshes with overlapping intervals on same row
  Mesh2DDevice A;
  A.num_rows = 1;
  A.num_intervals = 1;
  A.row_keys = Mesh2DDevice::RowKeyView("A_row_keys", 1);
  A.row_ptr = Mesh2DDevice::IndexView("A_row_ptr", 2);
  A.intervals = Mesh2DDevice::IntervalView("A_intervals", 1);

  Mesh2DDevice B;
  B.num_rows = 1;
  B.num_intervals = 1;
  B.row_keys = Mesh2DDevice::RowKeyView("B_row_keys", 1);
  B.row_ptr = Mesh2DDevice::IndexView("B_row_ptr", 2);
  B.intervals = Mesh2DDevice::IntervalView("B_intervals", 1);

  auto A_keys_h = Kokkos::create_mirror_view(A.row_keys);
  auto A_ptr_h = Kokkos::create_mirror_view(A.row_ptr);
  auto A_int_h = Kokkos::create_mirror_view(A.intervals);
  auto B_keys_h = Kokkos::create_mirror_view(B.row_keys);
  auto B_ptr_h = Kokkos::create_mirror_view(B.row_ptr);
  auto B_int_h = Kokkos::create_mirror_view(B.intervals);

  A_keys_h(0) = RowKey2D{0};
  A_ptr_h(0) = 0;
  A_ptr_h(1) = 1;
  A_int_h(0) = Interval{0, 10};

  B_keys_h(0) = RowKey2D{0};
  B_ptr_h(0) = 0;
  B_ptr_h(1) = 1;
  B_int_h(0) = Interval{5, 15};

  Kokkos::deep_copy(A.row_keys, A_keys_h);
  Kokkos::deep_copy(A.row_ptr, A_ptr_h);
  Kokkos::deep_copy(A.intervals, A_int_h);
  Kokkos::deep_copy(B.row_keys, B_keys_h);
  Kokkos::deep_copy(B.row_ptr, B_ptr_h);
  Kokkos::deep_copy(B.intervals, B_int_h);

  auto result_v1 = v1::intersect_meshes_2d(A, B);
  auto result_v2 = v2::intersect_meshes_2d(A, B);

  EXPECT_TRUE(meshes_equal(result_v1, result_v2))
      << "v1 and v2 produced different 2D intersection results (simple)";
}

TEST(ExperimentalV2Test, V1_vs_V2_NoIntersection_2D) {
  // Test: Two meshes with non-overlapping intervals
  Mesh2DDevice A;
  A.num_rows = 1;
  A.num_intervals = 1;
  A.row_keys = Mesh2DDevice::RowKeyView("A_row_keys", 1);
  A.row_ptr = Mesh2DDevice::IndexView("A_row_ptr", 2);
  A.intervals = Mesh2DDevice::IntervalView("A_intervals", 1);

  Mesh2DDevice B;
  B.num_rows = 1;
  B.num_intervals = 1;
  B.row_keys = Mesh2DDevice::RowKeyView("B_row_keys", 1);
  B.row_ptr = Mesh2DDevice::IndexView("B_row_ptr", 2);
  B.intervals = Mesh2DDevice::IntervalView("B_intervals", 1);

  auto A_keys_h = Kokkos::create_mirror_view(A.row_keys);
  auto A_ptr_h = Kokkos::create_mirror_view(A.row_ptr);
  auto A_int_h = Kokkos::create_mirror_view(A.intervals);
  auto B_keys_h = Kokkos::create_mirror_view(B.row_keys);
  auto B_ptr_h = Kokkos::create_mirror_view(B.row_ptr);
  auto B_int_h = Kokkos::create_mirror_view(B.intervals);

  A_keys_h(0) = RowKey2D{0};
  A_ptr_h(0) = 0;
  A_ptr_h(1) = 1;
  A_int_h(0) = Interval{0, 5};

  B_keys_h(0) = RowKey2D{0};
  B_ptr_h(0) = 0;
  B_ptr_h(1) = 1;
  B_int_h(0) = Interval{10, 20};

  Kokkos::deep_copy(A.row_keys, A_keys_h);
  Kokkos::deep_copy(A.row_ptr, A_ptr_h);
  Kokkos::deep_copy(A.intervals, A_int_h);
  Kokkos::deep_copy(B.row_keys, B_keys_h);
  Kokkos::deep_copy(B.row_ptr, B_ptr_h);
  Kokkos::deep_copy(B.intervals, B_int_h);

  auto result_v1 = v1::intersect_meshes_2d(A, B);
  auto result_v2 = v2::intersect_meshes_2d(A, B);

  EXPECT_TRUE(meshes_equal(result_v1, result_v2))
      << "v1 and v2 produced different 2D intersection results (no overlap)";
}

TEST(ExperimentalV2Test, V1_vs_V2_MultipleIntervals_2D) {
  // Test: Multiple intervals per row
  Mesh2DDevice A;
  A.num_rows = 1;
  A.num_intervals = 3;
  A.row_keys = Mesh2DDevice::RowKeyView("A_row_keys", 1);
  A.row_ptr = Mesh2DDevice::IndexView("A_row_ptr", 2);
  A.intervals = Mesh2DDevice::IntervalView("A_intervals", 3);

  Mesh2DDevice B;
  B.num_rows = 1;
  B.num_intervals = 2;
  B.row_keys = Mesh2DDevice::RowKeyView("B_row_keys", 1);
  B.row_ptr = Mesh2DDevice::IndexView("B_row_ptr", 2);
  B.intervals = Mesh2DDevice::IntervalView("B_intervals", 2);

  auto A_keys_h = Kokkos::create_mirror_view(A.row_keys);
  auto A_ptr_h = Kokkos::create_mirror_view(A.row_ptr);
  auto A_int_h = Kokkos::create_mirror_view(A.intervals);
  auto B_keys_h = Kokkos::create_mirror_view(B.row_keys);
  auto B_ptr_h = Kokkos::create_mirror_view(B.row_ptr);
  auto B_int_h = Kokkos::create_mirror_view(B.intervals);

  A_keys_h(0) = RowKey2D{0};
  A_ptr_h(0) = 0;
  A_ptr_h(1) = 3;
  A_int_h(0) = Interval{0, 10};
  A_int_h(1) = Interval{20, 30};
  A_int_h(2) = Interval{40, 50};

  B_keys_h(0) = RowKey2D{0};
  B_ptr_h(0) = 0;
  B_ptr_h(1) = 2;
  B_int_h(0) = Interval{5, 15};
  B_int_h(1) = Interval{25, 35};

  Kokkos::deep_copy(A.row_keys, A_keys_h);
  Kokkos::deep_copy(A.row_ptr, A_ptr_h);
  Kokkos::deep_copy(A.intervals, A_int_h);
  Kokkos::deep_copy(B.row_keys, B_keys_h);
  Kokkos::deep_copy(B.row_ptr, B_ptr_h);
  Kokkos::deep_copy(B.intervals, B_int_h);

  auto result_v1 = v1::intersect_meshes_2d(A, B);
  auto result_v2 = v2::intersect_meshes_2d(A, B);

  EXPECT_TRUE(meshes_equal(result_v1, result_v2))
      << "v1 and v2 produced different 2D intersection results (multiple intervals)";
}

TEST(ExperimentalV2Test, V1_vs_V2_DifferentRows_2D) {
  // Test: Meshes with different row keys
  Mesh2DDevice A;
  A.num_rows = 2;
  A.num_intervals = 2;
  A.row_keys = Mesh2DDevice::RowKeyView("A_row_keys", 2);
  A.row_ptr = Mesh2DDevice::IndexView("A_row_ptr", 3);
  A.intervals = Mesh2DDevice::IntervalView("A_intervals", 2);

  Mesh2DDevice B;
  B.num_rows = 2;
  B.num_intervals = 2;
  B.row_keys = Mesh2DDevice::RowKeyView("B_row_keys", 2);
  B.row_ptr = Mesh2DDevice::IndexView("B_row_ptr", 3);
  B.intervals = Mesh2DDevice::IntervalView("B_intervals", 2);

  auto A_keys_h = Kokkos::create_mirror_view(A.row_keys);
  auto A_ptr_h = Kokkos::create_mirror_view(A.row_ptr);
  auto A_int_h = Kokkos::create_mirror_view(A.intervals);
  auto B_keys_h = Kokkos::create_mirror_view(B.row_keys);
  auto B_ptr_h = Kokkos::create_mirror_view(B.row_ptr);
  auto B_int_h = Kokkos::create_mirror_view(B.intervals);

  A_keys_h(0) = RowKey2D{0};
  A_keys_h(1) = RowKey2D{1};
  A_ptr_h(0) = 0;
  A_ptr_h(1) = 1;
  A_ptr_h(2) = 2;
  A_int_h(0) = Interval{0, 10};
  A_int_h(1) = Interval{0, 10};

  B_keys_h(0) = RowKey2D{1};
  B_keys_h(1) = RowKey2D{2};
  B_ptr_h(0) = 0;
  B_ptr_h(1) = 1;
  B_ptr_h(2) = 2;
  B_int_h(0) = Interval{5, 15};
  B_int_h(1) = Interval{0, 10};

  Kokkos::deep_copy(A.row_keys, A_keys_h);
  Kokkos::deep_copy(A.row_ptr, A_ptr_h);
  Kokkos::deep_copy(A.intervals, A_int_h);
  Kokkos::deep_copy(B.row_keys, B_keys_h);
  Kokkos::deep_copy(B.row_ptr, B_ptr_h);
  Kokkos::deep_copy(B.intervals, B_int_h);

  auto result_v1 = v1::intersect_meshes_2d(A, B);
  auto result_v2 = v2::intersect_meshes_2d(A, B);

  EXPECT_TRUE(meshes_equal(result_v1, result_v2))
      << "v1 and v2 produced different 2D intersection results (different rows)";
}

// ============================================================================
// 3D Correctness Tests: v1 vs v2 comparison
// ============================================================================

TEST(ExperimentalV2Test, V1_vs_V2_SimpleIntersection_3D) {
  // Test: Two 3D meshes with overlapping intervals on same row
  Mesh3DDevice A;
  A.num_rows = 1;
  A.num_intervals = 1;
  A.row_keys = Mesh3DDevice::RowKeyView("A_row_keys", 1);
  A.row_ptr = Mesh3DDevice::IndexView("A_row_ptr", 2);
  A.intervals = Mesh3DDevice::IntervalView("A_intervals", 1);

  Mesh3DDevice B;
  B.num_rows = 1;
  B.num_intervals = 1;
  B.row_keys = Mesh3DDevice::RowKeyView("B_row_keys", 1);
  B.row_ptr = Mesh3DDevice::IndexView("B_row_ptr", 2);
  B.intervals = Mesh3DDevice::IntervalView("B_intervals", 1);

  auto A_keys_h = Kokkos::create_mirror_view(A.row_keys);
  auto A_ptr_h = Kokkos::create_mirror_view(A.row_ptr);
  auto A_int_h = Kokkos::create_mirror_view(A.intervals);
  auto B_keys_h = Kokkos::create_mirror_view(B.row_keys);
  auto B_ptr_h = Kokkos::create_mirror_view(B.row_ptr);
  auto B_int_h = Kokkos::create_mirror_view(B.intervals);

  A_keys_h(0) = RowKey3D{0, 0};
  A_ptr_h(0) = 0;
  A_ptr_h(1) = 1;
  A_int_h(0) = Interval{0, 10};

  B_keys_h(0) = RowKey3D{0, 0};
  B_ptr_h(0) = 0;
  B_ptr_h(1) = 1;
  B_int_h(0) = Interval{5, 15};

  Kokkos::deep_copy(A.row_keys, A_keys_h);
  Kokkos::deep_copy(A.row_ptr, A_ptr_h);
  Kokkos::deep_copy(A.intervals, A_int_h);
  Kokkos::deep_copy(B.row_keys, B_keys_h);
  Kokkos::deep_copy(B.row_ptr, B_ptr_h);
  Kokkos::deep_copy(B.intervals, B_int_h);

  auto result_v1 = v1::intersect_meshes_3d(A, B);
  auto result_v2 = v2::intersect_meshes_3d(A, B);

  EXPECT_TRUE(meshes_equal(result_v1, result_v2))
      << "v1 and v2 produced different 3D intersection results (simple)";
}

TEST(ExperimentalV2Test, V1_vs_V2_DifferentZRows_3D) {
  // Test: 3D meshes with different z coordinates
  Mesh3DDevice A;
  A.num_rows = 2;
  A.num_intervals = 2;
  A.row_keys = Mesh3DDevice::RowKeyView("A_row_keys", 2);
  A.row_ptr = Mesh3DDevice::IndexView("A_row_ptr", 3);
  A.intervals = Mesh3DDevice::IntervalView("A_intervals", 2);

  Mesh3DDevice B;
  B.num_rows = 2;
  B.num_intervals = 2;
  B.row_keys = Mesh3DDevice::RowKeyView("B_row_keys", 2);
  B.row_ptr = Mesh3DDevice::IndexView("B_row_ptr", 3);
  B.intervals = Mesh3DDevice::IntervalView("B_intervals", 2);

  auto A_keys_h = Kokkos::create_mirror_view(A.row_keys);
  auto A_ptr_h = Kokkos::create_mirror_view(A.row_ptr);
  auto A_int_h = Kokkos::create_mirror_view(A.intervals);
  auto B_keys_h = Kokkos::create_mirror_view(B.row_keys);
  auto B_ptr_h = Kokkos::create_mirror_view(B.row_ptr);
  auto B_int_h = Kokkos::create_mirror_view(B.intervals);

  A_keys_h(0) = RowKey3D{0, 0};
  A_keys_h(1) = RowKey3D{1, 0};
  A_ptr_h(0) = 0;
  A_ptr_h(1) = 1;
  A_ptr_h(2) = 2;
  A_int_h(0) = Interval{0, 10};
  A_int_h(1) = Interval{0, 10};

  B_keys_h(0) = RowKey3D{1, 0};
  B_keys_h(1) = RowKey3D{1, 1};
  B_ptr_h(0) = 0;
  B_ptr_h(1) = 1;
  B_ptr_h(2) = 2;
  B_int_h(0) = Interval{5, 15};
  B_int_h(1) = Interval{0, 10};

  Kokkos::deep_copy(A.row_keys, A_keys_h);
  Kokkos::deep_copy(A.row_ptr, A_ptr_h);
  Kokkos::deep_copy(A.intervals, A_int_h);
  Kokkos::deep_copy(B.row_keys, B_keys_h);
  Kokkos::deep_copy(B.row_ptr, B_ptr_h);
  Kokkos::deep_copy(B.intervals, B_int_h);

  auto result_v1 = v1::intersect_meshes_3d(A, B);
  auto result_v2 = v2::intersect_meshes_3d(A, B);

  EXPECT_TRUE(meshes_equal(result_v1, result_v2))
      << "v1 and v2 produced different 3D intersection results (different z)";
}

// ============================================================================
// Type Traits (merged from v1_test)
// ============================================================================

TEST(ExperimentalV2Test, Mesh2D_TypeTraits) {
  // Test that 2D mesh types are correctly defined
  using MeshType = Mesh<2, Kokkos::DefaultExecutionSpace::memory_space>;

  EXPECT_EQ(MeshType::DIM, 2);

  // Test RowKey type
  constexpr bool has_row_key = std::same_as<typename MeshType::RowKey, RowKey2D>;
  EXPECT_TRUE(has_row_key);
}

TEST(ExperimentalV2Test, Mesh3D_TypeTraits) {
  // Test that 3D mesh types are correctly defined
  using MeshType = Mesh<3, Kokkos::DefaultExecutionSpace::memory_space>;

  EXPECT_EQ(MeshType::DIM, 3);

  // Test RowKey type
  constexpr bool has_row_key = std::same_as<typename MeshType::RowKey, RowKey3D>;
  EXPECT_TRUE(has_row_key);
}

// ============================================================================
// Memory space conversion (merged from v1_test)
// ============================================================================

TEST(ExperimentalV2Test, Mesh2D_HostDeviceConversion) {
  // Test that mesh_to compiles for 2D
  Mesh2DDevice device_mesh;
  device_mesh.num_rows = 0;
  device_mesh.num_intervals = 0;

  auto host_mesh = v1::mesh_to<2, Kokkos::HostSpace>(device_mesh);

  EXPECT_EQ(host_mesh.num_rows, 0);
  EXPECT_EQ(host_mesh.num_intervals, 0);
}

TEST(ExperimentalV2Test, Mesh3D_HostDeviceConversion) {
  // Test that mesh_to compiles for 3D
  Mesh3DDevice device_mesh;
  device_mesh.num_rows = 0;
  device_mesh.num_intervals = 0;

  auto host_mesh = v1::mesh_to<3, Kokkos::HostSpace>(device_mesh);

  EXPECT_EQ(host_mesh.num_rows, 0);
  EXPECT_EQ(host_mesh.num_intervals, 0);
}

#endif // SUBSETIX_ENABLE_EXPERIMENTAL
