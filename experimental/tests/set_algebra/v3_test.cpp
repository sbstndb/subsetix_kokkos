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
// v3 Basic Tests
// ============================================================================

TEST(ExperimentalV3Test, EmptyMeshIntersection_2D) {
  Mesh2DDevice A, B;
  A.num_rows = 0;
  A.num_intervals = 0;
  B.num_rows = 0;
  B.num_intervals = 0;

  auto result = v3::intersect_meshes_2d(A, B);
  EXPECT_EQ(result.num_rows, 0);
  EXPECT_EQ(result.num_intervals, 0);
}

TEST(ExperimentalV3Test, EmptyMeshIntersection_3D) {
  Mesh3DDevice A, B;
  A.num_rows = 0;
  A.num_intervals = 0;
  B.num_rows = 0;
  B.num_intervals = 0;

  auto result = v3::intersect_meshes_3d(A, B);
  EXPECT_EQ(result.num_rows, 0);
  EXPECT_EQ(result.num_intervals, 0);
}

// ============================================================================
// v3 Correctness Tests: v3 vs v1 comparison
// ============================================================================

TEST(ExperimentalV3Test, V1_vs_V3_SimpleIntersection_2D) {
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
  auto result_v3 = v3::intersect_meshes_2d(A, B);

  EXPECT_TRUE(meshes_equal(result_v1, result_v3))
      << "v1 and v3 produced different 2D intersection results (simple)";
}

TEST(ExperimentalV3Test, V1_vs_V3_NoIntersection_2D) {
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
  auto result_v3 = v3::intersect_meshes_2d(A, B);

  EXPECT_TRUE(meshes_equal(result_v1, result_v3))
      << "v1 and v3 produced different 2D intersection results (no overlap)";
}

TEST(ExperimentalV3Test, V1_vs_V3_MultipleIntervals_2D) {
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
  auto result_v3 = v3::intersect_meshes_2d(A, B);

  EXPECT_TRUE(meshes_equal(result_v1, result_v3))
      << "v1 and v3 produced different 2D intersection results (multiple intervals)";
}

TEST(ExperimentalV3Test, V1_vs_V3_DifferentRows_2D) {
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
  auto result_v3 = v3::intersect_meshes_2d(A, B);

  EXPECT_TRUE(meshes_equal(result_v1, result_v3))
      << "v1 and v3 produced different 2D intersection results (different rows)";
}

TEST(ExperimentalV3Test, V1_vs_V3_SimpleIntersection_3D) {
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
  auto result_v3 = v3::intersect_meshes_3d(A, B);

  EXPECT_TRUE(meshes_equal(result_v1, result_v3))
      << "v1 and v3 produced different 3D intersection results (simple)";
}

TEST(ExperimentalV3Test, V1_vs_V3_DifferentZRows_3D) {
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
  auto result_v3 = v3::intersect_meshes_3d(A, B);

  EXPECT_TRUE(meshes_equal(result_v1, result_v3))
      << "v1 and v3 produced different 3D intersection results (different z)";
}

// ============================================================================
// Bounding Box Early Termination Test
// ============================================================================

TEST(ExperimentalV3Test, EarlyTermination_NonOverlappingBoundingBoxes) {
  Mesh2DDevice A;
  A.num_rows = 10;
  A.num_intervals = 10;
  A.row_keys = Mesh2DDevice::RowKeyView("A_row_keys", 10);
  A.row_ptr = Mesh2DDevice::IndexView("A_row_ptr", 11);
  A.intervals = Mesh2DDevice::IntervalView("A_intervals", 10);

  Mesh2DDevice B;
  B.num_rows = 10;
  B.num_intervals = 10;
  B.row_keys = Mesh2DDevice::RowKeyView("B_row_keys", 10);
  B.row_ptr = Mesh2DDevice::IndexView("B_row_ptr", 11);
  B.intervals = Mesh2DDevice::IntervalView("B_intervals", 10);

  auto A_keys_h = Kokkos::create_mirror_view(A.row_keys);
  auto A_ptr_h = Kokkos::create_mirror_view(A.row_ptr);
  auto A_int_h = Kokkos::create_mirror_view(A.intervals);
  auto B_keys_h = Kokkos::create_mirror_view(B.row_keys);
  auto B_ptr_h = Kokkos::create_mirror_view(B.row_ptr);
  auto B_int_h = Kokkos::create_mirror_view(B.intervals);

  // A has rows y=0 to 9, B has rows y=100 to 109 - no overlap!
  for (int i = 0; i < 10; ++i) {
    A_keys_h(i) = RowKey2D{i};
    A_ptr_h(i) = i;
    A_int_h(i) = Interval{0, 10};

    B_keys_h(i) = RowKey2D{100 + i};
    B_ptr_h(i) = i;
    B_int_h(i) = Interval{5, 15};
  }
  A_ptr_h(10) = 10;
  B_ptr_h(10) = 10;

  Kokkos::deep_copy(A.row_keys, A_keys_h);
  Kokkos::deep_copy(A.row_ptr, A_ptr_h);
  Kokkos::deep_copy(A.intervals, A_int_h);
  Kokkos::deep_copy(B.row_keys, B_keys_h);
  Kokkos::deep_copy(B.row_ptr, B_ptr_h);
  Kokkos::deep_copy(B.intervals, B_int_h);

  auto result = v3::intersect_meshes_2d(A, B);

  // Should return immediately due to bounding box check
  EXPECT_EQ(result.num_rows, 0);
  EXPECT_EQ(result.num_intervals, 0);
}

#endif // SUBSETIX_ENABLE_EXPERIMENTAL
