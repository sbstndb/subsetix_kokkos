// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <gtest/gtest.h>
#include <experimental/subsetix/csr/set_algebra/v1.hpp>
#include <Kokkos_Core.hpp>

using namespace experimental::subsetix::csr;

// ============================================================================
// Helper: Deep mesh comparison
// ============================================================================

template <int DIM, typename MemorySpace>
bool meshes_equal(const Mesh<DIM, MemorySpace>& a, const Mesh<DIM, MemorySpace>& b) {
  if (a.num_rows != b.num_rows) return false;
  if (a.num_intervals != b.num_intervals) return false;
  if (a.num_rows == 0) return true;

  auto a_row_keys = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a.row_keys);
  auto a_row_ptr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a.row_ptr);
  auto a_intervals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a.intervals);

  auto b_row_keys = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b.row_keys);
  auto b_row_ptr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b.row_ptr);
  auto b_intervals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b.intervals);

  for (std::size_t i = 0; i < a.num_rows; ++i) {
    if (a_row_keys(i) != b_row_keys(i)) return false;
  }
  for (std::size_t i = 0; i <= a.num_rows; ++i) {
    if (a_row_ptr(i) != b_row_ptr(i)) return false;
  }
  for (std::size_t i = 0; i < a.num_intervals; ++i) {
    const auto& ia = a_intervals(i);
    const auto& ib = b_intervals(i);
    if (ia.begin != ib.begin || ia.end != ib.end) return false;
  }
  return true;
}

// ============================================================================
// v1 Correctness Tests - Validate v1 produces MATHEMATICALLY CORRECT results
// ============================================================================

/**
 * @brief Helper: Create a simple 2D mesh with one row and multiple intervals
 */
Mesh2DDevice create_simple_mesh_2d(int row_y, const std::vector<Interval>& intervals) {
  Mesh2DDevice mesh;
  mesh.num_rows = 1;
  mesh.num_intervals = intervals.size();
  mesh.row_keys = Mesh2DDevice::RowKeyView("row_keys", 1);
  mesh.row_ptr = Mesh2DDevice::IndexView("row_ptr", 2);
  mesh.intervals = Mesh2DDevice::IntervalView("intervals", intervals.size());

  auto keys_h = Kokkos::create_mirror_view(mesh.row_keys);
  auto ptr_h = Kokkos::create_mirror_view(mesh.row_ptr);
  auto ints_h = Kokkos::create_mirror_view(mesh.intervals);

  keys_h(0) = RowKey2D{row_y};
  ptr_h(0) = 0;
  ptr_h(1) = intervals.size();

  for (size_t i = 0; i < intervals.size(); ++i) {
    ints_h(i) = intervals[i];
  }

  Kokkos::deep_copy(mesh.row_keys, keys_h);
  Kokkos::deep_copy(mesh.row_ptr, ptr_h);
  Kokkos::deep_copy(mesh.intervals, ints_h);

  return mesh;
}

// ============================================================================
// Oracle Tests - Known inputs with verified expected outputs
// ============================================================================

/**
 * Test: Simple intersection with known result
 * A: [0, 10), [20, 30), [40, 50)
 * B: [5, 15), [25, 35)
 * Expected: [5, 10), [25, 30)
 */
TEST(V1Correctness, SimpleIntersection_KnownResult) {
  auto mesh_a = create_simple_mesh_2d(0, {{0, 10}, {20, 30}, {40, 50}});
  auto mesh_b = create_simple_mesh_2d(0, {{5, 15}, {25, 35}});

  auto result = v1::intersect_meshes_2d(mesh_a, mesh_b);

  EXPECT_EQ(result.num_rows, 1);
  EXPECT_EQ(result.num_intervals, 2);

  // Verify intervals are correct
  auto result_intervals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.intervals);

  EXPECT_EQ(result_intervals(0).begin, 5);
  EXPECT_EQ(result_intervals(0).end, 10);
  EXPECT_EQ(result_intervals(1).begin, 25);
  EXPECT_EQ(result_intervals(1).end, 30);
}

/**
 * Test: No overlap produces empty result
 */
TEST(V1Correctness, NoOverlap_EmptyResult) {
  auto mesh_a = create_simple_mesh_2d(0, {{0, 10}, {20, 30}});
  auto mesh_b = create_simple_mesh_2d(0, {{40, 50}, {60, 70}});

  auto result = v1::intersect_meshes_2d(mesh_a, mesh_b);

  EXPECT_EQ(result.num_rows, 0);
  EXPECT_EQ(result.num_intervals, 0);
}

/**
 * Test: Touching intervals [0,10) and [10,20) do NOT overlap
 * (half-open intervals: end is exclusive)
 */
TEST(V1Correctness, TouchingIntervals_NoOverlap) {
  auto mesh_a = create_simple_mesh_2d(0, {{0, 10}, {20, 30}});
  auto mesh_b = create_simple_mesh_2d(0, {{10, 20}});

  auto result = v1::intersect_meshes_2d(mesh_a, mesh_b);

  EXPECT_EQ(result.num_rows, 0);
  EXPECT_EQ(result.num_intervals, 0);
}

/**
 * Test: Single interval subset
 * A: [0, 100)
 * B: [25, 75)
 * Expected: [25, 75)
 */
TEST(V1Correctness, Subset_SingleInterval) {
  auto mesh_a = create_simple_mesh_2d(0, {{0, 100}});
  auto mesh_b = create_simple_mesh_2d(0, {{25, 75}});

  auto result = v1::intersect_meshes_2d(mesh_a, mesh_b);

  EXPECT_EQ(result.num_rows, 1);
  EXPECT_EQ(result.num_intervals, 1);

  auto result_intervals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.intervals);

  EXPECT_EQ(result_intervals(0).begin, 25);
  EXPECT_EQ(result_intervals(0).end, 75);
}

// ============================================================================
// Mathematical Property Tests
// ============================================================================

/**
 * Test: Commutativity - A ∩ B = B ∩ A
 */
TEST(V1Correctness, Commutativity) {
  auto mesh_a = create_simple_mesh_2d(0, {{0, 10}, {20, 30}, {40, 50}});
  auto mesh_b = create_simple_mesh_2d(0, {{5, 15}, {25, 35}, {45, 55}});

  auto result_ab = v1::intersect_meshes_2d(mesh_a, mesh_b);
  auto result_ba = v1::intersect_meshes_2d(mesh_b, mesh_a);

  EXPECT_TRUE(meshes_equal(result_ab, result_ba))
      << "Intersection should be commutative: A∩B = B∩A";
}

/**
 * Test: Idempotence - A ∩ A = A
 */
TEST(V1Correctness, Idempotence) {
  auto mesh_a = create_simple_mesh_2d(0, {{0, 10}, {20, 30}, {40, 50}});

  auto result = v1::intersect_meshes_2d(mesh_a, mesh_a);

  EXPECT_TRUE(meshes_equal(result, mesh_a))
      << "Intersection should be idempotent: A∩A = A";
}

/**
 * Test: Associativity with subsets - (A ∩ B) ∩ C = A ∩ (B ∩ C)
 * Note: This is complex to test with general sets, so we test with specific subsets
 */
TEST(V1Correctness, Associativity_WithSubsets) {
  // Create three meshes where each is a subset of the previous
  auto mesh_a = create_simple_mesh_2d(0, {{0, 100}});
  auto mesh_b = create_simple_mesh_2d(0, {{0, 50}});       // subset of A
  auto mesh_c = create_simple_mesh_2d(0, {{0, 25}});       // subset of B

  auto result_ab_c = v1::intersect_meshes_2d(
      v1::intersect_meshes_2d(mesh_a, mesh_b),
      mesh_c
  );
  auto result_a_bc = v1::intersect_meshes_2d(
      mesh_a,
      v1::intersect_meshes_2d(mesh_b, mesh_c)
  );

  EXPECT_TRUE(meshes_equal(result_ab_c, result_a_bc))
      << "Intersection should be associative: (A∩B)∩C = A∩(B∩C)";
}

// ============================================================================
// Invariant Tests
// ============================================================================

/**
 * Test: Result row_keys preserve order from input
 * Note: v1 does NOT sort row_keys, it preserves the order from input mesh
 */
TEST(V1Correctness, ResultRowKeysPreserveInputOrder) {
  // Create mesh A with sorted row keys
  Mesh2DDevice sorted_a;
  sorted_a.num_rows = 3;
  sorted_a.num_intervals = 3;
  sorted_a.row_keys = Mesh2DDevice::RowKeyView("keys_a", 3);
  sorted_a.row_ptr = Mesh2DDevice::IndexView("ptr_a", 4);
  sorted_a.intervals = Mesh2DDevice::IntervalView("ints_a", 3);

  // Create mesh B with sorted row keys
  Mesh2DDevice sorted_b;
  sorted_b.num_rows = 3;
  sorted_b.num_intervals = 3;
  sorted_b.row_keys = Mesh2DDevice::RowKeyView("keys_b", 3);
  sorted_b.row_ptr = Mesh2DDevice::IndexView("ptr_b", 4);
  sorted_b.intervals = Mesh2DDevice::IntervalView("ints_b", 3);

  auto keys_a_h = Kokkos::create_mirror_view(sorted_a.row_keys);
  auto keys_b_h = Kokkos::create_mirror_view(sorted_b.row_keys);
  auto ptr_a_h = Kokkos::create_mirror_view(sorted_a.row_ptr);
  auto ptr_b_h = Kokkos::create_mirror_view(sorted_b.row_ptr);
  auto ints_a_h = Kokkos::create_mirror_view(sorted_a.intervals);
  auto ints_b_h = Kokkos::create_mirror_view(sorted_b.intervals);

  // Both have sorted row keys: [5, 10, 15]
  for (int i = 0; i < 3; ++i) {
    keys_a_h(i) = RowKey2D{5 + i * 5};
    keys_b_h(i) = RowKey2D{5 + i * 5};
    ptr_a_h(i) = i;
    ptr_b_h(i) = i;
    ints_a_h(i) = Interval{0, 100};
    ints_b_h(i) = Interval{0, 100};
  }
  ptr_a_h(3) = 3;
  ptr_b_h(3) = 3;

  Kokkos::deep_copy(sorted_a.row_keys, keys_a_h);
  Kokkos::deep_copy(sorted_a.row_ptr, ptr_a_h);
  Kokkos::deep_copy(sorted_a.intervals, ints_a_h);
  Kokkos::deep_copy(sorted_b.row_keys, keys_b_h);
  Kokkos::deep_copy(sorted_b.row_ptr, ptr_b_h);
  Kokkos::deep_copy(sorted_b.intervals, ints_b_h);

  auto result = v1::intersect_meshes_2d(sorted_a, sorted_b);

  // Verify row_keys are sorted when inputs are sorted
  auto result_keys = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.row_keys);

  for (size_t i = 1; i < result.num_rows; ++i) {
    EXPECT_GE(result_keys(i).y, result_keys(i-1).y)
        << "Row keys should be sorted when inputs are sorted";
  }
}

/**
 * Test: Result intervals are sorted within each row
 */
TEST(V1Correctness, ResultIntervalsAreSortedWithinRow) {
  auto mesh_a = create_simple_mesh_2d(0, {{0, 20}, {40, 60}, {80, 100}});
  auto mesh_b = create_simple_mesh_2d(0, {{10, 30}, {50, 70}, {90, 110}});

  auto result = v1::intersect_meshes_2d(mesh_a, mesh_b);

  ASSERT_EQ(result.num_rows, 1);
  ASSERT_EQ(result.num_intervals, 3);

  auto result_intervals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.intervals);

  // Verify intervals are sorted by begin
  for (size_t i = 1; i < result.num_intervals; ++i) {
    EXPECT_GE(result_intervals(i).begin, result_intervals(i-1).begin)
        << "Intervals should be sorted by begin coordinate";
  }
}

/**
 * Test: Result intervals do not overlap
 */
TEST(V1Correctness, ResultIntervalsDoNotOverlap) {
  auto mesh_a = create_simple_mesh_2d(0, {{0, 50}, {60, 100}});
  auto mesh_b = create_simple_mesh_2d(0, {{25, 75}});

  auto result = v1::intersect_meshes_2d(mesh_a, mesh_b);

  auto result_intervals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.intervals);

  // Verify no interval overlaps with another
  for (size_t i = 1; i < result.num_intervals; ++i) {
    EXPECT_GE(result_intervals(i).begin, result_intervals(i-1).end)
        << "Intervals should not overlap: interval[" << i << "].begin >= interval[" << (i-1) << "].end";
  }
}

// ============================================================================
// Edge Cases
// ============================================================================

/**
 * Test: Empty mesh (0 rows) produces empty result
 */
TEST(V1Correctness, EmptyMesh) {
  Mesh2DDevice empty_a, empty_b;
  empty_a.num_rows = 0;
  empty_a.num_intervals = 0;
  empty_b.num_rows = 0;
  empty_b.num_intervals = 0;

  auto result = v1::intersect_meshes_2d(empty_a, empty_b);

  EXPECT_EQ(result.num_rows, 0);
  EXPECT_EQ(result.num_intervals, 0);
}

/**
 * Test: Empty mesh with non-empty mesh produces empty result
 */
TEST(V1Correctness, EmptyMeshWithNonEmpty) {
  auto non_empty = create_simple_mesh_2d(0, {{0, 10}});

  Mesh2DDevice empty;
  empty.num_rows = 0;
  empty.num_intervals = 0;

  auto result1 = v1::intersect_meshes_2d(non_empty, empty);
  auto result2 = v1::intersect_meshes_2d(empty, non_empty);

  EXPECT_EQ(result1.num_rows, 0);
  EXPECT_EQ(result1.num_intervals, 0);
  EXPECT_EQ(result2.num_rows, 0);
  EXPECT_EQ(result2.num_intervals, 0);
}

/**
 * Test: Point intersection [0,1) and [1,2) should not overlap
 */
TEST(V1Correctness, PointIntersection_NoOverlap) {
  auto mesh_a = create_simple_mesh_2d(0, {{0, 1}});
  auto mesh_b = create_simple_mesh_2d(0, {{1, 2}});

  auto result = v1::intersect_meshes_2d(mesh_a, mesh_b);

  EXPECT_EQ(result.num_rows, 0);
  EXPECT_EQ(result.num_intervals, 0);
}

/**
 * Test: Zero-width interval should be handled gracefully
 */
TEST(V1Correctness, ZeroWidthInterval) {
  auto mesh_a = create_simple_mesh_2d(0, {{0, 10}, {20, 20}});  // Note: [20,20) is empty
  auto mesh_b = create_simple_mesh_2d(0, {{5, 15}});

  auto result = v1::intersect_meshes_2d(mesh_a, mesh_b);

  // Result should only contain [5, 10), the zero-width interval is ignored
  EXPECT_EQ(result.num_rows, 1);
  EXPECT_EQ(result.num_intervals, 1);

  auto result_intervals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.intervals);
  EXPECT_EQ(result_intervals(0).begin, 5);
  EXPECT_EQ(result_intervals(0).end, 10);
}

// ============================================================================
// 3D Correctness Tests
// ============================================================================

Mesh3DDevice create_simple_mesh_3d(int y, int z, const std::vector<Interval>& intervals) {
  Mesh3DDevice mesh;
  mesh.num_rows = 1;
  mesh.num_intervals = intervals.size();
  mesh.row_keys = Mesh3DDevice::RowKeyView("row_keys", 1);
  mesh.row_ptr = Mesh3DDevice::IndexView("row_ptr", 2);
  mesh.intervals = Mesh3DDevice::IntervalView("intervals", intervals.size());

  auto keys_h = Kokkos::create_mirror_view(mesh.row_keys);
  auto ptr_h = Kokkos::create_mirror_view(mesh.row_ptr);
  auto ints_h = Kokkos::create_mirror_view(mesh.intervals);

  keys_h(0) = RowKey3D{y, z};
  ptr_h(0) = 0;
  ptr_h(1) = intervals.size();

  for (size_t i = 0; i < intervals.size(); ++i) {
    ints_h(i) = intervals[i];
  }

  Kokkos::deep_copy(mesh.row_keys, keys_h);
  Kokkos::deep_copy(mesh.row_ptr, ptr_h);
  Kokkos::deep_copy(mesh.intervals, ints_h);

  return mesh;
}

TEST(V1Correctness3D, SimpleIntersection_KnownResult) {
  auto mesh_a = create_simple_mesh_3d(0, 0, {{0, 10}, {20, 30}});
  auto mesh_b = create_simple_mesh_3d(0, 0, {{5, 15}, {25, 35}});

  auto result = v1::intersect_meshes_3d(mesh_a, mesh_b);

  EXPECT_EQ(result.num_rows, 1);
  EXPECT_EQ(result.num_intervals, 2);

  auto result_intervals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.intervals);

  EXPECT_EQ(result_intervals(0).begin, 5);
  EXPECT_EQ(result_intervals(0).end, 10);
  EXPECT_EQ(result_intervals(1).begin, 25);
  EXPECT_EQ(result_intervals(1).end, 30);
}

TEST(V1Correctness3D, DifferentZ_NoOverlap) {
  auto mesh_a = create_simple_mesh_3d(0, 0, {{0, 10}});
  auto mesh_b = create_simple_mesh_3d(0, 1, {{0, 10}});  // Different Z coordinate

  auto result = v1::intersect_meshes_3d(mesh_a, mesh_b);

  EXPECT_EQ(result.num_rows, 0);
  EXPECT_EQ(result.num_intervals, 0);
}

#endif
