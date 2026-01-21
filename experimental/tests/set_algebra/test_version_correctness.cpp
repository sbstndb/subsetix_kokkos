// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <gtest/gtest.h>
#include "test_correctness.hpp"
#include <vector>

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
// Helper: Create simple 2D mesh
// ============================================================================

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
// Typed Test Fixture for Oracle Tests
// ============================================================================

template <typename Version>
class CorrectnessTest : public ::testing::Test {
protected:
  void SetUp() override {
    if constexpr (!std::is_same_v<Version, V1Intersection>) {
      workspace_ = Version::create_workspace();
    }
  }

  template <typename MeshType>
  auto intersect(const MeshType& a, const MeshType& b) {
    if constexpr (std::is_same_v<Version, V2Intersection>) {
      return Version::intersect_2d(a, b, workspace_);
    } else {
      return Version::intersect_2d(a, b);
    }
  }

  typename Version::WorkspaceType workspace_;
};

TYPED_TEST_SUITE_P(CorrectnessTest);

// ============================================================================
// Oracle Tests - Known inputs with verified expected outputs
// ============================================================================

TYPED_TEST_P(CorrectnessTest, SimpleIntersection_KnownResult) {
  // A: [0, 10), [20, 30), [40, 50)
  // B: [5, 15), [25, 35)
  // Expected: [5, 10), [25, 30)
  auto mesh_a = create_simple_mesh_2d(0, {{0, 10}, {20, 30}, {40, 50}});
  auto mesh_b = create_simple_mesh_2d(0, {{5, 15}, {25, 35}});

  auto result = this->intersect(mesh_a, mesh_b);

  EXPECT_EQ(result.num_rows, 1);
  EXPECT_EQ(result.num_intervals, 2);

  auto result_intervals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.intervals);

  EXPECT_EQ(result_intervals(0).begin, 5);
  EXPECT_EQ(result_intervals(0).end, 10);
  EXPECT_EQ(result_intervals(1).begin, 25);
  EXPECT_EQ(result_intervals(1).end, 30);
}

TYPED_TEST_P(CorrectnessTest, NoOverlap_EmptyResult) {
  auto mesh_a = create_simple_mesh_2d(0, {{0, 10}, {20, 30}});
  auto mesh_b = create_simple_mesh_2d(0, {{40, 50}, {60, 70}});

  auto result = this->intersect(mesh_a, mesh_b);

  EXPECT_EQ(result.num_rows, 0);
  EXPECT_EQ(result.num_intervals, 0);
}

TYPED_TEST_P(CorrectnessTest, TouchingIntervals_NoOverlap) {
  // [0,10) and [10,20) should NOT overlap (half-open intervals)
  auto mesh_a = create_simple_mesh_2d(0, {{0, 10}, {20, 30}});
  auto mesh_b = create_simple_mesh_2d(0, {{10, 20}});

  auto result = this->intersect(mesh_a, mesh_b);

  EXPECT_EQ(result.num_rows, 0);
  EXPECT_EQ(result.num_intervals, 0);
}

TYPED_TEST_P(CorrectnessTest, Subset_SingleInterval) {
  // A: [0, 100), B: [25, 75) → Expected: [25, 75)
  auto mesh_a = create_simple_mesh_2d(0, {{0, 100}});
  auto mesh_b = create_simple_mesh_2d(0, {{25, 75}});

  auto result = this->intersect(mesh_a, mesh_b);

  EXPECT_EQ(result.num_rows, 1);
  EXPECT_EQ(result.num_intervals, 1);

  auto result_intervals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.intervals);

  EXPECT_EQ(result_intervals(0).begin, 25);
  EXPECT_EQ(result_intervals(0).end, 75);
}

// ============================================================================
// Mathematical Property Tests
// ============================================================================

TYPED_TEST_P(CorrectnessTest, Commutativity) {
  // A ∩ B = B ∩ A
  auto mesh_a = create_simple_mesh_2d(0, {{0, 10}, {20, 30}, {40, 50}});
  auto mesh_b = create_simple_mesh_2d(0, {{5, 15}, {25, 35}, {45, 55}});

  auto result_ab = this->intersect(mesh_a, mesh_b);
  auto result_ba = this->intersect(mesh_b, mesh_a);

  EXPECT_TRUE(meshes_equal(result_ab, result_ba))
      << "Intersection should be commutative: A∩B = B∩A";
}

TYPED_TEST_P(CorrectnessTest, Idempotence) {
  // A ∩ A = A
  auto mesh_a = create_simple_mesh_2d(0, {{0, 10}, {20, 30}, {40, 50}});

  auto result = this->intersect(mesh_a, mesh_a);

  EXPECT_TRUE(meshes_equal(result, mesh_a))
      << "Intersection should be idempotent: A∩A = A";
}

// ============================================================================
// Invariant Tests
// ============================================================================

TYPED_TEST_P(CorrectnessTest, ResultRowKeysPreserveInputOrder) {
  // When inputs are sorted, output should be sorted
  Mesh2DDevice sorted_a, sorted_b;

  for (auto& mesh : {&sorted_a, &sorted_b}) {
    mesh->num_rows = 3;
    mesh->num_intervals = 3;
    mesh->row_keys = Mesh2DDevice::RowKeyView("keys", 3);
    mesh->row_ptr = Mesh2DDevice::IndexView("ptr", 4);
    mesh->intervals = Mesh2DDevice::IntervalView("ints", 3);

    auto keys_h = Kokkos::create_mirror_view(mesh->row_keys);
    auto ptr_h = Kokkos::create_mirror_view(mesh->row_ptr);
    auto ints_h = Kokkos::create_mirror_view(mesh->intervals);

    for (int i = 0; i < 3; ++i) {
      keys_h(i) = RowKey2D{5 + i * 5};
      ptr_h(i) = i;
      ints_h(i) = Interval{0, 100};
    }
    ptr_h(3) = 3;

    Kokkos::deep_copy(mesh->row_keys, keys_h);
    Kokkos::deep_copy(mesh->row_ptr, ptr_h);
    Kokkos::deep_copy(mesh->intervals, ints_h);
  }

  auto result = this->intersect(sorted_a, sorted_b);

  auto result_keys = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.row_keys);

  for (size_t i = 1; i < result.num_rows; ++i) {
    EXPECT_GE(result_keys(i).y, result_keys(i-1).y)
        << "Row keys should be sorted when inputs are sorted";
  }
}

TYPED_TEST_P(CorrectnessTest, ResultIntervalsDoNotOverlap) {
  auto mesh_a = create_simple_mesh_2d(0, {{0, 50}, {60, 100}});
  auto mesh_b = create_simple_mesh_2d(0, {{25, 75}});

  auto result = this->intersect(mesh_a, mesh_b);

  if (result.num_intervals > 1) {
    auto result_intervals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.intervals);

    for (size_t i = 1; i < result.num_intervals; ++i) {
      EXPECT_GE(result_intervals(i).begin, result_intervals(i-1).end)
          << "Intervals should not overlap";
    }
  }
}

// ============================================================================
// Edge Cases
// ============================================================================

TYPED_TEST_P(CorrectnessTest, EmptyMesh) {
  Mesh2DDevice empty_a, empty_b;
  empty_a.num_rows = 0;
  empty_a.num_intervals = 0;
  empty_b.num_rows = 0;
  empty_b.num_intervals = 0;

  auto result = this->intersect(empty_a, empty_b);

  EXPECT_EQ(result.num_rows, 0);
  EXPECT_EQ(result.num_intervals, 0);
}

TYPED_TEST_P(CorrectnessTest, EmptyMeshWithNonEmpty) {
  auto non_empty = create_simple_mesh_2d(0, {{0, 10}});

  Mesh2DDevice empty;
  empty.num_rows = 0;
  empty.num_intervals = 0;

  auto result1 = this->intersect(non_empty, empty);
  auto result2 = this->intersect(empty, non_empty);

  EXPECT_EQ(result1.num_rows, 0);
  EXPECT_EQ(result1.num_intervals, 0);
  EXPECT_EQ(result2.num_rows, 0);
  EXPECT_EQ(result2.num_intervals, 0);
}

TYPED_TEST_P(CorrectnessTest, PointIntersection_NoOverlap) {
  // [0,1) and [1,2) should not overlap
  auto mesh_a = create_simple_mesh_2d(0, {{0, 1}});
  auto mesh_b = create_simple_mesh_2d(0, {{1, 2}});

  auto result = this->intersect(mesh_a, mesh_b);

  EXPECT_EQ(result.num_rows, 0);
  EXPECT_EQ(result.num_intervals, 0);
}

// Register all test instances for each version
REGISTER_TYPED_TEST_SUITE_P(CorrectnessTest,
  SimpleIntersection_KnownResult,
  NoOverlap_EmptyResult,
  TouchingIntervals_NoOverlap,
  Subset_SingleInterval,
  Commutativity,
  Idempotence,
  ResultRowKeysPreserveInputOrder,
  ResultIntervalsDoNotOverlap,
  EmptyMesh,
  EmptyMeshWithNonEmpty,
  PointIntersection_NoOverlap
);

INSTANTIATE_TYPED_TEST_SUITE_P(IntersectionCorrectness, CorrectnessTest, IntersectionVersions);

#endif
