// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#ifdef SUBSETIX_ENABLE_PLAYGROUND

#include <gtest/gtest.h>
#include <playground/subsetix/csr/intersection/algorithm/optimized.hpp>
#include "test_common_format.hpp"
#include "test_random_mesh_generator.hpp"
#include <Kokkos_Core.hpp>

using namespace playground::subsetix::csr::intersection;
using namespace playground::subsetix::csr::intersection::test;

// ============================================================================
// optimized-Specific Tests with Common Format Conversion
// ============================================================================

/**
 * @brief Test suite for optimized algorithm using common format conversion
 */
class OptimizedConversionTest : public ::testing::Test {
  // The wrapper functions from test_random_mesh_generator.hpp are used directly
};

// ============================================================================
// Oracle Tests - Known inputs with verified expected outputs
// ============================================================================

TEST_F(OptimizedConversionTest, SimpleIntersection_KnownResult) {
  // A: [0, 10), [20, 30), [40, 50)
  // B: [5, 15), [25, 35)
  // Expected: [5, 10), [25, 30)
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 10}, {20, 30}, {40, 50}}});
  b.rows.push_back({0, {{5, 15}, {25, 35}}});

  auto result = intersect_2d(a, b);

  ASSERT_EQ(result.num_rows(), 1);
  ASSERT_EQ(result.num_intervals(), 2);

  EXPECT_EQ(result.rows[0].y, 0);
  ASSERT_EQ(result.rows[0].intervals.size(), 2);
  EXPECT_EQ(result.rows[0].intervals[0].begin, 5);
  EXPECT_EQ(result.rows[0].intervals[0].end, 10);
  EXPECT_EQ(result.rows[0].intervals[1].begin, 25);
  EXPECT_EQ(result.rows[0].intervals[1].end, 30);
}

TEST_F(OptimizedConversionTest, NoOverlap_EmptyResult) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 10}, {20, 30}}});
  b.rows.push_back({0, {{40, 50}, {60, 70}}});

  auto result = intersect_2d(a, b);

  EXPECT_EQ(result.num_rows(), 0);
  EXPECT_EQ(result.num_intervals(), 0);
}

TEST_F(OptimizedConversionTest, TouchingIntervals_NoOverlap) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 10}, {20, 30}}});
  b.rows.push_back({0, {{10, 20}}});

  auto result = intersect_2d(a, b);

  EXPECT_EQ(result.num_rows(), 0);
  EXPECT_EQ(result.num_intervals(), 0);
}

TEST_F(OptimizedConversionTest, Subset_SingleInterval) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 100}}});
  b.rows.push_back({0, {{25, 75}}});

  auto result = intersect_2d(a, b);

  ASSERT_EQ(result.num_rows(), 1);
  ASSERT_EQ(result.rows[0].intervals.size(), 1);
  EXPECT_EQ(result.rows[0].intervals[0].begin, 25);
  EXPECT_EQ(result.rows[0].intervals[0].end, 75);
}

TEST_F(OptimizedConversionTest, MultipleRows_PartialOverlap) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 100}}});
  a.rows.push_back({10, {{0, 100}}});
  a.rows.push_back({20, {{0, 100}}});

  b.rows.push_back({5, {{0, 100}}});
  b.rows.push_back({10, {{50, 150}}});
  b.rows.push_back({25, {{0, 100}}});

  auto result = intersect_2d(a, b);

  ASSERT_EQ(result.num_rows(), 1);
  EXPECT_EQ(result.rows[0].y, 10);
  ASSERT_EQ(result.rows[0].intervals.size(), 1);
  EXPECT_EQ(result.rows[0].intervals[0].begin, 50);
  EXPECT_EQ(result.rows[0].intervals[0].end, 100);
}

// ============================================================================
// Mathematical Property Tests
// ============================================================================

TEST_F(OptimizedConversionTest, Commutativity) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 10}, {20, 30}, {40, 50}}});
  b.rows.push_back({0, {{5, 15}, {25, 35}, {45, 55}}});

  auto result_ab = intersect_2d(a, b);
  auto result_ba = intersect_2d(b, a);

  EXPECT_TRUE(common_meshes_equal(result_ab, result_ba))
      << "Intersection should be commutative: A∩B = B∩A";
}

TEST_F(OptimizedConversionTest, Idempotence) {
  DefaultCommonMesh2D a;
  a.rows.push_back({0, {{0, 10}, {20, 30}, {40, 50}}});

  auto result = intersect_2d(a, a);

  EXPECT_TRUE(common_meshes_equal(result, a))
      << "Intersection should be idempotent: A∩A = A";
}

TEST_F(OptimizedConversionTest, Associativity_WithSubsets) {
  DefaultCommonMesh2D a, b, c;
  a.rows.push_back({0, {{0, 100}}});
  b.rows.push_back({0, {{0, 50}}});
  c.rows.push_back({0, {{0, 25}}});

  auto ab = intersect_2d(a, b);
  auto abc_left = intersect_2d(ab, c);

  auto bc = intersect_2d(b, c);
  auto abc_right = intersect_2d(a, bc);

  EXPECT_TRUE(common_meshes_equal(abc_left, abc_right))
      << "Intersection should be associative: (A∩B)∩C = A∩(B∩C)";
}

TEST_F(OptimizedConversionTest, AbsorbingElement) {
  DefaultCommonMesh2D a, empty;
  a.rows.push_back({0, {{0, 10}, {20, 30}}});

  auto result = intersect_2d(a, empty);

  EXPECT_EQ(result.num_rows(), 0);
  EXPECT_EQ(result.num_intervals(), 0);
}

// ============================================================================
// Invariant Tests
// ============================================================================

TEST_F(OptimizedConversionTest, ResultIntervalsDoNotOverlap) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 50}, {60, 100}}});
  b.rows.push_back({0, {{25, 75}}});

  auto result = intersect_2d(a, b);

  for (const auto& row : result.rows) {
    for (size_t i = 1; i < row.intervals.size(); ++i) {
      EXPECT_GE(row.intervals[i].begin, row.intervals[i-1].end)
          << "Intervals should not overlap";
    }
  }
}

TEST_F(OptimizedConversionTest, ResultIntervalsAreNonEmpty) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 100}}});
  b.rows.push_back({0, {{25, 75}}});

  auto result = intersect_2d(a, b);

  for (const auto& row : result.rows) {
    for (const auto& interval : row.intervals) {
      EXPECT_LT(interval.begin, interval.end)
          << "All intervals should be non-empty";
    }
  }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(OptimizedConversionTest, EmptyMesh_EmptyResult) {
  DefaultCommonMesh2D empty_a, empty_b;

  auto result = intersect_2d(empty_a, empty_b);

  EXPECT_EQ(result.num_rows(), 0);
  EXPECT_EQ(result.num_intervals(), 0);
}

TEST_F(OptimizedConversionTest, EmptyMesh_NonEmptyGivesEmpty) {
  DefaultCommonMesh2D a, empty;
  a.rows.push_back({0, {{0, 10}}});

  auto result1 = intersect_2d(a, empty);
  auto result2 = intersect_2d(empty, a);

  EXPECT_EQ(result1.num_rows(), 0);
  EXPECT_EQ(result2.num_rows(), 0);
}

TEST_F(OptimizedConversionTest, PointIntersection_NoOverlap) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 1}}});
  b.rows.push_back({0, {{1, 2}}});

  auto result = intersect_2d(a, b);

  EXPECT_EQ(result.num_rows(), 0);
  EXPECT_EQ(result.num_intervals(), 0);
}

TEST_F(OptimizedConversionTest, LargeIntervals) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{1000000, 2000000}}});
  b.rows.push_back({0, {{1500000, 2500000}}});

  auto result = intersect_2d(a, b);

  ASSERT_EQ(result.num_rows(), 1);
  ASSERT_EQ(result.rows[0].intervals.size(), 1);
  EXPECT_EQ(result.rows[0].intervals[0].begin, 1500000);
  EXPECT_EQ(result.rows[0].intervals[0].end, 2000000);
}

TEST_F(OptimizedConversionTest, NegativeCoordinates) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({-10, {{-100, -50}, {-20, 0}}});
  b.rows.push_back({-10, {{-75, -25}}});

  auto result = intersect_2d(a, b);

  ASSERT_EQ(result.num_rows(), 1);
  ASSERT_EQ(result.rows[0].intervals.size(), 1);
  EXPECT_EQ(result.rows[0].intervals[0].begin, -75);
  EXPECT_EQ(result.rows[0].intervals[0].end, -50);
}

// ============================================================================
// 3D Tests with Conversion
// ============================================================================

TEST_F(OptimizedConversionTest, Simple3DIntersection_KnownResult) {
  DefaultCommonMesh3D a, b;
  a.rows.push_back({0, 0, {{0, 10}, {20, 30}}});
  b.rows.push_back({0, 0, {{5, 15}, {25, 35}}});

  auto result = intersect_3d(a, b);

  ASSERT_EQ(result.num_rows(), 1);
  EXPECT_EQ(result.rows[0].y, 0);
  EXPECT_EQ(result.rows[0].z, 0);
  ASSERT_EQ(result.rows[0].intervals.size(), 2);
  EXPECT_EQ(result.rows[0].intervals[0].begin, 5);
  EXPECT_EQ(result.rows[0].intervals[0].end, 10);
}

TEST_F(OptimizedConversionTest, Different3DZ_NoOverlap) {
  DefaultCommonMesh3D a, b;
  a.rows.push_back({0, 0, {{0, 10}}});
  a.rows.push_back({0, 5, {{0, 10}}});
  a.rows.push_back({0, 10, {{0, 10}}});
  b.rows.push_back({0, 1, {{0, 10}}});  // Different Z, same Y
  b.rows.push_back({0, 6, {{0, 10}}});
  b.rows.push_back({0, 11, {{0, 10}}});

  auto result = intersect_3d(a, b);

  EXPECT_EQ(result.num_rows(), 0);
  EXPECT_EQ(result.num_intervals(), 0);
}

TEST_F(OptimizedConversionTest, Multiple3DRowsWithDifferentZ) {
  DefaultCommonMesh3D a, b;
  // Y scope: [0, 10], Z scope: [0, 10]
  a.rows.push_back({0, 0, {{0, 100}}});
  a.rows.push_back({10, 5, {{0, 100}}});
  a.rows.push_back({5, 10, {{0, 100}}});

  b.rows.push_back({0, 0, {{50, 150}}});
  b.rows.push_back({10, 3, {{0, 100}}});  // Different Z, same Y - no overlap
  b.rows.push_back({2, 10, {{0, 100}}});   // Different Y, same Z - no overlap

  auto result = intersect_3d(a, b);

  // Only first row should overlap
  ASSERT_EQ(result.num_rows(), 1);
  EXPECT_EQ(result.rows[0].y, 0);
  EXPECT_EQ(result.rows[0].z, 0);
  EXPECT_EQ(result.rows[0].intervals[0].begin, 50);
  EXPECT_EQ(result.rows[0].intervals[0].end, 100);
}

// ============================================================================
// Round-trip Conversion Tests
// ============================================================================

TEST_F(OptimizedConversionTest, RoundTripConversion_PreservesData) {
  DefaultCommonMesh2D original;
  original.rows.push_back({0, {{0, 10}, {20, 30}}});
  original.rows.push_back({10, {{5, 15}}});
  original.rows.push_back({20, {{100, 200}}});

  auto device = from_common_2d(original);
  auto converted = to_common_2d(device);

  EXPECT_TRUE(common_meshes_equal(original, converted));
}

TEST_F(OptimizedConversionTest, RoundTrip3DConversion_PreservesData) {
  DefaultCommonMesh3D original;
  original.rows.push_back({0, 0, {{0, 10}}});
  original.rows.push_back({5, 3, {{20, 30}}});
  original.rows.push_back({10, 0, {{100, 200}}});

  auto device = from_common_3d(original);
  auto converted = to_common_3d(device);

  EXPECT_TRUE(common_meshes_equal(original, converted));
}

#endif
