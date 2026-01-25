// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#ifdef SUBSETIX_ENABLE_PLAYGROUND

#include <gtest/gtest.h>
#include <playground/subsetix/csr/intersection/algorithm/v4_hash.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v5_parallel_merge.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v6_direct_index.hpp>
#include "test_common_format.hpp"
#include "test_random_mesh_generator.hpp"
#include <Kokkos_Core.hpp>

using namespace playground::subsetix::csr::intersection;
using namespace playground::subsetix::csr::intersection::test;

// ============================================================================
// Helper functions to run intersection with each version
// ============================================================================

namespace version_helpers {

// v4 (hash-based)
inline DefaultCommonMesh2D intersect_v4_2d(const DefaultCommonMesh2D& a, const DefaultCommonMesh2D& b) {
  return test::intersect_v4_hash_2d(a, b);
}
inline DefaultCommonMesh3D intersect_v4_3d(const DefaultCommonMesh3D& a, const DefaultCommonMesh3D& b) {
  return test::intersect_v4_hash_3d(a, b);
}

// v5 (parallel merge)
inline DefaultCommonMesh2D intersect_v5_2d(const DefaultCommonMesh2D& a, const DefaultCommonMesh2D& b) {
  return test::intersect_v5_parallel_merge_2d(a, b);
}
inline DefaultCommonMesh3D intersect_v5_3d(const DefaultCommonMesh3D& a, const DefaultCommonMesh3D& b) {
  return test::intersect_v5_parallel_merge_3d(a, b);
}

// v6 (direct index)
inline DefaultCommonMesh2D intersect_v6_2d(const DefaultCommonMesh2D& a, const DefaultCommonMesh2D& b) {
  return test::intersect_v6_direct_index_2d(a, b);
}
inline DefaultCommonMesh3D intersect_v6_3d(const DefaultCommonMesh3D& a, const DefaultCommonMesh3D& b) {
  return test::intersect_v6_direct_index_3d(a, b);
}

} // namespace version_helpers

// ============================================================================
// Test suite for v4 (hash-based)
// ============================================================================

class V4HashConversionTest : public ::testing::Test {
protected:
  using Intersect2D = DefaultCommonMesh2D (*)(const DefaultCommonMesh2D&, const DefaultCommonMesh2D&);
  using Intersect3D = DefaultCommonMesh3D (*)(const DefaultCommonMesh3D&, const DefaultCommonMesh3D&);

  void run_oracle_test_2d(
      const std::function<void(DefaultCommonMesh2D&, DefaultCommonMesh2D&)>& setup,
      const std::function<void(const DefaultCommonMesh2D&)>& verify,
      Intersect2D intersect_fn) {
    DefaultCommonMesh2D a, b;
    setup(a, b);
    auto result = intersect_fn(a, b);
    verify(result);
  }

  void run_oracle_test_3d(
      const std::function<void(DefaultCommonMesh3D&, DefaultCommonMesh3D&)>& setup,
      const std::function<void(const DefaultCommonMesh3D&)>& verify,
      Intersect3D intersect_fn) {
    DefaultCommonMesh3D a, b;
    setup(a, b);
    auto result = intersect_fn(a, b);
    verify(result);
  }
};

// ============================================================================
// Oracle Tests - Same as baseline, just testing v4 produces same results
// ============================================================================

TEST_F(V4HashConversionTest, SimpleIntersection_KnownResult) {
  // A: [0, 10), [20, 30), [40, 50)
  // B: [5, 15), [25, 35)
  // Expected: [5, 10), [25, 30)
  run_oracle_test_2d(
      [](DefaultCommonMesh2D& a, DefaultCommonMesh2D& b) {
        a.rows.push_back({0, {{0, 10}, {20, 30}, {40, 50}}});
        b.rows.push_back({0, {{5, 15}, {25, 35}}});
      },
      [](const DefaultCommonMesh2D& result) {
        ASSERT_EQ(result.num_rows(), 1);
        ASSERT_EQ(result.num_intervals(), 2);
        EXPECT_EQ(result.rows[0].y, 0);
        ASSERT_EQ(result.rows[0].intervals.size(), 2);
        EXPECT_EQ(result.rows[0].intervals[0].begin, 5);
        EXPECT_EQ(result.rows[0].intervals[0].end, 10);
        EXPECT_EQ(result.rows[0].intervals[1].begin, 25);
        EXPECT_EQ(result.rows[0].intervals[1].end, 30);
      },
      version_helpers::intersect_v4_2d);
}

TEST_F(V4HashConversionTest, NoOverlap_EmptyResult) {
  run_oracle_test_2d(
      [](DefaultCommonMesh2D& a, DefaultCommonMesh2D& b) {
        a.rows.push_back({0, {{0, 10}, {20, 30}}});
        b.rows.push_back({0, {{40, 50}, {60, 70}}});
      },
      [](const DefaultCommonMesh2D& result) {
        EXPECT_EQ(result.num_rows(), 0);
        EXPECT_EQ(result.num_intervals(), 0);
      },
      version_helpers::intersect_v4_2d);
}

TEST_F(V4HashConversionTest, TouchingIntervals_NoOverlap) {
  run_oracle_test_2d(
      [](DefaultCommonMesh2D& a, DefaultCommonMesh2D& b) {
        a.rows.push_back({0, {{0, 10}, {20, 30}}});
        b.rows.push_back({0, {{10, 20}}});
      },
      [](const DefaultCommonMesh2D& result) {
        EXPECT_EQ(result.num_rows(), 0);
        EXPECT_EQ(result.num_intervals(), 0);
      },
      version_helpers::intersect_v4_2d);
}

TEST_F(V4HashConversionTest, Subset_SingleInterval) {
  run_oracle_test_2d(
      [](DefaultCommonMesh2D& a, DefaultCommonMesh2D& b) {
        a.rows.push_back({0, {{0, 100}}});
        b.rows.push_back({0, {{25, 75}}});
      },
      [](const DefaultCommonMesh2D& result) {
        ASSERT_EQ(result.num_rows(), 1);
        ASSERT_EQ(result.rows[0].intervals.size(), 1);
        EXPECT_EQ(result.rows[0].intervals[0].begin, 25);
        EXPECT_EQ(result.rows[0].intervals[0].end, 75);
      },
      version_helpers::intersect_v4_2d);
}

TEST_F(V4HashConversionTest, MultipleRows_PartialOverlap) {
  run_oracle_test_2d(
      [](DefaultCommonMesh2D& a, DefaultCommonMesh2D& b) {
        a.rows.push_back({0, {{0, 100}}});
        a.rows.push_back({10, {{0, 100}}});
        a.rows.push_back({20, {{0, 100}}});
        b.rows.push_back({5, {{0, 100}}});
        b.rows.push_back({10, {{50, 150}}});
        b.rows.push_back({25, {{0, 100}}});
      },
      [](const DefaultCommonMesh2D& result) {
        ASSERT_EQ(result.num_rows(), 1);
        EXPECT_EQ(result.rows[0].y, 10);
        ASSERT_EQ(result.rows[0].intervals.size(), 1);
        EXPECT_EQ(result.rows[0].intervals[0].begin, 50);
        EXPECT_EQ(result.rows[0].intervals[0].end, 100);
      },
      version_helpers::intersect_v4_2d);
}

// ============================================================================
// Mathematical Property Tests - v4
// ============================================================================

TEST_F(V4HashConversionTest, Commutativity) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 10}, {20, 30}, {40, 50}}});
  b.rows.push_back({0, {{5, 15}, {25, 35}, {45, 55}}});

  auto result_ab = version_helpers::intersect_v4_2d(a, b);
  auto result_ba = version_helpers::intersect_v4_2d(b, a);

  EXPECT_TRUE(common_meshes_equal(result_ab, result_ba))
      << "v4 (hash): Intersection should be commutative: A∩B = B∩A";
}

TEST_F(V4HashConversionTest, Idempotence) {
  DefaultCommonMesh2D a;
  a.rows.push_back({0, {{0, 10}, {20, 30}, {40, 50}}});

  auto result = version_helpers::intersect_v4_2d(a, a);

  EXPECT_TRUE(common_meshes_equal(result, a))
      << "v4 (hash): Intersection should be idempotent: A∩A = A";
}

TEST_F(V4HashConversionTest, Associativity_WithSubsets) {
  DefaultCommonMesh2D a, b, c;
  a.rows.push_back({0, {{0, 100}}});
  b.rows.push_back({0, {{0, 50}}});   // subset of A
  c.rows.push_back({0, {{0, 25}}});   // subset of B

  auto ab = version_helpers::intersect_v4_2d(a, b);
  auto abc_left = version_helpers::intersect_v4_2d(ab, c);

  auto bc = version_helpers::intersect_v4_2d(b, c);
  auto abc_right = version_helpers::intersect_v4_2d(a, bc);

  EXPECT_TRUE(common_meshes_equal(abc_left, abc_right))
      << "v4 (hash): Intersection should be associative: (A∩B)∩C = A∩(B∩C)";
}

TEST_F(V4HashConversionTest, AbsorbingElement) {
  DefaultCommonMesh2D a, empty;
  a.rows.push_back({0, {{0, 10}, {20, 30}}});

  auto result = version_helpers::intersect_v4_2d(a, empty);

  EXPECT_EQ(result.num_rows(), 0);
  EXPECT_EQ(result.num_intervals(), 0);
}

// ============================================================================
// Invariant Tests - v4
// ============================================================================

TEST_F(V4HashConversionTest, ResultIntervalsDoNotOverlap) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 50}, {60, 100}}});
  b.rows.push_back({0, {{25, 75}}});

  auto result = version_helpers::intersect_v4_2d(a, b);

  for (const auto& row : result.rows) {
    for (size_t i = 1; i < row.intervals.size(); ++i) {
      EXPECT_GE(row.intervals[i].begin, row.intervals[i-1].end)
          << "v4 (hash): Intervals should not overlap";
    }
  }
}

TEST_F(V4HashConversionTest, ResultIntervalsAreNonEmpty) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 100}}});
  b.rows.push_back({0, {{25, 75}}});

  auto result = version_helpers::intersect_v4_2d(a, b);

  for (const auto& row : result.rows) {
    for (const auto& interval : row.intervals) {
      EXPECT_LT(interval.begin, interval.end)
          << "v4 (hash): All intervals should be non-empty";
    }
  }
}

TEST_F(V4HashConversionTest, ResultIsSubsetOfBoth) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 10}, {20, 30}, {40, 50}}});
  b.rows.push_back({0, {{5, 15}, {25, 35}, {45, 55}}});

  auto result = version_helpers::intersect_v4_2d(a, b);

  EXPECT_LE(result.num_intervals(), a.num_intervals());
  EXPECT_LE(result.num_intervals(), b.num_intervals());
}

// ============================================================================
// Edge Cases - v4
// ============================================================================

TEST_F(V4HashConversionTest, EmptyMesh_EmptyResult) {
  DefaultCommonMesh2D empty_a, empty_b;
  auto result = version_helpers::intersect_v4_2d(empty_a, empty_b);
  EXPECT_EQ(result.num_rows(), 0);
  EXPECT_EQ(result.num_intervals(), 0);
}

TEST_F(V4HashConversionTest, EmptyMesh_NonEmptyGivesEmpty) {
  DefaultCommonMesh2D a, empty;
  a.rows.push_back({0, {{0, 10}}});

  auto result1 = version_helpers::intersect_v4_2d(a, empty);
  auto result2 = version_helpers::intersect_v4_2d(empty, a);

  EXPECT_EQ(result1.num_rows(), 0);
  EXPECT_EQ(result2.num_rows(), 0);
}

TEST_F(V4HashConversionTest, PointIntersection_NoOverlap) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 1}}});
  b.rows.push_back({0, {{1, 2}}});

  auto result = version_helpers::intersect_v4_2d(a, b);

  EXPECT_EQ(result.num_rows(), 0);
  EXPECT_EQ(result.num_intervals(), 0);
}

TEST_F(V4HashConversionTest, SinglePointOverlap_TreatedAsOverlap) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 10}}});
  b.rows.push_back({0, {{9, 20}}});

  auto result = version_helpers::intersect_v4_2d(a, b);

  ASSERT_EQ(result.num_rows(), 1);
  ASSERT_EQ(result.rows[0].intervals.size(), 1);
  EXPECT_EQ(result.rows[0].intervals[0].begin, 9);
  EXPECT_EQ(result.rows[0].intervals[0].end, 10);
}

TEST_F(V4HashConversionTest, LargeIntervals) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{1000000, 2000000}}});
  b.rows.push_back({0, {{1500000, 2500000}}});

  auto result = version_helpers::intersect_v4_2d(a, b);

  ASSERT_EQ(result.num_rows(), 1);
  ASSERT_EQ(result.rows[0].intervals.size(), 1);
  EXPECT_EQ(result.rows[0].intervals[0].begin, 1500000);
  EXPECT_EQ(result.rows[0].intervals[0].end, 2000000);
}

TEST_F(V4HashConversionTest, NegativeCoordinates) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({-10, {{-100, -50}, {-20, 0}}});
  b.rows.push_back({-10, {{-75, -25}}});

  auto result = version_helpers::intersect_v4_2d(a, b);

  ASSERT_EQ(result.num_rows(), 1);
  ASSERT_EQ(result.rows[0].intervals.size(), 1);
  EXPECT_EQ(result.rows[0].intervals[0].begin, -75);
  EXPECT_EQ(result.rows[0].intervals[0].end, -50);
}

// ============================================================================
// 3D Tests - v4
// ============================================================================

TEST_F(V4HashConversionTest, Simple3DIntersection_KnownResult) {
  run_oracle_test_3d(
      [](DefaultCommonMesh3D& a, DefaultCommonMesh3D& b) {
        a.rows.push_back({0, 0, {{0, 10}, {20, 30}}});
        b.rows.push_back({0, 0, {{5, 15}, {25, 35}}});
      },
      [](const DefaultCommonMesh3D& result) {
        ASSERT_EQ(result.num_rows(), 1);
        EXPECT_EQ(result.rows[0].y, 0);
        EXPECT_EQ(result.rows[0].z, 0);
        ASSERT_EQ(result.rows[0].intervals.size(), 2);
        EXPECT_EQ(result.rows[0].intervals[0].begin, 5);
        EXPECT_EQ(result.rows[0].intervals[0].end, 10);
      },
      version_helpers::intersect_v4_3d);
}

TEST_F(V4HashConversionTest, Different3DZ_NoOverlap) {
  run_oracle_test_3d(
      [](DefaultCommonMesh3D& a, DefaultCommonMesh3D& b) {
        a.rows.push_back({0, 0, {{0, 10}}});
        a.rows.push_back({0, 5, {{0, 10}}});
        a.rows.push_back({0, 10, {{0, 10}}});
        b.rows.push_back({0, 1, {{0, 10}}});
        b.rows.push_back({0, 6, {{0, 10}}});
        b.rows.push_back({0, 11, {{0, 10}}});
      },
      [](const DefaultCommonMesh3D& result) {
        EXPECT_EQ(result.num_rows(), 0);
        EXPECT_EQ(result.num_intervals(), 0);
      },
      version_helpers::intersect_v4_3d);
}

TEST_F(V4HashConversionTest, Multiple3DRowsWithDifferentZ) {
  run_oracle_test_3d(
      [](DefaultCommonMesh3D& a, DefaultCommonMesh3D& b) {
        a.rows.push_back({0, 0, {{0, 100}}});
        a.rows.push_back({10, 5, {{0, 100}}});
        a.rows.push_back({5, 10, {{0, 100}}});
        b.rows.push_back({0, 0, {{50, 150}}});
        b.rows.push_back({10, 3, {{0, 100}}});
        b.rows.push_back({2, 10, {{0, 100}}});
      },
      [](const DefaultCommonMesh3D& result) {
        ASSERT_EQ(result.num_rows(), 1);
        EXPECT_EQ(result.rows[0].y, 0);
        EXPECT_EQ(result.rows[0].z, 0);
        EXPECT_EQ(result.rows[0].intervals[0].begin, 50);
        EXPECT_EQ(result.rows[0].intervals[0].end, 100);
      },
      version_helpers::intersect_v4_3d);
}

TEST_F(V4HashConversionTest, RoundTrip3DConversion_PreservesData) {
  DefaultCommonMesh3D original;
  original.rows.push_back({0, 0, {{0, 10}}});
  original.rows.push_back({5, 3, {{20, 30}}});
  original.rows.push_back({10, 0, {{100, 200}}});

  auto device = MeshConverter3D<hash_based::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(original);
  auto converted = MeshConverter3D<hash_based::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::to_common(device);

  EXPECT_TRUE(common_meshes_equal(original, converted));
}

// ============================================================================
// Test suite for v5 (parallel merge)
// ============================================================================

class V5ParallelMergeConversionTest : public ::testing::Test {
protected:
  using Intersect2D = DefaultCommonMesh2D (*)(const DefaultCommonMesh2D&, const DefaultCommonMesh2D&);
  using Intersect3D = DefaultCommonMesh3D (*)(const DefaultCommonMesh3D&, const DefaultCommonMesh3D&);

  void run_oracle_test_2d(
      const std::function<void(DefaultCommonMesh2D&, DefaultCommonMesh2D&)>& setup,
      const std::function<void(const DefaultCommonMesh2D&)>& verify,
      Intersect2D intersect_fn) {
    DefaultCommonMesh2D a, b;
    setup(a, b);
    auto result = intersect_fn(a, b);
    verify(result);
  }

  void run_oracle_test_3d(
      const std::function<void(DefaultCommonMesh3D&, DefaultCommonMesh3D&)>& setup,
      const std::function<void(const DefaultCommonMesh3D&)>& verify,
      Intersect3D intersect_fn) {
    DefaultCommonMesh3D a, b;
    setup(a, b);
    auto result = intersect_fn(a, b);
    verify(result);
  }
};

// Reuse all the same oracle tests for v5
#define V5_TEST_2D(name, setup, verify) \
TEST_F(V5ParallelMergeConversionTest, name) { \
  run_oracle_test_2d(setup, verify, version_helpers::intersect_v5_2d); \
}

#define V5_TEST_3D(name, setup, verify) \
TEST_F(V5ParallelMergeConversionTest, name) { \
  run_oracle_test_3d(setup, verify, version_helpers::intersect_v5_3d); \
}

// Oracle tests 2D
V5_TEST_2D(SimpleIntersection_KnownResult,
  [](DefaultCommonMesh2D& a, DefaultCommonMesh2D& b) {
    a.rows.push_back({0, {{0, 10}, {20, 30}, {40, 50}}});
    b.rows.push_back({0, {{5, 15}, {25, 35}}});
  },
  [](const DefaultCommonMesh2D& result) {
    ASSERT_EQ(result.num_rows(), 1);
    ASSERT_EQ(result.num_intervals(), 2);
    EXPECT_EQ(result.rows[0].intervals[0].begin, 5);
    EXPECT_EQ(result.rows[0].intervals[0].end, 10);
    EXPECT_EQ(result.rows[0].intervals[1].begin, 25);
    EXPECT_EQ(result.rows[0].intervals[1].end, 30);
  })

V5_TEST_2D(NoOverlap_EmptyResult,
  [](DefaultCommonMesh2D& a, DefaultCommonMesh2D& b) {
    a.rows.push_back({0, {{0, 10}, {20, 30}}});
    b.rows.push_back({0, {{40, 50}, {60, 70}}});
  },
  [](const DefaultCommonMesh2D& result) {
    EXPECT_EQ(result.num_rows(), 0);
    EXPECT_EQ(result.num_intervals(), 0);
  })

// Oracle tests 3D
V5_TEST_3D(Simple3DIntersection_KnownResult,
  [](DefaultCommonMesh3D& a, DefaultCommonMesh3D& b) {
    a.rows.push_back({0, 0, {{0, 10}, {20, 30}}});
    b.rows.push_back({0, 0, {{5, 15}, {25, 35}}});
  },
  [](const DefaultCommonMesh3D& result) {
    ASSERT_EQ(result.num_rows(), 1);
    EXPECT_EQ(result.rows[0].intervals.size(), 2);
    EXPECT_EQ(result.rows[0].intervals[0].begin, 5);
    EXPECT_EQ(result.rows[0].intervals[0].end, 10);
  })

#undef V5_TEST_2D
#undef V5_TEST_3D

// ============================================================================
// Test suite for v6 (direct index)
// ============================================================================

class V6DirectIndexConversionTest : public ::testing::Test {
protected:
  using Intersect2D = DefaultCommonMesh2D (*)(const DefaultCommonMesh2D&, const DefaultCommonMesh2D&);
  using Intersect3D = DefaultCommonMesh3D (*)(const DefaultCommonMesh3D&, const DefaultCommonMesh3D&);

  void run_oracle_test_2d(
      const std::function<void(DefaultCommonMesh2D&, DefaultCommonMesh2D&)>& setup,
      const std::function<void(const DefaultCommonMesh2D&)>& verify,
      Intersect2D intersect_fn) {
    DefaultCommonMesh2D a, b;
    setup(a, b);
    auto result = intersect_fn(a, b);
    verify(result);
  }

  void run_oracle_test_3d(
      const std::function<void(DefaultCommonMesh3D&, DefaultCommonMesh3D&)>& setup,
      const std::function<void(const DefaultCommonMesh3D&)>& verify,
      Intersect3D intersect_fn) {
    DefaultCommonMesh3D a, b;
    setup(a, b);
    auto result = intersect_fn(a, b);
    verify(result);
  }
};

// Reuse all the same oracle tests for v6
#define V6_TEST_2D(name, setup, verify) \
TEST_F(V6DirectIndexConversionTest, name) { \
  run_oracle_test_2d(setup, verify, version_helpers::intersect_v6_2d); \
}

#define V6_TEST_3D(name, setup, verify) \
TEST_F(V6DirectIndexConversionTest, name) { \
  run_oracle_test_3d(setup, verify, version_helpers::intersect_v6_3d); \
}

// Oracle tests 2D
V6_TEST_2D(SimpleIntersection_KnownResult,
  [](DefaultCommonMesh2D& a, DefaultCommonMesh2D& b) {
    a.rows.push_back({0, {{0, 10}, {20, 30}, {40, 50}}});
    b.rows.push_back({0, {{5, 15}, {25, 35}}});
  },
  [](const DefaultCommonMesh2D& result) {
    ASSERT_EQ(result.num_rows(), 1);
    ASSERT_EQ(result.num_intervals(), 2);
    EXPECT_EQ(result.rows[0].intervals[0].begin, 5);
    EXPECT_EQ(result.rows[0].intervals[0].end, 10);
    EXPECT_EQ(result.rows[0].intervals[1].begin, 25);
    EXPECT_EQ(result.rows[0].intervals[1].end, 30);
  })

V6_TEST_2D(NoOverlap_EmptyResult,
  [](DefaultCommonMesh2D& a, DefaultCommonMesh2D& b) {
    a.rows.push_back({0, {{0, 10}, {20, 30}}});
    b.rows.push_back({0, {{40, 50}, {60, 70}}});
  },
  [](const DefaultCommonMesh2D& result) {
    EXPECT_EQ(result.num_rows(), 0);
    EXPECT_EQ(result.num_intervals(), 0);
  })

// Oracle tests 3D
V6_TEST_3D(Simple3DIntersection_KnownResult,
  [](DefaultCommonMesh3D& a, DefaultCommonMesh3D& b) {
    a.rows.push_back({0, 0, {{0, 10}, {20, 30}}});
    b.rows.push_back({0, 0, {{5, 15}, {25, 35}}});
  },
  [](const DefaultCommonMesh3D& result) {
    ASSERT_EQ(result.num_rows(), 1);
    EXPECT_EQ(result.rows[0].intervals.size(), 2);
    EXPECT_EQ(result.rows[0].intervals[0].begin, 5);
    EXPECT_EQ(result.rows[0].intervals[0].end, 10);
  })

// ============================================================================
// Additional tests for V5 - matching baseline coverage
// ============================================================================

TEST_F(V5ParallelMergeConversionTest, TouchingIntervals_NoOverlap) {
  run_oracle_test_2d(
    [](DefaultCommonMesh2D& a, DefaultCommonMesh2D& b) {
      a.rows.push_back({0, {{0, 10}, {20, 30}}});
      b.rows.push_back({0, {{10, 20}}});
    },
    [](const DefaultCommonMesh2D& result) {
      EXPECT_EQ(result.num_rows(), 0);
      EXPECT_EQ(result.num_intervals(), 0);
    },
    version_helpers::intersect_v5_2d);
}

TEST_F(V5ParallelMergeConversionTest, Subset_SingleInterval) {
  run_oracle_test_2d(
    [](DefaultCommonMesh2D& a, DefaultCommonMesh2D& b) {
      a.rows.push_back({0, {{0, 100}}});
      b.rows.push_back({0, {{25, 75}}});
    },
    [](const DefaultCommonMesh2D& result) {
      ASSERT_EQ(result.num_rows(), 1);
      ASSERT_EQ(result.rows[0].intervals.size(), 1);
      EXPECT_EQ(result.rows[0].intervals[0].begin, 25);
      EXPECT_EQ(result.rows[0].intervals[0].end, 75);
    },
    version_helpers::intersect_v5_2d);
}

TEST_F(V5ParallelMergeConversionTest, MultipleRows_PartialOverlap) {
  run_oracle_test_2d(
    [](DefaultCommonMesh2D& a, DefaultCommonMesh2D& b) {
      a.rows.push_back({0, {{0, 100}}});
      a.rows.push_back({10, {{0, 100}}});
      a.rows.push_back({20, {{0, 100}}});
      b.rows.push_back({5, {{0, 100}}});
      b.rows.push_back({10, {{50, 150}}});
      b.rows.push_back({25, {{0, 100}}});
    },
    [](const DefaultCommonMesh2D& result) {
      ASSERT_EQ(result.num_rows(), 1);
      EXPECT_EQ(result.rows[0].y, 10);
      ASSERT_EQ(result.rows[0].intervals.size(), 1);
      EXPECT_EQ(result.rows[0].intervals[0].begin, 50);
      EXPECT_EQ(result.rows[0].intervals[0].end, 100);
    },
    version_helpers::intersect_v5_2d);
}

TEST_F(V5ParallelMergeConversionTest, Commutativity) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 10}, {20, 30}, {40, 50}}});
  b.rows.push_back({0, {{5, 15}, {25, 35}, {45, 55}}});

  auto result_ab = version_helpers::intersect_v5_2d(a, b);
  auto result_ba = version_helpers::intersect_v5_2d(b, a);

  EXPECT_TRUE(common_meshes_equal(result_ab, result_ba))
      << "v5 (parallel merge): Intersection should be commutative: A∩B = B∩A";
}

TEST_F(V5ParallelMergeConversionTest, Idempotence) {
  DefaultCommonMesh2D a;
  a.rows.push_back({0, {{0, 10}, {20, 30}, {40, 50}}});

  auto result = version_helpers::intersect_v5_2d(a, a);

  EXPECT_TRUE(common_meshes_equal(result, a))
      << "v5 (parallel merge): Intersection should be idempotent: A∩A = A";
}

TEST_F(V5ParallelMergeConversionTest, Associativity_WithSubsets) {
  DefaultCommonMesh2D a, b, c;
  a.rows.push_back({0, {{0, 100}}});
  b.rows.push_back({0, {{0, 50}}});
  c.rows.push_back({0, {{0, 25}}});

  auto ab = version_helpers::intersect_v5_2d(a, b);
  auto abc_left = version_helpers::intersect_v5_2d(ab, c);

  auto bc = version_helpers::intersect_v5_2d(b, c);
  auto abc_right = version_helpers::intersect_v5_2d(a, bc);

  EXPECT_TRUE(common_meshes_equal(abc_left, abc_right))
      << "v5 (parallel merge): Intersection should be associative: (A∩B)∩C = A∩(B∩C)";
}

TEST_F(V5ParallelMergeConversionTest, AbsorbingElement) {
  DefaultCommonMesh2D a, empty;
  a.rows.push_back({0, {{0, 10}, {20, 30}}});

  auto result = version_helpers::intersect_v5_2d(a, empty);

  EXPECT_EQ(result.num_rows(), 0);
  EXPECT_EQ(result.num_intervals(), 0);
}

TEST_F(V5ParallelMergeConversionTest, ResultIntervalsDoNotOverlap) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 50}, {60, 100}}});
  b.rows.push_back({0, {{25, 75}}});

  auto result = version_helpers::intersect_v5_2d(a, b);

  for (const auto& row : result.rows) {
    for (size_t i = 1; i < row.intervals.size(); ++i) {
      EXPECT_GE(row.intervals[i].begin, row.intervals[i-1].end)
          << "v5 (parallel merge): Intervals should not overlap";
    }
  }
}

TEST_F(V5ParallelMergeConversionTest, ResultIntervalsAreNonEmpty) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 100}}});
  b.rows.push_back({0, {{25, 75}}});

  auto result = version_helpers::intersect_v5_2d(a, b);

  for (const auto& row : result.rows) {
    for (const auto& interval : row.intervals) {
      EXPECT_LT(interval.begin, interval.end)
          << "v5 (parallel merge): All intervals should be non-empty";
    }
  }
}

TEST_F(V5ParallelMergeConversionTest, ResultIsSubsetOfBoth) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 10}, {20, 30}, {40, 50}}});
  b.rows.push_back({0, {{5, 15}, {25, 35}, {45, 55}}});

  auto result = version_helpers::intersect_v5_2d(a, b);

  EXPECT_LE(result.num_intervals(), a.num_intervals());
  EXPECT_LE(result.num_intervals(), b.num_intervals());
}

TEST_F(V5ParallelMergeConversionTest, EmptyMesh_EmptyResult) {
  DefaultCommonMesh2D empty_a, empty_b;
  auto result = version_helpers::intersect_v5_2d(empty_a, empty_b);
  EXPECT_EQ(result.num_rows(), 0);
  EXPECT_EQ(result.num_intervals(), 0);
}

TEST_F(V5ParallelMergeConversionTest, EmptyMesh_NonEmptyGivesEmpty) {
  DefaultCommonMesh2D a, empty;
  a.rows.push_back({0, {{0, 10}}});

  auto result1 = version_helpers::intersect_v5_2d(a, empty);
  auto result2 = version_helpers::intersect_v5_2d(empty, a);

  EXPECT_EQ(result1.num_rows(), 0);
  EXPECT_EQ(result2.num_rows(), 0);
}

TEST_F(V5ParallelMergeConversionTest, PointIntersection_NoOverlap) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 1}}});
  b.rows.push_back({0, {{1, 2}}});

  auto result = version_helpers::intersect_v5_2d(a, b);

  EXPECT_EQ(result.num_rows(), 0);
  EXPECT_EQ(result.num_intervals(), 0);
}

TEST_F(V5ParallelMergeConversionTest, SinglePointOverlap_TreatedAsOverlap) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 10}}});
  b.rows.push_back({0, {{9, 20}}});

  auto result = version_helpers::intersect_v5_2d(a, b);

  ASSERT_EQ(result.num_rows(), 1);
  ASSERT_EQ(result.rows[0].intervals.size(), 1);
  EXPECT_EQ(result.rows[0].intervals[0].begin, 9);
  EXPECT_EQ(result.rows[0].intervals[0].end, 10);
}

TEST_F(V5ParallelMergeConversionTest, LargeIntervals) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{1000000, 2000000}}});
  b.rows.push_back({0, {{1500000, 2500000}}});

  auto result = version_helpers::intersect_v5_2d(a, b);

  ASSERT_EQ(result.num_rows(), 1);
  ASSERT_EQ(result.rows[0].intervals.size(), 1);
  EXPECT_EQ(result.rows[0].intervals[0].begin, 1500000);
  EXPECT_EQ(result.rows[0].intervals[0].end, 2000000);
}

TEST_F(V5ParallelMergeConversionTest, NegativeCoordinates) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({-10, {{-100, -50}, {-20, 0}}});
  b.rows.push_back({-10, {{-75, -25}}});

  auto result = version_helpers::intersect_v5_2d(a, b);

  ASSERT_EQ(result.num_rows(), 1);
  ASSERT_EQ(result.rows[0].intervals.size(), 1);
  EXPECT_EQ(result.rows[0].intervals[0].begin, -75);
  EXPECT_EQ(result.rows[0].intervals[0].end, -50);
}

TEST_F(V5ParallelMergeConversionTest, Different3DZ_NoOverlap) {
  run_oracle_test_3d(
    [](DefaultCommonMesh3D& a, DefaultCommonMesh3D& b) {
      a.rows.push_back({0, 0, {{0, 10}}});
      a.rows.push_back({0, 5, {{0, 10}}});
      a.rows.push_back({0, 10, {{0, 10}}});
      b.rows.push_back({0, 1, {{0, 10}}});
      b.rows.push_back({0, 6, {{0, 10}}});
      b.rows.push_back({0, 11, {{0, 10}}});
    },
    [](const DefaultCommonMesh3D& result) {
      EXPECT_EQ(result.num_rows(), 0);
      EXPECT_EQ(result.num_intervals(), 0);
    },
    version_helpers::intersect_v5_3d);
}

TEST_F(V5ParallelMergeConversionTest, Multiple3DRowsWithDifferentZ) {
  run_oracle_test_3d(
    [](DefaultCommonMesh3D& a, DefaultCommonMesh3D& b) {
      a.rows.push_back({0, 0, {{0, 100}}});
      a.rows.push_back({10, 5, {{0, 100}}});
      a.rows.push_back({5, 10, {{0, 100}}});
      b.rows.push_back({0, 0, {{50, 150}}});
      b.rows.push_back({10, 3, {{0, 100}}});
      b.rows.push_back({2, 10, {{0, 100}}});
    },
    [](const DefaultCommonMesh3D& result) {
      ASSERT_EQ(result.num_rows(), 1);
      EXPECT_EQ(result.rows[0].y, 0);
      EXPECT_EQ(result.rows[0].z, 0);
      EXPECT_EQ(result.rows[0].intervals[0].begin, 50);
      EXPECT_EQ(result.rows[0].intervals[0].end, 100);
    },
    version_helpers::intersect_v5_3d);
}

TEST_F(V5ParallelMergeConversionTest, RoundTrip3DConversion_PreservesData) {
  DefaultCommonMesh3D original;
  original.rows.push_back({0, 0, {{0, 10}}});
  original.rows.push_back({5, 3, {{20, 30}}});
  original.rows.push_back({10, 0, {{100, 200}}});

  auto device = MeshConverter3D<parallel_merge::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(original);
  auto converted = MeshConverter3D<parallel_merge::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::to_common(device);

  EXPECT_TRUE(common_meshes_equal(original, converted));
}

// ============================================================================
// Additional tests for V6 - matching baseline coverage
// ============================================================================

TEST_F(V6DirectIndexConversionTest, TouchingIntervals_NoOverlap) {
  run_oracle_test_2d(
    [](DefaultCommonMesh2D& a, DefaultCommonMesh2D& b) {
      a.rows.push_back({0, {{0, 10}, {20, 30}}});
      b.rows.push_back({0, {{10, 20}}});
    },
    [](const DefaultCommonMesh2D& result) {
      EXPECT_EQ(result.num_rows(), 0);
      EXPECT_EQ(result.num_intervals(), 0);
    },
    version_helpers::intersect_v6_2d);
}

TEST_F(V6DirectIndexConversionTest, Subset_SingleInterval) {
  run_oracle_test_2d(
    [](DefaultCommonMesh2D& a, DefaultCommonMesh2D& b) {
      a.rows.push_back({0, {{0, 100}}});
      b.rows.push_back({0, {{25, 75}}});
    },
    [](const DefaultCommonMesh2D& result) {
      ASSERT_EQ(result.num_rows(), 1);
      ASSERT_EQ(result.rows[0].intervals.size(), 1);
      EXPECT_EQ(result.rows[0].intervals[0].begin, 25);
      EXPECT_EQ(result.rows[0].intervals[0].end, 75);
    },
    version_helpers::intersect_v6_2d);
}

TEST_F(V6DirectIndexConversionTest, MultipleRows_PartialOverlap) {
  run_oracle_test_2d(
    [](DefaultCommonMesh2D& a, DefaultCommonMesh2D& b) {
      a.rows.push_back({0, {{0, 100}}});
      a.rows.push_back({10, {{0, 100}}});
      a.rows.push_back({20, {{0, 100}}});
      b.rows.push_back({5, {{0, 100}}});
      b.rows.push_back({10, {{50, 150}}});
      b.rows.push_back({25, {{0, 100}}});
    },
    [](const DefaultCommonMesh2D& result) {
      ASSERT_EQ(result.num_rows(), 1);
      EXPECT_EQ(result.rows[0].y, 10);
      ASSERT_EQ(result.rows[0].intervals.size(), 1);
      EXPECT_EQ(result.rows[0].intervals[0].begin, 50);
      EXPECT_EQ(result.rows[0].intervals[0].end, 100);
    },
    version_helpers::intersect_v6_2d);
}

TEST_F(V6DirectIndexConversionTest, Commutativity) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 10}, {20, 30}, {40, 50}}});
  b.rows.push_back({0, {{5, 15}, {25, 35}, {45, 55}}});

  auto result_ab = version_helpers::intersect_v6_2d(a, b);
  auto result_ba = version_helpers::intersect_v6_2d(b, a);

  EXPECT_TRUE(common_meshes_equal(result_ab, result_ba))
      << "v6 (direct index): Intersection should be commutative: A∩B = B∩A";
}

TEST_F(V6DirectIndexConversionTest, Idempotence) {
  DefaultCommonMesh2D a;
  a.rows.push_back({0, {{0, 10}, {20, 30}, {40, 50}}});

  auto result = version_helpers::intersect_v6_2d(a, a);

  EXPECT_TRUE(common_meshes_equal(result, a))
      << "v6 (direct index): Intersection should be idempotent: A∩A = A";
}

TEST_F(V6DirectIndexConversionTest, Associativity_WithSubsets) {
  DefaultCommonMesh2D a, b, c;
  a.rows.push_back({0, {{0, 100}}});
  b.rows.push_back({0, {{0, 50}}});
  c.rows.push_back({0, {{0, 25}}});

  auto ab = version_helpers::intersect_v6_2d(a, b);
  auto abc_left = version_helpers::intersect_v6_2d(ab, c);

  auto bc = version_helpers::intersect_v6_2d(b, c);
  auto abc_right = version_helpers::intersect_v6_2d(a, bc);

  EXPECT_TRUE(common_meshes_equal(abc_left, abc_right))
      << "v6 (direct index): Intersection should be associative: (A∩B)∩C = A∩(B∩C)";
}

TEST_F(V6DirectIndexConversionTest, AbsorbingElement) {
  DefaultCommonMesh2D a, empty;
  a.rows.push_back({0, {{0, 10}, {20, 30}}});

  auto result = version_helpers::intersect_v6_2d(a, empty);

  EXPECT_EQ(result.num_rows(), 0);
  EXPECT_EQ(result.num_intervals(), 0);
}

TEST_F(V6DirectIndexConversionTest, ResultIntervalsDoNotOverlap) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 50}, {60, 100}}});
  b.rows.push_back({0, {{25, 75}}});

  auto result = version_helpers::intersect_v6_2d(a, b);

  for (const auto& row : result.rows) {
    for (size_t i = 1; i < row.intervals.size(); ++i) {
      EXPECT_GE(row.intervals[i].begin, row.intervals[i-1].end)
          << "v6 (direct index): Intervals should not overlap";
    }
  }
}

TEST_F(V6DirectIndexConversionTest, ResultIntervalsAreNonEmpty) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 100}}});
  b.rows.push_back({0, {{25, 75}}});

  auto result = version_helpers::intersect_v6_2d(a, b);

  for (const auto& row : result.rows) {
    for (const auto& interval : row.intervals) {
      EXPECT_LT(interval.begin, interval.end)
          << "v6 (direct index): All intervals should be non-empty";
    }
  }
}

TEST_F(V6DirectIndexConversionTest, ResultIsSubsetOfBoth) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 10}, {20, 30}, {40, 50}}});
  b.rows.push_back({0, {{5, 15}, {25, 35}, {45, 55}}});

  auto result = version_helpers::intersect_v6_2d(a, b);

  EXPECT_LE(result.num_intervals(), a.num_intervals());
  EXPECT_LE(result.num_intervals(), b.num_intervals());
}

TEST_F(V6DirectIndexConversionTest, EmptyMesh_EmptyResult) {
  DefaultCommonMesh2D empty_a, empty_b;
  auto result = version_helpers::intersect_v6_2d(empty_a, empty_b);
  EXPECT_EQ(result.num_rows(), 0);
  EXPECT_EQ(result.num_intervals(), 0);
}

TEST_F(V6DirectIndexConversionTest, EmptyMesh_NonEmptyGivesEmpty) {
  DefaultCommonMesh2D a, empty;
  a.rows.push_back({0, {{0, 10}}});

  auto result1 = version_helpers::intersect_v6_2d(a, empty);
  auto result2 = version_helpers::intersect_v6_2d(empty, a);

  EXPECT_EQ(result1.num_rows(), 0);
  EXPECT_EQ(result2.num_rows(), 0);
}

TEST_F(V6DirectIndexConversionTest, PointIntersection_NoOverlap) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 1}}});
  b.rows.push_back({0, {{1, 2}}});

  auto result = version_helpers::intersect_v6_2d(a, b);

  EXPECT_EQ(result.num_rows(), 0);
  EXPECT_EQ(result.num_intervals(), 0);
}

TEST_F(V6DirectIndexConversionTest, SinglePointOverlap_TreatedAsOverlap) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{0, 10}}});
  b.rows.push_back({0, {{9, 20}}});

  auto result = version_helpers::intersect_v6_2d(a, b);

  ASSERT_EQ(result.num_rows(), 1);
  ASSERT_EQ(result.rows[0].intervals.size(), 1);
  EXPECT_EQ(result.rows[0].intervals[0].begin, 9);
  EXPECT_EQ(result.rows[0].intervals[0].end, 10);
}

TEST_F(V6DirectIndexConversionTest, LargeIntervals) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({0, {{1000000, 2000000}}});
  b.rows.push_back({0, {{1500000, 2500000}}});

  auto result = version_helpers::intersect_v6_2d(a, b);

  ASSERT_EQ(result.num_rows(), 1);
  ASSERT_EQ(result.rows[0].intervals.size(), 1);
  EXPECT_EQ(result.rows[0].intervals[0].begin, 1500000);
  EXPECT_EQ(result.rows[0].intervals[0].end, 2000000);
}

TEST_F(V6DirectIndexConversionTest, NegativeCoordinates) {
  DefaultCommonMesh2D a, b;
  a.rows.push_back({-10, {{-100, -50}, {-20, 0}}});
  b.rows.push_back({-10, {{-75, -25}}});

  auto result = version_helpers::intersect_v6_2d(a, b);

  ASSERT_EQ(result.num_rows(), 1);
  ASSERT_EQ(result.rows[0].intervals.size(), 1);
  EXPECT_EQ(result.rows[0].intervals[0].begin, -75);
  EXPECT_EQ(result.rows[0].intervals[0].end, -50);
}

TEST_F(V6DirectIndexConversionTest, Different3DZ_NoOverlap) {
  run_oracle_test_3d(
    [](DefaultCommonMesh3D& a, DefaultCommonMesh3D& b) {
      a.rows.push_back({0, 0, {{0, 10}}});
      a.rows.push_back({0, 5, {{0, 10}}});
      a.rows.push_back({0, 10, {{0, 10}}});
      b.rows.push_back({0, 1, {{0, 10}}});
      b.rows.push_back({0, 6, {{0, 10}}});
      b.rows.push_back({0, 11, {{0, 10}}});
    },
    [](const DefaultCommonMesh3D& result) {
      EXPECT_EQ(result.num_rows(), 0);
      EXPECT_EQ(result.num_intervals(), 0);
    },
    version_helpers::intersect_v6_3d);
}

TEST_F(V6DirectIndexConversionTest, Multiple3DRowsWithDifferentZ) {
  run_oracle_test_3d(
    [](DefaultCommonMesh3D& a, DefaultCommonMesh3D& b) {
      a.rows.push_back({0, 0, {{0, 100}}});
      a.rows.push_back({10, 5, {{0, 100}}});
      a.rows.push_back({5, 10, {{0, 100}}});
      b.rows.push_back({0, 0, {{50, 150}}});
      b.rows.push_back({10, 3, {{0, 100}}});
      b.rows.push_back({2, 10, {{0, 100}}});
    },
    [](const DefaultCommonMesh3D& result) {
      ASSERT_EQ(result.num_rows(), 1);
      EXPECT_EQ(result.rows[0].y, 0);
      EXPECT_EQ(result.rows[0].z, 0);
      EXPECT_EQ(result.rows[0].intervals[0].begin, 50);
      EXPECT_EQ(result.rows[0].intervals[0].end, 100);
    },
    version_helpers::intersect_v6_3d);
}

TEST_F(V6DirectIndexConversionTest, RoundTrip3DConversion_PreservesData) {
  DefaultCommonMesh3D original;
  original.rows.push_back({0, 0, {{0, 10}}});
  original.rows.push_back({5, 3, {{20, 30}}});
  original.rows.push_back({10, 0, {{100, 200}}});

  auto device = MeshConverter3D<direct_index::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(original);
  auto converted = MeshConverter3D<direct_index::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::to_common(device);

  EXPECT_TRUE(common_meshes_equal(original, converted));
}

#undef V6_TEST_2D
#undef V6_TEST_3D

#endif // SUBSETIX_ENABLE_PLAYGROUND
