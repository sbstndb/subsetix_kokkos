// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <gtest/gtest.h>
#include <experimental/subsetix/csr/set_algebra/v1.hpp>
#include <experimental/subsetix/csr/set_algebra/v2.hpp>
#include <experimental/subsetix/csr/set_algebra/v3.hpp>
#include "test_common_format.hpp"
#include "test_random_mesh_generator.hpp"
#include <Kokkos_Core.hpp>

using namespace experimental::subsetix::csr;
using namespace experimental::subsetix::csr::test;

// ============================================================================
// Random Comparison Tests: v1 vs v2 vs v3 using Random Geometries
// ============================================================================

/**
 * @brief Test suite for comparing v1, v2, v3 using random geometries
 *
 * These tests generate random meshes and verify that all versions
 * produce bitwise identical results.
 */
class RandomComparisonTest : public ::testing::Test {
protected:
  // Run intersection with v1 (reference)
  DefaultCommonMesh2D intersect_v1_2d(const DefaultCommonMesh2D& a, const DefaultCommonMesh2D& b) {
    return test::intersect_v1_2d(a, b);
  }

  DefaultCommonMesh3D intersect_v1_3d(const DefaultCommonMesh3D& a, const DefaultCommonMesh3D& b) {
    return test::intersect_v1_3d(a, b);
  }

  // Run intersection with v2
  DefaultCommonMesh2D intersect_v2_2d(const DefaultCommonMesh2D& a, const DefaultCommonMesh2D& b) {
    return test::intersect_v2_2d(a, b);
  }

  DefaultCommonMesh3D intersect_v2_3d(const DefaultCommonMesh3D& a, const DefaultCommonMesh3D& b) {
    return test::intersect_v2_3d(a, b);
  }

  // Run intersection with v3
  DefaultCommonMesh2D intersect_v3_2d(const DefaultCommonMesh2D& a, const DefaultCommonMesh2D& b) {
    return test::intersect_v3_2d(a, b);
  }

  DefaultCommonMesh3D intersect_v3_3d(const DefaultCommonMesh3D& a, const DefaultCommonMesh3D& b) {
    return test::intersect_v3_3d(a, b);
  }
};

// ============================================================================
// 2D Random Test with Random Bounds
// ============================================================================

/**
 * @brief Test v1/v2/v3 produce identical results on 2D random meshes
 *
 * This test generates random configuration parameters (num_rows, intervals, seed)
 * within reasonable bounds and runs multiple iterations to cover a wide range
 * of test cases without code duplication.
 */
TEST_F(RandomComparisonTest, AllVersions2D_RandomBounds) {
  // Fixed seed for test reproducibility
  std::mt19937 seed_gen(42);

  std::uniform_int_distribution<int> rows_dist(1, 1024);
  std::uniform_int_distribution<int> intervals_dist(1, 10);
  std::uniform_int_distribution<int> seed_dist(1, 10000);
  std::uniform_int_distribution<int> length_dist(1, 500);

  const int num_iterations = 15;

  for (int iter = 0; iter < num_iterations; ++iter) {
    // Generate random config parameters
    int num_rows = rows_dist(seed_gen);
    int min_intervals = intervals_dist(seed_gen);
    int max_intervals = std::min(min_intervals + intervals_dist(seed_gen), 20);
    int seed = seed_dist(seed_gen);
    int max_length = length_dist(seed_gen);

    // Build config from MediumConfig baseline
    RandomMeshConfig config = MediumConfig();
    config.seed = seed;
    config.num_rows_min = num_rows;
    config.num_rows_max = num_rows + 1;  // Exact row count
    config.intervals_per_row_min = min_intervals;
    config.intervals_per_row_max = max_intervals;
    config.interval_length_max = max_length;

    // Generate two random meshes
    auto mesh_a = RandomMeshGenerator::generate_2d(config);
    config.seed++;
    auto mesh_b = RandomMeshGenerator::generate_2d(config);

    // Run all versions
    auto result_v1 = intersect_v1_2d(mesh_a, mesh_b);
    auto result_v2 = intersect_v2_2d(mesh_a, mesh_b);
    auto result_v3 = intersect_v3_2d(mesh_a, mesh_b);

    // All should be identical (bitwise comparison)
    EXPECT_TRUE(common_meshes_equal(result_v1, result_v2))
        << "v1 and v2 produced different 2D results (iteration=" << iter
        << ", seed=" << seed << ", rows=" << num_rows << ")";
    EXPECT_TRUE(common_meshes_equal(result_v1, result_v3))
        << "v1 and v3 produced different 2D results (iteration=" << iter
        << ", seed=" << seed << ", rows=" << num_rows << ")";
  }
}

// ============================================================================
// 3D Random Test with Random Bounds
// ============================================================================

/**
 * @brief Test v1/v2/v3 produce identical results on 3D random meshes
 *
 * Same logic as 2D test but for 3D meshes with (y, z) row keys.
 */
TEST_F(RandomComparisonTest, AllVersions3D_RandomBounds) {
  std::mt19937 seed_gen(43);
  std::uniform_int_distribution<int> rows_dist(1, 30);  // Fewer rows for 3D (faster)
  std::uniform_int_distribution<int> intervals_dist(1, 8);
  std::uniform_int_distribution<int> seed_dist(1, 10000);
  std::uniform_int_distribution<int> length_dist(1, 500);

  const int num_iterations = 10;

  for (int iter = 0; iter < num_iterations; ++iter) {
    int num_rows = rows_dist(seed_gen);
    int min_intervals = intervals_dist(seed_gen);
    int max_intervals = std::min(min_intervals + intervals_dist(seed_gen), 15);
    int seed = seed_dist(seed_gen);
    int max_length = length_dist(seed_gen);

    // Build config from MediumConfig baseline (includes proper z_max=1024)
    RandomMeshConfig config = MediumConfig();
    config.seed = seed;
    config.num_rows_min = num_rows;
    config.num_rows_max = num_rows + 1;
    config.intervals_per_row_min = min_intervals;
    config.intervals_per_row_max = max_intervals;
    config.interval_length_max = max_length;

    auto mesh_a = RandomMeshGenerator::generate_3d(config);
    config.seed++;
    auto mesh_b = RandomMeshGenerator::generate_3d(config);

    auto result_v1 = intersect_v1_3d(mesh_a, mesh_b);
    auto result_v2 = intersect_v2_3d(mesh_a, mesh_b);
    auto result_v3 = intersect_v3_3d(mesh_a, mesh_b);

    EXPECT_TRUE(common_meshes_equal(result_v1, result_v2))
        << "v1 and v2 produced different 3D results (iteration=" << iter
        << ", seed=" << seed << ", rows=" << num_rows << ")";
    EXPECT_TRUE(common_meshes_equal(result_v1, result_v3))
        << "v1 and v3 produced different 3D results (iteration=" << iter
        << ", seed=" << seed << ", rows=" << num_rows << ")";
  }
}

// ============================================================================
// Mathematical Properties Test with Random Meshes
// ============================================================================

/**
 * @brief Test mathematical properties (commutativity, associativity) on random meshes
 *
 * Verifies that intersection satisfies:
 * - Commutativity: A ∩ B = B ∩ A
 * - Associativity: (A ∩ B) ∩ C = A ∩ (B ∩ C)
 * - Idempotence: A ∩ A = A
 */
TEST_F(RandomComparisonTest, AllVersions_MathProperties_Random) {
  std::mt19937 seed_gen(44);
  std::uniform_int_distribution<int> rows_dist(5, 25);
  std::uniform_int_distribution<int> intervals_dist(1, 6);
  std::uniform_int_distribution<int> seed_dist(1, 5000);

  const int num_iterations = 8;

  for (int iter = 0; iter < num_iterations; ++iter) {
    // Generate random config with high y_max to avoid duplicate row keys
    int num_rows = rows_dist(seed_gen);
    int min_intervals = intervals_dist(seed_gen);
    int max_intervals = std::min(min_intervals + intervals_dist(seed_gen), 10);
    int seed = seed_dist(seed_gen);

    RandomMeshConfig config;
    config.seed = seed;
    config.num_rows_min = num_rows;
    config.num_rows_max = num_rows + 1;
    config.intervals_per_row_min = min_intervals;
    config.intervals_per_row_max = max_intervals;
    config.y_max = 10000;  // High y_max ensures unique row keys for math properties

    // Generate three random meshes
    DefaultCommonMesh2D mesh_a = RandomMeshGenerator::generate_2d(config);
    config.seed++;
    DefaultCommonMesh2D mesh_b = RandomMeshGenerator::generate_2d(config);
    config.seed++;
    DefaultCommonMesh2D mesh_c = RandomMeshGenerator::generate_2d(config);

    // Test Commutativity: A ∩ B = B ∩ A
    auto v1_ab = intersect_v1_2d(mesh_a, mesh_b);
    auto v1_ba = intersect_v1_2d(mesh_b, mesh_a);
    EXPECT_TRUE(common_meshes_equal(v1_ab, v1_ba))
        << "v1 is not commutative (iteration=" << iter << ")";

    auto v2_ab = intersect_v2_2d(mesh_a, mesh_b);
    auto v2_ba = intersect_v2_2d(mesh_b, mesh_a);
    EXPECT_TRUE(common_meshes_equal(v2_ab, v2_ba))
        << "v2 is not commutative (iteration=" << iter << ")";

    auto v3_ab = intersect_v3_2d(mesh_a, mesh_b);
    auto v3_ba = intersect_v3_2d(mesh_b, mesh_a);
    EXPECT_TRUE(common_meshes_equal(v3_ab, v3_ba))
        << "v3 is not commutative (iteration=" << iter << ")";

    // Test Associativity: (A ∩ B) ∩ C = A ∩ (B ∩ C)
    auto v1_ab_c = intersect_v1_2d(v1_ab, mesh_c);  // (A ∩ B) ∩ C
    auto v1_bc = intersect_v1_2d(mesh_b, mesh_c);
    auto v1_a_bc = intersect_v1_2d(mesh_a, v1_bc);   // A ∩ (B ∩ C)
    EXPECT_TRUE(common_meshes_equal(v1_ab_c, v1_a_bc))
        << "v1 is not associative (iteration=" << iter << ")";

    auto v2_ab_c = intersect_v2_2d(v2_ab, mesh_c);
    auto v2_bc = intersect_v2_2d(mesh_b, mesh_c);
    auto v2_a_bc = intersect_v2_2d(mesh_a, v2_bc);
    EXPECT_TRUE(common_meshes_equal(v2_ab_c, v2_a_bc))
        << "v2 is not associative (iteration=" << iter << ")";

    auto v3_ab_c = intersect_v3_2d(v3_ab, mesh_c);
    auto v3_bc = intersect_v3_2d(mesh_b, mesh_c);
    auto v3_a_bc = intersect_v3_2d(mesh_a, v3_bc);
    EXPECT_TRUE(common_meshes_equal(v3_ab_c, v3_a_bc))
        << "v3 is not associative (iteration=" << iter << ")";

    // Test Idempotence: A ∩ A = A
    auto v1_aa = intersect_v1_2d(mesh_a, mesh_a);
    EXPECT_TRUE(common_meshes_equal(v1_aa, mesh_a))
        << "v1 is not idempotent (iteration=" << iter << ")";

    auto v2_aa = intersect_v2_2d(mesh_a, mesh_a);
    EXPECT_TRUE(common_meshes_equal(v2_aa, mesh_a))
        << "v2 is not idempotent (iteration=" << iter << ")";

    auto v3_aa = intersect_v3_2d(mesh_a, mesh_a);
    EXPECT_TRUE(common_meshes_equal(v3_aa, mesh_a))
        << "v3 is not idempotent (iteration=" << iter << ")";
  }
}

#endif
