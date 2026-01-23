// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#ifdef SUBSETIX_ENABLE_PLAYGROUND

#include <gtest/gtest.h>
#include <playground/subsetix/csr/intersection/algorithm/baseline.hpp>
#include <playground/subsetix/csr/intersection/algorithm/optimized.hpp>
#include "test_common_format.hpp"
#include "test_random_mesh_generator.hpp"
#include <Kokkos_Core.hpp>

using namespace playground::subsetix::csr::intersection;
using namespace playground::subsetix::csr::intersection::test;

// ============================================================================
// Random Comparison Tests: baseline vs optimized using Random Geometries
// ============================================================================

/**
 * @brief Test suite for comparing baseline and optimized using random geometries
 *
 * These tests generate random meshes and verify that both versions
 * produce bitwise identical results.
 */
class RandomComparisonTest : public ::testing::Test {
protected:
  // Run intersection with baseline (reference)
  DefaultCommonMesh2D intersect_baseline_2d(const DefaultCommonMesh2D& a, const DefaultCommonMesh2D& b) {
    return test::intersect_baseline_2d(a, b);
  }

  DefaultCommonMesh3D intersect_baseline_3d(const DefaultCommonMesh3D& a, const DefaultCommonMesh3D& b) {
    return test::intersect_baseline_3d(a, b);
  }

  // Run intersection with optimized
  DefaultCommonMesh2D intersect_optimized_2d(const DefaultCommonMesh2D& a, const DefaultCommonMesh2D& b) {
    return test::intersect_optimized_2d(a, b);
  }

  DefaultCommonMesh3D intersect_optimized_3d(const DefaultCommonMesh3D& a, const DefaultCommonMesh3D& b) {
    return test::intersect_optimized_3d(a, b);
  }
};

// ============================================================================
// 2D Random Test with Random Bounds
// ============================================================================

/**
 * @brief Test baseline/optimized produce identical results on 2D random meshes
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
    // Calculate sparsity for desired row count
    config.sparsity = static_cast<double>(num_rows) / (config.y_max - config.y_min);
    config.intervals_per_row_min = min_intervals;
    config.intervals_per_row_max = max_intervals;
    config.interval_length_max = max_length;

    // Generate two random meshes
    auto mesh_a = RandomMeshGenerator::generate_2d(config);
    config.seed++;
    auto mesh_b = RandomMeshGenerator::generate_2d(config);

    // Run both versions
    auto result_baseline = intersect_baseline_2d(mesh_a, mesh_b);
    auto result_optimized = intersect_optimized_2d(mesh_a, mesh_b);

    // Both should be identical (bitwise comparison)
    EXPECT_TRUE(common_meshes_equal(result_baseline, result_optimized))
        << "baseline and optimized produced different 2D results (iteration=" << iter
        << ", seed=" << seed << ", rows=" << num_rows << ")";
  }
}

// ============================================================================
// 3D Random Test with Random Bounds
// ============================================================================

/**
 * @brief Test baseline/optimized produce identical results on 3D random meshes
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

    // Build config from MediumConfig baseline (includes proper z_max=512)
    RandomMeshConfig config = MediumConfig();
    config.seed = seed;
    // Calculate sparsity for desired row count
    // For 3D: num_rows = sparsity * y_extent * z_extent
    int y_extent = config.y_max - config.y_min;
    int z_extent = config.z_max - config.z_min;
    config.sparsity = static_cast<double>(num_rows) / (y_extent * z_extent);
    config.intervals_per_row_min = min_intervals;
    config.intervals_per_row_max = max_intervals;
    config.interval_length_max = max_length;

    auto mesh_a = RandomMeshGenerator::generate_3d(config);
    config.seed++;
    auto mesh_b = RandomMeshGenerator::generate_3d(config);

    auto result_baseline = intersect_baseline_3d(mesh_a, mesh_b);
    auto result_optimized = intersect_optimized_3d(mesh_a, mesh_b);

    EXPECT_TRUE(common_meshes_equal(result_baseline, result_optimized))
        << "baseline and optimized produced different 3D results (iteration=" << iter
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
    // Calculate sparsity for desired row count
    config.y_max = 10000;  // High y_max ensures unique row keys for math properties
    config.sparsity = static_cast<double>(num_rows) / (config.y_max - config.y_min);
    config.intervals_per_row_min = min_intervals;
    config.intervals_per_row_max = max_intervals;

    // Generate three random meshes
    DefaultCommonMesh2D mesh_a = RandomMeshGenerator::generate_2d(config);
    config.seed++;
    DefaultCommonMesh2D mesh_b = RandomMeshGenerator::generate_2d(config);
    config.seed++;
    DefaultCommonMesh2D mesh_c = RandomMeshGenerator::generate_2d(config);

    // Test Commutativity: A ∩ B = B ∩ A
    auto baseline_ab = intersect_baseline_2d(mesh_a, mesh_b);
    auto baseline_ba = intersect_baseline_2d(mesh_b, mesh_a);
    EXPECT_TRUE(common_meshes_equal(baseline_ab, baseline_ba))
        << "baseline is not commutative (iteration=" << iter << ")";

    auto optimized_ab = intersect_optimized_2d(mesh_a, mesh_b);
    auto optimized_ba = intersect_optimized_2d(mesh_b, mesh_a);
    EXPECT_TRUE(common_meshes_equal(optimized_ab, optimized_ba))
        << "optimized is not commutative (iteration=" << iter << ")";

    // Test Associativity: (A ∩ B) ∩ C = A ∩ (B ∩ C)
    auto baseline_ab_c = intersect_baseline_2d(baseline_ab, mesh_c);  // (A ∩ B) ∩ C
    auto baseline_bc = intersect_baseline_2d(mesh_b, mesh_c);
    auto baseline_a_bc = intersect_baseline_2d(mesh_a, baseline_bc);   // A ∩ (B ∩ C)
    EXPECT_TRUE(common_meshes_equal(baseline_ab_c, baseline_a_bc))
        << "baseline is not associative (iteration=" << iter << ")";

    auto optimized_ab_c = intersect_optimized_2d(optimized_ab, mesh_c);
    auto optimized_bc = intersect_optimized_2d(mesh_b, mesh_c);
    auto optimized_a_bc = intersect_optimized_2d(mesh_a, optimized_bc);
    EXPECT_TRUE(common_meshes_equal(optimized_ab_c, optimized_a_bc))
        << "optimized is not associative (iteration=" << iter << ")";

    // Test Idempotence: A ∩ A = A
    auto baseline_aa = intersect_baseline_2d(mesh_a, mesh_a);
    EXPECT_TRUE(common_meshes_equal(baseline_aa, mesh_a))
        << "baseline is not idempotent (iteration=" << iter << ")";

    auto optimized_aa = intersect_optimized_2d(mesh_a, mesh_a);
    EXPECT_TRUE(common_meshes_equal(optimized_aa, mesh_a))
        << "optimized is not idempotent (iteration=" << iter << ")";
  }
}

#endif
