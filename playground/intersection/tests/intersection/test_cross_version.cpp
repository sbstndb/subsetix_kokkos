// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#ifdef SUBSETIX_ENABLE_PLAYGROUND

#include <gtest/gtest.h>
#include <playground/subsetix/csr/intersection/algorithm/baseline.hpp>
#include <playground/subsetix/csr/intersection/algorithm/optimized.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v4_hash.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v5_parallel_merge.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v6_direct_index.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v7_soa_optimized.hpp>
#include "test_common_format.hpp"
#include "test_random_mesh_generator.hpp"
#include <Kokkos_Core.hpp>

using namespace playground::subsetix::csr::intersection;
using namespace playground::subsetix::csr::intersection::test;

// ============================================================================
// Random Comparison Tests: baseline vs all other versions
// ============================================================================

/**
 * @brief Test suite for comparing baseline with all other versions
 *
 * These tests generate random meshes and verify that all versions
 * produce bitwise identical results.
 */
class AllVersionsComparisonTest : public ::testing::Test {
protected:
  // Helper to get baseline result (reference)
  DefaultCommonMesh2D get_baseline_2d(const DefaultCommonMesh2D& a, const DefaultCommonMesh2D& b) {
    return test::intersect_baseline_2d(a, b);
  }

  DefaultCommonMesh3D get_baseline_3d(const DefaultCommonMesh3D& a, const DefaultCommonMesh3D& b) {
    return test::intersect_baseline_3d(a, b);
  }
};

// ============================================================================
// 2D Random Test with Random Bounds
// ============================================================================

/**
 * @brief Test all versions produce identical results to baseline on 2D random meshes
 *
 * This test generates random configuration parameters and runs multiple iterations
 * to cover a wide range of test cases.
 */
TEST_F(AllVersionsComparisonTest, AllVersions2D_RandomBounds) {
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
    config.sparsity = static_cast<double>(num_rows) / (config.y_max - config.y_min);
    config.intervals_per_row_min = min_intervals;
    config.intervals_per_row_max = max_intervals;
    config.interval_length_max = max_length;

    // Generate two random meshes
    auto mesh_a = RandomMeshGenerator::generate_2d(config);
    config.seed++;
    auto mesh_b = RandomMeshGenerator::generate_2d(config);

    // Get baseline result (reference)
    auto baseline_result = get_baseline_2d(mesh_a, mesh_b);

    // Test all versions against baseline
    auto optimized_result = test::intersect_optimized_2d(mesh_a, mesh_b);
    auto v4_result = test::intersect_v4_hash_2d(mesh_a, mesh_b);
    auto v5_result = test::intersect_v5_parallel_merge_2d(mesh_a, mesh_b);
    auto v6_result = test::intersect_v6_direct_index_2d(mesh_a, mesh_b);
    auto v7_result = test::intersect_v7_soa_optimized_2d(mesh_a, mesh_b);

    // All should be identical to baseline (bitwise comparison)
    EXPECT_TRUE(common_meshes_equal(baseline_result, optimized_result))
        << "optimized differs from baseline in 2D (iteration=" << iter
        << ", seed=" << seed << ", rows=" << num_rows << ")";

    EXPECT_TRUE(common_meshes_equal(baseline_result, v4_result))
        << "v4 (hash) differs from baseline in 2D (iteration=" << iter
        << ", seed=" << seed << ", rows=" << num_rows << ")";

    EXPECT_TRUE(common_meshes_equal(baseline_result, v5_result))
        << "v5 (parallel_merge) differs from baseline in 2D (iteration=" << iter
        << ", seed=" << seed << ", rows=" << num_rows << ")";

    EXPECT_TRUE(common_meshes_equal(baseline_result, v6_result))
        << "v6 (direct_index) differs from baseline in 2D (iteration=" << iter
        << ", seed=" << seed << ", rows=" << num_rows << ")";

    EXPECT_TRUE(common_meshes_equal(baseline_result, v7_result))
        << "v7 (soa_optimized) differs from baseline in 2D (iteration=" << iter
        << ", seed=" << seed << ", rows=" << num_rows << ")";
  }
}

// ============================================================================
// 3D Random Test with Random Bounds
// ============================================================================

/**
 * @brief Test all versions produce identical results to baseline on 3D random meshes
 *
 * Same logic as 2D test but for 3D meshes with (y, z) row keys.
 */
TEST_F(AllVersionsComparisonTest, AllVersions3D_RandomBounds) {
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
    int y_extent = config.y_max - config.y_min;
    int z_extent = config.z_max - config.z_min;
    config.sparsity = static_cast<double>(num_rows) / (y_extent * z_extent);
    config.intervals_per_row_min = min_intervals;
    config.intervals_per_row_max = max_intervals;
    config.interval_length_max = max_length;

    auto mesh_a = RandomMeshGenerator::generate_3d(config);
    config.seed++;
    auto mesh_b = RandomMeshGenerator::generate_3d(config);

    auto baseline_result = get_baseline_3d(mesh_a, mesh_b);

    auto optimized_result = test::intersect_optimized_3d(mesh_a, mesh_b);
    auto v4_result = test::intersect_v4_hash_3d(mesh_a, mesh_b);
    auto v5_result = test::intersect_v5_parallel_merge_3d(mesh_a, mesh_b);
    auto v6_result = test::intersect_v6_direct_index_3d(mesh_a, mesh_b);
    auto v7_result = test::intersect_v7_soa_optimized_3d(mesh_a, mesh_b);

    EXPECT_TRUE(common_meshes_equal(baseline_result, optimized_result))
        << "optimized differs from baseline in 3D (iteration=" << iter
        << ", seed=" << seed << ", rows=" << num_rows << ")";

    EXPECT_TRUE(common_meshes_equal(baseline_result, v4_result))
        << "v4 (hash) differs from baseline in 3D (iteration=" << iter
        << ", seed=" << seed << ", rows=" << num_rows << ")";

    EXPECT_TRUE(common_meshes_equal(baseline_result, v5_result))
        << "v5 (parallel_merge) differs from baseline in 3D (iteration=" << iter
        << ", seed=" << seed << ", rows=" << num_rows << ")";

    EXPECT_TRUE(common_meshes_equal(baseline_result, v6_result))
        << "v6 (direct_index) differs from baseline in 3D (iteration=" << iter
        << ", seed=" << seed << ", rows=" << num_rows << ")";

    EXPECT_TRUE(common_meshes_equal(baseline_result, v7_result))
        << "v7 (soa_optimized) differs from baseline in 3D (iteration=" << iter
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
 *
 * Tests baseline, optimized, v4, v5, v6, v7.
 */
TEST_F(AllVersionsComparisonTest, AllVersions_MathProperties_Random) {
  std::mt19937 seed_gen(44);
  std::uniform_int_distribution<int> rows_dist(5, 25);
  std::uniform_int_distribution<int> intervals_dist(1, 6);
  std::uniform_int_distribution<int> seed_dist(1, 5000);

  const int num_iterations = 8;

  for (int iter = 0; iter < num_iterations; ++iter) {
    int num_rows = rows_dist(seed_gen);
    int min_intervals = intervals_dist(seed_gen);
    int max_intervals = std::min(min_intervals + intervals_dist(seed_gen), 10);
    int seed = seed_dist(seed_gen);

    RandomMeshConfig config;
    config.seed = seed;
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

    // Test each version's math properties

    // === baseline ===
    auto baseline_ab = test::intersect_baseline_2d(mesh_a, mesh_b);
    auto baseline_ba = test::intersect_baseline_2d(mesh_b, mesh_a);
    EXPECT_TRUE(common_meshes_equal(baseline_ab, baseline_ba))
        << "baseline is not commutative (iteration=" << iter << ")";

    auto baseline_ab_c = test::intersect_baseline_2d(baseline_ab, mesh_c);
    auto baseline_bc = test::intersect_baseline_2d(mesh_b, mesh_c);
    auto baseline_a_bc = test::intersect_baseline_2d(mesh_a, baseline_bc);
    EXPECT_TRUE(common_meshes_equal(baseline_ab_c, baseline_a_bc))
        << "baseline is not associative (iteration=" << iter << ")";

    auto baseline_aa = test::intersect_baseline_2d(mesh_a, mesh_a);
    EXPECT_TRUE(common_meshes_equal(baseline_aa, mesh_a))
        << "baseline is not idempotent (iteration=" << iter << ")";

    // === optimized ===
    auto optimized_ab = test::intersect_optimized_2d(mesh_a, mesh_b);
    auto optimized_ba = test::intersect_optimized_2d(mesh_b, mesh_a);
    EXPECT_TRUE(common_meshes_equal(optimized_ab, optimized_ba))
        << "optimized is not commutative (iteration=" << iter << ")";

    auto optimized_ab_c = test::intersect_optimized_2d(optimized_ab, mesh_c);
    auto optimized_bc = test::intersect_optimized_2d(mesh_b, mesh_c);
    auto optimized_a_bc = test::intersect_optimized_2d(mesh_a, optimized_bc);
    EXPECT_TRUE(common_meshes_equal(optimized_ab_c, optimized_a_bc))
        << "optimized is not associative (iteration=" << iter << ")";

    auto optimized_aa = test::intersect_optimized_2d(mesh_a, mesh_a);
    EXPECT_TRUE(common_meshes_equal(optimized_aa, mesh_a))
        << "optimized is not idempotent (iteration=" << iter << ")";

    // === v4 (hash-based) ===
    auto v4_ab = test::intersect_v4_hash_2d(mesh_a, mesh_b);
    auto v4_ba = test::intersect_v4_hash_2d(mesh_b, mesh_a);
    EXPECT_TRUE(common_meshes_equal(v4_ab, v4_ba))
        << "v4 (hash) is not commutative (iteration=" << iter << ")";

    auto v4_ab_c = test::intersect_v4_hash_2d(v4_ab, mesh_c);
    auto v4_bc = test::intersect_v4_hash_2d(mesh_b, mesh_c);
    auto v4_a_bc = test::intersect_v4_hash_2d(mesh_a, v4_bc);
    EXPECT_TRUE(common_meshes_equal(v4_ab_c, v4_a_bc))
        << "v4 (hash) is not associative (iteration=" << iter << ")";

    auto v4_aa = test::intersect_v4_hash_2d(mesh_a, mesh_a);
    EXPECT_TRUE(common_meshes_equal(v4_aa, mesh_a))
        << "v4 (hash) is not idempotent (iteration=" << iter << ")";

    // === v5 (parallel merge) ===
    auto v5_ab = test::intersect_v5_parallel_merge_2d(mesh_a, mesh_b);
    auto v5_ba = test::intersect_v5_parallel_merge_2d(mesh_b, mesh_a);
    EXPECT_TRUE(common_meshes_equal(v5_ab, v5_ba))
        << "v5 (parallel_merge) is not commutative (iteration=" << iter << ")";

    auto v5_ab_c = test::intersect_v5_parallel_merge_2d(v5_ab, mesh_c);
    auto v5_bc = test::intersect_v5_parallel_merge_2d(mesh_b, mesh_c);
    auto v5_a_bc = test::intersect_v5_parallel_merge_2d(mesh_a, v5_bc);
    EXPECT_TRUE(common_meshes_equal(v5_ab_c, v5_a_bc))
        << "v5 (parallel_merge) is not associative (iteration=" << iter << ")";

    auto v5_aa = test::intersect_v5_parallel_merge_2d(mesh_a, mesh_a);
    EXPECT_TRUE(common_meshes_equal(v5_aa, mesh_a))
        << "v5 (parallel_merge) is not idempotent (iteration=" << iter << ")";

    // === v6 (direct index) ===
    auto v6_ab = test::intersect_v6_direct_index_2d(mesh_a, mesh_b);
    auto v6_ba = test::intersect_v6_direct_index_2d(mesh_b, mesh_a);
    EXPECT_TRUE(common_meshes_equal(v6_ab, v6_ba))
        << "v6 (direct_index) is not commutative (iteration=" << iter << ")";

    auto v6_ab_c = test::intersect_v6_direct_index_2d(v6_ab, mesh_c);
    auto v6_bc = test::intersect_v6_direct_index_2d(mesh_b, mesh_c);
    auto v6_a_bc = test::intersect_v6_direct_index_2d(mesh_a, v6_bc);
    EXPECT_TRUE(common_meshes_equal(v6_ab_c, v6_a_bc))
        << "v6 (direct_index) is not associative (iteration=" << iter << ")";

    auto v6_aa = test::intersect_v6_direct_index_2d(mesh_a, mesh_a);
    EXPECT_TRUE(common_meshes_equal(v6_aa, mesh_a))
        << "v6 (direct_index) is not idempotent (iteration=" << iter << ")";

    // === v7 (SOA optimized) ===
    auto v7_ab = test::intersect_v7_soa_optimized_2d(mesh_a, mesh_b);
    auto v7_ba = test::intersect_v7_soa_optimized_2d(mesh_b, mesh_a);
    EXPECT_TRUE(common_meshes_equal(v7_ab, v7_ba))
        << "v7 (soa_optimized) is not commutative (iteration=" << iter << ")";

    auto v7_ab_c = test::intersect_v7_soa_optimized_2d(v7_ab, mesh_c);
    auto v7_bc = test::intersect_v7_soa_optimized_2d(mesh_b, mesh_c);
    auto v7_a_bc = test::intersect_v7_soa_optimized_2d(mesh_a, v7_bc);
    EXPECT_TRUE(common_meshes_equal(v7_ab_c, v7_a_bc))
        << "v7 (soa_optimized) is not associative (iteration=" << iter << ")";

    auto v7_aa = test::intersect_v7_soa_optimized_2d(mesh_a, mesh_a);
    EXPECT_TRUE(common_meshes_equal(v7_aa, mesh_a))
        << "v7 (soa_optimized) is not idempotent (iteration=" << iter << ")";
  }
}

#endif // SUBSETIX_ENABLE_PLAYGROUND
