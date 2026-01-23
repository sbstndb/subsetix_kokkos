// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <gtest/gtest.h>
#include <set>
#include "test_random_mesh_generator.hpp"

using namespace experimental::subsetix::csr::test;

// ============================================================================
// 2D Uniqueness Tests
// ============================================================================

/**
 * @brief Test that 2D generated meshes have unique Y coordinates
 */
TEST(RandomMeshGeneratorTest, Generated2D_HasUniqueYCoords) {
  auto config = SmallConfig();
  config.sparsity = 0.5;

  auto mesh = RandomMeshGenerator::generate_2d(config);

  std::set<int> y_coords;
  for (const auto& row : mesh.rows) {
    EXPECT_FALSE(y_coords.contains(row.y))
        << "Duplicate Y coordinate found: " << row.y;
    y_coords.insert(row.y);
  }

  // All rows should be unique
  EXPECT_EQ(y_coords.size(), mesh.rows.size());
}

/**
 * @brief Test that 2D generated meshes are sorted when requested
 */
TEST(RandomMeshGeneratorTest, Generated2D_SortedWhenRequested) {
  auto config = SmallConfig();
  config.sparsity = 0.5;
  config.sorted_rows = true;

  auto mesh = RandomMeshGenerator::generate_2d(config);

  for (size_t i = 1; i < mesh.rows.size(); ++i) {
    EXPECT_LT(mesh.rows[i-1].y, mesh.rows[i].y)
        << "Rows not sorted at index " << i;
  }
}

// ============================================================================
// 3D Uniqueness Tests
// ============================================================================

/**
 * @brief Test that 3D generated meshes have unique (Y, Z) coordinate pairs
 */
TEST(RandomMeshGeneratorTest, Generated3D_HasUniqueYZPairs) {
  auto config = SmallConfig();
  config.sparsity = 0.3;

  auto mesh = RandomMeshGenerator::generate_3d(config);

  std::set<std::pair<int, int>> yz_pairs;
  for (const auto& row : mesh.rows) {
    auto pair = std::make_pair(row.y, row.z);
    EXPECT_FALSE(yz_pairs.contains(pair))
        << "Duplicate (Y, Z) pair found: (" << row.y << ", " << row.z << ")";
    yz_pairs.insert(pair);
  }

  // All rows should be unique
  EXPECT_EQ(yz_pairs.size(), mesh.rows.size());
}

/**
 * @brief Test that 3D generated meshes are sorted lexicographically when requested
 */
TEST(RandomMeshGeneratorTest, Generated3D_SortedWhenRequested) {
  auto config = SmallConfig();
  config.sparsity = 0.3;
  config.sorted_rows = true;

  auto mesh = RandomMeshGenerator::generate_3d(config);

  for (size_t i = 1; i < mesh.rows.size(); ++i) {
    const auto& prev = mesh.rows[i-1];
    const auto& curr = mesh.rows[i];
    EXPECT_TRUE(prev < curr)
        << "Rows not sorted at index " << i
        << ": (" << prev.y << ", " << prev.z << ") >= (" << curr.y << ", " << curr.z << ")";
  }
}

// ============================================================================
// Sparsity Accuracy Tests (2D)
// ============================================================================

/**
 * @brief Test that sparsity produces correct row count (2D)
 */
TEST(RandomMeshGeneratorTest, SparsityAccuracy_2D) {
  auto config = SmallConfig();
  config.y_max = 100;
  config.sparsity = 0.3;

  auto mesh = RandomMeshGenerator::generate_2d(config);

  int expected = std::max(1, static_cast<int>(std::round(0.3 * 100)));
  EXPECT_EQ(mesh.rows.size(), static_cast<std::size_t>(expected))
      << "Expected " << expected << " rows for sparsity=0.3, y_max=100";
}

/**
 * @brief Test that sparsity=0 produces empty mesh (2D)
 */
TEST(RandomMeshGeneratorTest, SparsityZero_EmptyMesh_2D) {
  auto config = SmallConfig();
  config.sparsity = 0.0;

  auto mesh = RandomMeshGenerator::generate_2d(config);

  EXPECT_EQ(mesh.rows.size(), 0u) << "Sparsity=0 should produce empty mesh";
}

/**
 * @brief Test that sparsity=1 produces full mesh (2D)
 */
TEST(RandomMeshGeneratorTest, SparsityOne_FullMesh_2D) {
  auto config = SmallConfig();
  config.y_max = 64;
  config.sparsity = 1.0;

  auto mesh = RandomMeshGenerator::generate_2d(config);

  EXPECT_EQ(mesh.rows.size(), 64u) << "Sparsity=1.0 should produce 64 rows for y_max=64";

  // Verify all Y coordinates are present
  std::set<int> y_coords;
  for (const auto& row : mesh.rows) {
    y_coords.insert(row.y);
  }
  EXPECT_EQ(y_coords.size(), 64u);
}

/**
 * @brief Test various sparsity values (2D)
 */
TEST(RandomMeshGeneratorTest, SparsityVariations_2D) {
  struct TestCase {
    double sparsity;
    int y_extent;
    int expected_rows;
  };

  std::vector<TestCase> test_cases = {
    {0.0, 100, 0},
    {0.1, 100, 10},
    {0.25, 100, 25},
    {0.5, 100, 50},
    {0.75, 100, 75},
    {1.0, 100, 100}
  };

  for (const auto& tc : test_cases) {
    auto config = SmallConfig();
    config.y_max = tc.y_extent;
    config.sparsity = tc.sparsity;

    auto mesh = RandomMeshGenerator::generate_2d(config);

    EXPECT_EQ(mesh.rows.size(), static_cast<std::size_t>(tc.expected_rows))
        << "Failed for sparsity=" << tc.sparsity << ", y_extent=" << tc.y_extent;
  }
}

// ============================================================================
// Sparsity Accuracy Tests (3D)
// ============================================================================

/**
 * @brief Test that sparsity produces correct row count (3D)
 */
TEST(RandomMeshGeneratorTest, SparsityAccuracy_3D) {
  auto config = SmallConfig();
  config.y_max = 10;
  config.z_max = 10;
  config.sparsity = 0.3;

  auto mesh = RandomMeshGenerator::generate_3d(config);

  int expected = std::max(1, static_cast<int>(std::round(0.3 * 10 * 10)));
  EXPECT_EQ(mesh.rows.size(), static_cast<std::size_t>(expected))
      << "Expected " << expected << " rows for sparsity=0.3, y_max=10, z_max=10";
}

/**
 * @brief Test that sparsity=0 produces empty mesh (3D)
 */
TEST(RandomMeshGeneratorTest, SparsityZero_EmptyMesh_3D) {
  auto config = SmallConfig();
  config.sparsity = 0.0;

  auto mesh = RandomMeshGenerator::generate_3d(config);

  EXPECT_EQ(mesh.rows.size(), 0u) << "Sparsity=0 should produce empty mesh";
}

/**
 * @brief Test that sparsity=1 produces full mesh (3D)
 */
TEST(RandomMeshGeneratorTest, SparsityOne_FullMesh_3D) {
  auto config = SmallConfig();
  config.y_max = 8;
  config.z_max = 8;
  config.sparsity = 1.0;

  auto mesh = RandomMeshGenerator::generate_3d(config);

  EXPECT_EQ(mesh.rows.size(), 64u) << "Sparsity=1.0 should produce 64 rows for y_max=8, z_max=8";

  // Verify all (Y, Z) pairs are present
  std::set<std::pair<int, int>> yz_pairs;
  for (const auto& row : mesh.rows) {
    yz_pairs.emplace(row.y, row.z);
  }
  EXPECT_EQ(yz_pairs.size(), 64u);
}

// ============================================================================
// Predefined Configuration Tests
// ============================================================================

/**
 * @brief Test that SmallConfig produces expected row counts
 */
TEST(RandomMeshGeneratorTest, SmallConfig_RowCounts) {
  auto config = SmallConfig();

  // 2D
  auto mesh_2d = RandomMeshGenerator::generate_2d(config);
  int expected_2d = std::max(1, static_cast<int>(std::round(0.3 * 64)));
  EXPECT_EQ(mesh_2d.rows.size(), static_cast<std::size_t>(expected_2d))
      << "SmallConfig 2D: expected ~" << expected_2d << " rows";

  // 3D
  auto mesh_3d = RandomMeshGenerator::generate_3d(config);
  int expected_3d = std::max(1, static_cast<int>(std::round(0.3 * 64 * 64)));
  EXPECT_EQ(mesh_3d.rows.size(), static_cast<std::size_t>(expected_3d))
      << "SmallConfig 3D: expected ~" << expected_3d << " rows";
}

/**
 * @brief Test that MediumConfig produces expected row counts
 */
TEST(RandomMeshGeneratorTest, MediumConfig_RowCounts) {
  auto config = MediumConfig();

  // 2D
  auto mesh_2d = RandomMeshGenerator::generate_2d(config);
  int expected_2d = std::max(1, static_cast<int>(std::round(0.3 * 512)));
  EXPECT_EQ(mesh_2d.rows.size(), static_cast<std::size_t>(expected_2d))
      << "MediumConfig 2D: expected ~" << expected_2d << " rows";

  // 3D
  auto mesh_3d = RandomMeshGenerator::generate_3d(config);
  int expected_3d = std::max(1, static_cast<int>(std::round(0.3 * 512 * 512)));
  EXPECT_EQ(mesh_3d.rows.size(), static_cast<std::size_t>(expected_3d))
      << "MediumConfig 3D: expected ~" << expected_3d << " rows";
}

/**
 * @brief Test that LargeConfig produces expected row counts
 */
TEST(RandomMeshGeneratorTest, LargeConfig_RowCounts) {
  auto config = LargeConfig();

  // 2D
  auto mesh_2d = RandomMeshGenerator::generate_2d(config);
  int expected_2d = std::max(1, static_cast<int>(std::round(0.3 * 4096)));
  EXPECT_EQ(mesh_2d.rows.size(), static_cast<std::size_t>(expected_2d))
      << "LargeConfig 2D: expected ~" << expected_2d << " rows";

  // 3D: Too large to test directly, just verify it runs without error
  auto mesh_3d = RandomMeshGenerator::generate_3d(config);
  EXPECT_GT(mesh_3d.rows.size(), 0u) << "LargeConfig 3D should generate rows";
}

// ============================================================================
// Determinism Tests
// ============================================================================

/**
 * @brief Test that same seed produces same mesh (2D)
 */
TEST(RandomMeshGeneratorTest, Determinism_SameSeed_2D) {
  auto config = SmallConfig();
  config.seed = 12345;

  auto mesh1 = RandomMeshGenerator::generate_2d(config);
  auto mesh2 = RandomMeshGenerator::generate_2d(config);

  EXPECT_EQ(mesh1.rows.size(), mesh2.rows.size());
  if (mesh1.rows.size() == mesh2.rows.size()) {
    for (size_t i = 0; i < mesh1.rows.size(); ++i) {
      EXPECT_EQ(mesh1.rows[i].y, mesh2.rows[i].y);
    }
  }
}

/**
 * @brief Test that same seed produces same mesh (3D)
 */
TEST(RandomMeshGeneratorTest, Determinism_SameSeed_3D) {
  auto config = SmallConfig();
  config.seed = 12345;

  auto mesh1 = RandomMeshGenerator::generate_3d(config);
  auto mesh2 = RandomMeshGenerator::generate_3d(config);

  EXPECT_EQ(mesh1.rows.size(), mesh2.rows.size());
  if (mesh1.rows.size() == mesh2.rows.size()) {
    for (size_t i = 0; i < mesh1.rows.size(); ++i) {
      EXPECT_EQ(mesh1.rows[i].y, mesh2.rows[i].y);
      EXPECT_EQ(mesh1.rows[i].z, mesh2.rows[i].z);
    }
  }
}

/**
 * @brief Test that different seeds produce different meshes (2D)
 */
TEST(RandomMeshGeneratorTest, Determinism_DifferentSeeds_2D) {
  auto config = SmallConfig();

  config.seed = 12345;
  auto mesh1 = RandomMeshGenerator::generate_2d(config);

  config.seed = 54321;
  auto mesh2 = RandomMeshGenerator::generate_2d(config);

  // Different seeds should generally produce different meshes
  // (This is probabilistic but very likely to be true for non-trivial meshes)
  bool all_same = (mesh1.rows.size() == mesh2.rows.size());
  if (all_same && mesh1.rows.size() > 10) {
    for (size_t i = 0; i < mesh1.rows.size(); ++i) {
      if (mesh1.rows[i].y != mesh2.rows[i].y) {
        all_same = false;
        break;
      }
    }
  }
  EXPECT_FALSE(all_same) << "Different seeds should produce different meshes";
}

#endif // SUBSETIX_ENABLE_EXPERIMENTAL
