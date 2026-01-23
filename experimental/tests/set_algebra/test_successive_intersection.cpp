// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <gtest/gtest.h>
#include <experimental/subsetix/csr/set_algebra/successive_intersection.hpp>
#include <experimental/subsetix/csr/set_algebra/v3.hpp>
#include "test_common_format.hpp"
#include "test_random_mesh_generator.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

using namespace experimental::subsetix::csr;
using namespace experimental::subsetix::csr::v3;
using namespace experimental::subsetix::csr::test;
using namespace experimental::subsetix::csr::successive;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Convert CommonMesh2D to device mesh (using v3)
 */
inline v3::Mesh2DDevice common_to_device_2d(const DefaultCommonMesh2D& common) {
  return MeshConverter2D<v3::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(common);
}

/**
 * @brief Convert device mesh to CommonMesh2D (using v3)
 */
inline DefaultCommonMesh2D device_to_common_2d(const v3::Mesh2DDevice& device) {
  return MeshConverter2D<v3::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::to_common(device);
}

/**
 * @brief Generate vector of device meshes from common meshes
 */
inline std::vector<v3::Mesh2DDevice> generate_mesh_vector_2d(
    const std::vector<DefaultCommonMesh2D>& common_meshes) {
  std::vector<v3::Mesh2DDevice> device_meshes;
  device_meshes.reserve(common_meshes.size());
  for (const auto& common : common_meshes) {
    device_meshes.push_back(common_to_device_2d(common));
  }
  return device_meshes;
}

// ============================================================================
// Test Fixtures
// ============================================================================

/**
 * @brief Fixture for successive intersection cross-validation tests
 *
 * This fixture provides a framework for comparing all three approaches
 * (naive, workspace, graph) to ensure they produce identical results.
 */
class SuccessiveIntersectionTest : public ::testing::Test {
protected:
  /**
   * @brief Helper to run intersection with all three approaches and compare
   *
   * @param common_meshes Vector of common meshes to intersect
   * @param test_name Name for error reporting
   */
  void validate_all_approaches(
      const std::vector<DefaultCommonMesh2D>& common_meshes,
      const std::string& test_name) {

    if (common_meshes.empty()) {
      return;  // Skip empty input
    }

    // Convert to device meshes
    auto device_meshes = generate_mesh_vector_2d(common_meshes);

    // Create workspace
    workspace::IntersectionWorkspace<2, Kokkos::DefaultExecutionSpace::memory_space> ws;

    // Run all three approaches
    auto result_naive = naive::intersect<2>(device_meshes);
    auto result_ws = workspace::intersect_successive<2>(device_meshes, ws);
    auto result_graph = graph::successive_intersection<2>(device_meshes);

    // Convert to common format for comparison
    auto common_naive = device_to_common_2d(result_naive);
    auto common_ws = device_to_common_2d(result_ws);
    auto common_graph = device_to_common_2d(result_graph);

    // Verify all approaches produce identical results
    EXPECT_TRUE(common_meshes_equal(common_naive, common_ws))
        << test_name << ": Naive and Workspace produced different results\n"
        << "  Naive: " << common_naive.num_rows() << " rows, " << common_naive.num_intervals() << " intervals\n"
        << "  Workspace: " << common_ws.num_rows() << " rows, " << common_ws.num_intervals() << " intervals";

    EXPECT_TRUE(common_meshes_equal(common_naive, common_graph))
        << test_name << ": Naive and Graph produced different results\n"
        << "  Naive: " << common_naive.num_rows() << " rows, " << common_naive.num_intervals() << " intervals\n"
        << "  Graph: " << common_graph.num_rows() << " rows, " << common_graph.num_intervals() << " intervals";

    // Verify bitwise equality for critical fields
    EXPECT_EQ(common_naive.num_rows(), common_ws.num_rows())
        << test_name << ": Row count mismatch";
    EXPECT_EQ(common_naive.num_rows(), common_graph.num_rows())
        << test_name << ": Row count mismatch";

    EXPECT_EQ(common_naive.num_intervals(), common_ws.num_intervals())
        << test_name << ": Interval count mismatch";
    EXPECT_EQ(common_naive.num_intervals(), common_graph.num_intervals())
        << test_name << ": Interval count mismatch";

    // Validate that the result is a valid mesh
    EXPECT_TRUE(validate_common_mesh_2d(common_naive))
        << test_name << ": Naive produced invalid mesh";
    EXPECT_TRUE(validate_common_mesh_2d(common_ws))
        << test_name << ": Workspace produced invalid mesh";
    EXPECT_TRUE(validate_common_mesh_2d(common_graph))
        << test_name << ": Graph produced invalid mesh";
  }

  /**
   * @brief Validate that successive intersection produces the same result
   *        as sequential binary intersections (mathematical correctness)
   */
  void validate_against_binary_intersection(
      const std::vector<DefaultCommonMesh2D>& common_meshes,
      const std::string& test_name) {

    if (common_meshes.size() < 2) {
      return;
    }

    // Compute result using successive intersection (naive as reference)
    auto device_meshes = generate_mesh_vector_2d(common_meshes);
    auto result_successive = naive::intersect<2>(device_meshes);
    auto common_successive = device_to_common_2d(result_successive);

    // Compute result using sequential binary intersections (mathematical definition)
    auto device_a = common_to_device_2d(common_meshes[0]);
    for (size_t i = 1; i < common_meshes.size(); ++i) {
      auto device_b = common_to_device_2d(common_meshes[i]);
      auto result = v3::intersect_meshes<2, int32_t, std::size_t>(device_a, device_b);
      device_a = result;  // Chain: result = (...((A ∩ B) ∩ C) ∩ D)
    }
    auto common_binary = device_to_common_2d(device_a);

    // They must be identical
    EXPECT_TRUE(common_meshes_equal(common_successive, common_binary))
        << test_name << ": Successive intersection does not match binary intersection chain\n"
        << "  Successive: " << common_successive.num_rows() << " rows, " << common_successive.num_intervals() << " intervals\n"
        << "  Binary: " << common_binary.num_rows() << " rows, " << common_binary.num_intervals() << " intervals";
  }
};

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST_F(SuccessiveIntersectionTest, EmptyVector_ReturnsEmpty) {
  std::vector<v3::Mesh2DDevice> empty_meshes;

  auto result_naive = naive::intersect<2>(empty_meshes);
  EXPECT_EQ(result_naive.num_rows, 0u);
  EXPECT_EQ(result_naive.num_intervals, 0u);
}

TEST_F(SuccessiveIntersectionTest, SingleMesh_ReturnsInput) {
  auto config = SmallConfig();
  auto common_mesh = RandomMeshGenerator::generate_2d(config);
  auto device_mesh = common_to_device_2d(common_mesh);

  std::vector<v3::Mesh2DDevice> meshes = {device_mesh};

  auto result = naive::intersect<2>(meshes);
  auto common_result = device_to_common_2d(result);

  EXPECT_TRUE(common_meshes_equal(common_mesh, common_result))
      << "Single mesh intersection should return the input unchanged";
}

TEST_F(SuccessiveIntersectionTest, AllEmptyMeshes_ReturnsEmpty) {
  // Create empty meshes
  DefaultCommonMesh2D empty_mesh;
  std::vector<DefaultCommonMesh2D> common_meshes = {empty_mesh, empty_mesh, empty_mesh};

  validate_all_approaches(common_meshes, "AllEmptyMeshes");
}

TEST_F(SuccessiveIntersectionTest, DisjointMeshes_ReturnsEmpty) {
  // Create two meshes that don't overlap
  auto config_a = SmallConfig();
  config_a.y_max = 100;
  config_a.seed = 1;

  auto config_b = SmallConfig();
  config_b.y_min = 200;  // No overlap with config_a
  config_b.y_max = 300;
  config_b.seed = 2;

  std::vector<DefaultCommonMesh2D> common_meshes = {
    RandomMeshGenerator::generate_2d(config_a),
    RandomMeshGenerator::generate_2d(config_b)
  };

  validate_all_approaches(common_meshes, "DisjointMeshes");

  // Result should be empty
  auto device_meshes = generate_mesh_vector_2d(common_meshes);
  auto result = naive::intersect<2>(device_meshes);
  EXPECT_EQ(result.num_rows, 0u);
  EXPECT_EQ(result.num_intervals, 0u);
}

TEST_F(SuccessiveIntersectionTest, FirstMeshEmpty_ReturnsEmpty) {
  DefaultCommonMesh2D empty_mesh;

  auto config = SmallConfig();
  auto non_empty = RandomMeshGenerator::generate_2d(config);

  std::vector<DefaultCommonMesh2D> common_meshes = {empty_mesh, non_empty};

  validate_all_approaches(common_meshes, "FirstMeshEmpty");

  auto device_meshes = generate_mesh_vector_2d(common_meshes);
  auto result = naive::intersect<2>(device_meshes);
  EXPECT_EQ(result.num_rows, 0u);
}

TEST_F(SuccessiveIntersectionTest, MiddleMeshEmpty_ReturnsEmpty) {
  auto config_a = SmallConfig();
  config_a.seed = 1;
  auto mesh_a = RandomMeshGenerator::generate_2d(config_a);

  DefaultCommonMesh2D empty_mesh;

  auto config_b = SmallConfig();
  config_b.seed = 2;
  auto mesh_b = RandomMeshGenerator::generate_2d(config_b);

  std::vector<DefaultCommonMesh2D> common_meshes = {mesh_a, empty_mesh, mesh_b};

  validate_all_approaches(common_meshes, "MiddleMeshEmpty");

  auto device_meshes = generate_mesh_vector_2d(common_meshes);
  auto result = naive::intersect<2>(device_meshes);
  EXPECT_EQ(result.num_rows, 0u);
}

// ============================================================================
// Two Meshes Tests (Baseline)
// ============================================================================

TEST_F(SuccessiveIntersectionTest, TwoMeshes_SmallCrossValidation) {
  auto config = SmallConfig();

  auto mesh_a = RandomMeshGenerator::generate_2d(config);
  config.seed++;
  auto mesh_b = RandomMeshGenerator::generate_2d(config);

  std::vector<DefaultCommonMesh2D> common_meshes = {mesh_a, mesh_b};

  validate_all_approaches(common_meshes, "TwoMeshes_Small");
  validate_against_binary_intersection(common_meshes, "TwoMeshes_Small");
}

TEST_F(SuccessiveIntersectionTest, TwoMeshes_MediumCrossValidation) {
  auto config = MediumConfig();

  auto mesh_a = RandomMeshGenerator::generate_2d(config);
  config.seed++;
  auto mesh_b = RandomMeshGenerator::generate_2d(config);

  std::vector<DefaultCommonMesh2D> common_meshes = {mesh_a, mesh_b};

  validate_all_approaches(common_meshes, "TwoMeshes_Medium");
  validate_against_binary_intersection(common_meshes, "TwoMeshes_Medium");
}

// ============================================================================
// Four Meshes Tests (Typical AMR)
// ============================================================================

TEST_F(SuccessiveIntersectionTest, FourMeshes_SmallCrossValidation) {
  auto config = SmallConfig();

  std::vector<DefaultCommonMesh2D> common_meshes;
  for (int i = 0; i < 4; ++i) {
    common_meshes.push_back(RandomMeshGenerator::generate_2d(config));
    config.seed++;
  }

  validate_all_approaches(common_meshes, "FourMeshes_Small");
  validate_against_binary_intersection(common_meshes, "FourMeshes_Small");
}

TEST_F(SuccessiveIntersectionTest, FourMeshes_MediumCrossValidation) {
  auto config = MediumConfig();

  std::vector<DefaultCommonMesh2D> common_meshes;
  for (int i = 0; i < 4; ++i) {
    common_meshes.push_back(RandomMeshGenerator::generate_2d(config));
    config.seed++;
  }

  validate_all_approaches(common_meshes, "FourMeshes_Medium");
  validate_against_binary_intersection(common_meshes, "FourMeshes_Medium");
}

// ============================================================================
// Eight Meshes Tests (Deep Refinement)
// ============================================================================

TEST_F(SuccessiveIntersectionTest, EightMeshes_SmallCrossValidation) {
  auto config = SmallConfig();

  std::vector<DefaultCommonMesh2D> common_meshes;
  for (int i = 0; i < 8; ++i) {
    common_meshes.push_back(RandomMeshGenerator::generate_2d(config));
    config.seed++;
  }

  validate_all_approaches(common_meshes, "EightMeshes_Small");
  validate_against_binary_intersection(common_meshes, "EightMeshes_Small");
}

TEST_F(SuccessiveIntersectionTest, EightMeshes_MediumCrossValidation) {
  auto config = MediumConfig();

  std::vector<DefaultCommonMesh2D> common_meshes;
  for (int i = 0; i < 8; ++i) {
    common_meshes.push_back(RandomMeshGenerator::generate_2d(config));
    config.seed++;
  }

  validate_all_approaches(common_meshes, "EightMeshes_Medium");
  validate_against_binary_intersection(common_meshes, "EightMeshes_Medium");
}

// ============================================================================
// Idempotence Tests
// ============================================================================

TEST_F(SuccessiveIntersectionTest, Idempotence_IdenticalMeshes) {
  auto config = SmallConfig();
  auto mesh = RandomMeshGenerator::generate_2d(config);

  // Create vector with 4 identical meshes
  std::vector<DefaultCommonMesh2D> common_meshes = {mesh, mesh, mesh, mesh};

  validate_all_approaches(common_meshes, "Idempotence_Identical");

  // Result should equal the input mesh
  auto device_meshes = generate_mesh_vector_2d(common_meshes);
  auto result = naive::intersect<2>(device_meshes);
  auto common_result = device_to_common_2d(result);

  EXPECT_TRUE(common_meshes_equal(mesh, common_result))
      << "Intersection of identical meshes should equal the input mesh";
}

// ============================================================================
// Random Mesh Cross-Validation Tests
// ============================================================================

TEST_F(SuccessiveIntersectionTest, RandomMeshes_VaryingSizes) {
  std::mt19937 seed_gen(42);
  std::uniform_int_distribution<int> count_dist(2, 6);
  std::uniform_int_distribution<int> rows_dist(10, 100);
  std::uniform_int_distribution<int> seed_dist(1, 10000);

  const int num_iterations = 10;

  for (int iter = 0; iter < num_iterations; ++iter) {
    int num_meshes = count_dist(seed_gen);
    int num_rows = rows_dist(seed_gen);
    int seed = seed_dist(seed_gen);

    // Build config
    RandomMeshConfig config = SmallConfig();
    config.seed = seed;
    config.sparsity = static_cast<double>(num_rows) / (config.y_max - config.y_min);

    // Generate meshes
    std::vector<DefaultCommonMesh2D> common_meshes;
    for (int i = 0; i < num_meshes; ++i) {
      common_meshes.push_back(RandomMeshGenerator::generate_2d(config));
      config.seed++;
    }

    validate_all_approaches(common_meshes,
        "RandomMeshes_VaryingSizes_iter" + std::to_string(iter));
    validate_against_binary_intersection(common_meshes,
        "RandomMeshes_VaryingSizes_iter" + std::to_string(iter));
  }
}

// ============================================================================
// Associativity Tests
// ============================================================================

TEST_F(SuccessiveIntersectionTest, Associativity_DifferentOrders) {
  auto config = SmallConfig();

  auto mesh_a = RandomMeshGenerator::generate_2d(config);
  config.seed++;
  auto mesh_b = RandomMeshGenerator::generate_2d(config);
  config.seed++;
  auto mesh_c = RandomMeshGenerator::generate_2d(config);
  config.seed++;
  auto mesh_d = RandomMeshGenerator::generate_2d(config);

  // Test (A ∩ B) ∩ (C ∩ D) vs A ∩ (B ∩ C) ∩ D
  // Both should give the same result due to associativity

  std::vector<DefaultCommonMesh2D> order1 = {mesh_a, mesh_b, mesh_c, mesh_d};
  std::vector<DefaultCommonMesh2D> order2 = {mesh_c, mesh_d, mesh_a, mesh_b};
  std::vector<DefaultCommonMesh2D> order3 = {mesh_b, mesh_c, mesh_d, mesh_a};

  auto devices1 = generate_mesh_vector_2d(order1);
  auto devices2 = generate_mesh_vector_2d(order2);
  auto devices3 = generate_mesh_vector_2d(order3);

  auto result1 = naive::intersect<2>(devices1);
  auto result2 = naive::intersect<2>(devices2);
  auto result3 = naive::intersect<2>(devices3);

  auto common1 = device_to_common_2d(result1);
  auto common2 = device_to_common_2d(result2);
  auto common3 = device_to_common_2d(result3);

  EXPECT_TRUE(common_meshes_equal(common1, common2))
      << "Different mesh orders should produce the same result (order1 vs order2)";
  EXPECT_TRUE(common_meshes_equal(common1, common3))
      << "Different mesh orders should produce the same result (order1 vs order3)";
}

// ============================================================================
// Unified API Tests
// ============================================================================

TEST_F(SuccessiveIntersectionTest, UnifiedAPI_StrategyDispatch) {
  auto config = SmallConfig();

  std::vector<DefaultCommonMesh2D> common_meshes;
  for (int i = 0; i < 4; ++i) {
    common_meshes.push_back(RandomMeshGenerator::generate_2d(config));
    config.seed++;
  }

  auto device_meshes = generate_mesh_vector_2d(common_meshes);

  // Test Naive strategy
  Config<2> config_naive;
  config_naive.strategy = Strategy::Naive;
  auto result_naive_api = intersect_successive(device_meshes, config_naive);
  auto common_naive_api = device_to_common_2d(result_naive_api);

  // Test Workspace strategy
  Config<2> config_ws;
  config_ws.strategy = Strategy::Workspace;
  config_ws.workspace.max_rows = 10000;
  config_ws.workspace.max_intervals = 100000;
  auto result_ws_api = intersect_successive(device_meshes, config_ws);
  auto common_ws_api = device_to_common_2d(result_ws_api);

  // Test Graph strategy
  Config<2> config_graph;
  config_graph.strategy = Strategy::Graph;
  auto result_graph_api = intersect_successive(device_meshes, config_graph);
  auto common_graph_api = device_to_common_2d(result_graph_api);

  // All should be equal
  auto result_naive_direct = naive::intersect<2>(device_meshes);
  auto common_naive_direct = device_to_common_2d(result_naive_direct);

  EXPECT_TRUE(common_meshes_equal(common_naive_direct, common_naive_api))
      << "Unified API (Naive) should match direct call";
  EXPECT_TRUE(common_meshes_equal(common_naive_direct, common_ws_api))
      << "Unified API (Workspace) should match naive result";
  EXPECT_TRUE(common_meshes_equal(common_naive_direct, common_graph_api))
      << "Unified API (Graph) should match naive result";
}

TEST_F(SuccessiveIntersectionTest, UnifiedAPI_DefaultConfig) {
  auto config = SmallConfig();

  std::vector<DefaultCommonMesh2D> common_meshes;
  for (int i = 0; i < 3; ++i) {
    common_meshes.push_back(RandomMeshGenerator::generate_2d(config));
    config.seed++;
  }

  auto device_meshes = generate_mesh_vector_2d(common_meshes);

  // Test with default config (should use Naive strategy)
  auto result_default = intersect_successive(device_meshes);
  auto result_naive = naive::intersect<2>(device_meshes);

  auto common_default = device_to_common_2d(result_default);
  auto common_naive = device_to_common_2d(result_naive);

  EXPECT_TRUE(common_meshes_equal(common_default, common_naive))
      << "Default config should use Naive strategy";
}

// ============================================================================
// Workspace Growth Tests
// ============================================================================

TEST_F(SuccessiveIntersectionTest, Workspace_AutoGrowth) {
  auto config = MediumConfig();

  std::vector<DefaultCommonMesh2D> common_meshes;
  for (int i = 0; i < 4; ++i) {
    common_meshes.push_back(RandomMeshGenerator::generate_2d(config));
    config.seed++;
  }

  auto device_meshes = generate_mesh_vector_2d(common_meshes);

  // Start with small workspace (should trigger growth)
  workspace::IntersectionWorkspace<2, Kokkos::DefaultExecutionSpace::memory_space> ws;
  ws.max_rows = 10;  // Too small
  ws.max_intervals = 100;  // Too small
  ws.growth_factor = 2.0;

  auto result_ws = workspace::intersect_successive<2>(device_meshes, ws);
  auto common_ws = device_to_common_2d(result_ws);

  // Compare with naive to ensure correctness
  auto result_naive = naive::intersect<2>(device_meshes);
  auto common_naive = device_to_common_2d(result_naive);

  EXPECT_TRUE(common_meshes_equal(common_naive, common_ws))
      << "Workspace with auto-growth should match naive result";

  // Verify workspace actually grew
  EXPECT_GT(ws.max_rows, 10u)
      << "Workspace should have grown beyond initial capacity";
}

// ============================================================================
// Large Mesh Tests (Stress Test)
// ============================================================================

TEST_F(SuccessiveIntersectionTest, DISABLED_LargeMesh_FourMeshesCrossValidation) {
  // This test is disabled by default as it may take time
  // Enable for stress testing or CI

  auto config = LargeConfig();

  std::vector<DefaultCommonMesh2D> common_meshes;
  for (int i = 0; i < 4; ++i) {
    common_meshes.push_back(RandomMeshGenerator::generate_2d(config));
    config.seed++;
  }

  validate_all_approaches(common_meshes, "LargeMesh_FourMeshes");
}

#endif // SUBSETIX_ENABLE_EXPERIMENTAL
