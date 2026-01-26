// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#ifdef SUBSETIX_ENABLE_PLAYGROUND

#include <gtest/gtest.h>
#include <playground/subsetix/csr/intersection/algorithm/baseline.hpp>
#include <playground/subsetix/csr/intersection/workspace.hpp>
#include "test_common_format.hpp"
#include "test_random_mesh_generator.hpp"
#include <Kokkos_Core.hpp>

using namespace playground::subsetix::csr::intersection;
using namespace playground::subsetix::csr::intersection::baseline;
using namespace playground::subsetix::csr::intersection::test;

// ============================================================================
// Oracle Tests: baseline (original) vs in_place (workspace)
// ============================================================================

/**
 * @brief Test that in_place (workspace) produces identical results to baseline
 *
 * This validates that the workspace-optimized version produces bitwise
 * identical results to the original baseline implementation.
 */
class WorkspaceValidationTest : public ::testing::Test {
protected:
  // Helper to convert device mesh to common format for comparison
  template <typename DeviceMesh>
  auto to_common_for_test(const DeviceMesh& device_mesh) {
    return MeshConverter2D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::to_common(device_mesh);
  }

  template <typename DeviceMesh>
  auto to_common_for_test_3d(const DeviceMesh& device_mesh) {
    return MeshConverter3D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::to_common(device_mesh);
  }

  // Helper to run in_place intersection (workspace version)
  DefaultCommonMesh2D intersect_in_place_2d(const DefaultCommonMesh2D& a, const DefaultCommonMesh2D& b) {
    auto device_a = MeshConverter2D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(a);
    auto device_b = MeshConverter2D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(b);

    std::size_t max_rows = std::max(device_a.num_rows, device_b.num_rows);
    std::size_t max_intervals = device_a.num_intervals + device_b.num_intervals;  // Worst case: all intervals are disjoint

    IntersectionWorkspace2D<Kokkos::DefaultExecutionSpace> ws;
    ws.ensure_capacity(max_rows, max_intervals);

    baseline::Mesh2DDevice result_device;
    result_device.row_keys = Kokkos::View<RowKey2D<int32_t>*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_keys", max_rows);
    result_device.row_ptr = Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_ptr", max_rows + 1);
    result_device.intervals = Kokkos::View<playground::subsetix::csr::intersection::Interval<int32_t>*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_intervals", max_intervals);

    baseline::intersect_meshes_2d_in_place(device_a, device_b, result_device, ws);

    return to_common_for_test(result_device);
  }

  DefaultCommonMesh3D intersect_in_place_3d(const DefaultCommonMesh3D& a, const DefaultCommonMesh3D& b) {
    auto device_a = MeshConverter3D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(a);
    auto device_b = MeshConverter3D<baseline::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(b);

    std::size_t max_rows = std::max(device_a.num_rows, device_b.num_rows);
    std::size_t max_intervals = device_a.num_intervals + device_b.num_intervals;  // Worst case: all intervals are disjoint

    IntersectionWorkspace3D<Kokkos::DefaultExecutionSpace> ws;
    ws.ensure_capacity(max_rows, max_intervals);

    baseline::Mesh3DDevice result_device;
    result_device.row_keys = Kokkos::View<RowKey3D<int32_t>*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_keys", max_rows);
    result_device.row_ptr = Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_ptr", max_rows + 1);
    result_device.intervals = Kokkos::View<playground::subsetix::csr::intersection::Interval<int32_t>*, Kokkos::DefaultExecutionSpace::memory_space>(
        "result_intervals", max_intervals);

    baseline::intersect_meshes_3d_in_place(device_a, device_b, result_device, ws);

    return to_common_for_test_3d(result_device);
  }
};

// ============================================================================
// Oracle Tests: Regular Mesh Configs
// ============================================================================

TEST_F(WorkspaceValidationTest, InPlaceMatchesBaseline_2D_SmallRegular) {
  auto cfg = SmallRegularConfig();
  auto mesh_a = RegularMeshGenerator::generate_2d(cfg);
  auto mesh_b = RegularMeshGenerator::generate_2d(cfg);

  auto result_baseline = test::intersect_baseline_2d(mesh_a, mesh_b);
  auto result_in_place = intersect_in_place_2d(mesh_a, mesh_b);

  EXPECT_TRUE(common_meshes_equal(result_baseline, result_in_place))
      << "Workspace version produced different result than baseline for SmallRegular 2D";
}

TEST_F(WorkspaceValidationTest, InPlaceMatchesBaseline_2D_MediumRegular) {
  auto cfg = MediumRegularConfig();
  auto mesh_a = RegularMeshGenerator::generate_2d(cfg);
  auto mesh_b = RegularMeshGenerator::generate_2d(cfg);

  auto result_baseline = test::intersect_baseline_2d(mesh_a, mesh_b);
  auto result_in_place = intersect_in_place_2d(mesh_a, mesh_b);

  EXPECT_TRUE(common_meshes_equal(result_baseline, result_in_place))
      << "Workspace version produced different result than baseline for MediumRegular 2D";
}

TEST_F(WorkspaceValidationTest, InPlaceMatchesBaseline_3D_SmallRegular) {
  auto cfg = SmallRegularConfig();
  auto mesh_a = RegularMeshGenerator::generate_3d(cfg);
  auto mesh_b = RegularMeshGenerator::generate_3d(cfg);

  auto result_baseline = test::intersect_baseline_3d(mesh_a, mesh_b);
  auto result_in_place = intersect_in_place_3d(mesh_a, mesh_b);

  EXPECT_TRUE(common_meshes_equal(result_baseline, result_in_place))
      << "Workspace version produced different result than baseline for SmallRegular 3D";
}

TEST_F(WorkspaceValidationTest, InPlaceMatchesBaseline_3D_MediumRegular) {
  auto cfg = MediumRegularConfig();
  auto mesh_a = RegularMeshGenerator::generate_3d(cfg);
  auto mesh_b = RegularMeshGenerator::generate_3d(cfg);

  auto result_baseline = test::intersect_baseline_3d(mesh_a, mesh_b);
  auto result_in_place = intersect_in_place_3d(mesh_a, mesh_b);

  EXPECT_TRUE(common_meshes_equal(result_baseline, result_in_place))
      << "Workspace version produced different result than baseline for MediumRegular 3D";
}

// ============================================================================
// Randomized Oracle Tests
// ============================================================================

TEST_F(WorkspaceValidationTest, InPlaceMatchesBaseline_2D_RandomBounds) {
  std::mt19937 seed_gen(42);

  std::uniform_int_distribution<int> rows_dist(1, 512);
  std::uniform_int_distribution<int> intervals_dist(1, 10);
  std::uniform_int_distribution<int> seed_dist(1, 10000);
  std::uniform_int_distribution<int> length_dist(1, 500);

  const int num_iterations = 15;

  for (int iter = 0; iter < num_iterations; ++iter) {
    int num_rows = rows_dist(seed_gen);
    int min_intervals = intervals_dist(seed_gen);
    int max_intervals = std::min(min_intervals + intervals_dist(seed_gen), 20);
    int seed = seed_dist(seed_gen);
    int max_length = length_dist(seed_gen);

    RandomMeshConfig config = MediumConfig();
    config.seed = seed;
    config.sparsity = static_cast<double>(num_rows) / (config.y_max - config.y_min);
    config.intervals_per_row_min = min_intervals;
    config.intervals_per_row_max = max_intervals;
    config.interval_length_max = max_length;

    auto mesh_a = RandomMeshGenerator::generate_2d(config);
    config.seed++;
    auto mesh_b = RandomMeshGenerator::generate_2d(config);

    auto result_baseline = test::intersect_baseline_2d(mesh_a, mesh_b);
    auto result_in_place = intersect_in_place_2d(mesh_a, mesh_b);

    EXPECT_TRUE(common_meshes_equal(result_baseline, result_in_place))
        << "Workspace version produced different result than baseline (2D random, iteration=" << iter
        << ", seed=" << seed << ", rows=" << num_rows << ")";
  }
}

TEST_F(WorkspaceValidationTest, InPlaceMatchesBaseline_3D_RandomBounds) {
  std::mt19937 seed_gen(43);

  std::uniform_int_distribution<int> rows_dist(1, 20);
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

    auto result_baseline = test::intersect_baseline_3d(mesh_a, mesh_b);
    auto result_in_place = intersect_in_place_3d(mesh_a, mesh_b);

    EXPECT_TRUE(common_meshes_equal(result_baseline, result_in_place))
        << "Workspace version produced different result than baseline (3D random, iteration=" << iter
        << ", seed=" << seed << ", rows=" << num_rows << ")";
  }
}

// ============================================================================
// Mathematical Properties Tests with Workspace
// ============================================================================

TEST_F(WorkspaceValidationTest, InPlace_IsCommutative_2D_Random) {
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
    config.y_max = 10000;
    config.sparsity = static_cast<double>(num_rows) / (config.y_max - config.y_min);
    config.intervals_per_row_min = min_intervals;
    config.intervals_per_row_max = max_intervals;

    auto mesh_a = RandomMeshGenerator::generate_2d(config);
    config.seed++;
    auto mesh_b = RandomMeshGenerator::generate_2d(config);

    // Test A ∩ B = B ∩ A
    auto result_ab = intersect_in_place_2d(mesh_a, mesh_b);
    auto result_ba = intersect_in_place_2d(mesh_b, mesh_a);

    EXPECT_TRUE(common_meshes_equal(result_ab, result_ba))
        << "Workspace version is not commutative (2D random, iteration=" << iter << ")";
  }
}

TEST_F(WorkspaceValidationTest, InPlace_IsAssociative_2D_Random) {
  std::mt19937 seed_gen(45);
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
    config.y_max = 10000;
    config.sparsity = static_cast<double>(num_rows) / (config.y_max - config.y_min);
    config.intervals_per_row_min = min_intervals;
    config.intervals_per_row_max = max_intervals;

    auto mesh_a = RandomMeshGenerator::generate_2d(config);
    config.seed++;
    auto mesh_b = RandomMeshGenerator::generate_2d(config);
    config.seed++;
    auto mesh_c = RandomMeshGenerator::generate_2d(config);

    // Test (A ∩ B) ∩ C = A ∩ (B ∩ C)
    auto result_ab = intersect_in_place_2d(mesh_a, mesh_b);
    auto result_ab_c = intersect_in_place_2d(result_ab, mesh_c);  // (A ∩ B) ∩ C

    auto result_bc = intersect_in_place_2d(mesh_b, mesh_c);
    auto result_a_bc = intersect_in_place_2d(mesh_a, result_bc);  // A ∩ (B ∩ C)

    EXPECT_TRUE(common_meshes_equal(result_ab_c, result_a_bc))
        << "Workspace version is not associative (2D random, iteration=" << iter << ")";
  }
}

TEST_F(WorkspaceValidationTest, InPlace_IsIdempotent_2D_Random) {
  std::mt19937 seed_gen(46);
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
    config.y_max = 10000;
    config.sparsity = static_cast<double>(num_rows) / (config.y_max - config.y_min);
    config.intervals_per_row_min = min_intervals;
    config.intervals_per_row_max = max_intervals;

    auto mesh_a = RandomMeshGenerator::generate_2d(config);

    // Test A ∩ A = A
    auto result_aa = intersect_in_place_2d(mesh_a, mesh_a);

    EXPECT_TRUE(common_meshes_equal(result_aa, mesh_a))
        << "Workspace version is not idempotent (2D random, iteration=" << iter << ")";
  }
}

#endif // SUBSETIX_ENABLE_PLAYGROUND
