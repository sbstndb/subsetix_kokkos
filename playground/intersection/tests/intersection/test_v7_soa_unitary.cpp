// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#ifdef SUBSETIX_ENABLE_PLAYGROUND

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <vector>

#include <playground/subsetix/csr/intersection/algorithm/optimized.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v7_soa_optimized.hpp>

using namespace playground::subsetix::csr::intersection;

// ============================================================================
// Test Fixture for v7 SOA Optimized
// ============================================================================

class V7SoaOptimizedTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  // Helper to create a simple 2D mesh
  optimized::Mesh2D<> create_simple_mesh_2d(const std::vector<int32_t>& y_coords) {
    optimized::Mesh2D<> mesh;
    mesh.num_rows = y_coords.size();
    mesh.num_intervals = y_coords.size();

    if (y_coords.empty()) {
      return mesh;
    }

    using RowKey = playground::subsetix::csr::intersection::RowKey2D<int32_t>;
    using Interval = playground::subsetix::csr::intersection::Interval<int32_t>;
    using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;

    mesh.row_keys = Kokkos::View<RowKey*, DeviceSpace>("row_keys", y_coords.size());
    mesh.row_ptr = Kokkos::View<std::size_t*, DeviceSpace>("row_ptr", y_coords.size() + 1);
    mesh.intervals = Kokkos::View<Interval*, DeviceSpace>("intervals", y_coords.size());

    auto host_row_keys = Kokkos::create_mirror_view(mesh.row_keys);
    auto host_row_ptr = Kokkos::create_mirror_view(mesh.row_ptr);
    auto host_intervals = Kokkos::create_mirror_view(mesh.intervals);

    for (std::size_t i = 0; i < y_coords.size(); ++i) {
      host_row_keys(i).y = y_coords[i];
      host_row_ptr(i) = i;
      host_intervals(i) = Interval{0, 10};
    }
    host_row_ptr(y_coords.size()) = y_coords.size();

    Kokkos::deep_copy(mesh.row_keys, host_row_keys);
    Kokkos::deep_copy(mesh.row_ptr, host_row_ptr);
    Kokkos::deep_copy(mesh.intervals, host_intervals);

    return mesh;
  }

  // Helper to compare two meshes
  bool meshes_equal_2d(const optimized::Mesh2D<>& a, const optimized::Mesh2D<>& b) {
    if (a.num_rows != b.num_rows || a.num_intervals != b.num_intervals) {
      return false;
    }
    if (a.num_rows == 0) {
      return true;
    }

    // Convert to host for comparison
    auto host_a_row_keys = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a.row_keys);
    auto host_a_row_ptr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a.row_ptr);
    auto host_a_intervals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a.intervals);

    auto host_b_row_keys = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b.row_keys);
    auto host_b_row_ptr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b.row_ptr);
    auto host_b_intervals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b.intervals);

    for (std::size_t i = 0; i < a.num_rows; ++i) {
      if (host_a_row_keys(i).y != host_b_row_keys(i).y) {
        return false;
      }
    }

    for (std::size_t i = 0; i <= a.num_rows; ++i) {
      if (host_a_row_ptr(i) != host_b_row_ptr(i)) {
        return false;
      }
    }

    for (std::size_t i = 0; i < a.num_intervals; ++i) {
      if (host_a_intervals(i).begin != host_b_intervals(i).begin ||
          host_a_intervals(i).end != host_b_intervals(i).end) {
        return false;
      }
    }

    return true;
  }
};

// ============================================================================
// 2D Tests - v7 should produce same results as baseline
// ============================================================================

TEST_F(V7SoaOptimizedTest, EmptyMeshes2D) {
  auto empty_a = create_simple_mesh_2d({});
  auto mesh_b = create_simple_mesh_2d({0, 1, 2});

  auto baseline_result = optimized::intersect_meshes_2d(empty_a, mesh_b);
  auto v7_result = soa_optimized::intersect_meshes_2d(empty_a, mesh_b);

  EXPECT_EQ(baseline_result.num_rows, 0);
  EXPECT_EQ(v7_result.num_rows, 0);
}

TEST_F(V7SoaOptimizedTest, AllRowsMatch2D) {
  auto mesh_a = create_simple_mesh_2d({0, 1, 2, 3, 4});
  auto mesh_b = create_simple_mesh_2d({0, 1, 2, 3, 4});

  auto baseline_result = optimized::intersect_meshes_2d(mesh_a, mesh_b);
  auto v7_result = soa_optimized::intersect_meshes_2d(mesh_a, mesh_b);

  EXPECT_TRUE(meshes_equal_2d(baseline_result, v7_result));
  EXPECT_EQ(baseline_result.num_rows, 5);
  EXPECT_EQ(v7_result.num_rows, 5);
}

TEST_F(V7SoaOptimizedTest, PartialOverlap2D) {
  auto mesh_a = create_simple_mesh_2d({0, 1, 2, 3, 4});
  auto mesh_b = create_simple_mesh_2d({2, 3, 4, 5, 6});

  auto baseline_result = optimized::intersect_meshes_2d(mesh_a, mesh_b);
  auto v7_result = soa_optimized::intersect_meshes_2d(mesh_a, mesh_b);

  EXPECT_TRUE(meshes_equal_2d(baseline_result, v7_result));
  EXPECT_EQ(baseline_result.num_rows, 3);
  EXPECT_EQ(v7_result.num_rows, 3);
}

TEST_F(V7SoaOptimizedTest, NoOverlap2D) {
  auto mesh_a = create_simple_mesh_2d({0, 1, 2});
  auto mesh_b = create_simple_mesh_2d({5, 6, 7});

  auto baseline_result = optimized::intersect_meshes_2d(mesh_a, mesh_b);
  auto v7_result = soa_optimized::intersect_meshes_2d(mesh_a, mesh_b);

  EXPECT_EQ(baseline_result.num_rows, 0);
  EXPECT_EQ(v7_result.num_rows, 0);
}

TEST_F(V7SoaOptimizedTest, DenseSequence2D) {
  // Test with larger dense meshes
  std::vector<int32_t> coords_a(100);
  std::vector<int32_t> coords_b(200);
  for (int i = 0; i < 100; ++i) coords_a[i] = i;
  for (int i = 0; i < 200; ++i) coords_b[i] = i;

  auto mesh_a = create_simple_mesh_2d(coords_a);
  auto mesh_b = create_simple_mesh_2d(coords_b);

  auto baseline_result = optimized::intersect_meshes_2d(mesh_a, mesh_b);
  auto v7_result = soa_optimized::intersect_meshes_2d(mesh_a, mesh_b);

  EXPECT_TRUE(meshes_equal_2d(baseline_result, v7_result));
  EXPECT_EQ(baseline_result.num_rows, 100);
  EXPECT_EQ(v7_result.num_rows, 100);
}

// ============================================================================
// 3D Tests
// ============================================================================

TEST_F(V7SoaOptimizedTest, AllRowsMatch3D) {
  // Create simple 3D meshes
  using RowKey = playground::subsetix::csr::intersection::RowKey3D<int32_t>;
  using Interval = playground::subsetix::csr::intersection::Interval<int32_t>;
  using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;

  optimized::Mesh3D<> mesh_a, mesh_b;

  std::vector<std::pair<int32_t, int32_t>> coords = {{0,0}, {0,1}, {1,0}, {1,1}, {2,0}};

  mesh_a.num_rows = coords.size();
  mesh_a.num_intervals = coords.size();
  mesh_b.num_rows = coords.size();
  mesh_b.num_intervals = coords.size();

  mesh_a.row_keys = Kokkos::View<RowKey*, DeviceSpace>("row_keys_a", coords.size());
  mesh_a.row_ptr = Kokkos::View<std::size_t*, DeviceSpace>("row_ptr_a", coords.size() + 1);
  mesh_a.intervals = Kokkos::View<Interval*, DeviceSpace>("intervals_a", coords.size());

  mesh_b.row_keys = Kokkos::View<RowKey*, DeviceSpace>("row_keys_b", coords.size());
  mesh_b.row_ptr = Kokkos::View<std::size_t*, DeviceSpace>("row_ptr_b", coords.size() + 1);
  mesh_b.intervals = Kokkos::View<Interval*, DeviceSpace>("intervals_b", coords.size());

  auto host_a_keys = Kokkos::create_mirror_view(mesh_a.row_keys);
  auto host_a_ptr = Kokkos::create_mirror_view(mesh_a.row_ptr);
  auto host_a_intervals = Kokkos::create_mirror_view(mesh_a.intervals);

  auto host_b_keys = Kokkos::create_mirror_view(mesh_b.row_keys);
  auto host_b_ptr = Kokkos::create_mirror_view(mesh_b.row_ptr);
  auto host_b_intervals = Kokkos::create_mirror_view(mesh_b.intervals);

  for (std::size_t i = 0; i < coords.size(); ++i) {
    host_a_keys(i).y = coords[i].first;
    host_a_keys(i).z = coords[i].second;
    host_a_ptr(i) = i;
    host_a_intervals(i) = Interval{0, 10};

    host_b_keys(i).y = coords[i].first;
    host_b_keys(i).z = coords[i].second;
    host_b_ptr(i) = i;
    host_b_intervals(i) = Interval{0, 10};
  }
  host_a_ptr(coords.size()) = coords.size();
  host_b_ptr(coords.size()) = coords.size();

  Kokkos::deep_copy(mesh_a.row_keys, host_a_keys);
  Kokkos::deep_copy(mesh_a.row_ptr, host_a_ptr);
  Kokkos::deep_copy(mesh_a.intervals, host_a_intervals);

  Kokkos::deep_copy(mesh_b.row_keys, host_b_keys);
  Kokkos::deep_copy(mesh_b.row_ptr, host_b_ptr);
  Kokkos::deep_copy(mesh_b.intervals, host_b_intervals);

  auto baseline_result = optimized::intersect_meshes_3d(mesh_a, mesh_b);
  auto v7_result = soa_optimized::intersect_meshes_3d(mesh_a, mesh_b);

  EXPECT_EQ(baseline_result.num_rows, 5);
  EXPECT_EQ(v7_result.num_rows, 5);
  EXPECT_EQ(baseline_result.num_intervals, 5);
  EXPECT_EQ(v7_result.num_intervals, 5);
}

#endif // SUBSETIX_ENABLE_PLAYGROUND
