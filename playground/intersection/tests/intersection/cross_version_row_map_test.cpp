// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#ifdef SUBSETIX_ENABLE_PLAYGROUND

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <vector>
#include <random>
#include <algorithm>

// Include all algorithm headers
#include <playground/subsetix/csr/intersection/algorithm/optimized.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v4_hash.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v5_parallel_merge.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v6_direct_index.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v7_soa_optimized.hpp>
// v8 and v9 have compilation errors, commented out for now
// #include <playground/subsetix/csr/intersection/algorithm/v8_hybrid_cpu_gpu.hpp>
// #include <playground/subsetix/csr/intersection/algorithm/v9_adaptive.hpp>

using namespace playground::subsetix::csr::intersection;

// ============================================================================
// Test Fixture
// ============================================================================

class CrossVersionRowMapTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Already initialized by test_main.cpp
  }

  void TearDown() override {
    // Will be finalized by test_main.cpp
  }

  // Type aliases for each version
  using OptimizedMesh2D = optimized::Mesh2D<>;
  using OptimizedMesh3D = optimized::Mesh3D<>;
  using HashMesh2D = hash_based::Mesh2D<>;
  using HashMesh3D = hash_based::Mesh3D<>;
  using MergeMesh2D = parallel_merge::Mesh2D<>;
  using MergeMesh3D = parallel_merge::Mesh3D<>;
  using DirectMesh2D = direct_index::Mesh2D<>;
  using DirectMesh3D = direct_index::Mesh3D<>;
  // v7 uses optimized::Mesh as input/output type, same as baseline
  // using SoaMesh2D = soa_optimized::Mesh2D<>;
  // using SoaMesh3D = soa_optimized::Mesh3D<>;
  // using HybridMesh2D = hybrid_cpu_gpu::Mesh2D<>;
  // using HybridMesh3D = hybrid_cpu_gpu::Mesh3D<>;
  // using AdaptiveMesh2D = adaptive::Mesh2D<>;
  // using AdaptiveMesh3D = adaptive::Mesh3D<>;

  using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
  using HostSpace = Kokkos::HostSpace;
};

// ============================================================================
// Comparison Helper Functions
// ============================================================================

/**
 * @brief Compare two meshes for equality (2D version)
 *
 * Converts both meshes to host space and compares all fields.
 */
template <class MeshA, class MeshB>
::testing::AssertionResult meshes_are_equal_2d(
    const std::string& name_a,
    const MeshA& mesh_a,
    const std::string& name_b,
    const MeshB& mesh_b) {

  using CoordType = typename MeshA::coord_type;
  using IndexType = typename MeshA::index_type;
  using RowKey = playground::subsetix::csr::intersection::RowKey2D<CoordType>;
  using Interval = playground::subsetix::csr::intersection::Interval<CoordType>;

  // Check basic properties
  if (mesh_a.num_rows != mesh_b.num_rows) {
    return ::testing::AssertionFailure()
        << name_a << " has " << mesh_a.num_rows << " rows, "
        << name_b << " has " << mesh_b.num_rows << " rows";
  }

  if (mesh_a.num_intervals != mesh_b.num_intervals) {
    return ::testing::AssertionFailure()
        << name_a << " has " << mesh_a.num_intervals << " intervals, "
        << name_b << " has " << mesh_b.num_intervals << " intervals";
  }

  // Empty meshes are equal
  if (mesh_a.num_rows == 0) {
    return ::testing::AssertionSuccess();
  }

  // Convert to host for comparison
  auto host_a = optimized::mesh_to<HostSpace>(mesh_a);
  auto host_b = optimized::mesh_to<HostSpace>(mesh_b);

  // Compare row keys
  for (std::size_t i = 0; i < mesh_a.num_rows; ++i) {
    const RowKey& key_a = host_a.row_keys(i);
    const RowKey& key_b = host_b.row_keys(i);

    if (key_a.y != key_b.y) {
      return ::testing::AssertionFailure()
          << "Row keys differ at index " << i << ": "
          << name_a << " has y=" << key_a.y << ", "
          << name_b << " has y=" << key_b.y;
    }
  }

  // Compare row_ptr
  for (std::size_t i = 0; i <= mesh_a.num_rows; ++i) {
    const IndexType ptr_a = host_a.row_ptr(i);
    const IndexType ptr_b = host_b.row_ptr(i);

    if (ptr_a != ptr_b) {
      return ::testing::AssertionFailure()
          << "Row ptr differs at index " << i << ": "
          << name_a << " has " << ptr_a << ", "
          << name_b << " has " << ptr_b;
    }
  }

  // Compare intervals
  for (std::size_t i = 0; i < mesh_a.num_intervals; ++i) {
    const Interval& int_a = host_a.intervals(i);
    const Interval& int_b = host_b.intervals(i);

    if (int_a.begin != int_b.begin || int_a.end != int_b.end) {
      return ::testing::AssertionFailure()
          << "Intervals differ at index " << i << ": "
          << name_a << " has [" << int_a.begin << "," << int_a.end << "), "
          << name_b << " has [" << int_b.begin << "," << int_b.end << ")";
    }
  }

  return ::testing::AssertionSuccess();
}

/**
 * @brief Compare two meshes for equality (3D version)
 */
template <class MeshA, class MeshB>
::testing::AssertionResult meshes_are_equal_3d(
    const std::string& name_a,
    const MeshA& mesh_a,
    const std::string& name_b,
    const MeshB& mesh_b) {

  using CoordType = typename MeshA::coord_type;
  using IndexType = typename MeshA::index_type;
  using RowKey = playground::subsetix::csr::intersection::RowKey3D<CoordType>;
  using Interval = playground::subsetix::csr::intersection::Interval<CoordType>;

  // Check basic properties
  if (mesh_a.num_rows != mesh_b.num_rows) {
    return ::testing::AssertionFailure()
        << name_a << " has " << mesh_a.num_rows << " rows, "
        << name_b << " has " << mesh_b.num_rows << " rows";
  }

  if (mesh_a.num_intervals != mesh_b.num_intervals) {
    return ::testing::AssertionFailure()
        << name_a << " has " << mesh_a.num_intervals << " intervals, "
        << name_b << " has " << mesh_b.num_intervals << " intervals";
  }

  // Empty meshes are equal
  if (mesh_a.num_rows == 0) {
    return ::testing::AssertionSuccess();
  }

  // Convert to host for comparison
  auto host_a = optimized::mesh_to<HostSpace>(mesh_a);
  auto host_b = optimized::mesh_to<HostSpace>(mesh_b);

  // Compare row keys (3D: y and z)
  for (std::size_t i = 0; i < mesh_a.num_rows; ++i) {
    const RowKey& key_a = host_a.row_keys(i);
    const RowKey& key_b = host_b.row_keys(i);

    if (key_a.y != key_b.y || key_a.z != key_b.z) {
      return ::testing::AssertionFailure()
          << "Row keys differ at index " << i << ": "
          << name_a << " has (y=" << key_a.y << ",z=" << key_a.z << "), "
          << name_b << " has (y=" << key_b.y << ",z=" << key_b.z << ")";
    }
  }

  // Compare row_ptr
  for (std::size_t i = 0; i <= mesh_a.num_rows; ++i) {
    const IndexType ptr_a = host_a.row_ptr(i);
    const IndexType ptr_b = host_b.row_ptr(i);

    if (ptr_a != ptr_b) {
      return ::testing::AssertionFailure()
          << "Row ptr differs at index " << i << ": "
          << name_a << " has " << ptr_a << ", "
          << name_b << " has " << ptr_b;
    }
  }

  // Compare intervals
  for (std::size_t i = 0; i < mesh_a.num_intervals; ++i) {
    const Interval& int_a = host_a.intervals(i);
    const Interval& int_b = host_b.intervals(i);

    if (int_a.begin != int_b.begin || int_a.end != int_b.end) {
      return ::testing::AssertionFailure()
          << "Intervals differ at index " << i << ": "
          << name_a << " has [" << int_a.begin << "," << int_a.end << "), "
          << name_b << " has [" << int_b.begin << "," << int_b.end << ")";
    }
  }

  return ::testing::AssertionSuccess();
}

// ============================================================================
// Helper Functions to Create Test Meshes
// ============================================================================

/**
 * @brief Create a simple 2D mesh from list of Y coordinates
 *
 * Each row will have one interval [0, 10) for simplicity.
 * The focus is on testing row mapping correctness.
 */
optimized::Mesh2D<> create_test_mesh_2d(const std::vector<int32_t>& y_coords) {
  optimized::Mesh2D<> mesh;
  mesh.num_rows = y_coords.size();
  mesh.num_intervals = y_coords.size();  // One interval per row

  if (y_coords.empty()) {
    return mesh;
  }

  using RowKey = playground::subsetix::csr::intersection::RowKey2D<int32_t>;
  using Interval = playground::subsetix::csr::intersection::Interval<int32_t>;
  using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;

  mesh.row_keys = Kokkos::View<RowKey*, DeviceSpace>("row_keys", y_coords.size());
  mesh.row_ptr = Kokkos::View<std::size_t*, DeviceSpace>("row_ptr", y_coords.size() + 1);
  mesh.intervals = Kokkos::View<Interval*, DeviceSpace>("intervals", y_coords.size());

  // Create host mirrors
  auto host_row_keys = Kokkos::create_mirror_view(mesh.row_keys);
  auto host_row_ptr = Kokkos::create_mirror_view(mesh.row_ptr);
  auto host_intervals = Kokkos::create_mirror_view(mesh.intervals);

  // Fill data
  for (std::size_t i = 0; i < y_coords.size(); ++i) {
    host_row_keys(i).y = y_coords[i];
    host_row_ptr(i) = i;
    host_intervals(i) = Interval{0, 10};  // Simple interval [0, 10)
  }
  host_row_ptr(y_coords.size()) = y_coords.size();

  // Copy to device
  Kokkos::deep_copy(mesh.row_keys, host_row_keys);
  Kokkos::deep_copy(mesh.row_ptr, host_row_ptr);
  Kokkos::deep_copy(mesh.intervals, host_intervals);

  return mesh;
}

/**
 * @brief Create dense mesh with consecutive Y coordinates
 */
optimized::Mesh2D<> create_test_mesh_2d_dense(int32_t y_start, std::size_t num_rows) {
  std::vector<int32_t> y_coords(num_rows);
  for (std::size_t i = 0; i < num_rows; ++i) {
    y_coords[i] = y_start + static_cast<int32_t>(i);
  }
  return create_test_mesh_2d(y_coords);
}

/**
 * @brief Create mesh with uniform stride
 */
optimized::Mesh2D<> create_test_mesh_2d_stride(int32_t y_start, int32_t y_end, int32_t stride) {
  std::vector<int32_t> y_coords;
  for (int32_t y = y_start; y <= y_end; y += stride) {
    y_coords.push_back(y);
  }
  return create_test_mesh_2d(y_coords);
}

/**
 * @brief Create random sparse mesh (for hash-based testing)
 */
optimized::Mesh2D<> create_random_mesh_2d(std::size_t num_rows, int32_t coord_range) {
  std::vector<int32_t> y_coords;

  // Use fixed seed for reproducibility
  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(0, coord_range - 1);

  // Generate unique random coordinates
  std::set<int32_t> unique_coords;
  while (unique_coords.size() < num_rows) {
    unique_coords.insert(dist(gen));
  }

  // Convert to vector (already sorted by std::set)
  y_coords.assign(unique_coords.begin(), unique_coords.end());

  return create_test_mesh_2d(y_coords);
}

/**
 * @brief Create a simple 3D mesh from list of (Y, Z) coordinate pairs
 */
optimized::Mesh3D<> create_test_mesh_3d(const std::vector<std::pair<int32_t, int32_t>>& yz_coords) {
  optimized::Mesh3D<> mesh;
  mesh.num_rows = yz_coords.size();
  mesh.num_intervals = yz_coords.size();  // One interval per row

  if (yz_coords.empty()) {
    return mesh;
  }

  using RowKey = playground::subsetix::csr::intersection::RowKey3D<int32_t>;
  using Interval = playground::subsetix::csr::intersection::Interval<int32_t>;
  using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;

  mesh.row_keys = Kokkos::View<RowKey*, DeviceSpace>("row_keys", yz_coords.size());
  mesh.row_ptr = Kokkos::View<std::size_t*, DeviceSpace>("row_ptr", yz_coords.size() + 1);
  mesh.intervals = Kokkos::View<Interval*, DeviceSpace>("intervals", yz_coords.size());

  // Create host mirrors
  auto host_row_keys = Kokkos::create_mirror_view(mesh.row_keys);
  auto host_row_ptr = Kokkos::create_mirror_view(mesh.row_ptr);
  auto host_intervals = Kokkos::create_mirror_view(mesh.intervals);

  // Fill data
  for (std::size_t i = 0; i < yz_coords.size(); ++i) {
    host_row_keys(i).y = yz_coords[i].first;
    host_row_keys(i).z = yz_coords[i].second;
    host_row_ptr(i) = i;
    host_intervals(i) = Interval{0, 10};  // Simple interval [0, 10)
  }
  host_row_ptr(yz_coords.size()) = yz_coords.size();

  // Copy to device
  Kokkos::deep_copy(mesh.row_keys, host_row_keys);
  Kokkos::deep_copy(mesh.row_ptr, host_row_ptr);
  Kokkos::deep_copy(mesh.intervals, host_intervals);

  return mesh;
}

/**
 * @brief Convert an optimized mesh to other version's mesh type
 *
 * This is needed because each version has its own Mesh type.
 */
template <typename DestMesh, typename SourceMesh>
DestMesh convert_mesh(const SourceMesh& src) {
  DestMesh dst;
  dst.num_rows = src.num_rows;
  dst.num_intervals = src.num_intervals;

  if (src.num_rows == 0) {
    return dst;
  }

  dst.row_keys = Kokkos::create_mirror_view_and_copy(
      typename DestMesh::memory_space{}, src.row_keys);
  dst.row_ptr = Kokkos::create_mirror_view_and_copy(
      typename DestMesh::memory_space{}, src.row_ptr);
  dst.intervals = Kokkos::create_mirror_view_and_copy(
      typename DestMesh::memory_space{}, src.intervals);

  return dst;
}

// ============================================================================
// 2D Test Cases
// ============================================================================

TEST_F(CrossVersionRowMapTest, EmptyMeshes2D) {
  // Create empty mesh A
  auto empty_a = create_test_mesh_2d({});
  // Create mesh B with some rows
  auto mesh_b = create_test_mesh_2d({0, 1, 2, 3, 4});

  // Convert to each version's mesh type
  auto empty_a_hash = convert_mesh<HashMesh2D>(empty_a);
  auto empty_a_merge = convert_mesh<MergeMesh2D>(empty_a);
  auto empty_a_direct = convert_mesh<DirectMesh2D>(empty_a);
  // v7 uses optimized mesh directly, no conversion needed
  // auto empty_a_hybrid = convert_mesh<HybridMesh2D>(empty_a);
//   auto empty_a_adaptive = convert_mesh<AdaptiveMesh2D>(empty_a);

  auto mesh_b_hash = convert_mesh<HashMesh2D>(mesh_b);
  auto mesh_b_merge = convert_mesh<MergeMesh2D>(mesh_b);
  auto mesh_b_direct = convert_mesh<DirectMesh2D>(mesh_b);
// v7 uses optimized mesh directly, no conversion needed
// //   auto mesh_b_hybrid = convert_mesh<HybridMesh2D>(mesh_b);
// //   auto mesh_b_adaptive = convert_mesh<AdaptiveMesh2D>(mesh_b);

  // Test all versions produce empty result
  auto baseline = optimized::intersect_meshes_2d(empty_a, mesh_b);
  auto v4 = hash_based::intersect_meshes_2d(empty_a_hash, mesh_b_hash);
  auto v5 = parallel_merge::intersect_meshes_2d(empty_a_merge, mesh_b_merge);
  auto v6 = direct_index::intersect_meshes_2d(empty_a_direct, mesh_b_direct);
  auto v7 = soa_optimized::intersect_meshes_2d(empty_a, mesh_b);
//   auto v8 = hybrid_cpu_gpu::intersect_meshes_2d(empty_a_hybrid, mesh_b_hybrid);
//   auto v9 = adaptive::intersect_meshes_2d(empty_a_adaptive, mesh_b_adaptive);

  EXPECT_EQ(baseline.num_rows, 0);
  EXPECT_EQ(v4.num_rows, 0);
  EXPECT_EQ(v5.num_rows, 0);
  EXPECT_EQ(v6.num_rows, 0);
  EXPECT_EQ(v7.num_rows, 0);
//   EXPECT_EQ(v8.num_rows, 0);
//   EXPECT_EQ(v9.num_rows, 0);
}

TEST_F(CrossVersionRowMapTest, AllRowsMatch2D) {
  // A: y = 0, 1, 2, 3, 4
  // B: y = 0, 1, 2, 3, 4 (same)
  auto mesh_a = create_test_mesh_2d({0, 1, 2, 3, 4});
  auto mesh_b = create_test_mesh_2d({0, 1, 2, 3, 4});

  // Convert to each version
  auto mesh_a_hash = convert_mesh<HashMesh2D>(mesh_a);
  auto mesh_a_merge = convert_mesh<MergeMesh2D>(mesh_a);
  auto mesh_a_direct = convert_mesh<DirectMesh2D>(mesh_a);
// v7 uses optimized mesh directly, no conversion needed
// //   auto mesh_a_hybrid = convert_mesh<HybridMesh2D>(mesh_a);
// //   auto mesh_a_adaptive = convert_mesh<AdaptiveMesh2D>(mesh_a);

  auto mesh_b_hash = convert_mesh<HashMesh2D>(mesh_b);
  auto mesh_b_merge = convert_mesh<MergeMesh2D>(mesh_b);
  auto mesh_b_direct = convert_mesh<DirectMesh2D>(mesh_b);
// v7 uses optimized mesh directly, no conversion needed
// //   auto mesh_b_hybrid = convert_mesh<HybridMesh2D>(mesh_b);
// //   auto mesh_b_adaptive = convert_mesh<AdaptiveMesh2D>(mesh_b);

  auto baseline = optimized::intersect_meshes_2d(mesh_a, mesh_b);
  auto v4 = hash_based::intersect_meshes_2d(mesh_a_hash, mesh_b_hash);
  auto v5 = parallel_merge::intersect_meshes_2d(mesh_a_merge, mesh_b_merge);
  auto v6 = direct_index::intersect_meshes_2d(mesh_a_direct, mesh_b_direct);
  auto v7 = soa_optimized::intersect_meshes_2d(mesh_a, mesh_b);
//   auto v8 = hybrid_cpu_gpu::intersect_meshes_2d(mesh_a_hybrid, mesh_b_hybrid);
//   auto v9 = adaptive::intersect_meshes_2d(mesh_a_adaptive, mesh_b_adaptive);

  // Compare all to baseline
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v4_hash", v4));
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v5_merge", v5));
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v6_direct", v6));
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v7_soa", v7));
//   // v8 has compilation errors, skipped
//   // v9 has compilation errors, skipped

  // Verify expected result: all 5 rows with 5 intervals each
  EXPECT_EQ(baseline.num_rows, 5);
  EXPECT_EQ(baseline.num_intervals, 5);
}

TEST_F(CrossVersionRowMapTest, PartialOverlap2D) {
  // A: y = 0, 1, 2, 3, 4
  // B: y = 2, 3, 4, 5, 6
  // Expected: y = 2, 3, 4
  auto mesh_a = create_test_mesh_2d({0, 1, 2, 3, 4});
  auto mesh_b = create_test_mesh_2d({2, 3, 4, 5, 6});

  auto mesh_a_hash = convert_mesh<HashMesh2D>(mesh_a);
  auto mesh_a_merge = convert_mesh<MergeMesh2D>(mesh_a);
  auto mesh_a_direct = convert_mesh<DirectMesh2D>(mesh_a);
// v7 uses optimized mesh directly, no conversion needed
// //   auto mesh_a_hybrid = convert_mesh<HybridMesh2D>(mesh_a);
// //   auto mesh_a_adaptive = convert_mesh<AdaptiveMesh2D>(mesh_a);

  auto mesh_b_hash = convert_mesh<HashMesh2D>(mesh_b);
  auto mesh_b_merge = convert_mesh<MergeMesh2D>(mesh_b);
  auto mesh_b_direct = convert_mesh<DirectMesh2D>(mesh_b);
// v7 uses optimized mesh directly, no conversion needed
// //   auto mesh_b_hybrid = convert_mesh<HybridMesh2D>(mesh_b);
// //   auto mesh_b_adaptive = convert_mesh<AdaptiveMesh2D>(mesh_b);

  auto baseline = optimized::intersect_meshes_2d(mesh_a, mesh_b);
  auto v4 = hash_based::intersect_meshes_2d(mesh_a_hash, mesh_b_hash);
  auto v5 = parallel_merge::intersect_meshes_2d(mesh_a_merge, mesh_b_merge);
  auto v6 = direct_index::intersect_meshes_2d(mesh_a_direct, mesh_b_direct);
  auto v7 = soa_optimized::intersect_meshes_2d(mesh_a, mesh_b);
//   auto v8 = hybrid_cpu_gpu::intersect_meshes_2d(mesh_a_hybrid, mesh_b_hybrid);
//   auto v9 = adaptive::intersect_meshes_2d(mesh_a_adaptive, mesh_b_adaptive);

  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v4_hash", v4));
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v5_merge", v5));
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v6_direct", v6));
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v7_soa", v7));
//   // v8 has compilation errors, skipped
//   // v9 has compilation errors, skipped

  // Verify expected result: 3 rows
  EXPECT_EQ(baseline.num_rows, 3);
  EXPECT_EQ(baseline.num_intervals, 3);
}

TEST_F(CrossVersionRowMapTest, DenseSequence2D) {
  // Test case optimized for v6 (direct index)
  // Dense consecutive coordinates
  auto mesh_a = create_test_mesh_2d_dense(0, 100);    // y = 0, 1, ..., 99
  auto mesh_b = create_test_mesh_2d_dense(0, 200);    // y = 0, 1, ..., 199

  auto mesh_a_hash = convert_mesh<HashMesh2D>(mesh_a);
  auto mesh_a_merge = convert_mesh<MergeMesh2D>(mesh_a);
  auto mesh_a_direct = convert_mesh<DirectMesh2D>(mesh_a);
// v7 uses optimized mesh directly, no conversion needed
// //   auto mesh_a_hybrid = convert_mesh<HybridMesh2D>(mesh_a);
// //   auto mesh_a_adaptive = convert_mesh<AdaptiveMesh2D>(mesh_a);

  auto mesh_b_hash = convert_mesh<HashMesh2D>(mesh_b);
  auto mesh_b_merge = convert_mesh<MergeMesh2D>(mesh_b);
  auto mesh_b_direct = convert_mesh<DirectMesh2D>(mesh_b);
// v7 uses optimized mesh directly, no conversion needed
// //   auto mesh_b_hybrid = convert_mesh<HybridMesh2D>(mesh_b);
// //   auto mesh_b_adaptive = convert_mesh<AdaptiveMesh2D>(mesh_b);

  auto baseline = optimized::intersect_meshes_2d(mesh_a, mesh_b);
  auto v4 = hash_based::intersect_meshes_2d(mesh_a_hash, mesh_b_hash);
  auto v5 = parallel_merge::intersect_meshes_2d(mesh_a_merge, mesh_b_merge);
  auto v6 = direct_index::intersect_meshes_2d(mesh_a_direct, mesh_b_direct);
  auto v7 = soa_optimized::intersect_meshes_2d(mesh_a, mesh_b);
//   auto v8 = hybrid_cpu_gpu::intersect_meshes_2d(mesh_a_hybrid, mesh_b_hybrid);
//   auto v9 = adaptive::intersect_meshes_2d(mesh_a_adaptive, mesh_b_adaptive);

  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v4_hash", v4));
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v5_merge", v5));
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v6_direct", v6));
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v7_soa", v7));
//   // v8 has compilation errors, skipped
//   // v9 has compilation errors, skipped

  // Verify expected result: 100 rows
  EXPECT_EQ(baseline.num_rows, 100);
  EXPECT_EQ(baseline.num_intervals, 100);
}

TEST_F(CrossVersionRowMapTest, UniformStride2D) {
  // Test case for v6 stride detection
  // A: y = 0, 5, 10, 15, 20
  // B: y = 0, 10, 20, 30
  // Expected: y = 0, 10, 20
  auto mesh_a = create_test_mesh_2d_stride(0, 20, 5);   // 0, 5, 10, 15, 20
  auto mesh_b = create_test_mesh_2d_stride(0, 30, 10);  // 0, 10, 20, 30

  auto mesh_a_hash = convert_mesh<HashMesh2D>(mesh_a);
  auto mesh_a_merge = convert_mesh<MergeMesh2D>(mesh_a);
  auto mesh_a_direct = convert_mesh<DirectMesh2D>(mesh_a);
// v7 uses optimized mesh directly, no conversion needed
// //   auto mesh_a_hybrid = convert_mesh<HybridMesh2D>(mesh_a);
// //   auto mesh_a_adaptive = convert_mesh<AdaptiveMesh2D>(mesh_a);

  auto mesh_b_hash = convert_mesh<HashMesh2D>(mesh_b);
  auto mesh_b_merge = convert_mesh<MergeMesh2D>(mesh_b);
  auto mesh_b_direct = convert_mesh<DirectMesh2D>(mesh_b);
// v7 uses optimized mesh directly, no conversion needed
// //   auto mesh_b_hybrid = convert_mesh<HybridMesh2D>(mesh_b);
// //   auto mesh_b_adaptive = convert_mesh<AdaptiveMesh2D>(mesh_b);

  auto baseline = optimized::intersect_meshes_2d(mesh_a, mesh_b);
  auto v4 = hash_based::intersect_meshes_2d(mesh_a_hash, mesh_b_hash);
  auto v5 = parallel_merge::intersect_meshes_2d(mesh_a_merge, mesh_b_merge);
  auto v6 = direct_index::intersect_meshes_2d(mesh_a_direct, mesh_b_direct);
  auto v7 = soa_optimized::intersect_meshes_2d(mesh_a, mesh_b);
//   auto v8 = hybrid_cpu_gpu::intersect_meshes_2d(mesh_a_hybrid, mesh_b_hybrid);
//   auto v9 = adaptive::intersect_meshes_2d(mesh_a_adaptive, mesh_b_adaptive);

  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v4_hash", v4));
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v5_merge", v5));
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v6_direct", v6));
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v7_soa", v7));
//   // v8 has compilation errors, skipped
//   // v9 has compilation errors, skipped

  // Verify expected result: 3 rows
  EXPECT_EQ(baseline.num_rows, 3);
  EXPECT_EQ(baseline.num_intervals, 3);
}

TEST_F(CrossVersionRowMapTest, RandomSparse2D) {
  // Test case for v4 (hash-based)
  // Random subsets of coordinate range
  auto mesh_a = create_random_mesh_2d(1000, 10000);  // 1000 rows from 0-10000
  auto mesh_b = create_random_mesh_2d(1000, 10000);

  auto mesh_a_hash = convert_mesh<HashMesh2D>(mesh_a);
  auto mesh_a_merge = convert_mesh<MergeMesh2D>(mesh_a);
  auto mesh_a_direct = convert_mesh<DirectMesh2D>(mesh_a);
// v7 uses optimized mesh directly, no conversion needed
// //   auto mesh_a_hybrid = convert_mesh<HybridMesh2D>(mesh_a);
// //   auto mesh_a_adaptive = convert_mesh<AdaptiveMesh2D>(mesh_a);

  auto mesh_b_hash = convert_mesh<HashMesh2D>(mesh_b);
  auto mesh_b_merge = convert_mesh<MergeMesh2D>(mesh_b);
  auto mesh_b_direct = convert_mesh<DirectMesh2D>(mesh_b);
// v7 uses optimized mesh directly, no conversion needed
// //   auto mesh_b_hybrid = convert_mesh<HybridMesh2D>(mesh_b);
// //   auto mesh_b_adaptive = convert_mesh<AdaptiveMesh2D>(mesh_b);

  auto baseline = optimized::intersect_meshes_2d(mesh_a, mesh_b);
  auto v4 = hash_based::intersect_meshes_2d(mesh_a_hash, mesh_b_hash);
  auto v5 = parallel_merge::intersect_meshes_2d(mesh_a_merge, mesh_b_merge);
  auto v6 = direct_index::intersect_meshes_2d(mesh_a_direct, mesh_b_direct);
  auto v7 = soa_optimized::intersect_meshes_2d(mesh_a, mesh_b);
//   auto v8 = hybrid_cpu_gpu::intersect_meshes_2d(mesh_a_hybrid, mesh_b_hybrid);
//   auto v9 = adaptive::intersect_meshes_2d(mesh_a_adaptive, mesh_b_adaptive);

  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v4_hash", v4));
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v5_merge", v5));
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v6_direct", v6));
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v7_soa", v7));
//   // v8 has compilation errors, skipped
//   // v9 has compilation errors, skipped
}

TEST_F(CrossVersionRowMapTest, SmallMeshes2D) {
  // < 100 rows
  auto mesh_a = create_test_mesh_2d({0, 1, 2, 3, 4, 5});
  auto mesh_b = create_test_mesh_2d({3, 4, 5, 6, 7, 8, 9});

  auto mesh_a_hash = convert_mesh<HashMesh2D>(mesh_a);
  auto mesh_a_merge = convert_mesh<MergeMesh2D>(mesh_a);
  auto mesh_a_direct = convert_mesh<DirectMesh2D>(mesh_a);
// v7 uses optimized mesh directly, no conversion needed
// //   auto mesh_a_hybrid = convert_mesh<HybridMesh2D>(mesh_a);
// //   auto mesh_a_adaptive = convert_mesh<AdaptiveMesh2D>(mesh_a);

  auto mesh_b_hash = convert_mesh<HashMesh2D>(mesh_b);
  auto mesh_b_merge = convert_mesh<MergeMesh2D>(mesh_b);
  auto mesh_b_direct = convert_mesh<DirectMesh2D>(mesh_b);
// v7 uses optimized mesh directly, no conversion needed
// //   auto mesh_b_hybrid = convert_mesh<HybridMesh2D>(mesh_b);
// //   auto mesh_b_adaptive = convert_mesh<AdaptiveMesh2D>(mesh_b);

  auto baseline = optimized::intersect_meshes_2d(mesh_a, mesh_b);
  auto v4 = hash_based::intersect_meshes_2d(mesh_a_hash, mesh_b_hash);
  auto v5 = parallel_merge::intersect_meshes_2d(mesh_a_merge, mesh_b_merge);
  auto v6 = direct_index::intersect_meshes_2d(mesh_a_direct, mesh_b_direct);
  auto v7 = soa_optimized::intersect_meshes_2d(mesh_a, mesh_b);
//   auto v8 = hybrid_cpu_gpu::intersect_meshes_2d(mesh_a_hybrid, mesh_b_hybrid);
//   auto v9 = adaptive::intersect_meshes_2d(mesh_a_adaptive, mesh_b_adaptive);

  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v4_hash", v4));
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v5_merge", v5));
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v6_direct", v6));
  EXPECT_TRUE(meshes_are_equal_2d("baseline", baseline, "v7_soa", v7));
//   // v8 has compilation errors, skipped
//   // v9 has compilation errors, skipped

  // Verify expected result: 3 rows
  EXPECT_EQ(baseline.num_rows, 3);
  EXPECT_EQ(baseline.num_intervals, 3);
}

// ============================================================================
// 3D Test Cases
// ============================================================================

TEST_F(CrossVersionRowMapTest, Basic3D) {
  // 3D: (y, z) lexicographic
  // A: (0,0), (0,1), (1,0), (1,1)
  // B: (0,1), (1,0), (1,1), (2,0)
  // Expected: (0,1), (1,0), (1,1)
  auto mesh_a = create_test_mesh_3d({{0,0}, {0,1}, {1,0}, {1,1}});
  auto mesh_b = create_test_mesh_3d({{0,1}, {1,0}, {1,1}, {2,0}});

  auto mesh_a_hash = convert_mesh<HashMesh3D>(mesh_a);
  auto mesh_a_merge = convert_mesh<MergeMesh3D>(mesh_a);
  auto mesh_a_direct = convert_mesh<DirectMesh3D>(mesh_a);
// v7 uses optimized mesh directly, no conversion needed
// //   auto mesh_a_hybrid = convert_mesh<HybridMesh3D>(mesh_a);
// //   auto mesh_a_adaptive = convert_mesh<AdaptiveMesh3D>(mesh_a);

  auto mesh_b_hash = convert_mesh<HashMesh3D>(mesh_b);
  auto mesh_b_merge = convert_mesh<MergeMesh3D>(mesh_b);
  auto mesh_b_direct = convert_mesh<DirectMesh3D>(mesh_b);
// v7 uses optimized mesh directly, no conversion needed
// //   auto mesh_b_hybrid = convert_mesh<HybridMesh3D>(mesh_b);
// //   auto mesh_b_adaptive = convert_mesh<AdaptiveMesh3D>(mesh_b);

  auto baseline = optimized::intersect_meshes_3d(mesh_a, mesh_b);
  auto v4 = hash_based::intersect_meshes_3d(mesh_a_hash, mesh_b_hash);
  auto v5 = parallel_merge::intersect_meshes_3d(mesh_a_merge, mesh_b_merge);
  auto v6 = direct_index::intersect_meshes_3d(mesh_a_direct, mesh_b_direct);
  auto v7 = soa_optimized::intersect_meshes_3d(mesh_a, mesh_b);
//   auto v8 = hybrid_cpu_gpu::intersect_meshes_3d(mesh_a_hybrid, mesh_b_hybrid);
//   auto v9 = adaptive::intersect_meshes_3d(mesh_a_adaptive, mesh_b_adaptive);

  EXPECT_TRUE(meshes_are_equal_3d("baseline", baseline, "v4_hash", v4));
  EXPECT_TRUE(meshes_are_equal_3d("baseline", baseline, "v5_merge", v5));
  EXPECT_TRUE(meshes_are_equal_3d("baseline", baseline, "v6_direct", v6));
  EXPECT_TRUE(meshes_are_equal_3d("baseline", baseline, "v7_soa", v7));
//   // v8 has compilation errors, skipped
//   // v9 has compilation errors, skipped

  // Verify expected result: 3 rows
  EXPECT_EQ(baseline.num_rows, 3);
  EXPECT_EQ(baseline.num_intervals, 3);
}

TEST_F(CrossVersionRowMapTest, EmptyMeshes3D) {
  // Create empty mesh A
  auto empty_a = create_test_mesh_3d({});
  // Create mesh B with some rows
  auto mesh_b = create_test_mesh_3d({{0,0}, {0,1}, {1,0}, {1,1}});

  auto empty_a_hash = convert_mesh<HashMesh3D>(empty_a);
  auto empty_a_merge = convert_mesh<MergeMesh3D>(empty_a);
  auto empty_a_direct = convert_mesh<DirectMesh3D>(empty_a);
  // v7 uses optimized mesh directly, no conversion needed
//   auto empty_a_hybrid = convert_mesh<HybridMesh3D>(empty_a);
//   auto empty_a_adaptive = convert_mesh<AdaptiveMesh3D>(empty_a);

  auto mesh_b_hash = convert_mesh<HashMesh3D>(mesh_b);
  auto mesh_b_merge = convert_mesh<MergeMesh3D>(mesh_b);
  auto mesh_b_direct = convert_mesh<DirectMesh3D>(mesh_b);
// v7 uses optimized mesh directly, no conversion needed
// //   auto mesh_b_hybrid = convert_mesh<HybridMesh3D>(mesh_b);
// //   auto mesh_b_adaptive = convert_mesh<AdaptiveMesh3D>(mesh_b);

  auto baseline = optimized::intersect_meshes_3d(empty_a, mesh_b);
  auto v4 = hash_based::intersect_meshes_3d(empty_a_hash, mesh_b_hash);
  auto v5 = parallel_merge::intersect_meshes_3d(empty_a_merge, mesh_b_merge);
  auto v6 = direct_index::intersect_meshes_3d(empty_a_direct, mesh_b_direct);
  auto v7 = soa_optimized::intersect_meshes_3d(empty_a, mesh_b);
//   auto v8 = hybrid_cpu_gpu::intersect_meshes_3d(empty_a_hybrid, mesh_b_hybrid);
//   auto v9 = adaptive::intersect_meshes_3d(empty_a_adaptive, mesh_b_adaptive);

  EXPECT_EQ(baseline.num_rows, 0);
  EXPECT_EQ(v4.num_rows, 0);
  EXPECT_EQ(v5.num_rows, 0);
  EXPECT_EQ(v6.num_rows, 0);
  EXPECT_EQ(v7.num_rows, 0);
//   EXPECT_EQ(v8.num_rows, 0);
//   EXPECT_EQ(v9.num_rows, 0);
}

TEST_F(CrossVersionRowMapTest, AllRowsMatch3D) {
  // A and B have same rows
  auto mesh_a = create_test_mesh_3d({{0,0}, {0,1}, {1,0}, {1,1}, {2,0}});
  auto mesh_b = create_test_mesh_3d({{0,0}, {0,1}, {1,0}, {1,1}, {2,0}});

  auto mesh_a_hash = convert_mesh<HashMesh3D>(mesh_a);
  auto mesh_a_merge = convert_mesh<MergeMesh3D>(mesh_a);
  auto mesh_a_direct = convert_mesh<DirectMesh3D>(mesh_a);
// v7 uses optimized mesh directly, no conversion needed
// //   auto mesh_a_hybrid = convert_mesh<HybridMesh3D>(mesh_a);
// //   auto mesh_a_adaptive = convert_mesh<AdaptiveMesh3D>(mesh_a);

  auto mesh_b_hash = convert_mesh<HashMesh3D>(mesh_b);
  auto mesh_b_merge = convert_mesh<MergeMesh3D>(mesh_b);
  auto mesh_b_direct = convert_mesh<DirectMesh3D>(mesh_b);
// v7 uses optimized mesh directly, no conversion needed
// //   auto mesh_b_hybrid = convert_mesh<HybridMesh3D>(mesh_b);
// //   auto mesh_b_adaptive = convert_mesh<AdaptiveMesh3D>(mesh_b);

  auto baseline = optimized::intersect_meshes_3d(mesh_a, mesh_b);
  auto v4 = hash_based::intersect_meshes_3d(mesh_a_hash, mesh_b_hash);
  auto v5 = parallel_merge::intersect_meshes_3d(mesh_a_merge, mesh_b_merge);
  auto v6 = direct_index::intersect_meshes_3d(mesh_a_direct, mesh_b_direct);
  auto v7 = soa_optimized::intersect_meshes_3d(mesh_a, mesh_b);
//   auto v8 = hybrid_cpu_gpu::intersect_meshes_3d(mesh_a_hybrid, mesh_b_hybrid);
//   auto v9 = adaptive::intersect_meshes_3d(mesh_a_adaptive, mesh_b_adaptive);

  EXPECT_TRUE(meshes_are_equal_3d("baseline", baseline, "v4_hash", v4));
  EXPECT_TRUE(meshes_are_equal_3d("baseline", baseline, "v5_merge", v5));
  EXPECT_TRUE(meshes_are_equal_3d("baseline", baseline, "v6_direct", v6));
  EXPECT_TRUE(meshes_are_equal_3d("baseline", baseline, "v7_soa", v7));
//   // v8 has compilation errors, skipped
//   // v9 has compilation errors, skipped

  // Verify expected result: 5 rows
  EXPECT_EQ(baseline.num_rows, 5);
  EXPECT_EQ(baseline.num_intervals, 5);
}

#endif // SUBSETIX_ENABLE_PLAYGROUND
