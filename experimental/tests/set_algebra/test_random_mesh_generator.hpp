// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include "test_common_format.hpp"
#include <experimental/subsetix/csr/set_algebra.hpp>
#include <experimental/subsetix/csr/set_algebra/v1.hpp>
#include <experimental/subsetix/csr/set_algebra/v2.hpp>
#include <experimental/subsetix/csr/set_algebra/v3.hpp>
#include <Kokkos_Random.hpp>
#include <algorithm>
#include <cmath>
#include <random>

// Convenience aliases for backward compatibility
using DefaultInterval = experimental::subsetix::csr::Interval<int32_t>;
using DefaultCommonRow2D = experimental::subsetix::csr::test::CommonRow2D<int32_t>;
using DefaultCommonRow3D = experimental::subsetix::csr::test::CommonRow3D<int32_t>;
using DefaultCommonMesh2D = experimental::subsetix::csr::test::CommonMesh2D<int32_t>;
using DefaultCommonMesh3D = experimental::subsetix::csr::test::CommonMesh3D<int32_t>;

// Bring version namespaces into scope for test helpers
using namespace experimental::subsetix::csr::v1;
using namespace experimental::subsetix::csr::v2;
using namespace experimental::subsetix::csr::v3;

namespace experimental::subsetix::csr::test {

// ============================================================================
// Configuration for Random Mesh Generation
// ============================================================================

/**
 * @brief Configuration for random mesh generation
 */
struct RandomMeshConfig {
  int seed = 42;                      // Random seed for reproducibility
  int num_rows_min = 10;              // Minimum number of rows
  int num_rows_max = 100;             // Maximum number of rows
  int intervals_per_row_min = 1;      // Minimum intervals per row
  int intervals_per_row_max = 10;     // Maximum intervals per row
  int interval_length_min = 1;        // Minimum interval length
  int interval_length_max = 100;      // Maximum interval length
  int gap_min = 0;                    // Minimum gap between intervals
  int gap_max = 50;                   // Maximum gap between intervals
  int y_min = 0;                      // Minimum Y coordinate
  int y_max = 10000;                  // Maximum Y coordinate
  int z_min = 0;                      // Minimum Z coordinate (3D only)
  int z_max = 100;                    // Maximum Z coordinate (3D only)
  bool sorted_rows = true;            // Whether row keys should be sorted
  double overlap_probability = 0.5;   // Probability of row overlap (for generating pairs)
};

// ============================================================================
// Predefined Mesh Configurations
// ============================================================================

/**
 * @brief Small mesh configuration for quick unit tests
 *
 * Suitable for:
 * - Fast unit tests
 * - Debugging
 * - CI/CD pipelines
 *
 * Parameters:
 * - Y scope: [0, 64]
 * - Z scope: [0, 64] (3D only)
 * - Max intervals per row: 4
 * - Row count: 400-1250
 * - 3D sparsity: ~30% (1250 / (64×64) ≈ 30%)
 */
inline RandomMeshConfig SmallConfig() {
  return RandomMeshConfig{
    .seed = 42,
    .num_rows_min = 400,
    .num_rows_max = 1250,
    .intervals_per_row_min = 1,
    .intervals_per_row_max = 4,
    .interval_length_min = 1,
    .interval_length_max = 100,
    .gap_min = 0,
    .gap_max = 50,
    .y_min = 0,
    .y_max = 64,
    .z_min = 0,
    .z_max = 64,
    .sorted_rows = true,
    .overlap_probability = 0.5
  };
}

/**
 * @brief Medium mesh configuration for standard tests
 *
 * Suitable for:
 * - Standard unit tests
 * - Integration tests
 * - Development testing
 *
 * Parameters:
 * - Y scope: [0, 512]
 * - Z scope: [0, 512] (3D only)
 * - Max intervals per row: 4
 * - Row count: 40000-78643
 * - 3D sparsity: ~30% (78643 / (512×512) ≈ 30%)
 */
inline RandomMeshConfig MediumConfig() {
  return RandomMeshConfig{
    .seed = 42,
    .num_rows_min = 40000,
    .num_rows_max = 78643,
    .intervals_per_row_min = 1,
    .intervals_per_row_max = 4,
    .interval_length_min = 1,
    .interval_length_max = 100,
    .gap_min = 0,
    .gap_max = 50,
    .y_min = 0,
    .y_max = 512,
    .z_min = 0,
    .z_max = 512,
    .sorted_rows = true,
    .overlap_probability = 0.5
  };
}

/**
 * @brief Large mesh configuration for stress tests and benchmarks
 *
 * Suitable for:
 * - Performance benchmarks
 * - Stress tests
 * - Large-scale validation
 *
 * Parameters:
 * - Y scope: [0, 4096]
 * - Z scope: [0, 4096] (3D only)
 * - Max intervals per row: 4
 * - Row count: 2.5M-5M
 * - 3D sparsity: ~30% (5M / (4096×4096) ≈ 30%)
 */
inline RandomMeshConfig LargeConfig() {
  return RandomMeshConfig{
    .seed = 42,
    .num_rows_min = 2500000,
    .num_rows_max = 5000000,
    .intervals_per_row_min = 1,
    .intervals_per_row_max = 4,
    .interval_length_min = 1,
    .interval_length_max = 100,
    .gap_min = 0,
    .gap_max = 50,
    .y_min = 0,
    .y_max = 4096,
    .z_min = 0,
    .z_max = 4096,
    .sorted_rows = true,
    .overlap_probability = 0.5
  };
}

/**
 * @brief Extra Large mesh configuration for extreme benchmarks
 *
 * Suitable for:
 * - Extreme performance benchmarks
 * - GPU stress tests (H100, B200)
 * - Maximum scale validation
 *
 * Parameters:
 * - Y scope: [0, 8192] (2x Large)
 * - Z scope: [0, 8192] (3D only, 2x Large)
 * - Max intervals per row: 4
 * - Row count: 5M-10M (2x Large)
 * - 3D sparsity: ~15% (10M / (8192×8192) ≈ 15%
 *
 * Memory estimate:
 * - 2D: ~10M intervals × 16 bytes = ~160 MB input data
 * - 3D: ~10M intervals × 24 bytes = ~240 MB input data
 */
inline RandomMeshConfig ExtraLargeConfig() {
  return RandomMeshConfig{
    .seed = 42,
    .num_rows_min = 5000000,
    .num_rows_max = 10000000,
    .intervals_per_row_min = 1,
    .intervals_per_row_max = 4,
    .interval_length_min = 1,
    .interval_length_max = 100,
    .gap_min = 0,
    .gap_max = 50,
    .y_min = 0,
    .y_max = 8192,
    .z_min = 0,
    .z_max = 8192,
    .sorted_rows = true,
    .overlap_probability = 0.5
  };
}

// ============================================================================
// Regular Mesh Configurations for Benchmarking
// ============================================================================

/**
 * @brief Configuration for regular mesh generation
 *
 * Regular meshes represent "optimal" performance scenarios:
 * - All rows are present (100% density, no gaps)
 * - Each row has exactly one interval covering the full X range
 * - Perfect alignment for set operations
 *
 * The num_rows parameter directly controls the number of rows (unlike random
 * which has y_max/z_max ranges with sparsity). This makes it easier to compare
 * regular vs random performance at similar mesh sizes.
 */
struct RegularMeshConfig {
  int num_rows_2d = 1250;   // Number of rows for 2D (y = 0, 1, ..., num_rows_2d-1)
  int num_rows_3d = 1250;   // Number of rows for 3D (uses y/z from random's y_max/z_max range)
};

/**
 * @brief Small regular mesh configuration (100% dense)
 *
 * Matches the number of rows in random SmallConfig but with 100% density:
 * - 2D: 1250 rows (same as random's ~1250)
 * - 3D: 4096 rows (64x64, which is 100% of random's y_max=64, z_max=64 range)
 *
 * The regular mesh has the same or more rows than random, but with NO sparsity.
 */
inline RegularMeshConfig SmallRegularConfig() {
  return RegularMeshConfig{.num_rows_2d = 1250, .num_rows_3d = 64 * 64};
}

/**
 * @brief Medium regular mesh configuration (100% dense)
 *
 * Matches the number of rows in random MediumConfig but with 100% density:
 * - 2D: 78643 rows (same as random's ~78643)
 * - 3D: 262144 rows (512x512, which is 100% of random's y_max=512, z_max=512 range)
 *
 * The regular mesh has the same or more rows than random, but with NO sparsity.
 */
inline RegularMeshConfig MediumRegularConfig() {
  return RegularMeshConfig{.num_rows_2d = 78643, .num_rows_3d = 512 * 512};
}

/**
 * @brief Large regular mesh configuration (100% dense)
 *
 * Matches the number of rows in random LargeConfig but with 100% density:
 * - 2D: 5M rows (same as random's ~5M)
 * - 3D: 1048576 rows (1024x1024, limited to avoid excessive memory vs random's 4096x4096)
 *
 * The regular mesh has the same or more rows than random, but with NO sparsity.
 */
inline RegularMeshConfig LargeRegularConfig() {
  return RegularMeshConfig{.num_rows_2d = 5000000, .num_rows_3d = 1024 * 1024};
}

// ============================================================================
// Regular Mesh Generators
// ============================================================================

/**
 * @brief Regular mesh generator for optimal performance benchmarking
 *
 * Generates fully dense meshes where:
 * - All rows are present (no gaps)
 * - Each row has exactly one interval [0, size)
 * - Perfect alignment for self-intersection (A ∩ A = A)
 *
 * This provides the "best case" performance scenario:
 * - No binary search misses
 * - All rows match perfectly
 * - All intervals match perfectly
 * - Minimal memory overhead (1 interval per row)
 */
class RegularMeshGenerator {
public:
  /**
   * @brief Generate a regular CommonMesh2D (100% dense)
   *
   * Creates a mesh where:
   * - y = 0, 1, 2, ..., num_rows-1 (ALL rows present, no gaps)
   * - Each row has one interval [0, num_rows) covering the full X range
   *
   * @param config Configuration containing num_rows_2d
   * @return Regular CommonMesh2D with num_rows_2d rows
   */
  static DefaultCommonMesh2D generate_2d(const RegularMeshConfig& config = RegularMeshConfig{}) {
    DefaultCommonMesh2D mesh;
    mesh.rows.reserve(config.num_rows_2d);

    for (int y = 0; y < config.num_rows_2d; ++y) {
      DefaultCommonRow2D row;
      row.y = static_cast<int32_t>(y);
      // Single interval covering the entire X range [0, num_rows)
      row.intervals.push_back(DefaultInterval{0, static_cast<int32_t>(config.num_rows_2d)});
      mesh.rows.push_back(std::move(row));
    }

    return mesh;
  }

  /**
   * @brief Generate a regular CommonMesh3D (100% dense cube)
   *
   * Creates a mesh where all (y,z) combinations in a square grid are present:
   * - (y, z) = (0,0), (0,1), ..., (0, sqrt)-1, (1,0), ..., (sqrt-1, sqrt-1)
   * - Total rows: num_rows_3d (which should be a perfect square)
   * - Each row has one interval [0, sqrt) covering the full X range
   *
   * @param config Configuration containing num_rows_3d
   * @return Regular CommonMesh3D with num_rows_3d rows
   */
  static DefaultCommonMesh3D generate_3d(const RegularMeshConfig& config = RegularMeshConfig{}) {
    DefaultCommonMesh3D mesh;
    int grid_size = static_cast<int>(std::sqrt(config.num_rows_3d));
    mesh.rows.reserve(config.num_rows_3d);

    for (int z = 0; z < grid_size; ++z) {
      for (int y = 0; y < grid_size; ++y) {
        DefaultCommonRow3D row;
        row.y = static_cast<int32_t>(y);
        row.z = static_cast<int32_t>(z);
        // Single interval covering the entire X range [0, grid_size)
        row.intervals.push_back(DefaultInterval{0, static_cast<int32_t>(grid_size)});
        mesh.rows.push_back(std::move(row));
      }
    }

    return mesh;
  }
};

// ============================================================================
// Random Mesh Generators
// ============================================================================

/**
 * @brief Random mesh generator using Kokkos_Random
 *
 * Generates random CommonMesh2D with configurable parameters.
 * Uses Kokkos_Random for GPU-compatible random number generation.
 */
class RandomMeshGenerator {
public:
  /**
   * @brief Generate a random CommonMesh2D
   *
   * @param config Configuration for generation
   * @return Random CommonMesh2D
   */
  static DefaultCommonMesh2D generate_2d(const RandomMeshConfig& config = RandomMeshConfig{}) {
    // Use std::mt19937 for host-side generation (simpler for test data)
    std::mt19937 gen(config.seed);

    // Generate number of rows
    std::uniform_int_distribution<int> rows_dist(config.num_rows_min, config.num_rows_max);
    int num_rows = rows_dist(gen);

    DefaultCommonMesh2D mesh;
    mesh.rows.reserve(num_rows);

    // Generate Y coordinates
    std::vector<int> y_coords;
    y_coords.reserve(num_rows);
    std::uniform_int_distribution<int> y_dist(config.y_min, config.y_max);

    for (int i = 0; i < num_rows; ++i) {
      y_coords.push_back(y_dist(gen));
    }

    // Sort if requested
    if (config.sorted_rows) {
      std::sort(y_coords.begin(), y_coords.end());
    }

    // Generate intervals for each row
    std::uniform_int_distribution<int> interval_count_dist(
      config.intervals_per_row_min,
      config.intervals_per_row_max
    );
    std::uniform_int_distribution<int> length_dist(
      config.interval_length_min,
      config.interval_length_max
    );
    std::uniform_int_distribution<int> gap_dist(config.gap_min, config.gap_max);

    for (int y : y_coords) {
      DefaultCommonRow2D row;
      row.y = static_cast<int32_t>(y);

      int num_intervals = interval_count_dist(gen);

      // Generate non-overlapping intervals by construction
      // Start at a random position
      std::uniform_int_distribution<int> start_dist(0, 1000);
      int current_x = start_dist(gen);

      for (int j = 0; j < num_intervals; ++j) {
        int length = length_dist(gen);
        DefaultInterval interval{static_cast<int32_t>(current_x), static_cast<int32_t>(current_x + length)};

        // Only add if non-empty
        if (!interval.empty()) {
          row.intervals.push_back(interval);
        }

        // Add gap before next interval
        int gap = gap_dist(gen);
        current_x += length + gap;
      }

      mesh.rows.push_back(std::move(row));
    }

    return mesh;
  }

  /**
   * @brief Generate a random CommonMesh3D
   *
   * @param config Configuration for generation
   * @return Random CommonMesh3D
   */
  static DefaultCommonMesh3D generate_3d(const RandomMeshConfig& config = RandomMeshConfig{}) {
    // Use std::mt19937 for host-side generation
    std::mt19937 gen(config.seed);

    // Generate number of rows
    std::uniform_int_distribution<int> rows_dist(config.num_rows_min, config.num_rows_max);
    int num_rows = rows_dist(gen);

    DefaultCommonMesh3D mesh;
    mesh.rows.reserve(num_rows);

    // Generate (Y, Z) coordinates
    std::uniform_int_distribution<int> y_dist(config.y_min, config.y_max);
    std::uniform_int_distribution<int> z_dist(config.z_min, config.z_max);

    for (int i = 0; i < num_rows; ++i) {
      DefaultCommonRow3D row;
      row.y = static_cast<int32_t>(y_dist(gen));
      row.z = static_cast<int32_t>(z_dist(gen));

      // Generate intervals by construction (non-overlapping)
      std::uniform_int_distribution<int> interval_count_dist(
        config.intervals_per_row_min,
        config.intervals_per_row_max
      );
      std::uniform_int_distribution<int> length_dist(
        config.interval_length_min,
        config.interval_length_max
      );
      std::uniform_int_distribution<int> gap_dist(config.gap_min, config.gap_max);

      int num_intervals = interval_count_dist(gen);

      // Random starting position
      std::uniform_int_distribution<int> start_dist(0, 1000);
      int current_x = start_dist(gen);

      for (int j = 0; j < num_intervals; ++j) {
        int length = length_dist(gen);
        DefaultInterval interval{static_cast<int32_t>(current_x), static_cast<int32_t>(current_x + length)};

        if (!interval.empty()) {
          row.intervals.push_back(interval);
        }

        int gap = gap_dist(gen);
        current_x += length + gap;
      }

      mesh.rows.push_back(std::move(row));
    }

    // Sort rows by (y, z) if requested
    if (config.sorted_rows) {
      std::sort(mesh.rows.begin(), mesh.rows.end());
    }

    return mesh;
  }

};

// ============================================================================
// Helper Functions for Common Mesh Operations
// ============================================================================

/**
 * @brief Check if two CommonMesh2D are equal (for testing)
 */
inline bool common_meshes_equal(const DefaultCommonMesh2D& a, const DefaultCommonMesh2D& b) {
  return a == b;
}

/**
 * @brief Check if two CommonMesh3D are equal (for testing)
 */
inline bool common_meshes_equal(const DefaultCommonMesh3D& a, const DefaultCommonMesh3D& b) {
  return a == b;
}

/**
 * @brief Validate that a CommonMesh2D satisfies invariants
 *
 * Invariants:
 * - Intervals in each row are sorted and non-overlapping
 * - Row keys are unique (if sorted_rows=true)
 * - All intervals are non-empty (begin < end)
 */
inline bool validate_common_mesh_2d(const DefaultCommonMesh2D& mesh, bool sorted_rows = true) {
  for (const auto& row : mesh.rows) {
    // Check intervals are sorted and non-overlapping
    for (size_t i = 0; i < row.intervals.size(); ++i) {
      const auto& interval = row.intervals[i];

      // Check non-empty
      if (interval.empty()) {
        return false;
      }

      // Check sorted
      if (i > 0 && row.intervals[i-1].end > interval.begin) {
        return false;  // Overlapping
      }
    }
  }

  // Check unique row keys if sorted
  if (sorted_rows && mesh.rows.size() > 1) {
    for (size_t i = 1; i < mesh.rows.size(); ++i) {
      if (mesh.rows[i-1].y >= mesh.rows[i].y) {
        return false;  // Not sorted or duplicate
      }
    }
  }

  return true;
}

/**
 * @brief Validate that a CommonMesh3D satisfies invariants
 */
inline bool validate_common_mesh_3d(const DefaultCommonMesh3D& mesh, bool sorted_rows = true) {
  for (const auto& row : mesh.rows) {
    // Check intervals are sorted and non-overlapping
    for (size_t i = 0; i < row.intervals.size(); ++i) {
      const auto& interval = row.intervals[i];

      // Check non-empty
      if (interval.empty()) {
        return false;
      }

      // Check sorted
      if (i > 0 && row.intervals[i-1].end > interval.begin) {
        return false;  // Overlapping
      }
    }
  }

  // Check unique row keys if sorted
  if (sorted_rows && mesh.rows.size() > 1) {
    for (size_t i = 1; i < mesh.rows.size(); ++i) {
      if (!(mesh.rows[i-1] < mesh.rows[i])) {
        return false;  // Not sorted or duplicate
      }
    }
  }

  return true;
}

/**
 * @brief Convert Mesh2DDevice to CommonMesh2D (convenience wrapper for v1)
 */
inline DefaultCommonMesh2D to_common_2d(const v1::Mesh2DDevice& device_mesh) {
  return v1_test::Converter2D<Kokkos::DefaultExecutionSpace::memory_space>::to_common(device_mesh);
}

/**
 * @brief Convert Mesh3DDevice to CommonMesh3D (convenience wrapper for v1)
 */
inline DefaultCommonMesh3D to_common_3d(const v1::Mesh3DDevice& device_mesh) {
  return v1_test::Converter3D<Kokkos::DefaultExecutionSpace::memory_space>::to_common(device_mesh);
}

/**
 * @brief Convert CommonMesh2D to Mesh2DDevice (convenience wrapper for v1)
 */
inline v1::Mesh2DDevice from_common_2d(const DefaultCommonMesh2D& common_mesh) {
  return v1_test::Converter2D<Kokkos::DefaultExecutionSpace::memory_space>::from_common(common_mesh);
}

/**
 * @brief Convert CommonMesh3D to Mesh3DDevice (convenience wrapper for v1)
 */
inline v1::Mesh3DDevice from_common_3d(const DefaultCommonMesh3D& common_mesh) {
  return v1_test::Converter3D<Kokkos::DefaultExecutionSpace::memory_space>::from_common(common_mesh);
}

// ============================================================================
// Compatibility wrappers for test code
// ============================================================================

/**
 * @brief Wrapper for v1::intersect_meshes<2> for backward compatibility
 * Returns CommonMesh2D for easy testing
 */
inline DefaultCommonMesh2D intersect_2d(const DefaultCommonMesh2D& a, const DefaultCommonMesh2D& b) {
  auto device_a = from_common_2d(a);
  auto device_b = from_common_2d(b);
  auto result = v1::intersect_meshes<2>(device_a, device_b);
  return to_common_2d(result);
}

/**
 * @brief Wrapper for v1::intersect_meshes<3> for backward compatibility
 * Returns CommonMesh3D for easy testing
 */
inline DefaultCommonMesh3D intersect_3d(const DefaultCommonMesh3D& a, const DefaultCommonMesh3D& b) {
  auto device_a = from_common_3d(a);
  auto device_b = from_common_3d(b);
  auto result = v1::intersect_meshes<3>(device_a, device_b);
  return to_common_3d(result);
}

/**
 * @brief Wrapper for v1::intersect_meshes<2> (explicit version name)
 */
inline DefaultCommonMesh2D intersect_v1_2d(const DefaultCommonMesh2D& a, const DefaultCommonMesh2D& b) {
  return intersect_2d(a, b);
}

/**
 * @brief Wrapper for v2::intersect_meshes<2> (explicit version name)
 */
inline DefaultCommonMesh2D intersect_v2_2d(const DefaultCommonMesh2D& a, const DefaultCommonMesh2D& b) {
  auto device_a = MeshConverter2D<v2::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(a);
  auto device_b = MeshConverter2D<v2::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(b);
  auto result = v2::intersect_meshes<2>(device_a, device_b);
  return MeshConverter2D<v2::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::to_common(result);
}

/**
 * @brief Wrapper for v3::intersect_meshes<2> (explicit version name)
 */
inline DefaultCommonMesh2D intersect_v3_2d(const DefaultCommonMesh2D& a, const DefaultCommonMesh2D& b) {
  auto device_a = MeshConverter2D<v3::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(a);
  auto device_b = MeshConverter2D<v3::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(b);
  auto result = v3::intersect_meshes<2>(device_a, device_b);
  return MeshConverter2D<v3::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::to_common(result);
}

/**
 * @brief Wrapper for v1::intersect_meshes<3> (explicit version name)
 */
inline DefaultCommonMesh3D intersect_v1_3d(const DefaultCommonMesh3D& a, const DefaultCommonMesh3D& b) {
  return intersect_3d(a, b);
}

/**
 * @brief Wrapper for v2::intersect_meshes<3> (explicit version name)
 */
inline DefaultCommonMesh3D intersect_v2_3d(const DefaultCommonMesh3D& a, const DefaultCommonMesh3D& b) {
  auto device_a = MeshConverter3D<v2::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(a);
  auto device_b = MeshConverter3D<v2::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(b);
  auto result = v2::intersect_meshes<3>(device_a, device_b);
  return MeshConverter3D<v2::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::to_common(result);
}

/**
 * @brief Wrapper for v3::intersect_meshes<3> (explicit version name)
 */
inline DefaultCommonMesh3D intersect_v3_3d(const DefaultCommonMesh3D& a, const DefaultCommonMesh3D& b) {
  auto device_a = MeshConverter3D<v3::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(a);
  auto device_b = MeshConverter3D<v3::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::from_common(b);
  auto result = v3::intersect_meshes<3>(device_a, device_b);
  return MeshConverter3D<v3::Mesh, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>::to_common(result);
}

} // namespace experimental::subsetix::csr::test

#endif
