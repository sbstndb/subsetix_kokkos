// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

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
 *
 * Sparsity-based row count:
 * - 2D: num_rows = round(sparsity * (y_max - y_min))
 * - 3D: num_rows = round(sparsity * (y_max - y_min) * (z_max - z_min))
 *
 * The sparsity parameter represents the fraction of grid positions that are occupied.
 * A sparsity of 1.0 means 100% dense (all grid positions), while 0.3 means 30% dense.
 *
 * Row keys (Y in 2D, (Y,Z) in 3D) are guaranteed to be unique through Fisher-Yates sampling.
 */
struct RandomMeshConfig {
  int seed = 42;                      // Random seed for reproducibility
  double sparsity = 0.3;              // Fraction of grid positions [0.0, 1.0]
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
 * - Sparsity: 30%
 * - 2D rows: ~19 (64 × 0.3)
 * - 3D rows: ~1229 (64 × 64 × 0.3)
 * - Max intervals per row: 4
 */
inline RandomMeshConfig SmallConfig() {
  return RandomMeshConfig{
    .seed = 42,
    .sparsity = 0.3,
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
 * - Sparsity: 30%
 * - 2D rows: ~154 (512 × 0.3)
 * - 3D rows: ~78643 (512 × 512 × 0.3)
 * - Max intervals per row: 4
 */
inline RandomMeshConfig MediumConfig() {
  return RandomMeshConfig{
    .seed = 42,
    .sparsity = 0.3,
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
 * - Sparsity: 30%
 * - 2D rows: ~1229 (4096 × 0.3)
 * - 3D rows: ~5.0M (4096 × 4096 × 0.3)
 * - Max intervals per row: 4
 */
inline RandomMeshConfig LargeConfig() {
  return RandomMeshConfig{
    .seed = 42,
    .sparsity = 0.3,
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
 * - Sparsity: 15% (reduced from 30% to manage memory)
 * - 2D rows: ~1229 (8192 × 0.15)
 * - 3D rows: ~10M (8192 × 8192 × 0.15)
 * - Max intervals per row: 4
 *
 * Memory estimate:
 * - 2D: ~5K intervals × 16 bytes = ~80 KB input data
 * - 3D: ~10M intervals × 24 bytes = ~240 MB input data
 */
inline RandomMeshConfig ExtraLargeConfig() {
  return RandomMeshConfig{
    .seed = 42,
    .sparsity = 0.15,
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
 * The parameters work as follows:
 * - num_rows_2d: Direct number of rows for 2D (y = 0, 1, ..., num_rows_2d-1)
 * - grid_size_3d: Size of ONE SIDE of the 3D grid (like y_max/z_max in random config)
 *                Total rows = grid_size_3d * grid_size_3d
 *
 * This matches the random config semantics where y_max/z_max define the grid extent.
 */
struct RegularMeshConfig {
  int num_rows_2d = 1250;   // Number of rows for 2D (y = 0, 1, ..., num_rows_2d-1)
  int grid_size_3d = 64;    // Grid size for 3D (total rows = grid_size_3d²)
};

/**
 * @brief Small regular mesh configuration (100% dense)
 *
 * Matches the random SmallConfig grid extent but with 100% density:
 * - 2D: 64 rows (100% of y_max=64, random has ~19 at 30% sparsity)
 * - 3D: grid_size=64 → 64×64=4096 rows (100% of random's y_max=64, z_max=64 range,
 *   random has ~1229 at 30% sparsity)
 *
 * The regular mesh has more rows than random (100% vs 30% density), but with NO sparsity.
 */
inline RegularMeshConfig SmallRegularConfig() {
  return RegularMeshConfig{.num_rows_2d = 64, .grid_size_3d = 64};
}

/**
 * @brief Medium regular mesh configuration (100% dense)
 *
 * Matches the random MediumConfig grid extent but with 100% density:
 * - 2D: 512 rows (100% of y_max=512, random has ~154 at 30% sparsity)
 * - 3D: grid_size=512 → 512×512=262144 rows (100% of random's y_max=512, z_max=512 range,
 *   random has ~78643 at 30% sparsity)
 *
 * The regular mesh has more rows than random (100% vs 30% density), but with NO sparsity.
 */
inline RegularMeshConfig MediumRegularConfig() {
  return RegularMeshConfig{.num_rows_2d = 512, .grid_size_3d = 512};
}

/**
 * @brief Large regular mesh configuration (100% dense)
 *
 * Matches the random LargeConfig grid extent but with 100% density:
 * - 2D: 4096 rows (100% of y_max=4096, random has ~1229 at 30% sparsity)
 * - 3D: grid_size=4096 → 4096×4096=16.8M rows (100% of random's y_max=4096, z_max=4096 range,
 *   random has ~5.0M at 30% sparsity)
 *
 * The regular mesh has more rows than random (100% vs 30% density), but with NO sparsity.
 */
inline RegularMeshConfig LargeRegularConfig() {
  return RegularMeshConfig{.num_rows_2d = 4096, .grid_size_3d = 4096};
}

/**
 * @brief Extra Large regular mesh configuration (100% dense)
 *
 * Matches the random ExtraLargeConfig grid extent but with 100% density:
 * - 2D: 8192 rows (100% of y_max=8192, random has ~1229 at 15% sparsity)
 * - 3D: grid_size=8192 → 8192×8192=67M rows (100% of random's y_max=8192, z_max=8192 range,
 *   random has ~10M at 15% sparsity)
 *
 * WARNING: 3D ExtraLarge requires >512MB GPU memory and will crash on most GPUs.
 * - Use 3D ExtraLarge for CPU benchmarking only
 * - 2D ExtraLarge is safe for GPU (8192 rows fits easily in GPU memory)
 *
 * NOTE: For CUDA, 3D benchmarks are limited to Large (16.8M rows) to avoid OOM.
 */
inline RegularMeshConfig ExtraLargeRegularConfig() {
  return RegularMeshConfig{.num_rows_2d = 8192, .grid_size_3d = 8192};
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
 *
 * Note: Regular meshes are generated deterministically with guaranteed
 * correctness (sorted rows, non-empty intervals, no overlaps), so
 * validation is not required unlike random meshes.
 */
class RegularMeshGenerator {
public:
  /**
   * @brief Generate a regular CommonMesh2D (100% dense)
   *
   * Creates a mesh where:
   * - y = 0, 1, 2, ..., num_rows_2d-1 (ALL rows present, no gaps)
   * - Each row has one interval [0, num_rows_2d) covering the full X range
   *
   * The generated mesh is guaranteed to be valid (sorted, non-overlapping).
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
      // Single interval covering the entire X range [0, num_rows_2d)
      row.intervals.push_back(DefaultInterval{0, static_cast<int32_t>(config.num_rows_2d)});
      mesh.rows.push_back(std::move(row));
    }

    return mesh;
  }

  /**
   * @brief Generate a regular CommonMesh3D (100% dense cube)
   *
   * Creates a mesh where all (y,z) combinations in a square grid are present:
   * - (y, z) = (0,0), (0,1), ..., (0, grid_size-1), (1,0), ..., (grid_size-1, grid_size-1)
   * - Total rows: grid_size_3d × grid_size_3d (perfect square grid)
   * - Each row has one interval [0, grid_size_3d) covering the full X range
   *
   * @param config Configuration containing grid_size_3d (side length of the grid)
   * @return Regular CommonMesh3D with grid_size_3d² rows
   */
  static DefaultCommonMesh3D generate_3d(const RegularMeshConfig& config = RegularMeshConfig{}) {
    DefaultCommonMesh3D mesh;
    int grid_size = config.grid_size_3d;
    mesh.rows.reserve(static_cast<std::size_t>(grid_size) * static_cast<std::size_t>(grid_size));

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
 * @brief Random mesh generator with sparsity-based unique row sampling
 *
 * Generates random CommonMesh2D/CommonMesh3D with configurable parameters.
 * Uses Fisher-Yates shuffle for unique row key sampling, ensuring no duplicates.
 *
 * The sparsity parameter determines what fraction of grid positions are occupied:
 * - 2D: num_rows = round(sparsity * (y_max - y_min))
 * - 3D: num_rows = round(sparsity * (y_max - y_min) * (z_max - z_min))
 *
 * Row keys (Y in 2D, (Y,Z) in 3D) are guaranteed to be unique.
 */
class RandomMeshGenerator {
private:
  /**
   * @brief Calculate number of rows from sparsity (2D)
   */
  static inline int calculate_num_rows_2d(const RandomMeshConfig& config) {
    int y_extent = config.y_max - config.y_min;
    int num_rows = static_cast<int>(std::round(config.sparsity * y_extent));
    return std::max(0, num_rows);  // Allow 0 rows for sparsity=0
  }

  /**
   * @brief Calculate number of rows from sparsity (3D)
   */
  static inline int calculate_num_rows_3d(const RandomMeshConfig& config) {
    int y_extent = config.y_max - config.y_min;
    int z_extent = config.z_max - config.z_min;
    long long grid_positions = static_cast<long long>(y_extent) * static_cast<long long>(z_extent);
    long long num_rows = static_cast<long long>(std::round(config.sparsity * grid_positions));
    return static_cast<int>(std::max(0LL, num_rows));  // Allow 0 rows for sparsity=0
  }

  /**
   * @brief Generate unique Y coordinates using partial Fisher-Yates shuffle (2D)
   *
   * This algorithm guarantees uniqueness without needing a set to track seen values.
   * It performs a partial Fisher-Yates shuffle, only shuffling the first N elements.
   */
  static inline std::vector<int> generate_unique_y_coords(const RandomMeshConfig& config, int num_rows) {
    // Generate all possible Y coordinates
    int grid_extent = config.y_max - config.y_min;
    std::vector<int> all_coords(grid_extent);
    std::iota(all_coords.begin(), all_coords.end(), config.y_min);

    // Handle edge case: num_rows > grid_extent (clamp to grid extent)
    int actual_rows = std::min(num_rows, grid_extent);
    if (actual_rows <= 0) {
      return {};  // Empty mesh for sparsity = 0
    }

    // Partial Fisher-Yates shuffle: only shuffle what we need
    std::mt19937 gen(config.seed);
    for (int i = 0; i < actual_rows; ++i) {
      std::uniform_int_distribution<int> dist(i, grid_extent - 1);
      int j = dist(gen);
      std::swap(all_coords[i], all_coords[j]);
    }

    // Keep only selected coordinates
    all_coords.resize(actual_rows);

    // Sort if requested
    if (config.sorted_rows) {
      std::sort(all_coords.begin(), all_coords.end());
    }

    return all_coords;
  }

  /**
   * @brief Generate unique (Y, Z) coordinate pairs using partial Fisher-Yates shuffle (3D)
   */
  static inline std::vector<std::pair<int, int>> generate_unique_yz_pairs(
      const RandomMeshConfig& config, int num_rows) {
    // Generate all possible (Y, Z) pairs in lexicographic order
    int y_extent = config.y_max - config.y_min;
    int z_extent = config.z_max - config.z_min;
    long long grid_positions = static_cast<long long>(y_extent) * static_cast<long long>(z_extent);

    // Handle edge case
    long long actual_rows = std::min(static_cast<long long>(num_rows), grid_positions);
    if (actual_rows <= 0) {
      return {};  // Empty mesh for sparsity = 0
    }

    std::vector<std::pair<int, int>> all_pairs;
    all_pairs.reserve(static_cast<std::size_t>(grid_positions));

    for (int z = config.z_min; z < config.z_max; ++z) {
      for (int y = config.y_min; y < config.y_max; ++y) {
        all_pairs.emplace_back(y, z);
      }
    }

    // Partial Fisher-Yates shuffle
    std::mt19937 gen(config.seed);
    for (long long i = 0; i < actual_rows; ++i) {
      std::uniform_int_distribution<long long> dist(i, grid_positions - 1);
      long long j = dist(gen);
      std::swap(all_pairs[static_cast<std::size_t>(i)], all_pairs[static_cast<std::size_t>(j)]);
    }

    // Keep only selected pairs
    all_pairs.resize(static_cast<std::size_t>(actual_rows));

    // Sort lexicographically if requested
    if (config.sorted_rows) {
      std::sort(all_pairs.begin(), all_pairs.end());
    }

    return all_pairs;
  }

public:
  /**
   * @brief Generate a random CommonMesh2D with unique Y coordinates
   *
   * Y coordinates are sampled without replacement using Fisher-Yates shuffle,
   * guaranteeing no duplicate row keys.
   *
   * @param config Configuration for generation
   * @return Random CommonMesh2D
   */
  static DefaultCommonMesh2D generate_2d(const RandomMeshConfig& config = RandomMeshConfig{}) {
    // Calculate number of rows from sparsity
    int num_rows = calculate_num_rows_2d(config);

    DefaultCommonMesh2D mesh;

    // Handle empty mesh (sparsity = 0)
    if (num_rows <= 0) {
      return mesh;
    }

    // Generate unique Y coordinates using Fisher-Yates
    std::vector<int> y_coords = generate_unique_y_coords(config, num_rows);

    mesh.rows.reserve(y_coords.size());

    // Setup random generators for intervals
    std::mt19937 gen(config.seed + 1);  // Different seed for intervals
    std::uniform_int_distribution<int> interval_count_dist(
      config.intervals_per_row_min,
      config.intervals_per_row_max
    );
    std::uniform_int_distribution<int> length_dist(
      config.interval_length_min,
      config.interval_length_max
    );
    std::uniform_int_distribution<int> gap_dist(config.gap_min, config.gap_max);

    // Generate intervals for each row
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
   * @brief Generate a random CommonMesh3D with unique (Y, Z) coordinate pairs
   *
   * (Y, Z) coordinate pairs are sampled without replacement using Fisher-Yates shuffle,
   * guaranteeing no duplicate row keys.
   *
   * @param config Configuration for generation
   * @return Random CommonMesh3D
   */
  static DefaultCommonMesh3D generate_3d(const RandomMeshConfig& config = RandomMeshConfig{}) {
    // Calculate number of rows from sparsity
    int num_rows = calculate_num_rows_3d(config);

    DefaultCommonMesh3D mesh;

    // Handle empty mesh (sparsity = 0)
    if (num_rows <= 0) {
      return mesh;
    }

    // Generate unique (Y, Z) pairs using Fisher-Yates
    std::vector<std::pair<int, int>> yz_pairs = generate_unique_yz_pairs(config, num_rows);

    mesh.rows.reserve(yz_pairs.size());

    // Setup random generators for intervals
    std::mt19937 gen(config.seed + 1);  // Different seed for intervals
    std::uniform_int_distribution<int> interval_count_dist(
      config.intervals_per_row_min,
      config.intervals_per_row_max
    );
    std::uniform_int_distribution<int> length_dist(
      config.interval_length_min,
      config.interval_length_max
    );
    std::uniform_int_distribution<int> gap_dist(config.gap_min, config.gap_max);

    // Generate intervals for each row
    for (const auto& [y, z] : yz_pairs) {
      DefaultCommonRow3D row;
      row.y = static_cast<int32_t>(y);
      row.z = static_cast<int32_t>(z);

      int num_intervals = interval_count_dist(gen);

      // Generate non-overlapping intervals by construction
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

    // Rows are already sorted if requested (in generate_unique_yz_pairs)
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
