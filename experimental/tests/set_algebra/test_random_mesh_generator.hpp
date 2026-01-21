// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include "test_common_format.hpp"
#include <Kokkos_Random.hpp>
#include <algorithm>
#include <random>

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
  static CommonMesh2D generate_2d(const RandomMeshConfig& config = RandomMeshConfig{}) {
    // Use std::mt19937 for host-side generation (simpler for test data)
    std::mt19937 gen(config.seed);

    // Generate number of rows
    std::uniform_int_distribution<int> rows_dist(config.num_rows_min, config.num_rows_max);
    int num_rows = rows_dist(gen);

    CommonMesh2D mesh;
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
      CommonRow2D row;
      row.y = y;

      int num_intervals = interval_count_dist(gen);

      // Generate non-overlapping intervals by construction
      // Start at a random position
      std::uniform_int_distribution<int> start_dist(0, 1000);
      int current_x = start_dist(gen);

      for (int j = 0; j < num_intervals; ++j) {
        int length = length_dist(gen);
        Interval interval{current_x, current_x + length};

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
  static CommonMesh3D generate_3d(const RandomMeshConfig& config = RandomMeshConfig{}) {
    // Use std::mt19937 for host-side generation
    std::mt19937 gen(config.seed);

    // Generate number of rows
    std::uniform_int_distribution<int> rows_dist(config.num_rows_min, config.num_rows_max);
    int num_rows = rows_dist(gen);

    CommonMesh3D mesh;
    mesh.rows.reserve(num_rows);

    // Generate (Y, Z) coordinates
    std::uniform_int_distribution<int> y_dist(config.y_min, config.y_max);
    std::uniform_int_distribution<int> z_dist(config.z_min, config.z_max);

    for (int i = 0; i < num_rows; ++i) {
      CommonRow3D row;
      row.y = y_dist(gen);
      row.z = z_dist(gen);

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
        Interval interval{current_x, current_x + length};

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
inline bool common_meshes_equal(const CommonMesh2D& a, const CommonMesh2D& b) {
  return a == b;
}

/**
 * @brief Check if two CommonMesh3D are equal (for testing)
 */
inline bool common_meshes_equal(const CommonMesh3D& a, const CommonMesh3D& b) {
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
inline bool validate_common_mesh_2d(const CommonMesh2D& mesh, bool sorted_rows = true) {
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
inline bool validate_common_mesh_3d(const CommonMesh3D& mesh, bool sorted_rows = true) {
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
 * @brief Convert Mesh2DDevice to CommonMesh2D (convenience wrapper)
 */
inline CommonMesh2D to_common_2d(const Mesh2DDevice& device_mesh) {
  return MeshConverter2D<Kokkos::DefaultExecutionSpace::memory_space>::to_common(device_mesh);
}

/**
 * @brief Convert Mesh3DDevice to CommonMesh3D (convenience wrapper)
 */
inline CommonMesh3D to_common_3d(const Mesh3DDevice& device_mesh) {
  return MeshConverter3D<Kokkos::DefaultExecutionSpace::memory_space>::to_common(device_mesh);
}

/**
 * @brief Convert CommonMesh2D to Mesh2DDevice (convenience wrapper)
 */
inline Mesh2DDevice from_common_2d(const CommonMesh2D& common_mesh) {
  return MeshConverter2D<Kokkos::DefaultExecutionSpace::memory_space>::from_common(common_mesh);
}

/**
 * @brief Convert CommonMesh3D to Mesh3DDevice (convenience wrapper)
 */
inline Mesh3DDevice from_common_3d(const CommonMesh3D& common_mesh) {
  return MeshConverter3D<Kokkos::DefaultExecutionSpace::memory_space>::from_common(common_mesh);
}

} // namespace experimental::subsetix::csr::test

#endif
