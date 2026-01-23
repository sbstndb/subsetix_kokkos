// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

/**
 * @file successive_intersection.hpp
 * @brief Unified API for successive intersection of multiple CSR meshes.
 *
 * This module provides three distinct strategies for computing the intersection
 * of N meshes (N >= 2):
 *
 * 1. **Naive Strategy**: Iteratively intersects meshes pairwise (m1 ∩ m2, then
 *    result ∩ m3, etc.). Simple but creates many intermediate meshes.
 *
 * 2. **Workspace Strategy**: Allocates reusable workspaces to minimize
 *    allocations. Better memory efficiency than naive for large N.
 *
 * 3. **Graph Strategy**: Analyzes mesh overlap patterns to build a directed
 *    acyclic graph (DAG) and computes intersections in optimal order. Most
 *    sophisticated, potentially fastest for sparse overlaps.
 *
 * All strategies are dimension-agnostic (support both 2D and 3D meshes) and
 * memory-space-agnostic (work with HostSpace or device memory spaces).
 *
 * @tparam DIM Dimension (2 for 2D meshes, 3 for 3D meshes)
 *
 * Example usage:
 * @code
 *   #include <experimental/subsetix/csr/set_algebra/successive_intersection.hpp>
 *
 *   using namespace experimental::subsetix::csr::successive;
 *
 *   // Create input meshes (all must be in the same memory space)
 *   std::vector<Mesh2D<Kokkos::HostSpace>> meshes = {mesh1, mesh2, mesh3, mesh4};
 *
 *   // Use default strategy (Naive)
 *   auto result = intersect_successive(meshes);
 *
 *   // Use Workspace strategy with custom configuration
 *   Config<2> cfg;
 *   cfg.strategy = Strategy::Workspace;
 *   cfg.workspace.max_rows = 10000;
 *   cfg.workspace.max_intervals = 50000;
 *   cfg.workspace.growth_factor = 2.0;
 *   auto result2 = intersect_successive(meshes, cfg);
 *
 *   // Use Graph strategy (best for sparse overlaps)
 *   Config<2> cfg3;
 *   cfg3.strategy = Strategy::Graph;
 *   auto result3 = intersect_successive(meshes, cfg3);
 * @endcode
 */

#include <experimental/subsetix/csr/mesh.hpp>
#include <experimental/subsetix/csr/set_algebra/v1.hpp>
#include <vector>
#include <cstddef>

// Include implementation headers
#include <experimental/subsetix/csr/set_algebra/detail/successive_intersection_naive.hpp>
#include <experimental/subsetix/csr/set_algebra/detail/successive_intersection_workspace.hpp>
#include <experimental/subsetix/csr/set_algebra/detail/successive_intersection_graph.hpp>

namespace experimental::subsetix::csr::successive {

// ============================================================================
// Strategy enumeration
// ============================================================================

/**
 * @brief Available strategies for successive intersection.
 *
 * Each strategy represents a different algorithmic approach with different
 * performance characteristics:
 *
 * - **Naive**: Simplest, easiest to understand, good for small N or debugging
 * - **Workspace**: Better memory efficiency, good for medium N (3-10 meshes)
 * - **Graph**: Most sophisticated, best for large N with sparse overlaps
 */
enum class Strategy {
  /**
   * Naive pairwise intersection strategy.
   *
   * Algorithm:
   * 1. result = meshes[0]
   * 2. For each subsequent mesh m: result = intersect(result, m)
   *
   * Pros:
   * - Simple implementation, easy to debug
   * - No analysis overhead
   * - Minimal memory footprint (only 2 meshes active at once)
   *
   * Cons:
   * - Creates N-1 intermediate meshes
   * - No optimization based on overlap patterns
   * - May intersect large meshes unnecessarily
   *
   * Best for: Small N (< 5), debugging, or when all meshes have similar sizes.
   */
  Naive,

  /**
   * Workspace reuse strategy.
   *
   * Algorithm:
   * 1. Allocate reusable workspaces (row_keys, row_ptr, intervals)
   * 2. Iteratively intersect, reusing workspaces to avoid allocations
   * 3. Grow workspaces as needed (using growth_factor)
   *
   * Pros:
   * - Minimizes memory allocations
   * - Better cache locality than naive
   * - Configurable workspace sizes
   *
   * Cons:
   * - Still creates N-1 intermediate meshes
   * - Requires estimating workspace sizes upfront
   * - Over-allocation wastes memory if estimates are too high
   *
   * Best for: Medium N (5-10 meshes), when memory allocation is a bottleneck.
   */
  Workspace,

  /**
   * Graph-based DAG strategy.
   *
   * Algorithm:
   * 1. Analyze pairwise overlaps between all meshes
   * 2. Build a DAG where edges represent "depends on" relationships
   * 3. Topologically sort to find optimal intersection order
   * 4. Compute intersections in parallel where possible
   *
   * Pros:
   * - Can skip unnecessary intersections
   * - Potential for parallel execution
   * - Optimal order for sparse overlaps
   *
   * Cons:
   * - O(N^2) overlap analysis phase
   * - More complex implementation
   * - Higher overhead for small N
   *
   * Best for: Large N (> 10) with sparse overlap patterns.
   */
  Graph
};

// ============================================================================
// Configuration structures
// ============================================================================

/**
 * @brief Configuration for the Workspace strategy.
 *
 * Controls the size and growth behavior of reusable workspaces.
 */
struct WorkspaceConfig {
  /**
   * Maximum number of rows to pre-allocate in workspace.
   *
   * Set to 0 to auto-detect from the first mesh (default behavior).
   */
  std::size_t max_rows = 0;

  /**
   * Maximum number of intervals to pre-allocate in workspace.
   *
   * Set to 0 to auto-detect from the first mesh (default behavior).
   */
  std::size_t max_intervals = 0;

  /**
   * Factor by which to grow workspace when capacity is exceeded.
   *
   * Typical values: 1.5 (conservative), 2.0 (aggressive).
   * Must be >= 1.0.
   */
  double growth_factor = 1.5;
};

/**
 * @brief Configuration for the Graph strategy.
 *
 * Controls the overlap analysis and DAG construction.
 */
struct GraphConfig {
  /**
   * Threshold for considering two meshes "overlapping".
   *
   * Two meshes are considered overlapping if their bounding boxes
   * overlap by at least this fraction (0.0 to 1.0).
   *
   * Lower values = more conservative (more edges in DAG)
   * Higher values = more aggressive (fewer edges, potential skipping)
   */
  double overlap_threshold = 0.0;

  /**
   * Whether to enable parallel intersection of independent branches.
   *
   * When true, independent branches of the DAG can be processed in parallel.
   * Requires OpenMP or CUDA backend.
   */
  bool enable_parallel = true;
};

/**
 * @brief Main configuration for successive intersection.
 *
 * @tparam DIM Dimension (2 or 3)
 *
 * This struct combines strategy selection with strategy-specific
 * configuration options.
 */
template<int DIM>
struct Config {
  /**
   * Which strategy to use for successive intersection.
   *
   * Default: Strategy::Naive (simplest, works well for small N)
   */
  Strategy strategy = Strategy::Naive;

  /**
   * Configuration for the Workspace strategy.
   *
   * Only used when strategy == Strategy::Workspace.
   */
  WorkspaceConfig workspace;

  /**
   * Configuration for the Graph strategy.
   *
   * Only used when strategy == Strategy::Graph.
   */
  GraphConfig graph;

  /**
   * Validate configuration parameters.
   *
   * @throws std::invalid_argument if configuration is invalid
   */
  void validate() const {
    if (workspace.growth_factor < 1.0) {
      throw std::invalid_argument("workspace.growth_factor must be >= 1.0");
    }
    if (graph.overlap_threshold < 0.0 || graph.overlap_threshold > 1.0) {
      throw std::invalid_argument("graph.overlap_threshold must be in [0.0, 1.0]");
    }
  }

  /**
   * Create a default configuration for a given strategy.
   *
   * @param s The strategy to configure
   * @return Config with defaults optimized for the chosen strategy
   */
  static Config for_strategy(Strategy s) {
    Config cfg;
    cfg.strategy = s;

    // Set strategy-specific defaults
    switch (s) {
      case Strategy::Workspace:
        cfg.workspace.max_rows = 0;      // Auto-detect
        cfg.workspace.max_intervals = 0; // Auto-detect
        cfg.workspace.growth_factor = 1.5;
        break;

      case Strategy::Graph:
        cfg.graph.overlap_threshold = 0.0;
        cfg.graph.enable_parallel = true;
        break;

      case Strategy::Naive:
      default:
        // Naive uses no additional config
        break;
    }

    return cfg;
  }
};

// ============================================================================
// Main API function
// ============================================================================

/**
 * @brief Compute the successive intersection of multiple meshes.
 *
 * This function computes m1 ∩ m2 ∩ m3 ∩ ... ∩ mN using the specified
 * strategy. All input meshes must have the same dimension and memory space.
 *
 * @tparam DIM Dimension (2 or 3)
 * @tparam MemorySpace Kokkos memory space (deduced from input meshes)
 *
 * @param meshes Vector of input meshes (must contain at least 2 meshes)
 * @param config Configuration controlling the intersection strategy
 *
 * @return Mesh containing the successive intersection
 *
 * @pre meshes.size() >= 2
 * @pre All meshes have the same DIM template parameter
 * @pre All meshes use the same MemorySpace
 * @pre config.validate() passes (no invalid parameters)
 *
 * @post The returned mesh contains only cells present in ALL input meshes
 * @post The returned mesh uses the same MemorySpace as input meshes
 *
 * @throws std::invalid_argument if meshes.size() < 2
 * @throws std::invalid_argument if configuration is invalid
 * @throws std::runtime_error if meshes have inconsistent memory spaces
 *
 * Example:
 * @code
 *   std::vector<Mesh2D<Kokkos::HostSpace>> meshes = {m1, m2, m3};
 *   auto result = intersect_successive<2>(meshes);
 * @endcode
 */
template<int DIM, class MemorySpace, class CoordType = int32_t, class IndexType = std::size_t>
v3::Mesh<DIM, MemorySpace, CoordType, IndexType> intersect_successive(
    const std::vector<v3::Mesh<DIM, MemorySpace, CoordType, IndexType>>& meshes,
    const Config<DIM>& config = {}) {

  // Validate inputs
  if (meshes.size() < 2) {
    throw std::invalid_argument(
        "intersect_successive requires at least 2 meshes, got " +
        std::to_string(meshes.size()));
  }

  // Validate configuration
  config.validate();

  // Dispatch to appropriate implementation based on strategy
  switch (config.strategy) {
    case Strategy::Naive:
      return naive::intersect<DIM, MemorySpace, CoordType, IndexType>(meshes);

    case Strategy::Workspace: {
      typename workspace::IntersectionWorkspace<DIM, MemorySpace> ws;
      return workspace::intersect_successive<DIM, MemorySpace, CoordType, IndexType>(meshes, ws);
    }

    case Strategy::Graph:
      return graph::successive_intersection<DIM, CoordType, IndexType>(meshes);

    default:
      throw std::invalid_argument("Unknown strategy");
  }
}

// ============================================================================
// Convenience wrappers for common use cases
// ============================================================================

/**
 * @brief Convenience function: naive intersection with default config.
 *
 * This is the simplest API for users who just want "intersection of all meshes"
 * without worrying about strategies or configuration.
 *
 * @tparam DIM Dimension (2 or 3)
 * @tparam MemorySpace Kokkos memory space (deduced from input)
 * @tparam CoordType Coordinate type (deduced from input)
 * @tparam IndexType Index type (deduced from input)
 *
 * @param meshes Vector of input meshes (at least 2)
 *
 * Example:
 * @code
 *   auto result = intersect_all<2>(meshes);
 * @endcode
 */
template<int DIM, class MemorySpace, class CoordType = int32_t, class IndexType = std::size_t>
v3::Mesh<DIM, MemorySpace, CoordType, IndexType> intersect_all(
    const std::vector<v3::Mesh<DIM, MemorySpace, CoordType, IndexType>>& meshes) {
  return intersect_successive<DIM, MemorySpace, CoordType, IndexType>(meshes, Config<DIM>::for_strategy(Strategy::Naive));
}

/**
 * @brief Convenience function: workspace-optimized intersection.
 *
 * Uses the Workspace strategy with auto-detected workspace sizes.
 *
 * @tparam DIM Dimension (2 or 3)
 * @tparam MemorySpace Kokkos memory space (deduced from input)
 * @tparam CoordType Coordinate type (deduced from input)
 * @tparam IndexType Index type (deduced from input)
 *
 * @param meshes Vector of input meshes (at least 2)
 * @param growth_factor Workspace growth factor (default 1.5)
 *
 * Example:
 * @code
 *   auto result = intersect_all_workspace<2>(meshes, 2.0);
 * @endcode
 */
template<int DIM, class MemorySpace, class CoordType = int32_t, class IndexType = std::size_t>
v3::Mesh<DIM, MemorySpace, CoordType, IndexType> intersect_all_workspace(
    const std::vector<v3::Mesh<DIM, MemorySpace, CoordType, IndexType>>& meshes,
    double growth_factor = 1.5) {

  Config<DIM> cfg = Config<DIM>::for_strategy(Strategy::Workspace);
  cfg.workspace.growth_factor = growth_factor;
  return intersect_successive<DIM, MemorySpace, CoordType, IndexType>(meshes, cfg);
}

/**
 * @brief Convenience function: graph-based intersection.
 *
 * Uses the Graph strategy for optimal intersection ordering.
 *
 * @tparam DIM Dimension (2 or 3)
 * @tparam MemorySpace Kokkos memory space (deduced from input)
 * @tparam CoordType Coordinate type (deduced from input)
 * @tparam IndexType Index type (deduced from input)
 *
 * @param meshes Vector of input meshes (at least 2)
 * @param overlap_threshold Overlap threshold for DAG construction (default 0.0)
 *
 * Example:
 * @code
 *   auto result = intersect_all_graph<2>(meshes, 0.1);
 * @endcode
 */
template<int DIM, class MemorySpace, class CoordType = int32_t, class IndexType = std::size_t>
v3::Mesh<DIM, MemorySpace, CoordType, IndexType> intersect_all_graph(
    const std::vector<v3::Mesh<DIM, MemorySpace, CoordType, IndexType>>& meshes,
    double overlap_threshold = 0.0) {

  Config<DIM> cfg = Config<DIM>::for_strategy(Strategy::Graph);
  cfg.graph.overlap_threshold = overlap_threshold;
  return intersect_successive<DIM, MemorySpace, CoordType, IndexType>(meshes, cfg);
}

} // namespace experimental::subsetix::csr::successive
