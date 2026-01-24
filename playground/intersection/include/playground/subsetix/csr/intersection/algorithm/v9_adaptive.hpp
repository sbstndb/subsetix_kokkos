// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#include <playground/subsetix/csr/intersection/algorithm/optimized.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v4_hash.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v5_parallel_merge.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v6_direct_index.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v7_soa_optimized.hpp>
#include <playground/subsetix/csr/intersection/algorithm/v8_hybrid_cpu_gpu.hpp>

#include <chrono>
#include <algorithm>

namespace playground::subsetix::csr::intersection::adaptive {

// ============================================================================
// Strategy Enumeration
// ============================================================================

/**
 * @brief Available row mapping strategies for adaptive selection.
 *
 * Each strategy corresponds to a specific algorithm implementation:
 * - OPTIMIZED: v2 baseline - safe default for small meshes
 * - HASH_BASED: v4 - hash table based mapping
 * - PARALLEL_MERGE: v5 - parallel merge algorithm
 * - DIRECT_INDEX: v6 - O(1) direct indexing for dense meshes
 * - SOA_OPTIMIZED: v7 - Structure of Arrays optimization
 * - HYBRID_CPU_GPU: v8 - Hybrid CPU-GPU algorithm
 */
enum class RowMappingStrategy {
  OPTIMIZED,          // v2 baseline
  HASH_BASED,         // v4
  PARALLEL_MERGE,     // v5
  DIRECT_INDEX,       // v6
  SOA_OPTIMIZED,      // v7
  HYBRID_CPU_GPU      // v8
};

/**
 * @brief Convert strategy enum to string for debugging.
 */
inline const char* strategy_to_string(RowMappingStrategy strategy) {
  switch (strategy) {
    case RowMappingStrategy::OPTIMIZED: return "OPTIMIZED (v2 baseline)";
    case RowMappingStrategy::HASH_BASED: return "HASH_BASED (v4)";
    case RowMappingStrategy::PARALLEL_MERGE: return "PARALLEL_MERGE (v5)";
    case RowMappingStrategy::DIRECT_INDEX: return "DIRECT_INDEX (v6)";
    case RowMappingStrategy::SOA_OPTIMIZED: return "SOA_OPTIMIZED (v7)";
    case RowMappingStrategy::HYBRID_CPU_GPU: return "HYBRID_CPU_GPU (v8)";
    default: return "UNKNOWN";
  }
}

// ============================================================================
// Selection Metrics (Optional Profiling)
// ============================================================================

/**
 * @brief Metrics collected during strategy selection.
 *
 * This struct can be used for profiling and analyzing the adaptive
 * selector's decisions. Pass a pointer to intersect_meshes() to fill it.
 */
struct SelectionMetrics {
  RowMappingStrategy strategy;
  std::size_t num_rows_a;
  std::size_t num_rows_b;
  double density_b;
  bool is_uniform;
  double selection_time_ms;
};

// ============================================================================
// Pattern Detection Helpers
// ============================================================================

/**
 * @brief Detect if mesh B has uniform stride (regular spacing).
 *
 * Samples first ~100 rows to check for constant y-gap, then verifies
 * the stride covers the full range correctly.
 *
 * @tparam RowKeyView Type of row_keys view
 * @tparam CoordType Coordinate type
 * @param rows_b Row keys view (device memory)
 * @param num_rows_b Number of rows in B
 * @param out_stride Output parameter for detected stride
 * @return true if uniform stride detected
 */
template <class RowKeyView, class CoordType>
bool detect_uniform_stride(const RowKeyView& rows_b,
                           std::size_t num_rows_b,
                           CoordType& out_stride) {
  if (num_rows_b < 2) return false;

  // Copy first ~100 rows to host for sampling
  constexpr std::size_t SAMPLE_SIZE = 100;
  const std::size_t sample = std::min(SAMPLE_SIZE, num_rows_b);

  auto host_rows = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, rows_b);

  // Check stride consistency in sample
  CoordType stride = host_rows(1).y - host_rows(0).y;
  if (stride <= 0) return false;

  for (std::size_t i = 2; i < sample; ++i) {
    const CoordType gap = host_rows(i).y - host_rows(i - 1).y;
    if (gap != stride) return false;
  }

  // Verify stride covers full range
  const CoordType y_min = host_rows(0).y;
  const CoordType y_max = host_rows(num_rows_b - 1).y;
  const std::size_t y_range = y_max - y_min + 1;

  if ((y_max - y_min) % stride != 0) return false;

  const std::size_t expected = (y_range + stride - 1) / stride;
  if (expected != num_rows_b) return false;

  out_stride = stride;
  return true;
}

/**
 * @brief Compute mesh density (fraction of Y coordinates occupied).
 *
 * @tparam CoordType Coordinate type
 * @param num_rows Number of rows in mesh
 * @param y_min Minimum Y coordinate
 * @param y_max Maximum Y coordinate
 * @return Density in [0, 1], where 1.0 means all Y coordinates are present
 */
template <class CoordType>
double compute_density(std::size_t num_rows, CoordType y_min, CoordType y_max) {
  const std::size_t y_range = static_cast<std::size_t>(y_max - y_min + 1);
  if (y_range == 0) return 0.0;
  return static_cast<double>(num_rows) / static_cast<double>(y_range);
}

// ============================================================================
// Strategy Selection
// ============================================================================

/**
 * @brief Select optimal row mapping strategy based on mesh characteristics.
 *
 * Decision tree:
 * 1. Small meshes (< 100 rows) → OPTIMIZED baseline
 * 2. Dense meshes (y_range == num_rows) → DIRECT_INDEX O(1)
 * 3. High density (> 0.8) with small range (< 10000) → DIRECT_INDEX
 * 4. Uniform stride detected → DIRECT_INDEX stride-based O(1)
 * 5. Large balanced meshes (ratio 0.1-10.0) → PARALLEL_MERGE
 * 6. Large unbalanced meshes → HYBRID_CPU_GPU
 * 7. Sparse large meshes (< 0.3 density, > 50000 rows) → HASH_BASED
 * 8. Default → SOA_OPTIMIZED (safe improvement for all cases)
 *
 * @tparam DIM Dimension (2 or 3)
 * @tparam CoordType Coordinate type
 * @tparam IndexType Index type
 * @param A First input mesh
 * @param B Second input mesh
 * @return Selected strategy
 */
template <int DIM, class CoordType, class IndexType>
RowMappingStrategy select_strategy(
    const optimized::Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>& A,
    const optimized::Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>& B) {

  const std::size_t n_a = A.num_rows;
  const std::size_t n_b = B.num_rows;

  // 1. Size-based decision: small meshes use baseline
  if (n_a < 100 || n_b < 100) {
    return RowMappingStrategy::OPTIMIZED;
  }

  // 2. Copy first/last rows to host for pattern analysis
  auto host_rows_b = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, B.row_keys);

  const CoordType y_min_b = host_rows_b(0).y;
  const CoordType y_max_b = host_rows_b(n_b - 1).y;
  const std::size_t y_range_b = static_cast<std::size_t>(y_max_b - y_min_b + 1);
  const double density_b = compute_density(n_b, y_min_b, y_max_b);

  // 3. Dense detection: consecutive coordinates (perfect O(1))
  if (y_range_b == n_b) {
    return RowMappingStrategy::DIRECT_INDEX;
  }

  // 4. High density with small range: lookup table efficient
  if (density_b > 0.8 && y_range_b < 10000) {
    return RowMappingStrategy::DIRECT_INDEX;
  }

  // 5. Uniform stride detection: stride-based O(1)
  CoordType stride;
  if (detect_uniform_stride(B.row_keys, n_b, stride)) {
    return RowMappingStrategy::DIRECT_INDEX;
  }

  // 6. Size-based for large meshes
  if (n_a > 10000 && n_b > 10000) {
    const double ratio = static_cast<double>(n_a) / static_cast<double>(n_b);

    if (ratio > 0.1 && ratio < 10.0) {
      // Balanced sizes: parallel merge is most efficient
      return RowMappingStrategy::PARALLEL_MERGE;
    } else {
      // Unbalanced: hybrid CPU-GPU wins
      return RowMappingStrategy::HYBRID_CPU_GPU;
    }
  }

  // 7. Sparse large meshes: hash table wins
  if (density_b < 0.3 && n_a > 50000) {
    return RowMappingStrategy::HASH_BASED;
  }

  // 8. Default: SoA optimized (safe improvement for all cases)
  return RowMappingStrategy::SOA_OPTIMIZED;
}

// ============================================================================
// Main Entry Point with Dispatch
// ============================================================================

/**
 * @brief Intersect two meshes with automatic algorithm selection.
 *
 * This function analyzes the input meshes and dispatches to the optimal
 * algorithm implementation (v2-v8) based on their characteristics.
 *
 * @tparam DIM Dimension (2 or 3)
 * @tparam CoordType Coordinate type (default: int32_t)
 * @tparam IndexType Index type (default: std::size_t)
 * @param A First input mesh
 * @param B Second input mesh
 * @param out_metrics Optional output parameter for selection metrics
 * @return Intersection result mesh
 */
template <int DIM, class CoordType = int32_t, class IndexType = std::size_t>
inline optimized::Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>
intersect_meshes(
    const optimized::Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>& A,
    const optimized::Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>& B,
    SelectionMetrics* out_metrics = nullptr) {

  using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
  using MeshType = optimized::Mesh<DIM, DeviceMemorySpace, CoordType, IndexType>;

  // Start timing selection (if metrics requested)
  auto start_time = std::chrono::high_resolution_clock::now();

  // Step 1: Select strategy based on mesh characteristics
  const auto strategy = select_strategy<DIM, CoordType, IndexType>(A, B);

  // Step 2: Convert to appropriate mesh type and dispatch
  MeshType result;

  // Helper lambda to convert between mesh types
  auto convert_to_optimized = [](const auto& source) -> MeshType {
    MeshType dest;
    dest.row_keys = source.row_keys;
    dest.row_ptr = source.row_ptr;
    dest.intervals = source.intervals;
    dest.num_rows = source.num_rows;
    dest.num_intervals = source.num_intervals;
    return dest;
  };

  switch (strategy) {
    case RowMappingStrategy::HASH_BASED: {
      // v4: hash-based mapping
      hash_based::Mesh<DIM, DeviceMemorySpace, CoordType, IndexType> mesh_a, mesh_b;
      mesh_a.row_keys = A.row_keys;
      mesh_a.row_ptr = A.row_ptr;
      mesh_a.intervals = A.intervals;
      mesh_a.num_rows = A.num_rows;
      mesh_a.num_intervals = A.num_intervals;

      mesh_b.row_keys = B.row_keys;
      mesh_b.row_ptr = B.row_ptr;
      mesh_b.intervals = B.intervals;
      mesh_b.num_rows = B.num_rows;
      mesh_b.num_intervals = B.num_intervals;

      auto hash_result = hash_based::intersect_meshes<DIM>(mesh_a, mesh_b);
      result = convert_to_optimized(hash_result);
      break;
    }

    case RowMappingStrategy::PARALLEL_MERGE: {
      // v5: parallel merge algorithm
      parallel_merge::Mesh<DIM, DeviceMemorySpace, CoordType, IndexType> mesh_a, mesh_b;
      mesh_a.row_keys = A.row_keys;
      mesh_a.row_ptr = A.row_ptr;
      mesh_a.intervals = A.intervals;
      mesh_a.num_rows = A.num_rows;
      mesh_a.num_intervals = A.num_intervals;

      mesh_b.row_keys = B.row_keys;
      mesh_b.row_ptr = B.row_ptr;
      mesh_b.intervals = B.intervals;
      mesh_b.num_rows = B.num_rows;
      mesh_b.num_intervals = B.num_intervals;

      auto merge_result = parallel_merge::intersect_meshes<DIM>(mesh_a, mesh_b);
      result = convert_to_optimized(merge_result);
      break;
    }

    case RowMappingStrategy::DIRECT_INDEX: {
      // v6: direct index O(1) lookup
      direct_index::Mesh<DIM, DeviceMemorySpace, CoordType, IndexType> mesh_a, mesh_b;
      mesh_a.row_keys = A.row_keys;
      mesh_a.row_ptr = A.row_ptr;
      mesh_a.intervals = A.intervals;
      mesh_a.num_rows = A.num_rows;
      mesh_a.num_intervals = A.num_intervals;

      mesh_b.row_keys = B.row_keys;
      mesh_b.row_ptr = B.row_ptr;
      mesh_b.intervals = B.intervals;
      mesh_b.num_rows = B.num_rows;
      mesh_b.num_intervals = B.num_intervals;

      auto direct_result = direct_index::intersect_meshes<DIM>(mesh_a, mesh_b);
      result = convert_to_optimized(direct_result);
      break;
    }

    case RowMappingStrategy::SOA_OPTIMIZED: {
      // v7: Structure of Arrays optimization
      result = soa_optimized::intersect_meshes<DIM, CoordType, IndexType>(A, B);
      break;
    }

    case RowMappingStrategy::HYBRID_CPU_GPU: {
      // v8: Hybrid CPU-GPU algorithm
      hybrid_cpu_gpu::Mesh<DIM, DeviceMemorySpace, CoordType, IndexType> mesh_a, mesh_b;
      mesh_a.row_keys = A.row_keys;
      mesh_a.row_ptr = A.row_ptr;
      mesh_a.intervals = A.intervals;
      mesh_a.num_rows = A.num_rows;
      mesh_a.num_intervals = A.num_intervals;

      mesh_b.row_keys = B.row_keys;
      mesh_b.row_ptr = B.row_ptr;
      mesh_b.intervals = B.intervals;
      mesh_b.num_rows = B.num_rows;
      mesh_b.num_intervals = B.num_intervals;

      auto hybrid_result = hybrid_cpu_gpu::intersect_meshes<DIM>(mesh_a, mesh_b);
      result = convert_to_optimized(hybrid_result);
      break;
    }

    default: {
      // v2: optimized baseline
      result = optimized::intersect_meshes<DIM>(A, B);
      break;
    }
  }

  // Step 3: Fill metrics (if requested)
  if (out_metrics) {
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

    out_metrics->strategy = strategy;
    out_metrics->num_rows_a = A.num_rows;
    out_metrics->num_rows_b = B.num_rows;

    // Compute density for metrics
    auto host_rows_b = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, B.row_keys);
    const CoordType y_min_b = host_rows_b(0).y;
    const CoordType y_max_b = host_rows_b(B.num_rows - 1).y;
    out_metrics->density_b = compute_density(B.num_rows, y_min_b, y_max_b);

    // Check uniform stride for metrics
    CoordType stride;
    out_metrics->is_uniform = detect_uniform_stride(B.row_keys, B.num_rows, stride);

    out_metrics->selection_time_ms = elapsed.count();
  }

  return result;
}

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * @brief 2D convenience wrapper for mesh intersection with adaptive selection.
 *
 * @tparam CoordType Coordinate type (default: int32_t)
 * @tparam IndexType Index type (default: std::size_t)
 * @param A First input 2D mesh
 * @param B Second input 2D mesh
 * @param out_metrics Optional output parameter for selection metrics
 * @return Intersection result mesh
 */
template <class CoordType = int32_t, class IndexType = std::size_t>
inline optimized::Mesh<2, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>
intersect_meshes_2d(
    const optimized::Mesh<2, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>& A,
    const optimized::Mesh<2, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>& B,
    SelectionMetrics* out_metrics = nullptr) {
  return intersect_meshes<2, CoordType, IndexType>(A, B, out_metrics);
}

/**
 * @brief 3D convenience wrapper for mesh intersection with adaptive selection.
 *
 * @tparam CoordType Coordinate type (default: int32_t)
 * @tparam IndexType Index type (default: std::size_t)
 * @param A First input 3D mesh
 * @param B Second input 3D mesh
 * @param out_metrics Optional output parameter for selection metrics
 * @return Intersection result mesh
 */
template <class CoordType = int32_t, class IndexType = std::size_t>
inline optimized::Mesh<3, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>
intersect_meshes_3d(
    const optimized::Mesh<3, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>& A,
    const optimized::Mesh<3, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>& B,
    SelectionMetrics* out_metrics = nullptr) {
  return intersect_meshes<3, CoordType, IndexType>(A, B, out_metrics);
}

} // namespace playground::subsetix::csr::intersection::adaptive
