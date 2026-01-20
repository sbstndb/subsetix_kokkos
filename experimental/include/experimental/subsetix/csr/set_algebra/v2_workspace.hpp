// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <experimental/subsetix/csr/mesh.hpp>
#include <Kokkos_Core.hpp>
#include <array>

namespace experimental::subsetix::csr::v2 {

// ============================================================================
// Workspace for v2 intersection algorithm
// ============================================================================

/**
 * @brief Reusable workspace for v2 mesh intersection.
 *
 * This workspace eliminates per-operation allocations by pre-allocating
 * scratch buffers that grow as needed. Inspired by the stable's
 * UnifiedCsrWorkspace pattern.
 *
 * Benefits:
 * - Reduces device allocations from 16+ to ~2-3 per intersection
 * - Enables memory reuse across multiple operations
 * - Reduces malloc overhead which is especially expensive on GPU
 */
template <class MemorySpace>
class MeshIntersectionWorkspace {
public:
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using IndexView = Kokkos::View<std::size_t*, MemorySpace>;
  using IntView = Kokkos::View<int*, MemorySpace>;

  // Buffer counts (tunable based on typical workload)
  static constexpr int NUM_INDEX_BUFS = 3;
  static constexpr int NUM_INT_BUFS = 3;  // Increased for tmp_idx_b

  // Scratch interval buffer - single pass, no over-allocation
  using IntervalView = Kokkos::View<Interval*, MemorySpace>;
  IntervalView scratch_intervals_;

  // Index buffers for row mapping and offsets
  std::array<IndexView, NUM_INDEX_BUFS> index_bufs_;

  // Int buffers for flags and indices
  std::array<IntView, NUM_INT_BUFS> int_bufs_;

  // Current capacities
  std::size_t scratch_capacity_ = 0;
  std::size_t index_capacity_ = 0;
  std::size_t int_capacity_ = 0;

  // Labels for debugging
  std::array<std::string, NUM_INDEX_BUFS> index_labels_ = {
    "workspace_index_0", "workspace_index_1", "workspace_index_2"
  };
  std::array<std::string, NUM_INT_BUFS> int_labels_ = {
    "workspace_int_0", "workspace_int_1", "workspace_int_2"
  };

  // FIX: Remove KOKKOS_INLINE_FUNCTION - default constructor is host-only
  MeshIntersectionWorkspace() = default;

  // ========================================================================
  // Scratch interval buffer (single-pass output)
  // ========================================================================

  /**
   * @brief Ensure scratch buffer has capacity for n intervals.
   *
   * Unlike v1 which allocates A.num + B.num (worst case),
   * v2 only allocates what's needed (max of inputs).
   */
  inline void ensure_scratch_capacity(std::size_t n) {
    if (scratch_capacity_ < n) {
      scratch_intervals_ = IntervalView("scratch_intervals", n);
      scratch_capacity_ = n;
    }
  }

  // FIX: Remove KOKKOS_INLINE_FUNCTION - these are host-only methods
  IntervalView scratch_intervals() const { return scratch_intervals_; }

  // ========================================================================
  // Index buffers
  // ========================================================================

  inline void ensure_index_capacity(std::size_t n) {
    if (index_capacity_ < n) {
      for (int i = 0; i < NUM_INDEX_BUFS; ++i) {
        index_bufs_[i] = IndexView(index_labels_[i], n);
      }
      index_capacity_ = n;
    }
  }

  IndexView index_buf(int idx) const { return index_bufs_[idx]; }

  // ========================================================================
  // Int buffers
  // ========================================================================

  inline void ensure_int_capacity(std::size_t n) {
    if (int_capacity_ < n) {
      for (int i = 0; i < NUM_INT_BUFS; ++i) {
        int_bufs_[i] = IntView(int_labels_[i], n);
      }
      int_capacity_ = n;
    }
  }

  IntView int_buf(int idx) const { return int_bufs_[idx]; }

  // ========================================================================
  // Utility: Reset for reuse
  // ========================================================================

  /**
   * @brief Clear workspace without freeing memory.
   *
   * Buffers retain their capacity for reuse.
   */
  // FIX: Remove KOKKOS_INLINE_FUNCTION - this is a host-only method
  inline void clear() {
    scratch_capacity_ = 0;
    index_capacity_ = 0;
    int_capacity_ = 0;
  }
};

// ==============================================================================
// Device-side workspace accessor (for use in kernels)
// ==============================================================================

/**
 * @brief Device-compatible view of workspace for kernel access.
 *
 * This lightweight struct can be passed to GPU kernels
 * without carrying the entire workspace.
 */
template <class MemorySpace>
struct DeviceWorkspaceView {
  using IntervalView = Kokkos::View<Interval*, MemorySpace>;
  using IndexView = Kokkos::View<std::size_t*, MemorySpace>;
  using IntView = Kokkos::View<int*, MemorySpace>;

  IntervalView scratch_intervals;
  IndexView index_bufs[3];
  IntView int_bufs[3];  // Increased to 3

  KOKKOS_INLINE_FUNCTION
  IntervalView scratch() const { return scratch_intervals; }

  KOKKOS_INLINE_FUNCTION
  IndexView index(int idx) const { return index_bufs[idx]; }

  KOKKOS_INLINE_FUNCTION
  IntView int_buf(int idx) const { return int_bufs[idx]; }
};

/**
 * @brief Create device view from workspace (host-side).
 */
template <class MemorySpace>
DeviceWorkspaceView<MemorySpace> make_device_view(MeshIntersectionWorkspace<MemorySpace>& ws) {
  DeviceWorkspaceView<MemorySpace> view;
  view.scratch_intervals = ws.scratch_intervals_;
  for (int i = 0; i < 3; ++i) view.index_bufs[i] = ws.index_bufs_[i];
  for (int i = 0; i < 3; ++i) view.int_bufs[i] = ws.int_bufs_[i];  // FIX: Copy 3 int bufs
  return view;
}

} // namespace experimental::subsetix::csr::v2
