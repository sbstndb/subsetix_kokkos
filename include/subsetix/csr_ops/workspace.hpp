#pragma once

#include <array>
#include <Kokkos_Core.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/detail/memory_utils.hpp>

namespace subsetix {
namespace csr {

namespace detail {

/**
 * @brief Unified workspace for CSR operations.
 *
 * Instead of having separate workspaces for each operation type (which wastes
 * memory when they are not used simultaneously), this workspace provides a
 * pool of generic buffers that operations can "checkout" and use as needed.
 */
struct UnifiedCsrWorkspace {
  // Buffer counts
  static constexpr std::size_t NUM_INT_BUFS = 5;
  static constexpr std::size_t NUM_SIZE_T_BUFS = 2;
  static constexpr std::size_t NUM_ROW_KEY_BUFS = 2;
  static constexpr std::size_t NUM_RAW_POOLS = 2;

  // Generic buffers for int arrays (e.g., indices, maps, flags)
  std::array<Kokkos::View<int*, DeviceMemorySpace>, NUM_INT_BUFS> int_bufs_;

  // Generic buffers for size_t arrays (e.g., row_ptr, counts, offsets)
  std::array<Kokkos::View<std::size_t*, DeviceMemorySpace>, NUM_SIZE_T_BUFS> size_t_bufs_;

  // Generic buffers for scalar values (e.g., totals)
  Kokkos::View<std::size_t, DeviceMemorySpace> scalar_size_t_buf_0;
  Kokkos::View<int, DeviceMemorySpace> scalar_int_buf_0;

  // Generic buffers for RowKey arrays
  std::array<IntervalSet2DDevice::RowKeyView, NUM_ROW_KEY_BUFS> row_key_bufs_;

  // Generic buffers for Interval arrays
  IntervalSet2DDevice::IntervalView interval_buf_0;

  // Generic raw memory pool for any type of values
  std::array<Kokkos::View<uint64_t*, DeviceMemorySpace>, NUM_RAW_POOLS> raw_pools_;

  // Label names for Kokkos views
  static constexpr const char* int_buf_labels_[NUM_INT_BUFS] = {
    "unified_ws_int_0", "unified_ws_int_1", "unified_ws_int_2",
    "unified_ws_int_3", "unified_ws_int_4"
  };
  static constexpr const char* size_t_buf_labels_[NUM_SIZE_T_BUFS] = {
    "unified_ws_size_t_0", "unified_ws_size_t_1"
  };
  static constexpr const char* row_key_buf_labels_[NUM_ROW_KEY_BUFS] = {
    "unified_ws_row_key_0", "unified_ws_row_key_1"
  };
  static constexpr const char* raw_pool_labels_[NUM_RAW_POOLS] = {
    "unified_ws_raw_0", "unified_ws_raw_1"
  };

  // Indexed accessors that ensure capacity

  Kokkos::View<int*, DeviceMemorySpace> get_int_buf(std::size_t idx, std::size_t size) {
    ensure_view_capacity(int_bufs_[idx], size, int_buf_labels_[idx]);
    return int_bufs_[idx];
  }

  Kokkos::View<std::size_t*, DeviceMemorySpace> get_size_t_buf(std::size_t idx, std::size_t size) {
    ensure_view_capacity(size_t_bufs_[idx], size, size_t_buf_labels_[idx]);
    return size_t_bufs_[idx];
  }

  IntervalSet2DDevice::RowKeyView get_row_key_buf(std::size_t idx, std::size_t size) {
    ensure_view_capacity(row_key_bufs_[idx], size, row_key_buf_labels_[idx]);
    return row_key_bufs_[idx];
  }

  // Backward-compatible wrappers for int buffers
  Kokkos::View<int*, DeviceMemorySpace> get_int_buf_0(std::size_t size) { return get_int_buf(0, size); }
  Kokkos::View<int*, DeviceMemorySpace> get_int_buf_1(std::size_t size) { return get_int_buf(1, size); }
  Kokkos::View<int*, DeviceMemorySpace> get_int_buf_2(std::size_t size) { return get_int_buf(2, size); }
  Kokkos::View<int*, DeviceMemorySpace> get_int_buf_3(std::size_t size) { return get_int_buf(3, size); }
  Kokkos::View<int*, DeviceMemorySpace> get_int_buf_4(std::size_t size) { return get_int_buf(4, size); }

  // Backward-compatible wrappers for size_t buffers
  Kokkos::View<std::size_t*, DeviceMemorySpace> get_size_t_buf_0(std::size_t size) { return get_size_t_buf(0, size); }
  Kokkos::View<std::size_t*, DeviceMemorySpace> get_size_t_buf_1(std::size_t size) { return get_size_t_buf(1, size); }

  // Backward-compatible wrappers for row_key buffers
  IntervalSet2DDevice::RowKeyView get_row_key_buf_0(std::size_t size) { return get_row_key_buf(0, size); }
  IntervalSet2DDevice::RowKeyView get_row_key_buf_1(std::size_t size) { return get_row_key_buf(1, size); }

  Kokkos::View<std::size_t, DeviceMemorySpace> get_scalar_size_t_buf_0() {
    if (!scalar_size_t_buf_0.data()) {
      scalar_size_t_buf_0 = Kokkos::View<std::size_t, DeviceMemorySpace>("unified_ws_scalar_size_t_0");
    }
    return scalar_size_t_buf_0;
  }

  Kokkos::View<int, DeviceMemorySpace> get_scalar_int_buf_0() {
    if (!scalar_int_buf_0.data()) {
      scalar_int_buf_0 = Kokkos::View<int, DeviceMemorySpace>("unified_ws_scalar_int_0");
    }
    return scalar_int_buf_0;
  }

  IntervalSet2DDevice::IntervalView get_interval_buf_0(std::size_t size) {
    ensure_view_capacity(interval_buf_0, size, "unified_ws_interval_0");
    return interval_buf_0;
  }

  /**
   * @brief Get a typed view from a generic raw memory pool.
   *
   * This allows reusing the same memory for double, float, int64_t, etc.
   * The returned view is Unmanaged (does not own memory).
   */
  template <typename T>
  Kokkos::View<T*, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
  get_value_buf(std::size_t idx, std::size_t size) {
    const std::size_t bytes_needed = size * sizeof(T);
    const std::size_t uint64_needed = (bytes_needed + sizeof(uint64_t) - 1) / sizeof(uint64_t);

    ensure_view_capacity(raw_pools_[idx], uint64_needed, raw_pool_labels_[idx]);

    return Kokkos::View<T*, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        reinterpret_cast<T*>(raw_pools_[idx].data()), size);
  }

  // Backward-compatible wrappers
  template <typename T>
  Kokkos::View<T*, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
  get_value_buf_0(std::size_t size) { return get_value_buf<T>(0, size); }

  template <typename T>
  Kokkos::View<T*, DeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
  get_value_buf_1(std::size_t size) { return get_value_buf<T>(1, size); }

  /**
   * @brief Reclaims memory by resetting all views to empty.
   *
   * This frees the device memory allocated by this workspace.
   */
  void clear() {
    for (std::size_t i = 0; i < NUM_INT_BUFS; ++i) {
      int_bufs_[i] = Kokkos::View<int*, DeviceMemorySpace>();
    }
    for (std::size_t i = 0; i < NUM_SIZE_T_BUFS; ++i) {
      size_t_bufs_[i] = Kokkos::View<std::size_t*, DeviceMemorySpace>();
    }
    for (std::size_t i = 0; i < NUM_ROW_KEY_BUFS; ++i) {
      row_key_bufs_[i] = IntervalSet2DDevice::RowKeyView();
    }
    for (std::size_t i = 0; i < NUM_RAW_POOLS; ++i) {
      raw_pools_[i] = Kokkos::View<uint64_t*, DeviceMemorySpace>();
    }

    scalar_size_t_buf_0 = Kokkos::View<std::size_t, DeviceMemorySpace>();
    scalar_int_buf_0 = Kokkos::View<int, DeviceMemorySpace>();
    interval_buf_0 = IntervalSet2DDevice::IntervalView();
  }
};

} // namespace detail

/**
 * @brief Context object carrying reusable workspaces for CSR set
 *        algebra operations.
 *
 * This allows applications to chain many set operations on device
 * (union, intersection, difference, AMR ops) without paying device
 * allocations for scratch buffers on every call.
 */
struct CsrSetAlgebraContext {
  detail::UnifiedCsrWorkspace workspace;
};

} // namespace csr
} // namespace subsetix
