#pragma once

#include <vector>
#include <Kokkos_Core.hpp>
#include <subsetix/csr_interval_set.hpp>
#include <subsetix/detail/memory_utils.hpp>

namespace subsetix {
namespace csr {

using ExecSpace = Kokkos::DefaultExecutionSpace;

namespace detail {

/**
 * @brief Unified workspace for CSR operations.
 *
 * Instead of having separate workspaces for each operation type (which wastes
 * memory when they are not used simultaneously), this workspace provides a
 * pool of generic buffers that operations can "checkout" and use as needed.
 */
struct UnifiedCsrWorkspace {
  // Generic buffers for int arrays (e.g., indices, maps, flags)
  Kokkos::View<int*, DeviceMemorySpace> int_buf_0;
  Kokkos::View<int*, DeviceMemorySpace> int_buf_1;
  Kokkos::View<int*, DeviceMemorySpace> int_buf_2;
  Kokkos::View<int*, DeviceMemorySpace> int_buf_3;
  Kokkos::View<int*, DeviceMemorySpace> int_buf_4;

  // Generic buffers for size_t arrays (e.g., row_ptr, counts, offsets)
  Kokkos::View<std::size_t*, DeviceMemorySpace> size_t_buf_0;
  Kokkos::View<std::size_t*, DeviceMemorySpace> size_t_buf_1;

  // Generic buffers for scalar values (e.g., totals)
  Kokkos::View<std::size_t, DeviceMemorySpace> scalar_size_t_buf_0;
  Kokkos::View<int, DeviceMemorySpace> scalar_int_buf_0;

  // Generic buffers for RowKey arrays
  IntervalSet2DDevice::RowKeyView row_key_buf_0;
  IntervalSet2DDevice::RowKeyView row_key_buf_1;

  // Generic buffers for Interval arrays
  IntervalSet2DDevice::IntervalView interval_buf_0;

  // Generic buffers for field values (double precision)
  Kokkos::View<double*, DeviceMemorySpace> double_buf_0;
  Kokkos::View<double*, DeviceMemorySpace> double_buf_1;

  // Generic buffers for field values (single precision)
  Kokkos::View<float*, DeviceMemorySpace> float_buf_0;
  Kokkos::View<float*, DeviceMemorySpace> float_buf_1;

  // Accessors that ensure capacity
  
  Kokkos::View<int*, DeviceMemorySpace> get_int_buf_0(std::size_t size) {
    ensure_view_capacity(int_buf_0, size, "unified_ws_int_0");
    return int_buf_0;
  }
  
  Kokkos::View<int*, DeviceMemorySpace> get_int_buf_1(std::size_t size) {
    ensure_view_capacity(int_buf_1, size, "unified_ws_int_1");
    return int_buf_1;
  }

  Kokkos::View<int*, DeviceMemorySpace> get_int_buf_2(std::size_t size) {
    ensure_view_capacity(int_buf_2, size, "unified_ws_int_2");
    return int_buf_2;
  }

  Kokkos::View<int*, DeviceMemorySpace> get_int_buf_3(std::size_t size) {
    ensure_view_capacity(int_buf_3, size, "unified_ws_int_3");
    return int_buf_3;
  }

  Kokkos::View<int*, DeviceMemorySpace> get_int_buf_4(std::size_t size) {
    ensure_view_capacity(int_buf_4, size, "unified_ws_int_4");
    return int_buf_4;
  }

  Kokkos::View<std::size_t*, DeviceMemorySpace> get_size_t_buf_0(std::size_t size) {
    ensure_view_capacity(size_t_buf_0, size, "unified_ws_size_t_0");
    return size_t_buf_0;
  }

  Kokkos::View<std::size_t*, DeviceMemorySpace> get_size_t_buf_1(std::size_t size) {
    ensure_view_capacity(size_t_buf_1, size, "unified_ws_size_t_1");
    return size_t_buf_1;
  }

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

  IntervalSet2DDevice::RowKeyView get_row_key_buf_0(std::size_t size) {
    ensure_view_capacity(row_key_buf_0, size, "unified_ws_row_key_0");
    return row_key_buf_0;
  }

  IntervalSet2DDevice::RowKeyView get_row_key_buf_1(std::size_t size) {
    ensure_view_capacity(row_key_buf_1, size, "unified_ws_row_key_1");
    return row_key_buf_1;
  }

  IntervalSet2DDevice::IntervalView get_interval_buf_0(std::size_t size) {
    ensure_view_capacity(interval_buf_0, size, "unified_ws_interval_0");
    return interval_buf_0;
  }

  Kokkos::View<double*, DeviceMemorySpace> get_double_buf_0(std::size_t size) {
    ensure_view_capacity(double_buf_0, size, "unified_ws_double_0");
    return double_buf_0;
  }

  Kokkos::View<double*, DeviceMemorySpace> get_double_buf_1(std::size_t size) {
    ensure_view_capacity(double_buf_1, size, "unified_ws_double_1");
    return double_buf_1;
  }

  Kokkos::View<float*, DeviceMemorySpace> get_float_buf_0(std::size_t size) {
    ensure_view_capacity(float_buf_0, size, "unified_ws_float_0");
    return float_buf_0;
  }

  Kokkos::View<float*, DeviceMemorySpace> get_float_buf_1(std::size_t size) {
    ensure_view_capacity(float_buf_1, size, "unified_ws_float_1");
    return float_buf_1;
  }

  /**
   * @brief Reclaims memory by resetting all views to empty.
   *
   * This frees the device memory allocated by this workspace.
   */
  void clear() {
    int_buf_0 = Kokkos::View<int*, DeviceMemorySpace>();
    int_buf_1 = Kokkos::View<int*, DeviceMemorySpace>();
    int_buf_2 = Kokkos::View<int*, DeviceMemorySpace>();
    int_buf_3 = Kokkos::View<int*, DeviceMemorySpace>();
    int_buf_4 = Kokkos::View<int*, DeviceMemorySpace>();

    size_t_buf_0 = Kokkos::View<std::size_t*, DeviceMemorySpace>();
    size_t_buf_1 = Kokkos::View<std::size_t*, DeviceMemorySpace>();

    scalar_size_t_buf_0 = Kokkos::View<std::size_t, DeviceMemorySpace>();
    scalar_int_buf_0 = Kokkos::View<int, DeviceMemorySpace>();

    row_key_buf_0 = IntervalSet2DDevice::RowKeyView();
    row_key_buf_1 = IntervalSet2DDevice::RowKeyView();
    interval_buf_0 = IntervalSet2DDevice::IntervalView();

    double_buf_0 = Kokkos::View<double*, DeviceMemorySpace>();
    double_buf_1 = Kokkos::View<double*, DeviceMemorySpace>();
    float_buf_0 = Kokkos::View<float*, DeviceMemorySpace>();
    float_buf_1 = Kokkos::View<float*, DeviceMemorySpace>();
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
