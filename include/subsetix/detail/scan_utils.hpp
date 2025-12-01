#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <subsetix/geometry/csr_backend.hpp>

namespace subsetix {
namespace csr {
namespace detail {

/**
 * @brief Perform an exclusive scan on counts, returning the total.
 *
 * For each i in [0, n), writes output(i) = sum of counts(0..i-1).
 * Returns the total sum of all counts.
 *
 * @tparam T The accumulator type (e.g., std::size_t, int)
 * @tparam CountView View type for input counts
 * @tparam OutputView View type for output offsets
 * @param label Kokkos kernel label
 * @param n Number of elements to scan
 * @param counts Input counts view
 * @param output Output offsets view (must have size >= n)
 * @return The total sum of all counts
 */
template <typename T, class CountView, class OutputView>
T exclusive_scan_with_total(
    const std::string& label,
    std::size_t n,
    const CountView& counts,
    OutputView& output) {
  if (n == 0) {
    return T(0);
  }

  // Create subviews with exact size n
  auto counts_sub = Kokkos::subview(counts, std::make_pair(std::size_t(0), n));
  auto output_sub = Kokkos::subview(output, std::make_pair(std::size_t(0), n));

  // Exclusive scan
  Kokkos::Experimental::exclusive_scan(
      ExecSpace(), counts_sub, output_sub, T(0));

  // Compute total separately (reduce)
  T total = T(0);
  Kokkos::parallel_reduce(
      label + "_total",
      Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t i, T& sum) {
        sum += static_cast<T>(counts(i));
      },
      total);

  return total;
}

/**
 * @brief Perform an exclusive scan for CSR row_ptr, returning the total.
 *
 * For each i in [0, n), writes row_ptr(i) = sum of counts(0..i-1).
 * Also writes row_ptr(n) = total.
 * Returns the total sum of all counts.
 *
 * @tparam T The accumulator type (e.g., std::size_t)
 * @tparam CountView View type for input counts
 * @tparam IndexView View type for output row_ptr (must have size >= n+1)
 * @param label Kokkos kernel label
 * @param n Number of rows (row_ptr must have n+1 entries)
 * @param counts Input counts view
 * @param row_ptr Output row pointer view
 * @return The total sum of all counts
 */
template <typename T, class CountView, class IndexView>
T exclusive_scan_csr_row_ptr(
    const std::string& label,
    std::size_t n,
    const CountView& counts,
    IndexView& row_ptr) {
  if (n == 0) {
    Kokkos::deep_copy(Kokkos::subview(row_ptr, 0), T(0));
    return T(0);
  }

  // Create subviews with exact size n
  auto counts_sub = Kokkos::subview(counts, std::make_pair(std::size_t(0), n));
  auto row_ptr_sub = Kokkos::subview(row_ptr, std::make_pair(std::size_t(0), n));

  // Exclusive scan into row_ptr[0..n)
  Kokkos::Experimental::exclusive_scan(
      ExecSpace(), counts_sub, row_ptr_sub, T(0));

  // Compute total and set row_ptr[n]
  T total = T(0);
  Kokkos::parallel_reduce(
      label + "_total",
      Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t i, T& sum) {
        sum += static_cast<T>(counts(i));
      },
      total);

  Kokkos::deep_copy(Kokkos::subview(row_ptr, n), total);
  return total;
}

/**
 * @brief Perform an exclusive scan for CSR row_ptr with a provided total view.
 *
 * Same as exclusive_scan_csr_row_ptr but stores the total in an existing
 * device view instead of creating a temporary one. Useful when the total
 * needs to be used in subsequent device kernels.
 *
 * @tparam T The accumulator type (e.g., std::size_t)
 * @tparam CountView View type for input counts
 * @tparam IndexView View type for output row_ptr
 * @tparam TotalView Scalar view type for the total
 * @param label Kokkos kernel label
 * @param n Number of rows
 * @param counts Input counts view
 * @param row_ptr Output row pointer view
 * @param total_view Device scalar view to store the total
 * @return The total sum (also stored in total_view)
 */
template <typename T, class CountView, class IndexView, class TotalView>
T exclusive_scan_csr_row_ptr(
    const std::string& label,
    std::size_t n,
    const CountView& counts,
    IndexView& row_ptr,
    TotalView& total_view) {
  if (n == 0) {
    Kokkos::deep_copy(Kokkos::subview(row_ptr, 0), T(0));
    Kokkos::deep_copy(total_view, T(0));
    return T(0);
  }

  Kokkos::parallel_scan(
      label,
      Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t i, T& update, const bool final_pass) {
        const T c = static_cast<T>(counts(i));
        if (final_pass) {
          row_ptr(i) = update;
          if (i + 1 == n) {
            row_ptr(n) = update + c;
            total_view() = update + c;
          }
        }
        update += c;
      });

  ExecSpace().fence();

  T host_total = T(0);
  Kokkos::deep_copy(host_total, total_view);
  return host_total;
}

} // namespace detail
} // namespace csr
} // namespace subsetix
