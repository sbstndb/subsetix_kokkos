#pragma once

#include <cstddef>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include <subsetix/csr_field.hpp>
#include <subsetix/csr_interval_set.hpp>

namespace subsetix {
namespace csr {

// ---------------------------------------------------------------------------
// Field Algebra Operations
// ---------------------------------------------------------------------------

/**
 * @brief Add two fields element-wise: result = a + b.
 *
 * Precondition: All three fields must have the same geometry (same row keys,
 * row_ptr, and intervals). Only the values are different.
 */
template <typename T>
inline void field_add_device(Field2DDevice<T>& result,
                             const Field2DDevice<T>& a,
                             const Field2DDevice<T>& b) {
  const std::size_t n = result.size();
  if (n == 0) {
    return;
  }

  auto a_vals = a.values;
  auto b_vals = b.values;
  auto result_vals = result.values;

  Kokkos::parallel_for(
      "subsetix_field2d_add",
      Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t i) {
        result_vals(i) = a_vals(i) + b_vals(i);
      });

  ExecSpace().fence();
}

/**
 * @brief Subtract two fields element-wise: result = a - b.
 *
 * Precondition: All three fields must have the same geometry.
 */
template <typename T>
inline void field_sub_device(Field2DDevice<T>& result,
                             const Field2DDevice<T>& a,
                             const Field2DDevice<T>& b) {
  const std::size_t n = result.size();
  if (n == 0) {
    return;
  }

  auto a_vals = a.values;
  auto b_vals = b.values;
  auto result_vals = result.values;

  Kokkos::parallel_for(
      "subsetix_field2d_sub",
      Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t i) {
        result_vals(i) = a_vals(i) - b_vals(i);
      });

  ExecSpace().fence();
}

/**
 * @brief Multiply two fields element-wise: result = a * b.
 *
 * Precondition: All three fields must have the same geometry.
 */
template <typename T>
inline void field_mul_device(Field2DDevice<T>& result,
                             const Field2DDevice<T>& a,
                             const Field2DDevice<T>& b) {
  const std::size_t n = result.size();
  if (n == 0) {
    return;
  }

  auto a_vals = a.values;
  auto b_vals = b.values;
  auto result_vals = result.values;

  Kokkos::parallel_for(
      "subsetix_field2d_mul",
      Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t i) {
        result_vals(i) = a_vals(i) * b_vals(i);
      });

  ExecSpace().fence();
}

/**
 * @brief Divide two fields element-wise: result = a / b.
 *
 * Precondition: All three fields must have the same geometry.
 * Warning: No division-by-zero check is performed.
 */
template <typename T>
inline void field_div_device(Field2DDevice<T>& result,
                             const Field2DDevice<T>& a,
                             const Field2DDevice<T>& b) {
  const std::size_t n = result.size();
  if (n == 0) {
    return;
  }

  auto a_vals = a.values;
  auto b_vals = b.values;
  auto result_vals = result.values;

  Kokkos::parallel_for(
      "subsetix_field2d_div",
      Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t i) {
        result_vals(i) = a_vals(i) / b_vals(i);
      });

  ExecSpace().fence();
}

/**
 * @brief Compute the absolute value of a field element-wise: result = |a|.
 *
 * Precondition: Both fields must have the same geometry.
 */
template <typename T>
inline void field_abs_device(Field2DDevice<T>& result,
                             const Field2DDevice<T>& a) {
  const std::size_t n = result.size();
  if (n == 0) {
    return;
  }

  auto a_vals = a.values;
  auto result_vals = result.values;

  Kokkos::parallel_for(
      "subsetix_field2d_abs",
      Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t i) {
        const T val = a_vals(i);
        result_vals(i) = (val < T(0)) ? -val : val;
      });

  ExecSpace().fence();
}

/**
 * @brief Compute a linear combination: result = alpha * a + beta * b.
 *
 * Precondition: All three fields must have the same geometry.
 */
template <typename T>
inline void field_axpby_device(Field2DDevice<T>& result,
                               const T& alpha,
                               const Field2DDevice<T>& a,
                               const T& beta,
                               const Field2DDevice<T>& b) {
  const std::size_t n = result.size();
  if (n == 0) {
    return;
  }

  const T alpha_copy = alpha;
  const T beta_copy = beta;
  auto a_vals = a.values;
  auto b_vals = b.values;
  auto result_vals = result.values;

  Kokkos::parallel_for(
      "subsetix_field2d_axpby",
      Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t i) {
        result_vals(i) = alpha_copy * a_vals(i) + beta_copy * b_vals(i);
      });

  ExecSpace().fence();
}

/**
 * @brief Compute the dot product of two fields: result = sum(a[i] * b[i]).
 *
 * Precondition: Both fields must have the same geometry.
 */
template <typename T>
inline T field_dot_device(const Field2DDevice<T>& a,
                          const Field2DDevice<T>& b) {
  const std::size_t n = a.size();
  if (n == 0) {
    return T(0);
  }

  auto a_vals = a.values;
  auto b_vals = b.values;

  T result = T(0);
  Kokkos::parallel_reduce(
      "subsetix_field2d_dot",
      Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t i, T& sum) {
        sum += a_vals(i) * b_vals(i);
      },
      result);

  return result;
}

/**
 * @brief Compute the L2 norm of a field: result = sqrt(sum(a[i]^2)).
 *
 * Precondition: Field must be initialized.
 */
template <typename T>
inline T field_norm_l2_device(const Field2DDevice<T>& a) {
  const std::size_t n = a.size();
  if (n == 0) {
    return T(0);
  }

  auto a_vals = a.values;

  T sum_sq = T(0);
  Kokkos::parallel_reduce(
      "subsetix_field2d_norm_l2",
      Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t i, T& local_sum) {
        const T val = a_vals(i);
        local_sum += val * val;
      },
      sum_sq);

  return Kokkos::sqrt(sum_sq);
}

} // namespace csr
} // namespace subsetix
