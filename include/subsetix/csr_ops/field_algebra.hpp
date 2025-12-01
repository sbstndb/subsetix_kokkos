#pragma once

#include <cstddef>
#include <stdexcept>

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include <subsetix/field/csr_field.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>

namespace subsetix {
namespace csr {

// ---------------------------------------------------------------------------
// Field Algebra Operations
// ---------------------------------------------------------------------------

namespace detail {

// Kokkos-compatible binary functors
template <typename T> struct Plus {
  KOKKOS_INLINE_FUNCTION T operator()(const T& a, const T& b) const { return a + b; }
};
template <typename T> struct Minus {
  KOKKOS_INLINE_FUNCTION T operator()(const T& a, const T& b) const { return a - b; }
};
template <typename T> struct Multiplies {
  KOKKOS_INLINE_FUNCTION T operator()(const T& a, const T& b) const { return a * b; }
};
template <typename T> struct Divides {
  KOKKOS_INLINE_FUNCTION T operator()(const T& a, const T& b) const { return a / b; }
};

/**
 * @brief Generic binary operation on two fields: result = Op(a, b).
 *
 * Precondition: All three fields must have the same geometry.
 */
template <typename T, typename BinaryOp>
inline void field_binary_op_device(Field2DDevice<T>& result,
                                   const Field2DDevice<T>& a,
                                   const Field2DDevice<T>& b,
                                   const char* /* label */) {
  const std::size_t n = result.size();
  if (n == 0) return;

  auto a_sub = Kokkos::subview(a.values, std::make_pair(std::size_t(0), n));
  auto b_sub = Kokkos::subview(b.values, std::make_pair(std::size_t(0), n));
  auto r_sub = Kokkos::subview(result.values, std::make_pair(std::size_t(0), n));

  Kokkos::Experimental::transform(ExecSpace(), a_sub, b_sub, r_sub, BinaryOp{});
  ExecSpace().fence();
}

} // namespace detail

/** @brief Add two fields element-wise: result = a + b. */
template <typename T>
inline void field_add_device(Field2DDevice<T>& result,
                             const Field2DDevice<T>& a,
                             const Field2DDevice<T>& b) {
  detail::field_binary_op_device<T, detail::Plus<T>>(result, a, b, "field_add");
}

/** @brief Subtract two fields element-wise: result = a - b. */
template <typename T>
inline void field_sub_device(Field2DDevice<T>& result,
                             const Field2DDevice<T>& a,
                             const Field2DDevice<T>& b) {
  detail::field_binary_op_device<T, detail::Minus<T>>(result, a, b, "field_sub");
}

/** @brief Multiply two fields element-wise: result = a * b. */
template <typename T>
inline void field_mul_device(Field2DDevice<T>& result,
                             const Field2DDevice<T>& a,
                             const Field2DDevice<T>& b) {
  detail::field_binary_op_device<T, detail::Multiplies<T>>(result, a, b, "field_mul");
}

/** @brief Divide two fields element-wise: result = a / b. (No div-by-zero check) */
template <typename T>
inline void field_div_device(Field2DDevice<T>& result,
                             const Field2DDevice<T>& a,
                             const Field2DDevice<T>& b) {
  detail::field_binary_op_device<T, detail::Divides<T>>(result, a, b, "field_div");
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
  if (n == 0) return;

  auto a_sub = Kokkos::subview(a.values, std::make_pair(std::size_t(0), n));
  auto r_sub = Kokkos::subview(result.values, std::make_pair(std::size_t(0), n));

  Kokkos::Experimental::transform(
      ExecSpace(), a_sub, r_sub,
      KOKKOS_LAMBDA(const T val) { return Kokkos::abs(val); });

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
  if (n == 0) return;

  const T alpha_copy = alpha;
  const T beta_copy = beta;

  auto a_sub = Kokkos::subview(a.values, std::make_pair(std::size_t(0), n));
  auto b_sub = Kokkos::subview(b.values, std::make_pair(std::size_t(0), n));
  auto r_sub = Kokkos::subview(result.values, std::make_pair(std::size_t(0), n));

  Kokkos::Experimental::transform(
      ExecSpace(), a_sub, b_sub, r_sub,
      KOKKOS_LAMBDA(const T av, const T bv) {
        return alpha_copy * av + beta_copy * bv;
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
  if (n == 0) return T(0);

  auto a_sub = Kokkos::subview(a.values, std::make_pair(std::size_t(0), n));
  auto b_sub = Kokkos::subview(b.values, std::make_pair(std::size_t(0), n));

  return Kokkos::Experimental::transform_reduce(
      ExecSpace(), a_sub, b_sub, T(0),
      KOKKOS_LAMBDA(const T x, const T y) { return x + y; },
      KOKKOS_LAMBDA(const T av, const T bv) { return av * bv; });
}

/**
 * @brief Compute the L2 norm of a field: result = sqrt(sum(a[i]^2)).
 *
 * Precondition: Field must be initialized.
 */
template <typename T>
inline T field_norm_l2_device(const Field2DDevice<T>& a) {
  const std::size_t n = a.size();
  if (n == 0) return T(0);

  auto a_sub = Kokkos::subview(a.values, std::make_pair(std::size_t(0), n));

  T sum_sq = Kokkos::Experimental::transform_reduce(
      ExecSpace(), a_sub, T(0),
      KOKKOS_LAMBDA(const T x, const T y) { return x + y; },
      KOKKOS_LAMBDA(const T val) { return val * val; });

  return Kokkos::sqrt(sum_sq);
}

} // namespace csr
} // namespace subsetix
