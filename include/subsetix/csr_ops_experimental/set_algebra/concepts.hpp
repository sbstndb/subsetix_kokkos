// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <subsetix/csr_ops_experimental/geometry/mesh.hpp>
#include <concepts>

namespace subsetix::experimental::concepts {

// ============================================================================
// Mesh concepts
// ============================================================================

/**
 * @brief Concept for a 2D mesh type.
 *
 * A 2D mesh must have:
 * - RowKey type compatible with RowKey2D
 * - RowKeyView, IndexView, IntervalView member types
 * - row_keys, row_ptr, intervals data members
 * - num_rows, num_intervals counters
 */
template<typename T>
concept Mesh2D = requires(T mesh) {
  typename T::RowKey;
  requires std::same_as<decltype(T::DIM), const int&> && T::DIM == 2;
  typename T::RowKeyView;
  typename T::IndexView;
  typename T::IntervalView;

  { mesh.row_keys } -> std::convertible_to<typename T::RowKeyView>;
  { mesh.row_ptr } -> std::convertible_to<typename T::IndexView>;
  { mesh.intervals } -> std::convertible_to<typename T::IntervalView>;
  { mesh.num_rows } -> std::convertible_to<std::size_t>;
  { mesh.num_intervals } -> std::convertible_to<std::size_t>;
};

/**
 * @brief Concept for a 3D mesh type.
 */
template<typename T>
concept Mesh3D = requires(T mesh) {
  typename T::RowKey;
  requires std::same_as<decltype(T::DIM), const int&> && T::DIM == 3;
  typename T::RowKeyView;
  typename T::IndexView;
  typename T::IntervalView;

  { mesh.row_keys } -> std::convertible_to<typename T::RowKeyView>;
  { mesh.row_ptr } -> std::convertible_to<typename T::IndexView>;
  { mesh.intervals } -> std::convertible_to<typename T::IntervalView>;
  { mesh.num_rows } -> std::convertible_to<std::size_t>;
  { mesh.num_intervals } -> std::convertible_to<std::size_t>;
};

/**
 * @brief Concept for any mesh type (2D or 3D).
 */
template<typename T>
concept MeshAny = Mesh2D<T> || Mesh3D<T>;

// ============================================================================
// Set algebra algorithm concepts
// ============================================================================

/**
 * @brief Concept for a mesh intersection algorithm.
 *
 * An intersection algorithm must provide a static intersect_meshes
 * function that takes two meshes and returns their intersection.
 */
template<typename Algo, int DIM>
concept MeshIntersectionAlgorithm = requires() {
  // Must have a static intersect_meshes function
  requires requires(
      const Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space>& A,
      const Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space>& B) {
    { Algo::intersect_meshes(A, B) } ->
      std::same_as<Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space>>;
  };
};

// ============================================================================
// Algorithm traits
// ============================================================================

/**
 * @brief Traits class for algorithm metadata.
 *
 * Specialize this for each algorithm to provide metadata like:
 * - name: Human-readable name
 * - complexity: Algorithmic complexity
 * - description: Short description
 */
template<typename Algo>
struct AlgorithmTraits;

/**
 * @brief Helper to get algorithm name.
 */
template<typename Algo>
inline constexpr auto algorithm_name = AlgorithmTraits<Algo>::name;

/**
 * @brief Helper to get algorithm complexity.
 */
template<typename Algo>
inline constexpr auto algorithm_complexity = AlgorithmTraits<Algo>::complexity;

// ============================================================================
// Algorithm registry (for runtime selection if needed)
// ============================================================================

/**
 * @brief Type-erased wrapper for intersection algorithms.
 *
 * This allows runtime selection of algorithms while maintaining
 * compile-time type safety through the concept constraints.
 */
template<int DIM>
class IntersectionAlgorithmWrapper {
public:
  using MeshType = Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space>;

  virtual ~IntersectionAlgorithmWrapper() = default;

  virtual MeshType intersect(const MeshType& A, const MeshType& B) const = 0;
  virtual std::string name() const = 0;
};

/**
 * @brief Concrete wrapper for a specific algorithm.
 */
template<int DIM, MeshIntersectionAlgorithm<DIM> Algo>
class ConcreteIntersectionAlgorithm : public IntersectionAlgorithmWrapper<DIM> {
public:
  using MeshType = Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space>;

  MeshType intersect(const MeshType& A, const MeshType& B) const override {
    return Algo::intersect_meshes(A, B);
  }

  std::string name() const override {
    if constexpr (DIM == 2) {
      return std::string("v1_2d");
    } else {
      return std::string("v1_3d");
    }
  }
};

} // namespace subsetix::experimental::concepts

// ============================================================================
// Algorithm traits for v1
// ============================================================================

namespace subsetix::experimental::v1 {

// v1 algorithm metadata (can be used for runtime reporting)
inline constexpr const char* algorithm_name = "v1";
inline constexpr const char* algorithm_complexity = "O(n + m) where n,m are row counts";
inline constexpr const char* algorithm_description =
    "Original subsetix_kokkos_2 intersection algorithm using "
    "two-pointer merge with row mapping via binary search";

} // namespace subsetix::experimental::v1
