// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#include <experimental/subsetix/csr/set_algebra/v3.hpp>
#include <vector>

namespace experimental::subsetix::csr::successive::naive {

/**
 * @brief Naive successive intersection for multiple meshes.
 *
 * This baseline implementation intersects meshes one pair at a time,
 * allocating a new mesh for each intermediate result.
 *
 * Algorithm:
 *   result = meshes[0]
 *   for i = 1 to n-1:
 *     result = intersect(result, meshes[i])
 *     if result is empty: break (early exit)
 *   return result
 *
 * @tparam DIM Dimension (2 for 2D, 3 for 3D)
 * @tparam MemorySpace Kokkos memory space
 * @tparam CoordType Coordinate type
 * @tparam IndexType Index type
 * @param meshes Vector of meshes to intersect
 * @return Intersection of all input meshes
 */
template <int DIM, class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
          class CoordType = int32_t, class IndexType = std::size_t>
inline v3::Mesh<DIM, MemorySpace, CoordType, IndexType>
intersect(const std::vector<v3::Mesh<DIM, MemorySpace, CoordType, IndexType>>& meshes) {
  using MeshType = v3::Mesh<DIM, MemorySpace, CoordType, IndexType>;

  // Edge case: empty vector
  if (meshes.empty()) {
    return MeshType{};
  }

  // Edge case: single mesh - return a copy
  if (meshes.size() == 1) {
    return v3::mesh_to<DIM, CoordType, IndexType, MemorySpace, MemorySpace>(meshes[0]);
  }

  // Start with the first mesh
  MeshType result = v3::mesh_to<DIM, CoordType, IndexType, MemorySpace, MemorySpace>(meshes[0]);

  // Successively intersect with each remaining mesh
  for (std::size_t i = 1; i < meshes.size(); ++i) {
    // Early exit if result is already empty
    if (result.num_rows == 0 || result.num_intervals == 0) {
      return MeshType{};
    }

    // Intersect with next mesh
    result = v3::intersect_meshes<DIM, CoordType, IndexType>(result, meshes[i]);
  }

  return result;
}

} // namespace experimental::subsetix::csr::successive::naive
