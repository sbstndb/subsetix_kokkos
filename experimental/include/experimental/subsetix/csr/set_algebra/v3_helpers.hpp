// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <experimental/subsetix/csr/mesh.hpp>
#include <Kokkos_Core.hpp>

namespace experimental::subsetix::csr::v3::detail {

// Import RowHashMap from v2::detail
using RowHashMap2D = experimental::subsetix::csr::v2::detail::RowHashMap<RowKey2D, Kokkos::DefaultExecutionSpace::memory_space>;
using RowHashMap3D = experimental::subsetix::csr::v2::detail::RowHashMap<RowKey3D, Kokkos::DefaultExecutionSpace::memory_space>;

// ============================================================================
// Bounding Box Structures and Utilities
// ============================================================================

struct BoundingBox2D {
  Coord y_min, y_max;

  KOKKOS_INLINE_FUNCTION
  BoundingBox2D() : y_min(0), y_max(0) {}
};

struct BoundingBox3D {
  Coord y_min, y_max;
  Coord z_min, z_max;

  KOKKOS_INLINE_FUNCTION
  BoundingBox3D() : y_min(0), y_max(0), z_min(0), z_max(0) {}
};

// Compute bounding box for 2D mesh
template <class MemorySpace>
inline BoundingBox2D compute_mesh_bbox(const Mesh<2, MemorySpace>& mesh) {
  using ExecSpace = Kokkos::DefaultExecutionSpace;

  BoundingBox2D bbox;
  auto bbox_y_min = Kokkos::View<Coord, MemorySpace>("bbox_y_min");
  auto bbox_y_max = Kokkos::View<Coord, MemorySpace>("bbox_y_max");

  // Initialize with first row's y coordinate
  if (mesh.num_rows > 0) {
    auto row_keys_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, mesh.row_keys);
    const Coord first_y = row_keys_host(0).y;

    Kokkos::deep_copy(bbox_y_min, first_y);
    Kokkos::deep_copy(bbox_y_max, first_y);
  }

  // Find min/max y coordinates
  Kokkos::parallel_reduce(
      "compute_bbox_2d",
      Kokkos::RangePolicy<ExecSpace>(0, mesh.num_rows),
      KOKKOS_LAMBDA(const std::size_t i, Coord& y_min, Coord& y_max) {
        const Coord y = mesh.row_keys(i).y;
        if (y < y_min) y_min = y;
        if (y > y_max) y_max = y;
      },
      Kokkos::Min<Coord>(bbox_y_min),
      Kokkos::Max<Coord>(bbox_y_max));

  Kokkos::fence();

  auto y_min_h = Kokkos::create_mirror_view(bbox_y_min);
  auto y_max_h = Kokkos::create_mirror_view(bbox_y_max);
  Kokkos::deep_copy(y_min_h, bbox_y_min);
  Kokkos::deep_copy(y_max_h, bbox_y_max);

  bbox.y_min = y_min_h();
  bbox.y_max = y_max_h();

  return bbox;
}

// Compute bounding box for 3D mesh
template <class MemorySpace>
inline BoundingBox3D compute_mesh_bbox(const Mesh<3, MemorySpace>& mesh) {
  using ExecSpace = Kokkos::DefaultExecutionSpace;

  BoundingBox3D bbox;
  auto bbox_y_min = Kokkos::View<Coord, MemorySpace>("bbox_y_min");
  auto bbox_y_max = Kokkos::View<Coord, MemorySpace>("bbox_y_max");
  auto bbox_z_min = Kokkos::View<Coord, MemorySpace>("bbox_z_min");
  auto bbox_z_max = Kokkos::View<Coord, MemorySpace>("bbox_z_max");

  // Initialize with first row's coordinates
  if (mesh.num_rows > 0) {
    auto row_keys_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, mesh.row_keys);
    const RowKey3D first_key = row_keys_host(0);

    Kokkos::deep_copy(bbox_y_min, first_key.y);
    Kokkos::deep_copy(bbox_y_max, first_key.y);
    Kokkos::deep_copy(bbox_z_min, first_key.z);
    Kokkos::deep_copy(bbox_z_max, first_key.z);
  }

  // Find min/max coordinates
  Kokkos::parallel_reduce(
      "compute_bbox_3d",
      Kokkos::RangePolicy<ExecSpace>(0, mesh.num_rows),
      KOKKOS_LAMBDA(const std::size_t i,
                    Coord& y_min, Coord& y_max,
                    Coord& z_min, Coord& z_max) {
        const RowKey3D key = mesh.row_keys(i);
        if (key.y < y_min) y_min = key.y;
        if (key.y > y_max) y_max = key.y;
        if (key.z < z_min) z_min = key.z;
        if (key.z > z_max) z_max = key.z;
      },
      Kokkos::Min<Coord>(bbox_y_min),
      Kokkos::Max<Coord>(bbox_y_max),
      Kokkos::Min<Coord>(bbox_z_min),
      Kokkos::Max<Coord>(bbox_z_max));

  Kokkos::fence();

  auto y_min_h = Kokkos::create_mirror_view(bbox_y_min);
  auto y_max_h = Kokkos::create_mirror_view(bbox_y_max);
  auto z_min_h = Kokkos::create_mirror_view(bbox_z_min);
  auto z_max_h = Kokkos::create_mirror_view(bbox_z_max);

  Kokkos::deep_copy(y_min_h, bbox_y_min);
  Kokkos::deep_copy(y_max_h, bbox_y_max);
  Kokkos::deep_copy(z_min_h, bbox_z_min);
  Kokkos::deep_copy(z_max_h, bbox_z_max);

  bbox.y_min = y_min_h();
  bbox.y_max = y_max_h();
  bbox.z_min = z_min_h();
  bbox.z_max = z_max_h();

  return bbox;
}

// Check if 2D bounding boxes overlap
KOKKOS_INLINE_FUNCTION
bool bboxes_overlap(const BoundingBox2D& a, const BoundingBox2D& b) {
  return !(a.y_max < b.y_min || b.y_max < a.y_min);
}

// Check if 3D bounding boxes overlap
KOKKOS_INLINE_FUNCTION
bool bboxes_overlap(const BoundingBox3D& a, const BoundingBox3D& b) {
  return !(a.y_max < b.y_min || b.y_max < a.y_min ||
           a.z_max < b.z_min || b.z_max < a.z_min);
}

// Overload for mesh types (deduces dimension)
template <int DIM, class MemorySpace>
auto compute_mesh_bbox(const Mesh<DIM, MemorySpace>& mesh) {
  if constexpr (DIM == 2) {
    return compute_mesh_bbox(static_cast<const Mesh<2, MemorySpace>&>(mesh));
  } else {
    return compute_mesh_bbox(static_cast<const Mesh<3, MemorySpace>&>(mesh));
  }
}

// ============================================================================
// Hash Map Building Utilities
// ============================================================================

// Helper to build hash map from mesh (serial build on host) - 2D version
template <class MemorySpace>
inline void build_hash_map_for_mesh(
    const Mesh<2, MemorySpace>& mesh,
    RowHashMap2D& hash_map_out,
    const std::string& label = "row_hash_map") {

  hash_map_out.reserve(mesh.num_rows, label);

  // Copy row keys to host for building
  auto row_keys_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, mesh.row_keys);

  for (std::size_t i = 0; i < mesh.num_rows; ++i) {
    RowKey2D key;
    key.y = row_keys_host(i).y;
    hash_map_out.insert(key, static_cast<int>(i));
  }
}

// Helper to build hash map from mesh (serial build on host) - 3D version
template <class MemorySpace>
inline void build_hash_map_for_mesh(
    const Mesh<3, MemorySpace>& mesh,
    RowHashMap3D& hash_map_out,
    const std::string& label = "row_hash_map") {

  hash_map_out.reserve(mesh.num_rows, label);

  // Copy row keys to host for building
  auto row_keys_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, mesh.row_keys);

  for (std::size_t i = 0; i < mesh.num_rows; ++i) {
    RowKey3D key;
    key.y = row_keys_host(i).y;
    key.z = row_keys_host(i).z;
    hash_map_out.insert(key, static_cast<int>(i));
  }
}

} // namespace experimental::subsetix::csr::v3::detail
