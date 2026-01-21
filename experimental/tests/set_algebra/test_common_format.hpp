// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <experimental/subsetix/csr/mesh.hpp>
#include <vector>
#include <cstdint>

namespace experimental::subsetix::csr::test {

// ============================================================================
// Common Format: CPU-side simple struct for test data
// ============================================================================

/**
 * @brief CPU-side row representation for testing
 *
 * This is a simple struct using std::vector for test data.
 * It can be easily created, compared, and validated on the host.
 */
struct CommonRow2D {
  Coord y = 0;                       // Row key (Y coordinate)
  std::vector<Interval> intervals;   // X intervals in this row

  bool operator==(const CommonRow2D& other) const {
    return y == other.y && intervals == other.intervals;
  }

  bool operator!=(const CommonRow2D& other) const {
    return !(*this == other);
  }
};

struct CommonRow3D {
  Coord y = 0;                       // Y coordinate
  Coord z = 0;                       // Z coordinate
  std::vector<Interval> intervals;   // X intervals in this row

  bool operator==(const CommonRow3D& other) const {
    return y == other.y && z == other.z && intervals == other.intervals;
  }

  bool operator!=(const CommonRow3D& other) const {
    return !(*this == other);
  }

  bool operator<(const CommonRow3D& other) const {
    if (y != other.y) return y < other.y;
    return z < other.z;
  }
};

/**
 * @brief CPU-side mesh representation for testing
 *
 * This is the common format used for all test data.
 * It uses std::vector for easy manipulation on the host.
 */
struct CommonMesh2D {
  std::vector<CommonRow2D> rows;

  std::size_t num_rows() const { return rows.size(); }

  std::size_t num_intervals() const {
    std::size_t total = 0;
    for (const auto& row : rows) {
      total += row.intervals.size();
    }
    return total;
  }

  bool operator==(const CommonMesh2D& other) const {
    return rows == other.rows;
  }

  bool operator!=(const CommonMesh2D& other) const {
    return !(*this == other);
  }
};

struct CommonMesh3D {
  std::vector<CommonRow3D> rows;

  std::size_t num_rows() const { return rows.size(); }

  std::size_t num_intervals() const {
    std::size_t total = 0;
    for (const auto& row : rows) {
      total += row.intervals.size();
    }
    return total;
  }

  bool operator==(const CommonMesh3D& other) const {
    return rows == other.rows;
  }

  bool operator!=(const CommonMesh3D& other) const {
    return !(*this == other);
  }
};

// ============================================================================
// Bidirectional Converters: Common Format <-> Version-Specific Format
// ============================================================================

/**
 * @brief Converter between CommonMesh2D and Mesh<DIM, MemorySpace>
 */
template <class MemorySpace>
struct MeshConverter2D {
  using DeviceMesh = Mesh<2, MemorySpace>;

  /**
   * @brief Convert from CommonMesh2D to DeviceMesh
   */
  static DeviceMesh from_common(const CommonMesh2D& common) {
    DeviceMesh mesh;
    mesh.num_rows = common.num_rows();
    mesh.num_intervals = common.num_intervals();

    if (mesh.num_rows == 0) {
      return mesh;
    }

    mesh.row_keys = typename DeviceMesh::RowKeyView("row_keys", mesh.num_rows);
    mesh.row_ptr = typename DeviceMesh::IndexView("row_ptr", mesh.num_rows + 1);
    mesh.intervals = typename DeviceMesh::IntervalView("intervals", mesh.num_intervals);

    // Create host mirrors
    auto keys_h = Kokkos::create_mirror_view(mesh.row_keys);
    auto ptr_h = Kokkos::create_mirror_view(mesh.row_ptr);
    auto ints_h = Kokkos::create_mirror_view(mesh.intervals);

    // Fill data
    std::size_t interval_idx = 0;
    for (std::size_t i = 0; i < common.rows.size(); ++i) {
      const auto& row = common.rows[i];
      keys_h(i) = RowKey2D{row.y};
      ptr_h(i) = interval_idx;

      for (const auto& interval : row.intervals) {
        ints_h(interval_idx++) = interval;
      }
    }
    ptr_h(common.rows.size()) = interval_idx;

    // Copy to device
    Kokkos::deep_copy(mesh.row_keys, keys_h);
    Kokkos::deep_copy(mesh.row_ptr, ptr_h);
    Kokkos::deep_copy(mesh.intervals, ints_h);

    return mesh;
  }

  /**
   * @brief Convert from DeviceMesh to CommonMesh2D
   */
  static CommonMesh2D to_common(const DeviceMesh& mesh) {
    CommonMesh2D common;

    if (mesh.num_rows == 0) {
      return common;
    }

    // Copy device data to host
    auto keys_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.row_keys);
    auto ptr_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.row_ptr);
    auto ints_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.intervals);

    // Build common mesh
    common.rows.reserve(mesh.num_rows);
    for (std::size_t i = 0; i < mesh.num_rows; ++i) {
      CommonRow2D row;
      row.y = keys_h(i).y;

      std::size_t start = ptr_h(i);
      std::size_t end = ptr_h(i + 1);
      row.intervals.reserve(end - start);

      for (std::size_t j = start; j < end; ++j) {
        row.intervals.push_back(ints_h(j));
      }

      common.rows.push_back(std::move(row));
    }

    return common;
  }
};

/**
 * @brief Converter between CommonMesh3D and Mesh<DIM, MemorySpace>
 */
template <class MemorySpace>
struct MeshConverter3D {
  using DeviceMesh = Mesh<3, MemorySpace>;

  /**
   * @brief Convert from CommonMesh3D to DeviceMesh
   */
  static DeviceMesh from_common(const CommonMesh3D& common) {
    DeviceMesh mesh;
    mesh.num_rows = common.num_rows();
    mesh.num_intervals = common.num_intervals();

    if (mesh.num_rows == 0) {
      return mesh;
    }

    mesh.row_keys = typename DeviceMesh::RowKeyView("row_keys", mesh.num_rows);
    mesh.row_ptr = typename DeviceMesh::IndexView("row_ptr", mesh.num_rows + 1);
    mesh.intervals = typename DeviceMesh::IntervalView("intervals", mesh.num_intervals);

    // Create host mirrors
    auto keys_h = Kokkos::create_mirror_view(mesh.row_keys);
    auto ptr_h = Kokkos::create_mirror_view(mesh.row_ptr);
    auto ints_h = Kokkos::create_mirror_view(mesh.intervals);

    // Fill data
    std::size_t interval_idx = 0;
    for (std::size_t i = 0; i < common.rows.size(); ++i) {
      const auto& row = common.rows[i];
      keys_h(i) = RowKey3D{row.y, row.z};
      ptr_h(i) = interval_idx;

      for (const auto& interval : row.intervals) {
        ints_h(interval_idx++) = interval;
      }
    }
    ptr_h(common.rows.size()) = interval_idx;

    // Copy to device
    Kokkos::deep_copy(mesh.row_keys, keys_h);
    Kokkos::deep_copy(mesh.row_ptr, ptr_h);
    Kokkos::deep_copy(mesh.intervals, ints_h);

    return mesh;
  }

  /**
   * @brief Convert from DeviceMesh to CommonMesh3D
   */
  static CommonMesh3D to_common(const DeviceMesh& mesh) {
    CommonMesh3D common;

    if (mesh.num_rows == 0) {
      return common;
    }

    // Copy device data to host
    auto keys_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.row_keys);
    auto ptr_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.row_ptr);
    auto ints_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.intervals);

    // Build common mesh
    common.rows.reserve(mesh.num_rows);
    for (std::size_t i = 0; i < mesh.num_rows; ++i) {
      CommonRow3D row;
      row.y = keys_h(i).y;
      row.z = keys_h(i).z;

      std::size_t start = ptr_h(i);
      std::size_t end = ptr_h(i + 1);
      row.intervals.reserve(end - start);

      for (std::size_t j = start; j < end; ++j) {
        row.intervals.push_back(ints_h(j));
      }

      common.rows.push_back(std::move(row));
    }

    return common;
  }
};

} // namespace experimental::subsetix::csr::test

#endif
