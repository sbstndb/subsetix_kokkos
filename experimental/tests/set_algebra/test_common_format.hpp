// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef SUBSETIX_ENABLE_EXPERIMENTAL

#include <experimental/subsetix/csr/types.hpp>
#include <vector>
#include <cstdint>

namespace experimental::subsetix::csr::test {

// ============================================================================
// Common Format: CPU-side simple struct for test data
// ============================================================================

/**
 * @brief CPU-side row representation for testing (2D)
 *
 * This is a simple struct using std::vector for test data.
 * It can be easily created, compared, and validated on the host.
 *
 * @tparam CoordType The coordinate type (default: int32_t)
 */
template<class CoordType = int32_t>
struct CommonRow2D {
  using coord_type = CoordType;
  CoordType y = 0;                              // Row key (Y coordinate)
  std::vector<csr::Interval<CoordType>> intervals;   // X intervals in this row

  bool operator==(const CommonRow2D& other) const {
    return y == other.y && intervals == other.intervals;
  }

  bool operator!=(const CommonRow2D& other) const {
    return !(*this == other);
  }
};

/**
 * @brief CPU-side row representation for testing (3D)
 *
 * @tparam CoordType The coordinate type (default: int32_t)
 */
template<class CoordType = int32_t>
struct CommonRow3D {
  using coord_type = CoordType;
  CoordType y = 0;                              // Y coordinate
  CoordType z = 0;                              // Z coordinate
  std::vector<csr::Interval<CoordType>> intervals;   // X intervals in this row

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
 * @brief CPU-side mesh representation for testing (2D)
 *
 * This is the common format used for all test data.
 * It uses std::vector for easy manipulation on the host.
 *
 * @tparam CoordType The coordinate type (default: int32_t)
 */
template<class CoordType = int32_t>
struct CommonMesh2D {
  using coord_type = CoordType;
  std::vector<CommonRow2D<CoordType>> rows;

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

/**
 * @brief CPU-side mesh representation for testing (3D)
 *
 * @tparam CoordType The coordinate type (default: int32_t)
 */
template<class CoordType = int32_t>
struct CommonMesh3D {
  using coord_type = CoordType;
  std::vector<CommonRow3D<CoordType>> rows;

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
// Default type aliases (for backward compatibility)
// ============================================================================

using DefaultCommonRow2D = CommonRow2D<int32_t>;
using DefaultCommonRow3D = CommonRow3D<int32_t>;
using DefaultCommonMesh2D = CommonMesh2D<int32_t>;
using DefaultCommonMesh3D = CommonMesh3D<int32_t>;

// ============================================================================
// Bidirectional Converters: Common Format <-> Version-Specific Format
// ============================================================================

/**
 * @brief Converter between CommonMesh2D and version-specific Mesh types
 *
 * This converter works with any version's Mesh type (v1, v2, v3)
 * as long as they follow the same CSR structure.
 *
 * @tparam VersionNamespace The version namespace (v1, v2, v3)
 * @tparam MemorySpace Kokkos memory space
 * @tparam CoordType Coordinate type
 * @tparam IndexType Index type for CSR row_ptr
 */
template <template<int, class, class, class> class MeshType,
          class MemorySpace,
          class CoordType = int32_t,
          class IndexType = std::size_t>
struct MeshConverter2D {
  using DeviceMesh = MeshType<2, MemorySpace, CoordType, IndexType>;
  using CommonMesh = CommonMesh2D<CoordType>;
  using CommonRow = CommonRow2D<CoordType>;
  using RowKey = csr::RowKey2D<CoordType>;
  using Interval = csr::Interval<CoordType>;

  /**
   * @brief Convert from CommonMesh2D to DeviceMesh
   */
  static DeviceMesh from_common(const CommonMesh& common) {
    DeviceMesh mesh;
    mesh.num_rows = common.num_rows();
    mesh.num_intervals = common.num_intervals();

    if (mesh.num_rows == 0) {
      return mesh;
    }

    mesh.row_keys = Kokkos::View<RowKey*, MemorySpace>("row_keys", mesh.num_rows);
    mesh.row_ptr = Kokkos::View<IndexType*, MemorySpace>("row_ptr", mesh.num_rows + 1);
    mesh.intervals = Kokkos::View<Interval*, MemorySpace>("intervals", mesh.num_intervals);

    // Create host mirrors
    auto keys_h = Kokkos::create_mirror_view(mesh.row_keys);
    auto ptr_h = Kokkos::create_mirror_view(mesh.row_ptr);
    auto ints_h = Kokkos::create_mirror_view(mesh.intervals);

    // Fill data
    std::size_t interval_idx = 0;
    for (std::size_t i = 0; i < common.rows.size(); ++i) {
      const auto& row = common.rows[i];
      keys_h(i) = RowKey{row.y};
      ptr_h(i) = static_cast<IndexType>(interval_idx);

      for (const auto& interval : row.intervals) {
        ints_h(interval_idx++) = interval;
      }
    }
    ptr_h(common.rows.size()) = static_cast<IndexType>(interval_idx);

    // Copy to device
    Kokkos::deep_copy(mesh.row_keys, keys_h);
    Kokkos::deep_copy(mesh.row_ptr, ptr_h);
    Kokkos::deep_copy(mesh.intervals, ints_h);

    return mesh;
  }

  /**
   * @brief Convert from DeviceMesh to CommonMesh2D
   */
  static CommonMesh to_common(const DeviceMesh& mesh) {
    CommonMesh common;

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
      CommonRow row;
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
 * @brief Converter between CommonMesh3D and version-specific Mesh types
 *
 * @tparam VersionNamespace The version namespace (v1, v2, v3)
 * @tparam MemorySpace Kokkos memory space
 * @tparam CoordType Coordinate type
 * @tparam IndexType Index type for CSR row_ptr
 */
template <template<int, class, class, class> class MeshType,
          class MemorySpace,
          class CoordType = int32_t,
          class IndexType = std::size_t>
struct MeshConverter3D {
  using DeviceMesh = MeshType<3, MemorySpace, CoordType, IndexType>;
  using CommonMesh = CommonMesh3D<CoordType>;
  using CommonRow = CommonRow3D<CoordType>;
  using RowKey = csr::RowKey3D<CoordType>;
  using Interval = csr::Interval<CoordType>;

  /**
   * @brief Convert from CommonMesh3D to DeviceMesh
   */
  static DeviceMesh from_common(const CommonMesh& common) {
    DeviceMesh mesh;
    mesh.num_rows = common.num_rows();
    mesh.num_intervals = common.num_intervals();

    if (mesh.num_rows == 0) {
      return mesh;
    }

    mesh.row_keys = Kokkos::View<RowKey*, MemorySpace>("row_keys", mesh.num_rows);
    mesh.row_ptr = Kokkos::View<IndexType*, MemorySpace>("row_ptr", mesh.num_rows + 1);
    mesh.intervals = Kokkos::View<Interval*, MemorySpace>("intervals", mesh.num_intervals);

    // Create host mirrors
    auto keys_h = Kokkos::create_mirror_view(mesh.row_keys);
    auto ptr_h = Kokkos::create_mirror_view(mesh.row_ptr);
    auto ints_h = Kokkos::create_mirror_view(mesh.intervals);

    // Fill data
    std::size_t interval_idx = 0;
    for (std::size_t i = 0; i < common.rows.size(); ++i) {
      const auto& row = common.rows[i];
      keys_h(i) = RowKey{row.y, row.z};
      ptr_h(i) = static_cast<IndexType>(interval_idx);

      for (const auto& interval : row.intervals) {
        ints_h(interval_idx++) = interval;
      }
    }
    ptr_h(common.rows.size()) = static_cast<IndexType>(interval_idx);

    // Copy to device
    Kokkos::deep_copy(mesh.row_keys, keys_h);
    Kokkos::deep_copy(mesh.row_ptr, ptr_h);
    Kokkos::deep_copy(mesh.intervals, ints_h);

    return mesh;
  }

  /**
   * @brief Convert from DeviceMesh to CommonMesh3D
   */
  static CommonMesh to_common(const DeviceMesh& mesh) {
    CommonMesh common;

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
      CommonRow row;
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

// ============================================================================
// Convenience aliases for v1
// ============================================================================

namespace v1_test {
  template<class MemorySpace, class CoordType = int32_t, class IndexType = std::size_t>
  using Converter2D = MeshConverter2D<experimental::subsetix::csr::v1::Mesh, MemorySpace, CoordType, IndexType>;

  template<class MemorySpace, class CoordType = int32_t, class IndexType = std::size_t>
  using Converter3D = MeshConverter3D<experimental::subsetix::csr::v1::Mesh, MemorySpace, CoordType, IndexType>;
}

namespace v2_test {
  template<class MemorySpace, class CoordType = int32_t, class IndexType = std::size_t>
  using Converter2D = MeshConverter2D<experimental::subsetix::csr::v2::Mesh, MemorySpace, CoordType, IndexType>;

  template<class MemorySpace, class CoordType = int32_t, class IndexType = std::size_t>
  using Converter3D = MeshConverter3D<experimental::subsetix::csr::v2::Mesh, MemorySpace, CoordType, IndexType>;
}

namespace v3_test {
  template<class MemorySpace, class CoordType = int32_t, class IndexType = std::size_t>
  using Converter2D = MeshConverter2D<experimental::subsetix::csr::v3::Mesh, MemorySpace, CoordType, IndexType>;

  template<class MemorySpace, class CoordType = int32_t, class IndexType = std::size_t>
  using Converter3D = MeshConverter3D<experimental::subsetix::csr::v3::Mesh, MemorySpace, CoordType, IndexType>;
}

} // namespace experimental::subsetix::csr::test

#endif
