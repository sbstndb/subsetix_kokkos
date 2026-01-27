// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include <playground/subsetix/csr/intersection/types.hpp>

namespace playground::subsetix::csr {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using DeviceMemorySpace = ExecSpace::memory_space;
using HostMemorySpace = Kokkos::HostSpace;

template <class CoordType, class IndexType = std::size_t>
struct FieldInterval {
  CoordType begin = 0;
  CoordType end = 0;
  IndexType value_offset = 0;

  KOKKOS_INLINE_FUNCTION
  CoordType size() const { return end - begin; }
};

template <int DIM, class CoordType>
using RowKeyForDim = std::conditional_t<DIM == 2,
                                       intersection::RowKey2D<CoordType>,
                                       intersection::RowKey3D<CoordType>>;

template <int DIM, class CoordType>
using IntervalForDim = intersection::Interval<CoordType>;

template <int DIM, class MemorySpace, class CoordType = int32_t,
          class IndexType = std::size_t>
struct FieldMesh {
  static constexpr int dim_value = DIM;
  using coord_type = CoordType;
  using index_type = IndexType;
  using memory_space = MemorySpace;

  using RowKey = RowKeyForDim<DIM, CoordType>;
  using Interval = IntervalForDim<DIM, CoordType>;

  using RowKeyView = Kokkos::View<RowKey*, MemorySpace>;
  using IndexView = Kokkos::View<IndexType*, MemorySpace>;
  using IntervalView = Kokkos::View<Interval*, MemorySpace>;

  RowKeyView row_keys;
  IndexView row_ptr;
  IntervalView intervals;
  IndexView cell_offsets;

  std::size_t num_rows = 0;
  std::size_t num_intervals = 0;
  IndexType total_cells = 0;
};

template <int DIM, class MemorySpace, class CoordType = int32_t,
          class IndexType = std::size_t>
struct CsrMesh {
  static constexpr int dim_value = DIM;
  using coord_type = CoordType;
  using index_type = IndexType;
  using memory_space = MemorySpace;

  using RowKey = RowKeyForDim<DIM, CoordType>;
  using Interval = IntervalForDim<DIM, CoordType>;

  using RowKeyView = Kokkos::View<RowKey*, MemorySpace>;
  using IndexView = Kokkos::View<IndexType*, MemorySpace>;
  using IntervalView = Kokkos::View<Interval*, MemorySpace>;

  RowKeyView row_keys;
  IndexView row_ptr;
  IntervalView intervals;

  std::size_t num_rows = 0;
  std::size_t num_intervals = 0;
};

template <int DIM, class ValueType = float, class MemorySpace = DeviceMemorySpace,
          class CoordType = int32_t, class IndexType = std::size_t>
struct Field {
  static constexpr int dim_value = DIM;
  using value_type = ValueType;
  using memory_space = MemorySpace;
  using coord_type = CoordType;
  using index_type = IndexType;

  using Mesh = FieldMesh<DIM, MemorySpace, CoordType, IndexType>;
  using ValueView = Kokkos::View<ValueType*, MemorySpace>;

  Mesh mesh;
  ValueView values;

  KOKKOS_INLINE_FUNCTION
  std::size_t size() const {
    return static_cast<std::size_t>(mesh.total_cells);
  }
};

template <int DIM, class CoordType = int32_t, class IndexType = std::size_t>
struct CsrMeshHost;

template <class CoordType, class IndexType>
struct CsrMeshHost<2, CoordType, IndexType> {
  using coord_type = CoordType;
  using index_type = IndexType;
  using RowKey = intersection::RowKey2D<CoordType>;
  using Interval = intersection::Interval<CoordType>;

  std::vector<RowKey> row_keys;
  std::vector<IndexType> row_ptr;
  std::vector<Interval> intervals;

  CsrMeshHost() { row_ptr.push_back(0); }

  std::size_t num_rows() const { return row_keys.size(); }
  std::size_t num_intervals() const { return intervals.size(); }

  void append_interval(CoordType y, CoordType begin, CoordType end) {
    if (!(begin < end)) return;

    if (row_keys.empty() || row_keys.back().y != y) {
      if (!row_keys.empty() && !(row_keys.back().y < y)) {
        return;
      }
      row_keys.push_back(RowKey{y});
      row_ptr.push_back(static_cast<IndexType>(intervals.size()));
    } else {
      if (!intervals.empty()) {
        const CoordType last_begin = intervals.back().begin;
        if (!(last_begin <= begin)) {
          return;
        }
      }
    }

    intervals.push_back(Interval{begin, end});
    row_ptr.back() = static_cast<IndexType>(intervals.size());
  }
};

template <class CoordType, class IndexType>
struct CsrMeshHost<3, CoordType, IndexType> {
  using coord_type = CoordType;
  using index_type = IndexType;
  using RowKey = intersection::RowKey3D<CoordType>;
  using Interval = intersection::Interval<CoordType>;

  std::vector<RowKey> row_keys;
  std::vector<IndexType> row_ptr;
  std::vector<Interval> intervals;

  CsrMeshHost() { row_ptr.push_back(0); }

  std::size_t num_rows() const { return row_keys.size(); }
  std::size_t num_intervals() const { return intervals.size(); }

  void append_interval(CoordType y, CoordType z, CoordType begin, CoordType end) {
    if (!(begin < end)) return;

    const RowKey key{y, z};
    if (row_keys.empty() || row_keys.back() != key) {
      if (!row_keys.empty() && !(row_keys.back() < key)) {
        return;
      }
      row_keys.push_back(key);
      row_ptr.push_back(static_cast<IndexType>(intervals.size()));
    } else {
      if (!intervals.empty()) {
        const CoordType last_begin = intervals.back().begin;
        if (!(last_begin <= begin)) {
          return;
        }
      }
    }

    intervals.push_back(Interval{begin, end});
    row_ptr.back() = static_cast<IndexType>(intervals.size());
  }
};

template <int DIM, class ValueType, class CoordType = int32_t,
          class IndexType = std::size_t>
struct IntervalFieldHost;

template <class ValueType, class CoordType, class IndexType>
struct IntervalFieldHost<2, ValueType, CoordType, IndexType> {
  using value_type = ValueType;
  using coord_type = CoordType;
  using index_type = IndexType;
  using RowKey = intersection::RowKey2D<CoordType>;

  std::vector<RowKey> row_keys;
  std::vector<IndexType> row_ptr;
  std::vector<FieldInterval<CoordType, IndexType>> intervals;
  std::vector<ValueType> values;

  IntervalFieldHost() { row_ptr.push_back(0); }

  std::size_t num_rows() const { return row_keys.size(); }
  std::size_t num_intervals() const { return intervals.size(); }
  std::size_t value_count() const { return values.size(); }

  void append_interval(CoordType y, CoordType begin,
                       const std::vector<ValueType>& vals) {
    if (vals.empty()) return;

    const CoordType end =
        static_cast<CoordType>(begin + static_cast<CoordType>(vals.size()));

    if (row_keys.empty() || row_keys.back().y != y) {
      if (!row_keys.empty() && !(row_keys.back().y < y)) {
        return;
      }
      row_keys.push_back(RowKey{y});
      row_ptr.push_back(static_cast<IndexType>(intervals.size()));
    } else {
      if (!intervals.empty()) {
        const CoordType last_begin = intervals.back().begin;
        if (!(last_begin <= begin)) {
          return;
        }
      }
    }

    const IndexType offset = static_cast<IndexType>(values.size());
    intervals.push_back(FieldInterval<CoordType, IndexType>{begin, end, offset});
    values.insert(values.end(), vals.begin(), vals.end());
    row_ptr.back() = static_cast<IndexType>(intervals.size());
  }
};

template <class ValueType, class CoordType, class IndexType>
struct IntervalFieldHost<3, ValueType, CoordType, IndexType> {
  using value_type = ValueType;
  using coord_type = CoordType;
  using index_type = IndexType;
  using RowKey = intersection::RowKey3D<CoordType>;

  std::vector<RowKey> row_keys;
  std::vector<IndexType> row_ptr;
  std::vector<FieldInterval<CoordType, IndexType>> intervals;
  std::vector<ValueType> values;

  IntervalFieldHost() { row_ptr.push_back(0); }

  std::size_t num_rows() const { return row_keys.size(); }
  std::size_t num_intervals() const { return intervals.size(); }
  std::size_t value_count() const { return values.size(); }

  void append_interval(CoordType y, CoordType z, CoordType begin,
                       const std::vector<ValueType>& vals) {
    if (vals.empty()) return;

    const CoordType end =
        static_cast<CoordType>(begin + static_cast<CoordType>(vals.size()));

    const RowKey key{y, z};

    if (row_keys.empty() || row_keys.back() != key) {
      if (!row_keys.empty() && !(row_keys.back() < key)) {
        return;
      }
      row_keys.push_back(key);
      row_ptr.push_back(static_cast<IndexType>(intervals.size()));
    } else {
      if (!intervals.empty()) {
        const CoordType last_begin = intervals.back().begin;
        if (!(last_begin <= begin)) {
          return;
        }
      }
    }

    const IndexType offset = static_cast<IndexType>(values.size());
    intervals.push_back(FieldInterval<CoordType, IndexType>{begin, end, offset});
    values.insert(values.end(), vals.begin(), vals.end());
    row_ptr.back() = static_cast<IndexType>(intervals.size());
  }
};

namespace detail {

template <class IntervalView, class OffsetView, class TotalView>
inline void compute_cell_offsets_device(const IntervalView& intervals,
                                        OffsetView& offsets,
                                        TotalView& total_cells,
                                        std::size_t num_intervals) {
  using IndexType = std::remove_reference_t<decltype(offsets(0))>;

  if (num_intervals == 0) {
    return;
  }

  Kokkos::parallel_scan(
      "playground_fields_scan_cell_offsets",
      Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(num_intervals)),
      KOKKOS_LAMBDA(const int i, IndexType& update, const bool final_pass) {
        const auto iv = intervals(static_cast<std::size_t>(i));
        const IndexType len = static_cast<IndexType>(iv.end - iv.begin);
        if (final_pass) {
          offsets(static_cast<std::size_t>(i)) = update;
        }
        update += len;
        if (final_pass && i == static_cast<int>(num_intervals - 1)) {
          total_cells() = update;
        }
      });
  ExecSpace().fence();
}

template <class RowKeyView>
KOKKOS_INLINE_FUNCTION
int find_row_2d(const RowKeyView& rows, std::size_t num_rows,
                const auto& y) {
  std::size_t lo = 0;
  std::size_t hi = num_rows;

  while (lo < hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    if (rows(mid).y < y) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  if (lo < num_rows && rows(lo).y == y) {
    return static_cast<int>(lo);
  }
  return -1;
}

template <class RowKeyView>
KOKKOS_INLINE_FUNCTION
int find_row_3d(const RowKeyView& rows, std::size_t num_rows,
                const auto& y, const auto& z) {
  std::size_t lo = 0;
  std::size_t hi = num_rows;

  while (lo < hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    const auto key = rows(mid);
    if (key.y < y || (key.y == y && key.z < z)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  if (lo < num_rows) {
    const auto key = rows(lo);
    if (key.y == y && key.z == z) {
      return static_cast<int>(lo);
    }
  }
  return -1;
}

}  // namespace detail

template <int DIM, class CoordType = int32_t, class IndexType = std::size_t>
inline CsrMesh<DIM, DeviceMemorySpace, CoordType, IndexType>
build_device_mesh_from_host(const CsrMeshHost<DIM, CoordType, IndexType>& host,
                            const std::string& label = "playground_mesh") {
  using MeshType = CsrMesh<DIM, DeviceMemorySpace, CoordType, IndexType>;
  using RowKey = typename MeshType::RowKey;
  using Interval = typename MeshType::Interval;

  MeshType dev;

  const std::size_t num_rows = host.row_keys.size();
  const std::size_t num_row_ptr = host.row_ptr.size();
  const std::size_t num_intervals = host.intervals.size();

  if (num_rows == 0 || num_row_ptr == 0) {
    return dev;
  }

  dev.num_rows = num_rows;
  dev.num_intervals = num_intervals;

  dev.row_keys = typename MeshType::RowKeyView(label + "_row_keys", num_rows);
  dev.row_ptr = typename MeshType::IndexView(label + "_row_ptr", num_row_ptr);
  dev.intervals =
      typename MeshType::IntervalView(label + "_intervals", num_intervals);

  auto h_row_keys = Kokkos::create_mirror_view(dev.row_keys);
  auto h_row_ptr = Kokkos::create_mirror_view(dev.row_ptr);
  auto h_intervals = Kokkos::create_mirror_view(dev.intervals);

  for (std::size_t i = 0; i < num_rows; ++i) {
    h_row_keys(i) = static_cast<RowKey>(host.row_keys[i]);
  }
  for (std::size_t i = 0; i < num_row_ptr; ++i) {
    h_row_ptr(i) = host.row_ptr[i];
  }
  for (std::size_t i = 0; i < num_intervals; ++i) {
    h_intervals(i) = static_cast<Interval>(host.intervals[i]);
  }

  Kokkos::deep_copy(dev.row_keys, h_row_keys);
  Kokkos::deep_copy(dev.row_ptr, h_row_ptr);
  Kokkos::deep_copy(dev.intervals, h_intervals);

  return dev;
}

template <int DIM, class ValueType = float, class CoordType = int32_t,
          class IndexType = std::size_t>
inline Field<DIM, ValueType, DeviceMemorySpace, CoordType, IndexType>
build_device_field_from_host(const IntervalFieldHost<DIM, ValueType, CoordType, IndexType>& host,
                             const std::string& label = "playground_field") {
  using FieldType = Field<DIM, ValueType, DeviceMemorySpace, CoordType, IndexType>;
  using MeshType = typename FieldType::Mesh;
  using RowKey = typename MeshType::RowKey;
  using Interval = typename MeshType::Interval;

  FieldType dev;

  const std::size_t num_rows = host.row_keys.size();
  const std::size_t num_row_ptr = host.row_ptr.size();
  const std::size_t num_intervals = host.intervals.size();
  const std::size_t total_cells = host.values.size();

  if (num_rows == 0 || num_row_ptr == 0) {
    return dev;
  }

  MeshType mesh;
  mesh.num_rows = num_rows;
  mesh.num_intervals = num_intervals;
  mesh.total_cells = static_cast<IndexType>(total_cells);

  mesh.row_keys = typename MeshType::RowKeyView("playground_field_row_keys", num_rows);
  mesh.row_ptr = typename MeshType::IndexView("playground_field_row_ptr", num_row_ptr);
  mesh.intervals = typename MeshType::IntervalView("playground_field_intervals", num_intervals);
  mesh.cell_offsets = typename MeshType::IndexView("playground_field_cell_offsets", num_intervals);

  auto h_row_keys = Kokkos::create_mirror_view(mesh.row_keys);
  auto h_row_ptr = Kokkos::create_mirror_view(mesh.row_ptr);
  auto h_intervals = Kokkos::create_mirror_view(mesh.intervals);
  auto h_offsets = Kokkos::create_mirror_view(mesh.cell_offsets);

  for (std::size_t i = 0; i < num_rows; ++i) {
    h_row_keys(i) = static_cast<RowKey>(host.row_keys[i]);
  }
  for (std::size_t i = 0; i < num_row_ptr; ++i) {
    h_row_ptr(i) = host.row_ptr[i];
  }
  for (std::size_t i = 0; i < num_intervals; ++i) {
    const auto& fi = host.intervals[i];
    h_intervals(i) = Interval{fi.begin, fi.end};
    h_offsets(i) = fi.value_offset;
  }

  Kokkos::deep_copy(mesh.row_keys, h_row_keys);
  Kokkos::deep_copy(mesh.row_ptr, h_row_ptr);
  Kokkos::deep_copy(mesh.intervals, h_intervals);
  Kokkos::deep_copy(mesh.cell_offsets, h_offsets);

  dev.mesh = mesh;

  if (total_cells > 0) {
    dev.values = typename FieldType::ValueView(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           label.empty() ? "playground_field_values"
                                         : label + "_values"),
        total_cells);
    auto h_values = Kokkos::create_mirror_view(dev.values);
    for (std::size_t i = 0; i < total_cells; ++i) {
      h_values(i) = host.values[i];
    }
    Kokkos::deep_copy(dev.values, h_values);
  }

  return dev;
}

template <int DIM, class ValueType = float, class CoordType = int32_t,
          class IndexType = std::size_t>
inline IntervalFieldHost<DIM, ValueType, CoordType, IndexType>
build_host_field_from_device(
    const Field<DIM, ValueType, DeviceMemorySpace, CoordType, IndexType>& dev) {
  IntervalFieldHost<DIM, ValueType, CoordType, IndexType> host;

  if (dev.mesh.num_rows == 0) {
    return host;
  }

  const std::size_t num_rows = dev.mesh.num_rows;
  const std::size_t num_intervals = dev.mesh.num_intervals;

  auto h_row_keys = Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, dev.mesh.row_keys);
  auto h_row_ptr = Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, dev.mesh.row_ptr);
  auto h_intervals = Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, dev.mesh.intervals);
  auto h_offsets = Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, dev.mesh.cell_offsets);

  host.row_keys.resize(num_rows);
  for (std::size_t i = 0; i < num_rows; ++i) {
    host.row_keys[i] = h_row_keys(i);
  }

  host.row_ptr.resize(num_rows + 1);
  for (std::size_t i = 0; i < num_rows + 1; ++i) {
    host.row_ptr[i] = h_row_ptr(i);
  }

  host.intervals.resize(num_intervals);
  for (std::size_t i = 0; i < num_intervals; ++i) {
    FieldInterval<CoordType, IndexType> fi;
    fi.begin = h_intervals(i).begin;
    fi.end = h_intervals(i).end;
    fi.value_offset = h_offsets(i);
    host.intervals[i] = fi;
  }

  const std::size_t total_cells = static_cast<std::size_t>(dev.mesh.total_cells);
  host.values.resize(total_cells);
  if (total_cells > 0) {
    auto h_values = Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, dev.values);
    for (std::size_t i = 0; i < total_cells; ++i) {
      host.values[i] = h_values(i);
    }
  }

  return host;
}

template <int DIM, class ValueType = float, class CoordType = int32_t,
          class IndexType = std::size_t, class MeshLike>
inline Field<DIM, ValueType, DeviceMemorySpace, CoordType, IndexType>
make_field_from_mesh(const MeshLike& mesh,
                     const ValueType& init_value = ValueType(),
                     const std::string& label = "playground_field") {
  using FieldType = Field<DIM, ValueType, DeviceMemorySpace, CoordType, IndexType>;
  using MeshType = typename FieldType::Mesh;

  FieldType out;
  if (mesh.num_rows == 0 || mesh.num_intervals == 0) {
    return out;
  }

  MeshType geom;
  geom.num_rows = mesh.num_rows;
  geom.num_intervals = mesh.num_intervals;
  geom.row_keys = mesh.row_keys;
  geom.row_ptr = mesh.row_ptr;
  geom.intervals = mesh.intervals;
  geom.cell_offsets =
      typename MeshType::IndexView(label + "_cell_offsets", geom.num_intervals);

  Kokkos::View<IndexType, DeviceMemorySpace> total(label + "_total_cells");
  detail::compute_cell_offsets_device(geom.intervals, geom.cell_offsets, total,
                                      geom.num_intervals);
  IndexType total_cells_host = 0;
  Kokkos::deep_copy(total_cells_host, total);
  geom.total_cells = total_cells_host;

  out.mesh = geom;

  if (geom.total_cells > 0) {
    out.values = typename FieldType::ValueView(
        label.empty() ? "playground_field_values" : label + "_values",
        static_cast<std::size_t>(geom.total_cells));
    Kokkos::deep_copy(out.values, init_value);
  }

  return out;
}

}  // namespace playground::subsetix::csr
