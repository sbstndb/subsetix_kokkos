#pragma once

#include <cstddef>
#include <vector>
#include <string>

#include <Kokkos_Core.hpp>

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_interval_subset.hpp>

namespace subsetix {
namespace csr {

/**
 * @brief Per-interval metadata for fields: geometry + offset into values array.
 *
 * Each interval stores its [begin, end) coordinates on X and the starting
 * offset of its cell values in the linear values buffer.
 */
struct FieldInterval {
  Coord begin = 0;
  Coord end = 0;
  std::size_t value_offset = 0;

  Coord size() const { return end - begin; }
};

/**
 * @brief Host-side sparse 2D field using CSR-of-intervals layout.
 *
 * Layout:
 *   - row_keys: one entry per non-empty row (Y coordinate)
 *   - row_ptr: CSR pointers into intervals (size = num_rows + 1)
 *   - intervals: FieldInterval entries (geometry + offset into values)
 *   - values: concatenation of all cell values (per interval, per row)
 *
 * Intervals must be appended in non-decreasing Y, and for a given Y in
 * non-decreasing X (begin). Intervals within a row are expected to be
 * non-overlapping.
 */
template <typename T>
struct IntervalField2DHost {
  std::vector<RowKey2D> row_keys;
  std::vector<std::size_t> row_ptr;
  std::vector<FieldInterval> intervals;
  std::vector<T> values;

  IntervalField2DHost() {
    row_ptr.push_back(0);
  }

  std::size_t num_rows() const { return row_keys.size(); }
  std::size_t num_intervals() const { return intervals.size(); }
  std::size_t value_count() const { return values.size(); }

  /**
   * @brief Append an interval [begin, begin+vals.size()) on row y with values.
   *
   * Rows must be appended in strictly increasing Y. For a given row, X
   * coordinates must be non-decreasing.
   */
  void append_interval(Coord y, Coord begin, const std::vector<T>& vals) {
    if (vals.empty()) {
      return;
    }

    const Coord end = static_cast<Coord>(begin + static_cast<Coord>(vals.size()));

    if (row_keys.empty() || row_keys.back().y != y) {
      // New row: enforce increasing Y and start a new CSR row.
      if (!row_keys.empty()) {
        if (!(row_keys.back().y < y)) {
          // Invalid ordering: ignore in release builds.
          return;
        }
      }
      row_keys.push_back(RowKey2D{y});
      row_ptr.push_back(intervals.size());
    } else {
      // Same row: enforce non-decreasing X.
      const std::size_t last_idx = intervals.empty() ? 0 : intervals.size() - 1;
      if (!intervals.empty()) {
        const Coord last_begin = intervals[last_idx].begin;
        if (!(last_begin <= begin)) {
          return;
        }
      }
    }

    const std::size_t offset = values.size();
    FieldInterval fi;
    fi.begin = begin;
    fi.end = end;
    fi.value_offset = offset;
    intervals.push_back(fi);

    values.insert(values.end(), vals.begin(), vals.end());

    // Update CSR pointer for current row to point past all intervals added so far.
    row_ptr.back() = intervals.size();
  }
};

/**
 * @brief Lightweight field wrapper that shares geometry with IntervalSet2DView.
 *
 * The geometry is provided by an IntervalSet2DView, while values are stored
 * in a contiguous Kokkos::View sized to geometry.total_cells.
 */
template <typename T, class MemorySpace = DeviceMemorySpace>
struct Field2D {
  using GeometryView = IntervalSet2DView<MemorySpace>;
  using RowKeyView = typename GeometryView::RowKeyView;
  using IndexView = typename GeometryView::IndexView;
  using IntervalView = typename GeometryView::IntervalView;
  using ValueView = Kokkos::View<T*, MemorySpace>;

  GeometryView geometry;
  ValueView values;

  Field2D() = default;

  Field2D(const GeometryView& geom, const std::string& name = "subsetix_field")
      : geometry(geom) {
    const std::size_t value_count = geometry.total_cells;
    if (value_count == 0) {
      values = ValueView();
    } else {
      const std::string view_label =
          name.empty() ? "subsetix_field_values" : name + "_values";
      values = ValueView(view_label, value_count);
    }
  }

  KOKKOS_INLINE_FUNCTION
  std::size_t size() const { return geometry.total_cells; }

  KOKKOS_INLINE_FUNCTION
  T& at(std::size_t interval_idx, Coord x) const {
    const Interval iv = geometry.intervals(interval_idx);
    const std::size_t base = geometry.cell_offsets(interval_idx);
    return values(base + static_cast<std::size_t>(x - iv.begin));
  }
};

template <typename T>
using Field2DDevice = Field2D<T, DeviceMemorySpace>;

template <typename T>
using Field2DHost = Field2D<T, HostMemorySpace>;

/**
 * @brief Build a device field from a host field.
 */
template <typename T>
inline Field2DDevice<T>
build_device_field_from_host(const IntervalField2DHost<T>& host,
                             const std::string& label = "subsetix_field") {
  Field2DDevice<T> dev;

  const std::size_t num_rows = host.row_keys.size();
  const std::size_t num_row_ptr = host.row_ptr.size();
  const std::size_t num_intervals = host.intervals.size();
  const std::size_t value_count = host.values.size();

  if (num_rows == 0 || num_row_ptr == 0) {
    return dev;
  }

  IntervalSet2DHost geom_host;
  geom_host.row_keys = host.row_keys;
  geom_host.row_ptr = host.row_ptr;
  geom_host.intervals.reserve(num_intervals);
  for (const FieldInterval& fi : host.intervals) {
    geom_host.intervals.push_back(Interval{fi.begin, fi.end});
  }
  geom_host.rebuild_mapping();

  dev.geometry = build_device_from_host(geom_host);

  if (value_count > 0) {
    dev.values = typename Field2DDevice<T>::ValueView(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           label.empty() ? "subsetix_csr_field_values"
                                         : label + "_values"),
        value_count);
    Kokkos::View<T*, HostMemorySpace> h_values(
        "subsetix_csr_field_values_host", value_count);
    for (std::size_t i = 0; i < value_count; ++i) {
      h_values(i) = host.values[i];
    }
    Kokkos::deep_copy(dev.values, h_values);
  }

  return dev;
}

/**
 * @brief Rebuild a host field from a device field.
 */
template <typename T>
inline IntervalField2DHost<T>
build_host_field_from_device(const Field2DDevice<T>& dev) {
  IntervalField2DHost<T> host;

  if (dev.geometry.num_rows == 0) {
    return host;
  }

  auto geom_host = build_host_from_device(dev.geometry);
  host.row_keys = geom_host.row_keys;
  host.row_ptr = geom_host.row_ptr;
  host.intervals.resize(geom_host.intervals.size());

  for (std::size_t i = 0; i < geom_host.intervals.size(); ++i) {
    FieldInterval fi;
    fi.begin = geom_host.intervals[i].begin;
    fi.end = geom_host.intervals[i].end;
    fi.value_offset = geom_host.cell_offsets[i];
    host.intervals[i] = fi;
  }

  const std::size_t value_count = geom_host.total_cells;
  host.values.resize(value_count);
  if (value_count > 0) {
    auto h_values =
        Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, dev.values);
    for (std::size_t i = 0; i < value_count; ++i) {
      host.values[i] = h_values(i);
    }
  }

  return host;
}

/**
 * @brief Build a host field filled with a constant value, using an existing
 *        IntervalSet2DHost geometry as template.
 */
template <typename T>
inline IntervalField2DHost<T>
make_field_like_geometry(const IntervalSet2DHost& geom,
                         const T& init_value) {
  IntervalField2DHost<T> field;

  const std::size_t num_rows = geom.row_keys.size();
  if (num_rows == 0 || geom.row_ptr.size() != num_rows + 1) {
    return field;
  }

  field.row_keys.clear();
  field.row_ptr.clear();
  field.intervals.clear();
  field.values.clear();

  field.row_keys.reserve(num_rows);
  field.row_ptr.reserve(num_rows + 1);

  field.row_ptr.push_back(0);

  for (std::size_t i = 0; i < num_rows; ++i) {
    const Coord y = geom.row_keys[i].y;
    field.row_keys.push_back(RowKey2D{y});

    const std::size_t begin = geom.row_ptr[i];
    const std::size_t end = geom.row_ptr[i + 1];

    for (std::size_t k = begin; k < end; ++k) {
      const Interval& iv = geom.intervals[k];
      const Coord len = static_cast<Coord>(iv.end - iv.begin);
      if (len <= 0) {
        continue;
      }

      FieldInterval fi;
      fi.begin = iv.begin;
      fi.end = iv.end;
      fi.value_offset = field.values.size();

      field.intervals.push_back(fi);
      field.values.insert(field.values.end(),
                          static_cast<std::size_t>(len),
                          init_value);
    }

    field.row_ptr.push_back(field.intervals.size());
  }

  return field;
}

/**
 * @brief Lightweight view representing a field restricted to a sub-geometry.
 *
 * This structure simply bundles a parent Field2D and the IntervalSet that
 * defines the subset of cells we want to operate on. It does not own data and
 * is safe to copy because it only stores Kokkos::View handles.
 */
template <typename T, class MemorySpace = DeviceMemorySpace>
struct Field2DSubView {
  using FieldView = Field2D<T, MemorySpace>;
  using GeometryView = IntervalSet2DView<MemorySpace>;
  using SubSetView = IntervalSubSet2DView<MemorySpace>;

  FieldView parent;
  GeometryView region;
  SubSetView subset;

  KOKKOS_INLINE_FUNCTION
  bool valid() const {
    return parent.geometry.num_rows > 0 && region.num_rows > 0;
  }

  KOKKOS_INLINE_FUNCTION
  std::size_t size() const { return region.total_cells; }

  KOKKOS_INLINE_FUNCTION
  bool has_subset() const {
    return subset.valid();
  }
};

template <typename T>
using Field2DSubViewDevice = Field2DSubView<T, DeviceMemorySpace>;

template <typename T>
using Field2DSubViewHost = Field2DSubView<T, HostMemorySpace>;

/**
 * @brief Build a subview from a parent field and a region geometry.
 *
 * No deep copy is performed. The returned view remains valid as long as the
 * parent field and the region geometry remain valid.
 */
template <typename T, class MemorySpace>
inline Field2DSubView<T, MemorySpace>
make_subview(Field2D<T, MemorySpace>& field,
             const IntervalSet2DView<MemorySpace>& region,
             const std::string& label = {}) {
  (void)label;
  Field2DSubView<T, MemorySpace> sub;
  sub.parent = field;
  sub.region = region;
  return sub;
}

} // namespace csr
} // namespace subsetix
