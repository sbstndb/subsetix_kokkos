#pragma once

#include <cstddef>
#include <fstream>
#include <string>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/field/csr_field.hpp>
#include <subsetix/multilevel/multilevel.hpp>

namespace subsetix {
namespace vtk {

using subsetix::csr::Coord;
using subsetix::csr::IntervalSet2DHost;
using subsetix::csr::FieldInterval;
using subsetix::csr::IntervalField2DHost;

namespace detail {

template <typename T>
inline T byte_swap(T value) {
  static_assert(std::is_trivially_copyable<T>::value,
                "byte_swap requires trivially copyable type");
  std::array<unsigned char, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(T));
  std::reverse(bytes.begin(), bytes.end());
  T out{};
  std::memcpy(&out, bytes.data(), sizeof(T));
  return out;
}

template <typename T>
inline T to_big_endian(T value) {
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
  return value;
#else
  return byte_swap(value);
#endif
}

template <typename T>
inline void write_binary(std::ofstream& ofs, T value) {
  const T be = to_big_endian(value);
  ofs.write(reinterpret_cast<const char*>(&be), sizeof(T));
}

// Generic CSR iteration: calls fn(x, y, interval_index, cell_in_interval)
template <typename RowKeysT, typename RowPtrT, typename IntervalsT, typename Fn>
inline void for_each_cell(const RowKeysT& row_keys, const RowPtrT& row_ptr,
                          const IntervalsT& intervals, std::size_t num_rows, Fn&& fn) {
  for (std::size_t i = 0; i < num_rows; ++i) {
    const Coord y = row_keys[i].y;
    const std::size_t begin = row_ptr[i];
    const std::size_t end = row_ptr[i + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto& iv = intervals[k];
      for (Coord x = iv.begin; x < iv.end; ++x) {
        fn(x, y, k, static_cast<std::size_t>(x - iv.begin));
      }
    }
  }
}

// Overload for IntervalSet2DView (uses operator() for access)
template <typename MemSpace, typename Fn>
inline void for_each_cell(const csr::IntervalSet2DView<MemSpace>& view, Fn&& fn) {
  for (std::size_t i = 0; i < view.num_rows; ++i) {
    const Coord y = view.row_keys(i).y;
    const std::size_t begin = view.row_ptr(i);
    const std::size_t end = view.row_ptr(i + 1);
    for (std::size_t k = begin; k < end; ++k) {
      const auto& iv = view.intervals(k);
      for (Coord x = iv.begin; x < iv.end; ++x) {
        fn(x, y, k, static_cast<std::size_t>(x - iv.begin));
      }
    }
  }
}

// Overload for Field2D (accesses geometry member)
template <typename T, typename MemSpace, typename Fn>
inline void for_each_cell(const csr::Field2D<T, MemSpace>& field, Fn&& fn) {
  for (std::size_t i = 0; i < field.geometry.num_rows; ++i) {
    const Coord y = field.geometry.row_keys(i).y;
    const std::size_t begin = field.geometry.row_ptr(i);
    const std::size_t end = field.geometry.row_ptr(i + 1);
    for (std::size_t k = begin; k < end; ++k) {
      const auto& iv = field.geometry.intervals(k);
      for (Coord x = iv.begin; x < iv.end; ++x) {
        fn(x, y, k, static_cast<std::size_t>(x - iv.begin));
      }
    }
  }
}

template <typename RowPtrT, typename IntervalsT>
inline std::size_t count_cells(const RowPtrT& row_ptr, const IntervalsT& intervals,
                               std::size_t num_rows) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < num_rows; ++i) {
    const std::size_t begin = row_ptr[i];
    const std::size_t end = row_ptr[i + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const Coord len = intervals[k].end - intervals[k].begin;
      if (len > 0) count += static_cast<std::size_t>(len);
    }
  }
  return count;
}

template <typename MemSpace>
inline std::size_t count_cells(const csr::IntervalSet2DView<MemSpace>& view) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < view.num_rows; ++i) {
    const std::size_t begin = view.row_ptr(i);
    const std::size_t end = view.row_ptr(i + 1);
    for (std::size_t k = begin; k < end; ++k) {
      const Coord len = view.intervals(k).end - view.intervals(k).begin;
      if (len > 0) count += static_cast<std::size_t>(len);
    }
  }
  return count;
}

// VTK quad writer - handles all VTK structure writing
class VtkQuadWriter {
public:
  VtkQuadWriter(const std::string& filename, bool binary = false)
      : ofs_(filename, binary ? std::ios::binary : std::ios::out),
        binary_(binary), num_cells_(0), cell_idx_(0) {}

  void write_header(const char* title, std::size_t num_cells) {
    num_cells_ = num_cells;
    ofs_ << "# vtk DataFile Version 3.0\n";
    ofs_ << title << "\n";
    ofs_ << (binary_ ? "BINARY\n" : "ASCII\n");
    ofs_ << "DATASET UNSTRUCTURED_GRID\n";
  }

  void write_empty(const char* title, const std::string& scalar_name = "") {
    write_header(title, 0);
    ofs_ << "POINTS 0 float\n";
    ofs_ << "CELLS 0 0\n";
    ofs_ << "CELL_TYPES 0\n";
    if (!scalar_name.empty()) {
      ofs_ << "CELL_DATA 0\n";
      ofs_ << "SCALARS " << scalar_name << " float 1\n";
      ofs_ << "LOOKUP_TABLE default\n";
    }
  }

  void begin_points() {
    ofs_ << "POINTS " << num_cells_ * 4 << " float\n";
  }

  void write_quad(double x0, double y0, double x1, double y1) {
    if (binary_) {
      write_binary(ofs_, static_cast<float>(x0)); write_binary(ofs_, static_cast<float>(y0)); write_binary(ofs_, 0.0f);
      write_binary(ofs_, static_cast<float>(x1)); write_binary(ofs_, static_cast<float>(y0)); write_binary(ofs_, 0.0f);
      write_binary(ofs_, static_cast<float>(x1)); write_binary(ofs_, static_cast<float>(y1)); write_binary(ofs_, 0.0f);
      write_binary(ofs_, static_cast<float>(x0)); write_binary(ofs_, static_cast<float>(y1)); write_binary(ofs_, 0.0f);
    } else {
      ofs_ << x0 << " " << y0 << " 0\n" << x1 << " " << y0 << " 0\n"
           << x1 << " " << y1 << " 0\n" << x0 << " " << y1 << " 0\n";
    }
  }

  void end_points() { if (binary_) ofs_ << "\n"; }

  void write_cells_and_types() {
    ofs_ << "CELLS " << num_cells_ << " " << num_cells_ * 5 << "\n";
    std::size_t pt = 0;
    if (binary_) {
      for (std::size_t c = 0; c < num_cells_; ++c, pt += 4) {
        write_binary(ofs_, std::uint32_t(4));
        write_binary(ofs_, std::uint32_t(pt)); write_binary(ofs_, std::uint32_t(pt + 1));
        write_binary(ofs_, std::uint32_t(pt + 2)); write_binary(ofs_, std::uint32_t(pt + 3));
      }
      ofs_ << "\n";
    } else {
      for (std::size_t c = 0; c < num_cells_; ++c, pt += 4) {
        ofs_ << "4 " << pt << " " << pt + 1 << " " << pt + 2 << " " << pt + 3 << "\n";
      }
    }
    ofs_ << "CELL_TYPES " << num_cells_ << "\n";
    if (binary_) {
      for (std::size_t c = 0; c < num_cells_; ++c) write_binary(ofs_, std::uint32_t(9));
      ofs_ << "\n";
    } else {
      for (std::size_t c = 0; c < num_cells_; ++c) ofs_ << "9\n";
    }
  }

  void begin_cell_data() { ofs_ << "CELL_DATA " << num_cells_ << "\n"; }

  void begin_scalar(const std::string& name, const char* type = "float") {
    ofs_ << "SCALARS " << name << " " << type << " 1\n";
    ofs_ << "LOOKUP_TABLE default\n";
  }

  template <typename T>
  void write_value(T v) {
    if (binary_) {
      write_binary(ofs_, v);
    } else {
      ofs_ << v << "\n";
    }
  }

  void end_scalar() { if (binary_) ofs_ << "\n"; }

private:
  std::ofstream ofs_;
  bool binary_;
  std::size_t num_cells_;
  std::size_t cell_idx_;
};

} // namespace detail

/**
 * @brief Export a CSR 2D interval set to a legacy VTK unstructured grid (.vtk).
 */
inline void write_legacy_quads(const IntervalSet2DHost& host,
                               const std::string& filename) {
  const std::size_t num_rows = host.row_keys.size();
  if (num_rows == 0 || host.row_ptr.size() != num_rows + 1) {
    detail::VtkQuadWriter w(filename);
    w.write_empty("Empty subsetix mesh");
    return;
  }

  const std::size_t num_cells = detail::count_cells(host.row_ptr, host.intervals, num_rows);
  detail::VtkQuadWriter w(filename);
  w.write_header("Subsetix CSR mesh", num_cells);
  w.begin_points();
  detail::for_each_cell(host.row_keys, host.row_ptr, host.intervals, num_rows,
    [&](Coord x, Coord y, std::size_t, std::size_t) {
      w.write_quad(x, y, x + 1, y + 1);
    });
  w.end_points();
  w.write_cells_and_types();
}

/**
 * @brief Export a CSR 2D field to a legacy VTK unstructured grid (.vtk).
 */
template <typename T>
inline void write_legacy_quads(const IntervalField2DHost<T>& field,
                               const std::string& filename,
                               const std::string& scalar_name = "field") {
  const std::size_t num_rows = field.row_keys.size();
  if (num_rows == 0 || field.row_ptr.size() != num_rows + 1) {
    detail::VtkQuadWriter w(filename);
    w.write_empty("Empty subsetix field", scalar_name);
    return;
  }

  const std::size_t num_cells = detail::count_cells(field.row_ptr, field.intervals, num_rows);
  detail::VtkQuadWriter w(filename);
  w.write_header("Subsetix CSR field", num_cells);
  w.begin_points();
  detail::for_each_cell(field.row_keys, field.row_ptr, field.intervals, num_rows,
    [&](Coord x, Coord y, std::size_t, std::size_t) {
      w.write_quad(x, y, x + 1, y + 1);
    });
  w.end_points();
  w.write_cells_and_types();

  w.begin_cell_data();
  w.begin_scalar(scalar_name);
  detail::for_each_cell(field.row_keys, field.row_ptr, field.intervals, num_rows,
    [&](Coord x, Coord, std::size_t k, std::size_t local_idx) {
      const std::size_t offset = field.intervals[k].value_offset;
      const std::size_t idx = offset + local_idx;
      float v = (idx < field.values.size()) ? static_cast<float>(field.values[idx]) : 0.0f;
      w.write_value(v);
    });
  w.end_scalar();
}

/**
 * @brief Export a MultilevelGeoHost to VTK, respecting physical coordinates.
 */
inline void write_multilevel_vtk(const MultilevelGeoHost& geo,
                                 const std::string& filename) {
  std::size_t num_cells = 0;
  for (int l = 0; l < geo.num_active_levels; ++l) {
    num_cells += detail::count_cells(geo.levels[l]);
  }

  detail::VtkQuadWriter w(filename);
  w.write_header("Subsetix Multilevel Mesh", num_cells);
  if (num_cells == 0) {
    w.write_empty("Subsetix Multilevel Mesh");
    return;
  }

  w.begin_points();
  for (int l = 0; l < geo.num_active_levels; ++l) {
    const auto& view = geo.levels[l];
    if (view.num_rows == 0) continue;
    const double dx = geo.dx_at(l), dy = geo.dy_at(l);
    detail::for_each_cell(view, [&](Coord x, Coord y, std::size_t, std::size_t) {
      w.write_quad(geo.origin_x + x * dx, geo.origin_y + y * dy,
                   geo.origin_x + (x + 1) * dx, geo.origin_y + (y + 1) * dy);
    });
  }
  w.end_points();
  w.write_cells_and_types();

  w.begin_cell_data();
  w.begin_scalar("Level", "int");
  for (int l = 0; l < geo.num_active_levels; ++l) {
    detail::for_each_cell(geo.levels[l], [&](Coord, Coord, std::size_t, std::size_t) {
      w.write_value(l);
    });
  }
  w.end_scalar();
}

/**
 * @brief Export a MultilevelFieldHost to VTK with physical coordinates.
 */
template <typename T>
inline void write_multilevel_field_vtk(const MultilevelFieldHost<T>& field,
                                       const MultilevelGeoHost& geo,
                                       const std::string& filename,
                                       const std::string& scalar_name = "field",
                                       bool binary = false) {
  std::size_t num_cells = 0;
  for (int l = 0; l < field.num_active_levels; ++l) {
    const auto& view = field.levels[l];
    for (std::size_t i = 0; i < view.geometry.num_rows; ++i) {
      const std::size_t begin = view.geometry.row_ptr(i);
      const std::size_t end = view.geometry.row_ptr(i + 1);
      for (std::size_t k = begin; k < end; ++k) {
        const Coord len = view.geometry.intervals(k).end - view.geometry.intervals(k).begin;
        if (len > 0) num_cells += static_cast<std::size_t>(len);
      }
    }
  }

  detail::VtkQuadWriter w(filename, binary);
  w.write_header("Subsetix Multilevel Field", num_cells);
  if (num_cells == 0) {
    w.write_empty("Subsetix Multilevel Field", scalar_name);
    return;
  }

  w.begin_points();
  for (int l = 0; l < field.num_active_levels; ++l) {
    const auto& view = field.levels[l];
    if (view.geometry.num_rows == 0) continue;
    const double dx = geo.dx_at(l), dy = geo.dy_at(l);
    detail::for_each_cell(view, [&](Coord x, Coord y, std::size_t, std::size_t) {
      w.write_quad(geo.origin_x + x * dx, geo.origin_y + y * dy,
                   geo.origin_x + (x + 1) * dx, geo.origin_y + (y + 1) * dy);
    });
  }
  w.end_points();
  w.write_cells_and_types();

  w.begin_cell_data();
  w.begin_scalar(scalar_name);
  for (int l = 0; l < field.num_active_levels; ++l) {
    const auto& view = field.levels[l];
    detail::for_each_cell(view, [&](Coord x, Coord, std::size_t k, std::size_t local_idx) {
      const std::size_t offset = view.geometry.cell_offsets(k);
      const std::size_t idx = offset + local_idx;
      float v = (idx < view.values.extent(0)) ? static_cast<float>(view.values(idx)) : 0.0f;
      w.write_value(v);
    });
  }
  w.end_scalar();

  w.begin_scalar("Level", "int");
  for (int l = 0; l < field.num_active_levels; ++l) {
    detail::for_each_cell(field.levels[l], [&](Coord, Coord, std::size_t, std::size_t) {
      w.write_value(l);
    });
  }
  w.end_scalar();
}

} // namespace vtk
} // namespace subsetix
