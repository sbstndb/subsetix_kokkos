#pragma once

#include <cstddef>
#include <fstream>
#include <string>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

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
  static_assert(std::is_trivially_copyable<T>::value, "byte_swap requires trivially copyable type");
  std::array<unsigned char, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(T));
  std::reverse(bytes.begin(), bytes.end());
  T out{};
  std::memcpy(&out, bytes.data(), sizeof(T));
  return out;
}

template <typename T>
inline T to_big_endian(T value) {
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
  return value;
#else
  return byte_swap(value);
#endif
}

// Generic CSR iteration: calls fn(x, y, interval_index, cell_in_interval)
template <typename RowKeysT, typename RowPtrT, typename IntervalsT, typename Fn>
inline void for_each_cell(const RowKeysT& row_keys, const RowPtrT& row_ptr,
                          const IntervalsT& intervals, std::size_t num_rows, Fn&& fn) {
  for (std::size_t i = 0; i < num_rows; ++i) {
    const Coord y = row_keys[i].y;
    const std::size_t begin = row_ptr[i], end = row_ptr[i + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto& iv = intervals[k];
      for (Coord x = iv.begin; x < iv.end; ++x)
        fn(x, y, k, static_cast<std::size_t>(x - iv.begin));
    }
  }
}

template <typename MemSpace, typename Fn>
inline void for_each_cell(const csr::IntervalSet2DView<MemSpace>& view, Fn&& fn) {
  for (std::size_t i = 0; i < view.num_rows; ++i) {
    const Coord y = view.row_keys(i).y;
    const std::size_t begin = view.row_ptr(i), end = view.row_ptr(i + 1);
    for (std::size_t k = begin; k < end; ++k) {
      const auto& iv = view.intervals(k);
      for (Coord x = iv.begin; x < iv.end; ++x)
        fn(x, y, k, static_cast<std::size_t>(x - iv.begin));
    }
  }
}

template <typename T, typename MemSpace, typename Fn>
inline void for_each_cell(const csr::Field2D<T, MemSpace>& field, Fn&& fn) {
  for (std::size_t i = 0; i < field.geometry.num_rows; ++i) {
    const Coord y = field.geometry.row_keys(i).y;
    const std::size_t begin = field.geometry.row_ptr(i), end = field.geometry.row_ptr(i + 1);
    for (std::size_t k = begin; k < end; ++k) {
      const auto& iv = field.geometry.intervals(k);
      for (Coord x = iv.begin; x < iv.end; ++x)
        fn(x, y, k, static_cast<std::size_t>(x - iv.begin));
    }
  }
}

template <typename RowPtrT, typename IntervalsT>
inline std::size_t count_cells(const RowPtrT& row_ptr, const IntervalsT& intervals, std::size_t num_rows) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < num_rows; ++i) {
    for (std::size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
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
    for (std::size_t k = view.row_ptr(i); k < view.row_ptr(i + 1); ++k) {
      const Coord len = view.intervals(k).end - view.intervals(k).begin;
      if (len > 0) count += static_cast<std::size_t>(len);
    }
  }
  return count;
}

// Buffered VTK writer - minimizes syscalls with 64KB buffer
class VtkQuadWriter {
  static constexpr std::size_t BUF_SIZE = 65536;
  std::ofstream ofs_;
  std::vector<char> buf_;
  std::size_t pos_ = 0;
  bool binary_;
  std::size_t num_cells_ = 0;

  void flush_buf() { if (pos_) { ofs_.write(buf_.data(), pos_); pos_ = 0; } }
  void ensure(std::size_t n) { if (pos_ + n > BUF_SIZE) flush_buf(); }

  void write_raw(const char* data, std::size_t n) {
    if (n >= BUF_SIZE) { flush_buf(); ofs_.write(data, n); return; }
    ensure(n);
    std::memcpy(buf_.data() + pos_, data, n);
    pos_ += n;
  }

  template<typename T> void write_be(T v) {
    T be = to_big_endian(v);
    ensure(sizeof(T));
    std::memcpy(buf_.data() + pos_, &be, sizeof(T));
    pos_ += sizeof(T);
  }

public:
  VtkQuadWriter(const std::string& filename, bool binary = false)
      : ofs_(filename, binary ? std::ios::binary : std::ios::out), buf_(BUF_SIZE), binary_(binary) {}
  ~VtkQuadWriter() { flush_buf(); }

  void write_header(const char* title, std::size_t num_cells) {
    num_cells_ = num_cells;
    char hdr[256];
    int n = std::snprintf(hdr, sizeof(hdr), "# vtk DataFile Version 3.0\n%s\n%s\nDATASET UNSTRUCTURED_GRID\n",
                          title, binary_ ? "BINARY" : "ASCII");
    write_raw(hdr, n);
  }

  void write_empty(const char* title, const std::string& scalar_name = "") {
    write_header(title, 0);
    const char* empty = "POINTS 0 float\nCELLS 0 0\nCELL_TYPES 0\n";
    write_raw(empty, std::strlen(empty));
    if (!scalar_name.empty()) {
      char s[128];
      int n = std::snprintf(s, sizeof(s), "CELL_DATA 0\nSCALARS %s float 1\nLOOKUP_TABLE default\n", scalar_name.c_str());
      write_raw(s, n);
    }
  }

  void begin_points() {
    char s[64];
    int n = std::snprintf(s, sizeof(s), "POINTS %zu float\n", num_cells_ * 4);
    write_raw(s, n);
  }

  void write_quad(double x0, double y0, double x1, double y1) {
    if (binary_) {
      float pts[12] = {float(x0), float(y0), 0.f, float(x1), float(y0), 0.f,
                       float(x1), float(y1), 0.f, float(x0), float(y1), 0.f};
      for (int i = 0; i < 12; ++i) write_be(pts[i]);
    } else {
      char s[256];
      int n = std::snprintf(s, sizeof(s), "%g %g 0\n%g %g 0\n%g %g 0\n%g %g 0\n", x0, y0, x1, y0, x1, y1, x0, y1);
      write_raw(s, n);
    }
  }

  void end_points() { if (binary_) write_raw("\n", 1); }

  void write_cells_and_types() {
    char s[64];
    int n = std::snprintf(s, sizeof(s), "CELLS %zu %zu\n", num_cells_, num_cells_ * 5);
    write_raw(s, n);

    if (binary_) {
      for (std::size_t c = 0, pt = 0; c < num_cells_; ++c, pt += 4) {
        write_be(std::uint32_t(4));
        write_be(std::uint32_t(pt)); write_be(std::uint32_t(pt+1));
        write_be(std::uint32_t(pt+2)); write_be(std::uint32_t(pt+3));
      }
      write_raw("\n", 1);
    } else {
      for (std::size_t c = 0, pt = 0; c < num_cells_; ++c, pt += 4) {
        char line[64];
        int len = std::snprintf(line, sizeof(line), "4 %zu %zu %zu %zu\n", pt, pt+1, pt+2, pt+3);
        write_raw(line, len);
      }
    }

    n = std::snprintf(s, sizeof(s), "CELL_TYPES %zu\n", num_cells_);
    write_raw(s, n);

    if (binary_) {
      const std::uint32_t nine = to_big_endian(std::uint32_t(9));
      ensure(sizeof(nine));
      for (std::size_t c = 0; c < num_cells_; ++c) {
        if (pos_ + sizeof(nine) > BUF_SIZE) flush_buf();
        std::memcpy(buf_.data() + pos_, &nine, sizeof(nine));
        pos_ += sizeof(nine);
      }
      write_raw("\n", 1);
    } else {
      // Batch write "9\n" - build block and repeat
      constexpr std::size_t BATCH = 512;
      char block[BATCH * 2];
      for (std::size_t i = 0; i < BATCH; ++i) { block[i*2] = '9'; block[i*2+1] = '\n'; }
      std::size_t full = num_cells_ / BATCH, rem = num_cells_ % BATCH;
      for (std::size_t i = 0; i < full; ++i) write_raw(block, BATCH * 2);
      if (rem) write_raw(block, rem * 2);
    }
  }

  void begin_cell_data() {
    char s[32];
    int n = std::snprintf(s, sizeof(s), "CELL_DATA %zu\n", num_cells_);
    write_raw(s, n);
  }

  void begin_scalar(const std::string& name, const char* type = "float") {
    char s[128];
    int n = std::snprintf(s, sizeof(s), "SCALARS %s %s 1\nLOOKUP_TABLE default\n", name.c_str(), type);
    write_raw(s, n);
  }

  template<typename T> void write_value(T v) {
    if (binary_) {
      write_be(v);
    } else {
      char s[32];
      int n;
      if constexpr (std::is_floating_point<T>::value) n = std::snprintf(s, sizeof(s), "%g\n", double(v));
      else n = std::snprintf(s, sizeof(s), "%d\n", int(v));
      write_raw(s, n);
    }
  }

  void end_scalar() { if (binary_) write_raw("\n", 1); }
};

} // namespace detail

/**
 * @brief Export a CSR 2D interval set to a legacy VTK unstructured grid (.vtk).
 */
inline void write_legacy_quads(const IntervalSet2DHost& host, const std::string& filename, bool binary = false) {
  const std::size_t num_rows = host.row_keys.size();
  if (num_rows == 0 || host.row_ptr.size() != num_rows + 1) {
    detail::VtkQuadWriter w(filename, binary);
    w.write_empty("Empty subsetix mesh");
    return;
  }
  const std::size_t num_cells = detail::count_cells(host.row_ptr, host.intervals, num_rows);
  detail::VtkQuadWriter w(filename, binary);
  w.write_header("Subsetix CSR mesh", num_cells);
  w.begin_points();
  detail::for_each_cell(host.row_keys, host.row_ptr, host.intervals, num_rows,
    [&](Coord x, Coord y, std::size_t, std::size_t) { w.write_quad(x, y, x + 1, y + 1); });
  w.end_points();
  w.write_cells_and_types();
}

/**
 * @brief Export a CSR 2D field to a legacy VTK unstructured grid (.vtk).
 */
template <typename T>
inline void write_legacy_quads(const IntervalField2DHost<T>& field, const std::string& filename,
                               const std::string& scalar_name = "field", bool binary = false) {
  const std::size_t num_rows = field.row_keys.size();
  if (num_rows == 0 || field.row_ptr.size() != num_rows + 1) {
    detail::VtkQuadWriter w(filename, binary);
    w.write_empty("Empty subsetix field", scalar_name);
    return;
  }
  const std::size_t num_cells = detail::count_cells(field.row_ptr, field.intervals, num_rows);
  detail::VtkQuadWriter w(filename, binary);
  w.write_header("Subsetix CSR field", num_cells);
  w.begin_points();
  detail::for_each_cell(field.row_keys, field.row_ptr, field.intervals, num_rows,
    [&](Coord x, Coord y, std::size_t, std::size_t) { w.write_quad(x, y, x + 1, y + 1); });
  w.end_points();
  w.write_cells_and_types();
  w.begin_cell_data();
  w.begin_scalar(scalar_name);
  detail::for_each_cell(field.row_keys, field.row_ptr, field.intervals, num_rows,
    [&](Coord, Coord, std::size_t k, std::size_t local_idx) {
      const std::size_t idx = field.intervals[k].value_offset + local_idx;
      w.write_value((idx < field.values.size()) ? static_cast<float>(field.values[idx]) : 0.0f);
    });
  w.end_scalar();
}

/**
 * @brief Export a MultilevelGeoHost to VTK, respecting physical coordinates.
 */
inline void write_multilevel_vtk(const MultilevelGeoHost& geo, const std::string& filename, bool binary = false) {
  std::size_t num_cells = 0;
  for (int l = 0; l < geo.num_active_levels; ++l) num_cells += detail::count_cells(geo.levels[l]);

  detail::VtkQuadWriter w(filename, binary);
  if (num_cells == 0) { w.write_empty("Subsetix Multilevel Mesh"); return; }
  w.write_header("Subsetix Multilevel Mesh", num_cells);
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
  for (int l = 0; l < geo.num_active_levels; ++l)
    detail::for_each_cell(geo.levels[l], [&](Coord, Coord, std::size_t, std::size_t) { w.write_value(l); });
  w.end_scalar();
}

/**
 * @brief Export a MultilevelFieldHost to VTK with physical coordinates.
 */
template <typename T>
inline void write_multilevel_field_vtk(const MultilevelFieldHost<T>& field, const MultilevelGeoHost& geo,
                                       const std::string& filename, const std::string& scalar_name = "field",
                                       bool binary = false) {
  std::size_t num_cells = 0;
  for (int l = 0; l < field.num_active_levels; ++l) {
    const auto& view = field.levels[l];
    for (std::size_t i = 0; i < view.geometry.num_rows; ++i)
      for (std::size_t k = view.geometry.row_ptr(i); k < view.geometry.row_ptr(i + 1); ++k) {
        const Coord len = view.geometry.intervals(k).end - view.geometry.intervals(k).begin;
        if (len > 0) num_cells += static_cast<std::size_t>(len);
      }
  }

  detail::VtkQuadWriter w(filename, binary);
  if (num_cells == 0) { w.write_empty("Subsetix Multilevel Field", scalar_name); return; }
  w.write_header("Subsetix Multilevel Field", num_cells);
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
    detail::for_each_cell(view, [&](Coord, Coord, std::size_t k, std::size_t local_idx) {
      const std::size_t idx = view.geometry.cell_offsets(k) + local_idx;
      w.write_value((idx < view.values.extent(0)) ? static_cast<float>(view.values(idx)) : 0.0f);
    });
  }
  w.end_scalar();
  w.begin_scalar("Level", "int");
  for (int l = 0; l < field.num_active_levels; ++l)
    detail::for_each_cell(field.levels[l], [&](Coord, Coord, std::size_t, std::size_t) { w.write_value(l); });
  w.end_scalar();
}

} // namespace vtk
} // namespace subsetix
