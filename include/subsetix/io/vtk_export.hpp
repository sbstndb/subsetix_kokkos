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

// Type traits for CSR data access
template <typename T> struct CsrTraits;

template <> struct CsrTraits<IntervalSet2DHost> {
  static std::size_t num_rows(const IntervalSet2DHost& h) { return h.row_keys.size(); }
  static Coord row_y(const IntervalSet2DHost& h, std::size_t i) { return h.row_keys[i].y; }
  static std::size_t row_begin(const IntervalSet2DHost& h, std::size_t i) { return h.row_ptr[i]; }
  static std::size_t row_end(const IntervalSet2DHost& h, std::size_t i) { return h.row_ptr[i + 1]; }
  static Coord iv_begin(const IntervalSet2DHost& h, std::size_t k) { return h.intervals[k].begin; }
  static Coord iv_end(const IntervalSet2DHost& h, std::size_t k) { return h.intervals[k].end; }
  static bool valid(const IntervalSet2DHost& h) { return h.row_keys.size() > 0 && h.row_ptr.size() == h.row_keys.size() + 1; }
};

template <typename T> struct CsrTraits<IntervalField2DHost<T>> {
  static std::size_t num_rows(const IntervalField2DHost<T>& f) { return f.row_keys.size(); }
  static Coord row_y(const IntervalField2DHost<T>& f, std::size_t i) { return f.row_keys[i].y; }
  static std::size_t row_begin(const IntervalField2DHost<T>& f, std::size_t i) { return f.row_ptr[i]; }
  static std::size_t row_end(const IntervalField2DHost<T>& f, std::size_t i) { return f.row_ptr[i + 1]; }
  static Coord iv_begin(const IntervalField2DHost<T>& f, std::size_t k) { return f.intervals[k].begin; }
  static Coord iv_end(const IntervalField2DHost<T>& f, std::size_t k) { return f.intervals[k].end; }
  static std::size_t value_offset(const IntervalField2DHost<T>& f, std::size_t k) { return f.intervals[k].value_offset; }
  static float get_value(const IntervalField2DHost<T>& f, std::size_t idx) {
    return idx < f.values.size() ? static_cast<float>(f.values[idx]) : 0.0f;
  }
  static bool valid(const IntervalField2DHost<T>& f) { return f.row_keys.size() > 0 && f.row_ptr.size() == f.row_keys.size() + 1; }
};

template <typename MemSpace> struct CsrTraits<csr::IntervalSet2DView<MemSpace>> {
  using View = csr::IntervalSet2DView<MemSpace>;
  static std::size_t num_rows(const View& v) { return v.num_rows; }
  static Coord row_y(const View& v, std::size_t i) { return v.row_keys(i).y; }
  static std::size_t row_begin(const View& v, std::size_t i) { return v.row_ptr(i); }
  static std::size_t row_end(const View& v, std::size_t i) { return v.row_ptr(i + 1); }
  static Coord iv_begin(const View& v, std::size_t k) { return v.intervals(k).begin; }
  static Coord iv_end(const View& v, std::size_t k) { return v.intervals(k).end; }
  static bool valid(const View& v) { return v.num_rows > 0; }
};

template <typename T, typename MemSpace> struct CsrTraits<csr::Field2D<T, MemSpace>> {
  using Field = csr::Field2D<T, MemSpace>;
  static std::size_t num_rows(const Field& f) { return f.geometry.num_rows; }
  static Coord row_y(const Field& f, std::size_t i) { return f.geometry.row_keys(i).y; }
  static std::size_t row_begin(const Field& f, std::size_t i) { return f.geometry.row_ptr(i); }
  static std::size_t row_end(const Field& f, std::size_t i) { return f.geometry.row_ptr(i + 1); }
  static Coord iv_begin(const Field& f, std::size_t k) { return f.geometry.intervals(k).begin; }
  static Coord iv_end(const Field& f, std::size_t k) { return f.geometry.intervals(k).end; }
  static std::size_t value_offset(const Field& f, std::size_t k) { return f.geometry.cell_offsets(k); }
  static float get_value(const Field& f, std::size_t idx) {
    return idx < f.values.extent(0) ? static_cast<float>(f.values(idx)) : 0.0f;
  }
  static bool valid(const Field& f) { return f.geometry.num_rows > 0; }
};

// Unified for_each_cell using traits
template <typename CsrT, typename Fn>
inline void for_each_cell(const CsrT& data, Fn&& fn) {
  using Tr = CsrTraits<CsrT>;
  for (std::size_t i = 0; i < Tr::num_rows(data); ++i) {
    const Coord y = Tr::row_y(data, i);
    for (std::size_t k = Tr::row_begin(data, i); k < Tr::row_end(data, i); ++k) {
      const Coord x0 = Tr::iv_begin(data, k), x1 = Tr::iv_end(data, k);
      for (Coord x = x0; x < x1; ++x)
        fn(x, y, k, static_cast<std::size_t>(x - x0));
    }
  }
}

// Unified count_cells using traits
template <typename CsrT>
inline std::size_t count_cells(const CsrT& data) {
  using Tr = CsrTraits<CsrT>;
  std::size_t count = 0;
  for (std::size_t i = 0; i < Tr::num_rows(data); ++i) {
    for (std::size_t k = Tr::row_begin(data, i); k < Tr::row_end(data, i); ++k) {
      const Coord len = Tr::iv_end(data, k) - Tr::iv_begin(data, k);
      if (len > 0) count += static_cast<std::size_t>(len);
    }
  }
  return count;
}

// Buffered binary VTK writer - minimizes syscalls with 64KB buffer
class VtkQuadWriter {
  static constexpr std::size_t BUF_SIZE = 65536;
  std::ofstream ofs_;
  std::vector<char> buf_;
  std::size_t pos_ = 0;
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

  template<typename T, std::size_t N>
  void write_be_array(const T (&arr)[N]) {
    ensure(sizeof(T) * N);
    for (std::size_t i = 0; i < N; ++i) {
      T be = to_big_endian(arr[i]);
      std::memcpy(buf_.data() + pos_, &be, sizeof(T));
      pos_ += sizeof(T);
    }
  }

public:
  explicit VtkQuadWriter(const std::string& filename)
      : ofs_(filename, std::ios::binary), buf_(BUF_SIZE) {}
  ~VtkQuadWriter() { flush_buf(); }

  void write_header(const char* title, std::size_t num_cells) {
    num_cells_ = num_cells;
    char hdr[256];
    int n = std::snprintf(hdr, sizeof(hdr), "# vtk DataFile Version 3.0\n%s\nBINARY\nDATASET UNSTRUCTURED_GRID\n", title);
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
    const float pts[12] = {float(x0), float(y0), 0.f, float(x1), float(y0), 0.f,
                           float(x1), float(y1), 0.f, float(x0), float(y1), 0.f};
    write_be_array(pts);
  }

  void end_points() { write_raw("\n", 1); }

  void write_cells_and_types() {
    char s[64];
    int n = std::snprintf(s, sizeof(s), "CELLS %zu %zu\n", num_cells_, num_cells_ * 5);
    write_raw(s, n);

    for (std::size_t c = 0, pt = 0; c < num_cells_; ++c, pt += 4) {
      const std::uint32_t cell[5] = {4, std::uint32_t(pt), std::uint32_t(pt+1),
                                     std::uint32_t(pt+2), std::uint32_t(pt+3)};
      write_be_array(cell);
    }
    write_raw("\n", 1);

    n = std::snprintf(s, sizeof(s), "CELL_TYPES %zu\n", num_cells_);
    write_raw(s, n);

    // Write cell types in batches for efficiency
    constexpr std::size_t BATCH = 256;
    std::uint32_t nines[BATCH];
    const std::uint32_t nine_be = to_big_endian(std::uint32_t(9));
    for (std::size_t i = 0; i < BATCH; ++i) nines[i] = nine_be;

    std::size_t remaining = num_cells_;
    while (remaining >= BATCH) {
      ensure(BATCH * sizeof(std::uint32_t));
      std::memcpy(buf_.data() + pos_, nines, BATCH * sizeof(std::uint32_t));
      pos_ += BATCH * sizeof(std::uint32_t);
      remaining -= BATCH;
    }
    if (remaining > 0) {
      ensure(remaining * sizeof(std::uint32_t));
      std::memcpy(buf_.data() + pos_, nines, remaining * sizeof(std::uint32_t));
      pos_ += remaining * sizeof(std::uint32_t);
    }
    write_raw("\n", 1);
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

  template<typename T> void write_value(T v) { write_be(v); }

  void end_scalar() { write_raw("\n", 1); }
};

// Trait to detect if CsrTraits<T> has get_value (i.e., is a field type)
template <typename T, typename = void>
struct has_values : std::false_type {};
template <typename T>
struct has_values<T, std::void_t<decltype(CsrTraits<T>::get_value(std::declval<T>(), 0))>> : std::true_type {};

} // namespace detail

/**
 * @brief Export a CSR 2D data structure to a legacy VTK unstructured grid (.vtk).
 * Works with both IntervalSet2DHost (geometry only) and IntervalField2DHost<T> (with scalar values).
 */
template <typename CsrT>
inline void write_legacy_quads(const CsrT& data, const std::string& filename,
                               const std::string& scalar_name = "field") {
  using Tr = detail::CsrTraits<CsrT>;
  constexpr bool is_field = detail::has_values<CsrT>::value;

  if (!Tr::valid(data)) {
    detail::VtkQuadWriter w(filename);
    if constexpr (is_field) w.write_empty("Empty subsetix field", scalar_name);
    else w.write_empty("Empty subsetix mesh");
    return;
  }

  const std::size_t num_cells = detail::count_cells(data);
  detail::VtkQuadWriter w(filename);
  w.write_header(is_field ? "Subsetix CSR field" : "Subsetix CSR mesh", num_cells);
  w.begin_points();
  detail::for_each_cell(data, [&](Coord x, Coord y, std::size_t, std::size_t) {
    w.write_quad(x, y, x + 1, y + 1);
  });
  w.end_points();
  w.write_cells_and_types();

  if constexpr (is_field) {
    w.begin_cell_data();
    w.begin_scalar(scalar_name);
    detail::for_each_cell(data, [&](Coord, Coord, std::size_t k, std::size_t local_idx) {
      w.write_value(Tr::get_value(data, Tr::value_offset(data, k) + local_idx));
    });
    w.end_scalar();
  }
}

/**
 * @brief Export a MultilevelGeoHost or MultilevelFieldHost to VTK with physical coordinates.
 * Uses compile-time detection to handle geometry-only vs field types.
 */
template <typename MultiT, bool WriteFieldValues>
inline void write_multilevel_vtk_impl(const MultiT& multi, const MultilevelGeoHost& geo,
                                      const std::string& filename, const std::string& scalar_name) {
  std::size_t num_cells = 0;
  for (int l = 0; l < multi.num_active_levels; ++l)
    num_cells += detail::count_cells(multi.levels[l]);

  detail::VtkQuadWriter w(filename);
  if (num_cells == 0) {
    if constexpr (WriteFieldValues) w.write_empty("Subsetix Multilevel Field", scalar_name);
    else w.write_empty("Subsetix Multilevel Mesh");
    return;
  }

  w.write_header(WriteFieldValues ? "Subsetix Multilevel Field" : "Subsetix Multilevel Mesh", num_cells);
  w.begin_points();
  for (int l = 0; l < multi.num_active_levels; ++l) {
    const auto& lv = multi.levels[l];
    using Tr = detail::CsrTraits<std::decay_t<decltype(lv)>>;
    if (Tr::num_rows(lv) == 0) continue;
    const double dx = geo.dx_at(l), dy = geo.dy_at(l);
    detail::for_each_cell(lv, [&](Coord x, Coord y, std::size_t, std::size_t) {
      w.write_quad(geo.origin_x + x * dx, geo.origin_y + y * dy,
                   geo.origin_x + (x + 1) * dx, geo.origin_y + (y + 1) * dy);
    });
  }
  w.end_points();
  w.write_cells_and_types();

  w.begin_cell_data();
  if constexpr (WriteFieldValues) {
    w.begin_scalar(scalar_name);
    for (int l = 0; l < multi.num_active_levels; ++l) {
      const auto& lv = multi.levels[l];
      using Tr = detail::CsrTraits<std::decay_t<decltype(lv)>>;
      detail::for_each_cell(lv, [&](Coord, Coord, std::size_t k, std::size_t local_idx) {
        w.write_value(Tr::get_value(lv, Tr::value_offset(lv, k) + local_idx));
      });
    }
    w.end_scalar();
  }
  w.begin_scalar("Level", "int");
  for (int l = 0; l < multi.num_active_levels; ++l)
    detail::for_each_cell(multi.levels[l], [&](Coord, Coord, std::size_t, std::size_t) { w.write_value(l); });
  w.end_scalar();
}

/// Export MultilevelGeoHost to VTK (geometry + level scalar only)
inline void write_multilevel_vtk(const MultilevelGeoHost& geo, const std::string& filename) {
  write_multilevel_vtk_impl<MultilevelGeoHost, false>(geo, geo, filename, "");
}

/// Export MultilevelFieldHost to VTK (geometry + field values + level scalar)
template <typename T>
inline void write_multilevel_field_vtk(const MultilevelFieldHost<T>& field, const MultilevelGeoHost& geo,
                                       const std::string& filename, const std::string& scalar_name = "field") {
  write_multilevel_vtk_impl<MultilevelFieldHost<T>, true>(field, geo, filename, scalar_name);
}

} // namespace vtk
} // namespace subsetix
