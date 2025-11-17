#pragma once

#include <cstddef>
#include <fstream>
#include <string>

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_field.hpp>

namespace subsetix {
namespace vtk {

using subsetix::csr::Coord;
using subsetix::csr::IntervalSet2DHost;
using subsetix::csr::FieldInterval;
using subsetix::csr::IntervalField2DHost;

/**
 * @brief Export a CSR 2D interval set to a legacy VTK unstructured grid (.vtk).
 *
 * Each integer cell [x, x+1) × [y, y+1) is exported as a VTK_QUAD lying in
 * the Z=0 plane. This is intended for debugging / visualisation of meshes.
 */
inline void write_legacy_quads(const IntervalSet2DHost& host,
                               const std::string& filename) {
  std::size_t num_cells = 0;

  const std::size_t num_rows = host.row_keys.size();
  if (num_rows == 0 || host.row_ptr.size() != num_rows + 1) {
    std::ofstream ofs(filename);
    ofs << "# vtk DataFile Version 3.0\n";
    ofs << "Empty subsetix mesh\n";
    ofs << "ASCII\n";
    ofs << "DATASET UNSTRUCTURED_GRID\n";
    ofs << "POINTS 0 float\n";
    ofs << "CELLS 0 0\n";
    ofs << "CELL_TYPES 0\n";
    return;
  }

  for (std::size_t i = 0; i < num_rows; ++i) {
    const Coord y = host.row_keys[i].y;
    (void)y;
    const std::size_t begin = host.row_ptr[i];
    const std::size_t end = host.row_ptr[i + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto& iv = host.intervals[k];
      const Coord x0 = iv.begin;
      const Coord x1 = iv.end;
      if (x1 > x0) {
        num_cells += static_cast<std::size_t>(x1 - x0);
      }
    }
  }

  const std::size_t num_points = num_cells * 4;

  std::ofstream ofs(filename);
  ofs << "# vtk DataFile Version 3.0\n";
  ofs << "Subsetix CSR mesh\n";
  ofs << "ASCII\n";
  ofs << "DATASET UNSTRUCTURED_GRID\n";

  ofs << "POINTS " << num_points << " float\n";
  // Emit 4 points per cell: (x,y,0), (x+1,y,0), (x+1,y+1,0), (x,y+1,0).
  for (std::size_t i = 0; i < num_rows; ++i) {
    const Coord y = host.row_keys[i].y;
    const float y0 = static_cast<float>(y);
    const float y1 = static_cast<float>(y + 1);

    const std::size_t begin = host.row_ptr[i];
    const std::size_t end = host.row_ptr[i + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto& iv = host.intervals[k];
      for (Coord x = iv.begin; x < iv.end; ++x) {
        const float x0 = static_cast<float>(x);
        const float x1 = static_cast<float>(x + 1);
        ofs << x0 << " " << y0 << " 0\n";
        ofs << x1 << " " << y0 << " 0\n";
        ofs << x1 << " " << y1 << " 0\n";
        ofs << x0 << " " << y1 << " 0\n";
      }
    }
  }

  ofs << "CELLS " << num_cells << " " << num_cells * 5 << "\n";
  std::size_t cell_idx = 0;
  for (std::size_t i = 0; i < num_rows; ++i) {
    const std::size_t begin = host.row_ptr[i];
    const std::size_t end = host.row_ptr[i + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto& iv = host.intervals[k];
      for (Coord x = iv.begin; x < iv.end; ++x) {
        (void)x;
        const std::size_t p0 = cell_idx * 4;
        const std::size_t p1 = p0 + 1;
        const std::size_t p2 = p0 + 2;
        const std::size_t p3 = p0 + 3;
        ofs << "4 " << p0 << " " << p1 << " "
            << p2 << " " << p3 << "\n";
        ++cell_idx;
      }
    }
  }

  ofs << "CELL_TYPES " << num_cells << "\n";
  for (std::size_t c = 0; c < num_cells; ++c) {
    ofs << "9\n"; // VTK_QUAD
  }
}

/**
 * @brief Export a CSR 2D field to a legacy VTK unstructured grid (.vtk).
 *
 * Geometry is taken from the field's intervals. Each cell carries a scalar
 * value written as CELL_DATA so that it can be visualised in ParaView.
 */
template <typename T>
inline void write_legacy_quads(const IntervalField2DHost<T>& field,
                               const std::string& filename,
                               const std::string& scalar_name = "field") {
  const std::size_t num_rows = field.row_keys.size();
  const std::size_t row_ptr_size = field.row_ptr.size();

  if (num_rows == 0 || row_ptr_size != num_rows + 1) {
    std::ofstream ofs(filename);
    ofs << "# vtk DataFile Version 3.0\n";
    ofs << "Empty subsetix field\n";
    ofs << "ASCII\n";
    ofs << "DATASET UNSTRUCTURED_GRID\n";
    ofs << "POINTS 0 float\n";
    ofs << "CELLS 0 0\n";
    ofs << "CELL_TYPES 0\n";
    ofs << "CELL_DATA 0\n";
    ofs << "SCALARS " << scalar_name << " float 1\n";
    ofs << "LOOKUP_TABLE default\n";
    return;
  }

  std::size_t num_cells = 0;
  for (std::size_t i = 0; i < num_rows; ++i) {
    const std::size_t begin = field.row_ptr[i];
    const std::size_t end = field.row_ptr[i + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const FieldInterval& iv = field.intervals[k];
      const Coord x0 = iv.begin;
      const Coord x1 = iv.end;
      if (x1 > x0) {
        num_cells += static_cast<std::size_t>(x1 - x0);
      }
    }
  }

  const std::size_t num_points = num_cells * 4;

  std::ofstream ofs(filename);
  ofs << "# vtk DataFile Version 3.0\n";
  ofs << "Subsetix CSR field\n";
  ofs << "ASCII\n";
  ofs << "DATASET UNSTRUCTURED_GRID\n";

  ofs << "POINTS " << num_points << " float\n";
  for (std::size_t i = 0; i < num_rows; ++i) {
    const Coord y = field.row_keys[i].y;
    const float y0 = static_cast<float>(y);
    const float y1 = static_cast<float>(y + 1);

    const std::size_t begin = field.row_ptr[i];
    const std::size_t end = field.row_ptr[i + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const FieldInterval& iv = field.intervals[k];
      for (Coord x = iv.begin; x < iv.end; ++x) {
        const float x0 = static_cast<float>(x);
        const float x1 = static_cast<float>(x + 1);
        ofs << x0 << " " << y0 << " 0\n";
        ofs << x1 << " " << y0 << " 0\n";
        ofs << x1 << " " << y1 << " 0\n";
        ofs << x0 << " " << y1 << " 0\n";
      }
    }
  }

  ofs << "CELLS " << num_cells << " " << num_cells * 5 << "\n";
  std::size_t cell_idx = 0;
  for (std::size_t i = 0; i < num_rows; ++i) {
    const std::size_t begin = field.row_ptr[i];
    const std::size_t end = field.row_ptr[i + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const FieldInterval& iv = field.intervals[k];
      for (Coord x = iv.begin; x < iv.end; ++x) {
        (void)x;
        const std::size_t p0 = cell_idx * 4;
        const std::size_t p1 = p0 + 1;
        const std::size_t p2 = p0 + 2;
        const std::size_t p3 = p0 + 3;
        ofs << "4 " << p0 << " " << p1 << " "
            << p2 << " " << p3 << "\n";
        ++cell_idx;
      }
    }
  }

  ofs << "CELL_TYPES " << num_cells << "\n";
  for (std::size_t c = 0; c < num_cells; ++c) {
    ofs << "9\n"; // VTK_QUAD
  }

  ofs << "CELL_DATA " << num_cells << "\n";
  ofs << "SCALARS " << scalar_name << " float 1\n";
  ofs << "LOOKUP_TABLE default\n";

  // Emit one scalar per cell, in the same order as cells were built.
  for (std::size_t i = 0; i < num_rows; ++i) {
    const std::size_t begin = field.row_ptr[i];
    const std::size_t end = field.row_ptr[i + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const FieldInterval& iv = field.intervals[k];
      const std::size_t offset = iv.value_offset;
      for (Coord x = iv.begin; x < iv.end; ++x) {
        const std::size_t idx =
            offset + static_cast<std::size_t>(x - iv.begin);
        float v = 0.0f;
        if (idx < field.values.size()) {
          v = static_cast<float>(field.values[idx]);
        }
        ofs << v << "\n";
      }
    }
  }
}

} // namespace vtk
} // namespace subsetix
