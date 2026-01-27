// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#include <cstddef>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include <playground/subsetix/field/csr_field.hpp>

namespace playground::subsetix::csr {

struct FieldMaskMapping {
  Kokkos::View<int*, DeviceMemorySpace> row_map;
  Kokkos::View<int*, DeviceMemorySpace> interval_to_row;
  Kokkos::View<int*, DeviceMemorySpace> interval_to_field_interval;
};

namespace detail {

template <int DIM, class MaskRowKeyView, class FieldRowKeyView>
inline Kokkos::View<int*, DeviceMemorySpace>
build_row_map(const MaskRowKeyView& mask_rows,
              const FieldRowKeyView& field_rows,
              std::size_t num_field_rows) {
  Kokkos::View<int*, DeviceMemorySpace> mapping(
      "playground_fields_row_map", mask_rows.extent(0));

  if (mask_rows.extent(0) == 0) {
    return mapping;
  }

  if constexpr (DIM == 2) {
    Kokkos::parallel_for(
        "playground_fields_row_map_kernel_2d",
        Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(mask_rows.extent(0))),
        KOKKOS_LAMBDA(const int i) {
          mapping(i) =
              detail::find_row_2d(field_rows, num_field_rows, mask_rows(i).y);
        });
  } else {
    Kokkos::parallel_for(
        "playground_fields_row_map_kernel_3d",
        Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(mask_rows.extent(0))),
        KOKKOS_LAMBDA(const int i) {
          mapping(i) = detail::find_row_3d(field_rows, num_field_rows,
                                           mask_rows(i).y, mask_rows(i).z);
        });
  }

  ExecSpace().fence();
  return mapping;
}

template <class MaskMesh>
inline Kokkos::View<int*, DeviceMemorySpace>
build_interval_to_row_mapping(const MaskMesh& mask) {
  Kokkos::View<int*, DeviceMemorySpace> interval_rows(
      "playground_fields_interval_rows", mask.num_intervals);

  if (mask.num_rows == 0 || mask.num_intervals == 0) {
    return interval_rows;
  }

  auto row_ptr = mask.row_ptr;

  Kokkos::parallel_for(
      "playground_fields_fill_interval_rows",
      Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(mask.num_rows)),
      KOKKOS_LAMBDA(const int row_idx) {
        const std::size_t begin = static_cast<std::size_t>(row_ptr(row_idx));
        const std::size_t end = static_cast<std::size_t>(row_ptr(row_idx + 1));
        for (std::size_t k = begin; k < end; ++k) {
          interval_rows(k) = row_idx;
        }
      });

  ExecSpace().fence();
  return interval_rows;
}

template <class IntervalView>
KOKKOS_INLINE_FUNCTION
int find_interval_by_x(const IntervalView& intervals,
                       std::size_t begin,
                       std::size_t end,
                       const auto& x) {
  std::size_t lo = begin;
  std::size_t hi = end;

  while (lo < hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    if (intervals(mid).end <= x) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  if (lo < end) {
    const auto iv = intervals(lo);
    if (iv.begin <= x && x < iv.end) {
      return static_cast<int>(lo);
    }
  }
  return -1;
}

template <int DIM, class FieldMesh, class MaskMesh>
inline Kokkos::View<int*, DeviceMemorySpace>
build_interval_to_field_interval_mapping(
    const MaskMesh& mask,
    const FieldMesh& field,
    const Kokkos::View<int*, DeviceMemorySpace>& row_map) {
  Kokkos::View<int*, DeviceMemorySpace> mapping(
      "playground_fields_interval_to_field_interval", mask.num_intervals);
  if (mask.num_rows == 0 || mask.num_intervals == 0) {
    return mapping;
  }

  Kokkos::deep_copy(mapping, -1);

  auto mask_row_ptr = mask.row_ptr;
  auto mask_intervals = mask.intervals;
  auto field_row_ptr = field.row_ptr;
  auto field_intervals = field.intervals;

  Kokkos::parallel_for(
      "playground_fields_fill_interval_to_field_interval",
      Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(mask.num_rows)),
      KOKKOS_LAMBDA(const int row_idx) {
        const int field_row = row_map(row_idx);
        if (field_row < 0) return;

        const std::size_t mask_begin =
            static_cast<std::size_t>(mask_row_ptr(row_idx));
        const std::size_t mask_end =
            static_cast<std::size_t>(mask_row_ptr(row_idx + 1));
        const std::size_t field_begin =
            static_cast<std::size_t>(field_row_ptr(field_row));
        const std::size_t field_end =
            static_cast<std::size_t>(field_row_ptr(field_row + 1));

        for (std::size_t mi = mask_begin; mi < mask_end; ++mi) {
          const auto mask_iv = mask_intervals(mi);
          const int fi = find_interval_by_x(field_intervals, field_begin, field_end,
                                            mask_iv.begin);
          if (fi < 0) continue;
          const auto field_iv = field_intervals(static_cast<std::size_t>(fi));
          if (mask_iv.end <= field_iv.end) {
            mapping(mi) = fi;
          }
        }
      });

  ExecSpace().fence();
  return mapping;
}

template <int DIM, class FieldMesh, class MaskMesh>
inline FieldMaskMapping
build_mask_field_mapping(const FieldMesh& field,
                         const MaskMesh& mask) {
  FieldMaskMapping mapping;
  if (mask.num_rows == 0 || mask.num_intervals == 0 ||
      field.num_rows == 0 || field.num_intervals == 0) {
    return mapping;
  }

  mapping.row_map =
      build_row_map<DIM>(mask.row_keys, field.row_keys, field.num_rows);
  mapping.interval_to_row = build_interval_to_row_mapping(mask);
  mapping.interval_to_field_interval =
      build_interval_to_field_interval_mapping<DIM>(mask, field, mapping.row_map);

  return mapping;
}

template <class FieldMesh, class IntervalView>
KOKKOS_INLINE_FUNCTION
int find_row_for_key(const FieldMesh& field, const auto& key) {
  if constexpr (FieldMesh::dim_value == 2) {
    return detail::find_row_2d(field.row_keys, field.num_rows, key.y);
  } else {
    return detail::find_row_3d(field.row_keys, field.num_rows, key.y, key.z);
  }
}

template <class FieldMesh, class MaskInterval>
KOKKOS_INLINE_FUNCTION
int find_interval_containing_mask(const FieldMesh& field,
                                  int field_row,
                                  const MaskInterval& mask_iv) {
  if (field_row < 0) return -1;
  const std::size_t begin =
      static_cast<std::size_t>(field.row_ptr(field_row));
  const std::size_t end =
      static_cast<std::size_t>(field.row_ptr(field_row + 1));
  const int interval_idx = find_interval_by_x(field.intervals, begin, end, mask_iv.begin);
  if (interval_idx < 0) return -1;
  const auto iv = field.intervals(static_cast<std::size_t>(interval_idx));
  if (mask_iv.end <= iv.end) {
    return interval_idx;
  }
  return -1;
}

}  // namespace detail

struct StencilNeighbourMapping2D {
  Kokkos::View<int*, DeviceMemorySpace> north_interval;
  Kokkos::View<int*, DeviceMemorySpace> south_interval;
};

struct StencilNeighbourMapping3D {
  Kokkos::View<int*, DeviceMemorySpace> north_interval;
  Kokkos::View<int*, DeviceMemorySpace> south_interval;
  Kokkos::View<int*, DeviceMemorySpace> up_interval;
  Kokkos::View<int*, DeviceMemorySpace> down_interval;
};

template <class FieldMesh, class MaskMesh>
inline StencilNeighbourMapping2D
build_stencil_neighbour_mapping_2d(
    const FieldMesh& field,
    const MaskMesh& mask,
    const FieldMaskMapping& mapping) {
  StencilNeighbourMapping2D out;
  out.north_interval =
      Kokkos::View<int*, DeviceMemorySpace>("playground_fields_north", mask.num_intervals);
  out.south_interval =
      Kokkos::View<int*, DeviceMemorySpace>("playground_fields_south", mask.num_intervals);

  if (mask.num_intervals == 0 || mask.num_rows == 0 ||
      field.num_rows == 0 || field.num_intervals == 0) {
    return out;
  }

  Kokkos::deep_copy(out.north_interval, -1);
  Kokkos::deep_copy(out.south_interval, -1);

  auto mask_row_keys = mask.row_keys;
  auto mask_intervals = mask.intervals;
  auto interval_to_row = mapping.interval_to_row;
  auto row_map = mapping.row_map;

  Kokkos::parallel_for(
      "playground_fields_build_neighbour_mapping_2d",
      Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(mask.num_intervals)),
      KOKKOS_LAMBDA(const int interval_idx) {
        const int mask_row = interval_to_row(interval_idx);
        const int field_row = row_map(mask_row);
        if (mask_row < 0 || field_row < 0) return;

        const auto key = mask_row_keys(mask_row);
        const auto mask_iv = mask_intervals(static_cast<std::size_t>(interval_idx));

        const int row_north =
            detail::find_row_2d(field.row_keys, field.num_rows, key.y + 1);
        const int row_south =
            detail::find_row_2d(field.row_keys, field.num_rows, key.y - 1);

        out.north_interval(interval_idx) =
            detail::find_interval_containing_mask(field, row_north, mask_iv);
        out.south_interval(interval_idx) =
            detail::find_interval_containing_mask(field, row_south, mask_iv);
      });

  ExecSpace().fence();
  return out;
}

template <class FieldMesh, class MaskMesh>
inline StencilNeighbourMapping3D
build_stencil_neighbour_mapping_3d(
    const FieldMesh& field,
    const MaskMesh& mask,
    const FieldMaskMapping& mapping) {
  StencilNeighbourMapping3D out;
  out.north_interval =
      Kokkos::View<int*, DeviceMemorySpace>("playground_fields_north", mask.num_intervals);
  out.south_interval =
      Kokkos::View<int*, DeviceMemorySpace>("playground_fields_south", mask.num_intervals);
  out.up_interval =
      Kokkos::View<int*, DeviceMemorySpace>("playground_fields_up", mask.num_intervals);
  out.down_interval =
      Kokkos::View<int*, DeviceMemorySpace>("playground_fields_down", mask.num_intervals);

  if (mask.num_intervals == 0 || mask.num_rows == 0 ||
      field.num_rows == 0 || field.num_intervals == 0) {
    return out;
  }

  Kokkos::deep_copy(out.north_interval, -1);
  Kokkos::deep_copy(out.south_interval, -1);
  Kokkos::deep_copy(out.up_interval, -1);
  Kokkos::deep_copy(out.down_interval, -1);

  auto mask_row_keys = mask.row_keys;
  auto mask_intervals = mask.intervals;
  auto interval_to_row = mapping.interval_to_row;
  auto row_map = mapping.row_map;

  Kokkos::parallel_for(
      "playground_fields_build_neighbour_mapping_3d",
      Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(mask.num_intervals)),
      KOKKOS_LAMBDA(const int interval_idx) {
        const int mask_row = interval_to_row(interval_idx);
        const int field_row = row_map(mask_row);
        if (mask_row < 0 || field_row < 0) return;

        const auto key = mask_row_keys(mask_row);
        const auto mask_iv = mask_intervals(static_cast<std::size_t>(interval_idx));

        const int row_north =
            detail::find_row_3d(field.row_keys, field.num_rows, key.y + 1, key.z);
        const int row_south =
            detail::find_row_3d(field.row_keys, field.num_rows, key.y - 1, key.z);
        const int row_up =
            detail::find_row_3d(field.row_keys, field.num_rows, key.y, key.z + 1);
        const int row_down =
            detail::find_row_3d(field.row_keys, field.num_rows, key.y, key.z - 1);

        out.north_interval(interval_idx) =
            detail::find_interval_containing_mask(field, row_north, mask_iv);
        out.south_interval(interval_idx) =
            detail::find_interval_containing_mask(field, row_south, mask_iv);
        out.up_interval(interval_idx) =
            detail::find_interval_containing_mask(field, row_up, mask_iv);
        out.down_interval(interval_idx) =
            detail::find_interval_containing_mask(field, row_down, mask_iv);
      });

  ExecSpace().fence();
  return out;
}

template <class T>
struct CsrStencilPoint5 {
  Kokkos::View<T*, DeviceMemorySpace> values;
  std::size_t idx_center = 0;
  std::size_t idx_west = 0;
  std::size_t idx_east = 0;
  std::size_t idx_south = 0;
  std::size_t idx_north = 0;

  KOKKOS_INLINE_FUNCTION
  T center() const { return values(idx_center); }
  KOKKOS_INLINE_FUNCTION
  T west() const { return values(idx_west); }
  KOKKOS_INLINE_FUNCTION
  T east() const { return values(idx_east); }
  KOKKOS_INLINE_FUNCTION
  T south() const { return values(idx_south); }
  KOKKOS_INLINE_FUNCTION
  T north() const { return values(idx_north); }
};

template <class T>
struct CsrStencilPoint7 {
  Kokkos::View<T*, DeviceMemorySpace> values;
  std::size_t idx_center = 0;
  std::size_t idx_west = 0;
  std::size_t idx_east = 0;
  std::size_t idx_south = 0;
  std::size_t idx_north = 0;
  std::size_t idx_down = 0;
  std::size_t idx_up = 0;

  KOKKOS_INLINE_FUNCTION
  T center() const { return values(idx_center); }
  KOKKOS_INLINE_FUNCTION
  T west() const { return values(idx_west); }
  KOKKOS_INLINE_FUNCTION
  T east() const { return values(idx_east); }
  KOKKOS_INLINE_FUNCTION
  T south() const { return values(idx_south); }
  KOKKOS_INLINE_FUNCTION
  T north() const { return values(idx_north); }
  KOKKOS_INLINE_FUNCTION
  T down() const { return values(idx_down); }
  KOKKOS_INLINE_FUNCTION
  T up() const { return values(idx_up); }
};

template <class OutT, class InT, class MaskMesh, class CoordType = int32_t,
          class IndexType = std::size_t, class StencilFunctor>
inline void apply_csr_stencil_5pt_on_mask_device(
    Field<2, OutT, DeviceMemorySpace, CoordType, IndexType>& field_out,
    const Field<2, InT, DeviceMemorySpace, CoordType, IndexType>& field_in,
    const MaskMesh& mask,
    const FieldMaskMapping& mapping,
    const StencilNeighbourMapping2D& neighbours,
    StencilFunctor stencil) {
#ifndef NDEBUG
  if (field_out.mesh.row_keys.data() != field_in.mesh.row_keys.data() ||
      field_out.mesh.row_ptr.data() != field_in.mesh.row_ptr.data() ||
      field_out.mesh.intervals.data() != field_in.mesh.intervals.data() ||
      field_out.mesh.cell_offsets.data() != field_in.mesh.cell_offsets.data()) {
    throw std::runtime_error("Fields must share the same geometry for stencil operations");
  }
#endif

  if (mask.num_rows == 0 || mask.num_intervals == 0) {
    return;
  }

  auto interval_to_row = mapping.interval_to_row;
  auto interval_to_field_interval = mapping.interval_to_field_interval;
  auto mask_row_keys = mask.row_keys;
  auto mask_intervals = mask.intervals;

  auto out_intervals = field_out.mesh.intervals;
  auto out_offsets = field_out.mesh.cell_offsets;
  auto in_intervals = field_in.mesh.intervals;
  auto in_offsets = field_in.mesh.cell_offsets;

  auto north_interval = neighbours.north_interval;
  auto south_interval = neighbours.south_interval;

  auto values_in = field_in.values;
  auto values_out = field_out.values;

  using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
  using MemberType = TeamPolicy::member_type;

  const TeamPolicy policy(static_cast<int>(mask.num_intervals), Kokkos::AUTO);

  Kokkos::parallel_for(
      "playground_fields_apply_5pt_stencil_on_mask",
      policy,
      KOKKOS_LAMBDA(const MemberType& team) {
        const int mask_interval_idx = team.league_rank();
        const int mask_row_idx = interval_to_row(mask_interval_idx);
        if (mask_row_idx < 0) return;

        const int field_interval_idx =
            interval_to_field_interval(mask_interval_idx);
        if (field_interval_idx < 0) return;

        const auto mask_iv = mask_intervals(static_cast<std::size_t>(mask_interval_idx));
        const auto in_iv = in_intervals(static_cast<std::size_t>(field_interval_idx));
        const auto out_iv = out_intervals(static_cast<std::size_t>(field_interval_idx));

        const std::size_t base_in =
            static_cast<std::size_t>(in_offsets(static_cast<std::size_t>(field_interval_idx)));
        const std::size_t base_out =
            static_cast<std::size_t>(out_offsets(static_cast<std::size_t>(field_interval_idx)));

        const int north_iv_idx = north_interval(mask_interval_idx);
        const int south_iv_idx = south_interval(mask_interval_idx);

        const auto north_iv = in_intervals(static_cast<std::size_t>(north_iv_idx));
        const auto south_iv = in_intervals(static_cast<std::size_t>(south_iv_idx));
        const std::size_t north_base =
            static_cast<std::size_t>(in_offsets(static_cast<std::size_t>(north_iv_idx)));
        const std::size_t south_base =
            static_cast<std::size_t>(in_offsets(static_cast<std::size_t>(south_iv_idx)));

        const auto key = mask_row_keys(mask_row_idx);
        const CoordType y = key.y;

        const int team_size = team.team_size();
        const int team_rank = team.team_rank();

        for (CoordType x = static_cast<CoordType>(mask_iv.begin + team_rank);
             x < mask_iv.end;
             x = static_cast<CoordType>(x + team_size)) {
          const std::size_t idx_center =
              base_in + static_cast<std::size_t>(x - in_iv.begin);

          CsrStencilPoint5<InT> p;
          p.values = values_in;
          p.idx_center = idx_center;
          p.idx_west = idx_center - 1;
          p.idx_east = idx_center + 1;
          p.idx_north =
              north_base + static_cast<std::size_t>(x - north_iv.begin);
          p.idx_south =
              south_base + static_cast<std::size_t>(x - south_iv.begin);

          const OutT out_val = stencil(x, y, p);
          const std::size_t idx_out =
              base_out + static_cast<std::size_t>(x - out_iv.begin);
          values_out(idx_out) = out_val;
        }
      });

  ExecSpace().fence();
}

template <class OutT, class InT, class MaskMesh, class CoordType = int32_t,
          class IndexType = std::size_t, class StencilFunctor>
inline void apply_csr_stencil_5pt_on_mask_device(
    Field<2, OutT, DeviceMemorySpace, CoordType, IndexType>& field_out,
    const Field<2, InT, DeviceMemorySpace, CoordType, IndexType>& field_in,
    const MaskMesh& mask,
    StencilFunctor stencil) {
  const auto mapping = detail::build_mask_field_mapping<2>(field_in.mesh, mask);
  const auto neighbours = build_stencil_neighbour_mapping_2d(field_in.mesh, mask, mapping);
  apply_csr_stencil_5pt_on_mask_device(field_out, field_in, mask, mapping,
                                       neighbours, stencil);
}

template <class OutT, class InT, class MaskMesh, class CoordType = int32_t,
          class IndexType = std::size_t, class StencilFunctor>
inline void apply_csr_stencil_7pt_on_mask_device(
    Field<3, OutT, DeviceMemorySpace, CoordType, IndexType>& field_out,
    const Field<3, InT, DeviceMemorySpace, CoordType, IndexType>& field_in,
    const MaskMesh& mask,
    const FieldMaskMapping& mapping,
    const StencilNeighbourMapping3D& neighbours,
    StencilFunctor stencil) {
#ifndef NDEBUG
  if (field_out.mesh.row_keys.data() != field_in.mesh.row_keys.data() ||
      field_out.mesh.row_ptr.data() != field_in.mesh.row_ptr.data() ||
      field_out.mesh.intervals.data() != field_in.mesh.intervals.data() ||
      field_out.mesh.cell_offsets.data() != field_in.mesh.cell_offsets.data()) {
    throw std::runtime_error("Fields must share the same geometry for stencil operations");
  }
#endif

  if (mask.num_rows == 0 || mask.num_intervals == 0) {
    return;
  }

  auto interval_to_row = mapping.interval_to_row;
  auto interval_to_field_interval = mapping.interval_to_field_interval;
  auto mask_row_keys = mask.row_keys;
  auto mask_intervals = mask.intervals;

  auto out_intervals = field_out.mesh.intervals;
  auto out_offsets = field_out.mesh.cell_offsets;
  auto in_intervals = field_in.mesh.intervals;
  auto in_offsets = field_in.mesh.cell_offsets;

  auto north_interval = neighbours.north_interval;
  auto south_interval = neighbours.south_interval;
  auto up_interval = neighbours.up_interval;
  auto down_interval = neighbours.down_interval;

  auto values_in = field_in.values;
  auto values_out = field_out.values;

  using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
  using MemberType = TeamPolicy::member_type;

  const TeamPolicy policy(static_cast<int>(mask.num_intervals), Kokkos::AUTO);

  Kokkos::parallel_for(
      "playground_fields_apply_7pt_stencil_on_mask",
      policy,
      KOKKOS_LAMBDA(const MemberType& team) {
        const int mask_interval_idx = team.league_rank();
        const int mask_row_idx = interval_to_row(mask_interval_idx);
        if (mask_row_idx < 0) return;

        const int field_interval_idx =
            interval_to_field_interval(mask_interval_idx);
        if (field_interval_idx < 0) return;

        const auto mask_iv = mask_intervals(static_cast<std::size_t>(mask_interval_idx));
        const auto in_iv = in_intervals(static_cast<std::size_t>(field_interval_idx));
        const auto out_iv = out_intervals(static_cast<std::size_t>(field_interval_idx));

        const std::size_t base_in =
            static_cast<std::size_t>(in_offsets(static_cast<std::size_t>(field_interval_idx)));
        const std::size_t base_out =
            static_cast<std::size_t>(out_offsets(static_cast<std::size_t>(field_interval_idx)));

        const int north_iv_idx = north_interval(mask_interval_idx);
        const int south_iv_idx = south_interval(mask_interval_idx);
        const int up_iv_idx = up_interval(mask_interval_idx);
        const int down_iv_idx = down_interval(mask_interval_idx);

        const auto north_iv = in_intervals(static_cast<std::size_t>(north_iv_idx));
        const auto south_iv = in_intervals(static_cast<std::size_t>(south_iv_idx));
        const auto up_iv = in_intervals(static_cast<std::size_t>(up_iv_idx));
        const auto down_iv = in_intervals(static_cast<std::size_t>(down_iv_idx));

        const std::size_t north_base =
            static_cast<std::size_t>(in_offsets(static_cast<std::size_t>(north_iv_idx)));
        const std::size_t south_base =
            static_cast<std::size_t>(in_offsets(static_cast<std::size_t>(south_iv_idx)));
        const std::size_t up_base =
            static_cast<std::size_t>(in_offsets(static_cast<std::size_t>(up_iv_idx)));
        const std::size_t down_base =
            static_cast<std::size_t>(in_offsets(static_cast<std::size_t>(down_iv_idx)));

        const auto key = mask_row_keys(mask_row_idx);
        const CoordType y = key.y;
        const CoordType z = key.z;

        const int team_size = team.team_size();
        const int team_rank = team.team_rank();

        for (CoordType x = static_cast<CoordType>(mask_iv.begin + team_rank);
             x < mask_iv.end;
             x = static_cast<CoordType>(x + team_size)) {
          const std::size_t idx_center =
              base_in + static_cast<std::size_t>(x - in_iv.begin);

          CsrStencilPoint7<InT> p;
          p.values = values_in;
          p.idx_center = idx_center;
          p.idx_west = idx_center - 1;
          p.idx_east = idx_center + 1;
          p.idx_north =
              north_base + static_cast<std::size_t>(x - north_iv.begin);
          p.idx_south =
              south_base + static_cast<std::size_t>(x - south_iv.begin);
          p.idx_up =
              up_base + static_cast<std::size_t>(x - up_iv.begin);
          p.idx_down =
              down_base + static_cast<std::size_t>(x - down_iv.begin);

          const OutT out_val = stencil(x, y, z, p);
          const std::size_t idx_out =
              base_out + static_cast<std::size_t>(x - out_iv.begin);
          values_out(idx_out) = out_val;
        }
      });

  ExecSpace().fence();
}

template <class OutT, class InT, class MaskMesh, class CoordType = int32_t,
          class IndexType = std::size_t, class StencilFunctor>
inline void apply_csr_stencil_7pt_on_mask_device(
    Field<3, OutT, DeviceMemorySpace, CoordType, IndexType>& field_out,
    const Field<3, InT, DeviceMemorySpace, CoordType, IndexType>& field_in,
    const MaskMesh& mask,
    StencilFunctor stencil) {
  const auto mapping = detail::build_mask_field_mapping<3>(field_in.mesh, mask);
  const auto neighbours = build_stencil_neighbour_mapping_3d(field_in.mesh, mask, mapping);
  apply_csr_stencil_7pt_on_mask_device(field_out, field_in, mask, mapping,
                                       neighbours, stencil);
}

}  // namespace playground::subsetix::csr
