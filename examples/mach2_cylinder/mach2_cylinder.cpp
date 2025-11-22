#include <Kokkos_Core.hpp>

#include "../example_output.hpp"

#include <subsetix/csr_field.hpp>
#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_ops/core.hpp>
#include <subsetix/csr_ops/field_core.hpp>
#include <subsetix/csr_ops/field_amr.hpp>
#include <subsetix/csr_ops/field_stencil.hpp>
#include <subsetix/csr_ops/field_subview.hpp>
#include <subsetix/csr_ops/amr.hpp>
#include <subsetix/csr_ops/threshold.hpp>
#include <subsetix/csr_ops/morphology.hpp>
#include <subsetix/detail/csr_utils.hpp>
#include <subsetix/multilevel.hpp>
#include <subsetix/vtk_export.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <filesystem>
#include <array>

namespace {

using Real = float;

using subsetix::csr::Box2D;
using subsetix::csr::Coord;
using subsetix::csr::CsrSetAlgebraContext;
using subsetix::csr::Disk2D;
using subsetix::csr::Field2DDevice;
using subsetix::csr::IntervalField2DHost;
using subsetix::csr::IntervalSet2DDevice;
using subsetix::csr::IntervalSet2DHost;
using subsetix::csr::make_box_device;
using subsetix::csr::make_disk_device;
using subsetix::csr::make_bitmap_device;
using subsetix::csr::set_difference_device;
using subsetix::csr::detail::FieldReadAccessor;
using subsetix::csr::detail::build_mask_field_mapping;
using subsetix::csr::Interval;
using subsetix::csr::copy_subview_device;
using subsetix::vtk::write_legacy_quads;
using subsetix::MultilevelGeoDevice;
using subsetix::MultilevelFieldDevice;

using Clock = std::chrono::steady_clock;
constexpr int MAX_AMR_LEVELS = 6; // level 0 + up to 5 refined levels

struct Conserved {
  Real rho;
  Real rhou;
  Real rhov;
  Real E;
};

struct Primitive {
  Real rho;
  Real u;
  Real v;
  Real p;
};

struct RunConfig {
  int nx = 400;
  int ny = 160;
  int cx = -1; // set later from nx if still negative
  int cy = -1; // set later from ny if still negative
  int radius = 20;

  Real mach_inlet = static_cast<Real>(2.0);
  Real rho = static_cast<Real>(1.0);
  Real p = static_cast<Real>(1.0);
  Real gamma = static_cast<Real>(1.4);
  Real cfl = static_cast<Real>(0.45);
  Real t_final = static_cast<Real>(0.01);
  int max_steps = 5000;
  int output_stride = 50;
  int max_amr_levels = 4; // includes the coarse level
  bool no_slip = false;
  bool enable_output = true;
  std::string pbm_path;

  bool enable_amr = true;
  Real amr_fraction = static_cast<Real>(0.5); // fraction of domain length refined in each direction
  int amr_guard = 2;         // coarse-cell guard radius around the refined zone
  int amr_remesh_stride = 0; // 0 => static AMR, >0 => remesh every N steps
};

struct RemeshTiming {
  double masks = 0.0;
  double mask_indicator = 0.0;
  double mask_reduce = 0.0;
  double mask_expand = 0.0;
  double mask_constrain = 0.0;
  double geom = 0.0;
  double prolong = 0.0;
  double overlap = 0.0;
};

template <typename T>
FieldReadAccessor<T>
make_accessor(const Field2DDevice<T>& field) {
  FieldReadAccessor<T> acc;
  acc.row_keys = field.geometry.row_keys;
  acc.row_ptr = field.geometry.row_ptr;
  acc.intervals = field.geometry.intervals;
  acc.offsets = field.geometry.cell_offsets;
  acc.values = field.values;
  acc.num_rows = field.geometry.num_rows;
  return acc;
}

KOKKOS_INLINE_FUNCTION
Primitive cons_to_prim(const Conserved& U, Real gamma) {
  constexpr Real eps = static_cast<Real>(1e-12);
  Primitive q;
  q.rho = U.rho;
  const Real inv_rho = static_cast<Real>(1.0) / (U.rho + eps);
  q.u = U.rhou * inv_rho;
  q.v = U.rhov * inv_rho;
  const Real kinetic = static_cast<Real>(0.5) * (q.u * q.u + q.v * q.v);
  const Real pressure = (gamma - static_cast<Real>(1.0)) * (U.E - U.rho * kinetic);
  q.p = (pressure > eps) ? pressure : eps;
  return q;
}

KOKKOS_INLINE_FUNCTION
Conserved prim_to_cons(const Primitive& q, Real gamma) {
  Conserved U;
  const Real kinetic = static_cast<Real>(0.5) * q.rho * (q.u * q.u + q.v * q.v);
  U.rho = q.rho;
  U.rhou = q.rho * q.u;
  U.rhov = q.rho * q.v;
  U.E = q.p / (gamma - static_cast<Real>(1.0)) + kinetic;
  return U;
}

KOKKOS_INLINE_FUNCTION
Real sound_speed(const Primitive& q, Real gamma) {
  constexpr Real eps = static_cast<Real>(1e-12);
  return std::sqrt(gamma * q.p / (q.rho + eps));
}

KOKKOS_INLINE_FUNCTION
Conserved flux_x(const Conserved& U, const Primitive& q) {
  Conserved F;
  F.rho = U.rhou;
  F.rhou = U.rho * q.u * q.u + q.p;
  F.rhov = U.rho * q.u * q.v;
  F.E = (U.E + q.p) * q.u;
  return F;
}

KOKKOS_INLINE_FUNCTION
Conserved flux_y(const Conserved& U, const Primitive& q) {
  Conserved F;
  F.rho = U.rhov;
  F.rhou = U.rho * q.u * q.v;
  F.rhov = U.rho * q.v * q.v + q.p;
  F.E = (U.E + q.p) * q.v;
  return F;
}

KOKKOS_INLINE_FUNCTION
Conserved rusanov_flux_x(const Conserved& UL,
                         const Conserved& UR,
                         const Primitive& qL,
                         const Primitive& qR,
                         Real gamma) {
  const Real aL = sound_speed(qL, gamma);
  const Real aR = sound_speed(qR, gamma);
  const Real smax = std::fmax(std::fabs(qL.u) + aL,
                              std::fabs(qR.u) + aR);

  const Conserved FL = flux_x(UL, qL);
  const Conserved FR = flux_x(UR, qR);

  Conserved F;
  F.rho = 0.5 * (FL.rho + FR.rho) - 0.5 * smax * (UR.rho - UL.rho);
  F.rhou = 0.5 * (FL.rhou + FR.rhou) - 0.5 * smax * (UR.rhou - UL.rhou);
  F.rhov = 0.5 * (FL.rhov + FR.rhov) - 0.5 * smax * (UR.rhov - UL.rhov);
  F.E = 0.5 * (FL.E + FR.E) - 0.5 * smax * (UR.E - UL.E);
  return F;
}

KOKKOS_INLINE_FUNCTION
Conserved rusanov_flux_y(const Conserved& UL,
                         const Conserved& UR,
                         const Primitive& qL,
                         const Primitive& qR,
                         Real gamma) {
  const Real aL = sound_speed(qL, gamma);
  const Real aR = sound_speed(qR, gamma);
  const Real smax = std::fmax(std::fabs(qL.v) + aL,
                              std::fabs(qR.v) + aR);

  const Conserved FL = flux_y(UL, qL);
  const Conserved FR = flux_y(UR, qR);

  Conserved F;
  F.rho = 0.5 * (FL.rho + FR.rho) - 0.5 * smax * (UR.rho - UL.rho);
  F.rhou = 0.5 * (FL.rhou + FR.rhou) - 0.5 * smax * (UR.rhou - UL.rhou);
  F.rhov = 0.5 * (FL.rhov + FR.rhov) - 0.5 * smax * (UR.rhov - UL.rhov);
  F.E = 0.5 * (FL.E + FR.E) - 0.5 * smax * (UR.E - UL.E);
  return F;
}

KOKKOS_INLINE_FUNCTION
bool in_domain(Coord x, Coord y, const Box2D& domain) {
  return (x >= domain.x_min && x < domain.x_max &&
          y >= domain.y_min && y < domain.y_max);
}

KOKKOS_INLINE_FUNCTION
bool contains_point(const IntervalSet2DDevice& set,
                    Coord x,
                    Coord y) {
  const int row_idx =
      subsetix::csr::detail::find_row_index(set.row_keys, set.num_rows, y);
  if (row_idx < 0) {
    return false;
  }
  const std::size_t begin = set.row_ptr(row_idx);
  const std::size_t end = set.row_ptr(row_idx + 1);
  const int interval_idx =
      subsetix::csr::detail::find_interval_index(set.intervals, begin, end, x);
  return interval_idx >= 0;
}

KOKKOS_INLINE_FUNCTION
Conserved make_wall_ghost(const Conserved& interior,
                          Real nx,
                          Real ny,
                          Real gamma,
                          bool no_slip) {
  Primitive q = cons_to_prim(interior, gamma);
  const Real un = q.u * nx + q.v * ny;
  if (no_slip) {
    q.u = 0.0;
    q.v = 0.0;
  } else {
    q.u = q.u - 2.0 * un * nx;
    q.v = q.v - 2.0 * un * ny;
  }
  return prim_to_cons(q, gamma);
}

KOKKOS_INLINE_FUNCTION
Conserved sample_neighbor(const Conserved& center,
                          Coord x,
                          Coord y,
                          int dx,
                          int dy,
                          const FieldReadAccessor<Conserved>& acc,
                          const Box2D& domain,
                          const Conserved& inflow,
                          Real gamma,
                          bool no_slip) {
  const Coord xn = static_cast<Coord>(x + dx);
  const Coord yn = static_cast<Coord>(y + dy);

  Conserved neigh;
  if (acc.try_get(xn, yn, neigh)) {
    return neigh; // fluid neighbor
  }

  if (!in_domain(xn, yn, domain)) {
    // Physical domain boundary
    if (dx == -1 && dy == 0) {
      // Inlet
      return inflow;
    }
    if (dx == 1 && dy == 0) {
      // Supersonic outflow: extrapolate
      return center;
    }
    // Top/bottom walls
    const Real nx = static_cast<Real>(0.0);
    const Real ny = (dy > 0) ? static_cast<Real>(1.0) : static_cast<Real>(-1.0);
    return make_wall_ghost(center, nx, ny, gamma, no_slip);
  }

  // Inside rectangular domain -> obstacle
  const Real nx = (dx != 0) ? ((dx > 0) ? static_cast<Real>(1.0) : static_cast<Real>(-1.0))
                            : static_cast<Real>(0.0);
  const Real ny = (dy != 0) ? ((dy > 0) ? static_cast<Real>(1.0) : static_cast<Real>(-1.0))
                            : static_cast<Real>(0.0);
  return make_wall_ghost(center, nx, ny, gamma, no_slip);
}

KOKKOS_INLINE_FUNCTION
Conserved sample_neighbor_with_coarse(
    const Conserved& center,
    Coord x,
    Coord y,
    int dx,
    int dy,
    const FieldReadAccessor<Conserved>& acc_fine,
    const FieldReadAccessor<Conserved>* acc_coarse,
    const Box2D& domain_fine,
    const Conserved& inflow,
    Real gamma,
    bool no_slip) {
  const Coord xn = static_cast<Coord>(x + dx);
  const Coord yn = static_cast<Coord>(y + dy);

  Conserved neigh;
  if (acc_fine.try_get(xn, yn, neigh)) {
    return neigh;
  }

  const bool inside = in_domain(xn, yn, domain_fine);
  if (inside && acc_coarse != nullptr) {
    const Coord xc = subsetix::csr::detail::floor_div2(xn);
    const Coord yc = subsetix::csr::detail::floor_div2(yn);
    if (acc_coarse->try_get(xc, yc, neigh)) {
      return neigh;
    }
  }

  if (!inside) {
    if (dx == -1 && dy == 0) {
      return inflow;
    }
    if (dx == 1 && dy == 0) {
      return center;
    }
    const Real nx = (dy == 0) ? ((dx > 0) ? static_cast<Real>(1.0) : static_cast<Real>(-1.0))
                              : static_cast<Real>(0.0);
    const Real ny = (dx == 0) ? ((dy > 0) ? static_cast<Real>(1.0) : static_cast<Real>(-1.0))
                              : static_cast<Real>(0.0);
    return make_wall_ghost(center, nx, ny, gamma, no_slip);
  }

  const Real nx = static_cast<Real>(-dx);
  const Real ny = static_cast<Real>(-dy);
  return make_wall_ghost(center, nx, ny, gamma, no_slip);
}

KOKKOS_INLINE_FUNCTION
Conserved make_wall_from_neighbor(const Conserved& interior,
                                  int dx,
                                  int dy,
                                  Real gamma,
                                  bool no_slip) {
  const Real nx = (dx != 0)
                      ? ((dx > 0) ? static_cast<Real>(1.0)
                                  : static_cast<Real>(-1.0))
                      : static_cast<Real>(0.0);
  const Real ny = (dy != 0)
                      ? ((dy > 0) ? static_cast<Real>(1.0)
                                  : static_cast<Real>(-1.0))
                      : static_cast<Real>(0.0);
  return make_wall_ghost(interior, nx, ny, gamma, no_slip);
}

void fill_ghost_cells(Field2DDevice<Conserved>& field,
                      const IntervalSet2DDevice& ghost_mask,
                      const IntervalSet2DDevice& base_mask,
                      const Box2D& domain,
                      const Conserved& inflow,
                      Real gamma,
                      bool no_slip) {
  if (ghost_mask.num_rows == 0 || ghost_mask.num_intervals == 0) {
    return;
  }

  const auto acc = make_accessor(field);

  apply_on_set_device(
      field, ghost_mask,
      KOKKOS_LAMBDA(
          Coord x, Coord y, Conserved& value, std::size_t /*idx*/) {
        const bool outside = !in_domain(x, y, domain);

        const auto clamp_coord = [&](Coord v, Coord vmin, Coord vmax) {
          return (v < vmin) ? vmin : ((v >= vmax) ? static_cast<Coord>(vmax - 1) : v);
        };

        if (outside) {
          const bool left = (x < domain.x_min);
          const bool right = (x >= domain.x_max);
          const bool bottom = (y < domain.y_min);

          if (left) {
            value = inflow;
            return;
          }
          if (right) {
            const Coord yc = clamp_coord(y, domain.y_min, domain.y_max);
            value = acc.value_at(static_cast<Coord>(domain.x_max - 1), yc);
            return;
          }

          const Coord xc = clamp_coord(x, domain.x_min, domain.x_max);
          const Coord yc = bottom ? static_cast<Coord>(domain.y_min)
                                  : static_cast<Coord>(domain.y_max - 1);
          const Conserved interior = acc.value_at(xc, yc);
          const Real ny = bottom ? static_cast<Real>(-1.0)
                                 : static_cast<Real>(1.0);
          value = make_wall_ghost(interior,
                                  static_cast<Real>(0.0), ny,
                                  gamma, no_slip);
          return;
        }

        auto has_base = [&](Coord xb, Coord yb) {
          return contains_point(base_mask, xb, yb);
        };

        Conserved neigh;
        int ndx = 0;
        int ndy = 0;

        auto try_dir = [&](int dx, int dy) {
          const Coord xn = static_cast<Coord>(x + dx);
          const Coord yn = static_cast<Coord>(y + dy);
          if (!has_base(xn, yn)) {
            return false;
          }
          neigh = acc.value_at(xn, yn);
          ndx = dx;
          ndy = dy;
          return true;
        };

        const bool found =
            try_dir(-1, 0) || try_dir(1, 0) ||
            try_dir(0, -1) || try_dir(0, 1);

        if (found) {
          value = make_wall_from_neighbor(neigh, ndx, ndy, gamma, no_slip);
          return;
        }

        // Fallback: use clamped interior as copy (should be rare)
        const Coord xc = clamp_coord(x, domain.x_min, domain.x_max);
        const Coord yc = clamp_coord(y, domain.y_min, domain.y_max);
        value = acc.value_at(xc, yc);
      });
}

Real compute_dt(const Field2DDevice<Conserved>& U,
                Real gamma,
                Real cfl,
                Real dx,
                Real dy) {
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  const Real inv_dx = static_cast<Real>(1.0) / dx;
  const Real inv_dy = static_cast<Real>(1.0) / dy;
  Real max_rate = static_cast<Real>(0.0);

  Kokkos::parallel_reduce(
      "mach2_cylinder_dt",
      Kokkos::RangePolicy<ExecSpace>(
          0, static_cast<int>(U.size())),
      KOKKOS_LAMBDA(const int idx, Real& lmax) {
        const Conserved s = U.values(idx);
        const Primitive q = cons_to_prim(s, gamma);
        const Real a = sound_speed(q, gamma);
        const Real rate = std::fabs(q.u) * inv_dx +
                          std::fabs(q.v) * inv_dy +
                          a * (inv_dx + inv_dy);
        if (rate > lmax) {
          lmax = rate;
        }
      },
      Kokkos::Max<Real>(max_rate));

  if (max_rate <= static_cast<Real>(0.0)) {
    return cfl * std::min(dx, dy);
  }
  return cfl / max_rate;
}

IntervalSet2DDevice ensure_subset(const IntervalSet2DDevice& region,
                                  const IntervalSet2DDevice& field_geom,
                                  CsrSetAlgebraContext& ctx);

struct AmrLayout {
  IntervalSet2DDevice coarse_mask;
  IntervalSet2DDevice fine_full;
  IntervalSet2DDevice fine_active;
  IntervalSet2DDevice fine_with_guard;
  IntervalSet2DDevice fine_guard;
  IntervalSet2DDevice projection_fine_on_coarse;
  Box2D fine_domain{0, 0, 0, 0};
  bool has_fine = false;
};

IntervalSet2DDevice build_refine_mask(const Field2DDevice<Conserved>& U,
                                      const IntervalSet2DDevice& fluid_dev,
                                      const Box2D& domain,
                                      const RunConfig& cfg,
                                      CsrSetAlgebraContext& ctx,
                                      RemeshTiming* timers = nullptr) {
  IntervalSet2DDevice mask;
  if (!cfg.enable_amr) {
    return mask;
  }

  const auto t_masks_begin = Clock::now();

  auto make_central_box_mask = [&]() {
    const Coord span_x = domain.x_max - domain.x_min;
    const Coord span_y = domain.y_max - domain.y_min;
    const Coord refine_w = static_cast<Coord>(
        std::max<int>(4, static_cast<int>(cfg.amr_fraction * span_x)));
    const Coord refine_h = static_cast<Coord>(
        std::max<int>(4, static_cast<int>(cfg.amr_fraction * span_y)));
    const Coord refine_x0 =
        domain.x_min + static_cast<Coord>((span_x - refine_w) / 2);
    const Coord refine_y0 =
        domain.y_min + static_cast<Coord>((span_y - refine_h) / 2);
    const Box2D refine_box{
        refine_x0, static_cast<Coord>(refine_x0 + refine_w),
        refine_y0, static_cast<Coord>(refine_y0 + refine_h)};

    const auto refine_box_dev = make_box_device(refine_box);
    IntervalSet2DDevice coarse_mask =
        subsetix::csr::allocate_interval_set_device(
            fluid_dev.num_rows,
            fluid_dev.num_intervals + refine_box_dev.num_intervals);
    set_intersection_device(fluid_dev, refine_box_dev, coarse_mask, ctx);
    subsetix::csr::compute_cell_offsets_device(coarse_mask);
    return coarse_mask;
  };

  Field2DDevice<Real> indicator(fluid_dev, "mach2_refine_indicator");
  auto acc = make_accessor(U);
  const Real dx = static_cast<Real>(1.0);
  const Real dy = static_cast<Real>(1.0);
  const Real inv_dx = static_cast<Real>(1.0) / dx;
  const Real inv_dy = static_cast<Real>(1.0) / dy;
  const auto t_indicator_begin = Clock::now();
  const IntervalSet2DDevice indicator_region =
      ensure_subset(fluid_dev, U.geometry, ctx);
  const auto mapping = build_mask_field_mapping(U, indicator_region);
  auto interval_to_row = mapping.interval_to_row;
  auto interval_to_field_interval = mapping.interval_to_field_interval;
  auto row_map = mapping.row_map;

  auto mask_row_keys = indicator_region.row_keys;
  auto mask_intervals = indicator_region.intervals;
  auto mask_offsets = indicator_region.cell_offsets;

  auto field_intervals = U.geometry.intervals;
  auto field_offsets = U.geometry.cell_offsets;
  auto field_values = U.values;
  auto indicator_values = indicator.values;

  const auto vertical = subsetix::csr::detail::build_vertical_interval_mapping(U);
  auto up_interval = vertical.up_interval;
  auto down_interval = vertical.down_interval;

  Kokkos::parallel_for(
      "mach2_refine_indicator",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
          0, static_cast<int>(indicator_region.num_intervals)),
      KOKKOS_LAMBDA(const int interval_idx) {
        const int row_idx = interval_to_row(interval_idx);
        const int field_row = row_map(row_idx);
        if (row_idx < 0 || field_row < 0) {
          return;
        }

        const int field_interval_idx =
            interval_to_field_interval(interval_idx);
        if (field_interval_idx < 0) {
          return;
        }

        const auto mask_iv = mask_intervals(interval_idx);
        const auto field_iv = field_intervals(field_interval_idx);
        const std::size_t out_base_offset = mask_offsets(interval_idx);
        const Coord out_base_begin = mask_iv.begin;
        const std::size_t in_base_offset = field_offsets(field_interval_idx);
        const Coord in_base_begin = field_iv.begin;

        for (Coord x = mask_iv.begin; x < mask_iv.end; ++x) {
          const std::size_t out_idx =
              out_base_offset +
              static_cast<std::size_t>(x - out_base_begin);
          const std::size_t in_idx =
              in_base_offset +
              static_cast<std::size_t>(x - in_base_begin);

          const Conserved center = field_values(in_idx);

          Real rho_w = center.rho;
          Real rho_e = center.rho;
          Real rho_n = center.rho;
          Real rho_s = center.rho;

          if (x > field_iv.begin) {
            rho_w = field_values(in_idx - 1).rho;
          }
          if (x + 1 < field_iv.end) {
            rho_e = field_values(in_idx + 1).rho;
          }

          const int up_idx = up_interval(field_interval_idx);
          if (up_idx >= 0) {
            const subsetix::csr::Interval iv_up = field_intervals(up_idx);
            if (x >= iv_up.begin && x < iv_up.end) {
              const std::size_t off =
                  field_offsets(up_idx) +
                  static_cast<std::size_t>(x - iv_up.begin);
              rho_n = field_values(off).rho;
            }
          }

          const int down_idx = down_interval(field_interval_idx);
          if (down_idx >= 0) {
            const subsetix::csr::Interval iv_down = field_intervals(down_idx);
            if (x >= iv_down.begin && x < iv_down.end) {
              const std::size_t off =
                  field_offsets(down_idx) +
                  static_cast<std::size_t>(x - iv_down.begin);
              rho_s = field_values(off).rho;
            }
          }

          const Real gx = static_cast<Real>(0.5) * (rho_e - rho_w) * inv_dx;
          const Real gy = static_cast<Real>(0.5) * (rho_n - rho_s) * inv_dy;
          indicator_values(out_idx) = std::fabs(gx) + std::fabs(gy);
        }
      });
  Kokkos::DefaultExecutionSpace().fence();
  const auto t_indicator_end = Clock::now();
  if (timers) {
    timers->mask_indicator += std::chrono::duration<double, std::milli>(
                                  t_indicator_end - t_indicator_begin)
                                  .count();
  }

  using ExecSpace = Kokkos::DefaultExecutionSpace;
  Real max_grad = static_cast<Real>(0.0);
  const auto t_reduce_begin = Clock::now();
  Kokkos::parallel_reduce(
      "mach2_refine_indicator_max",
      Kokkos::RangePolicy<ExecSpace>(
          0, static_cast<int>(indicator.size())),
      KOKKOS_LAMBDA(const int idx, Real& lmax) {
        const Real v = indicator_values(idx);
        if (v > lmax) {
          lmax = v;
        }
      },
      Kokkos::Max<Real>(max_grad));
  const auto t_reduce_end = Clock::now();
  if (timers) {
    timers->mask_reduce += std::chrono::duration<double, std::milli>(
                               t_reduce_end - t_reduce_begin)
                               .count();
  }

  Real threshold = max_grad * cfg.amr_fraction;
  constexpr Real min_thresh = static_cast<Real>(1e-10);
  if (threshold < min_thresh) {
    threshold = min_thresh;
  }
  mask = subsetix::csr::threshold_field(indicator, threshold);
  subsetix::csr::compute_cell_offsets_device(mask);

  if (mask.num_intervals == 0 || mask.num_rows == 0) {
    mask = make_central_box_mask();
  }

  const Coord smooth = static_cast<Coord>(1);
  if (smooth > 0 && mask.num_rows > 0 && mask.num_intervals > 0) {
    IntervalSet2DDevice expanded;
    const auto t_expand_begin = Clock::now();
    expand_device(mask, smooth, smooth, expanded, ctx);
    subsetix::csr::compute_cell_offsets_device(expanded);
    Kokkos::DefaultExecutionSpace().fence();
    const auto t_expand_end = Clock::now();
    if (timers) {
      timers->mask_expand += std::chrono::duration<double, std::milli>(
                                 t_expand_end - t_expand_begin)
                                 .count();
    }
    mask = expanded;
  }

  const auto t_masks_end = Clock::now();
  if (timers) {
    timers->masks += std::chrono::duration<double, std::milli>(
                         t_masks_end - t_masks_begin)
                         .count();
  }

  return mask;
}

AmrLayout build_fine_geometry(const IntervalSet2DDevice& fluid_dev,
                              const IntervalSet2DDevice& coarse_mask,
                              Coord guard_coarse,
                              const Box2D& coarse_domain,
                              CsrSetAlgebraContext& ctx) {
  AmrLayout layout;
  layout.coarse_mask = coarse_mask;
  if (coarse_mask.num_intervals == 0 || coarse_mask.num_rows == 0) {
    return layout;
  }

  IntervalSet2DDevice fine_full;
  refine_level_up_device(fluid_dev, fine_full, ctx);
  subsetix::csr::compute_cell_offsets_device(fine_full);
  layout.fine_full = fine_full;

  IntervalSet2DDevice fine_mask;
  refine_level_up_device(coarse_mask, fine_mask, ctx);
  subsetix::csr::compute_cell_offsets_device(fine_mask);

  layout.fine_active = subsetix::csr::allocate_interval_set_device(
      fine_full.num_rows, fine_full.num_intervals + fine_mask.num_intervals);
  set_intersection_device(fine_full, fine_mask, layout.fine_active, ctx);
  subsetix::csr::compute_cell_offsets_device(layout.fine_active);

  layout.fine_domain = Box2D{static_cast<Coord>(coarse_domain.x_min * 2),
                             static_cast<Coord>(coarse_domain.x_max * 2),
                             static_cast<Coord>(coarse_domain.y_min * 2),
                             static_cast<Coord>(coarse_domain.y_max * 2)};

  const Coord guard_fine = static_cast<Coord>(2 * guard_coarse);
  IntervalSet2DDevice fine_with_guard_raw;
  expand_device(layout.fine_active, guard_fine, guard_fine,
                fine_with_guard_raw, ctx);
  subsetix::csr::compute_cell_offsets_device(fine_with_guard_raw);

  // Clip guard to the refined fluid to avoid sampling obstacle/void coarse cells.
  layout.fine_with_guard = subsetix::csr::allocate_interval_set_device(
      fine_full.num_rows, fine_full.num_intervals + fine_with_guard_raw.num_intervals);
  set_intersection_device(fine_with_guard_raw, fine_full, layout.fine_with_guard, ctx);
  subsetix::csr::compute_cell_offsets_device(layout.fine_with_guard);

  layout.fine_guard = subsetix::csr::allocate_interval_set_device(
      layout.fine_with_guard.num_rows,
      layout.fine_with_guard.num_intervals + layout.fine_active.num_intervals);
  set_difference_device(layout.fine_with_guard, layout.fine_active,
                        layout.fine_guard, ctx);
  subsetix::csr::compute_cell_offsets_device(layout.fine_guard);

  project_level_down_device(layout.fine_active,
                            layout.projection_fine_on_coarse, ctx);
  subsetix::csr::compute_cell_offsets_device(layout.projection_fine_on_coarse);

  // Clip projection to the coarse fluid domain to guard against masks that
  // extend over physical boundaries during remesh.
  IntervalSet2DDevice projection_clipped = subsetix::csr::allocate_interval_set_device(
      layout.projection_fine_on_coarse.num_rows,
      layout.projection_fine_on_coarse.num_intervals);
  set_intersection_device(layout.projection_fine_on_coarse, fluid_dev,
                          projection_clipped, ctx);
  subsetix::csr::compute_cell_offsets_device(projection_clipped);
  layout.projection_fine_on_coarse = projection_clipped;

  layout.has_fine = layout.fine_active.num_rows > 0 &&
                    layout.fine_active.num_intervals > 0;
  return layout;
}

IntervalSet2DDevice constrain_mask_to_parent_interior(
    const IntervalSet2DDevice& mask,
    const IntervalSet2DDevice& parent_fluid,
    const IntervalSet2DDevice& parent_active,
    Coord buffer,
    CsrSetAlgebraContext& ctx) {
  if (mask.num_rows == 0 || mask.num_intervals == 0) {
    return mask;
  }

  const Coord guard = (buffer < static_cast<Coord>(1))
                          ? static_cast<Coord>(1)
                          : buffer;

  IntervalSet2DDevice parent_out = subsetix::csr::allocate_interval_set_device(
      parent_fluid.num_rows,
      parent_fluid.num_intervals + parent_active.num_intervals);
  set_difference_device(parent_fluid, parent_active, parent_out, ctx);
  subsetix::csr::compute_cell_offsets_device(parent_out);

  IntervalSet2DDevice parent_out_expanded;
  expand_device(parent_out, guard, guard, parent_out_expanded, ctx);
  subsetix::csr::compute_cell_offsets_device(parent_out_expanded);

  IntervalSet2DDevice allowed = subsetix::csr::allocate_interval_set_device(
      parent_fluid.num_rows,
      parent_fluid.num_intervals + parent_out_expanded.num_intervals);
  set_difference_device(parent_fluid, parent_out_expanded, allowed, ctx);
  subsetix::csr::compute_cell_offsets_device(allowed);

  IntervalSet2DDevice clipped = subsetix::csr::allocate_interval_set_device(
      std::max(mask.num_rows, allowed.num_rows),
      mask.num_intervals + allowed.num_intervals);
  set_intersection_device(mask, allowed, clipped, ctx);
  subsetix::csr::compute_cell_offsets_device(clipped);

  if (clipped.num_intervals == 0 || clipped.num_rows == 0) {
    clipped = allowed;
  }
  return clipped;
}

IntervalSet2DDevice ensure_subset(const IntervalSet2DDevice& region,
                                  const IntervalSet2DDevice& field_geom,
                                  CsrSetAlgebraContext& ctx) {
  if (region.num_rows == 0 || region.num_intervals == 0) {
    return region;
  }
  IntervalSet2DDevice subset = subsetix::csr::allocate_interval_set_device(
      std::max(region.num_rows, field_geom.num_rows),
      region.num_intervals + field_geom.num_intervals);
  set_intersection_device(region, field_geom, subset, ctx);
  subsetix::csr::compute_cell_offsets_device(subset);
  return subset;
}

Real compute_dt_on_set(const Field2DDevice<Conserved>& U,
                       const IntervalSet2DDevice& region,
                       Real gamma,
                       Real cfl,
                       Real dx,
                       Real dy) {
  if (region.num_intervals == 0) {
    return compute_dt(U, gamma, cfl, dx, dy);
  }

  const Real inv_dx = static_cast<Real>(1.0) / dx;
  const Real inv_dy = static_cast<Real>(1.0) / dy;
  Real max_rate = static_cast<Real>(0.0);

  const auto mapping = build_mask_field_mapping(U, region);
  auto interval_to_row = mapping.interval_to_row;
  auto interval_to_field_interval = mapping.interval_to_field_interval;
  auto row_map = mapping.row_map;

  auto mask_row_keys = region.row_keys;
  auto mask_intervals = region.intervals;

  auto field_intervals = U.geometry.intervals;
  auto field_offsets = U.geometry.cell_offsets;
  auto values = U.values;

  Kokkos::parallel_reduce(
      "mach2_cylinder_dt_on_set",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
          0, static_cast<int>(region.num_intervals)),
      KOKKOS_LAMBDA(const int interval_idx, Real& lmax) {
        const int row_idx = interval_to_row(interval_idx);
        const int field_row = row_map(row_idx);
        if (row_idx < 0 || field_row < 0) {
          return;
        }

        const int field_interval_idx =
            interval_to_field_interval(interval_idx);
        if (field_interval_idx < 0) {
          return;
        }

        const Interval mask_iv = mask_intervals(interval_idx);
        const Interval iv = field_intervals(field_interval_idx);
        const std::size_t base = field_offsets(field_interval_idx);
        const Coord base_begin = iv.begin;

        for (Coord x = mask_iv.begin; x < mask_iv.end; ++x) {
          const std::size_t idx =
              base + static_cast<std::size_t>(x - base_begin);
          const Conserved s = values(idx);
          const Primitive q = cons_to_prim(s, gamma);
          const Real a = sound_speed(q, gamma);
          const Real rate = std::fabs(q.u) * inv_dx +
                            std::fabs(q.v) * inv_dy +
                            a * (inv_dx + inv_dy);
          if (rate > lmax) {
            lmax = rate;
          }
        }
      },
      Kokkos::Max<Real>(max_rate));

  if (max_rate <= static_cast<Real>(0.0)) {
    return cfl * std::min(dx, dy);
  }
  return cfl / max_rate;
}

void prolong_full(Field2DDevice<Conserved>& fine_field,
                  const IntervalSet2DDevice& fine_mask,
                  const Field2DDevice<Conserved>& coarse_field,
                  CsrSetAlgebraContext& ctx) {
  (void)ctx;
  const auto coarse_acc = make_accessor(coarse_field);
  apply_on_set_device(
      fine_field, fine_mask,
      KOKKOS_LAMBDA(Coord x, Coord y, Conserved& value,
                    std::size_t /*linear_index*/) {
        const Coord xc = subsetix::csr::detail::floor_div2(x);
        const Coord yc = subsetix::csr::detail::floor_div2(y);
        value = coarse_acc.value_at(xc, yc);
      });
}

[[maybe_unused]] void copy_overlap(Field2DDevice<Conserved>& fine_dst,
                                   Field2DDevice<Conserved>& fine_src,
                                   const IntervalSet2DDevice& overlap,
                                   CsrSetAlgebraContext& ctx) {
  if (overlap.num_rows == 0 || overlap.num_intervals == 0) {
    return;
  }
  auto sub_dst = make_subview(fine_dst, overlap, "mach2_overlap_dst");
  auto sub_src = make_subview(fine_src, overlap, "mach2_overlap_src");
  copy_subview_device(sub_dst, sub_src, ctx);
}


void prolong_guard_from_coarse(Field2DDevice<Conserved>& fine_field,
                               const IntervalSet2DDevice& guard,
                               const FieldReadAccessor<Conserved>& coarse_acc) {
  if (guard.num_rows == 0 || guard.num_intervals == 0) {
    return;
  }

  apply_on_set_device(fine_field, guard, KOKKOS_LAMBDA(
                                             Coord x,
                                             Coord y,
                                             Conserved& value,
                                             std::size_t /*linear_index*/) {
    const Coord xc = subsetix::csr::detail::floor_div2(x);
    const Coord yc = subsetix::csr::detail::floor_div2(y);
    value = coarse_acc.value_at(xc, yc);
  });
}

void restrict_fine_to_coarse(Field2DDevice<Conserved>& coarse_field,
                             const Field2DDevice<Conserved>& fine_field,
                             const IntervalSet2DDevice& coarse_region) {
  if (coarse_region.num_rows == 0 || coarse_region.num_intervals == 0) {
    return;
  }

  const auto mapping =
      build_mask_field_mapping(coarse_field, coarse_region);
  auto interval_to_row = mapping.interval_to_row;
  auto interval_to_field_interval = mapping.interval_to_field_interval;
  auto row_map = mapping.row_map;

  auto mask_row_keys = coarse_region.row_keys;
  auto mask_intervals = coarse_region.intervals;

  auto coarse_intervals = coarse_field.geometry.intervals;
  auto coarse_offsets = coarse_field.geometry.cell_offsets;
  auto coarse_values = coarse_field.values;

  const auto fine_acc = make_accessor(fine_field);

  Kokkos::parallel_for(
      "mach2_cylinder_restrict",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
          0, static_cast<int>(coarse_region.num_intervals)),
      KOKKOS_LAMBDA(const int interval_idx) {
        const int row_idx = interval_to_row(interval_idx);
        const int field_row = row_map(row_idx);
        if (row_idx < 0 || field_row < 0) {
          return;
        }

        const int coarse_interval_idx =
            interval_to_field_interval(interval_idx);
        if (coarse_interval_idx < 0) {
          return;
        }

        const Interval mask_iv = mask_intervals(interval_idx);
        const Interval civ = coarse_intervals(coarse_interval_idx);
        const std::size_t base_offset = coarse_offsets(coarse_interval_idx);
        const Coord base_begin = civ.begin;
        const Coord y = mask_row_keys(row_idx).y;
        const Coord y_f0 = static_cast<Coord>(2 * y);
        const Coord y_f1 = static_cast<Coord>(2 * y + 1);

        for (Coord x = mask_iv.begin; x < mask_iv.end; ++x) {
          const std::size_t c_idx =
              base_offset + static_cast<std::size_t>(x - base_begin);
          const Coord x_f0 = static_cast<Coord>(2 * x);
          const Coord x_f1 = static_cast<Coord>(2 * x + 1);

          const Conserved v00 = fine_acc.value_at(x_f0, y_f0);
          const Conserved v01 = fine_acc.value_at(x_f1, y_f0);
          const Conserved v10 = fine_acc.value_at(x_f0, y_f1);
          const Conserved v11 = fine_acc.value_at(x_f1, y_f1);

          Conserved avg;
          avg.rho = 0.25 * (v00.rho + v01.rho + v10.rho + v11.rho);
          avg.rhou = 0.25 * (v00.rhou + v01.rhou + v10.rhou + v11.rhou);
          avg.rhov = 0.25 * (v00.rhov + v01.rhov + v10.rhov + v11.rhov);
          avg.E = 0.25 * (v00.E + v01.E + v10.E + v11.E);

          coarse_values(c_idx) = avg;
        }
      });
  Kokkos::DefaultExecutionSpace().fence();
}

struct CoarseEulerStencil {
  FieldReadAccessor<Conserved> acc;
  Box2D domain;
  Conserved inflow;
  Real gamma;
  Real dt;
  Real dx;
  Real dy;
  bool no_slip = false;

  KOKKOS_INLINE_FUNCTION
  Conserved operator()(Coord x, Coord y,
                       std::size_t linear_index,
                       int field_interval_idx,
                       const subsetix::csr::detail::FieldStencilContext<Conserved>& ctx) const {
    (void)field_interval_idx;
    const Conserved center = ctx.center(linear_index);
    const Primitive q_center = cons_to_prim(center, gamma);

    const Conserved left_state =
        sample_neighbor(center, x, y, -1, 0, acc, domain,
                        inflow, gamma, no_slip);
    const Conserved right_state =
        sample_neighbor(center, x, y, +1, 0, acc, domain,
                        inflow, gamma, no_slip);
    const Conserved down_state =
        sample_neighbor(center, x, y, 0, -1, acc, domain,
                        inflow, gamma, no_slip);
    const Conserved up_state =
        sample_neighbor(center, x, y, 0, +1, acc, domain,
                        inflow, gamma, no_slip);

    const Primitive q_left = cons_to_prim(left_state, gamma);
    const Primitive q_right = cons_to_prim(right_state, gamma);
    const Primitive q_down = cons_to_prim(down_state, gamma);
    const Primitive q_up = cons_to_prim(up_state, gamma);

    const Conserved flux_w =
        rusanov_flux_x(left_state, center, q_left,
                       q_center, gamma);
    const Conserved flux_e =
        rusanov_flux_x(center, right_state, q_center,
                       q_right, gamma);
    const Conserved flux_s =
        rusanov_flux_y(down_state, center, q_down,
                       q_center, gamma);
    const Conserved flux_n =
        rusanov_flux_y(center, up_state, q_center,
                       q_up, gamma);

    Conserved updated;
    updated.rho =
        center.rho -
        (dt / dx) * (flux_e.rho - flux_w.rho) -
        (dt / dy) * (flux_n.rho - flux_s.rho);
    updated.rhou =
        center.rhou -
        (dt / dx) * (flux_e.rhou - flux_w.rhou) -
        (dt / dy) * (flux_n.rhou - flux_s.rhou);
    updated.rhov =
        center.rhov -
        (dt / dx) * (flux_e.rhov - flux_w.rhov) -
        (dt / dy) * (flux_n.rhov - flux_s.rhov);
    updated.E =
        center.E -
        (dt / dx) * (flux_e.E - flux_w.E) -
        (dt / dy) * (flux_n.E - flux_s.E);
    return updated;
  }
};

struct FineEulerStencil {
  FieldReadAccessor<Conserved> acc_fine;
  FieldReadAccessor<Conserved> acc_coarse;
  Box2D fine_domain;
  Conserved inflow;
  Real gamma;
  Real dt;
  Real dx;
  Real dy;
  bool no_slip = false;

  KOKKOS_INLINE_FUNCTION
  Conserved operator()(Coord x, Coord y,
                       std::size_t linear_index,
                       int field_interval_idx,
                       const subsetix::csr::detail::FieldStencilContext<Conserved>& ctx) const {
    (void)field_interval_idx;
    const Conserved center = ctx.center(linear_index);
    const Primitive q_center = cons_to_prim(center, gamma);

    const Conserved left_state =
        sample_neighbor_with_coarse(center, x, y, -1, 0,
                                    acc_fine, &acc_coarse,
                                    fine_domain, inflow,
                                    gamma, no_slip);
    const Conserved right_state =
        sample_neighbor_with_coarse(center, x, y, +1, 0,
                                    acc_fine, &acc_coarse,
                                    fine_domain, inflow,
                                    gamma, no_slip);
    const Conserved down_state =
        sample_neighbor_with_coarse(center, x, y, 0, -1,
                                    acc_fine, &acc_coarse,
                                    fine_domain, inflow,
                                    gamma, no_slip);
    const Conserved up_state =
        sample_neighbor_with_coarse(center, x, y, 0, +1,
                                    acc_fine, &acc_coarse,
                                    fine_domain, inflow,
                                    gamma, no_slip);

    const Primitive q_left = cons_to_prim(left_state, gamma);
    const Primitive q_right = cons_to_prim(right_state, gamma);
    const Primitive q_down = cons_to_prim(down_state, gamma);
    const Primitive q_up = cons_to_prim(up_state, gamma);

    const Conserved flux_w =
        rusanov_flux_x(left_state, center, q_left,
                       q_center, gamma);
    const Conserved flux_e =
        rusanov_flux_x(center, right_state, q_center,
                       q_right, gamma);
    const Conserved flux_s =
        rusanov_flux_y(down_state, center, q_down,
                       q_center, gamma);
    const Conserved flux_n =
        rusanov_flux_y(center, up_state, q_center,
                       q_up, gamma);

    Conserved updated;
    updated.rho =
        center.rho -
        (dt / dx) * (flux_e.rho - flux_w.rho) -
        (dt / dy) * (flux_n.rho - flux_s.rho);
    updated.rhou =
        center.rhou -
        (dt / dx) * (flux_e.rhou - flux_w.rhou) -
        (dt / dy) * (flux_n.rhou - flux_s.rhou);
    updated.rhov =
        center.rhov -
        (dt / dx) * (flux_e.rhov - flux_w.rhov) -
        (dt / dy) * (flux_n.rhov - flux_s.rhov);
    updated.E =
        center.E -
        (dt / dx) * (flux_e.E - flux_w.E) -
        (dt / dy) * (flux_n.E - flux_s.E);
    return updated;
  }
};

void compute_diagnostics(const Field2DDevice<Conserved>& U,
                         Field2DDevice<Real>& density,
                         Field2DDevice<Real>& pressure,
                         Field2DDevice<Real>& mach,
                         Real gamma) {
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  auto values = U.values;
  auto rho_out = density.values;
  auto p_out = pressure.values;
  auto mach_out = mach.values;

  Kokkos::parallel_for(
      "mach2_cylinder_diagnostics",
      Kokkos::RangePolicy<ExecSpace>(
          0, static_cast<int>(U.size())),
      KOKKOS_LAMBDA(const int idx) {
        const Conserved s = values(idx);
        const Primitive q = cons_to_prim(s, gamma);
        const Real a = sound_speed(q, gamma);
        const Real vel = std::sqrt(q.u * q.u + q.v * q.v);
        rho_out(idx) = q.rho;
        p_out(idx) = q.p;
        mach_out(idx) = (a > static_cast<Real>(1e-12)) ? (vel / a) : static_cast<Real>(0.0);
      });
}

Real compute_total_mass(const Field2DDevice<Conserved>& U) {
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  Real total = static_cast<Real>(0.0);
  Kokkos::parallel_reduce(
      "mach2_cylinder_mass",
      Kokkos::RangePolicy<ExecSpace>(
          0, static_cast<int>(U.size())),
      KOKKOS_LAMBDA(const int idx, Real& sum) {
        sum += U.values(idx).rho;
      },
      total);
  return total;
}

RunConfig parse_args(int argc, char* argv[]) {
  RunConfig cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto read_int = [&](int& out) {
      if (i + 1 < argc) {
        out = std::stoi(argv[++i]);
      }
    };
    auto read_double = [&](Real& out) {
      if (i + 1 < argc) {
        out = static_cast<Real>(std::stod(argv[++i]));
      }
    };

    if (arg == "--nx") read_int(cfg.nx);
    else if (arg == "--ny") read_int(cfg.ny);
    else if (arg == "--cx") read_int(cfg.cx);
    else if (arg == "--cy") read_int(cfg.cy);
    else if (arg == "--radius") read_int(cfg.radius);
    else if (arg == "--mach-inlet") read_double(cfg.mach_inlet);
    else if (arg == "--rho") read_double(cfg.rho);
    else if (arg == "--p") read_double(cfg.p);
    else if (arg == "--gamma") read_double(cfg.gamma);
    else if (arg == "--cfl") read_double(cfg.cfl);
    else if (arg == "--t-final") read_double(cfg.t_final);
    else if (arg == "--max-steps") read_int(cfg.max_steps);
    else if (arg == "--output-stride") read_int(cfg.output_stride);
    else if (arg == "--no-slip") cfg.no_slip = true;
    else if (arg == "--no-output") cfg.enable_output = false;
    else if (arg == "--no-amr") cfg.enable_amr = false;
    else if (arg == "--amr") cfg.enable_amr = true;
    else if (arg == "--amr-fraction") read_double(cfg.amr_fraction);
    else if (arg == "--amr-guard") read_int(cfg.amr_guard);
    else if (arg == "--amr-levels") read_int(cfg.max_amr_levels);
    else if (arg == "--amr-remesh-stride") read_int(cfg.amr_remesh_stride);
    else if (arg == "--pbm" && i + 1 < argc) {
      cfg.pbm_path = argv[++i];
    }
  }
  if (cfg.cx < 0) {
    cfg.cx = cfg.nx / 4;
  }
  if (cfg.cy < 0) {
    cfg.cy = cfg.ny / 2;
  }
  cfg.amr_fraction = std::clamp(cfg.amr_fraction,
                                static_cast<Real>(0.05),
                                static_cast<Real>(0.95));
  if (cfg.amr_guard < 1) {
    cfg.amr_guard = 1;
  }
  if (cfg.amr_remesh_stride < 0) {
    cfg.amr_remesh_stride = 0;
  }
  cfg.max_amr_levels = std::clamp(cfg.max_amr_levels, 1, MAX_AMR_LEVELS);
  return cfg;
}

std::string vtk_filename(const std::filesystem::path& dir,
                         int step,
                         std::string_view suffix) {
  std::ostringstream oss;
  oss << "step_" << std::setw(5) << std::setfill('0') << step << "_"
      << suffix << ".vtk";
  return subsetix_examples::output_file(dir, oss.str());
}

Kokkos::View<std::uint8_t**, subsetix::csr::HostMemorySpace>
load_pbm(const std::filesystem::path& path) {
  using subsetix::csr::HostMemorySpace;
  std::ifstream in(path);
  if (!in) {
    return {};
  }

  std::string magic;
  in >> magic;
  if (magic != "P1") {
    return {};
  }

  auto skip_comments = [&]() {
    while (true) {
      in >> std::ws;
      if (in.peek() == '#') {
        std::string tmp;
        std::getline(in, tmp);
      } else {
        break;
      }
    }
  };

  skip_comments();

  std::size_t width = 0;
  std::size_t height = 0;
  in >> width >> height;
  if (width == 0 || height == 0) {
    return {};
  }

  Kokkos::View<std::uint8_t**, HostMemorySpace> mask(
      "pbm_mask_host", height, width);

  for (std::size_t y = 0; y < height; ++y) {
    for (std::size_t x = 0; x < width; ++x) {
      skip_comments();
      int bit = 0;
      in >> bit;
      mask(y, x) = static_cast<std::uint8_t>(bit ? 1 : 0);
    }
  }

  return mask;
}

IntervalSet2DDevice make_pbm_obstacle(const RunConfig& cfg,
                                      const Box2D& domain,
                                      bool& ok) {
  ok = false;
  IntervalSet2DDevice obstacle;
  auto h_mask = load_pbm(cfg.pbm_path);
  if (h_mask.extent(0) == 0 || h_mask.extent(1) == 0) {
    return obstacle;
  }

  // Flip Y so PBM top row maps to the highest Y visually.
  Kokkos::View<std::uint8_t**,
               subsetix::csr::HostMemorySpace> h_mask_flipped(
      "pbm_mask_flipped", h_mask.extent(0), h_mask.extent(1));
  for (std::size_t y = 0; y < h_mask.extent(0); ++y) {
    const std::size_t src_y = h_mask.extent(0) - 1 - y;
    for (std::size_t x = 0; x < h_mask.extent(1); ++x) {
      h_mask_flipped(y, x) = h_mask(src_y, x);
    }
  }

  const std::size_t width = h_mask_flipped.extent(1);
  const std::size_t height = h_mask_flipped.extent(0);

  if (width > static_cast<std::size_t>(domain.x_max - domain.x_min) ||
      height > static_cast<std::size_t>(domain.y_max - domain.y_min)) {
    return obstacle;
  }

  const Coord x_min = domain.x_min +
                      static_cast<Coord>((static_cast<long long>(domain.x_max - domain.x_min) -
                                          static_cast<long long>(width)) /
                                         2);
  const Coord y_min = domain.y_min +
                      static_cast<Coord>((static_cast<long long>(domain.y_max - domain.y_min) -
                                          static_cast<long long>(height)) /
                                         2);

  auto d_mask = Kokkos::create_mirror_view_and_copy(
      subsetix::csr::DeviceMemorySpace{}, h_mask_flipped);
  obstacle = make_bitmap_device(d_mask, x_min, y_min, 1);
  ok = true;
  return obstacle;
}

Conserved build_inflow_state(const RunConfig& cfg) {
  Primitive q;
  q.rho = cfg.rho;
  q.p = cfg.p;
  const Real a = std::sqrt(cfg.gamma * q.p / q.rho);
  q.u = cfg.mach_inlet * a;
  q.v = 0.0;
  return prim_to_cons(q, cfg.gamma);
}


template <int MaxLevels>
void write_multilevel_outputs(
    const std::array<IntervalSet2DDevice, MaxLevels>& geoms,
    const std::array<Field2DDevice<Real>, MaxLevels>& density,
    const std::array<Field2DDevice<Real>, MaxLevels>& pressure,
    const std::array<Field2DDevice<Real>, MaxLevels>& mach,
    const std::array<Field2DDevice<Conserved>, MaxLevels>& U_active,
    const std::array<bool, MaxLevels>& has_level,
    int max_active_level,
    Real gamma,
    const std::filesystem::path& out_dir,
    int step) {
  MultilevelGeoDevice geo;
  geo.origin_x = 0.0;
  geo.origin_y = 0.0;
  geo.root_dx = 1.0;
  geo.root_dy = 1.0;
  geo.num_active_levels = max_active_level + 1;

  MultilevelFieldDevice<Real> f_density;
  MultilevelFieldDevice<Real> f_pressure;
  MultilevelFieldDevice<Real> f_mach;
  f_density.num_active_levels = geo.num_active_levels;
  f_pressure.num_active_levels = geo.num_active_levels;
  f_mach.num_active_levels = geo.num_active_levels;

  for (int lvl = 0; lvl <= max_active_level; ++lvl) {
    if (!has_level[lvl]) {
      continue;
    }
    geo.levels[lvl] = geoms[lvl];
    f_density.levels[lvl] = density[lvl];
    f_pressure.levels[lvl] = pressure[lvl];
    f_mach.levels[lvl] = mach[lvl];
    compute_diagnostics(U_active[lvl],
                        f_density.levels[lvl],
                        f_pressure.levels[lvl],
                        f_mach.levels[lvl],
                        gamma);
  }

  const auto host_geo = subsetix::deep_copy_to_host(geo);
  const auto host_rho = subsetix::deep_copy_to_host(f_density);
  subsetix::vtk::write_multilevel_field_vtk(
      host_rho, host_geo, vtk_filename(out_dir, step, "density"), "rho", true);
}

} // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  {
    const RunConfig cfg = parse_args(argc, argv);
    const int LEVEL_LIMIT = std::max(
        0, std::min(cfg.max_amr_levels, MAX_AMR_LEVELS) - 1);
    std::filesystem::path output_dir;
    if (cfg.enable_output) {
      output_dir = subsetix_examples::make_example_output_dir("mach2_cylinder",
                                                              argc, argv);
    }

    const Box2D domain{
        0, static_cast<Coord>(cfg.nx),
        0, static_cast<Coord>(cfg.ny)};
    Disk2D obstacle;
    obstacle.cx = static_cast<Coord>(cfg.cx);
    obstacle.cy = static_cast<Coord>(cfg.cy);
    obstacle.radius = static_cast<Coord>(cfg.radius);

    auto domain_dev = make_box_device(domain);
    IntervalSet2DDevice obstacle_dev;
    bool obstacle_from_pbm = false;
    if (!cfg.pbm_path.empty()) {
      obstacle_dev = make_pbm_obstacle(cfg, domain, obstacle_from_pbm);
      if (!obstacle_from_pbm) {
        std::cout << "Warning: failed to load PBM mask '" << cfg.pbm_path
                  << "', falling back to disk obstacle.\n";
      }
    }
    if (!obstacle_from_pbm) {
      obstacle_dev = make_disk_device(obstacle);
    }

    CsrSetAlgebraContext ctx;
    auto fluid_dev = subsetix::csr::allocate_interval_set_device(
        domain_dev.num_rows,
        domain_dev.num_intervals + obstacle_dev.num_intervals);
    set_difference_device(domain_dev, obstacle_dev, fluid_dev, ctx);
    subsetix::csr::compute_cell_offsets_device(fluid_dev);

    std::array<IntervalSet2DDevice, MAX_AMR_LEVELS> fluid_full;
    std::array<IntervalSet2DDevice, MAX_AMR_LEVELS> active_set;
    std::array<IntervalSet2DDevice, MAX_AMR_LEVELS> with_guard_set;
    std::array<IntervalSet2DDevice, MAX_AMR_LEVELS> guard_set;
    std::array<IntervalSet2DDevice, MAX_AMR_LEVELS> projection_down;
    std::array<IntervalSet2DDevice, MAX_AMR_LEVELS> coarse_masks;
    std::array<IntervalSet2DDevice, MAX_AMR_LEVELS> field_geom;
    std::array<IntervalSet2DDevice, MAX_AMR_LEVELS> ghost_mask;
    std::array<Box2D, MAX_AMR_LEVELS> domains;
    std::array<bool, MAX_AMR_LEVELS> has_level;
    has_level.fill(false);

    std::array<Field2DDevice<Conserved>, MAX_AMR_LEVELS> U_levels;
    std::array<Field2DDevice<Conserved>, MAX_AMR_LEVELS> U_next_levels;
    std::array<Field2DDevice<Conserved>, MAX_AMR_LEVELS> U_active_levels;
    std::array<Field2DDevice<Real>, MAX_AMR_LEVELS> density_levels;
    std::array<Field2DDevice<Real>, MAX_AMR_LEVELS> pressure_levels;
    std::array<Field2DDevice<Real>, MAX_AMR_LEVELS> mach_levels;

    IntervalSet2DDevice field0;
    expand_device(fluid_dev,
                  static_cast<Coord>(1),
                  static_cast<Coord>(1),
                  field0, ctx);
    subsetix::csr::compute_cell_offsets_device(field0);
    field_geom[0] = field0;
    auto ghost0 = subsetix::csr::allocate_interval_set_device(
        field0.num_rows,
        field0.num_intervals + fluid_dev.num_intervals);
    set_difference_device(field0, fluid_dev, ghost0, ctx);
    subsetix::csr::compute_cell_offsets_device(ghost0);
    ghost_mask[0] = ghost0;

    const Conserved inflow = build_inflow_state(cfg);

    {
      Field2DDevice<Conserved> U(field_geom[0], "mach2_state");
      Field2DDevice<Conserved> U_next(field_geom[0], "mach2_state_next");
      fill_on_set_device(U, field_geom[0], inflow);
      fill_on_set_device(U_next, field_geom[0], inflow);
      U_levels[0] = U;
      U_next_levels[0] = U_next;
      fill_ghost_cells(U_levels[0], ghost_mask[0], fluid_dev,
                       domain, inflow, cfg.gamma, cfg.no_slip);
      fill_ghost_cells(U_next_levels[0], ghost_mask[0], fluid_dev,
                       domain, inflow, cfg.gamma, cfg.no_slip);
    }

    fluid_full[0] = fluid_dev;
    active_set[0] = fluid_dev;
    with_guard_set[0] = fluid_dev;
    guard_set[0] = IntervalSet2DDevice();
    projection_down[0] = IntervalSet2DDevice();
    domains[0] = domain;
    has_level[0] = true;
    coarse_masks[0] = IntervalSet2DDevice();
    density_levels[0] = Field2DDevice<Real>(fluid_dev, "mach2_density");
    pressure_levels[0] = Field2DDevice<Real>(fluid_dev, "mach2_pressure");
    mach_levels[0] = Field2DDevice<Real>(fluid_dev, "mach2_mach");
    U_active_levels[0] = Field2DDevice<Conserved>(fluid_dev, "mach2_state_active");

    auto build_level = [&](int lvl,
                           const IntervalSet2DDevice& parent_full,
                           const IntervalSet2DDevice& parent_active,
                           const Field2DDevice<Conserved>& parent_U,
                           const Box2D& parent_domain,
                           const IntervalSet2DDevice* prev_active = nullptr,
                           Field2DDevice<Conserved>* prev_U = nullptr,
                           RemeshTiming* timers = nullptr) {
      if (!cfg.enable_amr) {
        return;
      }
      if (lvl <= 0 || lvl >= MAX_AMR_LEVELS) {
        return;
      }
      IntervalSet2DDevice mask =
          build_refine_mask(parent_U, parent_full, parent_domain, cfg, ctx, timers);
      const auto t_constrain_begin = Clock::now();
      mask = constrain_mask_to_parent_interior(
          mask, parent_full, parent_active,
          static_cast<Coord>(std::max(1, cfg.amr_guard)), ctx);
      const auto t_constrain_end = Clock::now();
      if (timers) {
        timers->mask_constrain += std::chrono::duration<double, std::milli>(
                                      t_constrain_end - t_constrain_begin)
                                      .count();
      }
      const auto t_geom_begin = Clock::now();
      const AmrLayout amr = build_fine_geometry(
          parent_full, mask, static_cast<Coord>(cfg.amr_guard), parent_domain, ctx);
      const auto t_geom_end = Clock::now();
      if (timers) {
        timers->geom += std::chrono::duration<double, std::milli>(
                            t_geom_end - t_geom_begin)
                            .count();
      }

      fluid_full[lvl] = amr.fine_full;
      active_set[lvl] = ensure_subset(amr.fine_active, amr.fine_with_guard, ctx);
      with_guard_set[lvl] = amr.fine_with_guard;
      guard_set[lvl] = amr.fine_guard;
      projection_down[lvl] = ensure_subset(
          amr.projection_fine_on_coarse, parent_active, ctx);
      domains[lvl] = amr.fine_domain;
      coarse_masks[lvl] = mask;
      has_level[lvl] = amr.has_fine;
      if (!has_level[lvl]) {
        return;
      }

      IntervalSet2DDevice field_lvl;
      expand_device(with_guard_set[lvl],
                    static_cast<Coord>(1),
                    static_cast<Coord>(1),
                    field_lvl, ctx);
      subsetix::csr::compute_cell_offsets_device(field_lvl);
      field_geom[lvl] = field_lvl;
      auto ghost_lvl = subsetix::csr::allocate_interval_set_device(
          field_lvl.num_rows,
          field_lvl.num_intervals + with_guard_set[lvl].num_intervals);
      set_difference_device(field_lvl, with_guard_set[lvl], ghost_lvl, ctx);
      subsetix::csr::compute_cell_offsets_device(ghost_lvl);
      ghost_mask[lvl] = ghost_lvl;

      U_levels[lvl] = Field2DDevice<Conserved>(field_geom[lvl], "mach2_fine_lvl");
      U_next_levels[lvl] = Field2DDevice<Conserved>(field_geom[lvl],
                                                    "mach2_fine_lvl_next");
      U_active_levels[lvl] = Field2DDevice<Conserved>(active_set[lvl],
                                                      "mach2_fine_lvl_active");
      density_levels[lvl] = Field2DDevice<Real>(active_set[lvl], "mach2_density_lvl");
      pressure_levels[lvl] = Field2DDevice<Real>(active_set[lvl], "mach2_pressure_lvl");
      mach_levels[lvl] = Field2DDevice<Real>(active_set[lvl], "mach2_mach_lvl");

      const auto t_prolong_begin = Clock::now();
      prolong_full(U_levels[lvl], with_guard_set[lvl], parent_U, ctx);
      prolong_full(U_next_levels[lvl], with_guard_set[lvl], parent_U, ctx);
      Kokkos::DefaultExecutionSpace().fence();
      const auto t_prolong_end = Clock::now();
      if (timers) {
        timers->prolong += std::chrono::duration<double, std::milli>(
                               t_prolong_end - t_prolong_begin)
                               .count();
      }

      if (prev_active && prev_U) {
        const auto t_overlap_begin = Clock::now();
        const std::size_t overlap_rows_cap =
            std::min(prev_active->num_rows, active_set[lvl].num_rows);
        const std::size_t overlap_intervals_cap =
            prev_active->num_intervals + active_set[lvl].num_intervals;
        if (overlap_rows_cap > 0 && overlap_intervals_cap > 0) {
          auto overlap = subsetix::csr::allocate_interval_set_device(
              overlap_rows_cap, overlap_intervals_cap);
          set_intersection_device(*prev_active, active_set[lvl], overlap, ctx);
          copy_overlap(U_levels[lvl], *prev_U, overlap, ctx);
        }
        Kokkos::DefaultExecutionSpace().fence();
        const auto t_overlap_end = Clock::now();
        if (timers) {
          timers->overlap += std::chrono::duration<double, std::milli>(
                                 t_overlap_end - t_overlap_begin)
                                 .count();
        }
      }

      fill_ghost_cells(U_levels[lvl], ghost_mask[lvl], with_guard_set[lvl],
                       domains[lvl], inflow, cfg.gamma, cfg.no_slip);
      fill_ghost_cells(U_next_levels[lvl], ghost_mask[lvl], with_guard_set[lvl],
                       domains[lvl], inflow, cfg.gamma, cfg.no_slip);
    };

    // Initial hierarchy build up to MAX_AMR_LEVELS-1
    for (int lvl = 1; lvl <= LEVEL_LIMIT; ++lvl) {
      if (!has_level[lvl - 1]) {
        break;
      }
      build_level(lvl,
                  fluid_full[lvl - 1],
                  active_set[lvl - 1],
                  U_levels[lvl - 1],
                  domains[lvl - 1]);
      if (!has_level[lvl]) {
        break;
      }
    }

    if (cfg.enable_output) {
      const IntervalSet2DHost fluid_host =
          subsetix::csr::build_host_from_device(fluid_dev);
      const IntervalSet2DHost obstacle_host =
          subsetix::csr::build_host_from_device(obstacle_dev);
      write_legacy_quads(fluid_host,
                         subsetix_examples::output_file(output_dir, "fluid_geometry.vtk"));
      write_legacy_quads(obstacle_host,
                         subsetix_examples::output_file(output_dir, "obstacle_geometry.vtk"));
      for (int lvl = 1; lvl <= LEVEL_LIMIT; ++lvl) {
        if (!has_level[lvl]) {
          break;
        }
        if (coarse_masks[lvl].num_intervals > 0) {
          const IntervalSet2DHost mask_host =
              subsetix::csr::build_host_from_device(coarse_masks[lvl]);
          write_legacy_quads(mask_host,
                             subsetix_examples::output_file(
                                 output_dir, "refine_mask_lvl" + std::to_string(lvl) + ".vtk"));
        }
        const IntervalSet2DHost fine_host =
            subsetix::csr::build_host_from_device(active_set[lvl]);
        write_legacy_quads(fine_host,
                           subsetix_examples::output_file(
                               output_dir, "fine_geometry_lvl" + std::to_string(lvl) + ".vtk"));
      }
    } // initial outputs
    Real t = static_cast<Real>(0.0);
    int step = 0;
    std::array<Real, MAX_AMR_LEVELS> dx_levels;
    std::array<Real, MAX_AMR_LEVELS> dy_levels;
    dx_levels[0] = static_cast<Real>(1.0);
    dy_levels[0] = static_cast<Real>(1.0);
    for (int lvl = 1; lvl < MAX_AMR_LEVELS; ++lvl) {
      dx_levels[lvl] = static_cast<Real>(0.5) * dx_levels[lvl - 1];
      dy_levels[lvl] = static_cast<Real>(0.5) * dy_levels[lvl - 1];
    }

  std::cout << "Mach 2 cylinder setup: "
            << "nx=" << cfg.nx << " ny=" << cfg.ny
            << " cx=" << cfg.cx << " cy=" << cfg.cy
            << " r=" << cfg.radius
            << " pbm=" << (obstacle_from_pbm ? cfg.pbm_path : "(disk)")
            << " no-slip=" << (cfg.no_slip ? "yes" : "no")
            << " amr=" << (cfg.enable_amr ? "yes" : "no")
            << " remesh_stride=" << cfg.amr_remesh_stride
            << " output_dir=" << (cfg.enable_output ? output_dir.string() : "(disabled)") << "\n";

    Real total_mass0 = compute_total_mass(U_levels[0]);

  auto max_active_level = [&]() {
    int finest = 0;
    for (int lvl = 0; lvl < MAX_AMR_LEVELS; ++lvl) {
      if (has_level[lvl]) {
        finest = lvl;
      }
    }
    return finest;
  };

  while ((t < cfg.t_final) && (step < cfg.max_steps)) {
    const int finest_level = max_active_level();
    std::array<FieldReadAccessor<Conserved>, MAX_AMR_LEVELS> accessors;
    for (int lvl = 0; lvl <= finest_level; ++lvl) {
      if (has_level[lvl]) {
        accessors[lvl] = make_accessor(U_levels[lvl]);
      }
    }

    Real dt = compute_dt(U_levels[0], cfg.gamma, cfg.cfl,
                         dx_levels[0], dy_levels[0]);

    const auto t_step_begin = Clock::now();
    double time_prolong_ms = 0.0;
    double time_fine_ms = 0.0;
    double time_coarse_ms = 0.0;
    double time_restrict_ms = 0.0;
    double time_remesh_ms = 0.0;
    double time_remesh_masks_ms = 0.0;
    double time_remesh_mask_indicator_ms = 0.0;
    double time_remesh_mask_reduce_ms = 0.0;
    double time_remesh_mask_expand_ms = 0.0;
    double time_remesh_mask_constrain_ms = 0.0;
    double time_remesh_geom_ms = 0.0;
    double time_remesh_prolong_ms = 0.0;
    double time_remesh_overlap_ms = 0.0;
    double time_output_ms = 0.0;

    for (int lvl = 1; lvl <= finest_level; ++lvl) {
      if (!has_level[lvl]) {
        continue;
      }
      const auto t0 = Clock::now();
      prolong_guard_from_coarse(U_levels[lvl], guard_set[lvl], accessors[lvl - 1]);
      const auto t1 = Clock::now();
      time_prolong_ms +=
          std::chrono::duration<double, std::milli>(t1 - t0).count();
      IntervalSet2DDevice active_clipped =
          ensure_subset(active_set[lvl], U_levels[lvl].geometry, ctx);
      active_set[lvl] = active_clipped;
      const Real dt_lvl = compute_dt_on_set(
          U_levels[lvl], active_set[lvl], cfg.gamma, cfg.cfl,
          dx_levels[lvl], dy_levels[lvl]);
      dt = std::min(dt, dt_lvl);
    }

    if (t + dt > cfg.t_final) {
      dt = cfg.t_final - t;
    }

    for (int lvl = 1; lvl <= finest_level; ++lvl) {
      if (!has_level[lvl]) {
        continue;
      }
      fill_ghost_cells(U_levels[lvl], ghost_mask[lvl], with_guard_set[lvl],
                       domains[lvl], inflow, cfg.gamma, cfg.no_slip);
    }
    fill_ghost_cells(U_levels[0], ghost_mask[0], fluid_dev,
                     domain, inflow, cfg.gamma, cfg.no_slip);

    for (int lvl = finest_level; lvl >= 1; --lvl) {
      if (!has_level[lvl]) {
        continue;
      }
      const auto t0 = Clock::now();
      FineEulerStencil fine_stencil{accessors[lvl], accessors[lvl - 1],
                                    domains[lvl], inflow, cfg.gamma, dt,
                                    dx_levels[lvl], dy_levels[lvl], cfg.no_slip};
      apply_stencil_on_set_device(U_next_levels[lvl], U_levels[lvl],
                                  active_set[lvl], fine_stencil);
      const auto t1 = Clock::now();
      time_fine_ms +=
          std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    const auto t0_coarse = Clock::now();
    CoarseEulerStencil coarse_stencil{accessors[0], domain, inflow,
                                      cfg.gamma, dt, dx_levels[0], dy_levels[0],
                                      cfg.no_slip};
    apply_stencil_on_set_device(U_next_levels[0], U_levels[0],
                                fluid_dev, coarse_stencil);
    const auto t1_coarse = Clock::now();
    time_coarse_ms +=
        std::chrono::duration<double, std::milli>(t1_coarse - t0_coarse).count();

    for (int lvl = 0; lvl <= finest_level; ++lvl) {
      if (has_level[lvl]) {
        std::swap(U_levels[lvl].values, U_next_levels[lvl].values);
      }
    }

    for (int lvl = finest_level; lvl >= 1; --lvl) {
      if (!has_level[lvl]) {
        continue;
      }
      const auto t0 = Clock::now();
      projection_down[lvl] = ensure_subset(
          projection_down[lvl], U_levels[lvl - 1].geometry, ctx);
      restrict_fine_to_coarse(U_levels[lvl - 1], U_levels[lvl],
                              projection_down[lvl]);
      const auto t1 = Clock::now();
      time_restrict_ms +=
          std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    const bool do_remesh = cfg.enable_amr &&
                           cfg.amr_remesh_stride > 0 &&
                           (step % cfg.amr_remesh_stride == 0);
    if (do_remesh) {
      RemeshTiming remesh_timers;
      const auto t_remesh_begin = Clock::now();
      auto prev_active = active_set;
      auto prev_U = U_levels;
      for (int lvl = 1; lvl <= LEVEL_LIMIT; ++lvl) {
        if (!has_level[lvl - 1]) {
          has_level[lvl] = false;
          continue;
        }
        build_level(lvl,
                    fluid_full[lvl - 1],
                    active_set[lvl - 1],
                    U_levels[lvl - 1],
                    domains[lvl - 1],
                    has_level[lvl] ? &prev_active[lvl] : nullptr,
                    has_level[lvl] ? &prev_U[lvl] : nullptr,
                    &remesh_timers);
      }
      const auto t_remesh_end = Clock::now();
      time_remesh_ms += std::chrono::duration<double, std::milli>(
                            t_remesh_end - t_remesh_begin)
                            .count();
      time_remesh_masks_ms += remesh_timers.masks;
      time_remesh_mask_indicator_ms += remesh_timers.mask_indicator;
      time_remesh_mask_reduce_ms += remesh_timers.mask_reduce;
      time_remesh_mask_expand_ms += remesh_timers.mask_expand;
      time_remesh_mask_constrain_ms += remesh_timers.mask_constrain;
      time_remesh_geom_ms += remesh_timers.geom;
      time_remesh_prolong_ms += remesh_timers.prolong;
      time_remesh_overlap_ms += remesh_timers.overlap;
    }

    t += dt;
    ++step;

    if (step % cfg.output_stride == 0 || step == cfg.max_steps ||
        t >= cfg.t_final - 1e-12) {
      if (cfg.enable_output) {
        const auto t_out_begin = Clock::now();
        const int active_finest = max_active_level();
        {
          auto src0 = make_subview(U_levels[0], active_set[0],
                                   "coarse_active_src");
          auto dst0 = make_subview(U_active_levels[0], active_set[0],
                                   "coarse_active_dst");
          copy_subview_device(dst0, src0, ctx);
        }
        for (int lvl = 1; lvl <= active_finest; ++lvl) {
          if (!has_level[lvl]) {
            continue;
          }
          auto src = make_subview(U_levels[lvl], active_set[lvl],
                                  "fine_active_src_lvl");
          auto dst = make_subview(U_active_levels[lvl], active_set[lvl],
                                  "fine_active_dst_lvl");
          copy_subview_device(dst, src, ctx);
        }

        write_multilevel_outputs<MAX_AMR_LEVELS>(
            active_set,
            density_levels,
            pressure_levels,
            mach_levels,
            U_active_levels,
            has_level,
            active_finest,
            cfg.gamma,
            output_dir,
            step);
        const auto t_out_end = Clock::now();
        time_output_ms += std::chrono::duration<double, std::milli>(
                              t_out_end - t_out_begin)
                              .count();
      }

      const Real total_mass = compute_total_mass(U_levels[0]);
      const auto t_step_end = Clock::now();
      const double time_step_ms = std::chrono::duration<double, std::milli>(
                                      t_step_end - t_step_begin)
                                      .count();
      std::cout << "step=" << step
                << " t=" << t
                << " dt=" << dt
                << " mass=" << total_mass
                << " mass_drift=" << (total_mass - total_mass0)
                << " timings_ms:"
                << " prolong=" << time_prolong_ms
                << " fine=" << time_fine_ms
                << " coarse=" << time_coarse_ms
                << " restrict=" << time_restrict_ms
                << " remesh=" << time_remesh_ms
                << " remesh_masks=" << time_remesh_masks_ms
                << " remesh_mask_indicator=" << time_remesh_mask_indicator_ms
                << " remesh_mask_reduce=" << time_remesh_mask_reduce_ms
                << " remesh_mask_expand=" << time_remesh_mask_expand_ms
                << " remesh_mask_constrain=" << time_remesh_mask_constrain_ms
                << " remesh_geom=" << time_remesh_geom_ms
                << " remesh_prolong=" << time_remesh_prolong_ms
                << " remesh_overlap=" << time_remesh_overlap_ms
                << " output=" << time_output_ms
                << " total=" << time_step_ms
                << "\n";
    }
    }
  }
  return 0;
}
