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
#include <subsetix/csr_ops/morphology.hpp>
#include <subsetix/detail/csr_utils.hpp>
#include <subsetix/multilevel.hpp>
#include <subsetix/vtk_export.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <filesystem>

namespace {

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

struct Conserved {
  double rho;
  double rhou;
  double rhov;
  double E;
};

struct Primitive {
  double rho;
  double u;
  double v;
  double p;
};

struct RunConfig {
  int nx = 400;
  int ny = 160;
  int cx = -1; // set later from nx if still negative
  int cy = -1; // set later from ny if still negative
  int radius = 20;

  double mach_inlet = 2.0;
  double rho = 1.0;
  double p = 1.0;
  double gamma = 1.4;
  double cfl = 0.45;
  double t_final = 0.01;
  int max_steps = 5000;
  int output_stride = 50;
  bool no_slip = false;
  std::string pbm_path;

  bool enable_amr = true;
  double amr_fraction = 0.5; // fraction of domain length refined in each direction
  int amr_guard = 2;         // coarse-cell guard radius around the refined zone
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
Primitive cons_to_prim(const Conserved& U, double gamma) {
  constexpr double eps = 1e-12;
  Primitive q;
  q.rho = U.rho;
  const double inv_rho = 1.0 / (U.rho + eps);
  q.u = U.rhou * inv_rho;
  q.v = U.rhov * inv_rho;
  const double kinetic = 0.5 * (q.u * q.u + q.v * q.v);
  const double pressure = (gamma - 1.0) * (U.E - U.rho * kinetic);
  q.p = (pressure > eps) ? pressure : eps;
  return q;
}

KOKKOS_INLINE_FUNCTION
Conserved prim_to_cons(const Primitive& q, double gamma) {
  Conserved U;
  const double kinetic = 0.5 * q.rho * (q.u * q.u + q.v * q.v);
  U.rho = q.rho;
  U.rhou = q.rho * q.u;
  U.rhov = q.rho * q.v;
  U.E = q.p / (gamma - 1.0) + kinetic;
  return U;
}

KOKKOS_INLINE_FUNCTION
double sound_speed(const Primitive& q, double gamma) {
  constexpr double eps = 1e-12;
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
                         double gamma) {
  const double aL = sound_speed(qL, gamma);
  const double aR = sound_speed(qR, gamma);
  const double smax = std::fmax(std::fabs(qL.u) + aL,
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
                         double gamma) {
  const double aL = sound_speed(qL, gamma);
  const double aR = sound_speed(qR, gamma);
  const double smax = std::fmax(std::fabs(qL.v) + aL,
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
Conserved make_wall_ghost(const Conserved& interior,
                          double nx,
                          double ny,
                          double gamma,
                          bool no_slip) {
  Primitive q = cons_to_prim(interior, gamma);
  const double un = q.u * nx + q.v * ny;
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
                          double gamma,
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
    const double nx = 0.0;
    const double ny = (dy > 0) ? 1.0 : -1.0;
    return make_wall_ghost(center, nx, ny, gamma, no_slip);
  }

  // Inside rectangular domain -> obstacle
  const double nx = (dx != 0) ? ((dx > 0) ? 1.0 : -1.0) : 0.0;
  const double ny = (dy != 0) ? ((dy > 0) ? 1.0 : -1.0) : 0.0;
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
    double gamma,
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
    const double nx = (dy == 0) ? ((dx > 0) ? 1.0 : -1.0) : 0.0;
    const double ny = (dx == 0) ? ((dy > 0) ? 1.0 : -1.0) : 0.0;
    return make_wall_ghost(center, nx, ny, gamma, no_slip);
  }

  const double nx = static_cast<double>(-dx);
  const double ny = static_cast<double>(-dy);
  return make_wall_ghost(center, nx, ny, gamma, no_slip);
}

double compute_dt(const Field2DDevice<Conserved>& U,
                  double gamma,
                  double cfl,
                  double dx,
                  double dy) {
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  const double inv_dx = 1.0 / dx;
  const double inv_dy = 1.0 / dy;
  double max_rate = 0.0;

  Kokkos::parallel_reduce(
      "mach2_cylinder_dt",
      Kokkos::RangePolicy<ExecSpace>(
          0, static_cast<int>(U.size())),
      KOKKOS_LAMBDA(const int idx, double& lmax) {
        const Conserved s = U.values(idx);
        const Primitive q = cons_to_prim(s, gamma);
        const double a = sound_speed(q, gamma);
        const double rate = std::fabs(q.u) * inv_dx +
                            std::fabs(q.v) * inv_dy +
                            a * (inv_dx + inv_dy);
        if (rate > lmax) {
          lmax = rate;
        }
      },
      Kokkos::Max<double>(max_rate));

  if (max_rate <= 0.0) {
    return cfl * std::min(dx, dy);
  }
  return cfl / max_rate;
}

double compute_dt_on_set(const Field2DDevice<Conserved>& U,
                         const IntervalSet2DDevice& region,
                         double gamma,
                         double cfl,
                         double dx,
                         double dy) {
  if (region.num_intervals == 0) {
    return compute_dt(U, gamma, cfl, dx, dy);
  }

  const double inv_dx = 1.0 / dx;
  const double inv_dy = 1.0 / dy;
  double max_rate = 0.0;

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
      KOKKOS_LAMBDA(const int interval_idx, double& lmax) {
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
          const double a = sound_speed(q, gamma);
          const double rate = std::fabs(q.u) * inv_dx +
                              std::fabs(q.v) * inv_dy +
                              a * (inv_dx + inv_dy);
          if (rate > lmax) {
            lmax = rate;
          }
        }
      },
      Kokkos::Max<double>(max_rate));

  if (max_rate <= 0.0) {
    return cfl * std::min(dx, dy);
  }
  return cfl / max_rate;
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
  double gamma;
  double dt;
  double dx;
  double dy;
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
  double gamma;
  double dt;
  double dx;
  double dy;
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
                         Field2DDevice<double>& density,
                         Field2DDevice<double>& pressure,
                         Field2DDevice<double>& mach,
                         double gamma) {
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
        const double a = sound_speed(q, gamma);
        const double vel = std::sqrt(q.u * q.u + q.v * q.v);
        rho_out(idx) = q.rho;
        p_out(idx) = q.p;
        mach_out(idx) = (a > 1e-12) ? (vel / a) : 0.0;
      });
}

double compute_total_mass(const Field2DDevice<Conserved>& U) {
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  double total = 0.0;
  Kokkos::parallel_reduce(
      "mach2_cylinder_mass",
      Kokkos::RangePolicy<ExecSpace>(
          0, static_cast<int>(U.size())),
      KOKKOS_LAMBDA(const int idx, double& sum) {
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
    auto read_double = [&](double& out) {
      if (i + 1 < argc) {
        out = std::stod(argv[++i]);
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
    else if (arg == "--no-amr") cfg.enable_amr = false;
    else if (arg == "--amr") cfg.enable_amr = true;
    else if (arg == "--amr-fraction") read_double(cfg.amr_fraction);
    else if (arg == "--amr-guard") read_int(cfg.amr_guard);
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
  cfg.amr_fraction = std::clamp(cfg.amr_fraction, 0.05, 0.95);
  if (cfg.amr_guard < 1) {
    cfg.amr_guard = 1;
  }
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
  const double a = std::sqrt(cfg.gamma * q.p / q.rho);
  q.u = cfg.mach_inlet * a;
  q.v = 0.0;
  return prim_to_cons(q, cfg.gamma);
}


void write_step_outputs(const IntervalSet2DDevice& coarse_geom,
                        Field2DDevice<double>& density,
                        Field2DDevice<double>& pressure,
                        Field2DDevice<double>& mach,
                        double gamma,
                        const std::filesystem::path& out_dir,
                        int step,
                        const IntervalSet2DDevice* fine_geom = nullptr,
                        Field2DDevice<double>* density_fine = nullptr,
                        Field2DDevice<double>* pressure_fine = nullptr,
                        Field2DDevice<double>* mach_fine = nullptr,
                        const Field2DDevice<Conserved>* U_coarse = nullptr,
                        const Field2DDevice<Conserved>* U_fine_active = nullptr) {
  if (U_coarse) {
    compute_diagnostics(*U_coarse, density, pressure, mach, gamma);
  }
  if (U_fine_active && density_fine && pressure_fine && mach_fine &&
      U_fine_active->geometry.num_intervals > 0) {
    compute_diagnostics(*U_fine_active, *density_fine, *pressure_fine,
                        *mach_fine, gamma);
  }

  MultilevelGeoDevice geo;
  geo.origin_x = 0.0;
  geo.origin_y = 0.0;
  geo.root_dx = 1.0;
  geo.root_dy = 1.0;
  geo.num_active_levels = 1;
  geo.levels[0] = coarse_geom;

  MultilevelFieldDevice<double> f_density;
  MultilevelFieldDevice<double> f_pressure;
  MultilevelFieldDevice<double> f_mach;
  f_density.num_active_levels = 1;
  f_pressure.num_active_levels = 1;
  f_mach.num_active_levels = 1;
  f_density.levels[0] = density;
  f_pressure.levels[0] = pressure;
  f_mach.levels[0] = mach;

  const bool has_fine = (fine_geom && fine_geom->num_intervals > 0 &&
                         density_fine && pressure_fine && mach_fine &&
                         U_fine_active && U_fine_active->geometry.num_intervals > 0);
  if (has_fine) {
    geo.num_active_levels = 2;
    geo.levels[1] = *fine_geom;
    f_density.num_active_levels = 2;
    f_pressure.num_active_levels = 2;
    f_mach.num_active_levels = 2;
    f_density.levels[1] = *density_fine;
    f_pressure.levels[1] = *pressure_fine;
    f_mach.levels[1] = *mach_fine;
  }

  const auto host_geo = subsetix::deep_copy_to_host(geo);
  const auto host_rho = subsetix::deep_copy_to_host(f_density);
  const auto host_p = subsetix::deep_copy_to_host(f_pressure);
  const auto host_m = subsetix::deep_copy_to_host(f_mach);

  subsetix::vtk::write_multilevel_field_vtk(
      host_rho, host_geo, vtk_filename(out_dir, step, "density"), "rho");
  subsetix::vtk::write_multilevel_field_vtk(
      host_p, host_geo, vtk_filename(out_dir, step, "pressure"), "p");
  subsetix::vtk::write_multilevel_field_vtk(
      host_m, host_geo, vtk_filename(out_dir, step, "mach"), "mach");
}

} // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  {
    const RunConfig cfg = parse_args(argc, argv);
    const std::filesystem::path output_dir =
        subsetix_examples::make_example_output_dir("mach2_cylinder",
                                                   argc, argv);

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

    IntervalSet2DDevice fine_active;
    IntervalSet2DDevice fine_with_guard;
    IntervalSet2DDevice fine_guard;
    IntervalSet2DDevice projection_fine_on_coarse;
    Box2D fine_domain{0, 0, 0, 0};
    bool has_fine = false;

    if (cfg.enable_amr) {
      const Coord refine_w = static_cast<Coord>(
          std::max<int>(4, static_cast<int>(cfg.amr_fraction * cfg.nx)));
      const Coord refine_h = static_cast<Coord>(
          std::max<int>(4, static_cast<int>(cfg.amr_fraction * cfg.ny)));
      const Coord refine_x0 = domain.x_min +
                              static_cast<Coord>((cfg.nx - refine_w) / 2);
      const Coord refine_y0 = domain.y_min +
                              static_cast<Coord>((cfg.ny - refine_h) / 2);

      const Box2D refine_box{
          refine_x0, static_cast<Coord>(refine_x0 + refine_w),
          refine_y0, static_cast<Coord>(refine_y0 + refine_h)};

      auto refine_box_dev = make_box_device(refine_box);
      IntervalSet2DDevice refine_mask =
          subsetix::csr::allocate_interval_set_device(
              fluid_dev.num_rows,
              fluid_dev.num_intervals + refine_box_dev.num_intervals);
      set_intersection_device(fluid_dev, refine_box_dev, refine_mask, ctx);
      subsetix::csr::compute_cell_offsets_device(refine_mask);

      const Coord grow = static_cast<Coord>(cfg.amr_guard);
      IntervalSet2DDevice refine_mask_grown;
      expand_device(refine_mask, grow, grow, refine_mask_grown, ctx);
      subsetix::csr::compute_cell_offsets_device(refine_mask_grown);

      refine_level_up_device(refine_mask, fine_active, ctx);
      subsetix::csr::compute_cell_offsets_device(fine_active);

      refine_level_up_device(refine_mask_grown, fine_with_guard, ctx);
      subsetix::csr::compute_cell_offsets_device(fine_with_guard);

      fine_guard = subsetix::csr::allocate_interval_set_device(
          fine_with_guard.num_rows,
          fine_with_guard.num_intervals + fine_active.num_intervals);
      set_difference_device(fine_with_guard, fine_active, fine_guard, ctx);
      subsetix::csr::compute_cell_offsets_device(fine_guard);

      project_level_down_device(fine_active, projection_fine_on_coarse, ctx);
      subsetix::csr::compute_cell_offsets_device(projection_fine_on_coarse);

      has_fine = fine_active.num_rows > 0 && fine_active.num_intervals > 0;
      fine_domain = Box2D{static_cast<Coord>(domain.x_min * 2),
                          static_cast<Coord>(domain.x_max * 2),
                          static_cast<Coord>(domain.y_min * 2),
                          static_cast<Coord>(domain.y_max * 2)};
    }

    Field2DDevice<Conserved> U(fluid_dev, "mach2_state");
    Field2DDevice<Conserved> U_next(fluid_dev, "mach2_state_next");
    Field2DDevice<double> density(fluid_dev, "mach2_density");
    Field2DDevice<double> pressure(fluid_dev, "mach2_pressure");
    Field2DDevice<double> mach_field(fluid_dev, "mach2_mach");

    Field2DDevice<Conserved> U_fine;
    Field2DDevice<Conserved> U_fine_next;
    Field2DDevice<Conserved> U_fine_active;
    Field2DDevice<double> density_fine;
    Field2DDevice<double> pressure_fine;
    Field2DDevice<double> mach_fine;

    if (has_fine) {
      U_fine = Field2DDevice<Conserved>(fine_with_guard, "mach2_fine");
      U_fine_next = Field2DDevice<Conserved>(fine_with_guard,
                                             "mach2_fine_next");
      U_fine_active = Field2DDevice<Conserved>(fine_active,
                                               "mach2_fine_active");
      density_fine = Field2DDevice<double>(fine_active, "mach2_fine_density");
      pressure_fine = Field2DDevice<double>(fine_active, "mach2_fine_pressure");
      mach_fine = Field2DDevice<double>(fine_active, "mach2_fine_mach");
    }

    const Conserved inflow = build_inflow_state(cfg);

    {
      using ExecSpace = Kokkos::DefaultExecutionSpace;
      auto values = U.values;
      auto values_next = U_next.values;
      Kokkos::parallel_for(
          "mach2_cylinder_init",
          Kokkos::RangePolicy<ExecSpace>(
              0, static_cast<int>(U.size())),
          KOKKOS_LAMBDA(const int idx) {
            values(idx) = inflow;
            values_next(idx) = inflow;
          });

      if (has_fine) {
        auto fv = U_fine.values;
        auto fv_next = U_fine_next.values;
        Kokkos::parallel_for(
            "mach2_cylinder_init_fine",
            Kokkos::RangePolicy<ExecSpace>(
                0, static_cast<int>(U_fine.size())),
            KOKKOS_LAMBDA(const int idx) {
              fv(idx) = inflow;
              fv_next(idx) = inflow;
            });
      }
    }

    {
      const IntervalSet2DHost fluid_host =
          subsetix::csr::build_host_from_device(fluid_dev);
      const IntervalSet2DHost obstacle_host =
          subsetix::csr::build_host_from_device(obstacle_dev);
      write_legacy_quads(fluid_host,
                         subsetix_examples::output_file(output_dir, "fluid_geometry.vtk"));
      write_legacy_quads(obstacle_host,
                         subsetix_examples::output_file(output_dir, "obstacle_geometry.vtk"));
      if (has_fine) {
        const IntervalSet2DHost fine_host =
            subsetix::csr::build_host_from_device(fine_active);
        write_legacy_quads(fine_host,
                           subsetix_examples::output_file(output_dir, "fine_geometry.vtk"));
      }
    }

    double t = 0.0;
    int step = 0;
    const double dx = 1.0;
    const double dy = 1.0;
    const double dx_fine = 0.5 * dx;
    const double dy_fine = 0.5 * dy;

    std::cout << "Mach 2 cylinder setup: "
              << "nx=" << cfg.nx << " ny=" << cfg.ny
              << " cx=" << cfg.cx << " cy=" << cfg.cy
              << " r=" << cfg.radius
              << " pbm=" << (obstacle_from_pbm ? cfg.pbm_path : "(disk)")
              << " no-slip=" << (cfg.no_slip ? "yes" : "no")
              << " amr=" << (has_fine ? "enabled" : "disabled")
              << " output_dir=" << output_dir << "\n";

    double total_mass0 = compute_total_mass(U);

    while ((t < cfg.t_final) && (step < cfg.max_steps)) {
      const FieldReadAccessor<Conserved> acc_coarse = make_accessor(U);
      double dt = compute_dt(U, cfg.gamma, cfg.cfl, dx, dy);

      FieldReadAccessor<Conserved> acc_fine;
      if (has_fine) {
        prolong_guard_from_coarse(U_fine, fine_guard, acc_coarse);
        acc_fine = make_accessor(U_fine);
        const double dt_fine =
            compute_dt_on_set(U_fine, fine_active, cfg.gamma, cfg.cfl,
                              dx_fine, dy_fine);
        dt = std::min(dt, dt_fine);
      }

      if (t + dt > cfg.t_final) {
        dt = cfg.t_final - t;
      }

      if (has_fine) {
        FineEulerStencil fine_stencil{acc_fine, acc_coarse, fine_domain,
                                      inflow, cfg.gamma, dt, dx_fine, dy_fine,
                                      cfg.no_slip};
        apply_stencil_on_set_device(U_fine_next, U_fine, fine_active,
                                    fine_stencil);
      }

      CoarseEulerStencil coarse_stencil{acc_coarse, domain, inflow,
                                        cfg.gamma, dt, dx, dy, cfg.no_slip};
      apply_stencil_on_set_device(U_next, U, fluid_dev, coarse_stencil);

      std::swap(U.values, U_next.values);
      if (has_fine) {
        std::swap(U_fine.values, U_fine_next.values);
      }

      if (has_fine) {
        restrict_fine_to_coarse(U, U_fine, projection_fine_on_coarse);
      }

      t += dt;
      ++step;

      if (step % cfg.output_stride == 0 || step == cfg.max_steps ||
          t >= cfg.t_final - 1e-12) {
        if (has_fine) {
          auto fine_src = make_subview(U_fine, fine_active, "fine_active_src");
          auto fine_dst = make_subview(U_fine_active, fine_active,
                                       "fine_active_dst");
          copy_subview_device(fine_dst, fine_src, ctx);
        }

        write_step_outputs(fluid_dev,
                           density,
                           pressure,
                           mach_field,
                           cfg.gamma,
                           output_dir,
                           step,
                           has_fine ? &fine_active : nullptr,
                           has_fine ? &density_fine : nullptr,
                           has_fine ? &pressure_fine : nullptr,
                           has_fine ? &mach_fine : nullptr,
                           &U,
                           has_fine ? &U_fine_active : nullptr);

        const double total_mass = compute_total_mass(U);
        std::cout << "step=" << step
                  << " t=" << t
                  << " dt=" << dt
                  << " mass=" << total_mass
                  << " mass_drift=" << (total_mass - total_mass0)
                  << "\n";
      }
    }
  }
  return 0;
}
