#include <Kokkos_Core.hpp>

#include "../example_output.hpp"

#include <subsetix/field/csr_field.hpp>
#include <subsetix/field/csr_field_ops.hpp>
#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>
#include <subsetix/csr_ops/core.hpp>
#include <subsetix/csr_ops/morphology.hpp>
#include <subsetix/csr_ops/threshold.hpp>
#include <subsetix/csr_ops/field_stencil.hpp>
#include <subsetix/io/vtk_export.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Real = double;
using subsetix::csr::Box2D;
using subsetix::csr::Coord;
using subsetix::csr::DeviceMemorySpace;
using subsetix::csr::ExecSpace;
using subsetix::csr::HostMemorySpace;
using subsetix::csr::Field2DDevice;
using subsetix::csr::Interval;
using subsetix::csr::IntervalSet2DDevice;
using subsetix::csr::RowKey2D;
using subsetix::csr::IntervalSet2DHost;
using subsetix::csr::build_device_field_from_host;
using subsetix::csr::build_host_field_from_device;
using subsetix::csr::compute_cell_offsets_host;
using subsetix::csr::make_box_device;
using subsetix::csr::make_field_like_geometry;
using subsetix::csr::to;
using subsetix::vtk::write_legacy_quads;
using subsetix_examples::make_example_output_dir;
using subsetix_examples::output_file;

struct RunConfig {
  int nx = 160;
  int ny = 200;
  int max_steps = 1500;
  Real cfl = 0.4;
  Real diffusion = 0.0008;
  Real omega_diffusion = 0.0008;
  Real base_up = 0.4;
  Real base_right = 0.08;
  Real vort_amp = 0.6;
  Real vort_wave = 0.08;
  Real vort_time = 0.6;
  Real vort_buoyancy = 1.0;
  int poisson_iters = 30;
  Real buoyancy = 2.5;
  Real smoke_buoyancy = 1.0;
  Real ambient_temp = 0.0;
  Real source_smoke = 1.0;
  Real source_temp = 1.0;
  int source_width = 32;
  int source_height = 3;
  int output_stride = 50;
  Real output_dt = static_cast<Real>(0.02);
  Real threshold = 1e-4;
  int expand_margin = 12;
  int expand_x = 12;
  int expand_y = 12;
  int check_expand_stride = 5;
  int diag_stride = 25;
  bool enable_output = true;
  bool verbose = false;
};

template <typename T>
subsetix::csr::detail::FieldReadAccessor<T>
make_accessor(const Field2DDevice<T>& field) {
  subsetix::csr::detail::FieldReadAccessor<T> acc;
  acc.row_keys = field.geometry.row_keys;
  acc.row_ptr = field.geometry.row_ptr;
  acc.intervals = field.geometry.intervals;
  acc.offsets = field.geometry.cell_offsets;
  acc.values = field.values;
  acc.num_rows = field.geometry.num_rows;
  return acc;
}

Field2DDevice<Real>
make_field_for_box(const Box2D& box, Real init_value) {
  const IntervalSet2DDevice geom_dev = make_box_device(box);
  const auto geom_host = to<HostMemorySpace>(geom_dev);
  const auto host = make_field_like_geometry<Real>(geom_host, init_value);
  return build_device_field_from_host(host);
}

template <typename T>
Field2DDevice<T>
remap_to_box(const Field2DDevice<T>& src,
             const Box2D& old_box,
             const Box2D& new_box,
             T ambient) {
  Field2DDevice<T> dst = make_field_for_box(new_box, ambient);
  if (src.geometry.num_rows == 0 || src.geometry.num_intervals == 0) {
    return dst;
  }

  const auto src_acc = make_accessor(src);
  const auto intervals = dst.geometry.intervals;
  const auto offsets = dst.geometry.cell_offsets;
  const auto row_ptr = dst.geometry.row_ptr;
  const auto row_keys = dst.geometry.row_keys;
  auto out = dst.values;

  Kokkos::parallel_for(
      "subsetix_smoke_remap",
      Kokkos::RangePolicy<ExecSpace>(
          0, static_cast<int>(dst.geometry.num_rows)),
      KOKKOS_LAMBDA(const int row_idx) {
        const Coord y = row_keys(row_idx).y;
        const std::size_t iv_idx = row_ptr(row_idx);
        const Interval iv = intervals(iv_idx);
        const std::size_t base = offsets(iv_idx);

        for (Coord x = iv.begin; x < iv.end; ++x) {
          if (x >= old_box.x_min && x < old_box.x_max &&
              y >= old_box.y_min && y < old_box.y_max) {
            const std::size_t idx = base + static_cast<std::size_t>(x - iv.begin);
            out(idx) = src_acc.value_at(x, y);
          }
        }
      });
  ExecSpace().fence();
  return dst;
}

template <typename T>
Field2DDevice<T>
remap_to_geometry(const Field2DDevice<T>& src,
                  const IntervalSet2DDevice& new_geom,
                  T ambient) {
  const auto geom_host = to<HostMemorySpace>(new_geom);
  const auto host_field = make_field_like_geometry<T>(geom_host, ambient);
  Field2DDevice<T> dst = build_device_field_from_host(host_field);

  const auto src_acc = make_accessor(src);
  const auto intervals = dst.geometry.intervals;
  const auto offsets = dst.geometry.cell_offsets;
  const auto row_ptr = dst.geometry.row_ptr;
  const auto row_keys = dst.geometry.row_keys;
  auto out = dst.values;

  Kokkos::parallel_for(
      "subsetix_smoke_remap_geom",
      Kokkos::RangePolicy<ExecSpace>(
          0, static_cast<int>(dst.geometry.num_rows)),
      KOKKOS_LAMBDA(const int row_idx) {
        const Coord y = row_keys(row_idx).y;
        const std::size_t iv_begin = row_ptr(row_idx);
        const std::size_t iv_end = row_ptr(row_idx + 1);
        if (iv_begin == iv_end) {
          return;
        }
        const std::size_t iv_idx = iv_begin;
        const Interval iv = intervals(iv_idx);
        const std::size_t base = offsets(iv_idx);
        for (Coord x = iv.begin; x < iv.end; ++x) {
          const std::size_t idx = base + static_cast<std::size_t>(x - iv.begin);
          out(idx) = src_acc.value_at(x, y);
        }
      });
  ExecSpace().fence();
  return dst;
}

Box2D
bounds_from_set(const IntervalSet2DDevice& geom) {
  Box2D box{};
  if (geom.num_rows == 0 || geom.num_intervals == 0) {
    return box;
  }
  const auto host = to<HostMemorySpace>(geom);
  box.y_min = host.row_keys(0).y;
  box.y_max = host.row_keys(host.num_rows - 1).y + 1;
  Coord xmin = std::numeric_limits<Coord>::max();
  Coord xmax = std::numeric_limits<Coord>::min();
  for (std::size_t i = 0; i < host.num_intervals; ++i) {
    const auto iv = host.intervals(i);
    if (iv.begin < xmin) xmin = iv.begin;
    if (iv.end > xmax) xmax = iv.end;
  }
  box.x_min = xmin;
  box.x_max = xmax;
  return box;
}

IntervalSet2DDevice
build_expanded_mask(const Field2DDevice<Real>& smoke,
                    const RunConfig& cfg,
                    subsetix::csr::CsrSetAlgebraContext& ctx) {
  IntervalSet2DDevice active = subsetix::csr::threshold_field(
      smoke, static_cast<double>(cfg.threshold));
  IntervalSet2DDevice expanded;
  expand_device(active, cfg.expand_x, cfg.expand_y, expanded, ctx);
  // Clamp to y >= 0 (keep floor fixed).
  const auto host = to<HostMemorySpace>(expanded);

  // First pass: count valid rows and intervals
  std::vector<RowKey2D> temp_row_keys;
  std::vector<std::size_t> temp_row_ptr;
  std::vector<Interval> temp_intervals;
  temp_row_ptr.push_back(0);

  for (std::size_t i = 0; i < host.num_rows; ++i) {
    if (host.row_keys(i).y < 0) {
      continue;
    }
    const std::size_t begin = host.row_ptr(i);
    const std::size_t end = host.row_ptr(i + 1);
    const std::size_t count = end - begin;
    if (count == 0) {
      continue;
    }
    temp_row_keys.push_back(host.row_keys(i));
    for (std::size_t k = begin; k < end; ++k) {
      temp_intervals.push_back(host.intervals(k));
    }
    temp_row_ptr.push_back(temp_row_ptr.back() + count);
  }

  // Build filtered IntervalSet2DHost
  IntervalSet2DHost filtered;
  filtered.num_rows = temp_row_keys.size();
  filtered.num_intervals = temp_intervals.size();

  if (filtered.num_rows == 0) {
    return to<DeviceMemorySpace>(filtered);
  }

  filtered.row_keys = IntervalSet2DHost::RowKeyView("row_keys", filtered.num_rows);
  filtered.row_ptr = IntervalSet2DHost::IndexView("row_ptr", filtered.num_rows + 1);
  filtered.intervals = IntervalSet2DHost::IntervalView("intervals", filtered.num_intervals);

  for (std::size_t i = 0; i < temp_row_keys.size(); ++i) {
    filtered.row_keys(i) = temp_row_keys[i];
  }
  for (std::size_t i = 0; i < temp_row_ptr.size(); ++i) {
    filtered.row_ptr(i) = temp_row_ptr[i];
  }
  for (std::size_t i = 0; i < temp_intervals.size(); ++i) {
    filtered.intervals(i) = temp_intervals[i];
  }

  compute_cell_offsets_host(filtered);
  return to<DeviceMemorySpace>(filtered);
}

Real
compute_dt(const Field2DDevice<Real>& smoke,
           const Field2DDevice<Real>& temperature,
           const Field2DDevice<Real>& u_field,
           const Field2DDevice<Real>& v_field,
           const RunConfig& cfg,
           Real dx,
           Real dy) {
  Real max_vx = 0.0;
  const auto u_vals = u_field.values;
  const auto v_vals = v_field.values;
  const std::size_t n = std::min(u_vals.extent(0), v_vals.extent(0));
  Kokkos::parallel_reduce(
      "subsetix_smoke_max_speed",
      Kokkos::RangePolicy<ExecSpace>(
          0, static_cast<int>(n)),
      KOKKOS_LAMBDA(const int i, Real& max_v) {
        const Real ux = cfg.base_right + u_vals(i);
        const Real uy = cfg.base_up + v_vals(i);
        const Real mag = std::fabs(ux) + std::fabs(uy);
        if (mag > max_v) {
          max_v = mag;
        }
      },
      Kokkos::Max<Real>(max_vx));

  const Real adv_x = (max_vx > 0) ? (dx / max_vx) : std::numeric_limits<Real>::max();
  const Real adv_y = (max_vx > 0) ? (dy / max_vx) : std::numeric_limits<Real>::max();
  const Real diff_term = (cfg.diffusion > 0)
                           ? static_cast<Real>(0.25) *
                                 std::min(dx * dx, dy * dy) / cfg.diffusion
                           : std::numeric_limits<Real>::max();

  const Real dt_candidate = std::min({adv_x, adv_y, diff_term});
  const Real dt_safe =
      (dt_candidate == std::numeric_limits<Real>::max())
          ? static_cast<Real>(0.01)
          : dt_candidate;
  return cfg.cfl * dt_safe;
}

struct SmokeDiagnostics {
  static constexpr int inactive_y = -2147483647;
  Real mass = 0.0;
  Real max_value = 0.0;
  int max_y = inactive_y;
};

struct SmokeDiagnosticsReducer {
  using value_type = SmokeDiagnostics;

  subsetix::csr::IntervalSet2DDevice::IntervalView intervals;
  subsetix::csr::IntervalSet2DDevice::IndexView offsets;
  subsetix::csr::IntervalSet2DDevice::IndexView row_ptr;
  subsetix::csr::IntervalSet2DDevice::RowKeyView row_keys;
  Kokkos::View<Real*> values;
  Real threshold;

  KOKKOS_FUNCTION
  void operator()(const int row_idx, value_type& out) const {
    const Coord y = row_keys(row_idx).y;
    const std::size_t first_iv = row_ptr(row_idx);
    const std::size_t end_iv = row_ptr(row_idx + 1);
    for (std::size_t iv_idx = first_iv; iv_idx < end_iv; ++iv_idx) {
      const Interval iv = intervals(iv_idx);
      const std::size_t base = offsets(iv_idx);
      for (Coord x = iv.begin; x < iv.end; ++x) {
        const std::size_t idx = base + static_cast<std::size_t>(x - iv.begin);
        const Real v = values(idx);
        if (v > threshold) {
          out.mass += v;
          if (v > out.max_value) {
            out.max_value = v;
          }
          if (y > out.max_y) {
            out.max_y = static_cast<int>(y);
          }
        }
      }
    }
  }

  KOKKOS_FUNCTION
  static void join(value_type& dst, const value_type& src) {
    dst.mass += src.mass;
    if (src.max_value > dst.max_value) {
      dst.max_value = src.max_value;
    }
    if (src.max_y > dst.max_y) {
      dst.max_y = src.max_y;
    }
  }

  KOKKOS_FUNCTION
  static void init(value_type& val) {
    val.mass = 0.0;
    val.max_value = 0.0;
    val.max_y = value_type::inactive_y;
  }
};

SmokeDiagnostics
compute_diagnostics(const Field2DDevice<Real>& smoke, Real threshold) {
  SmokeDiagnostics diag;
  if (smoke.geometry.num_rows == 0 || smoke.geometry.num_intervals == 0) {
    return diag;
  }

  SmokeDiagnosticsReducer reducer;
  reducer.intervals = smoke.geometry.intervals;
  reducer.offsets = smoke.geometry.cell_offsets;
  reducer.row_ptr = smoke.geometry.row_ptr;
  reducer.row_keys = smoke.geometry.row_keys;
  reducer.values = smoke.values;
  reducer.threshold = threshold;

  Kokkos::parallel_reduce(
      "subsetix_smoke_diag",
      Kokkos::RangePolicy<ExecSpace>(
          0, static_cast<int>(smoke.geometry.num_rows)),
      reducer,
      diag);
  return diag;
}

void
advance_omega(const Field2DDevice<Real>& omega,
              Field2DDevice<Real>& omega_next,
              const Field2DDevice<Real>& u_field,
              const Field2DDevice<Real>& v_field,
              const Field2DDevice<Real>& temperature,
              const RunConfig& cfg,
              Real time,
              Real dt,
              Real dx,
              Real dy) {
  const auto intervals = omega.geometry.intervals;
  const auto offsets = omega.geometry.cell_offsets;
  const auto row_ptr = omega.geometry.row_ptr;
  const auto row_keys = omega.geometry.row_keys;
  const auto vals = omega.values;
  auto out_vals = omega_next.values;
  const auto u_vals = u_field.values;
  const auto v_vals = v_field.values;
  const auto temp_vals = temperature.values;

  Kokkos::parallel_for(
      "subsetix_smoke_advect_omega",
      Kokkos::RangePolicy<ExecSpace>(
          0, static_cast<int>(omega.geometry.num_rows)),
      KOKKOS_LAMBDA(const int row_idx) {
        const std::size_t iv_begin = row_ptr(row_idx);
        const std::size_t iv_end = row_ptr(row_idx + 1);
        if (iv_begin == iv_end) {
          return;
        }
        const Coord y = row_keys(row_idx).y;
        const bool has_down = (row_idx > 0);
        const bool has_up =
            (static_cast<std::size_t>(row_idx + 1) < omega.geometry.num_rows);
        std::size_t down_iv = 0;
        std::size_t up_iv = 0;
        if (has_down) {
          down_iv = row_ptr(row_idx - 1);
        }
        if (has_up) {
          up_iv = row_ptr(row_idx + 1);
        }
        const Interval down_interval =
            has_down ? intervals(down_iv) : Interval{};
        const Interval up_interval =
            has_up ? intervals(up_iv) : Interval{};
        const std::size_t down_base =
            has_down ? offsets(down_iv) : std::size_t(0);
        const std::size_t up_base =
            has_up ? offsets(up_iv) : std::size_t(0);

        for (std::size_t iv_idx = iv_begin; iv_idx < iv_end; ++iv_idx) {
          const Interval iv = intervals(iv_idx);
          const std::size_t base = offsets(iv_idx);
          for (Coord x = iv.begin; x < iv.end; ++x) {
            const std::size_t idx = base + static_cast<std::size_t>(x - iv.begin);
            const Real w = vals(idx);

            Real left = w;
            Real right = w;
            if (x > iv.begin) {
              left = vals(idx - 1);
            }
            if (x + 1 < iv.end) {
              right = vals(idx + 1);
            }

            Real down = w;
            if (has_down && x >= down_interval.begin && x < down_interval.end) {
              down = vals(down_base + static_cast<std::size_t>(x - down_interval.begin));
            }

            Real up = w;
            if (has_up && x >= up_interval.begin && x < up_interval.end) {
              up = vals(up_base + static_cast<std::size_t>(x - up_interval.begin));
            }

            // Temperature gradient source (baroclinic/vorticity generation)
            Real t_left = temp_vals(idx);
            Real t_right = temp_vals(idx);
            if (x > iv.begin) {
              t_left = temp_vals(idx - 1);
            }
            if (x + 1 < iv.end) {
              t_right = temp_vals(idx + 1);
            }
            const Real dTdx = (t_right - t_left) * static_cast<Real>(0.5) / dx;

            const Real phase =
                cfg.vort_wave * static_cast<Real>(y) + cfg.vort_time * time;
            const Real ux =
                cfg.base_right + u_vals(idx) + cfg.vort_amp * std::sin(phase);
            const Real vy =
                cfg.base_up + v_vals(idx);

            const Real wx = (ux >= 0.0)
                                ? (w - left) / dx
                                : (right - w) / dx;
            const Real wy = (vy >= 0.0)
                                ? (w - down) / dy
                                : (up - w) / dy;

            const Real lap =
                (left - static_cast<Real>(2.0) * w + right) / (dx * dx) +
                (down - static_cast<Real>(2.0) * w + up) / (dy * dy);

            Real next = w - dt * (ux * wx + vy * wy)
                        + dt * cfg.omega_diffusion * lap
                        + dt * cfg.vort_buoyancy * dTdx;
            out_vals(idx) = next;
          }
        }
      });
  ExecSpace().fence();
}

void
solve_poisson(const Field2DDevice<Real>& omega,
              Field2DDevice<Real>& psi,
              Field2DDevice<Real>& psi_tmp,
              const RunConfig& cfg,
              Real dx,
              Real dy) {
  const Real dx2 = dx * dx;
  const Real dy2 = dy * dy;
  const Real denom = static_cast<Real>(2.0) * (dx2 + dy2);

  const auto intervals = psi.geometry.intervals;
  const auto offsets = psi.geometry.cell_offsets;
  const auto row_ptr = psi.geometry.row_ptr;
  const auto row_keys = psi.geometry.row_keys;
  auto psi_vals = psi.values;
  auto tmp_vals = psi_tmp.values;
  const auto omega_vals = omega.values;

  for (int iter = 0; iter < cfg.poisson_iters; ++iter) {
    Kokkos::parallel_for(
        "subsetix_smoke_poisson",
        Kokkos::RangePolicy<ExecSpace>(
            0, static_cast<int>(psi.geometry.num_rows)),
        KOKKOS_LAMBDA(const int row_idx) {
          const std::size_t iv_begin = row_ptr(row_idx);
          const std::size_t iv_end = row_ptr(row_idx + 1);
          if (iv_begin == iv_end) {
            return;
          }
          const bool has_down = (row_idx > 0);
          const bool has_up =
              (static_cast<std::size_t>(row_idx + 1) < psi.geometry.num_rows);
          std::size_t down_iv = 0;
          std::size_t up_iv = 0;
          if (has_down) down_iv = row_ptr(row_idx - 1);
          if (has_up) up_iv = row_ptr(row_idx + 1);
          const Interval down_interval = has_down ? intervals(down_iv) : Interval{};
          const Interval up_interval = has_up ? intervals(up_iv) : Interval{};
          const std::size_t down_base = has_down ? offsets(down_iv) : std::size_t(0);
          const std::size_t up_base = has_up ? offsets(up_iv) : std::size_t(0);

          for (std::size_t iv_idx = iv_begin; iv_idx < iv_end; ++iv_idx) {
            const Interval iv = intervals(iv_idx);
            const std::size_t base = offsets(iv_idx);
            for (Coord x = iv.begin; x < iv.end; ++x) {
              const std::size_t idx = base + static_cast<std::size_t>(x - iv.begin);
              const Real psi_c = psi_vals(idx);

              Real psi_l = psi_c;
              Real psi_r = psi_c;
              if (x > iv.begin) {
                psi_l = psi_vals(idx - 1);
              }
              if (x + 1 < iv.end) {
                psi_r = psi_vals(idx + 1);
              }

              Real psi_d = 0.0;
              if (has_down && x >= down_interval.begin && x < down_interval.end) {
                psi_d = psi_vals(down_base + static_cast<std::size_t>(x - down_interval.begin));
              }
              Real psi_u = 0.0;
              if (has_up && x >= up_interval.begin && x < up_interval.end) {
                psi_u = psi_vals(up_base + static_cast<std::size_t>(x - up_interval.begin));
              }

              const Real rhs = -omega_vals(idx);
              const Real num = (psi_l + psi_r) * dy2 + (psi_u + psi_d) * dx2 + rhs * dx2 * dy2;
              tmp_vals(idx) = num / denom;
            }
          }
        });
    ExecSpace().fence();
    std::swap(psi_vals, tmp_vals);
  }
  // Ensure psi holds latest values
  psi.values = psi_vals;
  psi_tmp.values = tmp_vals;
}

void
compute_velocity(const Field2DDevice<Real>& psi,
                 Field2DDevice<Real>& u_field,
                 Field2DDevice<Real>& v_field,
                 Real dx,
                 Real dy) {
  const auto intervals = psi.geometry.intervals;
  const auto offsets = psi.geometry.cell_offsets;
  const auto row_ptr = psi.geometry.row_ptr;
  const auto row_keys = psi.geometry.row_keys;
  const auto psi_vals = psi.values;
  auto u_vals = u_field.values;
  auto v_vals = v_field.values;

  Kokkos::parallel_for(
      "subsetix_smoke_velocity",
      Kokkos::RangePolicy<ExecSpace>(
          0, static_cast<int>(psi.geometry.num_rows)),
      KOKKOS_LAMBDA(const int row_idx) {
        const std::size_t iv_begin = row_ptr(row_idx);
        const std::size_t iv_end = row_ptr(row_idx + 1);
        if (iv_begin == iv_end) {
          return;
        }
        const bool has_down = (row_idx > 0);
        const bool has_up =
            (static_cast<std::size_t>(row_idx + 1) < psi.geometry.num_rows);
        std::size_t down_iv = 0;
        std::size_t up_iv = 0;
        if (has_down) down_iv = row_ptr(row_idx - 1);
        if (has_up) up_iv = row_ptr(row_idx + 1);
        const Interval down_interval = has_down ? intervals(down_iv) : Interval{};
        const Interval up_interval = has_up ? intervals(up_iv) : Interval{};
        const std::size_t down_base = has_down ? offsets(down_iv) : std::size_t(0);
        const std::size_t up_base = has_up ? offsets(up_iv) : std::size_t(0);

        for (std::size_t iv_idx = iv_begin; iv_idx < iv_end; ++iv_idx) {
          const Interval iv = intervals(iv_idx);
          const std::size_t base = offsets(iv_idx);
          for (Coord x = iv.begin; x < iv.end; ++x) {
            const std::size_t idx = base + static_cast<std::size_t>(x - iv.begin);
            Real psi_l = psi_vals(idx);
            Real psi_r = psi_vals(idx);
            if (x > iv.begin) {
              psi_l = psi_vals(idx - 1);
            }
            if (x + 1 < iv.end) {
              psi_r = psi_vals(idx + 1);
            }

            Real psi_d = psi_vals(idx);
            if (has_down && x >= down_interval.begin && x < down_interval.end) {
              psi_d = psi_vals(down_base + static_cast<std::size_t>(x - down_interval.begin));
            }
            Real psi_u = psi_vals(idx);
            if (has_up && x >= up_interval.begin && x < up_interval.end) {
              psi_u = psi_vals(up_base + static_cast<std::size_t>(x - up_interval.begin));
            }

            u_vals(idx) = (psi_u - psi_d) * static_cast<Real>(0.5) / dy;
            v_vals(idx) = -(psi_r - psi_l) * static_cast<Real>(0.5) / dx;
          }
        }
      });
  ExecSpace().fence();
}
void
advance_scalar(const Field2DDevice<Real>& value,
               Field2DDevice<Real>& next_value,
               const Field2DDevice<Real>& u_field,
               const Field2DDevice<Real>& v_field,
               const Field2DDevice<Real>& smoke,
               const Field2DDevice<Real>& temperature,
               const RunConfig& cfg,
               Real time,
               Real dt,
               Real dx,
               Real dy,
               Real diffusion) {
  const auto intervals = value.geometry.intervals;
  const auto offsets = value.geometry.cell_offsets;
  const auto row_ptr = value.geometry.row_ptr;
  const auto row_keys = value.geometry.row_keys;
  const auto vals = value.values;
  auto out_vals = next_value.values;
  const auto smoke_vals = smoke.values;
  const auto temp_vals = temperature.values;
  const auto u_vals = u_field.values;
  const auto v_vals = v_field.values;

  Kokkos::parallel_for(
      "subsetix_smoke_advance",
      Kokkos::RangePolicy<ExecSpace>(
          0, static_cast<int>(value.geometry.num_rows)),
      KOKKOS_LAMBDA(const int row_idx) {
        const std::size_t iv_begin = row_ptr(row_idx);
        const std::size_t iv_end = row_ptr(row_idx + 1);
        if (iv_begin == iv_end) {
          return;
        }
        const Coord y = row_keys(row_idx).y;
        for (std::size_t iv_idx = iv_begin; iv_idx < iv_end; ++iv_idx) {
          const Interval iv = intervals(iv_idx);
          const std::size_t base = offsets(iv_idx);
          const bool has_down = (row_idx > 0);
          const bool has_up =
              (static_cast<std::size_t>(row_idx + 1) < value.geometry.num_rows);
          std::size_t down_iv = 0;
          std::size_t up_iv = 0;
          if (has_down) {
            down_iv = row_ptr(row_idx - 1);
          }
          if (has_up) {
            up_iv = row_ptr(row_idx + 1);
          }

          const Interval down_interval =
              has_down ? intervals(down_iv) : Interval{};
          const Interval up_interval =
              has_up ? intervals(up_iv) : Interval{};
          const std::size_t down_base =
              has_down ? offsets(down_iv) : std::size_t(0);
          const std::size_t up_base =
              has_up ? offsets(up_iv) : std::size_t(0);

          for (Coord x = iv.begin; x < iv.end; ++x) {
            const std::size_t idx = base + static_cast<std::size_t>(x - iv.begin);
            const Real c = vals(idx);

            Real left = c;
            Real right = c;
            if (x > iv.begin) {
              left = vals(idx - 1);
            }
            if (x + 1 < iv.end) {
              right = vals(idx + 1);
            }

            Real down = c;
            if (has_down && x >= down_interval.begin && x < down_interval.end) {
              down = vals(down_base + static_cast<std::size_t>(x - down_interval.begin));
            }

            Real up = c;
            if (has_up && x >= up_interval.begin && x < up_interval.end) {
              up = vals(up_base + static_cast<std::size_t>(x - up_interval.begin));
            }

            const Real phase =
                cfg.vort_wave * static_cast<Real>(y) + cfg.vort_time * time;
            const Real ux =
                cfg.base_right + u_vals(idx) + cfg.vort_amp * std::sin(phase);
            const Real vy =
                cfg.base_up + v_vals(idx);

            const Real wx = (ux >= 0.0)
                                ? (c - left) / dx
                                : (right - c) / dx;
            const Real wy = (vy >= 0.0)
                                ? (c - down) / dy
                                : (up - c) / dy;

            const Real lap =
                (left - static_cast<Real>(2.0) * c + right) / (dx * dx) +
                (down - static_cast<Real>(2.0) * c + up) / (dy * dy);

            Real next = c - dt * (ux * wx + vy * wy) + dt * diffusion * lap;
            if (next < 0.0) {
              next = 0.0;
            }

            out_vals(idx) = next;
          }
        }
      });
  ExecSpace().fence();
}

void
apply_source(Field2DDevice<Real>& smoke,
             Field2DDevice<Real>& temperature,
             const RunConfig& cfg,
             const Box2D& box,
             Coord source_center_x) {
  const int half_w = cfg.source_width / 2;
  const Coord xmin = static_cast<Coord>(source_center_x - half_w);
  const Coord xmax = static_cast<Coord>(source_center_x + half_w);
  const Coord y_top = static_cast<Coord>(box.y_min + cfg.source_height);

  auto smoke_vals = smoke.values;
  auto temp_vals = temperature.values;
  const auto intervals = smoke.geometry.intervals;
  const auto offsets = smoke.geometry.cell_offsets;
  const auto row_ptr = smoke.geometry.row_ptr;
  const auto row_keys = smoke.geometry.row_keys;

  Kokkos::parallel_for(
      "subsetix_smoke_source",
      Kokkos::RangePolicy<ExecSpace>(
          0, static_cast<int>(smoke.geometry.num_rows)),
      KOKKOS_LAMBDA(const int row_idx) {
        const Coord y = row_keys(row_idx).y;
        if (y < box.y_min || y >= y_top) {
          return;
        }

        const std::size_t iv_begin = row_ptr(row_idx);
        const std::size_t iv_end = row_ptr(row_idx + 1);
        if (iv_begin == iv_end) {
          return;
        }
        const std::size_t iv_idx = iv_begin;
        const Interval iv = intervals(iv_idx);
        const std::size_t base = offsets(iv_idx);

        const Coord x0 = (iv.begin > xmin) ? iv.begin : xmin;
        const Coord x1 = (iv.end < xmax) ? iv.end : xmax;
        if (x0 >= x1) {
          return;
        }

        for (Coord x = x0; x < x1; ++x) {
          const std::size_t idx = base + static_cast<std::size_t>(x - iv.begin);
          smoke_vals(idx) = cfg.source_smoke;
          temp_vals(idx) = cfg.source_temp;
        }
      });
  ExecSpace().fence();
}

std::string
format_step(int step) {
  std::ostringstream oss;
  oss << std::setw(4) << std::setfill('0') << step;
  return oss.str();
}

RunConfig
parse_args(int argc, char* argv[]) {
  RunConfig cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    auto read_int = [&](int& target) {
      if (i + 1 < argc) {
        target = std::atoi(argv[++i]);
      }
    };
    auto read_real = [&](Real& target) {
      if (i + 1 < argc) {
        target = std::atof(argv[++i]);
      }
    };

    if (arg == "--nx") read_int(cfg.nx);
    else if (arg == "--ny") read_int(cfg.ny);
    else if (arg == "--steps") read_int(cfg.max_steps);
    else if (arg == "--cfl") read_real(cfg.cfl);
    else if (arg == "--diffusion") read_real(cfg.diffusion);
    else if (arg == "--omega-diffusion") read_real(cfg.omega_diffusion);
    else if (arg == "--base-up") read_real(cfg.base_up);
    else if (arg == "--base-right") read_real(cfg.base_right);
    else if (arg == "--vort-amp") read_real(cfg.vort_amp);
    else if (arg == "--vort-wave") read_real(cfg.vort_wave);
    else if (arg == "--vort-time") read_real(cfg.vort_time);
    else if (arg == "--vort-buoyancy") read_real(cfg.vort_buoyancy);
    else if (arg == "--poisson-iters") read_int(cfg.poisson_iters);
    else if (arg == "--buoyancy") read_real(cfg.buoyancy);
    else if (arg == "--smoke-buoyancy") read_real(cfg.smoke_buoyancy);
    else if (arg == "--source-smoke") read_real(cfg.source_smoke);
    else if (arg == "--source-temp") read_real(cfg.source_temp);
    else if (arg == "--source-width") read_int(cfg.source_width);
    else if (arg == "--source-height") read_int(cfg.source_height);
    else if (arg == "--threshold") read_real(cfg.threshold);
    else if (arg == "--output-stride") read_int(cfg.output_stride);
    else if (arg == "--output-dt") read_real(cfg.output_dt);
    else if (arg == "--expand-margin") read_int(cfg.expand_margin);
    else if (arg == "--expand-x") read_int(cfg.expand_x);
    else if (arg == "--expand-y") read_int(cfg.expand_y);
    else if (arg == "--expand-stride") read_int(cfg.check_expand_stride);
    else if (arg == "--diag-stride") read_int(cfg.diag_stride);
    else if (arg == "--no-output") cfg.enable_output = false;
    else if (arg == "--verbose") cfg.verbose = true;
  }
  return cfg;
}

void
export_fields(const Field2DDevice<Real>& smoke,
              const Field2DDevice<Real>& temperature,
              const std::filesystem::path& output_dir,
              int step) {
  const auto smoke_host = build_host_field_from_device(smoke);
  const auto temp_host = build_host_field_from_device(temperature);
  const std::string smoke_name = "smoke_step" + format_step(step) + ".vtk";
  const std::string temp_name = "temperature_step" + format_step(step) + ".vtk";
  write_legacy_quads(smoke_host, output_file(output_dir, smoke_name), "smoke");
  write_legacy_quads(temp_host, output_file(output_dir, temp_name), "temperature");
}

int run(int argc, char* argv[]) {
  RunConfig cfg = parse_args(argc, argv);
  if (cfg.nx <= 0 || cfg.ny <= 0) {
    std::cerr << "nx and ny must be positive\n";
    return 1;
  }

  const Real dx = static_cast<Real>(1.0);
  const Real dy = static_cast<Real>(1.0);

  const std::filesystem::path output_dir =
      cfg.enable_output
          ? make_example_output_dir("smoke_plume", argc, argv)
          : std::filesystem::path();

  Box2D domain;
  domain.x_min = 0;
  domain.x_max = cfg.nx;
  domain.y_min = 0;
  domain.y_max = cfg.ny;

  const Coord source_center_x =
      static_cast<Coord>(domain.x_min + (domain.x_max - domain.x_min) / 2);

  Field2DDevice<Real> smoke = make_field_for_box(domain, static_cast<Real>(0.0));
  Field2DDevice<Real> smoke_next = make_field_for_box(domain, static_cast<Real>(0.0));
  Field2DDevice<Real> temperature =
      make_field_for_box(domain, cfg.ambient_temp);
  Field2DDevice<Real> temperature_next =
      make_field_for_box(domain, cfg.ambient_temp);
  Field2DDevice<Real> omega = make_field_for_box(domain, static_cast<Real>(0.0));
  Field2DDevice<Real> omega_next = make_field_for_box(domain, static_cast<Real>(0.0));
  Field2DDevice<Real> psi = make_field_for_box(domain, static_cast<Real>(0.0));
  Field2DDevice<Real> psi_tmp = make_field_for_box(domain, static_cast<Real>(0.0));
  Field2DDevice<Real> u_field = make_field_for_box(domain, static_cast<Real>(0.0));
  Field2DDevice<Real> v_field = make_field_for_box(domain, static_cast<Real>(0.0));

  apply_source(smoke, temperature, cfg, domain, source_center_x);

  subsetix::csr::CsrSetAlgebraContext set_ctx;

  Real t = 0.0;
  Real last_output_t = 0.0;

  if (cfg.enable_output) {
    export_fields(smoke, temperature, output_dir, 0);
  }

  std::cout << "Smoke plume: nx=" << cfg.nx
            << " ny=" << cfg.ny
            << " steps=" << cfg.max_steps
            << " diffusion=" << cfg.diffusion
            << " base_up=" << cfg.base_up
            << " base_right=" << cfg.base_right
            << " buoyancy=" << cfg.buoyancy
            << " smoke_buoyancy=" << cfg.smoke_buoyancy
            << " vort_buoyancy=" << cfg.vort_buoyancy
            << " poisson_iters=" << cfg.poisson_iters
            << " expand_margin=" << cfg.expand_margin
            << " expand_stride=" << cfg.check_expand_stride
            << " output=" << (cfg.enable_output ? output_dir.string() : "(disabled)")
            << "\n";

  for (int step = 1; step <= cfg.max_steps; ++step) {
    const Real dt = compute_dt(smoke, temperature, u_field, v_field, cfg, dx, dy);

    advance_omega(omega, omega_next, u_field, v_field, temperature, cfg, t, dt, dx, dy);
    std::swap(omega.values, omega_next.values);

    solve_poisson(omega, psi, psi_tmp, cfg, dx, dy);
    compute_velocity(psi, u_field, v_field, dx, dy);

    advance_scalar(smoke, smoke_next, u_field, v_field, smoke, temperature, cfg, t, dt, dx, dy, cfg.diffusion);
    advance_scalar(temperature, temperature_next, u_field, v_field, smoke, temperature, cfg, t, dt, dx, dy, cfg.diffusion);
    std::swap(smoke.values, smoke_next.values);
    std::swap(temperature.values, temperature_next.values);

    apply_source(smoke, temperature, cfg, domain, source_center_x);

    t += dt;

    if (cfg.verbose && step % 25 == 0) {
      std::cout << "step " << step << " t=" << t << " dt=" << dt
                << " domain=[" << domain.x_min << "," << domain.x_max
                << "]x[" << domain.y_min << "," << domain.y_max << "]\n";
    }

    if (cfg.enable_output && (step % cfg.output_stride == 0 || t - last_output_t >= cfg.output_dt)) {
      export_fields(smoke, temperature, output_dir, step);
      last_output_t = t;
    }

    if (cfg.diag_stride > 0 && step % cfg.diag_stride == 0) {
      const SmokeDiagnostics diag = compute_diagnostics(smoke, cfg.threshold);
      const int front = (diag.max_y == SmokeDiagnostics::inactive_y)
                            ? -1
                            : diag.max_y;
      std::cout << "diag step=" << step
                << " t=" << t
                << " dt=" << dt
                << " mass=" << diag.mass
                << " max=" << diag.max_value
                << " front_y=" << front
                << " domain_h=" << (domain.y_max - domain.y_min)
                << " domain_w=" << (domain.x_max - domain.x_min)
                << "\n";
    }

    if (step % cfg.check_expand_stride == 0) {
      IntervalSet2DDevice new_geom = build_expanded_mask(smoke, cfg, set_ctx);
      const bool has_geom =
          new_geom.num_rows > 0 && new_geom.num_intervals > 0;
      const bool changed =
          has_geom &&
          (new_geom.num_rows != smoke.geometry.num_rows ||
           new_geom.num_intervals != smoke.geometry.num_intervals);
      if (has_geom && changed) {
        smoke = remap_to_geometry(smoke, new_geom, static_cast<Real>(0.0));
        smoke_next = remap_to_geometry(smoke_next, new_geom, static_cast<Real>(0.0));
        temperature = remap_to_geometry(temperature, new_geom, cfg.ambient_temp);
        temperature_next =
            remap_to_geometry(temperature_next, new_geom, cfg.ambient_temp);
        omega = remap_to_geometry(omega, new_geom, static_cast<Real>(0.0));
        omega_next = remap_to_geometry(omega_next, new_geom, static_cast<Real>(0.0));
        psi = remap_to_geometry(psi, new_geom, static_cast<Real>(0.0));
        psi_tmp = remap_to_geometry(psi_tmp, new_geom, static_cast<Real>(0.0));
        u_field = remap_to_geometry(u_field, new_geom, static_cast<Real>(0.0));
        v_field = remap_to_geometry(v_field, new_geom, static_cast<Real>(0.0));
        domain = bounds_from_set(new_geom);
        if (cfg.verbose) {
          std::cout << "remesh: rows=" << new_geom.num_rows
                    << " intervals=" << new_geom.num_intervals
                    << " bounds=[" << domain.x_min << "," << domain.x_max
                    << "]x[" << domain.y_min << "," << domain.y_max << "]\n";
        }
      }
    }
  }

  if (cfg.enable_output) {
    export_fields(smoke, temperature, output_dir, cfg.max_steps);
  }

  return 0;
}

} // namespace

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int err = 0;
  {
    err = run(argc, argv);
  }
  Kokkos::finalize();
  return err;
}
