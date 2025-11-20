#include <Kokkos_Core.hpp>

#include "../example_output.hpp"

#include <subsetix/csr_field.hpp>
#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_ops/core.hpp>
#include <subsetix/csr_ops/field_core.hpp>
#include <subsetix/csr_ops/field_stencil.hpp>
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
using subsetix::csr::set_difference_device;
using subsetix::csr::detail::FieldReadAccessor;
using subsetix::csr::detail::build_mask_interval_to_row_mapping;
using subsetix::vtk::write_legacy_quads;

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

void update_solution(const Field2DDevice<Conserved>& U,
                     Field2DDevice<Conserved>& U_next,
                     const IntervalSet2DDevice& geom,
                     const Box2D& domain,
                     FieldReadAccessor<Conserved> acc,
                     const Kokkos::View<int*, subsetix::csr::DeviceMemorySpace>& interval_to_row,
                     const Conserved& inflow,
                     double gamma,
                     double dt,
                     double dx,
                     double dy,
                     bool no_slip) {
  using ExecSpace = Kokkos::DefaultExecutionSpace;

  auto row_keys = geom.row_keys;
  auto intervals = geom.intervals;
  auto offsets = geom.cell_offsets;

  auto values_in = U.values;
  auto values_out = U_next.values;

  Kokkos::parallel_for(
      "mach2_cylinder_update",
      Kokkos::RangePolicy<ExecSpace>(
          0, static_cast<int>(geom.num_intervals)),
      KOKKOS_LAMBDA(const int interval_idx) {
        const int row_idx = interval_to_row(interval_idx);
        const Coord y = row_keys(row_idx).y;
        const auto iv = intervals(interval_idx);
        const std::size_t base = offsets(interval_idx);

        for (Coord x = iv.begin; x < iv.end; ++x) {
          const std::size_t idx =
              base + static_cast<std::size_t>(x - iv.begin);

          const Conserved center = values_in(idx);
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

          values_out(idx) = updated;
        }
      });
}

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
  }
  if (cfg.cx < 0) {
    cfg.cx = cfg.nx / 4;
  }
  if (cfg.cy < 0) {
    cfg.cy = cfg.ny / 2;
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

Conserved build_inflow_state(const RunConfig& cfg) {
  Primitive q;
  q.rho = cfg.rho;
  q.p = cfg.p;
  const double a = std::sqrt(cfg.gamma * q.p / q.rho);
  q.u = cfg.mach_inlet * a;
  q.v = 0.0;
  return prim_to_cons(q, cfg.gamma);
}

void write_step_outputs(const Field2DDevice<Conserved>& U,
                        const IntervalSet2DDevice& fluid_geom,
                        Field2DDevice<double>& density,
                        Field2DDevice<double>& pressure,
                        Field2DDevice<double>& mach,
                        double gamma,
                        const std::filesystem::path& out_dir,
                        int step) {
  compute_diagnostics(U, density, pressure, mach, gamma);

  const auto rho_host = subsetix::csr::build_host_field_from_device(density);
  const auto p_host = subsetix::csr::build_host_field_from_device(pressure);
  const auto m_host = subsetix::csr::build_host_field_from_device(mach);

  write_legacy_quads(rho_host, vtk_filename(out_dir, step, "density"), "rho");
  write_legacy_quads(p_host, vtk_filename(out_dir, step, "pressure"), "p");
  write_legacy_quads(m_host, vtk_filename(out_dir, step, "mach"), "mach");

  (void)fluid_geom;
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
    auto obstacle_dev = make_disk_device(obstacle);

    CsrSetAlgebraContext ctx;
    auto fluid_dev = subsetix::csr::allocate_interval_set_device(
        domain_dev.num_rows,
        domain_dev.num_intervals + obstacle_dev.num_intervals);
    set_difference_device(domain_dev, obstacle_dev, fluid_dev, ctx);
    subsetix::csr::compute_cell_offsets_device(fluid_dev);

    auto interval_to_row = build_mask_interval_to_row_mapping(fluid_dev);

    Field2DDevice<Conserved> U(fluid_dev, "mach2_state");
    Field2DDevice<Conserved> U_next(fluid_dev, "mach2_state_next");
    Field2DDevice<double> density(fluid_dev, "mach2_density");
    Field2DDevice<double> pressure(fluid_dev, "mach2_pressure");
    Field2DDevice<double> mach_field(fluid_dev, "mach2_mach");

    const Conserved inflow = build_inflow_state(cfg);

    // Initialize with inflow everywhere in the fluid domain.
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
    }

    // Export initial geometries
    {
      const IntervalSet2DHost fluid_host =
          subsetix::csr::build_host_from_device(fluid_dev);
      const IntervalSet2DHost obstacle_host =
          subsetix::csr::build_host_from_device(obstacle_dev);
      write_legacy_quads(fluid_host,
                         subsetix_examples::output_file(output_dir, "fluid_geometry.vtk"));
      write_legacy_quads(obstacle_host,
                         subsetix_examples::output_file(output_dir, "obstacle_geometry.vtk"));
    }

    double t = 0.0;
    int step = 0;
    const double dx = 1.0;
    const double dy = 1.0;

    std::cout << "Mach 2 cylinder setup: "
              << "nx=" << cfg.nx << " ny=" << cfg.ny
              << " cx=" << cfg.cx << " cy=" << cfg.cy
              << " r=" << cfg.radius
              << " no-slip=" << (cfg.no_slip ? "yes" : "no")
              << " output_dir=" << output_dir << "\n";

    double total_mass0 = compute_total_mass(U);

    while ((t < cfg.t_final) && (step < cfg.max_steps)) {
      const FieldReadAccessor<Conserved> acc = make_accessor(U);
      double dt = compute_dt(U, cfg.gamma, cfg.cfl, dx, dy);
      if (t + dt > cfg.t_final) {
        dt = cfg.t_final - t;
      }

      update_solution(U, U_next, fluid_dev, domain, acc,
                      interval_to_row, inflow, cfg.gamma,
                      dt, dx, dy, cfg.no_slip);
      Kokkos::DefaultExecutionSpace().fence();

      // Swap
      std::swap(U.values, U_next.values);

      t += dt;
      ++step;

      if (step % cfg.output_stride == 0 || step == cfg.max_steps ||
          t >= cfg.t_final - 1e-12) {
        write_step_outputs(U, fluid_dev, density, pressure, mach_field,
                           cfg.gamma, output_dir, step);
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
