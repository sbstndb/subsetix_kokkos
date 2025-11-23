// Example: 2D advection with adaptive mesh refinement (2 levels, global fine dt)
// The fine grid follows a moving square by thresholding the coarse solution
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <chrono>
#include <utility>
#include <sstream>
#include <string>

#include <Kokkos_Core.hpp>

#include <subsetix/field/csr_field.hpp>
#include <subsetix/field/csr_field_ops.hpp>
#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>
#include <subsetix/csr_ops/amr.hpp>
#include <subsetix/csr_ops/field_amr.hpp>
#include <subsetix/csr_ops/field_stencil.hpp>
#include <subsetix/csr_ops/field_subview.hpp>
#include <subsetix/csr_ops/morphology.hpp>
#include <subsetix/csr_ops/threshold.hpp>
#include <subsetix/multilevel/multilevel.hpp>
#include <subsetix/io/vtk_export.hpp>

using namespace subsetix;
using namespace subsetix::csr;
using subsetix::csr::ExecSpace;

template <typename T>
detail::FieldReadAccessor<T>
make_accessor(const Field2DDevice<T>& field) {
  detail::FieldReadAccessor<T> acc;
  acc.row_keys = field.geometry.row_keys;
  acc.row_ptr = field.geometry.row_ptr;
  acc.intervals = field.geometry.intervals;
  acc.offsets = field.geometry.cell_offsets;
  acc.values = field.values;
  acc.num_rows = field.geometry.num_rows;
  return acc;
}

template <typename T>
Field2DDevice<T> make_field_like_device(const IntervalSet2DDevice& geom,
                                        T init_val = T()) {
  auto geom_host = build_host_from_device(geom);
  auto field_host = make_field_like_geometry<T>(geom_host, init_val);
  return build_device_field_from_host(field_host);
}

void prolong_by_coords(Field2DDevice<double>& fine_field,
                       const Field2DDevice<double>& coarse_field) {
  if (fine_field.geometry.num_rows == 0) {
    return;
  }

  const auto fine_rows = fine_field.geometry.row_keys;
  const auto fine_row_ptr = fine_field.geometry.row_ptr;
  const auto fine_intervals = fine_field.geometry.intervals;
  const auto fine_offsets = fine_field.geometry.cell_offsets;
  auto fine_values = fine_field.values;

  const auto coarse_acc = make_accessor(coarse_field);

  Kokkos::parallel_for(
      "subsetix_prolong_by_coords",
      Kokkos::RangePolicy<ExecSpace>(
          0, static_cast<int>(fine_field.geometry.num_rows)),
      KOKKOS_LAMBDA(const int row_idx) {
        const Coord y_f = fine_rows(row_idx).y;
        const Coord y_c = detail::floor_div2(y_f);

        const std::size_t begin = fine_row_ptr(row_idx);
        const std::size_t end = fine_row_ptr(row_idx + 1);
        for (std::size_t k = begin; k < end; ++k) {
          const Interval iv = fine_intervals(k);
          const std::size_t base_offset = fine_offsets(k);
          for (Coord x = iv.begin; x < iv.end; ++x) {
            const Coord x_c = detail::floor_div2(x);
            const std::size_t idx =
                base_offset + static_cast<std::size_t>(x - iv.begin);
            fine_values(idx) = coarse_acc.value_at(x_c, y_c);
          }
        }
      });
  ExecSpace().fence();
}

void restrict_by_coords(Field2DDevice<double>& coarse_field,
                        const Field2DDevice<double>& fine_field,
                        const IntervalSet2DDevice& coarse_region) {
  if (coarse_region.num_rows == 0 || coarse_region.num_intervals == 0) {
    return;
  }

  const auto mapping =
      detail::build_mask_field_mapping(coarse_field, coarse_region);
  auto interval_to_row = mapping.interval_to_row;
  auto interval_to_field_interval = mapping.interval_to_field_interval;
  auto row_map = mapping.row_map;

  const auto mask_row_keys = coarse_region.row_keys;
  const auto mask_intervals = coarse_region.intervals;

  const auto coarse_intervals = coarse_field.geometry.intervals;
  const auto coarse_offsets = coarse_field.geometry.cell_offsets;
  auto coarse_values = coarse_field.values;

  const auto fine_acc = make_accessor(fine_field);

  Kokkos::parallel_for(
      "subsetix_restrict_by_coords",
      Kokkos::RangePolicy<ExecSpace>(
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

          const double v00 = fine_acc.value_at(x_f0, y_f0);
          const double v01 = fine_acc.value_at(x_f1, y_f0);
          const double v10 = fine_acc.value_at(x_f0, y_f1);
          const double v11 = fine_acc.value_at(x_f1, y_f1);

          coarse_values(c_idx) =
              0.25 * (v00 + v01 + v10 + v11);
        }
      });
  ExecSpace().fence();
}

struct AdvectionStencil {
  double vx = 0.0;
  double vy = 0.0;
  double dt = 0.0;
  double dx = 1.0;
  double dy = 1.0;

  KOKKOS_INLINE_FUNCTION
  double operator()(Coord x, Coord /*y*/, std::size_t linear_index,
                    int interval_idx,
                    const detail::FieldStencilContext<double>& ctx) const {
    const Interval iv = ctx.intervals(interval_idx);
    const bool has_left = (x > iv.begin);
    const bool has_right = (x + 1 < iv.end);

    const double u_center = ctx.center(linear_index);

    double u_left = u_center;
    double u_right = u_center;
    if (has_left) {
      u_left = ctx.left(linear_index);
    }
    if (has_right) {
      u_right = ctx.right(linear_index);
    }

    // Vertical neighbors: guard against missing rows
    double u_down = u_center;
    double u_up = u_center;

    const int down_idx = ctx.down_interval(interval_idx);
    if (down_idx >= 0) {
      const Interval iv_down = ctx.intervals(down_idx);
      if (x >= iv_down.begin && x < iv_down.end) {
        u_down = ctx.south(x, interval_idx);
      }
    }

    const int up_idx = ctx.up_interval(interval_idx);
    if (up_idx >= 0) {
      const Interval iv_up = ctx.intervals(up_idx);
      if (x >= iv_up.begin && x < iv_up.end) {
        u_up = ctx.north(x, interval_idx);
      }
    }

    double flux_x = 0.0;
    if (vx >= 0.0) {
      flux_x = (u_center - u_left) / dx;
    } else {
      flux_x = (u_right - u_center) / dx;
    }

    double flux_y = 0.0;
    if (vy >= 0.0) {
      flux_y = (u_center - u_down) / dy;
    } else {
      flux_y = (u_up - u_center) / dy;
    }

    return u_center - dt * (vx * flux_x + vy * flux_y);
  }
};

double compute_stable_dt(double dx, double dy, double vx, double vy,
                         double cfl) {
  double inv_dt = 0.0;
  if (vx != 0.0) {
    inv_dt += std::abs(vx) / dx;
  }
  if (vy != 0.0) {
    inv_dt += std::abs(vy) / dy;
  }
  if (inv_dt == 0.0) {
    return std::numeric_limits<double>::max();
  }
  return cfl / inv_dt;
}

IntervalSet2DDevice build_coarse_geometry(Coord nx, Coord ny) {
  Box2D domain;
  domain.x_min = 0;
  domain.x_max = nx;
  domain.y_min = 0;
  domain.y_max = ny;
  return make_box_device(domain);
}

void initialize_square(Field2DDevice<double>& field, double origin_x,
                       double origin_y, double dx, double dy, double size_x,
                       double size_y) {
  const double x0 = origin_x + 0.05;
  const double y0 = origin_y + 0.05;
  const double x1 = x0 + size_x;
  const double y1 = y0 + size_y;

  auto row_keys = field.geometry.row_keys;
  auto row_ptr = field.geometry.row_ptr;
  auto intervals = field.geometry.intervals;
  auto offsets = field.geometry.cell_offsets;
  auto values = field.values;

  Kokkos::parallel_for(
      "subsetix_init_square",
      Kokkos::RangePolicy<ExecSpace>(
          0, static_cast<int>(field.geometry.num_rows)),
      KOKKOS_LAMBDA(const int i) {
        const Coord y_idx = row_keys(i).y;
        const double y_center = origin_y + (static_cast<double>(y_idx) + 0.5) * dy;

        const std::size_t begin = row_ptr(i);
        const std::size_t end = row_ptr(i + 1);

        for (std::size_t k = begin; k < end; ++k) {
          const Interval iv = intervals(k);
          const std::size_t base_offset = offsets(k);
          for (Coord x_idx = iv.begin; x_idx < iv.end; ++x_idx) {
            const double x_center =
                origin_x + (static_cast<double>(x_idx) + 0.5) * dx;
            const bool inside =
                (x_center >= x0 && x_center <= x1 && y_center >= y0 &&
                 y_center <= y1);
            const std::size_t linear_index =
                base_offset + static_cast<std::size_t>(x_idx - iv.begin);
            values(linear_index) = inside ? 1.0 : 0.0;
          }
        }
      });
  ExecSpace().fence();
}

IntervalSet2DDevice build_refine_mask(const Field2DDevice<double>& coarse_field,
                                      double threshold, Coord grow,
                                      CsrSetAlgebraContext& ctx) {
  IntervalSet2DDevice mask = threshold_field(coarse_field, threshold);
  IntervalSet2DDevice expanded;
  expand_device(mask, grow, grow, expanded, ctx);
  return expanded;
}

IntervalSet2DDevice build_fine_geometry(const IntervalSet2DDevice& coarse_geo,
                                        const IntervalSet2DDevice& coarse_mask,
                                        CsrSetAlgebraContext& ctx) {
  IntervalSet2DDevice fine_full;
  refine_level_up_device(coarse_geo, fine_full, ctx);

  IntervalSet2DDevice fine_mask;
  refine_level_up_device(coarse_mask, fine_mask, ctx);

  IntervalSet2DDevice fine_geo =
      allocate_interval_set_device(fine_full.num_rows, fine_full.num_intervals);
  set_intersection_device(fine_full, fine_mask, fine_geo, ctx);
  return fine_geo;
}

void prolong_full(Field2DDevice<double>& fine_field,
                  const Field2DDevice<double>& coarse_field,
                  CsrSetAlgebraContext& ctx) {
  (void)ctx;
  prolong_by_coords(fine_field, coarse_field);
}

void copy_overlap(Field2DDevice<double>& fine_dst,
                  Field2DDevice<double>& fine_src,
                  const IntervalSet2DDevice& overlap,
                  CsrSetAlgebraContext& ctx) {
  if (overlap.num_rows == 0 || overlap.num_intervals == 0) {
    return;
  }
  auto sub_dst = make_subview(fine_dst, overlap, "overlap_dst");
  auto sub_src = make_subview(fine_src, overlap, "overlap_src");
  copy_subview_device(sub_dst, sub_src, ctx);
}

void write_vtk_step(int step, const MultilevelGeoDevice& geo,
                    const Field2DDevice<double>& u0,
                    const Field2DDevice<double>& u1) {
  MultilevelFieldDevice<double> field;
  field.num_active_levels = 2;
  field.levels[0] = u0;
  field.levels[1] = u1;

  auto host_geo = deep_copy_to_host(geo);
  auto host_field = deep_copy_to_host(field);

  std::ostringstream geo_name;
  geo_name << "amr_advection_geo_step" << std::setfill('0') << std::setw(4)
           << step << ".vtk";
  std::ostringstream field_name;
  field_name << "amr_advection_field_step" << std::setfill('0') << std::setw(4)
             << step << ".vtk";

  subsetix::vtk::write_multilevel_vtk(host_geo, geo_name.str());
  subsetix::vtk::write_multilevel_field_vtk(host_field, host_geo,
                                            field_name.str(), "u");
}

void run_demo(bool write_vtk) {
  using Clock = std::chrono::high_resolution_clock;

  const Coord nx = 8192;
  const Coord ny = 8192;
  const double domain_length = 1.0;

  const double vx = 0.75;
  const double vy = 0.55;
  const double cfl = 0.4;

  const double refine_threshold = 0.05;
  const Coord grow = 2;

  const int num_steps = 50;
  const int output_every = 10;

  CsrSetAlgebraContext ctx;

  MultilevelGeoDevice geo;
  geo.origin_x = 0.0;
  geo.origin_y = 0.0;
  geo.root_dx = domain_length / static_cast<double>(nx);
  geo.root_dy = domain_length / static_cast<double>(ny);
  geo.num_active_levels = 2;

  // Build sparse coarse geometry: full box minus a central hole
  IntervalSet2DDevice coarse_full = build_coarse_geometry(nx, ny);
  Disk2D hole;
  hole.cx = nx / 2;
  hole.cy = ny / 2;
  hole.radius = nx / 8;
  IntervalSet2DDevice hole_set = make_disk_device(hole);
  geo.levels[0] = allocate_interval_set_device(
      coarse_full.num_rows, coarse_full.num_intervals + hole_set.num_intervals);
  set_difference_device(coarse_full, hole_set, geo.levels[0], ctx);

  Field2DDevice<double> u0 = make_field_like_device<double>(geo.levels[0], 0.0);
  Field2DDevice<double> u0_tmp =
      make_field_like_device<double>(geo.levels[0], 0.0);

  initialize_square(u0, geo.origin_x, geo.origin_y, geo.root_dx, geo.root_dy,
                    0.18, 0.18);

  IntervalSet2DDevice coarse_mask =
      build_refine_mask(u0, refine_threshold, grow, ctx);
  IntervalSet2DDevice fine_geo =
      build_fine_geometry(geo.levels[0], coarse_mask, ctx);

  geo.levels[1] = fine_geo;

  Field2DDevice<double> u1 =
      make_field_like_device<double>(geo.levels[1], 0.0);
  Field2DDevice<double> u1_tmp =
      make_field_like_device<double>(geo.levels[1], 0.0);

  prolong_full(u1, u0, ctx);

  const double dx0 = geo.root_dx;
  const double dy0 = geo.root_dy;
  const double dx1 = geo.root_dx * 0.5;
  const double dy1 = geo.root_dy * 0.5;
  const double dt = compute_stable_dt(dx1, dy1, vx, vy, cfl);

  std::cout << "Running AMR advection demo\n";
  std::cout << "Grid coarse: " << nx << " x " << ny << "\n";
  std::cout << "Initial fine rows: " << geo.levels[1].num_rows << "\n";
  std::cout << "dt (fine-based): " << dt << "\n";

  IntervalSet2DDevice projection_fine_on_coarse;
  IntervalSet2DDevice fine_geo_new;
  IntervalSet2DDevice overlap;

  auto t_start = Clock::now();
  double t_step_sum = 0.0;
  double t_adapt_sum = 0.0;
  double t_stencil_coarse = 0.0;
  double t_stencil_fine = 0.0;
  double t_restrict = 0.0;
  double t_mask = 0.0;
  double t_build_fine = 0.0;
  double t_prolong = 0.0;
  double t_overlap = 0.0;
  double t_allocate = 0.0;

  if (write_vtk) {
    write_vtk_step(0, geo, u0, u1);
  }

  for (int step = 1; step <= num_steps; ++step) {
    auto t_step_begin = Clock::now();
    const AdvectionStencil coarse_stencil{vx, vy, dt, dx0, dy0};
    const AdvectionStencil fine_stencil{vx, vy, dt, dx1, dy1};

    auto t0 = Clock::now();
    apply_stencil_on_set_device(u0_tmp, u0, geo.levels[0], coarse_stencil);
    auto t1 = Clock::now();
    t_stencil_coarse += std::chrono::duration<double>(t1 - t0).count();

    if (geo.levels[1].num_rows > 0 && geo.levels[1].num_intervals > 0) {
      auto t_fine0 = Clock::now();
      auto sub_src = make_subview(u1, geo.levels[1], "fine_src");
      auto sub_dst = make_subview(u1_tmp, geo.levels[1], "fine_dst");
      apply_stencil_on_subview_device(sub_dst, sub_src, fine_stencil, ctx);
      auto t_fine1 = Clock::now();
      t_stencil_fine += std::chrono::duration<double>(t_fine1 - t_fine0).count();
    }

    std::swap(u0, u0_tmp);
    std::swap(u1, u1_tmp);

    if (geo.levels[1].num_rows > 0 && geo.levels[1].num_intervals > 0) {
      auto t_r0 = Clock::now();
      project_level_down_device(geo.levels[1], projection_fine_on_coarse, ctx);
      restrict_by_coords(u0, u1, projection_fine_on_coarse);
      auto t_r1 = Clock::now();
      t_restrict += std::chrono::duration<double>(t_r1 - t_r0).count();
    }

    if (write_vtk && (step % output_every == 0)) {
      write_vtk_step(step, geo, u0, u1);
    }

    // Adaptivity: rebuild fine grid following the square
    auto t_mask0 = Clock::now();
    coarse_mask = build_refine_mask(u0, refine_threshold, grow, ctx);
    auto t_mask1 = Clock::now();
    t_mask += std::chrono::duration<double>(t_mask1 - t_mask0).count();

    auto t_build0 = Clock::now();
    fine_geo_new = build_fine_geometry(geo.levels[0], coarse_mask, ctx);
    auto t_build1 = Clock::now();
    t_build_fine += std::chrono::duration<double>(t_build1 - t_build0).count();

    auto t_alloc0 = Clock::now();
    Field2DDevice<double> u1_new =
        make_field_like_device<double>(fine_geo_new, 0.0);
    auto t_alloc1 = Clock::now();
    t_allocate += std::chrono::duration<double>(t_alloc1 - t_alloc0).count();

    auto t_prol0 = Clock::now();
    prolong_full(u1_new, u0, ctx);
    auto t_prol1 = Clock::now();
    t_prolong += std::chrono::duration<double>(t_prol1 - t_prol0).count();

    auto t_adapt_begin = std::chrono::high_resolution_clock::now();
    if (geo.levels[1].num_rows > 0 && fine_geo_new.num_rows > 0 &&
        geo.levels[1].num_intervals > 0 && fine_geo_new.num_intervals > 0) {
      auto t_ov0 = Clock::now();
      const std::size_t overlap_rows_cap =
          std::min(geo.levels[1].num_rows, fine_geo_new.num_rows);
      const std::size_t overlap_intervals_cap =
          std::min(geo.levels[1].num_intervals, fine_geo_new.num_intervals);
      overlap =
          allocate_interval_set_device(overlap_rows_cap, overlap_intervals_cap);
      set_intersection_device(geo.levels[1], fine_geo_new, overlap, ctx);
      copy_overlap(u1_new, u1, overlap, ctx);
      auto t_ov1 = Clock::now();
      t_overlap += std::chrono::duration<double>(t_ov1 - t_ov0).count();
    }
    auto t_adapt_end = Clock::now();

    geo.levels[1] = fine_geo_new;
    u1 = u1_new;
    u1_tmp = make_field_like_device<double>(geo.levels[1], 0.0);

    auto t_step_end = Clock::now();
    t_adapt_sum += std::chrono::duration<double>(t_adapt_end - t_adapt_begin).count();
    t_step_sum += std::chrono::duration<double>(t_step_end - t_step_begin).count();
  }

  auto t_end = Clock::now();
  double t_total = std::chrono::duration<double>(t_end - t_start).count();

  std::cout << "Timing (wall): total=" << t_total << "s"
            << " step_sum=" << t_step_sum << "s"
            << " adapt_sum=" << t_adapt_sum << "s"
            << " stencil_coarse=" << t_stencil_coarse << "s"
            << " stencil_fine=" << t_stencil_fine << "s"
            << " restrict=" << t_restrict << "s"
            << " mask=" << t_mask << "s"
            << " build_fine=" << t_build_fine << "s"
            << " prolong=" << t_prolong << "s"
            << " overlap=" << t_overlap << "s"
            << " allocate=" << t_allocate << "s"
            << std::endl;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int err = 0;
  try {
    bool write_vtk = false;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--vtk") {
        write_vtk = true;
      } else if (arg == "--no-vtk") {
        write_vtk = false;
      }
    }

    run_demo(write_vtk);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    err = 1;
  }
  Kokkos::finalize();
  return err;
}
