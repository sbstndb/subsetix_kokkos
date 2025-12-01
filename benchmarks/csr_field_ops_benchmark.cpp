#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>
#include <chrono>
#include <vector>

#include <subsetix/field/csr_field.hpp>
#include <subsetix/field/csr_field_ops.hpp>
#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>
#include <subsetix/csr_ops/amr.hpp>
#include <subsetix/csr_ops/workspace.hpp>

using namespace subsetix::csr;

namespace {

struct RectConfig {
  Coord x_min;
  Coord x_max;
  Coord y_min;
  Coord y_max;
};

RectConfig make_rect(std::size_t extent) {
  return RectConfig{0,
                    static_cast<Coord>(extent),
                    0,
                    static_cast<Coord>(extent)};
}

std::size_t cells_in_rect(const RectConfig& cfg) {
  const std::size_t width =
      static_cast<std::size_t>(cfg.x_max - cfg.x_min);
  const std::size_t height =
      static_cast<std::size_t>(cfg.y_max - cfg.y_min);
  return width * height;
}

IntervalSet2DDevice make_mask(const RectConfig& cfg) {
  Box2D box;
  box.x_min = cfg.x_min;
  box.x_max = cfg.x_max;
  box.y_min = cfg.y_min;
  box.y_max = cfg.y_max;
  return make_box_device(box);
}

RectConfig interior_rect(const RectConfig& cfg) {
  RectConfig r = cfg;
  if (r.x_max - r.x_min > 2) {
    ++r.x_min;
    --r.x_max;
  }
  if (r.y_max - r.y_min > 2) {
    ++r.y_min;
    --r.y_max;
  }
  return r;
}

Field2DDevice<double> make_field(
    const RectConfig& cfg, double init_value) {
  auto geom = make_mask(cfg);
  auto geom_host = to<HostMemorySpace>(geom);
  auto field_host =
      make_field_like_geometry<double>(geom_host, init_value);
  return build_device_field_from_host(field_host);
}

template <class Op>
void bench_field_op(benchmark::State& state,
                    const RectConfig& geom_cfg,
                    const RectConfig& mask_cfg,
                    Op op) {
  Field2DDevice<double> field =
      make_field(geom_cfg, 0.0);
  IntervalSet2DDevice mask = make_mask(mask_cfg);
  const std::size_t cells = cells_in_rect(mask_cfg);
  if (cells == 0) {
    state.SkipWithError("Mask has zero cells");
    return;
  }

  double total_seconds = 0.0;

  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    op(field, mask);
    const auto t1 = std::chrono::steady_clock::now();
    total_seconds +=
        std::chrono::duration<double>(t1 - t0).count();

    benchmark::DoNotOptimize(field.values.data());
  }

  const double total_cells =
      static_cast<double>(cells) *
      static_cast<double>(state.iterations());
  state.counters["ns_per_cell"] =
      (total_seconds / total_cells) * 1e9;
}

template <class Op>
void bench_subview_op(benchmark::State& state,
                      const RectConfig& geom_cfg,
                      const RectConfig& mask_cfg,
                      Op op,
                      bool use_subset = false) {
  Field2DDevice<double> field = make_field(geom_cfg, 0.0);
  IntervalSet2DDevice mask = make_mask(mask_cfg);
  const std::size_t cells = cells_in_rect(mask_cfg);
  if (cells == 0) {
    state.SkipWithError("SubView mask has zero cells");
    return;
  }

  auto sub = make_subview(field, mask, "bench_subview");
  CsrSetAlgebraContext ctx;
  if (use_subset) {
    subsetix::csr::detail::ensure_subview_subset(sub, &ctx);
  }
  double total_seconds = 0.0;

  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    op(sub);
    const auto t1 = std::chrono::steady_clock::now();
    total_seconds +=
        std::chrono::duration<double>(t1 - t0).count();
    benchmark::DoNotOptimize(field.values.data());
  }

  const double total_cells =
      static_cast<double>(cells) *
      static_cast<double>(state.iterations());
  state.counters["ns_per_cell"] =
      (total_seconds / total_cells) * 1e9;
}

struct FivePointAverage {
  KOKKOS_INLINE_FUNCTION
  double operator()(Coord x,
                    Coord y,
                    std::size_t idx,
                    int interval_index,
                    const detail::FieldStencilContext<double>& ctx) const {
    const double center = ctx.center(idx);
    const double east = ctx.right(idx);
    const double west = ctx.left(idx);
    const double north = ctx.north(x, interval_index);
    const double south = ctx.south(x, interval_index);
    return (center + east + west + north + south) / 5.0;
  }
};

void bench_stencil(benchmark::State& state,
                   const RectConfig& geom_cfg,
                   const RectConfig& mask_cfg) {
  Field2DDevice<double> input =
      make_field(geom_cfg, 0.0);
  Field2DDevice<double> output =
      make_field(geom_cfg, 0.0);
  IntervalSet2DDevice mask = make_mask(mask_cfg);
  const std::size_t cells = cells_in_rect(mask_cfg);

  // Initialize input with a simple pattern on host for repeatability.
  auto input_host = build_host_field_from_device(input);
  for (std::size_t row = 0; row < input_host.row_keys.size();
       ++row) {
    const Coord y = input_host.row_keys[row].y;
    const std::size_t begin = input_host.row_ptr[row];
    const std::size_t end = input_host.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = input_host.intervals[k];
      for (Coord x = fi.begin; x < fi.end; ++x) {
        const std::size_t offset =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        input_host.values[offset] =
            static_cast<double>(x + 10 * y);
      }
    }
  }
  input = build_device_field_from_host(input_host);

  double total_seconds = 0.0;

  for (auto _ : state) {
    Kokkos::deep_copy(output.values, 0.0);
    const auto t0 = std::chrono::steady_clock::now();
    apply_stencil_on_set_device(output, input, mask,
                                FivePointAverage{});
    const auto t1 = std::chrono::steady_clock::now();
    total_seconds +=
        std::chrono::duration<double>(t1 - t0).count();

    benchmark::DoNotOptimize(output.values.data());
  }

  const double total_cells =
      static_cast<double>(cells) *
      static_cast<double>(state.iterations());
  state.counters["ns_per_cell"] =
      (total_seconds / total_cells) * 1e9;
}

void bench_stencil_repeated(benchmark::State& state,
                            const RectConfig& geom_cfg,
                            const RectConfig& mask_cfg,
                            int repeats) {
  Field2DDevice<double> input =
      make_field(geom_cfg, 0.0);
  Field2DDevice<double> output =
      make_field(geom_cfg, 0.0);
  IntervalSet2DDevice mask = make_mask(mask_cfg);
  const std::size_t cells = cells_in_rect(mask_cfg);

  // Initialize input with a simple pattern on host for repeatability.
  auto input_host = build_host_field_from_device(input);
  for (std::size_t row = 0; row < input_host.row_keys.size();
       ++row) {
    const Coord y = input_host.row_keys[row].y;
    const std::size_t begin = input_host.row_ptr[row];
    const std::size_t end = input_host.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = input_host.intervals[k];
      for (Coord x = fi.begin; x < fi.end; ++x) {
        const std::size_t offset =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        input_host.values[offset] =
            static_cast<double>(x + 10 * y);
      }
    }
  }
  input = build_device_field_from_host(input_host);

  double total_seconds = 0.0;

  for (auto _ : state) {
    // Ensure output starts from a clean buffer.
    Kokkos::deep_copy(output.values, 0.0);

    auto* src = &input;
    auto* dst = &output;

    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) {
      apply_stencil_on_set_device(*dst, *src, mask,
                                  FivePointAverage{});
      auto* tmp = src;
      src = dst;
      dst = tmp;
    }
    const auto t1 = std::chrono::steady_clock::now();
    total_seconds +=
        std::chrono::duration<double>(t1 - t0).count();

    benchmark::DoNotOptimize(src->values.data());
  }

  const double total_cells =
      static_cast<double>(cells) *
      static_cast<double>(repeats) *
      static_cast<double>(state.iterations());
  state.counters["ns_per_cell"] =
      (total_seconds / total_cells) * 1e9;
}

void bench_subview_stencil(benchmark::State& state,
                           const RectConfig& geom_cfg,
                           const RectConfig& mask_cfg,
                           bool use_subset = false) {
  Field2DDevice<double> input =
      make_field(geom_cfg, 0.0);
  Field2DDevice<double> output =
      make_field(geom_cfg, 0.0);
  IntervalSet2DDevice mask = make_mask(mask_cfg);
  const std::size_t cells = cells_in_rect(mask_cfg);
  if (cells == 0) {
    state.SkipWithError("SubView stencil mask has zero cells");
    return;
  }

  auto input_host = build_host_field_from_device(input);
  for (std::size_t row = 0; row < input_host.row_keys.size();
       ++row) {
    const Coord y = input_host.row_keys[row].y;
    const std::size_t begin = input_host.row_ptr[row];
    const std::size_t end = input_host.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = input_host.intervals[k];
      for (Coord x = fi.begin; x < fi.end; ++x) {
        const std::size_t offset =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        input_host.values[offset] =
            static_cast<double>(x + 10 * y);
      }
    }
  }
  input = build_device_field_from_host(input_host);

  auto src = make_subview(input, mask, "subview_stencil_src");
  auto dst =
      make_subview(output, mask, "subview_stencil_dst");

  CsrSetAlgebraContext ctx;
  if (use_subset) {
    subsetix::csr::detail::ensure_subview_subset(dst, &ctx);
    subsetix::csr::detail::ensure_subview_subset(src, &ctx);
  }

  double total_seconds = 0.0;

  for (auto _ : state) {
    Kokkos::deep_copy(output.values, 0.0);
    const auto t0 = std::chrono::steady_clock::now();
    apply_stencil_on_subview_device(dst, src,
                                    FivePointAverage{});
    const auto t1 = std::chrono::steady_clock::now();
    total_seconds +=
        std::chrono::duration<double>(t1 - t0).count();

    benchmark::DoNotOptimize(output.values.data());
  }

  const double total_cells =
      static_cast<double>(cells) *
      static_cast<double>(state.iterations());
  state.counters["ns_per_cell"] =
      (total_seconds / total_cells) * 1e9;
}

void bench_subview_stencil_repeated(benchmark::State& state,
                                    const RectConfig& geom_cfg,
                                    const RectConfig& mask_cfg,
                                    bool use_subset,
                                    int repeats) {
  Field2DDevice<double> input =
      make_field(geom_cfg, 0.0);
  Field2DDevice<double> output =
      make_field(geom_cfg, 0.0);
  IntervalSet2DDevice mask = make_mask(mask_cfg);
  const std::size_t cells = cells_in_rect(mask_cfg);
  if (cells == 0) {
    state.SkipWithError("SubView stencil repeated mask has zero cells");
    return;
  }

  auto input_host = build_host_field_from_device(input);
  for (std::size_t row = 0; row < input_host.row_keys.size();
       ++row) {
    const Coord y = input_host.row_keys[row].y;
    const std::size_t begin = input_host.row_ptr[row];
    const std::size_t end = input_host.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = input_host.intervals[k];
      for (Coord x = fi.begin; x < fi.end; ++x) {
        const std::size_t offset =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        input_host.values[offset] =
            static_cast<double>(x + 10 * y);
      }
    }
  }
  input = build_device_field_from_host(input_host);

  auto src = make_subview(input, mask, "subview_stencil_rep_src");
  auto dst =
      make_subview(output, mask, "subview_stencil_rep_dst");

  CsrSetAlgebraContext ctx;
  if (use_subset) {
    subsetix::csr::detail::ensure_subview_subset(dst, &ctx);
    subsetix::csr::detail::ensure_subview_subset(src, &ctx);
  }

  double total_seconds = 0.0;

  for (auto _ : state) {
    Kokkos::deep_copy(output.values, 0.0);

    auto* src_view = &src;
    auto* dst_view = &dst;

    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) {
      apply_stencil_on_subview_device(*dst_view, *src_view,
                                      FivePointAverage{});
      auto* tmp = src_view;
      src_view = dst_view;
      dst_view = tmp;
    }
    const auto t1 = std::chrono::steady_clock::now();
    total_seconds +=
        std::chrono::duration<double>(t1 - t0).count();

    benchmark::DoNotOptimize(src_view->parent.values.data());
  }

  const double total_cells =
      static_cast<double>(cells) *
      static_cast<double>(repeats) *
      static_cast<double>(state.iterations());
  state.counters["ns_per_cell"] =
      (total_seconds / total_cells) * 1e9;
}

void bench_restriction(benchmark::State& state,
                       const RectConfig& coarse_cfg) {
  Field2DDevice<double> coarse =
      make_field(coarse_cfg, 0.0);
  IntervalSet2DDevice coarse_mask = make_mask(coarse_cfg);

  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice fine_geom;
  refine_level_up_device(coarse_mask, fine_geom, ctx);
  auto fine_geom_host = to<HostMemorySpace>(fine_geom);
  auto fine_field_host =
      make_field_like_geometry<double>(fine_geom_host, 0.0);
  for (std::size_t row = 0;
       row < fine_field_host.row_keys.size(); ++row) {
    const Coord y = fine_field_host.row_keys[row].y;
    const std::size_t begin = fine_field_host.row_ptr[row];
    const std::size_t end = fine_field_host.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = fine_field_host.intervals[k];
      for (Coord x = fi.begin; x < fi.end; ++x) {
        const std::size_t offset =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        fine_field_host.values[offset] =
            static_cast<double>(x + 10 * y);
      }
    }
  }
  Field2DDevice<double> fine =
      build_device_field_from_host(fine_field_host);

  const std::size_t cells = cells_in_rect(coarse_cfg);
  double total_seconds = 0.0;

  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    restrict_field_on_set_device(coarse, fine, coarse_mask);
    const auto t1 = std::chrono::steady_clock::now();
    total_seconds +=
        std::chrono::duration<double>(t1 - t0).count();
    benchmark::DoNotOptimize(coarse.values.data());
  }

  const double total_cells =
      static_cast<double>(cells) *
      static_cast<double>(state.iterations());
  state.counters["ns_per_cell"] =
      (total_seconds / total_cells) * 1e9;
}

void bench_subview_restriction(benchmark::State& state,
                               const RectConfig& coarse_cfg,
                               bool use_subset = false) {
  Field2DDevice<double> coarse =
      make_field(coarse_cfg, 0.0);
  IntervalSet2DDevice coarse_mask = make_mask(coarse_cfg);
  auto coarse_sub =
      make_subview(coarse, coarse_mask, "subview_restrict");

  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice fine_geom;
  refine_level_up_device(coarse_mask, fine_geom, ctx);
  auto fine_geom_host = to<HostMemorySpace>(fine_geom);
  auto fine_field_host =
      make_field_like_geometry<double>(fine_geom_host, 0.0);
  for (std::size_t row = 0;
       row < fine_field_host.row_keys.size(); ++row) {
    const Coord y = fine_field_host.row_keys[row].y;
    const std::size_t begin = fine_field_host.row_ptr[row];
    const std::size_t end = fine_field_host.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = fine_field_host.intervals[k];
      for (Coord x = fi.begin; x < fi.end; ++x) {
        const std::size_t offset =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        fine_field_host.values[offset] =
            static_cast<double>(x + 10 * y);
      }
    }
  }
  Field2DDevice<double> fine =
      build_device_field_from_host(fine_field_host);

  const std::size_t cells = cells_in_rect(coarse_cfg);
  if (use_subset) {
    subsetix::csr::detail::ensure_subview_subset(coarse_sub, &ctx);
  }
  double total_seconds = 0.0;

  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    restrict_field_subview_device(coarse_sub, fine);
    const auto t1 = std::chrono::steady_clock::now();
    total_seconds +=
        std::chrono::duration<double>(t1 - t0).count();
    benchmark::DoNotOptimize(coarse.values.data());
  }

  const double total_cells =
      static_cast<double>(cells) *
      static_cast<double>(state.iterations());
  state.counters["ns_per_cell"] =
      (total_seconds / total_cells) * 1e9;
}

void bench_prolongation(benchmark::State& state,
                        const RectConfig& coarse_cfg) {
  Field2DDevice<double> coarse =
      make_field(coarse_cfg, 0.0);
  auto coarse_host = build_host_field_from_device(coarse);
  for (std::size_t row = 0; row < coarse_host.row_keys.size();
       ++row) {
    const Coord y = coarse_host.row_keys[row].y;
    const std::size_t begin = coarse_host.row_ptr[row];
    const std::size_t end = coarse_host.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = coarse_host.intervals[k];
      for (Coord x = fi.begin; x < fi.end; ++x) {
        const std::size_t offset =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        coarse_host.values[offset] =
            static_cast<double>(100 + 3 * x + 5 * y);
      }
    }
  }
  coarse = build_device_field_from_host(coarse_host);

  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice fine_geom;
  refine_level_up_device(make_mask(coarse_cfg), fine_geom, ctx);
  auto fine_geom_host = to<HostMemorySpace>(fine_geom);
  auto fine_field_host =
      make_field_like_geometry<double>(fine_geom_host, 0.0);
  Field2DDevice<double> fine =
      build_device_field_from_host(fine_field_host);

  IntervalSet2DDevice fine_mask = fine_geom;
  const std::size_t cells =
      cells_in_rect(RectConfig{
          0,
          static_cast<Coord>(2 * (coarse_cfg.x_max - coarse_cfg.x_min)),
          0,
          static_cast<Coord>(2 * (coarse_cfg.y_max - coarse_cfg.y_min))});

  double total_seconds = 0.0;

  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    prolong_field_on_set_device(fine, coarse, fine_mask);
    const auto t1 = std::chrono::steady_clock::now();
    total_seconds +=
        std::chrono::duration<double>(t1 - t0).count();
    benchmark::DoNotOptimize(fine.values.data());
  }

  const double total_cells =
      static_cast<double>(cells) *
      static_cast<double>(state.iterations());
  state.counters["ns_per_cell"] =
      (total_seconds / total_cells) * 1e9;
}

void bench_subview_prolongation(benchmark::State& state,
                                const RectConfig& coarse_cfg,
                                bool use_subset = false) {
  Field2DDevice<double> coarse =
      make_field(coarse_cfg, 0.0);
  auto coarse_host = build_host_field_from_device(coarse);
  for (std::size_t row = 0; row < coarse_host.row_keys.size();
       ++row) {
    const Coord y = coarse_host.row_keys[row].y;
    const std::size_t begin = coarse_host.row_ptr[row];
    const std::size_t end = coarse_host.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = coarse_host.intervals[k];
      for (Coord x = fi.begin; x < fi.end; ++x) {
        const std::size_t offset =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        coarse_host.values[offset] =
            static_cast<double>(100 + 3 * x + 5 * y);
      }
    }
  }
  coarse = build_device_field_from_host(coarse_host);

  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice fine_geom;
  refine_level_up_device(make_mask(coarse_cfg), fine_geom, ctx);
  auto fine_geom_host = to<HostMemorySpace>(fine_geom);
  auto fine_field_host =
      make_field_like_geometry<double>(fine_geom_host, 0.0);
  Field2DDevice<double> fine =
      build_device_field_from_host(fine_field_host);

  IntervalSet2DDevice fine_mask = fine_geom;
  auto fine_sub =
      make_subview(fine, fine_mask, "subview_prolong");

  const std::size_t cells =
      cells_in_rect(RectConfig{
          0,
          static_cast<Coord>(2 * (coarse_cfg.x_max - coarse_cfg.x_min)),
          0,
          static_cast<Coord>(2 * (coarse_cfg.y_max - coarse_cfg.y_min))});
  if (use_subset) {
    subsetix::csr::detail::ensure_subview_subset(fine_sub, &ctx);
  }

  double total_seconds = 0.0;

  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    prolong_field_subview_device(fine_sub, coarse);
    const auto t1 = std::chrono::steady_clock::now();
    total_seconds +=
        std::chrono::duration<double>(t1 - t0).count();
    benchmark::DoNotOptimize(fine.values.data());
  }

  const double total_cells =
      static_cast<double>(cells) *
      static_cast<double>(state.iterations());
  state.counters["ns_per_cell"] =
      (total_seconds / total_cells) * 1e9;
}

void bench_prolongation_prediction(benchmark::State& state,
                                   const RectConfig& coarse_cfg) {
  Field2DDevice<double> coarse =
      make_field(coarse_cfg, 0.0);
  auto coarse_host = build_host_field_from_device(coarse);
  for (std::size_t row = 0; row < coarse_host.row_keys.size();
       ++row) {
    const Coord y = coarse_host.row_keys[row].y;
    const std::size_t begin = coarse_host.row_ptr[row];
    const std::size_t end = coarse_host.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = coarse_host.intervals[k];
      for (Coord x = fi.begin; x < fi.end; ++x) {
        const std::size_t offset =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        coarse_host.values[offset] =
            static_cast<double>(100 + 3 * x + 5 * y);
      }
    }
  }
  coarse = build_device_field_from_host(coarse_host);

  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice fine_geom;
  refine_level_up_device(make_mask(coarse_cfg), fine_geom, ctx);
  auto fine_geom_host = to<HostMemorySpace>(fine_geom);
  auto fine_field_host =
      make_field_like_geometry<double>(fine_geom_host, 0.0);
  Field2DDevice<double> fine =
      build_device_field_from_host(fine_field_host);

  IntervalSet2DDevice fine_mask = fine_geom;
  const std::size_t cells =
      cells_in_rect(RectConfig{
          0,
          static_cast<Coord>(2 * (coarse_cfg.x_max - coarse_cfg.x_min)),
          0,
          static_cast<Coord>(2 * (coarse_cfg.y_max - coarse_cfg.y_min))});

  double total_seconds = 0.0;

  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    prolong_field_prediction_device(fine, coarse, fine_mask);
    const auto t1 = std::chrono::steady_clock::now();
    total_seconds +=
        std::chrono::duration<double>(t1 - t0).count();
    benchmark::DoNotOptimize(fine.values.data());
  }

  const double total_cells =
      static_cast<double>(cells) *
      static_cast<double>(state.iterations());
  state.counters["ns_per_cell"] =
      (total_seconds / total_cells) * 1e9;
}

void bench_subview_prolongation_prediction(
    benchmark::State& state,
    const RectConfig& coarse_cfg,
    bool use_subset = false) {
  Field2DDevice<double> coarse =
      make_field(coarse_cfg, 0.0);
  auto coarse_host = build_host_field_from_device(coarse);
  for (std::size_t row = 0; row < coarse_host.row_keys.size();
       ++row) {
    const Coord y = coarse_host.row_keys[row].y;
    const std::size_t begin = coarse_host.row_ptr[row];
    const std::size_t end = coarse_host.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = coarse_host.intervals[k];
      for (Coord x = fi.begin; x < fi.end; ++x) {
        const std::size_t offset =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        coarse_host.values[offset] =
            static_cast<double>(100 + 3 * x + 5 * y);
      }
    }
  }
  coarse = build_device_field_from_host(coarse_host);

  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice fine_geom;
  refine_level_up_device(make_mask(coarse_cfg), fine_geom, ctx);
  auto fine_geom_host = to<HostMemorySpace>(fine_geom);
  auto fine_field_host =
      make_field_like_geometry<double>(fine_geom_host, 0.0);
  Field2DDevice<double> fine =
      build_device_field_from_host(fine_field_host);

  IntervalSet2DDevice fine_mask = fine_geom;
  auto fine_sub =
      make_subview(fine, fine_mask, "subview_prolong_pred");

  const std::size_t cells =
      cells_in_rect(RectConfig{
          0,
          static_cast<Coord>(2 * (coarse_cfg.x_max - coarse_cfg.x_min)),
          0,
          static_cast<Coord>(2 * (coarse_cfg.y_max - coarse_cfg.y_min))});
  if (use_subset) {
    subsetix::csr::detail::ensure_subview_subset(fine_sub, &ctx);
  }

  double total_seconds = 0.0;

  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    prolong_field_prediction_subview_device(fine_sub, coarse);
    const auto t1 = std::chrono::steady_clock::now();
    total_seconds +=
        std::chrono::duration<double>(t1 - t0).count();
    benchmark::DoNotOptimize(fine.values.data());
  }

  const double total_cells =
      static_cast<double>(cells) *
      static_cast<double>(state.iterations());
  state.counters["ns_per_cell"] =
      (total_seconds / total_cells) * 1e9;
}

// --- Fill benchmarks ---

void BM_FieldFill_Tiny(benchmark::State& state) {
  const RectConfig cfg = make_rect(64);
  bench_field_op(state, cfg, cfg, [](auto& field, const auto& mask) {
    fill_on_set_device(field, mask, 1.0);
  });
}

void BM_FieldFill_Small(benchmark::State& state) {
  const RectConfig cfg = make_rect(256);
  bench_field_op(state, cfg, cfg, [](auto& field, const auto& mask) {
    fill_on_set_device(field, mask, 1.0);
  });
}

void BM_FieldFill_Medium(benchmark::State& state) {
  const RectConfig cfg = make_rect(1024);
  bench_field_op(state, cfg, cfg, [](auto& field, const auto& mask) {
    fill_on_set_device(field, mask, 1.0);
  });
}

void BM_FieldFill_Large(benchmark::State& state) {
  const RectConfig cfg = make_rect(2048);
  bench_field_op(state, cfg, cfg, [](auto& field, const auto& mask) {
    fill_on_set_device(field, mask, 1.0);
  });
}

// --- Stencil benchmarks ---

void BM_FieldStencil_Tiny(benchmark::State& state) {
  const RectConfig cfg = make_rect(64);
  bench_stencil(state, cfg, interior_rect(cfg));
}

void BM_FieldStencil_Small(benchmark::State& state) {
  const RectConfig cfg = make_rect(256);
  bench_stencil(state, cfg, interior_rect(cfg));
}

void BM_FieldStencil_Medium(benchmark::State& state) {
  const RectConfig cfg = make_rect(1024);
  bench_stencil(state, cfg, interior_rect(cfg));
}

void BM_FieldStencil_Large(benchmark::State& state) {
  const RectConfig cfg = make_rect(2048);
  bench_stencil(state, cfg, interior_rect(cfg));
}

void BM_FieldStencil10_Large(benchmark::State& state) {
  const RectConfig cfg = make_rect(2048);
  bench_stencil_repeated(state, cfg, interior_rect(cfg), 10);
}

// --- Restriction benchmarks ---

void BM_FieldRestrict_Tiny(benchmark::State& state) {
  bench_restriction(state, make_rect(64));
}

void BM_FieldRestrict_Small(benchmark::State& state) {
  bench_restriction(state, make_rect(256));
}

void BM_FieldRestrict_Medium(benchmark::State& state) {
  bench_restriction(state, make_rect(512));
}

void BM_FieldRestrict_Large(benchmark::State& state) {
  bench_restriction(state, make_rect(1024));
}

// --- Prolongation benchmarks ---

void BM_FieldProlong_Tiny(benchmark::State& state) {
  bench_prolongation(state, make_rect(64));
}

void BM_FieldProlong_Small(benchmark::State& state) {
  bench_prolongation(state, make_rect(256));
}

void BM_FieldProlong_Medium(benchmark::State& state) {
  bench_prolongation(state, make_rect(512));
}

void BM_FieldProlong_Large(benchmark::State& state) {
  bench_prolongation(state, make_rect(1024));
}

// --- Prolongation Prediction benchmarks ---

void BM_FieldProlongPrediction_Tiny(benchmark::State& state) {
  bench_prolongation_prediction(state, make_rect(64));
}

void BM_FieldProlongPrediction_Small(benchmark::State& state) {
  bench_prolongation_prediction(state, make_rect(256));
}

void BM_FieldProlongPrediction_Medium(benchmark::State& state) {
  bench_prolongation_prediction(state, make_rect(512));
}

void BM_FieldProlongPrediction_Large(benchmark::State& state) {
  bench_prolongation_prediction(state, make_rect(1024));
}

// --- Subview Fill benchmarks ---

void BM_FieldSubViewFill_Tiny(benchmark::State& state) {
  const RectConfig cfg = make_rect(64);
  bench_subview_op(state, cfg, cfg, [](auto& sub) {
    fill_subview_device(sub, 1.0);
  });
}

void BM_FieldSubViewFill_Small(benchmark::State& state) {
  const RectConfig cfg = make_rect(256);
  bench_subview_op(state, cfg, cfg, [](auto& sub) {
    fill_subview_device(sub, 1.0);
  });
}

void BM_FieldSubViewFill_Medium(benchmark::State& state) {
  const RectConfig cfg = make_rect(1024);
  bench_subview_op(state, cfg, cfg, [](auto& sub) {
    fill_subview_device(sub, 1.0);
  });
}

void BM_FieldSubViewFill_Large(benchmark::State& state) {
  const RectConfig cfg = make_rect(2048);
  bench_subview_op(state, cfg, cfg, [](auto& sub) {
    fill_subview_device(sub, 1.0);
  });
}

void BM_FieldSubViewFillSubset_Tiny(benchmark::State& state) {
  const RectConfig cfg = make_rect(64);
  bench_subview_op(state, cfg, cfg, [](auto& sub) {
    fill_subview_device(sub, 1.0);
  }, true);
}

void BM_FieldSubViewFillSubset_Small(benchmark::State& state) {
  const RectConfig cfg = make_rect(256);
  bench_subview_op(state, cfg, cfg, [](auto& sub) {
    fill_subview_device(sub, 1.0);
  }, true);
}

void BM_FieldSubViewFillSubset_Medium(benchmark::State& state) {
  const RectConfig cfg = make_rect(1024);
  bench_subview_op(state, cfg, cfg, [](auto& sub) {
    fill_subview_device(sub, 1.0);
  }, true);
}

void BM_FieldSubViewFillSubset_Large(benchmark::State& state) {
  const RectConfig cfg = make_rect(2048);
  bench_subview_op(state, cfg, cfg, [](auto& sub) {
    fill_subview_device(sub, 1.0);
  }, true);
}

// --- Subview Stencil benchmarks ---

void BM_FieldSubViewStencil_Tiny(benchmark::State& state) {
  const RectConfig cfg = make_rect(64);
  bench_subview_stencil(state, cfg, interior_rect(cfg));
}

void BM_FieldSubViewStencil_Small(benchmark::State& state) {
  const RectConfig cfg = make_rect(256);
  bench_subview_stencil(state, cfg, interior_rect(cfg));
}

void BM_FieldSubViewStencil_Medium(benchmark::State& state) {
  const RectConfig cfg = make_rect(1024);
  bench_subview_stencil(state, cfg, interior_rect(cfg));
}

void BM_FieldSubViewStencil_Large(benchmark::State& state) {
  const RectConfig cfg = make_rect(2048);
  bench_subview_stencil(state, cfg, interior_rect(cfg));
}

void BM_FieldSubViewStencil10_Large(benchmark::State& state) {
  const RectConfig cfg = make_rect(2048);
  bench_subview_stencil_repeated(state, cfg, interior_rect(cfg),
                                 false, 10);
}

void BM_FieldSubViewStencilSubset10_Large(benchmark::State& state) {
  const RectConfig cfg = make_rect(2048);
  bench_subview_stencil_repeated(state, cfg, interior_rect(cfg),
                                 true, 10);
}

void BM_FieldSubViewStencilSubset_Tiny(benchmark::State& state) {
  const RectConfig cfg = make_rect(64);
  bench_subview_stencil(state, cfg, interior_rect(cfg), true);
}

void BM_FieldSubViewStencilSubset_Small(benchmark::State& state) {
  const RectConfig cfg = make_rect(256);
  bench_subview_stencil(state, cfg, interior_rect(cfg), true);
}

void BM_FieldSubViewStencilSubset_Medium(benchmark::State& state) {
  const RectConfig cfg = make_rect(1024);
  bench_subview_stencil(state, cfg, interior_rect(cfg), true);
}

void BM_FieldSubViewStencilSubset_Large(benchmark::State& state) {
  const RectConfig cfg = make_rect(2048);
  bench_subview_stencil(state, cfg, interior_rect(cfg), true);
}

// --- Subview Restriction benchmarks ---

void BM_FieldSubViewRestrict_Tiny(benchmark::State& state) {
  bench_subview_restriction(state, make_rect(64));
}

void BM_FieldSubViewRestrict_Small(benchmark::State& state) {
  bench_subview_restriction(state, make_rect(256));
}

void BM_FieldSubViewRestrict_Medium(benchmark::State& state) {
  bench_subview_restriction(state, make_rect(512));
}

void BM_FieldSubViewRestrict_Large(benchmark::State& state) {
  bench_subview_restriction(state, make_rect(1024));
}

void BM_FieldSubViewRestrictSubset_Tiny(benchmark::State& state) {
  bench_subview_restriction(state, make_rect(64), true);
}

void BM_FieldSubViewRestrictSubset_Small(benchmark::State& state) {
  bench_subview_restriction(state, make_rect(256), true);
}

void BM_FieldSubViewRestrictSubset_Medium(benchmark::State& state) {
  bench_subview_restriction(state, make_rect(512), true);
}

void BM_FieldSubViewRestrictSubset_Large(benchmark::State& state) {
  bench_subview_restriction(state, make_rect(1024), true);
}

// --- Subview Prolongation benchmarks ---

void BM_FieldSubViewProlong_Tiny(benchmark::State& state) {
  bench_subview_prolongation(state, make_rect(64));
}

void BM_FieldSubViewProlong_Small(benchmark::State& state) {
  bench_subview_prolongation(state, make_rect(256));
}

void BM_FieldSubViewProlong_Medium(benchmark::State& state) {
  bench_subview_prolongation(state, make_rect(512));
}

void BM_FieldSubViewProlong_Large(benchmark::State& state) {
  bench_subview_prolongation(state, make_rect(1024));
}

void BM_FieldSubViewProlongSubset_Tiny(benchmark::State& state) {
  bench_subview_prolongation(state, make_rect(64), true);
}

void BM_FieldSubViewProlongSubset_Small(benchmark::State& state) {
  bench_subview_prolongation(state, make_rect(256), true);
}

void BM_FieldSubViewProlongSubset_Medium(benchmark::State& state) {
  bench_subview_prolongation(state, make_rect(512), true);
}

void BM_FieldSubViewProlongSubset_Large(benchmark::State& state) {
  bench_subview_prolongation(state, make_rect(1024), true);
}

// --- Subview Prolongation Prediction benchmarks ---

void BM_FieldSubViewProlongPrediction_Tiny(benchmark::State& state) {
  bench_subview_prolongation_prediction(state, make_rect(64));
}

void BM_FieldSubViewProlongPrediction_Small(benchmark::State& state) {
  bench_subview_prolongation_prediction(state, make_rect(256));
}

void BM_FieldSubViewProlongPrediction_Medium(benchmark::State& state) {
  bench_subview_prolongation_prediction(state, make_rect(512));
}

void BM_FieldSubViewProlongPrediction_Large(benchmark::State& state) {
  bench_subview_prolongation_prediction(state, make_rect(1024));
}

void BM_FieldSubViewProlongPredictionSubset_Tiny(benchmark::State& state) {
  bench_subview_prolongation_prediction(state, make_rect(64), true);
}

void BM_FieldSubViewProlongPredictionSubset_Small(benchmark::State& state) {
  bench_subview_prolongation_prediction(state, make_rect(256), true);
}

void BM_FieldSubViewProlongPredictionSubset_Medium(benchmark::State& state) {
  bench_subview_prolongation_prediction(state, make_rect(512), true);
}

void BM_FieldSubViewProlongPredictionSubset_Large(benchmark::State& state) {
  bench_subview_prolongation_prediction(state, make_rect(1024), true);
}

} // namespace

BENCHMARK(BM_FieldFill_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldFill_Small)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldFill_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldFill_Large)->Unit(benchmark::kNanosecond);

BENCHMARK(BM_FieldStencil_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldStencil_Small)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldStencil_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldStencil_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldStencil10_Large)->Unit(benchmark::kNanosecond);

BENCHMARK(BM_FieldRestrict_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldRestrict_Small)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldRestrict_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldRestrict_Large)->Unit(benchmark::kNanosecond);

BENCHMARK(BM_FieldProlong_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldProlong_Small)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldProlong_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldProlong_Large)->Unit(benchmark::kNanosecond);

BENCHMARK(BM_FieldProlongPrediction_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldProlongPrediction_Small)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldProlongPrediction_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldProlongPrediction_Large)->Unit(benchmark::kNanosecond);

BENCHMARK(BM_FieldSubViewFill_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewFill_Small)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewFill_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewFill_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewFillSubset_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewFillSubset_Small)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewFillSubset_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewFillSubset_Large)->Unit(benchmark::kNanosecond);

BENCHMARK(BM_FieldSubViewStencil_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewStencil_Small)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewStencil_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewStencil_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewStencil10_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewStencilSubset10_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewStencilSubset_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewStencilSubset_Small)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewStencilSubset_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewStencilSubset_Large)->Unit(benchmark::kNanosecond);

BENCHMARK(BM_FieldSubViewRestrict_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewRestrict_Small)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewRestrict_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewRestrict_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewRestrictSubset_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewRestrictSubset_Small)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewRestrictSubset_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewRestrictSubset_Large)->Unit(benchmark::kNanosecond);

BENCHMARK(BM_FieldSubViewProlong_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewProlong_Small)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewProlong_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewProlong_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewProlongSubset_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewProlongSubset_Small)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewProlongSubset_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewProlongSubset_Large)->Unit(benchmark::kNanosecond);

BENCHMARK(BM_FieldSubViewProlongPrediction_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewProlongPrediction_Small)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewProlongPrediction_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewProlongPrediction_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewProlongPredictionSubset_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewProlongPredictionSubset_Small)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewProlongPredictionSubset_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_FieldSubViewProlongPredictionSubset_Large)->Unit(benchmark::kNanosecond);

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  ::benchmark::RunSpecifiedBenchmarks();
  Kokkos::finalize();
  return 0;
}
