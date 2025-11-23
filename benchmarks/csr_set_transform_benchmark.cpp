#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>
#include <chrono>

#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>

using namespace subsetix::csr;

namespace {

struct RectConfig {
  Coord x_min;
  Coord x_max;
  Coord y_min;
  Coord y_max;
};

IntervalSet2DDevice make_box(const RectConfig& cfg) {
  Box2D box;
  box.x_min = cfg.x_min;
  box.x_max = cfg.x_max;
  box.y_min = cfg.y_min;
  box.y_max = cfg.y_max;
  return make_box_device(box);
}

template <class UnaryOp>
void bench_unary_op(benchmark::State& state,
                    const RectConfig& cfg,
                    UnaryOp op) {
  IntervalSet2DDevice A = make_box(cfg);
  const std::size_t intervals_in = A.num_intervals;

  // Create context outside the loop to measure steady-state performance
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice out;

  double total_seconds = 0.0;

  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    op(A, out, ctx);
    const auto t1 = std::chrono::steady_clock::now();

    const std::chrono::duration<double> dt = t1 - t0;
    total_seconds += dt.count();

    benchmark::DoNotOptimize(out.row_keys.data());
    benchmark::DoNotOptimize(out.row_ptr.data());
    benchmark::DoNotOptimize(out.intervals.data());
  }

  if (intervals_in > 0) {
    const double total_intervals =
        static_cast<double>(intervals_in) *
        static_cast<double>(state.iterations());
    const double seconds_per_interval =
        total_seconds / total_intervals;
    state.counters["ns_per_interval"] =
        seconds_per_interval * 1e9;
  }
}

void bench_projection(benchmark::State& state,
                      const RectConfig& cfg) {
  IntervalSet2DDevice coarse = make_box(cfg);
  
  // Create context outside the loop
  CsrSetAlgebraContext ctx;
  
  // Use context for refine as well
  IntervalSet2DDevice fine;
  refine_level_up_device(coarse, fine, ctx);
  const std::size_t intervals_in = fine.num_intervals;

  IntervalSet2DDevice out;
  double total_seconds = 0.0;

  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    project_level_down_device(fine, out, ctx);
    const auto t1 = std::chrono::steady_clock::now();

    const std::chrono::duration<double> dt = t1 - t0;
    total_seconds += dt.count();

    benchmark::DoNotOptimize(out.row_keys.data());
    benchmark::DoNotOptimize(out.row_ptr.data());
    benchmark::DoNotOptimize(out.intervals.data());
  }

  if (intervals_in > 0) {
    const double total_intervals =
        static_cast<double>(intervals_in) *
        static_cast<double>(state.iterations());
    const double seconds_per_interval =
        total_seconds / total_intervals;
    state.counters["ns_per_interval"] =
        seconds_per_interval * 1e9;
  }
}

void bench_allocation(benchmark::State& state,
                      const RectConfig& cfg) {
  IntervalSet2DDevice source = make_box(cfg);
  const std::size_t intervals =
      source.num_intervals > 0 ? source.num_intervals : 1;

  double total_seconds = 0.0;

  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();

    IntervalSet2DDevice out;
    out.num_rows = source.num_rows;
    out.num_intervals = source.num_intervals;
    if (out.num_rows > 0) {
      out.row_keys = IntervalSet2DDevice::RowKeyView(
          "subsetix_alloc_row_keys", out.num_rows);
      out.row_ptr = IntervalSet2DDevice::IndexView(
          "subsetix_alloc_row_ptr", out.num_rows + 1);
      out.intervals = IntervalSet2DDevice::IntervalView(
          "subsetix_alloc_intervals", out.num_intervals);
    }

    const auto t1 = std::chrono::steady_clock::now();
    total_seconds += std::chrono::duration<double>(t1 - t0).count();

    benchmark::DoNotOptimize(out.row_keys.data());
    benchmark::DoNotOptimize(out.row_ptr.data());
    benchmark::DoNotOptimize(out.intervals.data());
  }

  const double total_intervals =
      static_cast<double>(intervals) *
      static_cast<double>(state.iterations());
  const double seconds_per_interval =
      total_seconds / total_intervals;
  state.counters["ns_per_interval"] =
      seconds_per_interval * 1e9;
}

// --- Translation X ---

void BM_CSRTranslateX_Tiny(benchmark::State& state) {
  RectConfig cfg{0, 128, 0, 128};
  bench_unary_op(state, cfg, [](const IntervalSet2DDevice& A, IntervalSet2DDevice& out, CsrSetAlgebraContext& ctx) {
    translate_x_device(A, 5, out, ctx);
  });
}

void BM_CSRTranslateX_Medium(benchmark::State& state) {
  RectConfig cfg{0, 1280, 0, 1280};
  bench_unary_op(state, cfg, [](const IntervalSet2DDevice& A, IntervalSet2DDevice& out, CsrSetAlgebraContext& ctx) {
    translate_x_device(A, 5, out, ctx);
  });
}

void BM_CSRTranslateX_Large(benchmark::State& state) {
  RectConfig cfg{0, 12800, 0, 12800};
  bench_unary_op(state, cfg, [](const IntervalSet2DDevice& A, IntervalSet2DDevice& out, CsrSetAlgebraContext& ctx) {
    translate_x_device(A, 5, out, ctx);
  });
}

void BM_CSRTranslateX_XLarge(benchmark::State& state) {
  RectConfig cfg{0, 128000, 0, 128000};
  bench_unary_op(state, cfg, [](const IntervalSet2DDevice& A, IntervalSet2DDevice& out, CsrSetAlgebraContext& ctx) {
    translate_x_device(A, 5, out, ctx);
  });
}

// --- Translation Y ---

void BM_CSRTranslateY_Tiny(benchmark::State& state) {
  RectConfig cfg{0, 128, 0, 128};
  bench_unary_op(state, cfg, [](const IntervalSet2DDevice& A, IntervalSet2DDevice& out, CsrSetAlgebraContext& ctx) {
    translate_y_device(A, 5, out, ctx);
  });
}

void BM_CSRTranslateY_Medium(benchmark::State& state) {
  RectConfig cfg{0, 1280, 0, 1280};
  bench_unary_op(state, cfg, [](const IntervalSet2DDevice& A, IntervalSet2DDevice& out, CsrSetAlgebraContext& ctx) {
    translate_y_device(A, 5, out, ctx);
  });
}

void BM_CSRTranslateY_Large(benchmark::State& state) {
  RectConfig cfg{0, 12800, 0, 12800};
  bench_unary_op(state, cfg, [](const IntervalSet2DDevice& A, IntervalSet2DDevice& out, CsrSetAlgebraContext& ctx) {
    translate_y_device(A, 5, out, ctx);
  });
}

void BM_CSRTranslateY_XLarge(benchmark::State& state) {
  RectConfig cfg{0, 128000, 0, 128000};
  bench_unary_op(state, cfg, [](const IntervalSet2DDevice& A, IntervalSet2DDevice& out, CsrSetAlgebraContext& ctx) {
    translate_y_device(A, 5, out, ctx);
  });
}

// --- Refinement (prediction) ---

void BM_CSRRefine_Tiny(benchmark::State& state) {
  RectConfig cfg{0, 128, 0, 128};
  bench_unary_op(state, cfg, [](const IntervalSet2DDevice& A, IntervalSet2DDevice& out, CsrSetAlgebraContext& ctx) {
    refine_level_up_device(A, out, ctx);
  });
}

void BM_CSRRefine_Medium(benchmark::State& state) {
  RectConfig cfg{0, 1280, 0, 1280};
  bench_unary_op(state, cfg, [](const IntervalSet2DDevice& A, IntervalSet2DDevice& out, CsrSetAlgebraContext& ctx) {
    refine_level_up_device(A, out, ctx);
  });
}

void BM_CSRRefine_Large(benchmark::State& state) {
  RectConfig cfg{0, 12800, 0, 12800};
  bench_unary_op(state, cfg, [](const IntervalSet2DDevice& A, IntervalSet2DDevice& out, CsrSetAlgebraContext& ctx) {
    refine_level_up_device(A, out, ctx);
  });
}

void BM_CSRRefine_XLarge(benchmark::State& state) {
  RectConfig cfg{0, 128000, 0, 128000};
  bench_unary_op(state, cfg, [](const IntervalSet2DDevice& A, IntervalSet2DDevice& out, CsrSetAlgebraContext& ctx) {
    refine_level_up_device(A, out, ctx);
  });
}

// --- Projection ---

void BM_CSRProject_Tiny(benchmark::State& state) {
  RectConfig cfg{0, 128, 0, 128};
  bench_projection(state, cfg);
}

void BM_CSRProject_Medium(benchmark::State& state) {
  RectConfig cfg{0, 1280, 0, 1280};
  bench_projection(state, cfg);
}

void BM_CSRProject_Large(benchmark::State& state) {
  RectConfig cfg{0, 12800, 0, 12800};
  bench_projection(state, cfg);
}

void BM_CSRProject_XLarge(benchmark::State& state) {
  RectConfig cfg{0, 128000, 0, 128000};
  bench_projection(state, cfg);
}

// --- Allocations ---

void BM_CSRAllocate_Tiny(benchmark::State& state) {
  RectConfig cfg{0, 128, 0, 128};
  bench_allocation(state, cfg);
}

void BM_CSRAllocate_Medium(benchmark::State& state) {
  RectConfig cfg{0, 1280, 0, 1280};
  bench_allocation(state, cfg);
}

void BM_CSRAllocate_Large(benchmark::State& state) {
  RectConfig cfg{0, 12800, 0, 12800};
  bench_allocation(state, cfg);
}

void BM_CSRAllocate_XLarge(benchmark::State& state) {
  RectConfig cfg{0, 128000, 0, 128000};
  bench_allocation(state, cfg);
}

} // namespace

BENCHMARK(BM_CSRTranslateX_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRTranslateX_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRTranslateX_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRTranslateX_XLarge)->Unit(benchmark::kNanosecond);

BENCHMARK(BM_CSRTranslateY_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRTranslateY_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRTranslateY_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRTranslateY_XLarge)->Unit(benchmark::kNanosecond);

BENCHMARK(BM_CSRRefine_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRRefine_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRRefine_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRRefine_XLarge)->Unit(benchmark::kNanosecond);

BENCHMARK(BM_CSRProject_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRProject_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRProject_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRProject_XLarge)->Unit(benchmark::kNanosecond);

BENCHMARK(BM_CSRAllocate_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRAllocate_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRAllocate_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRAllocate_XLarge)->Unit(benchmark::kNanosecond);

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
