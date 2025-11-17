#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>
#include <chrono>

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_set_ops.hpp>

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

void bench_intersection_rectangles(benchmark::State& state,
                                   const RectConfig& a_cfg,
                                   const RectConfig& b_cfg) {
  IntervalSet2DDevice A = make_box(a_cfg);
  IntervalSet2DDevice B = make_box(b_cfg);

  const std::size_t intervals_in =
      A.num_intervals + B.num_intervals;

  double total_seconds = 0.0;

  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    auto out = set_intersection_device(A, B);
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

void BM_CSRIntersection_Tiny(benchmark::State& state) {
  RectConfig a{0, 128, 0, 128};          // 128x128
  const Coord N = 128;
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N}; // carré 128x128
  bench_intersection_rectangles(state, a, b);
}

void BM_CSRIntersection_Medium(benchmark::State& state) {
  const Coord N = 1280;
  RectConfig a{0, N, 0, N};              // 1280x1280
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N}; // carré 1280x1280
  bench_intersection_rectangles(state, a, b);
}

void BM_CSRIntersection_Large(benchmark::State& state) {
  const Coord N = 12800;
  RectConfig a{0, N, 0, N};              // 12800x12800
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N}; // carré 12800x12800
  bench_intersection_rectangles(state, a, b);
}

void BM_CSRIntersection_XLarge(benchmark::State& state) {
  const Coord N = 128000;
  RectConfig a{0, N, 0, N};              // 128000x128000
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N}; // carré 128000x128000
  bench_intersection_rectangles(state, a, b);
}

} // namespace

BENCHMARK(BM_CSRIntersection_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_XLarge)->Unit(benchmark::kNanosecond);

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
