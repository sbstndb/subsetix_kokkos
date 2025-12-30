#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>
#include <chrono>

#include <subsetix/benchmark_sizes.hpp>
#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>

using namespace subsetix::csr;
using namespace subsetix::benchmark;

namespace {

// Use standardized benchmark sizes from benchmark_sizes.hpp
constexpr Coord kSizeTiny   = kIntervalsTiny;
constexpr Coord kSizeMedium = kIntervalsMedium;
constexpr Coord kSizeLarge  = kIntervalsLarge;
constexpr Coord kSizeXLarge = kIntervalsXLarge;

IntervalSet2DDevice make_bench_box(Coord size) {
  Box2D box;
  box.x_min = 0;
  box.x_max = size;
  box.y_min = 0;
  box.y_max = size;
  return make_box_device(box);
}

// Generic template for morphology operations (Expand/Shrink)
template <class Op>
void bench_morphology(benchmark::State& state, Coord size, Coord rx, Coord ry, Op op) {
  IntervalSet2DDevice A = make_bench_box(size);
  
  // Heuristic preallocation
  IntervalSet2DDevice out = allocate_interval_set_device(
      A.num_rows * 2, // Expand might add rows
      A.num_intervals * 2
  );
  
  CsrSetAlgebraContext ctx;
  
  double total_seconds = 0.0;
  
  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    op(A, rx, ry, out, ctx);
    const auto t1 = std::chrono::steady_clock::now();
    
    total_seconds += std::chrono::duration<double>(t1 - t0).count();
    
    benchmark::DoNotOptimize(out.row_keys.data());
    benchmark::DoNotOptimize(out.intervals.data());
  }
  
  if (A.num_intervals > 0) {
      double total_intervals = static_cast<double>(A.num_intervals) * static_cast<double>(state.iterations());
      state.counters["ns_per_input_interval"] = (total_seconds / total_intervals) * 1e9;
  }
}

void BM_Expand_R1_Tiny(benchmark::State& state) {
    bench_morphology(state, kSizeTiny, 1, 1, expand_device);
}
void BM_Expand_R1_Medium(benchmark::State& state) {
    bench_morphology(state, kSizeMedium, 1, 1, expand_device);
}
void BM_Expand_R1_Large(benchmark::State& state) {
    bench_morphology(state, kSizeLarge, 1, 1, expand_device);
}
void BM_Expand_R1_XLarge(benchmark::State& state) {
    bench_morphology(state, kSizeXLarge, 1, 1, expand_device);
}

void BM_Shrink_R1_Tiny(benchmark::State& state) {
    bench_morphology(state, kSizeTiny, 1, 1, shrink_device);
}
void BM_Shrink_R1_Medium(benchmark::State& state) {
    bench_morphology(state, kSizeMedium, 1, 1, shrink_device);
}
void BM_Shrink_R1_Large(benchmark::State& state) {
    bench_morphology(state, kSizeLarge, 1, 1, shrink_device);
}
void BM_Shrink_R1_XLarge(benchmark::State& state) {
    bench_morphology(state, kSizeXLarge, 1, 1, shrink_device);
}

// Radius 2 benchmarks
void BM_Expand_R2_Medium(benchmark::State& state) {
    bench_morphology(state, kSizeMedium, 2, 2, expand_device);
}
void BM_Shrink_R2_Medium(benchmark::State& state) {
    bench_morphology(state, kSizeMedium, 2, 2, shrink_device);
}

} // namespace

BENCHMARK(BM_Expand_R1_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Expand_R1_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Expand_R1_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Expand_R1_XLarge)->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Shrink_R1_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Shrink_R1_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Shrink_R1_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Shrink_R1_XLarge)->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Expand_R2_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Shrink_R2_Medium)->Unit(benchmark::kNanosecond);

// Standard main for benchmarks
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  Kokkos::finalize();
  return 0;
}
