#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>
#include <chrono>
#include <vector>
#include <random>

#include <subsetix/field/csr_field.hpp>
#include <subsetix/field/csr_field_ops.hpp>
#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>
#include <subsetix/csr_ops/threshold.hpp>

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

Field2DDevice<double> make_random_field(
    const RectConfig& cfg, double fill_ratio) {
  auto geom = make_mask(cfg);
  auto geom_host = to<HostMemorySpace>(geom);
  
  // Create field on host
  auto field_host = make_field_like_geometry<double>(geom_host, 0.0);

  // Fill with random values
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  for (auto& val : field_host.values) {
    val = dist(gen);
  }

  return build_device_field_from_host(field_host);
}

void bench_threshold(benchmark::State& state, const RectConfig& cfg) {
  // Create a fully dense field
  Field2DDevice<double> field = make_random_field(cfg, 1.0);
  
  const std::size_t cells = cells_in_rect(cfg);
  
  // Threshold at 0.5 (approx 50% of values should pass if uniform [-1,1])
  const double epsilon = 0.5;

  double total_seconds = 0.0;

  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    auto result = threshold_field(field, epsilon);
    Kokkos::fence();
    const auto t1 = std::chrono::steady_clock::now();
    
    total_seconds +=
        std::chrono::duration<double>(t1 - t0).count();

    // Prevent optimization
    benchmark::DoNotOptimize(result.num_intervals);
  }

  const double total_cells =
      static_cast<double>(cells) *
      static_cast<double>(state.iterations());
  state.counters["ns_per_cell"] =
      (total_seconds / total_cells) * 1e9;
}

void BM_Threshold_Tiny(benchmark::State& state) {
  bench_threshold(state, make_rect(64));
}

void BM_Threshold_Small(benchmark::State& state) {
  bench_threshold(state, make_rect(256));
}

void BM_Threshold_Medium(benchmark::State& state) {
  bench_threshold(state, make_rect(1024));
}

void BM_Threshold_Large(benchmark::State& state) {
  bench_threshold(state, make_rect(2048));
}

void BM_Threshold_XLarge(benchmark::State& state) {
  bench_threshold(state, make_rect(4096));
}

void BM_Threshold_Huge(benchmark::State& state) {
  bench_threshold(state, make_rect(8192));
}

} // namespace

BENCHMARK(BM_Threshold_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Threshold_Small)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Threshold_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Threshold_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Threshold_XLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Threshold_Huge)->Unit(benchmark::kNanosecond);

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
