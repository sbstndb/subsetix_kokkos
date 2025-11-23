#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>

#include <subsetix/multilevel/multilevel.hpp>
#include <subsetix/geometry.hpp>

using namespace subsetix;
using namespace subsetix::csr;
using subsetix::csr::ExecSpace;

// Global variables to hold data across benchmark runs to avoid reallocation noise
// We allocate ONCE for the max size, and then benchmarks use a subset.
IntervalSet2DDevice global_mesh_max;
MultilevelGeoDevice global_multi_max;

// Max size ~ 16 Million items
constexpr int MAX_BENCH_SIZE = 16 * 1024 * 1024;

void InitializeData() {
    // Create a huge box. 
    // For simplicity, let's make a box that results in exactly MAX_BENCH_SIZE rows.
    // Box2D height determines num_rows.
    Box2D box{0, 10, 0, MAX_BENCH_SIZE}; 
    global_mesh_max = make_box_device(box);
    
    global_multi_max.num_active_levels = 1;
    global_multi_max.levels[0] = global_mesh_max;
}

void CleanupData() {
    global_mesh_max = IntervalSet2DDevice();
    global_multi_max = MultilevelGeoDevice();
}

// ----------------------------------------------------------------------------
// Benchmark 1: Direct View Access
// ----------------------------------------------------------------------------
void BM_DirectViewAccess(benchmark::State& state) {
    // Use a subset of rows determined by the benchmark arg
    const int num_rows = static_cast<int>(state.range(0));
    
    // Check bounds to be safe
    if (num_rows > static_cast<int>(global_mesh_max.num_rows)) {
        state.SkipWithError("Benchmark size exceeds pre-allocated size");
        return;
    }

    auto row_keys = global_mesh_max.row_keys; // Capture View directly
    
    for (auto _ : state) {
        Kokkos::parallel_for("DirectAccess", Kokkos::RangePolicy<ExecSpace>(0, num_rows),
            KOKKOS_LAMBDA(const int i) {
                volatile Coord y = row_keys(i).y; 
                (void)y;
            });
        ExecSpace().fence();
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_rows);
}

// ----------------------------------------------------------------------------
// Benchmark 2: Multilevel Struct Access
// ----------------------------------------------------------------------------
void BM_MultilevelStructAccess(benchmark::State& state) {
    const int num_rows = static_cast<int>(state.range(0));
    
    MultilevelGeoDevice geo = global_multi_max; 
    
    for (auto _ : state) {
        Kokkos::parallel_for("StructAccess", Kokkos::RangePolicy<ExecSpace>(0, num_rows),
            KOKKOS_LAMBDA(const int i) {
                volatile Coord y = geo.levels[0].row_keys(i).y;
                (void)y;
            });
        ExecSpace().fence();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_rows);
}

// ----------------------------------------------------------------------------
// Benchmark 3: Dynamic Index Access
// ----------------------------------------------------------------------------
void BM_MultilevelDynamicAccess(benchmark::State& state) {
    const int num_rows = static_cast<int>(state.range(0));
    
    MultilevelGeoDevice geo = global_multi_max;
    
    for (auto _ : state) {
        Kokkos::parallel_for("DynamicAccess", Kokkos::RangePolicy<ExecSpace>(0, num_rows),
            KOKKOS_LAMBDA(const int i) {
                int level_idx = 0; 
                volatile Coord y = geo.levels[level_idx].row_keys(i).y;
                (void)y;
            });
        ExecSpace().fence();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_rows);
}

// Register Benchmarks with Ranges
// Range from 1k to 16M
BENCHMARK(BM_DirectViewAccess)
    ->RangeMultiplier(8)->Range(1024, MAX_BENCH_SIZE)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_MultilevelStructAccess)
    ->RangeMultiplier(8)->Range(1024, MAX_BENCH_SIZE)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_MultilevelDynamicAccess)
    ->RangeMultiplier(8)->Range(1024, MAX_BENCH_SIZE)
    ->Unit(benchmark::kMicrosecond);

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        InitializeData();
        
        ::benchmark::Initialize(&argc, argv);
        if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
             CleanupData();
             Kokkos::finalize();
             return 1;
        }
        ::benchmark::RunSpecifiedBenchmarks();
        
        CleanupData();
    }
    Kokkos::finalize();
    return 0;
}
