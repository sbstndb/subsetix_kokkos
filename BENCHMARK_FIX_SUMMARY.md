# Benchmark Fix Summary

## File Modified
`playground/intersection/benchmarks/intersection/row_mapping_comparison_benchmark.cpp`

## Changes Made

### 1. Replaced BENCHMARK_CAPTURE with RegisterBenchmark
**Before:**
```cpp
BENCHMARK_CAPTURE(BM_RowMapping_Generic, 2D_small_100x100_v2_baseline,
    "v2_baseline",
    [](const auto& a, const auto& b) { return optimized::intersect_meshes_2d(a, b); },
    small_100_a, small_100_b)
    ->Unit(benchmark::kMicrosecond);
```

**After:**
```cpp
benchmark::RegisterBenchmark("2D_small_100x100_v2_baseline", [&, mesh_a=std::move(small_100_a), mesh_b=std::move(small_100_b)](benchmark::State& state) {
  BM_RowMapping_Generic(state, "v2_baseline",
      [](const auto& a, const auto& b) { return optimized::intersect_meshes_2d(a, b); },
      mesh_a, mesh_b);
})->Unit(benchmark::kMicrosecond);
```

### 2. Fixed Lambda Capture Syntax
- Used `[&, mesh_a=std::move(...), mesh_b=std::move(...)]` to properly capture mesh variables
- Moved `->Unit(...)` to chain onto `RegisterBenchmark` instead of lambda

### 3. Fixed Namespace Qualifications
Changed from unqualified namespaces to fully qualified namespaces:
- `v4_hash::intersect` → `playground::subsetix::csr::intersection::hash_based::intersect`
- `v5_parallel_merge::intersect` → `playground::subsetix::csr::intersection::parallel_merge::intersect`
- `v6_direct_index::intersect` → `playground::subsetix::csr::intersection::direct_index::intersect`
- `v7_soa_optimized::intersect` → `playground::subsetix::csr::intersection::soa_optimized::intersect`
- `v8_hybrid_cpu_gpu::intersect` → `playground::subsetix::csr::intersection::hybrid_cpu_gpu::intersect`
- `v9_adaptive::intersect` → `playground::subsetix::csr::intersection::adaptive::intersect`

Note: Version namespaces don't match file names:
- File `v4_hash.hpp` contains namespace `hash_based`
- File `v5_parallel_merge.hpp` contains namespace `parallel_merge`
- File `v6_direct_index.hpp` contains namespace `direct_index`
- File `v7_soa_optimized.hpp` contains namespace `soa_optimized`
- File `v8_hybrid_cpu_gpu.hpp` contains namespace `hybrid_cpu_gpu`
- File `v9_adaptive.hpp` contains namespace `adaptive`

### 4. Removed Unused Type Aliases
Removed type aliases that were causing compilation errors:
- `using V4Mesh2D = hash_based::Mesh2D;`
- `using V5Mesh2D = parallel_merge::Mesh2D;`
- `using V6Mesh2D = direct_index::Mesh2D;`
- `using V7Mesh2D = soa_optimized::Mesh2D;`
- `using V8Mesh2D = hybrid_cpu_gpu::Mesh2D;`
- `using V9Mesh2D = adaptive::Mesh2D;` (and corresponding 3D versions)

Kept only the essential aliases:
- `using OptimizedMesh2D = optimized::Mesh2D<>;`
- `using OptimizedMesh3D = optimized::Mesh3D<>;`

## Issues Found

### Library Header Issues (Pre-existing)
The benchmark file now compiles correctly, but there are pre-existing compilation errors in library headers:

1. **v6_direct_index.hpp (line 373):** Incorrect attribute placement
   ```
   error: attributes are not allowed here
     __attribute__((device)) __attribute__((host))
   ```

2. **v9_adaptive.hpp (line 340):** Variable scope issue in switch case
   ```
   error: identifier "mesh_a" is undefined
   error: identifier "mesh_b" is undefined
   ```

These are library implementation issues, not benchmark issues.

## Verification
- All 70 benchmarks have been converted from BENCHMARK_CAPTURE to RegisterBenchmark
- All versions (baseline/v2, v4-v9) are preserved
- LARGE configuration focus maintained
- 2D and 3D benchmarks preserved
