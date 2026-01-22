# V1/V2/V3 Performance Comparison - 3D LargeConfig

## Benchmark Results (without profiler overhead)

| Version | Time | Throughput | vs V1 |
|---------|------|------------|-------|
| V1 | 5.393s | 4.839M/s | baseline |
| V2 | 5.127s | 5.025M/s | **+5.0%** faster |
| V3 | 5.108s | 5.033M/s | **+5.3%** faster |

**Note**: ncu profiling adds ~5.3s overhead (8 passes per kernel). Results above are from clean benchmark runs.

---

## V1 `intersect_meshes` - Detailed Kernel Breakdown

### Summary

| Metric | Value |
|--------|-------|
| **Total GPU time** | 5.35 ms |
| ParallelFor | 4.09 ms |
| ParallelScan | 1.26 ms |
| Kernels | 14 total |

### Per-Phase Analysis (sorted by duration)

| Phase | Instance | Grid | Duration | Bottleneck | Metrics |
|-------|----------|------|----------|------------|---------|
| **1. Row mapping** | For #1 | 39,063×128 | **1.87 ms** | **COMPUTE 69%** | Mem: 48%, Occ: 103%, Warps: 49.4/SM |
| **4. Fill intervals** | For #4 | 10,054×128 | **967 µs** | **MEMORY 90%** | Comp: 18%, Occ: 91%, Warps: 43.7/SM |
| **2. Count intervals** | For #3 | 10,054×128 | **933 µs** | **MEMORY 90%** | Comp: 14%, Occ: 91%, Warps: 43.8/SM |
| 1. Row scan | Scan #1 | 16,340×128 | 430 µs | MEMORY 49% | Comp: 49%, Occ: 81%, Warps: 38.9/SM |
| 3. Row ptr scan | Scan #2 | 10,054×128 | 345 µs | MEMORY 92% | Comp: 92%, Occ: 78%, Warps: 37.4/SM |
| 5. Compact scan | Scan #3 | 10,054×128 | 218 µs | MEMORY 54% | Comp: 54%, Occ: 77%, Warps: 36.7/SM |
| 1. Row compact | For #2 | 39,063×128 | 185 µs | MEMORY 90% | Comp: 20%, Occ: 93%, Warps: 44.8/SM |
| 2. Count | For #5 | 10,054×128 | 62 µs | MEMORY 88% | Comp: 26%, Occ: 81%, Warps: 38.9/SM |
| 5. Copy intervals | For #8 | 5,664×128 | 37 µs | MEMORY 84% | Comp: 31%, Occ: 81%, Warps: 38.9/SM |
| 5. Scan (warmup) | Scan #1b | 16,340×128 | 36 µs | MEMORY 80% | Comp: 43%, Occ: 81%, Warps: 38.7/SM |
| 1. Compact (warmup) | For #2b | 39,063×128 | 36 µs | MEMORY 81% | Comp: 43%, Occ: 91%, Warps: 43.6/SM |
| 5. Scan (warmup) | Scan #3b | 10,054×128 | 13 µs | MEMORY 68% | Comp: 36%, Occ: 78%, Warps: 37.4/SM |
| 5. Final ptr | For #7 | 1×128 | 4 µs | trivial | Negligible |

### Algorithm Phases (from source code)

| Phase | Description | Bottleneck |
|-------|-------------|------------|
| **1. Row mapping** | Find rows of A in B via `find_row_by_yz` (binary search) | **COMPUTE** - binary search per row |
| 2. Count intervals | Count intersections per row (without storing) | MEMORY - read A/B intervals |
| 3. Scan offsets | Compute `row_ptr` offsets via prefix sum | MEMORY |
| **4. Fill intervals** | Write intersected intervals to output | **MEMORY** - write results |
| 5. Compact | Remove empty rows | MEMORY - copy operations |

### Key Finding

**Primary bottleneck**: Phase 1 (row mapping) - 1.87ms (35% of total)
- COMPUTE-bound at 69% (vs 48% memory)
- Binary search `find_row_by_yz` for each row
- Grid: 39,063 blocks (largest kernel)

**Optimization opportunity**: Replace binary search with hash map or merge-based approach.

---

## NCU Profiling Files

- `compare_v/v1.ncu-rep` (2.35 MB)
- `compare_v/v2.ncu-rep` (2.35 MB)
- `compare_v/v3.ncu-rep` (2.35 MB)

View with: `ncu-ui compare_v/v1.ncu-rep`

---

## GPU Specifications

- Device: NVIDIA ADA
- Compute Capability: 8.9
- SMs: 20
- Peak Memory Bandwidth: ~1000 GB/s (theoretical)

---

## Notes

- Profiling was done with ncu using `--set basic` section set
- Each kernel runs 8 passes to collect different metrics
- Overhead is significant; use clean benchmarks for timing comparisons
- V2/V3 detailed analysis pending
