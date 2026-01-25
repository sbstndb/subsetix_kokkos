# Comprehensive Intersection Algorithm Performance Benchmark

## Test Configuration

- **Date**: 2026-01-24
- **Build**: Debug mode (with optimization disabled for accurate algorithm comparison)
- **Repetitions**: 3 per benchmark
- **Min Time**: 3 seconds per benchmark
- **Platform**: 24 X 4139.76 MHz CPU

## Algorithm Versions

1. **Baseline** - Original row mapping with std::lower_bound
2. **Optimized** - Optimized row mapping with custom binary search
3. **V4 Hash** - Hash-based row mapping
4. **V5 Parallel Merge** - Parallel merge-based row mapping
5. **V6 Direct Index** - Direct index-based row mapping

---

## 2D Large Config Results

| Algorithm | Mean Time (ns) | Throughput (Mi/s) | Speedup vs Baseline |
|-----------|----------------|-------------------|---------------------|
| **Baseline** | 630,428 | 74.94 | 1.00x |
| **Optimized** | 631,697 | 74.79 | 0.998x |
| **V4 Hash** | 611,158 | 77.30 | **1.031x** |
| **V5 Parallel Merge** | 415,563 | 113.68 | **1.517x** |
| **V6 Direct Index** | 626,606 | 75.39 | 1.006x |

### Key Finding: 2D Large
- **V5 Parallel Merge** is the clear winner with **51.7% speedup** over baseline
- V4 Hash shows modest 3.1% improvement
- V6 Direct Index is essentially tied with baseline

---

## 3D Large Config Results

| Algorithm | Mean Time (ns) | Throughput (Mi/s) | Speedup vs Baseline |
|-----------|----------------|-------------------|---------------------|
| **Baseline** | 3,494,677,741 | 54.95 | 1.00x |
| **Optimized** | 3,441,201,368 | 55.82 | **1.016x** |
| **V4 Hash** | 4,812,362,226 | 39.91 | 0.726x |
| **V5 Parallel Merge** | 1,742,145,128 | 110.24 | **2.006x** |
| **V6 Direct Index** | 1,007,245,970 | 190.65 | **3.469x** |

### Key Finding: 3D Large
- **V6 Direct Index** dominates with **247% speedup** (3.47x faster)
- **V5 Parallel Merge** also excellent at **101% speedup** (2.01x faster)
- V4 Hash performs poorly on 3D (27% slower than baseline)

---

## 2D ExtraLarge Config Results (2x Large)

| Algorithm | Mean Time (ns) | Throughput (Mi/s) | Speedup vs Baseline |
|-----------|----------------|-------------------|---------------------|
| **Baseline** | 519,279 | 90.98 | 1.00x |
| **Optimized** | 531,068 | 88.96 | 0.977x |
| **V4 Hash** | 535,111 | 88.28 | 0.970x |
| **V5 Parallel Merge** | 335,733 | 140.71 | **1.547x** |
| **V6 Direct Index** | 535,131 | 88.28 | 0.970x |

### Key Finding: 2D ExtraLarge
- **V5 Parallel Merge** maintains leadership with **54.7% speedup**
- Other algorithms are within 3% of baseline

---

## 3D ExtraLarge Config Results (2x Large)

| Algorithm | Mean Time (ns) | Throughput (Mi/s) | Speedup vs Baseline |
|-----------|----------------|-------------------|---------------------|
| **Baseline** | 6,066,469,784 | 63.31 | 1.00x |
| **Optimized** | 6,006,309,295 | 63.94 | **1.010x** |
| **V4 Hash** | 7,089,506,363 | 54.18 | 0.856x |
| **V5 Parallel Merge** | 2,692,836,791 | 142.62 | **2.253x** |
| **V6 Direct Index** | 2,123,298,443 | 180.88 | **2.857x** |

### Key Finding: 3D ExtraLarge
- **V6 Direct Index** leads with **186% speedup** (2.86x faster)
- **V5 Parallel Merge** strong at **125% speedup** (2.25x faster)
- V4 Hash continues to underperform on 3D

---

## Summary and Recommendations

### Performance Winners by Configuration

| Config | Winner | Speedup | Runner-up | Speedup |
|--------|--------|---------|-----------|---------|
| **2D Large** | V5 Parallel Merge | 1.52x | V4 Hash | 1.03x |
| **3D Large** | V6 Direct Index | 3.47x | V5 Parallel Merge | 2.01x |
| **2D ExtraLarge** | V5 Parallel Merge | 1.55x | Baseline | 1.00x |
| **3D ExtraLarge** | V6 Direct Index | 2.86x | V5 Parallel Merge | 2.25x |

### Key Insights

1. **V5 Parallel Merge** excels on 2D workloads with consistent ~50% speedup
2. **V6 Direct Index** dominates on 3D workloads with 2.8-3.5x speedup
3. **V4 Hash** is only marginally better on 2D and performs poorly on 3D
4. **Optimized** version provides minimal improvement over baseline

### Recommendation

- **For 2D workloads**: Use **V5 Parallel Merge** algorithm
- **For 3D workloads**: Use **V6 Direct Index** algorithm
- Consider adaptive selection based on problem dimension for best overall performance

---

## Notes

- All benchmarks run with 3 repetitions per configuration
- Results show mean CPU time across repetitions
- CV (coefficient of variation) consistently < 2.5% showing good stability
- Debug build may not represent production performance but allows algorithm comparison
