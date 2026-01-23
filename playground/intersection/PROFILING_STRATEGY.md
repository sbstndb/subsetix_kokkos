# Comprehensive Profiling Strategy: Playground Intersection Algorithms

<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique
-->

This document provides a systematic profiling strategy for the intersection algorithms in `playground/intersection/`. It covers CPU profiling (Linux perf), GPU profiling (Nsight), and Kokkos profiling tools with specific focus on the 5-phase intersection algorithm.

---

## Table of Contents

1. [Algorithm Overview](#algorithm-overview)
2. [CPU Profiling Strategy](#cpu-profiling-strategy)
3. [GPU Profiling Strategy](#gpu-profiling-strategy)
4. [Kokkos Profiling Strategy](#kokkos-profiling-strategy)
5. [Benchmark Interpretation](#benchmark-interpretation)
6. [Profiling Checklist](#profiling-checklist)
7. [Quick Wins Identification](#quick-wins-identification)
8. [Advanced Analysis](#advanced-analysis)

---

## Algorithm Overview

### v1 Intersection Algorithm (5-Phase)

The v1 algorithm (`/home/sbstndbs/subsetix_kokkos/playground/intersection/include/playground/subsetix/csr/intersection/algorithm/v1.hpp`) implements:

1. **Phase 1: Row Mapping** - Binary search to find common rows (O(n log n))
   - Kernels: `intersection_row_map_2d`, `intersection_row_map_3d`
   - Operations: Binary search per row, flag setting

2. **Phase 2: Count** - Count intersecting intervals per matched row
   - Kernel: `intersection_count`
   - Operations: Two-pointer merge per row (count-only mode)

3. **Phase 3: Scan** - Parallel prefix sum for CSR offsets
   - Kernel: `intersection_scan`
   - Operations: Kokkos parallel_scan

4. **Phase 4: Fill** - Write intersected intervals
   - Kernel: `intersection_fill`
   - Operations: Two-pointer merge per row (write mode)

5. **Phase 5: Compact** - Remove empty rows
   - Kernels: `intersection_mark_rows`, `intersection_compact_scan`, `intersection_compact_copy`, `intersection_compact_final_ptr`, `intersection_compact_intervals`

### Benchmark Configurations

| Config | 2D Rows | 3D Rows | Use Case |
|--------|---------|---------|----------|
| Small | ~19 | ~1,229 | Fast iteration, CI, debugging |
| Medium | ~154 | ~78,643 | Development testing |
| Large | ~1,229 | ~5.0M | Performance benchmarks |
| ExtraLarge | ~1,229 | ~10M | GPU stress tests |

---

## CPU Profiling Strategy

### 1. Quick Performance Baseline

Start with real-time statistics to identify bottlenecks:

```bash
# Build with perf support
cmake --preset playground-perf-serial
cmake --build --preset playground-perf-serial

# Quick stat for Small 2D
perf stat -e cycles,instructions,cache-misses,branches,branch-misses \
  ./build-playground-perf-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_SmallConfig"

# Quick stat for Large 2D
perf stat -e cycles,instructions,cache-misses,branches,branch-misses \
  ./build-playground-perf-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_LargeConfig"
```

**Key Metrics to Check:**
- **IPC (Instructions Per Cycle)**: `instructions / cycles`
  - Good: > 1.5 (well-optimized CPU code)
  - Acceptable: 0.8 - 1.5
  - Poor: < 0.8 (indicates memory bottlenecks or inefficient code)
- **Cache Miss Rate**: `cache-misses / cache-references`
  - Good: < 5%
  - Warning: 5-10%
  - Critical: > 10%
- **Branch Misprediction Rate**: `branch-misses / branches`
  - Good: < 2%
  - Warning: 2-5%
  - Critical: > 5%

### 2. Detailed CPU Profiling with Call Graphs

For deeper analysis, capture detailed profiling data:

```bash
# Profile specific configuration
./scripts/profile_benchmark_perf.sh playground-perf-serial Small 2D

# Or manually
perf record --call-graph dwarf \
  -o perf_output/v1_small_2d.data \
  ./build-playground-perf-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_SmallConfig"

# View report
perf report -i perf_output/v1_small_2d.data
```

**Navigation Tips:**
- `Enter` - Expand function
- `+` - Zoom into function
- `-` - Zoom out
- `H` - Sort by hot paths
- `q` - Quit

### 3. Phase-Specific Profiling

Profile each phase individually to identify bottlenecks:

```bash
# Phase 1 focus: Row mapping
perf record -e cycles,instructions,L1-dcache-load-misses \
  --call-graph dwarf \
  ./build-playground-perf-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_MediumConfig"

# Annotate row intersection function
perf annotate -s playground::subsetix::csr::intersection::v1::intersect_meshes
```

**What to Look For:**

**Phase 1 (Row Mapping):**
- High cache misses in binary search
- Consideration: Use hash-based lookup for dense meshes

**Phase 2 & 4 (Two-Pointer Merge):**
- Branch mispredictions in merge loop
- IPC < 1.0 indicates memory-bound code
- Cache line utilization: Each interval is 8 bytes (2× int32_t)

**Phase 3 & 5 (Scan/Compact):**
- Scan efficiency: Should be memory-bandwidth bound
- Look for high IPC (> 1.5) indicating compute waste

### 4. Memory Access Pattern Analysis

```bash
# Detailed cache analysis
perf record -e cache-references,cache-misses,L1-dcache-load-misses,L1-dcache-loads,LLC-load-misses \
  --call-graph dwarf \
  ./build-playground-perf-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_MediumConfig"

# Check memory bandwidth
perf stat -e cycles,instructions,cache-misses,cache-references \
  -e stall-cycles-frontend,stall-cycles-backend \
  ./build-playground-perf-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_LargeConfig"
```

**Metrics Interpretation:**
- **Frontend Stalls**: Instruction fetch/cache issues
- **Backend Stalls**: Data dependencies/memory issues
- **High Backend Stalls + Low IPC**: Memory bottleneck

### 5. Flamegraph Generation

```bash
# Install FlameGraph tools (if not already installed)
git clone https://github.com/brendangregg/FlameGraph.git

# Generate flamegraph
perf script -i perf_output/v1_medium_2d.data | \
  ./FlameGraph/stackcollapse-perf.pl | \
  ./FlameGraph/flamegraph.pl > perf_output/v1_medium_2d_flamegraph.svg

# Open in browser
firefox perf_output/v1_medium_2d_flamegraph.svg
```

**What to Look For:**
- Wide bars: Hot functions consuming most CPU time
- Deep stacks: Call overhead
- Function names: Identify kernel phases

### 6. Advanced perf Events

```bash
# TLB misses (address translation)
perf stat -e dTLB-load-misses,dTLB-loads,iTLB-load-misses \
  ./build-playground-perf-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_LargeConfig"

# SIMD efficiency
perf stat -e simd_instructions,simd_instructions_retired \
  ./build-playground-perf-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_MediumConfig"
```

---

## GPU Profiling Strategy

### 1. GPU Architecture Detection

**CRITICAL**: Always detect GPU architecture first:

```bash
nvidia-smi -L

# Then use appropriate architecture flag
cmake --preset playground-cuda -DKokkos_ARCH_ADA89=ON  # RTX 40xx
cmake --build --preset playground-cuda
```

### 2. Nsight Compute (ncu) - Deep Kernel Analysis

For detailed GPU kernel profiling:

```bash
# Configure with Nsight support
cmake --preset profiling-nsight-cuda -DKokkos_ARCH_ADA89=ON
cmake --build --preset profiling-nsight-cuda

# Basic profiling (fast, ~190 metrics)
./scripts/profiling/run_ncu.sh --benchmark "V1_2D_MediumConfig" --section-set basic

# Detailed profiling (recommended, ~1200 metrics)
./scripts/profiling/run_ncu.sh --benchmark "V1_3D_LargeConfig" --section-set detailed

# Full profiling (comprehensive, ~5900 metrics, very slow)
./scripts/profiling/run_ncu.sh --benchmark "V1_2D_LargeConfig" --section-set full
```

**Or manual invocation:**

```bash
# Profile specific kernel
ncu --set detailed \
  --section WarpStateStats \
  --section InstructionStats \
  -o profiling_output_ncu/intersection_v1_medium \
  ./build-profiling-nsight-cuda/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_MediumConfig"

# View results
ncu --import profiling_output_ncu/intersection_v1_medium.ncu-rep
```

### 3. Key GPU Metrics

#### 3.1 Memory Bandwidth vs Compute Bound

**Metric**: `dram__throughput.avg.pct_of_peak` and `smsp__sass_thread_inst_executed_op_hadd_pred_on.sum`

```bash
ncu --metrics dram__throughput.avg.pct_of_peak,smsp__sass_thread_inst_executed_op_hadd_pred_on.sum \
  ./build-profiling-nsight-cuda/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_MediumConfig"
```

**Interpretation:**

| DRAM Throughput | Instruction Throughput | Conclusion |
|-----------------|------------------------|------------|
| > 80% | < 30% | **Memory Bound** - Focus on memory access patterns |
| < 30% | > 80% | **Compute Bound** - Focus on algorithm efficiency |
| 50-80% | 30-80% | **Balanced** - Both matter |

**For Intersection Algorithm:**
- **Phases 1, 3, 5** (scan/compact): Should be memory-bound
- **Phases 2, 4** (merge): Could be compute or memory bound depending on interval count

#### 3.2 Warp Efficiency

**Metrics**: `smsp__thread_inst_executed_per_warp.avg.pct`

```bash
ncu --metrics smsp__thread_inst_executed_per_warp.avg.pct \
  --section WarpStateStats \
  ./build-profiling-nsight-cuda/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_LargeConfig"
```

**Interpretation:**
- **> 90%**: Excellent warp utilization
- **70-90%**: Good (some divergence expected in row intersection)
- **< 70%**: Poor (branch divergence or load imbalance)

**Intersection Algorithm Notes:**
- Binary search in Phase 1 causes some divergence
- Two-pointer merge in Phases 2/4 has varying loop counts
- This is expected - focus on overall throughput

#### 3.3 Occupancy

**Metrics**: `smsp__occupancy.avg.pct_of_peak`

```bash
ncu --metrics smsp__occupancy.avg.pct_of_peak \
  ./build-profiling-nsight-cuda/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_MediumConfig"
```

**Interpretation:**
- **> 75%**: Good resource utilization
- **50-75%**: Acceptable
- **< 50%**: Underutilized (increase thread block size or reduce register pressure)

#### 3.4 Memory Coalescing

**Metrics**: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`,
`l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`

**Calculate**: Sectors per request (should be ~1 for good coalescing)

```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
  ./build-profiling-nsight-cuda/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_LargeConfig"
```

**Interpretation:**
- **~1.0**: Perfect coalescing
- **1.0-2.0**: Good
- **> 2.0**: Poor (memory access pattern needs work)

### 4. Nsight Systems (nsys) - Timeline Analysis

For system-wide timeline and CPU/GPU overlap:

```bash
# Quick timeline profiling
./scripts/profiling/run_nsys.sh --benchmark "V1_3D_MediumConfig"

# Or manually
nsys profile -o profiling_output/intersection_v1_timeline \
  --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  ./build-profiling-nsight-cuda/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_MediumConfig"

# View GUI
nsys-ui profiling_output/intersection_v1_timeline.nsys-rep
```

**What to Look For:**
- **Kernel durations**: Identify longest kernels
- **CPU-GPU overlap**: Check for gaps (compute/memory transfers)
- **Memory copy overhead**: H2D/D2H transfer time
- **Kernel launch overhead**: Time between launch and execution

### 5. Kernel-Specific Profiling

Profile individual phases:

```bash
# Profile row mapping kernel (Phase 1)
ncu -k "intersection_row_map_2d" \
  --set detailed \
  -o profiling_output_ncu/phase1_row_map \
  ./build-profiling-nsight-cuda/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_MediumConfig"

# Profile count kernel (Phase 2)
ncu -k "intersection_count" \
  --set detailed \
  -o profiling_output_ncu/phase2_count \
  ./build-profiling-nsight-cuda/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_MediumConfig"

# Profile fill kernel (Phase 4)
ncu -k "intersection_fill" \
  --set detailed \
  -o profiling_output_ncu/phase4_fill \
  ./build-profiling-nsight-cuda/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_MediumConfig"
```

### 6. Roofline Analysis

```bash
# Generate roofline data
ncu --set roofline \
  -o profiling_output_ncu/roofline_v1 \
  ./build-profiling-nsight-cuda/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_LargeConfig"

# View in ncu UI
ncu --import profiling_output_ncu/roofline_v1.ncu-rep
```

**Interpretation:**
- Points on left of roofline: Memory bandwidth limited
- Points on right of roofline: Compute limited
- Aim to move kernels toward roofline peak

---

## Kokkos Profiling Strategy

### 1. Kernel-Level Timing (kernel-timer)

Best for quantitative timing of each Kokkos kernel:

```bash
# Build with Kokkos tools
cmake --preset playground-serial-profile
cmake --build --preset playground-serial-profile

# Profile with kernel-timer
./scripts/profile_benchmark.sh playground-serial-profile kernel-timer "V1_.*MediumConfig"

# Profile with sampling (reduce overhead)
./scripts/profile_benchmark.sh playground-serial-profile kernel-timer "V1_LargeConfig" -s 5

# Convert .dat to JSON for analysis
BUILD_DIR="./build-playground-serial-profile"
BUILD_DIR/_deps/kokkos_tools-build/profiling/simple-kernel-timer/kp_json_writer \
  profiling_output/<timestamp>-kernel-timer/*.dat
```

**Output Analysis:**
```
Kernel Name                      |  Calls  |  Total Time (s) |  Avg Time (s) |  Max Time (s)
---------------------------------|---------|-----------------|---------------|---------------
intersection_row_map_2d          |    50   |        0.125    |     0.0025    |       0.0030
intersection_count               |    50   |        0.450    |     0.0090    |       0.0095
intersection_scan                |    50   |        0.075    |     0.0015    |       0.0018
intersection_fill                |    50   |        0.380    |     0.0076    |       0.0080
intersection_compact_*           |   200   |        0.120    |     0.0006    |       0.0010
```

**What to Look For:**
- **Phases 2 & 4** (count/fill) should dominate
- **Phase 1** (row mapping) should be < 20% of total
- **Phase 5** (compact) should be < 10% of total
- **High variance** in timing: Check for load imbalance

### 2. Chrome Tracing (Timeline Visualization)

Best for visualizing kernel execution order:

```bash
# Profile with chrome-tracing
./scripts/profile_benchmark.sh playground-openmp-profile chrome-tracing "V1_2D_LargeConfig" -t 22 -s 10

# Open in Chrome
# 1. Open chrome://tracing
# 2. Click "Load"
# 3. Select profiling_output/<timestamp>-chrome-tracing/*.json
```

**What to Look For:**
- **Sequential vs parallel**: Kernels should execute in parallel (OpenMP)
- **Gaps**: Idle time between kernels
- **Kernel overlap**: Good for CPU-GPU overlap (CUDA)
- **Long-running kernels**: Candidates for optimization

### 3. Space-Time Stack (Memory + Time)

Best for memory hierarchy analysis:

```bash
# Profile with space-time-stack (heavy overhead, use sampling)
./scripts/profile_benchmark.sh playground-serial-profile space-time-stack "V1_MediumConfig" -s 5

# View report
cat profiling_output/<timestamp>-space-time-stack/benchmark_output.txt
```

**Sample Output:**
```
======================================================================
KokkosP: Space-Time Stack Profiling Report
======================================================================

Kernel: intersection_count
  Total Time: 8.92 ms
  Time in Kokkos: 8.90 ms (99.8%)

  Memory Spaces:
    HostSpace: 8.90 ms
      - allocations: 2
      - deallocations: 1

  Nested Kernels:
    - None

======================================================================
```

**What to Look For:**
- **Memory space usage**: Device vs Host
- **Allocation overhead**: Frequent allocations in kernels
- **Nested parallelism**: Opportunity for optimization

### 4. Memory Profiling

Track memory usage and high-water mark:

```bash
# Memory high-water mark
export KOKKOS_PROFILE_LIBRARY=./build-playground-serial-profile/_deps/kokkos_tools-build/profiling/memory-hwm/libkp_hwm_fixed.so
./build-playground-serial-profile/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_LargeConfig"

# Look for: "KokkosP: High water mark memory consumption: XXX kB"
```

**What to Check:**
- **Memory growth**: Allocations per iteration
- **Peak memory**: Should be stable across iterations
- **Memory leaks**: Increasing HWM over time

### 5. Kokkos Profiling Region Annotation

For custom profiling regions in the code:

```cpp
// In v1.hpp, add profiling regions
#include <Kokkos_Profiling_ProfileSection.hpp>

template <int DIM, class CoordType, class IndexType>
inline Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>
intersect_meshes(const Mesh<...>& A, const Mesh<...>& B) {
  Kokkos::Profiling::ProfilingSection phase1("intersection_phase1_row_map");
  // Phase 1 code...
  phase1.stop();

  Kokkos::Profiling::ProfilingSection phase2("intersection_phase2_count");
  // Phase 2 code...
  phase2.stop();

  // ... etc for all phases
}
```

**Then profile with chrome-tracing to see custom regions.**

---

## Benchmark Interpretation

### 1. Scaling Analysis

Compare across configurations to identify scaling bottlenecks:

```bash
# Run all configurations
./build-playground-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_.*Config" \
  --benchmark_format=json > benchmark_results.json

# Or use convenience script
./scripts/profile_all_benchmarks.sh playground-serial
```

**Metrics to Extract:**

```python
import json

with open('benchmark_results.json') as f:
    data = json.load(f)

for benchmark in data['benchmarks']:
    name = benchmark['name']
    time = benchmark['real_time']
    items = benchmark['items']  # intervals processed
    bytes_processed = benchmark['bytes']

    # Calculate derived metrics
    intervals_per_sec = items / time * 1e9  # ns to s
    bandwidth_gb_s = bytes_processed / time * 1e9 / 1e9

    print(f"{name}:")
    print(f"  Time: {time:.2f} ns")
    print(f"  Throughput: {intervals_per_sec:.2f} intervals/s")
    print(f"  Bandwidth: {bandwidth_gb_s:.2f} GB/s")
```

### 2. Algorithmic Complexity Analysis

Plot runtime vs input size:

| Config | 2D Rows | 3D Rows | Expected Complexity |
|--------|---------|---------|---------------------|
| Small | ~19 | ~1,229 | O(n log n) - binary search dominated |
| Medium | ~154 | ~78,643 | O(n) - merge dominated |
| Large | ~1,229 | ~5.0M | O(n) - merge dominated |

**Expected Scaling:**
- **Phase 1** (row mapping): O(n log n) where n = rows in A
- **Phases 2,4** (merge): O(m + n) where m,n = intervals in matched rows
- **Phases 3,5** (scan/compact): O(k) where k = output rows

**Check for:**
- **Super-linear scaling**: Cache issues, memory bandwidth saturation
- **Sub-linear scaling**: Parallel efficiency loss

### 3. Version Comparison (v1 vs v2 vs v3)

```bash
# Run all versions
./build-playground-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter=".*_MediumConfig"
```

**Output Comparison:**
```
Benchmark                           Time           CPU       Iterations
---------------------------------------------------------------------
V1_2D_MediumConfig               45230 ns      45180 ns        15678
V2_2D_MediumConfig               45180 ns      45200 ns        15702
V3_2D_MediumConfig               45250 ns      45190 ns        15695

V1_3D_MediumConfig             8924500 ns    8921000 ns           78
V2_3D_MediumConfig             8923000 ns    8922000 ns           78
V3_3D_MediumConfig             8924000 ns    8921000 ns           78
```

**What to Look For:**
- **Identical timings**: v2/v3 are copies of v1 (expected)
- **> 5% difference**: Actual algorithm differences
- **High variance**: Instability in implementation

### 4. Backend Comparison (Serial vs OpenMP vs CUDA)

```bash
# Serial
./build-playground-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_.*LargeConfig" > serial.txt

# OpenMP
OMP_NUM_THREADS=22 ./build-playground-openmp/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_.*LargeConfig" > openmp.txt

# CUDA
./build-playground-cuda/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_.*LargeConfig" > cuda.txt
```

**Speedup Calculation:**
- **OpenMP Speedup**: Serial Time / OpenMP Time
  - Expected: 10-20x on 22 cores (memory bandwidth limited)
- **CUDA Speedup**: Serial Time / CUDA Time
  - Expected: 50-200x depending on GPU

### 5. Bottleneck Identification

For each configuration, identify the dominant phase:

```bash
# Profile with kernel-timer
./scripts/profile_benchmark.sh playground-serial-profile kernel-timer "V1_MediumConfig"

# Parse output
# Example: phase2_count is 40% of total time
```

**Decision Tree:**

```
Is Phase 1 (row_map) dominant (> 30%)?
├─ Yes: Binary search overhead
│  ├─ For dense meshes: Consider hash-based lookup
│  └─ For sparse meshes: Acceptable
└─ No: Check Phase 2/4

Are Phases 2/4 (count/fill) dominant (> 50%)?
├─ Yes: Two-pointer merge is the bottleneck
│  ├─ Memory bound: Check memory access patterns
│  └─ Compute bound: Check branch efficiency
└─ No: Check Phase 3/5

Are Phases 3/5 (scan/compact) dominant (> 20%)?
├─ Yes: Scan/compact overhead
│  └─ Consider in-place compaction
└─ Balanced algorithm
```

---

## Profiling Checklist

### Phase 1: Initial Profiling (Quick Check)

**Goal:** Identify gross bottlenecks in < 30 minutes

- [ ] **Build with profiling support**
  ```bash
  cmake --preset playground-perf-serial
  cmake --build --preset playground-perf-serial
  ```

- [ ] **Run baseline benchmarks**
  ```bash
  ./build-playground-perf-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
    --benchmark_filter="V1_.*MediumConfig"
  ```

- [ ] **Quick perf stat check**
  ```bash
  perf stat -e cycles,instructions,cache-misses,branches,branch-misses \
    ./build-playground-perf-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
    --benchmark_filter="V1_MediumConfig"
  ```

- [ ] **Check key metrics**
  - [ ] IPC > 1.0?
  - [ ] Cache miss rate < 10%?
  - [ ] Branch misprediction < 5%?

- [ ] **Quick kernel-timer profiling**
  ```bash
  cmake --preset playground-serial-profile
  cmake --build --preset playground-serial-profile
  ./scripts/profile_benchmark.sh playground-serial-profile kernel-timer "V1_MediumConfig" -s 10
  ```

- [ ] **Identify dominant phase**
  - [ ] Phase 1 (row_map) < 20%?
  - [ ] Phases 2/4 (count/fill) < 70%?
  - [ ] Phases 3/5 (scan/compact) < 10%?

### Phase 2: Deep CPU Profiling (Detailed Analysis)

**Goal:** Understand CPU-level bottlenecks in 1-2 hours

- [ ] **Detailed perf recording**
  ```bash
  perf record --call-graph dwarf \
    -o perf_output/v1_detailed.data \
    ./build-playground-perf-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
    --benchmark_filter="V1_LargeConfig"
  ```

- [ ] **Generate flamegraph**
  ```bash
  perf script -i perf_output/v1_detailed.data | \
    ./FlameGraph/stackcollapse-perf.pl | \
    ./FlameGraph/flamegraph.pl > v1_flamegraph.svg
  ```

- [ ] **Annotate hot functions**
  ```bash
  perf annotate -s playground::subsetix::csr::intersection::v1::intersect_meshes
  ```

- [ ] **Check memory access patterns**
  ```bash
  perf stat -e cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses \
    ./build-playground-perf-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
    --benchmark_filter="V1_LargeConfig"
  ```

- [ ] **Document findings**
  - [ ] Hot functions list
  - [ ] Cache miss hotspots
  - [ ] Branch misprediction locations

### Phase 3: GPU Profiling (CUDA Only)

**Goal:** Optimize GPU kernels in 2-4 hours

- [ ] **Detect GPU architecture**
  ```bash
  nvidia-smi -L
  # Use appropriate -DKokkos_ARCH_XXX=ON flag
  ```

- [ ] **Build with Nsight support**
  ```bash
  cmake --preset profiling-nsight-cuda -DKokkos_ARCH_ADA89=ON
  cmake --build --preset profiling-nsight-cuda
  ```

- [ ] **Quick Nsight Systems timeline**
  ```bash
  ./scripts/profiling/run_nsys.sh --benchmark "V1_MediumConfig"
  ```

- [ ] **Detailed Nsight Compute profiling**
  ```bash
  ./scripts/profiling/run_ncu.sh --benchmark "V1_3D_MediumConfig" --section-set detailed
  ```

- [ ] **Check key GPU metrics**
  - [ ] Memory bandwidth utilization: `dram__throughput.avg.pct_of_peak`
  - [ ] Warp efficiency: `smsp__thread_inst_executed_per_warp.avg.pct`
  - [ ] Occupancy: `smsp__occupancy.avg.pct_of_peak`
  - [ ] Memory coalescing: Sectors per request ~1.0?

- [ ] **Identify bottleneck type**
  - [ ] Memory-bound vs Compute-bound?
  - [ ] Kernel with lowest efficiency?
  - [ ] Divergent warps?

### Phase 4: Cross-Backend Comparison

**Goal:** Understand scaling across backends

- [ ] **Build all backends**
  ```bash
  cmake --preset playground-serial && cmake --build --preset playground-serial
  cmake --preset playground-openmp && cmake --build --preset playground-openmp
  cmake --preset playground-cuda -DKokkos_ARCH_ADA89=ON && cmake --build --preset playground-cuda
  ```

- [ ] **Run benchmarks on all backends**
  ```bash
  for preset in serial openmp cuda; do
    ./build-playground-${preset}/playground/intersection/benchmarks/unified_comparison_benchmark \
      --benchmark_filter="V1_.*LargeConfig" > ${preset}_results.txt
  done
  ```

- [ ] **Calculate speedups**
  - [ ] OpenMP vs Serial
  - [ ] CUDA vs Serial
  - [ ] CUDA vs OpenMP

- [ ] **Check parallel efficiency**
  - [ ] OpenMP: Speedup / cores
  - [ ] CUDA: Speedup vs theoretical peak

### Phase 5: Optimization Iteration

**Goal:** Targeted optimization based on findings

- [ ] **Choose optimization target**
  - [ ] Dominant phase identified?
  - [ ] Bottleneck type (memory/compute)?
  - [ ] Backend-specific optimization?

- [ ] **Implement optimization**
  - [ ] Code change
  - [ ] Add benchmark guard
  - [ ] Update documentation

- [ ] **Validate**
  - [ ] Run same profiling steps
  - [ ] Compare before/after
  - [ ] Check correctness

- [ ] **Document**
  - [ ] Performance improvement
  - [ ] Trade-offs
  - [ ] Recommendations

---

## Quick Wins Identification

### 1. Most Impactful Metrics (Pareto Principle)

Focus on metrics that yield 80% of optimization gains:

| Metric | Impact | Ease | Priority |
|--------|--------|------|----------|
| **Memory bandwidth utilization** | High | Medium | **HIGH** |
| **Cache miss rate** | High | Low | Medium |
| **Branch misprediction** | Medium | Low | Medium |
| **Occupancy (GPU)** | Medium | High | **HIGH** |
| **Kernel launch overhead** | Low | High | Low |

### 2. Rapid Optimization Checks

**< 5 minutes each:**

#### Check 1: Memory Bandwidth Saturation
```bash
# Serial/OpenMP: Check IPC
perf stat -e cycles,instructions \
  ./build-playground-perf-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_LargeConfig"
# If IPC < 0.8: Memory bandwidth limited

# CUDA: Check DRAM throughput
ncu --metrics dram__throughput.avg.pct_of_peak \
  ./build-profiling-nsight-cuda/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_LargeConfig"
# If < 60%: Not bandwidth limited, check compute
```

**Quick Win:** If memory bandwidth limited, focus on:
- Better cache utilization
- Memory access coalescing (GPU)
- Reducing memory footprint

#### Check 2: Kernel Balance
```bash
./scripts/profile_benchmark.sh playground-serial-profile kernel-timer "V1_MediumConfig" -s 10
```

**Quick Win:** If one phase dominates (> 60%):
- Focus optimization efforts on that phase
- Other phases have limited impact

#### Check 3: Parallel Efficiency
```bash
# Single thread
OMP_NUM_THREADS=1 ./build-playground-openmp/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_MediumConfig" > 1thread.txt

# All threads
OMP_NUM_THREADS=22 ./build-playground-openmp/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_MediumConfig" > 22threads.txt

# Compare
# Speedup = time(1thread) / time(22threads)
# Efficiency = speedup / 22
```

**Quick Win:** If efficiency < 50%:
- Check for false sharing
- Check for load imbalance
- Check for serialization

### 3. Common Bottleneck Patterns

| Pattern | Symptom | Fix | Complexity |
|---------|---------|-----|------------|
| **Row mapping dominates** | Phase 1 > 30% | Hash table for dense meshes | Medium |
| **High cache misses** | L1 miss rate > 10% | Restructure data layout | Hard |
| **Poor GPU occupancy** | Occupancy < 50% | Increase block size | Easy |
| **Branch divergence** | Warp efficiency < 70% | Restructure conditionals | Medium |
| **Low parallel efficiency** | Efficiency < 40% | Load balancing | Hard |

### 4. Decision Matrix

**When to optimize what:**

```
IF Phase 1 (row_map) > 30%:
  AND mesh is dense (> 50% rows):
    → Hash-based row lookup (Medium effort)
  ELSE:
    → Accept as cost of sparsity

IF Phases 2/4 (count/fill) > 60%:
  AND memory bandwidth saturated:
    → Better memory access pattern (Hard)
  AND compute bound:
    → Optimize merge logic (Medium)
  AND GPU occupancy < 50%:
    → Increase thread block size (Easy)

IF Phases 3/5 (scan/compact) > 20%:
  → In-place compaction (Medium)
```

### 5. Optimization Priority Queue

Based on intersection algorithm analysis:

1. **GPU Occupancy** (if CUDA + occupancy < 50%)
   - Quick win: Adjust kernel launch parameters
   - Impact: 2-5x speedup potential

2. **Memory Access Coalescing** (if GPU + sectors/request > 2)
   - Quick win: Restructure array accesses
   - Impact: 1.5-3x speedup potential

3. **Phase 2/4 Merge Optimization** (if dominates)
   - Medium effort: Loop unrolling, SIMD
   - Impact: 1.2-1.5x speedup potential

4. **In-place Compaction** (if Phase 5 dominates)
   - Medium effort: Eliminate intermediate allocation
   - Impact: 1.3-1.5x speedup potential

5. **Hash-Based Row Lookup** (if Phase 1 dominates + dense)
   - Hard effort: New data structure
   - Impact: 2-10x speedup potential (specific case)

---

## Advanced Analysis

### 1. Roofline Modeling

For theoretical performance ceiling:

```python
# System parameters (example for RTX 4090)
peak_performance_gflops = 82900  # GFLOP/s
peak_bandwidth_gb_s = 1008       # GB/s

# Intersection algorithm arithmetic intensity
# Operations per interval: ~20-30 (two-pointer merge)
bytes_per_interval = 16  # 2× int32_t

arithmetic_intensity = 25 / 16  # flops/byte

# Theoretical performance bound
if arithmetic_intensity < peak_performance_gflops / peak_bandwidth_gb_s:
    # Memory-bound
    bound = arithmetic_intensity * peak_bandwidth_gb_s
else:
    # Compute-bound
    bound = peak_performance_gflops

print(f"Theoretical peak: {bound:.2f} GFLOP/s")
```

**Use Nsight roofline section for empirical data.**

### 2. Autocorrelation Analysis

Check for performance variability:

```bash
# Run multiple times
for i in {1..10}; do
  ./build-playground-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
    --benchmark_filter="V1_LargeConfig" --benchmark_repetitions=1
done | grep "V1_2D_LargeConfig" > timings.txt

# Check for variability
# If std/mean > 5%: System noise or thermal throttling
```

### 3. Strong vs Weak Scaling

**Strong Scaling** (fixed problem size):
```bash
for threads in 1 2 4 8 16 22; do
  OMP_NUM_THREADS=$threads \
    ./build-playground-openmp/playground/intersection/benchmarks/unified_comparison_benchmark \
    --benchmark_filter="V1_LargeConfig"
done
```

**Weak Scaling** (problem size scales with threads):
```bash
# Requires custom benchmark with scalable problem size
```

### 4. Amdahl's Law Analysis

Maximum speedup given serial portion:

```
Speedup_max = 1 / (S + P/N)

Where:
  S = Serial fraction (from profiling)
  P = Parallel fraction (1 - S)
  N = Number of processors

Example:
  Phase 1 (row_map) is 20% serial
  Phase 2-5 are 80% parallel

  On 22 cores:
  Speedup_max = 1 / (0.2 + 0.8/22) = 4.2x

  Actual: If achieving 3.5x, efficiency = 3.5/4.2 = 83%
```

---

## Summary

### Profiling Workflow (Recommended)

1. **Quick Start** (30 min)
   - Run baseline benchmarks
   - Check perf stat (IPC, cache, branch)
   - Identify dominant phase with kernel-timer

2. **Deep Dive** (2-4 hours)
   - CPU: perf record + flamegraph
   - GPU: ncu detailed profiling
   - Identify bottleneck type (memory/compute)

3. **Targeted Optimization** (varies)
   - Focus on dominant phase
   - Use quick wins checklist
   - Validate with same profiling

4. **Cross-Validation** (1-2 hours)
   - Compare across backends
   - Check scaling
   - Document findings

### Key Takeaways

- **Phase 2/4 (two-pointer merge)** typically dominates
- **Memory bandwidth** is usually the bottleneck
- **GPU occupancy** is an easy win if low
- **Cache misses** are hard but impactful
- **Profiling overhead** is significant (2-3x), account for it

### Common Pitfalls

- **Profiling with small inputs**: Results not representative
- **Ignoring variance**: Single run may be misleading
- **Over-optimizing**: < 5% improvements not worth it
- **GPU without architecture flag**: Wrong code generation
- **Forgetting to fence**: Kokkos async execution hides issues

---

**Document Version:** 1.0
**Last Updated:** 2025-01-23
**For:** Subsetix Kokkos Playground Intersection Module
