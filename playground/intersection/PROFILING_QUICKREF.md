# Profiling Quick Reference: Intersection Algorithms

<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique
-->

Quick commands for common profiling tasks. See [PROFILING_STRATEGY.md](PROFILING_STRATEGY.md) for detailed guidance.

---

## 1. Quick Profiling (5 minutes)

### CPU Baseline
```bash
# Build
cmake --preset playground-perf-serial && cmake --build --preset playground-perf-serial

# Quick stat
perf stat -e cycles,instructions,cache-misses,branches,branch-misses \
  ./build-playground-perf-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_MediumConfig"
```

**Key Metrics:**
- IPC = instructions/cycles (target > 1.0)
- Cache miss rate < 10%
- Branch miss rate < 5%

### GPU Baseline
```bash
# Detect GPU first
nvidia-smi -L

# Build (use appropriate arch flag)
cmake --preset profiling-nsight-cuda -DKokkos_ARCH_ADA89=ON && cmake --build --preset profiling-nsight-cuda

# Quick timeline
./scripts/profiling/run_nsys.sh --benchmark "V1_MediumConfig"
```

---

## 2. Identify Bottlenecks (10 minutes)

### Kernel-Level Timing
```bash
cmake --preset playground-serial-profile && cmake --build --preset playground-serial-profile
./scripts/profile_benchmark.sh playground-serial-profile kernel-timer "V1_.*MediumConfig" -s 10
```

**Expected Phase Distribution:**
- Phase 1 (row_map): ~20%
- Phases 2+4 (count+fill): ~60%
- Phases 3+5 (scan+compact): ~20%

### Generate Flamegraph
```bash
# CPU only
perf record --call-graph dwarf \
  ./build-playground-perf-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_LargeConfig"
perf script | ./FlameGraph/stackcollapse-perf.pl | ./FlameGraph/flamegraph.pl > flamegraph.svg
```

---

## 3. GPU Deep Dive (30 minutes)

### Detailed GPU Metrics
```bash
# Fast basic profile (~190 metrics)
./scripts/profiling/run_ncu.sh --benchmark "V1_2D_MediumConfig" --section-set basic

# Detailed profile (~1200 metrics)
./scripts/profiling/run_ncu.sh --benchmark "V1_3D_LargeConfig" --section-set detailed

# Full profile (~5900 metrics, very slow)
./scripts/profiling/run_ncu.sh --benchmark "V1_LargeConfig" --section-set full
```

### Key GPU Metrics
```bash
# Memory vs Compute bound
ncu --metrics dram__throughput.avg.pct_of_peak,smsp__sass_thread_inst_executed_op_hadd_pred_on.sum \
  ./build-profiling-nsight-cuda/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_MediumConfig"

# Interpret:
# - DRAM > 80%, Inst < 30%: Memory bound
# - DRAM < 30%, Inst > 80%: Compute bound

# Warp efficiency
ncu --metrics smsp__thread_inst_executed_per_warp.avg.pct \
  ./build-profiling-nsight-cuda/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_LargeConfig"

# Occupancy
ncu --metrics smsp__occupancy.avg.pct_of_peak \
  ./build-profiling-nsight-cuda/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_MediumConfig"
```

---

## 4. Cross-Backend Comparison (15 minutes)

### All Backends
```bash
# Serial
cmake --preset playground-serial && cmake --build --preset playground-serial
./build-playground-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_.*LargeConfig" > serial.txt

# OpenMP
cmake --preset playground-openmp && cmake --build --preset playground-openmp
OMP_NUM_THREADS=22 ./build-playground-openmp/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_.*LargeConfig" > openmp.txt

# CUDA (with architecture)
cmake --preset playground-cuda -DKokkos_ARCH_ADA89=ON && cmake --build --preset playground-cuda
./build-playground-cuda/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_.*LargeConfig" > cuda.txt
```

### Speedup Calculation
```python
speedup_openmp = time_serial / time_openmp
speedup_cuda = time_serial / time_cuda
efficiency_openmp = speedup_openmp / 22  # cores
```

---

## 5. Memory Profiling (10 minutes)

### Memory Access Patterns
```bash
# CPU
perf stat -e cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses \
  ./build-playground-perf-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_LargeConfig"

# GPU coalescing
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
  ./build-profiling-nsight-cuda/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_LargeConfig"
# Calculate: sectors/request (should be ~1.0 for good coalescing)
```

### Memory Usage
```bash
# High-water mark
cmake --preset playground-serial-profile && cmake --build --preset playground-serial-profile
export KOKKOS_PROFILE_LIBRARY=./build-playground-serial-profile/_deps/kokkos_tools-build/profiling/memory-hwm/libkp_hwm_fixed.so
./build-playground-serial-profile/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="V1_LargeConfig"
# Look for: "KokkosP: High water mark memory consumption: XXX kB"
```

---

## 6. Decision Tree

```
Quick Check (5 min):
├─ IPC < 0.8? → Memory bandwidth issue → Go to Memory Profiling
├─ Cache miss > 10%? → Cache inefficient → Restructure data layout
└─ Branch miss > 5%? → Branch misprediction → Restructure conditionals

Kernel-Timer (10 min):
├─ Phase 1 > 30%? → Row mapping overhead
│  └─ Dense mesh? → Hash-based lookup (2-10x speedup)
├─ Phases 2+4 > 70%? → Two-pointer merge dominates
│  ├─ Memory bound? → Memory access pattern
│  └─ Compute bound? → Optimize merge logic
└─ Phases 3+5 > 20%? → Scan/compact overhead
   └─ In-place compaction (1.3-1.5x speedup)

GPU (30 min):
├─ Occupancy < 50%? → Increase block size (Easy win, 2-5x)
├─ Sectors/request > 2? → Improve coalescing (1.5-3x)
└─ Warp efficiency < 70%? → Reduce divergence (1.2-1.5x)
```

---

## 7. Quick Wins Priority

| Priority | Check | Command | Fix | Impact |
|----------|-------|---------|-----|--------|
| 1 | GPU occupancy | `ncu --metrics smsp__occupancy.avg.pct_of_peak` | Increase block size | 2-5x |
| 2 | GPU coalescing | `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | Restructure accesses | 1.5-3x |
| 3 | Phase 2/4 dominates | Kernel-timer | Optimize merge | 1.2-1.5x |
| 4 | Phase 5 dominates | Kernel-timer | In-place compact | 1.3-1.5x |
| 5 | Phase 1 dominates + dense | Kernel-timer | Hash lookup | 2-10x |

---

## 8. Common Issues

| Issue | Symptom | Check | Fix |
|-------|---------|-------|-----|
| Memory bandwidth saturation | IPC < 0.8 | `perf stat -e cycles,instructions` | Reduce memory footprint |
| Poor cache utilization | L1 miss > 10% | `perf stat -e L1-dcache-load-misses` | Restructure data layout |
| GPU low occupancy | Occupancy < 50% | `ncu --metrics smsp__occupancy.avg.pct_of_peak` | Increase block size |
| Memory not coalesced | sectors/request > 2 | `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | Fix access pattern |
| Row mapping overhead | Phase 1 > 30% | Kernel-timer | Hash table (if dense) |

---

## 9. Benchmark Configurations

| Config | 2D Rows | 3D Rows | Use For |
|--------|---------|---------|---------|
| Small | ~19 | ~1,229 | Quick testing |
| Medium | ~154 | ~78,643 | Development |
| Large | ~1,229 | ~5.0M | Performance |
| ExtraLarge | ~1,229 | ~10M | GPU stress |

**Benchmark Names:**
- 2D: `V1_SmallConfig`, `V1_MediumConfig`, `V1_LargeConfig`
- 3D: `V1_3D_SmallConfig`, `V1_3D_MediumConfig`, `V1_3D_LargeConfig`

---

## 10. One-Liners

### Everything at once
```bash
# Build and profile CPU
cmake --preset playground-perf-serial && cmake --build --preset playground-perf-serial && \
  perf stat -e cycles,instructions,cache-misses,branches,branch-misses \
    ./build-playground-perf-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
    --benchmark_filter="V1_MediumConfig"

# Build and profile GPU
nvidia-smi -L && \
  cmake --preset profiling-nsight-cuda -DKokkos_ARCH_ADA89=ON && cmake --build --preset profiling-nsight-cuda && \
  ./scripts/profiling/run_ncu.sh --benchmark "V1_MediumConfig" --section-set detailed

# Kernel timing
cmake --preset playground-serial-profile && cmake --build --preset playground-serial-profile && \
  ./scripts/profile_benchmark.sh playground-serial-profile kernel-timer "V1_.*MediumConfig" -s 5
```

### Compare versions
```bash
./build-playground-serial/playground/intersection/benchmarks/unified_comparison_benchmark \
  --benchmark_filter=".*_MediumConfig" | grep -E "V[123]_"
```

### Check parallel efficiency
```bash
echo "1 thread:" && OMP_NUM_THREADS=1 ./build-playground-openmp/playground/intersection/benchmarks/unified_comparison_benchmark --benchmark_filter="V1_MediumConfig" && \
  echo "22 threads:" && OMP_NUM_THREADS=22 ./build-playground-openmp/playground/intersection/benchmarks/unified_comparison_benchmark --benchmark_filter="V1_MediumConfig"
```

---

## Reference Files

- **Detailed Strategy**: `PROFILING_STRATEGY.md`
- **Main Profiling Guide**: `/home/sbstndbs/subsetix_kokkos/PROFILING.md`
- **perf Guide**: `/home/sbstndbs/subsetix_kokkos/docs/PERF_PROFILING.md`
- **Algorithm Source**: `/home/sbstndbs/subsetix_kokkos/playground/intersection/include/playground/subsetix/csr/intersection/algorithm/v1.hpp`

---

**Version:** 1.0
**For:** Intersection algorithms in `playground/intersection/`
