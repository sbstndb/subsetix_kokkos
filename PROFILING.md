# Profiling Guide

This guide explains how to profile the Subsetix Kokkos library using different tools for different analysis needs.

## Quick Start

### Choose Your Profiling Tool

| Tool | Best For | Backend | Overhead |
|------|----------|---------|----------|
| **Linux perf** | CPU performance analysis (hot paths, cache, branch prediction) | Serial, OpenMP | ~2.3x-2.5x |
| **Nsight (ncu/nsys)** | GPU kernel profiling and timeline visualization | CUDA | ~2.9x |
| **Kokkos tools** | Kernel-level timing and memory hierarchy analysis | All | ~2.3x-2.9x |

### Quick Commands

```bash
# Linux perf (Serial/OpenMP)
cmake --preset experimental-perf-serial && cmake --build --preset experimental-perf-serial
./scripts/perf_profile.sh ./build-experimental-perf-serial/experimental/benchmarks/experimental_comparison_benchmark

# Nsight (CUDA)
cmake --preset profiling-nsight-cuda-gcc12 && cmake --build --preset profiling-nsight-cuda-gcc12
./scripts/profiling/run_ncu.sh experimental_comparison_benchmark

# Kokkos profiling tools (All backends)
cmake --preset experimental-serial-profile && cmake --build --preset experimental-serial-profile
./scripts/profile_benchmark.sh experimental-serial-profile kernel-timer "SmallConfig"
```

---

## Profiling Tools

### 1. Linux perf

**Purpose**: Low-level CPU performance analysis with debug symbols.

**Supported backends**: Serial, OpenMP

**What it provides**:
- CPU cycles, instructions, cache misses
- Branch prediction statistics
- Call graph analysis (dwarf, fp, lbr)
- Flame graph generation

**Build presets**:
- `experimental-perf-serial` - Serial backend with perf support
- `experimental-perf-openmp` - OpenMP backend with perf support

**Usage**:
```bash
# Configure and build
cmake --preset experimental-perf-serial
cmake --build --preset experimental-perf-serial

# Profile a specific executable
./scripts/perf_profile.sh ./build-experimental-perf-serial/experimental/benchmarks/experimental_comparison_benchmark

# Profile all benchmarks
./scripts/profile_all_benchmarks.sh experimental-perf-serial

# Compare two runs
./scripts/compare_perf.sh ./perf_output/before.data ./perf_output/after.data
```

**Documentation**: See `docs/PERF_PROFILING.md` for detailed perf usage.

---

### 2. Nsight Compute (ncu) & Nsight Systems (nsys)

**Purpose**: GPU kernel profiling and system-wide timeline analysis.

**Supported backends**: CUDA

**What it provides**:
- Detailed GPU kernel analysis (ncu)
- System-wide timeline with CPU/GPU overlap (nsys)
- Memory bandwidth analysis
- Warp efficiency and occupancy metrics

**Build presets**:
- `profiling-nsight-cuda-gcc12` - RelWithDebInfo build
- `profiling-nsight-cuda-gcc12-release` - Release with debug symbols

**Usage**:
```bash
# Configure and build
cmake --preset profiling-nsight-cuda-gcc12
cmake --build --preset profiling-nsight-cuda-gcc12

# Run Nsight Compute (detailed kernel profiling)
./scripts/profiling/run_ncu.sh experimental_comparison_benchmark

# Run Nsight Systems (timeline)
./scripts/profiling/run_nsys.sh experimental_comparison_benchmark

# Quick profiling with SmallConfig
./scripts/profiling/run_nsys_quick.sh

# Analyze results
./scripts/profiling/analyze_results.sh
```

**Output**:
- NCU reports: `<benchmark>.ncu-rep`
- Nsys reports: `nsys_<filter>_<timestamp>.nsys-rep`

---

### 3. Kokkos Profiling Tools

**Purpose**: Kernel-level timing and memory hierarchy analysis via Kokkos tools.

**Supported backends**: Serial, OpenMP, CUDA

**Available tools**:

| Tool | Library | Output | Best For |
|------|---------|--------|----------|
| **kernel-timer** | `libkp_kernel_timer.so` | `.dat` + JSON | Quantitative kernel timing |
| **chrome-tracing** | `libkp_chrome_tracing.so` | `.json` | Timeline visualization |
| **space-time-stack** | `libkp_space_time_stack.so` | stdout | Time + memory hierarchy |
| **memory-hwm** | `libkp_hwm.so` | stdout | Memory high-water mark |
| **memory-usage** | `libkp_memory_usage.so` | stdout | Per-allocation tracking |

**Build presets**:
- `experimental-serial-profile` - Serial + Kokkos tools
- `experimental-openmp-profile` - OpenMP + Kokkos tools
- `experimental-cuda-gcc12-profile` - CUDA + Kokkos tools

**Usage**:
```bash
# Configure and build (downloads and builds kokkos-tools)
cmake --preset experimental-serial-profile
cmake --build --preset experimental-serial-profile

# Profile with a specific tool
./scripts/profile_benchmark.sh experimental-serial-profile kernel-timer "SmallConfig"

# Profile all backends
./scripts/profiling/profile_all_backends.sh kernel-timer

# Analyze traces
./scripts/profiling/analyze_traces.sh ./profiling_output/<timestamp>-kernel-timer/

# Compare multiple runs
./scripts/profiling/compare_runs.sh ./profiling_output/
```

**Environment variables**:
```bash
# Set profiling library
export KOKKOS_PROFILE_LIBRARY=<path-to-lib>/libkokkosp.so

# Or use KOKKOS_TOOLS_LIBS (alternative)
export KOKKOS_TOOLS_LIBS=<path-to-lib>/libkp_kernel_timer.so
```

**Output organization**:
```
profiling_output/
└── <timestamp>-<tool>/
    ├── Serial/
    ├── OpenMP/
    └── CUDA/
```

**Documentation**: See `scripts/profiling/README.md` for detailed script usage.

---

## Important Notes

### Profiling Overhead

Profiling adds significant overhead. Expect slowdowns:

| Backend | Overhead |
|---------|----------|
| Serial | ~2.3x |
| OpenMP | ~2.5x |
| CUDA | ~2.9x |

### Sampling Support

For Kokkos tools, you can reduce overhead with sampling:

```bash
# Sample only 10% of kernels
export KOKKOS_TOOLS_SAMPLER_PROB=10

# Enable verbose sampling output
export KOKKOS_TOOLS_SAMPLER_VERBOSE=1

# Recommended sampling rates:
# - space-time-stack: 5-10% (50-70% overhead reduction)
# - chrome-tracing: 10-20% (40-60% reduction)
# - kernel-timer: 1-5% (20-40% reduction)
```

### Google Benchmark Limitation

Google Benchmark exits via `std::_Exit(0)`, which skips Kokkos finalization. This prevents:
- `memory-hwm` from displaying final report
- `memory-usage` from flushing accumulated data

**Workaround**: Use fixed memory tools (`kp_hwm_fixed.so`, `kp_memory_usage_fixed.so`) which use `atexit()` and periodic writes.

See `experimental/profiling_patches/` for the fixed versions.

### Kokkos Version Compatibility

The unified branch uses **Kokkos 5.0.1** (upgraded from 4.5.00) for B200/BLACKWELL support. Kokkos tools are fetched from the `develop` branch for compatibility.

---

## Build Presets Reference

### Profiling Presets

| Preset | Backend | Build Dir | Options |
|--------|---------|-----------|---------|
| `experimental-perf-serial` | Serial | `build-experimental-perf-serial` | `SUBSETIX_ENABLE_PROFILING_PERF=ON` |
| `experimental-perf-openmp` | OpenMP | `build-experimental-perf-openmp` | `SUBSETIX_ENABLE_PROFILING_PERF=ON` |
| `experimental-serial-profile` | Serial | `build-experimental-serial-profile` | `SUBSETIX_ENABLE_PROFILING_KOKKOS=ON` |
| `experimental-openmp-profile` | OpenMP | `build-experimental-openmp-profile` | `SUBSETIX_ENABLE_PROFILING_KOKKOS=ON` |
| `experimental-cuda-gcc12-profile` | CUDA | `build-experimental-cuda-gcc12-profile` | `SUBSETIX_ENABLE_PROFILING_KOKKOS=ON` |
| `profiling-nsight-cuda-gcc12` | CUDA | `build-profiling-nsight-cuda-gcc12` | `SUBSETIX_ENABLE_PROFILING_CUDA=ON` |
| `profiling-nsight-cuda-gcc12-release` | CUDA | `build-profiling-nsight-cuda-gcc12-release` | `SUBSETIX_ENABLE_PROFILING_CUDA=ON` |

All profiling presets:
- Enable experimental module only (`SUBSETIX_ENABLE_EXPERIMENTAL=ON`)
- Disable stable components to avoid linking errors
- Add debug symbols (`-g`) for profiling tools

---

## Script Reference

### Linux perf Scripts

| Script | Purpose |
|--------|---------|
| `scripts/perf_profile.sh` | Generic profiling for any executable |
| `scripts/profile_benchmark.sh` | Profile specific benchmark configurations |
| `scripts/profile_all_benchmarks.sh` | Profile all benchmark configurations |
| `scripts/compare_perf.sh` | Compare two perf data files |
| `scripts/generate_perf_report.sh` | Generate consolidated report |

### Nsight Scripts

| Script | Purpose |
|--------|---------|
| `scripts/profiling/run_ncu.sh` | Nsight Compute profiling |
| `scripts/profiling/run_nsys.sh` | Nsight Systems timeline profiling |
| `scripts/profiling/run_nsys_quick.sh` | Quick profiling with SmallConfig |
| `scripts/profiling/analyze_results.sh` | Result analysis helper |

### Kokkos Tools Scripts

| Script | Purpose |
|--------|---------|
| `scripts/profile_benchmark.sh` | Profile with Kokkos tools |
| `scripts/profiling/profile_all_backends.sh` | Profile all backends |
| `scripts/profiling/analyze_traces.sh` | Analyze Kokkos traces |
| `scripts/profiling/compare_runs.sh` | Compare profiling runs |

---

## Output Directories

| Tool | Output Directory |
|------|------------------|
| Linux perf | `./perf_output/` |
| Nsight Compute | `./profiling_output_ncu/` |
| Nsight Systems | `./profiling_output/` |
| Kokkos tools | `./profiling_output/<timestamp>-<tool>/` |

---

## Further Reading

- **Linux perf**: See `docs/PERF_PROFILING.md` for detailed perf guide
- **Modal GPU CI**: See `docs/MODAL.md` for cloud GPU profiling
- **Kokkos tools scripts**: See `scripts/profiling/README.md`
- **Main documentation**: See `CLAUDE.md` and `AGENTS.md`
