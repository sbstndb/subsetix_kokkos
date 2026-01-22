# Kokkos Profiling Guide

## Quick Start

### Automated Profiling (Recommended)

Use the profiling scripts for automatic trace generation and analysis:

```bash
# Profile all backends at once
./scripts/profiling/profile_all_backends.sh chrome-tracing "3D.*LargeConfig"

# Profile specific backend
./scripts/profile_benchmark.sh experimental-serial-profile kernel-timer "3D.*LargeConfig"

# Analyze generated traces
./scripts/profiling/analyze_traces.sh profiling_output/<timestamp>/*
```

See `scripts/profiling/README.md` for complete documentation.

### Manual Profiling

```bash
# Build with profiling support
cmake --preset experimental-serial-profile && cmake --build --preset experimental-serial-profile

# Run with profiling tool
KOKKOS_PROFILE_LIBRARY=build-experimental-serial-profile/_deps/kokkos_tools-build/profiling/simple-kernel-timer/libkp_kernel_timer.so \
  ./build-experimental-serial-profile/experimental/benchmarks/experimental_comparison_benchmark

# For OpenMP (set thread binding)
KOKKOS_PROFILE_LIBRARY=build-experimental-openmp-profile/_deps/kokkos_tools-build/profiling/simple-kernel-timer/libkp_kernel_timer.so \
  OMP_NUM_THREADS=22 OMP_PROC_BIND=spread \
  ./build-experimental-openmp-profile/experimental/benchmarks/experimental_comparison_benchmark
```

## Available Profiling Tools

| Tool | Library | Output | Usage |
|------|---------|--------|-------|
| **kernel-timer** | `libkp_kernel_timer.so` | `.dat` + JSON | Quantitative kernel timing analysis |
| **chrome-tracing** | `libkp_chrome_tracing.so` | `.json` | Timeline visualization (chrome://tracing) |
| **space-time-stack** | `libkp_space_time_stack.so` | stdout | Detailed time + memory report |
| **memory-hwm** | `libkp_hwm.so` | stdout | High water mark memory at program end |
| **memory-usage** | `libkp_memory_usage.so` | stdout | Memory usage with timestamps |

### Tool Selection Guide

- **kernel-timer**: Best for quantitative analysis, export to JSON for custom processing
- **chrome-tracing**: Best for visualizing kernel execution order and parallelism
- **space-time-stack**: Best for comprehensive analysis (time + memory hierarchy)
- **memory-hwm/memory-usage**: For memory profiling only

### Converting kernel-timer Output

```bash
# Convert .dat to JSON
./build-experimental-serial-profile/_deps/kokkos_tools-build/profiling/simple-kernel-timer/kp_json_writer <pid>.dat
```

## Important Notes

### Profiling Overhead

Kokkos profiling tools add **significant overhead** due to kernel synchronization and trace writing:
- **Serial**: ~2.3x slower
- **OpenMP**: ~2.5x slower
- **CUDA**: ~2.9x slower

Always compare against baseline (no profiling) for accurate performance metrics.

### Google Benchmark Limitation

Google Benchmark uses `std::_Exit(0)` which **skips Kokkos finalization**. This means:
- **memory-hwm** reports will not display
- **memory-usage** final reports will not display
- Other tools are unaffected (they write during execution)

### Environment Variables

- `KOKKOS_PROFILE_LIBRARY`: Deprecated in Kokkos 4.x but works reliably
- `KOKKOS_TOOLS_LIBS`: New syntax but may cause issues in some configurations

### Build Presets

| Preset | Backend | Build Dir |
|--------|---------|-----------|
| `experimental-serial-profile` | Serial | `build-experimental-serial-profile` |
| `experimental-openmp-profile` | OpenMP | `build-experimental-openmp-profile` |
| `experimental-cuda-gcc12-profile` | CUDA | `build-experimental-cuda-gcc12-profile` |

All profiling presets include `SUBSETIX_ENABLE_KOKKOS_TOOLS=ON` and automatically fetch kokkos-tools via FetchContent.
