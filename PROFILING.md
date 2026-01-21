# Kokkos Profiling Guide

## Quick Start

### Build with Profiling Support

```bash
# Configure with kokkos-tools enabled
cmake --preset experimental-serial-profile      # Serial
cmake --preset experimental-openmp-profile      # OpenMP
cmake --preset experimental-cuda-gcc12-profile  # CUDA

# Build
cmake --build --preset experimental-serial-profile
```

### Run Benchmarks with Profiling

```bash
# Using KOKKOS_PROFILE_LIBRARY (deprecated but works)
KOKKOS_PROFILE_LIBRARY=build-experimental-serial-profile/_deps/kokkos_tools-build/profiling/simple-kernel-timer/libkp_kernel_timer.so \
  ./build-experimental-serial-profile/experimental/benchmarks/experimental_comparison_benchmark

# For OpenMP (recommended: set thread binding)
KOKKOS_PROFILE_LIBRARY=build-experimental-openmp-profile/_deps/kokkos_tools-build/profiling/simple-kernel-timer/libkp_kernel_timer.so \
  OMP_NUM_THREADS=22 OMP_PROC_BIND=spread \
  ./build-experimental-openmp-profile/experimental/benchmarks/experimental_comparison_benchmark
```

### Convert Profiling Data to JSON

Profiling generates `.dat` files (binary format). Convert to JSON:

```bash
./build-*/_deps/kokkos_tools-build/profiling/simple-kernel-timer/kp_json_writer <pid>.dat
```

## Available Profiling Libraries

| Library | Path | Description |
|---------|------|-------------|
| `libkp_kernel_timer.so` | `profiling/simple-kernel-timer/` | Simple kernel timing (recommended) |
| `libkp_chrome_tracing.so` | `profiling/chrome-tracing/` | Chrome trace format |
| `libkokkostools.so` | `profiling/all/` | Monolithic (all profilers) - may segfault |

## Notes

- `KOKKOS_PROFILE_LIBRARY` is deprecated in Kokkos 4.x; use `KOKKOS_TOOLS_LIBS` instead (but may cause segfaults in some cases)
- Profiling data files are generated in the current working directory
- For CUDA, consider using Nsight Systems with nvtx connector
