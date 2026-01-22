# Profiling Guide

This document describes the profiling infrastructure and best practices for the Subsetix Kokkos project.

## Overview

Profiling is essential for understanding and optimizing performance. The project provides:
- **CMake presets** for profiling-enabled builds
- **Shell scripts** for running profiling on experimental benchmarks
- Support for **Nsight Compute (`ncu`)** and **Nsight Systems (`nsys`)**

## Profiler Selection Guide

| Profiler | Purpose | When to Use |
|----------|---------|-------------|
| **Nsight Compute (`ncu`)** | Detailed GPU kernel analysis | Deep dive into kernel performance, optimization work |
| **Nsight Systems (`nsys`)** | System-wide tracing | Application timeline, CPU-GPU interaction |

**Recommendation**: Use `ncu` for detailed GPU kernel profiling and algorithm optimization.

## CMake Presets for Profiling

### Available Presets

| Preset | Backend | Build Type | Description |
|--------|---------|------------|-------------|
| `profiling-cuda-gcc12` | CUDA | RelWithDebInfo | Profiling with debug symbols (default) |
| `profiling-cuda-gcc12-release` | CUDA | Release + symbols | Release build with profiling info |

### Usage

```bash
# Configure with profiling preset
cmake --preset profiling-cuda-gcc12

# Build
cmake --build --preset profiling-cuda-gcc12

# The benchmark binary will be at:
# build-profiling-cuda-gcc12/experimental/benchmarks/unified_comparison_benchmark
```

## Quick Start

### Nsight Compute (Recommended for GPU Kernel Analysis)

```bash
# Profile 3D LargeConfig with detailed metrics
./scripts/profiling/run_ncu.sh --benchmark "3D_LargeConfig"

# Quick profiling with basic metrics
./scripts/profiling/run_ncu.sh --benchmark "SmallConfig" --section-set basic

# View detailed results
/usr/local/cuda-12.8/bin/ncu --import profiling_output_ncu/*.ncu-rep --page=details
```

### Nsight Systems (Timeline Analysis)

```bash
# Quick profiling on 3D SmallConfig (fast)
./scripts/profiling/run_nsys_quick.sh

# Profile specific benchmark
./scripts/profiling/run_nsys.sh --benchmark "3D_LargeConfig"

# View results
nsys-ui profiling_output/*.nsys-rep
```

## Profiling Scripts

The profiling scripts are located in `scripts/profiling/`:

| Script | Purpose | Profiler |
|--------|---------|----------|
| `run_ncu.sh` | Detailed GPU kernel profiling | Nsight Compute |
| `run_nsys.sh` | System-wide timeline profiling | Nsight Systems |
| `run_nsys_quick.sh` | Quick profiling with SmallConfig | Nsight Systems |
| `analyze_results.sh` | Analyze profiling results | Both |

See `scripts/profiling/README.md` for detailed documentation.

## Benchmark Filters

The benchmark executables support filtering with `--benchmark_filter`:

| Pattern | Matches |
|---------|---------|
| `SmallConfig` | All SmallConfig benchmarks |
| `MediumConfig` | All MediumConfig benchmarks |
| `LargeConfig` | All LargeConfig benchmarks |
| `3D` | All 3D benchmarks |
| `2D` | All 2D benchmarks |
| `V1_3D_LargeConfig` | V1 3D LargeConfig only |
| `V2_SmallConfig` | V2 SmallConfig only |

## Manual Profiling

### Using Nsight Compute (Recommended for GPU Analysis)

```bash
# Find ncu location (may vary by system)
NCU_BIN=/usr/local/cuda-12.8/bin/ncu

# Basic profiling with default metrics
$NCU_BIN -o output.ncu-rep \
  build-profiling-cuda-gcc12/experimental/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="3D_LargeConfig"

# With specific section set (basic, full, or custom)
$NCU_BIN --set full -o output.ncu-rep \
  build-profiling-cuda-gcc12/experimental/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="LargeConfig"

# Profile specific kernel (by name)
$NCU_BIN -k "intersect_meshes" -o output.ncu-rep \
  build-profiling-cuda-gcc12/experimental/benchmarks/unified_comparison_benchmark

# Limit number of kernel launches
$NCU_BIN -c 10 -o output.ncu-rep \
  build-profiling-cuda-gcc12/experimental/benchmarks/unified_comparison_benchmark
```

### Using Nsight Systems

```bash
# Basic profiling
nsys profile -o output.nsys-rep \
  build-profiling-cuda-gcc12/experimental/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="3D_LargeConfig"

# With specific traces
nsys profile --trace=cuda,nvtx -o output.nsys-rep \
  build-profiling-cuda-gcc12/experimental/benchmarks/unified_comparison_benchmark \
  --benchmark_filter="LargeConfig"

# With CPU sampling (more overhead)
nsys profile --trace=cuda --sample=cpu -o output.nsys-rep \
  build-profiling-cuda-gcc12/experimental/benchmarks/unified_comparison_benchmark
```

## Analyzing Results

### Nsight Compute

```bash
# View detailed results (page: details, raw, source, session)
/usr/local/cuda-12.8/bin/ncu --import output.ncu-rep --page=details

# View specific sections
/usr/local/cuda-12.8/bin/ncu --import output.ncu-rep --section SpeedOfLight

# Export to CSV
/usr/local/cuda-12.8/bin/ncu --import output.ncu-rep --format csv --export output.csv

# Compare two profiles
/usr/local/cuda-12.8/bin/ncu --import profile1.ncu-rep --import profile2.ncu-rep
```

Key metrics to look for:
- **GPU Speed Of Light Throughput**: Compute and memory utilization
- **Occupancy**: How well the GPU SMs are utilized
- **Launch Statistics**: Block size, grid size, registers per thread
- **Workload Analysis**: Compute vs memory breakdown

### Nsight Systems

```bash
# GUI (recommended)
nsys-ui output.nsys-rep

# CLI statistics
nsys stats output.nsys-rep

# Export to CSV
nsys stats output.nsys-rep --format csv --output stats.csv

# Specific reports
nsys stats output.nsys-rep --report gpumemtimesum,gpumemsizesum
```

## Profiling Best Practices

### 1. Start Small

Use SmallConfig for rapid iterations during development:

```bash
./scripts/profiling/run_ncu.sh --benchmark "SmallConfig" --section-set basic
```

### 2. Target Specific Benchmarks

Use specific filters to reduce profiling time:

```bash
# Profile only v2 3D large mesh
./scripts/profiling/run_ncu.sh --benchmark "V2_3D_LargeConfig"
```

### 3. Understand Overhead

Profiling adds overhead. For accurate measurements:
- Use `--section-set basic` for faster profiling with ncu
- Use `--set full` only for detailed analysis (slow)
- Profile specific kernels with `-k` to reduce scope

### 4. Compare Versions

Profile multiple algorithm versions to compare performance:

```bash
# Profile v1, v2, v3 on same benchmark
for version in V1 V2 V3; do
  ./scripts/profiling/run_ncu.sh \
    --benchmark "${version}_3D_LargeConfig" \
    --output-dir "profiling_comparison/${version}"
done
```

### 5. Build Once, Profile Many Times

Once built, use `--kernel-only` to skip rebuild:

```bash
# First run: builds and profiles
./scripts/profiling/run_ncu.sh --benchmark "3D_MediumConfig"

# Subsequent runs: skip build
./scripts/profiling/run_ncu.sh --benchmark "3D_MediumConfig" --kernel-only
```

## Common Workflows

### Development Iteration

```bash
# 1. Make code changes
# 2. Quick test with SmallConfig
./scripts/profiling/run_ncu.sh --benchmark "SmallConfig" --section-set basic --kernel-only
# 3. View results
/usr/local/cuda-12.8/bin/ncu --import profiling_output_ncu/*.ncu-rep --page=details
```

### Performance Comparison

```bash
# Compare before/after optimization
git checkout before-optimization
./scripts/profiling/run_ncu.sh --benchmark "3D_LargeConfig" --output-dir "before"

git checkout after-optimization
./scripts/profiling/run_ncu.sh --benchmark "3D_LargeConfig" --output-dir "after"

# Compare the two .ncu-rep files
/usr/local/cuda-12.8/bin/ncu --import before/*.ncu-rep --import after/*.ncu-rep
```

### Full Benchmark Suite Profiling

```bash
# For comprehensive profiling, use nsys for timeline analysis
./scripts/profiling/run_nsys_all.sh --output-dir "profiling_full"

# Or profile individual configs with ncu for detailed GPU analysis
for config in SmallConfig MediumConfig LargeConfig; do
  ./scripts/profiling/run_ncu.sh --benchmark "3D_${config}" --kernel-only
done
```

## Troubleshooting

### "ncu not found"

The script searches in multiple locations. If not found, add to PATH or specify directly:
```bash
# Add to PATH
export PATH=/usr/local/cuda-12.8/bin:$PATH

# Or find and use directly
find /usr -name "ncu" 2>/dev/null
```

### "nsys not found"

Install Nsight Systems from: https://developer.nvidia.com/tools-overview

Or add to PATH:
```bash
export PATH=/opt/nvidia/nsight-systems/2024_6/bin:$PATH
```

### Build Errors

Make sure you're using a profiling preset:
```bash
cmake --preset profiling-cuda-gcc12
cmake --build --preset profiling-cuda-gcc12
```

### Large Trace Files

Profiling can generate large trace files (100MB+). To reduce size:
- Use specific benchmark filters
- Profile fewer repetitions
- Use `--trace=cuda` only (exclude OS runtime, CPU sampling)

## Integration with Development Workflow

Profiling should be part of the development process for performance-critical code:

1. **Before optimization**: Profile to identify bottlenecks
2. **During optimization**: Use quick profiling to verify improvements
3. **After optimization**: Full profiling to confirm gains

### For Experimental Module Development

When working on new algorithm versions in `experimental/`:

```bash
# 1. Build experimental with profiling
cmake --preset profiling-cuda-gcc12
cmake --build --preset profiling-cuda-gcc12

# 2. Profile your new version with detailed GPU metrics
./scripts/profiling/run_ncu.sh --benchmark "V4_3D_MediumConfig"

# 3. Compare against baseline
./scripts/profiling/run_ncu.sh --benchmark "V1_3D_MediumConfig" --output-dir "baseline"

# 4. Analyze the results
/usr/local/cuda-12.8/bin/ncu --import profiling_output_ncu/V4_3D_MediumConfig_*.ncu-rep --page=details
```

## Additional Resources

- **Nsight Systems Documentation**: https://docs.nvidia.com/nsight-systems/
- **Nsight Compute Documentation**: https://docs.nvidia.com/nsight-compute/
- **Kokkos Profiling**: https://kokkos.org/kokkos-core-wiki/Profiling.html
- **Project Scripts**: See `scripts/profiling/README.md` for complete script documentation
