<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique
-->
# Modal GPU CI Documentation

This document describes how to use Modal for GPU-based CI/CD of the experimental subsetix_kokkos module.

## Overview

Modal is a serverless platform for running Python functions on cloud GPUs. The `experimental/modal/` directory contains scripts to:

1. Build the experimental module with CUDA backend
2. Run all experimental tests
3. Execute performance benchmarks across different GPU architectures

## Quick Start

### Prerequisites

```bash
# Install Modal CLI
pip install modal

# Authenticate
modal token new
```

### Basic Usage

```bash
# Run on T4 (most affordable)
modal run experimental/modal/run_gpu_ci.py::run_t4_entry

# Run on A100 (balanced performance/cost)
modal run experimental/modal/run_gpu_ci.py::run_a100_entry

# Run on H100 (best performance)
modal run experimental/modal/run_gpu_ci.py::run_h100_entry
```

### Parallel Execution

To run benchmarks on multiple GPUs simultaneously, launch separate commands in parallel:

```bash
# Launch all 3 working GPUs in parallel
modal run experimental/modal/run_gpu_ci.py::run_t4_entry > /tmp/t4_results.txt 2>&1 &
modal run experimental/modal/run_gpu_ci.py::run_a100_entry > /tmp/a100_results.txt 2>&1 &
modal run experimental/modal/run_gpu_ci.py::run_h100_entry > /tmp/h100_results.txt 2>&1 &
wait
cat /tmp/*_results.txt
```

## Architecture

### GPU Support Matrix

| GPU | Kokkos Arch | Compute Capability | Kokkos 4.5.0<br/>+ CUDA 12.3 | Kokkos 5.0.1+<br/>+ CUDA 13.0 |
|-----|-------------|-------------------|---------------------------|-------------------------------|
| T4 | TURING75 | 7.5 | ✅ Working | ✅ Working |
| A100 | AMPERE80 | 8.0 | ✅ Working | ✅ Working |
| H100 | HOPPER90 | 9.0 | ✅ Working | ✅ Working |
| B200 | BLACKWELL | 10.0 | ❌ Kokkos can't detect CC 10.0 | ✅ **Full support** |

**B200 Requirements:**
- Kokkos 5.0.1+ (to detect Compute Capability 10.0)
- CUDA 13.0+ (nvcc must support `sm_100`)

### Container Image

The Modal image is based on:
- **Base**: `nvidia/cuda:13.0.0-devel-ubuntu22.04` (CUDA 13.0 required for sm_100/B200)
- **Python**: 3.11
- **Compiler**: GCC 12
- **Build tools**: CMake, Ninja

**Note:** For B200 (BLACKWELL), CUDA 13.0 is required because CUDA 12.x doesn't support `sm_100`.

### Build Configuration

```bash
cmake -S /workspace -B /tmp/build-experimental-cuda -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSUBSETIX_ENABLE_EXPERIMENTAL=ON \
    -DSUBSETIX_BUILD_STABLE_LIBS=OFF \
    -DSUBSETIX_BUILD_STABLE_TESTS=OFF \
    -DSUBSETIX_BUILD_STABLE_BENCHMARKS=OFF \
    -DSUBSETIX_KOKKOS_CUDA=ON \
    -DCMAKE_CXX_COMPILER=g++-12
```

**Important**: Uses `SUBSETIX_KOKKOS_CUDA=ON` (project option), not `Kokkos_ENABLE_CUDA=ON` (native Kokkos option).

## Entry Points

### Individual GPU Entry Points

For parallel execution, use these entry points:

```bash
modal run experimental/modal/run_gpu_ci.py::run_t4_entry
modal run experimental/modal/run_gpu_ci.py::run_a100_entry
modal run experimental/modal/run_gpu_ci.py::run_h100_entry
modal run experimental/modal/run_gpu_ci.py::run_b200_entry
```

### Main Entry Point

For sequential execution (all GPUs one after another):

```bash
modal run experimental/modal/run_gpu_ci.py::main
```

## Output Format

The script outputs:

1. **GPU Status** - nvidia-smi information
2. **CMake Configuration** - Kokkos backends and architectures
3. **Test Results** - All 7 experimental tests
4. **Benchmark Results** - Full Google Benchmark output

Example output:
```
🎯 GPU: T4 | CUDA ARCH: TURING75
🎮 GPU Status:
Tesla T4, 7.5, 580.95.05, 15360 MiB, 14913 MiB, 0 MiB

🔍 CMake Configuration Output:
-- Kokkos Backends: SERIAL;CUDA
-- Device Parallel: Kokkos::Cuda
-- Architectures: TURING75

TESTS:
1/7 Test #1: ExperimentalSortedRowsTest ........   Passed    0.86 sec
...
100% tests passed, 0 tests failed out of 7

ALL BENCHMARK RESULTS:
V1RandomMeshBenchmark2D<GetLargeConfig>/V1_LargeConfig ... 53ms
V2RandomMeshBenchmark2D<GetLargeConfig>/V2_LargeConfig ... 41ms
V3RandomMeshBenchmark2D<GetLargeConfig>/V3_LargeConfig ... 43ms
...
```

## Cost Estimation

| GPU | Hourly Rate | Build Time | Total Cost |
|-----|-------------|------------|------------|
| T4 | ~$0.70/hour | ~3 min | ~$0.04 |
| A100 | ~$1.50/hour | ~3 min | ~$0.08 |
| H100 | ~$1.50/hour | ~3 min | ~$0.08 |

Running all 3 in parallel: **~$0.20 total**

## Troubleshooting

### B200 fails with architecture error

**Error**: `CUDA enabled but no NVIDIA GPU architecture currently enabled`

**Cause (Kokkos 4.5.0)**: Kokkos 4.5.0 doesn't support Compute Capability 10.0 (BLACKWELL).

**Solution:** Use the `feat/modal-gpu-ci-kokkos-5` branch which includes Kokkos 5.0.1:
```bash
git checkout feat/modal-gpu-ci-kokkos-5
modal run experimental/modal/run_gpu_ci.py::run_b200_entry
```

### Build fails with CUDA errors

**Check**:
1. Verify `SUBSETIX_KOKKOS_CUDA=ON` is set (not `Kokkos_ENABLE_CUDA=ON`)
2. Check CMake output shows `Kokkos Backends: SERIAL;CUDA`
3. Ensure GPU architecture is detected (e.g., `TURING75`)

### Tests timeout

**Increase timeout** in `run_gpu_ci.py`:
```python
@app.function(gpu="T4", cpu=16.0, timeout=1800)  # 30 minutes
def run_t4() -> str:
    ...
```

## Development

### Adding New GPU Support

Edit `GPU_ARCH_MAP` in `run_gpu_ci.py`:

```python
GPU_ARCH_MAP = {
    "T4": "TURING75",
    "A100": "AMPERE80",
    "H100": "HOPPER90",
    "L40": "AD102",  # Add new GPU here
}
```

Then add the corresponding function and entry point:

```python
@app.function(gpu="L40", cpu=16.0, timeout=1200)
def run_l40() -> str:
    return run_benchmarks("L40", GPU_ARCH_MAP["L40"])

@app.local_entrypoint()
def run_l40_entry():
    print("🚀 Running L40 benchmarks...")
    print(run_l40.remote())
```

### Modifying Build Configuration

Edit the `cmake_cmd` list in `run_benchmarks()`:

```python
cmake_cmd = [
    "cmake", "-S", str(repo_root), "-B", str(build_dir), "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DSUBSETIX_ENABLE_EXPERIMENTAL=ON",
    # Add your options here
    "-DMY_CUSTOM_OPTION=ON",
]
```

## Related Documentation

- **CLAUDE.md** - Build and test commands
- **AGENTS.md** - Development guidelines
- **README.md** - Project overview
