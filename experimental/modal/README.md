# Modal GPU CI for Experimental Module

This directory contains Modal scripts to run the experimental subsetix_kokkos tests and benchmarks on NVIDIA GPUs.

## Prerequisites

- Modal CLI installed and authenticated
- Modal account with GPU access (billing enabled)

## Quick Start

```bash
# Run on specific GPU (individual entry points for parallel execution)
modal run experimental/modal/run_gpu_ci.py::run_t4_entry
modal run experimental/modal/run_gpu_ci.py::run_a100_entry
modal run experimental/modal/run_gpu_ci.py::run_h100_entry
modal run experimental/modal/run_gpu_ci.py::run_b200_entry

# Run all GPUs sequentially (not recommended - use individual entry points in parallel)
modal run experimental/modal/run_gpu_ci.py::main
```

## What It Does

The Modal script will:

1. **Mount your local code** into the Modal container
2. **Build** the experimental module with CUDA backend (Kokkos + CUDA)
3. **Run tests** - All experimental unit tests
4. **Run benchmarks** - Performance comparison benchmarks (v1, v2, v3)

Results are printed directly to your terminal.

## GPU Options

| GPU | Architecture | Status | Cost (approx.) |
|-----|--------------|--------|----------------|
| T4 | TURING75 | ✅ Working | ~$0.07/run |
| A100 | AMPERE80 | ✅ Working | ~$0.15/run |
| H100 | HOPPER90 | ✅ Working | ~$0.15/run |
| B200 | BLACKWELL | ✅ Supported (Kokkos 5.0.1+) | ~$0.30/run |

**Note:** This branch uses Kokkos 5.0.1 which supports Compute Capability 10.0 (BLACKWELL/B200).

## Execution Modes

### Parallel Execution (Recommended)
Run multiple GPUs in parallel from separate terminals:
```bash
# Terminal 1
modal run experimental/modal/run_gpu_ci.py::run_t4_entry

# Terminal 2 (simultaneously)
modal run experimental/modal/run_gpu_ci.py::run_a100_entry

# etc.
```

### Sequential Execution
Run all GPUs one after another:
```bash
modal run experimental/modal/run_gpu_ci.py::main
```

## Troubleshooting

### `modal: command not found`
Install Modal: `pip install modal`

### `Not authenticated`
Run: `modal token new`

### Build fails
Check that GCC 12 and CUDA toolkit are available in the container. The script installs:
- `gcc-12`, `g++-12`
- `cmake`, `ninja-build`
- `libfmt-dev`, `libmpfr-dev`

### CUDA version mismatch
Modal's NVIDIA driver version may differ. The script uses CUDA 12.3.2 from NVIDIA containers which should be compatible with most drivers.

### B200 fails with "CUDA enabled but no NVIDIA GPU architecture"

**On Kokkos 4.5.0:** This is expected - Kokkos 4.5.0 doesn't support Compute Capability 10.0 (BLACKWELL).

**Solution:** Switch to the `feat/modal-gpu-ci-kokkos-5` branch which uses Kokkos 5.0.1:
```bash
git checkout feat/modal-gpu-ci-kokkos-5
```

## Cost Estimate

Per GPU execution:
- Build: ~2-3 minutes
- Tests: ~5-10 seconds
- Benchmarks: ~1-2 minutes

Total per GPU: ~$0.07-$0.15 (T4: $0.07, A100/H100: $0.15)

Running all 3 working GPUs in parallel: ~$0.37 total
