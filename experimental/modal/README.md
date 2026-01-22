# Modal GPU CI for Experimental Module

This directory contains Modal scripts to run the experimental subsetix_kokkos tests and benchmarks on NVIDIA GPUs.

## Prerequisites

- Modal CLI installed and authenticated
- Modal account with GPU access (billing enabled)

## Quick Start

```bash
# Run everything (build + tests + benchmarks)
modal run experimental/modal/run_gpu_ci.py::simple

# Or with flags (not yet implemented)
modal run experimental/modal/run_gpu_ci.py --tests-only
modal run experimental/modal/run_gpu_ci.py --bench-only
```

## What It Does

The Modal script will:

1. **Mount your local code** into the Modal container
2. **Build** the experimental module with CUDA backend (Kokkos + CUDA)
3. **Run tests** - All experimental unit tests
4. **Run benchmarks** - Performance comparison benchmarks

Results are printed directly to your terminal.

## GPU Options

The script requests `any` GPU, which will give you:
- Usually an NVIDIA T4 (Turing) on Modal
- ~$1-2/hour depending on region

## Troubleshooting

### `modal: command not found`
Install Modal: `pip install modal`

### `Not authenticated`
Run: `modal token new`

### Build fails
Check that GCC 12 and CUDA toolkit are available in the container. The script installs:
- `gcc-12`, `g++-12`
- `cuda-toolkit-12`
- `cmake`, `ninja-build`

### CUDA version mismatch
Modal's NVIDIA driver version may differ. The script uses CUDA 12 toolkit which should be compatible with most drivers.

## Cost Estimate

- Build: ~2-5 minutes
- Tests: ~1-2 minutes
- Benchmarks: ~2-5 minutes

Total: ~5-12 minutes GPU time = ~$0.10-$0.40 per run
