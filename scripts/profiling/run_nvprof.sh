#!/bin/bash
# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause
#
# run_nvprof.sh - Run nvprof profiling on experimental benchmarks
#
# Usage:
#   ./run_nvprof.sh [OPTIONS]
#
# Options:
#   --preset PRESET        CMake preset to use (default: profiling-cuda-gcc12)
#   --benchmark FILTER     Benchmark filter (default: LargeConfig)
#   --output-dir DIR       Output directory for profiling results (default: profiling_output)
#   --nvprof-opts OPTS     Extra options to pass to nvprof (default: see NVPROF_DEFAULT_OPTS)
#   --build-only           Only build, don't run profiling
#   --run-only             Skip build, run profiling only
#   --help                 Show this help message
#
# Examples:
#   # Profile 3D LargeConfig benchmark
#   ./run_nvprof.sh --benchmark "LargeConfig" --output-dir profiling_output/3d_large
#
#   # Profile all 3D benchmarks
#   ./run_nvprof.sh --benchmark "3D" --output-dir profiling_output/3d_all
#
#   # Profile with detailed GPU metrics
#   ./run_nvprof.sh --benchmark "LargeConfig" --nvprof-opts "--metrics all"

set -euo pipefail

# Default values
PRESET="profiling-cuda-gcc12"
BENCHMARK_FILTER="LargeConfig"
OUTPUT_DIR="profiling_output"
BUILD_ONLY=0
RUN_ONLY=0

# Default nvprof options - good balance between detail and overhead
NVPROF_DEFAULT_OPTS="
  --print-gpu-trace
  --trace malloc
  --unified-memory-profiling off
  --export-profile %(output)s.prof
"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --preset)
      PRESET="$2"
      shift 2
      ;;
    --benchmark)
      BENCHMARK_FILTER="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --nvprof-opts)
      NVPROF_EXTRA_OPTS="$2"
      shift 2
      ;;
    --build-only)
      BUILD_ONLY=1
      shift
      ;;
    --run-only)
      RUN_ONLY=1
      shift
      ;;
    --help)
      grep '^#' "$0" | cut -c 4- | sed 's/^#$//' | sed 's/^# //'
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build-profiling-cuda-gcc12"
BENCHMARK_BIN="${BUILD_DIR}/experimental/benchmarks/unified_comparison_benchmark"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Sanitize benchmark filter for filename
SAFE_FILTER=$(echo "${BENCHMARK_FILTER}" | tr -cs '[:alnum:]_' '_')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PREFIX="${OUTPUT_DIR}/nvprof_${SAFE_FILTER}_${TIMESTAMP}"

echo "=================================="
echo "NVProf Profiling Script"
echo "=================================="
echo "Preset:              ${PRESET}"
echo "Benchmark filter:    ${BENCHMARK_FILTER}"
echo "Output directory:    ${OUTPUT_DIR}"
echo "Output prefix:       ${OUTPUT_PREFIX}"
echo ""

# Build step
if [ "${RUN_ONLY}" -eq 0 ]; then
  echo "Configuring..."
  cmake --preset "${PRESET}" || {
    echo "Error: CMake configuration failed"
    echo "Make sure CUDA is properly installed and the preset exists."
    exit 1
  }

  echo "Building..."
  cmake --build --preset "${PRESET}" || {
    echo "Error: Build failed"
    exit 1
  }
  echo "Build complete!"
  echo ""
fi

# Skip profiling if build-only
if [ "${BUILD_ONLY}" -eq 1 ]; then
  echo "Build-only mode - exiting"
  exit 0
fi

# Check if benchmark exists
if [ ! -f "${BENCHMARK_BIN}" ]; then
  echo "Error: Benchmark binary not found: ${BENCHMARK_BIN}"
  exit 1
fi

# Check if nvprof is available
if ! command -v nvprof &> /dev/null; then
  echo "Error: nvprof not found in PATH"
  echo ""
  echo "nvprof is part of the NVIDIA CUDA Toolkit."
  echo "Make sure you have CUDA installed and nvprof is in your PATH."
  exit 1
fi

# Prepare nvprof command
NVPROF_CMD="nvprof ${NVPROF_DEFAULT_OPTS} ${NVPROF_EXTRA_OPTS:-}"

echo "=================================="
echo "Running nvprof profiling..."
echo "=================================="
echo "Command: nvprof ${NVPROF_DEFAULT_OPTS} ${NVPROF_EXTRA_OPTS:-}"
echo ""

# Run profiling
# Note: We use --benchmark_filter to select specific benchmarks
eval nvprof --log-file "${OUTPUT_PREFIX}.log" \
  --print-gpu-trace \
  --trace malloc \
  ${NVPROF_EXTRA_OPTS:-} \
  "${BENCHMARK_BIN}" \
  --benchmark_filter="${BENCHMARK_FILTER}" \
  2>&1 | tee "${OUTPUT_PREFIX}_stdout.log"

echo ""
echo "=================================="
echo "Profiling complete!"
echo "=================================="
echo "Output files:"
echo "  - ${OUTPUT_PREFIX}.log (nvprof output)"
echo "  - ${OUTPUT_PREFIX}_stdout.log (benchmark stdout)"
echo ""

# Generate summary if possible
if command -v nvprof &> /dev/null; then
  echo "To view the profile in NVIDIA Visual Profiler:"
  echo "  nvprof -i ${OUTPUT_PREFIX}.prof"
  echo ""
  echo "Or use the standalone Visual Profiler:"
  echo "  nvvp ${OUTPUT_PREFIX}.prof"
  echo ""
fi

echo "Common nvprof options for future runs:"
echo "  --metrics all              : All available metrics"
echo "  --metrics evt:            : Event-based sampling"
echo "  --devices 0                : Profile specific GPU"
echo "  --print-gpu-summary        : Summary only (faster)"
echo "  --cpu-profiling            : Add CPU profiling"
