#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique
#
# run_nsys.sh - Run Nsight Systems profiling on experimental benchmarks
#
# Usage:
#   ./run_nsys.sh [OPTIONS]
#
# Options:
#   --preset PRESET        CMake preset to use (default: profiling-nsight-cuda)
#   --benchmark FILTER     Benchmark filter (default: LargeConfig)
#   --output-dir DIR       Output directory for profiling results (default: profiling_output)
#   --nsys-opts OPTS       Extra options to pass to nsys (default: see NSYS_DEFAULT_OPTS)
#   --trace-only           Skip build, run profiling only
#   --build-only           Only build, don't run profiling
#   --help                 Show this help message
#
# Examples:
#   # Profile 3D LargeConfig benchmark
#   ./run_nsys.sh --benchmark "3D_LargeConfig" --output-dir profiling_output/3d_large
#
#   # Profile all 3D benchmarks
#   ./run_nsys.sh --benchmark "3D" --output-dir profiling_output/3d_all
#
#   # Profile with detailed GPU tracing
#   ./run_nsys.sh --benchmark "LargeConfig" --nsys-opts "--trace=cuda,nvtx"

set -euo pipefail

# Default values
PRESET="profiling-nsight-cuda"
BENCHMARK_FILTER="LargeConfig"
OUTPUT_DIR="profiling_output"
BUILD_ONLY=0
TRACE_ONLY=0
EXPORT_STATS=1  # Automatically export stats after profiling

# Default nsys options - good balance between detail and overhead
# Note: --force-overwrite=true to avoid prompts
#       --stats=true generates summary statistics
NSYS_DEFAULT_OPTS="profile --force-overwrite=true --stats=true --trace=cuda,nvtx,osrt"

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
    --nsys-opts)
      NSYS_EXTRA_OPTS="$2"
      shift 2
      ;;
    --trace-only)
      TRACE_ONLY=1
      shift
      ;;
    --build-only)
      BUILD_ONLY=1
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
BUILD_DIR="${PROJECT_ROOT}/build-profiling-nsight-cuda"
BENCHMARK_BIN="${BUILD_DIR}/experimental/benchmarks/experimental_comparison_benchmark"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Sanitize benchmark filter for filename
SAFE_FILTER=$(echo "${BENCHMARK_FILTER}" | tr -cs '[:alnum:]_' '_')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PREFIX="${OUTPUT_DIR}/nsys_${SAFE_FILTER}_${TIMESTAMP}"

echo "=================================="
echo "Nsight Systems Profiling Script"
echo "=================================="
echo "Preset:              ${PRESET}"
echo "Benchmark filter:    ${BENCHMARK_FILTER}"
echo "Output directory:    ${OUTPUT_DIR}"
echo "Output prefix:       ${OUTPUT_PREFIX}"
echo ""

# Build step
if [ "${TRACE_ONLY}" -eq 0 ]; then
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

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
  echo "Error: nsys not found in PATH"
  echo ""
  echo "Nsight Systems is part of the NVIDIA CUDA Toolkit."
  echo "Install from: https://developer.nvidia.com/tools-overview"
  exit 1
fi

# Prepare nsys command
NSYS_CMD="${NSYS_DEFAULT_OPTS} ${NSYS_EXTRA_OPTS:-} -o ${OUTPUT_PREFIX}"

echo "=================================="
echo "Running Nsight Systems profiling..."
echo "=================================="
echo "Command: nsys ${NSYS_CMD}"
echo ""

# Run profiling
# Note: We use --benchmark_filter to select specific benchmarks
eval nsys ${NSYS_DEFAULT_OPTS} ${NSYS_EXTRA_OPTS:-} \
  -o "${OUTPUT_PREFIX}" \
  "${BENCHMARK_BIN}" \
  --benchmark_filter="${BENCHMARK_FILTER}" \
  2>&1 | tee "${OUTPUT_PREFIX}_stdout.log"

echo ""
echo "=================================="
echo "Profiling complete!"
echo "=================================="
echo "Output files:"
echo "  - ${OUTPUT_PREFIX}.nsys-rep (Nsight Systems report)"
echo "  - ${OUTPUT_PREFIX}_stdout.log (benchmark stdout)"
echo ""

# Export stats reports
if [ "${EXPORT_STATS}" -eq 1 ]; then
  echo "=================================="
  echo "Exporting stats reports..."
  echo "=================================="

  # List of useful reports for GPU kernel analysis
  REPORTS=(
    "cuda_gpu_kern_sum"
    "cuda_kern_exec_sum"
    "cuda_gpu_mem_time_sum"
    "cuda_api_sum"
  )

  for report in "${REPORTS[@]}"; do
    echo "Exporting: ${report}"
    nsys stats --report "${report}" --format csv --output "${OUTPUT_PREFIX}_${report}.csv" "${OUTPUT_PREFIX}.nsys-rep" 2>/dev/null || true
    nsys stats --report "${report}" --format table --output "${OUTPUT_PREFIX}_${report}.txt" "${OUTPUT_PREFIX}.nsys-rep" 2>/dev/null || true
  done

  echo ""
  echo "Stats exported:"
  for report in "${REPORTS[@]}"; do
    echo "  - ${OUTPUT_PREFIX}_${report}.csv"
    echo "  - ${OUTPUT_PREFIX}_${report}.txt"
  done
  echo ""
fi

# Generate summary
echo "To view the profile in Nsight Systems GUI:"
echo "  nsys-ui ${OUTPUT_PREFIX}.nsys-rep"
echo ""
echo "To export additional stats (requires nsys):"
echo "  nsys stats ${OUTPUT_PREFIX}.nsys-rep --report <report_name>"
echo ""
echo "Available reports:"
echo "  cuda_gpu_kern_sum     : GPU kernel summary (by time)"
echo "  cuda_kern_exec_sum    : Kernel launch vs execution time (queue overhead)"
echo "  cuda_gpu_mem_time_sum : Memory operations by time"
echo "  cuda_api_sum          : CUDA API summary"
echo ""
echo "Common nsys options for future runs:"
echo "  --trace=cuda,nvtx,osrt  : Trace CUDA, NVTX, OS runtime (default)"
echo "  --trace=cuda            : CUDA tracing only (faster)"
echo "  --sample=cpu            : Add CPU sampling"
echo "  --delay=10 --duration=20 : Start tracing after 10s, trace for 20s"
