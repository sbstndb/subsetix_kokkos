#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique
#
# run_ncu.sh - Run Nsight Compute profiling on experimental benchmarks
#
# Nsight Compute (ncu) provides detailed GPU kernel profiling and analysis.
# This is the recommended tool for deep GPU performance analysis.
#
# Usage:
#   ./run_ncu.sh [OPTIONS]
#
# Options:
#   --preset PRESET        CMake preset to use (default: profiling-nsight-cuda)
#   --benchmark FILTER     Benchmark filter (default: 3D_LargeConfig)
#   --output-dir DIR       Output directory for profiling results (default: profiling_output_ncu)
#   --ncu-opts OPTS        Extra options to pass to ncu
#   --section-set SET      Section set to use (default: detailed)
#   --extra-sections SEC   Additional sections (default: WarpStateStats,InstructionStats)
#   --kernel-only          Skip build, run profiling only
#   --build-only           Only build, don't run profiling
#   --help                 Show this help message
#
# Examples:
#   # Profile 3D LargeConfig with default detailed metrics
#   ./run_ncu.sh --benchmark "3D_LargeConfig"
#
#   # Profile with basic metrics (faster)
#   ./run_ncu.sh --benchmark "LargeConfig" --section-set basic --extra-sections ""
#
#   # Profile with full metrics (very slow, ~6000 metrics)
#   ./run_ncu.sh --section-set full --extra-sections ""
#
#   # Profile specific version
#   ./run_ncu.sh --benchmark "V2_3D_MediumConfig"

set -euo pipefail

# Default values
PRESET="profiling-nsight-cuda"
BENCHMARK_FILTER="3D_LargeConfig"
OUTPUT_DIR="profiling_output_ncu"
BUILD_ONLY=0
KERNEL_ONLY=0
SECTION_SET="detailed"  # Includes ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, SourceCounters, Roofline
EXTRA_SECTIONS="WarpStateStats,InstructionStats"  # Additional detailed sections

# Find ncu - check multiple common paths
find_ncu() {
  local ncu_paths=(
    "ncu"  # Try PATH first
    "/opt/nvidia/nsight-compute/ncu"
    "/usr/local/cuda/bin/ncu"
  )

  for path in "${ncu_paths[@]}"; do
    if command -v "$path" &> /dev/null; then
      echo "$path"
      return 0
    fi
  done

  return 1
}

NCU_BIN=$(find_ncu)

if [[ -z "$NCU_BIN" ]]; then
  echo "Error: ncu not found"
  echo ""
  echo "Install Nsight Compute from:"
  echo "  https://developer.nvidia.com/nsight-compute"
  echo ""
  echo "Or ensure ncu is in your PATH"
  exit 1
fi

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
    --ncu-opts)
      NCU_EXTRA_OPTS="$2"
      shift 2
      ;;
    --section-set)
      SECTION_SET="$2"
      shift 2
      ;;
    --kernel-only)
      KERNEL_ONLY=1
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
OUTPUT_PREFIX="${OUTPUT_DIR}/ncu_${SAFE_FILTER}_${TIMESTAMP}"

echo "=================================="
echo "Nsight Compute Profiling Script"
echo "=================================="
echo "Preset:              ${PRESET}"
echo "Benchmark filter:    ${BENCHMARK_FILTER}"
echo "Output directory:    ${OUTPUT_DIR}"
echo "Output prefix:       ${OUTPUT_PREFIX}"
echo "ncu binary:          ${NCU_BIN}"
echo ""

# Build step
if [ "${KERNEL_ONLY}" -eq 0 ]; then
  echo "Configuring..."
  cmake --preset "${PRESET}" || {
    echo "Error: CMake configuration failed"
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

# Prepare ncu command
NCU_CMD="${NCU_BIN}"

if [ -n "${SECTION_SET}" ]; then
  NCU_CMD="${NCU_CMD} --set ${SECTION_SET}"
fi

echo "=================================="
echo "Running Nsight Compute profiling..."
echo "=================================="
echo "Section set:         ${SECTION_SET}"
echo "Extra sections:      ${EXTRA_SECTIONS}"
echo "Total metrics:       ~1200 (detailed) + ~3500 (full sections)"
echo "Expected overhead:   3-5x basic profiling"
echo ""

# Build sections list
SECTION_ARGS=""
if [ -n "${SECTION_SET}" ]; then
  SECTION_ARGS="--set ${SECTION_SET}"
fi
if [ -n "${EXTRA_SECTIONS}" ]; then
  IFS=',' read -ra SECTIONS <<< "${EXTRA_SECTIONS}"
  for section in "${SECTIONS[@]}"; do
    SECTION_ARGS="${SECTION_ARGS} --section ${section}"
  done
fi

# Run profiling
# Note: ncu does verbose profiling with multiple passes
eval ${NCU_BIN} \
  ${SECTION_ARGS} \
  ${NCU_EXTRA_OPTS:-} \
  -o "${OUTPUT_PREFIX}" \
  "${BENCHMARK_BIN}" \
  --benchmark_filter="${BENCHMARK_FILTER}" \
  2>&1 | tee "${OUTPUT_PREFIX}_stdout.log"

echo ""
echo "=================================="
echo "Profiling complete!"
echo "=================================="
echo "Output files:"
echo "  - ${OUTPUT_PREFIX}.ncu-rep (Nsight Compute report)"
echo "  - ${OUTPUT_PREFIX}_stdout.log (benchmark stdout)"
echo ""

# Show quick summary
echo "To view detailed results in CLI:"
echo "  ${NCU_BIN} --import ${OUTPUT_PREFIX}.ncu-rep --page=details"
echo ""
echo "To view specific sections:"
echo "  ${NCU_BIN} --import ${OUTPUT_PREFIX}.ncu-rep --page=raw"
echo "  ${NCU_BIN} --import ${OUTPUT_PREFIX}.ncu-rep --page=source"
echo ""
echo "Section sets available:"
echo "  --set basic      : Basic metrics (~190 metrics, fast)"
echo "  --set detailed   : Compute+Memory workload + Source + Roofline (~1200 metrics, default)"
echo "  --set full       : All available metrics (~5900 metrics, very slow)"
echo ""
echo "Additional sections:"
echo "  --section WarpStateStats       : Warp stalled reasons, scheduler efficiency"
echo "  --section InstructionStats     : Instruction mix, branch prediction"
echo "  --section SchedulerStats       : Warp scheduler details"
echo "  --section SourceCounters       : Source line-level profiling"
echo ""
echo "To filter kernels:"
echo "  -k <kernel_name>     : Profile specific kernel"
echo "  -c <count>           : Limit number of kernel launches"
echo "  -s <skip>            : Skip first N launches"
