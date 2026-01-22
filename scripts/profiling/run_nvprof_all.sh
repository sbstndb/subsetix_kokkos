#!/bin/bash
# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause
#
# run_nvprof_all.sh - Run nvprof profiling on all experimental benchmarks
#
# This script runs nvprof on multiple benchmark configurations and organizes
# the results in a structured output directory.
#
# Usage:
#   ./run_nvprof_all.sh [OPTIONS]
#
# Options:
#   --preset PRESET        CMake preset to use (default: profiling-cuda-gcc12)
#   --output-dir DIR       Output directory for profiling results (default: profiling_output_all)
#   --skip-build           Skip the build step
#   --versions V1,V2,V3    Comma-separated list of versions to profile (default: all)
#   --configs S,M,L        Comma-separated list of configs (default: all)
#   --dimensions 2D,3D     Comma-separated list of dimensions (default: all)
#   --help                 Show this help message
#
# Examples:
#   # Profile everything
#   ./run_nvprof_all.sh
#
#   # Profile only 3D benchmarks, skip build
#   ./run_nvprof_all.sh --dimensions 3D --skip-build
#
#   # Profile only v2 and v3, medium and large configs
#   ./run_nvprof_all.sh --versions V2,V3 --configs M,L

set -euo pipefail

# Default values
PRESET="profiling-cuda-gcc12"
OUTPUT_DIR="profiling_output_all"
SKIP_BUILD=0
VERSIONS="V1,V2,V3"
CONFIGS="S,M,L"
DIMENSIONS="2D,3D"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --preset)
      PRESET="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --versions)
      VERSIONS="$2"
      shift 2
      ;;
    --configs)
      CONFIGS="$2"
      shift 2
      ;;
    --dimensions)
      DIMENSIONS="$2"
      shift 2
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
RUN_NVPROF="${SCRIPT_DIR}/run_nvprof.sh"

echo "=================================="
echo "NVProf Comprehensive Profiling"
echo "=================================="
echo "Preset:          ${PRESET}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "Versions:        ${VERSIONS}"
echo "Configs:         ${CONFIGS}"
echo "Dimensions:      ${DIMENSIONS}"
echo ""

# Build if needed
if [ "${SKIP_BUILD}" -eq 0 ]; then
  echo "Building..."
  "${RUN_NVPROF}" --preset "${PRESET}" --build-only || {
    echo "Error: Build failed"
    exit 1
  }
fi

# Parse selections
IFS=',' read -ra VERSION_ARRAY <<< "${VERSIONS}"
IFS=',' read -ra CONFIG_ARRAY <<< "${CONFIGS}"
IFS=',' read -ra DIMENSION_ARRAY <<< "${DIMENSIONS}"

# Mapping from config code to benchmark filter
declare -A CONFIG_MAP=(
  ["S"]="SmallConfig"
  ["M"]="MediumConfig"
  ["L"]="LargeConfig"
)

# Mapping from dimension to benchmark filter suffix
declare -A DIMENSION_MAP=(
  ["2D"]=""
  ["3D"]="3D"
)

# Total runs
TOTAL=0
for VERSION in "${VERSION_ARRAY[@]}"; do
  for CONFIG in "${CONFIG_ARRAY[@]}"; do
    for DIMENSION in "${DIMENSION_ARRAY[@]}"; do
      ((TOTAL++))
    done
  done
done

CURRENT=0

# Run profiling for each combination
for VERSION in "${VERSION_ARRAY[@]}"; do
  for CONFIG in "${CONFIG_ARRAY[@]}"; do
    for DIMENSION in "${DIMENSION_ARRAY[@]}"; do
      ((CURRENT++))

      CONFIG_NAME="${CONFIG_MAP[$CONFIG]}"
      DIMENSION_SUFFIX="${DIMENSION_MAP[$DIMENSION]}"

      # Build benchmark filter
      if [ -z "${DIMENSION_SUFFIX}" ]; then
        # 2D benchmarks
        FILTER="${VERSION}_${CONFIG_NAME}"
      else
        # 3D benchmarks
        FILTER="${VERSION}_${DIMENSION_SUFFIX}_${CONFIG_NAME}"
      fi

      # Build subdirectory
      SUBDIR="${OUTPUT_DIR}/${DIMENSION}/${CONFIG_NAME}/${VERSION}"
      OUTPUT_ARG="--output-dir ${SUBDIR}"

      echo ""
      echo "=================================="
      echo "Run ${CURRENT}/${TOTAL}"
      echo "=================================="
      echo "Version:   ${VERSION}"
      echo "Config:    ${CONFIG_NAME}"
      echo "Dimension: ${DIMENSION}"
      echo "Filter:    ${FILTER}"
      echo ""

      "${RUN_NVPROF}" \
        --preset "${PRESET}" \
        --benchmark "${FILTER}" \
        --output-dir "${SUBDIR}" \
        --run-only || {
        echo "Warning: Profiling failed for ${FILTER}"
        continue
      }
    done
  done
done

echo ""
echo "=================================="
echo "All profiling runs complete!"
echo "=================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Directory structure:"
echo "  ${OUTPUT_DIR}/"
echo "    ├── 2D/"
echo "    │   ├── SmallConfig/"
echo "    │   │   ├── V1/"
echo "    │   │   ├── V2/"
echo "    │   │   └── V3/"
echo "    │   ├── MediumConfig/"
echo "    │   └── LargeConfig/"
echo "    └── 3D/"
echo "        └── ..."
