#!/bin/bash
# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause
#
# run_nsys_quick.sh - Quick profiling with SmallConfig benchmarks using Nsight Systems
#
# This script is designed for rapid profiling iterations during development.
# It uses SmallConfig benchmarks which run quickly, and provides a focused
# set of profiling metrics.
#
# Usage:
#   ./run_nsys_quick.sh [OPTIONS]
#
# Options:
#   --dimension 2D|3D|both  Which dimension to profile (default: 3D)
#   --version V1|V2|V3|all  Which version to profile (default: all)
#   --output-dir DIR        Output directory (default: profiling_output_quick)
#   --skip-build            Skip the build step
#   --detailed              Use more detailed profiling options
#   --help                  Show this help message
#
# Examples:
#   # Quick 3D profiling, all versions
#   ./run_nsys_quick.sh
#
#   # Quick 2D profiling, v2 only
#   ./run_nsys_quick.sh --dimension 2D --version V2

set -euo pipefail

# Default values
DIMENSION="3D"
VERSION="all"
OUTPUT_DIR="profiling_output_quick"
SKIP_BUILD=0
DETAILED=0

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dimension)
      DIMENSION="$2"
      shift 2
      ;;
    --version)
      VERSION="$2"
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
    --detailed)
      DETAILED=1
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
RUN_NSYS="${SCRIPT_DIR}/run_nsys.sh"

echo "=================================="
echo "Quick Nsight Systems Profiling"
echo "=================================="
echo "Dimension:   ${DIMENSION}"
echo "Version:     ${VERSION}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Detailed:    ${DETAILED}"
echo ""

# Build if needed
if [ "${SKIP_BUILD}" -eq 0 ]; then
  echo "Building with profiling preset..."
  "${RUN_NSYS}" --preset profiling-cuda-gcc12 --build-only || {
    echo "Error: Build failed"
    exit 1
  }
fi

# Set up profiling options based on detail level
if [ "${DETAILED}" -eq 1 ]; then
  NSYS_OPTS="--sample=cpu"  # Add CPU sampling to default trace options
else
  # Quick profiling - use default trace options from run_nsys.sh
  NSYS_OPTS=""
fi

# Run profiling
if [ "${VERSION}" = "all" ]; then
  if [ "${DIMENSION}" = "both" ]; then
    echo "Running quick profiling: 2D and 3D, all versions..."
    "${RUN_NSYS}" \
      --preset profiling-cuda-gcc12 \
      --benchmark "SmallConfig" \
      --output-dir "${OUTPUT_DIR}/all" \
      --nsys-opts "${NSYS_OPTS}" \
      --trace-only
  elif [ "${DIMENSION}" = "3D" ]; then
    echo "Running quick profiling: 3D, all versions..."
    "${RUN_NSYS}" \
      --preset profiling-cuda-gcc12 \
      --benchmark "3D_SmallConfig" \
      --output-dir "${OUTPUT_DIR}/3d_all" \
      --nsys-opts "${NSYS_OPTS}" \
      --trace-only
  else
    echo "Running quick profiling: 2D, all versions..."
    "${RUN_NSYS}" \
      --preset profiling-cuda-gcc12 \
      --benchmark "SmallConfig" \
      --output-dir "${OUTPUT_DIR}/2d_all" \
      --nsys-opts "${NSYS_OPTS}" \
      --trace-only
  fi
else
  # Specific version
  if [ "${DIMENSION}" = "3D" ]; then
    FILTER="${VERSION}_3D_SmallConfig"
    SUBDIR="${OUTPUT_DIR}/3d_${VERSION}"
  elif [ "${DIMENSION}" = "2D" ]; then
    FILTER="${VERSION}_SmallConfig"
    SUBDIR="${OUTPUT_DIR}/2d_${VERSION}"
  else
    echo "Error: When specifying a version, dimension must be 2D or 3D"
    exit 1
  fi

  echo "Running quick profiling: ${FILTER}..."
  "${RUN_NSYS}" \
    --preset profiling-cuda-gcc12 \
    --benchmark "${FILTER}" \
    --output-dir "${SUBDIR}" \
    --nsys-opts "${NSYS_OPTS}" \
    --trace-only
fi

echo ""
echo "=================================="
echo "Quick profiling complete!"
echo "=================================="
echo "Results in: ${OUTPUT_DIR}"
echo ""
echo "For detailed analysis, use:"
echo "  nsys-ui ${OUTPUT_DIR}/*/*.nsys-rep"
echo ""
echo "To view statistics:"
echo "  nsys stats ${OUTPUT_DIR}/*/*.nsys-rep"
