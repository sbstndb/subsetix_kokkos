#!/bin/bash
# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause
#
# analyze_results.sh - Analyze and summarize nvprof profiling results
#
# This script helps extract and summarize key information from nvprof output.
#
# Usage:
#   ./analyze_results.sh [PROFILING_DIR]
#
# If no directory is specified, it will use the most recent profiling output.

set -euo pipefail

# Find the profiling directory
if [ $# -eq 0 ]; then
  # Find most recent profiling output
  PROFILING_DIR=$(find . -maxdepth 2 -type d -name "profiling_output*" 2>/dev/null | sort -r | head -1)
  if [ -z "${PROFILING_DIR}" ]; then
    echo "Error: No profiling output directories found"
    echo ""
    echo "Usage: $0 [PROFILING_DIR]"
    exit 1
  fi
else
  PROFILING_DIR="$1"
fi

if [ ! -d "${PROFILING_DIR}" ]; then
  echo "Error: Directory not found: ${PROFILING_DIR}"
  exit 1
fi

echo "=================================="
echo "NVProf Results Analysis"
echo "=================================="
echo "Directory: ${PROFILING_DIR}"
echo ""

# Function to extract timing info from nvprof log
extract_timing() {
  local log_file="$1"
  if [ ! -f "${log_file}" ]; then
    return
  fi

  # Extract GPU time from log (look for "GPU activities" section)
  echo "  File: $(basename "${log_file}")"

  # Try to extract total time from the benchmark output
  if grep -q "real" "${log_file}" 2>/dev/null; then
    local time=$(grep "^real" "${log_file}" | awk '{print $2}')
    echo "    Real time: ${time}"
  fi

  # Count kernel launches
  if grep -q "CUDA" "${log_file}" 2>/dev/null; then
    local kernels=$(grep -c "CUDA" "${log_file}" 2>/dev/null || echo "0")
    echo "    CUDA operations: ${kernels}"
  fi

  echo ""
}

# Find all log files
LOG_FILES=$(find "${PROFILING_DIR}" -name "*.log" -type f 2>/dev/null)

if [ -z "${LOG_FILES}" ]; then
  echo "No log files found in ${PROFILING_DIR}"
  exit 0
fi

echo "Found $(echo "${LOG_FILES}" | wc -l) log file(s)"
echo ""

# Process each log file
for log_file in ${LOG_FILES}; do
  extract_timing "${log_file}"
done

echo "=================================="
echo "Summary"
echo "=================================="
echo "Profiling directory: ${PROFILING_DIR}"
echo ""
echo "To view detailed profiles with NVIDIA Visual Profiler:"
echo "  nvvp ${PROFILING_DIR}/*/*.prof"
echo ""
echo "To export profile data to text:"
echo "  nvprof --export-profile output.prof <your_command>"
echo "  nvprof -i output.prof --print-gpu-trace > trace.txt"
