#!/bin/bash
# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause
#
# analyze_results.sh - Analyze and summarize profiling results
#
# This script helps extract and summarize key information from profiling output.
# Supports ncu and nsys output files.
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
echo "Profiling Results Analysis"
echo "=================================="
echo "Directory: ${PROFILING_DIR}"
echo ""

# Detect profiler type and count files
NCU_FILES=$(find "${PROFILING_DIR}" -name "*.ncu-rep" -type f 2>/dev/null | wc -l)
NSYS_FILES=$(find "${PROFILING_DIR}" -name "*.nsys-rep" -type f 2>/dev/null | wc -l)
LOG_FILES=$(find "${PROFILING_DIR}" -name "*.log" -type f 2>/dev/null)

echo "Found files:"
if [ "${NCU_FILES}" -gt 0 ]; then
  echo "  - ${NCU_FILES} ncu report(s) (*.ncu-rep)"
fi
if [ "${NSYS_FILES}" -gt 0 ]; then
  echo "  - ${NSYS_FILES} nsys report(s) (*.nsys-rep)"
fi
if [ -n "${LOG_FILES}" ]; then
  LOG_COUNT=$(echo "${LOG_FILES}" | wc -l)
  echo "  - ${LOG_COUNT} log file(s)"
fi
echo ""

# Function to extract timing info from log files
extract_log_timing() {
  local log_file="$1"
  if [ ! -f "${log_file}" ]; then
    return
  fi

  echo "  File: $(basename "${log_file}")"

  # Try to extract total time from the benchmark output
  if grep -q "real" "${log_file}" 2>/dev/null; then
    local time=$(grep "^real" "${log_file}" | awk '{print $2}')
    echo "    Real time: ${time}"
  fi

  # Try to extract benchmark results
  if grep -q "items_per_second" "${log_file}" 2>/dev/null; then
    local ips=$(grep -oP 'items_per_second=\K[\d.]+' "${log_file}" 2>/dev/null | head -1)
    if [ -n "${ips}" ]; then
      echo "    Items/sec: ${ips}"
    fi
  fi

  echo ""
}

# Process log files if they exist
if [ -n "${LOG_FILES}" ]; then
  echo "=================================="
  echo "Log Files Summary"
  echo "=================================="
  for log_file in ${LOG_FILES}; do
    extract_log_timing "${log_file}"
  done
fi

# Print viewing instructions based on available files
echo "=================================="
echo "Viewing Instructions"
echo "=================================="

if [ "${NCU_FILES}" -gt 0 ]; then
  NCU_BIN=$(find /usr -name "ncu" 2>/dev/null | head -1)
  if [ -z "${NCU_BIN}" ]; then
    NCU_BIN="ncu"
  fi
  echo ""
  echo "To view ncu results:"
  echo "  ${NCU_BIN} --import ${PROFILING_DIR}/*.ncu-rep --page=details"
fi

if [ "${NSYS_FILES}" -gt 0 ]; then
  echo ""
  echo "To view nsys results:"
  echo "  nsys-ui ${PROFILING_DIR}/*.nsys-rep"
  echo "  nsys stats ${PROFILING_DIR}/*.nsys-rep"
fi

echo ""
echo "=================================="
echo "Summary"
echo "=================================="
echo "Profiling directory: ${PROFILING_DIR}"
