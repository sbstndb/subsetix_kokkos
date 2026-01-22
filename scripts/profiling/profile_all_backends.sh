#!/bin/bash
# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

#
# profile_all_backends.sh - Profile benchmark across all backends
#
# Usage:
#   ./scripts/profiling/profile_all_backends.sh <tool> <benchmark_filter>
#
# Example:
#   ./scripts/profiling/profile_all_backends.sh kernel-timer "3D.*LargeConfig"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

usage() {
    cat << EOF
${BLUE}Usage:${NC}
    $(basename "$0") <tool> <benchmark_filter>

${BLUE}Arguments:${NC}
    <tool>              Profiling tool: kernel-timer, chrome-tracing, space-time-stack
    <benchmark_filter>    Google Benchmark filter

${BLUE}Example:${NC}
    $(basename "$0") kernel-timer "3D.*LargeConfig"

${BLUE}Description:${NC}
    Run profiling for all backends (Serial, OpenMP, CUDA) and compare results.
    Uses experimental-*-profile presets.

${BLUE}Output:${NC}
    Traces saved to: profiling_output/<timestamp>-<tool>/
EOF
    exit 1
}

if [[ $# -lt 2 ]]; then
    usage
fi

TOOL="$1"
BENCHMARK_FILTER="$2"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_BASE="${PROJECT_ROOT}/profiling_output/${TIMESTAMP}-${TOOL}"

# Presets to profile
PRESETS=("experimental-serial-profile" "experimental-openmp-profile" "experimental-cuda-gcc12-profile")
BACKEND_NAMES=("Serial" "OpenMP" "CUDA")

# Validate tool
declare -A TOOL_LIBS=(
    ["kernel-timer"]="profiling/simple-kernel-timer/libkp_kernel_timer.so"
    ["chrome-tracing"]="profiling/chrome-tracing/libkp_chrome_tracing.so"
    ["space-time-stack"]="profiling/space-time-stack/libkp_space_time_stack.so"
)

if [[ -z "${TOOL_LIBS[$TOOL]}" ]]; then
    echo "${RED}Error: Unknown tool '$TOOL'. Available: ${!TOOL_LIBS[@]}${NC}"
    exit 1
fi

echo "${CYAN}=== Kokkos Profiling - All Backends ===${NC}"
echo ""
echo "${BLUE}Tool:${NC}          $TOOL"
echo "${BLUE}Benchmark:${NC}     $BENCHMARK_FILTER"
echo "${BLUE}Output base:${NC}  $OUTPUT_BASE"
echo "${BLUE}Timestamp:${NC}    $TIMESTAMP"
echo ""

# Run profiling for each backend
for i in "${!PRESETS[@]}"; do
    PRESET="${PRESETS[$i]}"
    BACKEND="${BACKEND_NAMES[$i]}"
    OUTPUT_DIR="${OUTPUT_BASE}/${BACKEND}"

    echo "${CYAN}--- Profiling $BACKEND ---${NC}"

    "${PROJECT_ROOT}/scripts/profile_benchmark.sh" \
        "$PRESET" \
        "$TOOL" \
        "$BENCHMARK_FILTER" \
        -o "$OUTPUT_DIR"

    echo ""
done

# Compare results
echo ""
echo "${CYAN}=== Comparing Results ===${NC}"
"$SCRIPT_DIR/analyze_traces.sh" "$OUTPUT_BASE"/*

echo ""
echo "${GREEN}=== All done! ===${NC}"
echo "${BLUE}Output directories:${NC}"
for i in "${!PRESETS[@]}"; do
    BACKEND="${BACKEND_NAMES[$i]}"
    echo "  $OUTPUT_BASE/${BACKEND}"
done
