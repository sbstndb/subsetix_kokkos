#!/bin/bash
# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

#
# analyze_traces.sh - Analyze Kokkos profiling traces
#
# Usage:
#   ./scripts/profiling/analyze_traces.sh <trace_dir>
#   ./scripts/profiling/analyze_traces.sh --help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    cat << EOF
${BLUE}Usage:${NC}
    $(basename "$0") <trace_directory>
    $(basename "$0") [${YELLOW}-h${NC}|${YELLOW}--help${NC}]

${BLUE}Description:${NC}
    Analyze Kokkos profiling traces and generate human-readable reports.

${BLUE}Arguments:${NC}
    trace_directory    Path to the profiling output directory to analyze

${BLUE}Options:${NC}
    ${YELLOW}-h, --help${NC}     Show this help message

${BLUE}Examples:${NC}
    $(basename "$0") profiling_output/20250122-145000
EOF
    exit 0
}

if [[ $# -lt 1 ]]; then
    usage
fi

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

TRACE_DIR="$1"

if [[ ! -d "$TRACE_DIR" ]]; then
    echo "${RED}Error: Directory not found: $TRACE_DIR${NC}"
    exit 1
fi

echo "${BLUE}=== Analyzing traces in: $TRACE_DIR ===${NC}"

# Check for kernel-timer traces
KERNEL_TIMER_DATS=($(find "$TRACE_DIR" -name "*.dat" 2>/dev/null))
if [[ ${#KERNEL_TIMER_DATS[@]} -gt 0 ]]; then
    echo ""
    echo "${GREEN}Found kernel-timer traces (.dat files):${NC} ${#KERNEL_TIMER_DATS[@]}"
    echo "${BLUE}Converting to JSON...${NC}"

    # Find kp_json_writer
    JSON_WRITER=""
    for build_dir in "$PROJECT_ROOT"/build-experimental-*-profile; do
        if [[ -f "$build_dir/_deps/kokkos_tools-build/profiling/simple-kernel-timer/kp_json_writer" ]]; then
            JSON_WRITER="$build_dir/_deps/kokkos_tools-build/profiling/simple-kernel-timer/kp_json_writer"
            break
        fi
    done

    if [[ -z "$JSON_WRITER" ]]; then
        echo "${YELLOW}Warning: kp_json_writer not found. Skipping JSON conversion.${NC}"
    else
        for dat_file in "${KERNEL_TIMER_DATS[@]}"; do
            json_file="${dat_file%.dat}.json"
            echo "  Converting $(basename "$dat_file") -> $(basename "$json_file")"
            "$JSON_WRITER" "$dat_file" > "$json_file" 2>/dev/null || true
        done
        echo "${GREEN}✓ JSON files created${NC}"
    fi
fi

# Check for chrome-tracing traces
CHROME_TRACES=($(find "$TRACE_DIR" -name "*.json" ! -name "summary.txt" 2>/dev/null))
if [[ ${#CHROME_TRACES[@]} -gt 0 ]]; then
    echo ""
    echo "${GREEN}Found chrome-tracing traces (.json files):${NC} ${#CHROME_TRACES[@]}"
    echo ""
    echo "${BLUE}Top kernels by duration (first 10):${NC}"
    for trace_file in "${CHROME_TRACES[@]}"; do
        echo "  ${BLUE}$(basename "$trace_file"):${NC}"
        # Extract kernel names and durations
        python3 << EOF 2>/dev/null || echo "    (Python not available for analysis)"
import json
import sys

try:
    with open("$trace_file", "r") as f:
        events = json.load(f)

    # Group by kernel name and sum durations
    kernels = {}
    for event in events:
        if "name" in event and "dur" in event:
            name = event["name"]
            dur = event.get("dur", 0)
            kernels[name] = kernels.get(name, 0) + dur

    # Sort by duration
    sorted_kernels = sorted(kernels.items(), key=lambda x: x[1], reverse=True)[:10]

    for name, dur in sorted_kernels:
        print(f"    {dur/1000:>8.1f} μs  {name}")
except:
    pass
EOF
        break
    done
fi

# Check for space-time-stack reports
if grep -q "BEGIN KOKKOS PROFILING REPORT" "$TRACE_DIR/benchmark_output.txt" 2>/dev/null; then
    echo ""
    echo "${GREEN}Found space-time-stack report${NC}"
    echo ""
    echo "${BLUE}TOP-DOWN TIME TREE:${NC}"
    grep -A 20 "TOP-DOWN TIME TREE" "$TRACE_DIR/benchmark_output.txt" | tail -20
    echo ""
    echo "${BLUE}MEMORY ALLOCATION:${NC}"
    grep -A 15 "MAX MEMORY ALLOCATED" "$TRACE_DIR/benchmark_output.txt" | tail -16
fi

echo ""
echo "${GREEN}=== Analysis complete! ===${NC}"
