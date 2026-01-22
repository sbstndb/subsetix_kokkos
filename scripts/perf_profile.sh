#!/bin/bash
# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

# Performance profiling script using perf
# Requires: perf tool, appropriate permissions

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if perf is available
if ! command -v perf &> /dev/null; then
    echo -e "${RED}Error: perf is not installed${NC}"
    echo "Install with: sudo apt-get install linux-tools-common linux-tools-generic"
    exit 1
fi

# Check permissions
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "-1")
if [ "$PARANOID" -ge 2 ]; then
    echo -e "${YELLOW}Warning: perf_event_paranoid is $PARANOID (restrictive)${NC}"
    echo "Consider running: sudo sysctl -w kernel.perf_event_paranoid=1"
    echo ""
fi

# Function to show usage
usage() {
    cat << EOF
Usage: $0 <preset> <executable> [args]

Presets:
  experimental-perf-serial    - Serial backend profiling
  experimental-perf-openmp    - OpenMP backend profiling

Examples:
  $0 experimental-perf-serial ./build-experimental-perf-serial/experimental/tests/experimental_v1_unitary_test
  $0 experimental-perf-openmp ./build-experimental-perf-openmp/experimental/benchmarks/experimental_comparison_benchmark --benchmark_filter=Small

Environment variables:
  PERF_OUTPUT_DIR  - Output directory for perf data (default: ./perf_output)
  PERF_EVENTS      - Comma-separated events to record (default: cycles,instructions,cache-misses)
  PERF_CALL_GRAPH  - Call graph mode: dwarf, fp, lbr (default: dwarf)

EOF
    exit 1
}

# Parse arguments
if [ $# -lt 2 ]; then
    usage
fi

PRESET=$1
shift
EXECUTABLE=$@
OUTPUT_DIR=${PERF_OUTPUT_DIR:-./perf_output}
CALL_GRAPH=${PERF_CALL_GRAPH:-dwarf}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate output filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXEC_NAME=$(basename "$EXECUTABLE" | sed 's/[^a-zA-Z0-9._-]/_/g')
PERF_DATA="$OUTPUT_DIR/perf_${EXEC_NAME}_${TIMESTAMP}.data"

echo -e "${GREEN}=== Perf Profiling Session ===${NC}"
echo "Preset: $PRESET"
echo "Executable: $EXECUTABLE"
echo "Output: $PERF_DATA"
echo "Call graph: $CALL_GRAPH"
echo ""

# Run perf record
echo -e "${GREEN}Recording performance data...${NC}"
perf record --call-graph $CALL_GRAPH -o "$PERF_DATA" $EXECUTABLE || {
    echo -e "${RED}perf record failed${NC}"
    echo ""
    echo "If you get a permission error, try:"
    echo "  sudo sysctl -w kernel.perf_event_paranoid=1"
    echo "  or run with sudo"
    exit 1
}

echo ""
echo -e "${GREEN}=== Profiling complete ===${NC}"
echo ""
echo "View the report with:"
echo "  perf report -i $PERF_DATA"
echo ""
echo "Generate an annotated report:"
echo "  perf annotate -i $PERF_DATA"
echo ""
echo "Export to flamegraph (requires FlameGraph tools):"
echo "  perf script -i $PERF_DATA | ./FlameGraph/stackcollapse-perf.pl | ./FlameGraph/flamegraph.pl > ${PERF_DATA%.data}_flamegraph.svg"
