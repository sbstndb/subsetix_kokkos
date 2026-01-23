#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

# Script to profile experimental benchmarks with Linux perf
# Usage: ./scripts/profile_benchmark_perf.sh <preset> <config> <dimension> [options]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
PRESET=""
CONFIG=""
DIMENSION=""
OUTPUT_DIR="${PERF_OUTPUT_DIR:-./perf_output}"
CALL_GRAPH="${PERF_CALL_GRAPH:-dwarf}"
EXTRA_ARGS=""

# Help function
usage() {
    cat << EOF
Usage: $0 <preset> <config> <dimension> [options]

Profile experimental benchmarks with perf.

Arguments:
  preset      Build preset (experimental-perf-serial or experimental-perf-openmp)
  config      Benchmark size (Small, Medium, Large)
  dimension   Dimension (2D or 3D)

Options:
  -o, --output DIR     Output directory (default: ./perf_output)
  -c, --call-graph TYPE Call graph mode: dwarf, fp, lbr (default: dwarf)
  -e, --events EVENTS  Custom perf events (default: cycles)
  --stat              Use perf stat instead of perf record
  -h, --help          Show this help

Examples:
  # Profile Small 2D benchmarks with serial backend
  $0 experimental-perf-serial Small 2D

  # Profile Medium 2D benchmarks with OpenMP, custom output
  $0 experimental-perf-openmp Medium 2D -o ./my_perf

  # Profile Large 3D benchmarks with custom events
  $0 experimental-perf-serial Large 3D -e cycles,instructions,cache-misses

  # Use perf stat for real-time statistics
  $0 experimental-perf-serial Small 2D --stat

EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--call-graph)
            CALL_GRAPH="$2"
            shift 2
            ;;
        -e|--events)
            PERF_EVENTS="$2"
            shift 2
            ;;
        --stat)
            USE_STAT=1
            shift
            ;;
        experimental-perf-serial|experimental-perf-openmp)
            PRESET="$1"
            shift
            ;;
        Small|Medium|Large)
            CONFIG="$1"
            shift
            ;;
        2D|3D)
            DIMENSION="$1"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$PRESET" ] || [ -z "$CONFIG" ] || [ -z "$DIMENSION" ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    usage
fi

# Check perf availability
if ! command -v perf &> /dev/null; then
    echo -e "${RED}Error: perf is not installed${NC}"
    echo "Install with: sudo apt-get install linux-tools-common linux-tools-generic"
    exit 1
fi

# Determine build directory
BUILD_DIR=""
case $PRESET in
    experimental-perf-serial)
        BUILD_DIR="./build-experimental-perf-serial"
        ;;
    experimental-perf-openmp)
        BUILD_DIR="./build-experimental-perf-openmp"
        ;;
    *)
        echo -e "${RED}Error: Unknown preset '$PRESET'${NC}"
        exit 1
        ;;
esac

BENCHMARK_BIN="$BUILD_DIR/experimental/benchmarks/experimental_comparison_benchmark"

# Check if benchmark exists
if [ ! -f "$BENCHMARK_BIN" ]; then
    echo -e "${RED}Error: Benchmark binary not found: $BENCHMARK_BIN${NC}"
    echo "Build first with: cmake --preset $PRESET && cmake --build --preset $PRESET"
    exit 1
fi

# Build filter string
# Benchmark names: V1_SmallConfig, V2_SmallConfig, V3_SmallConfig (2D)
#                  V1_3D_SmallConfig, V2_3D_SmallConfig, V3_3D_SmallConfig (3D)
if [ "$DIMENSION" = "3D" ]; then
  FILTER="3D_${CONFIG}Config"
else
  FILTER="${CONFIG}Config"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate output filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PERF_DATA="$OUTPUT_DIR/perf_${PRESET}_${FILTER}_${TIMESTAMP}.data"

echo -e "${GREEN}=== Benchmark Profiling Session ===${NC}"
echo "Preset:        $PRESET"
echo "Config:        $CONFIG"
echo "Dimension:     $DIMENSION"
echo "Filter:        $FILTER"
echo "Binary:        $BENCHMARK_BIN"
echo "Output:        $PERF_DATA"
echo "Call graph:    $CALL_GRAPH"
if [ -n "$PERF_EVENTS" ]; then
    echo "Events:        $PERF_EVENTS"
fi
echo ""

# Run perf
if [ "$USE_STAT" = "1" ]; then
    # Use perf stat for real-time statistics
    echo -e "${GREEN}Running perf stat...${NC}"
    if [ -n "$PERF_EVENTS" ]; then
        perf stat -e $PERF_EVENTS "$BENCHMARK_BIN" --benchmark_filter="$FILTER"
    else
        perf stat -e cycles,instructions,cache-misses,branches,branch-misses \
            "$BENCHMARK_BIN" --benchmark_filter="$FILTER"
    fi
else
    # Use perf record for detailed analysis
    echo -e "${GREEN}Recording performance data...${NC}"
    RECORD_CMD="perf record --call-graph $CALL_GRAPH -o $PERF_DATA"

    if [ -n "$PERF_EVENTS" ]; then
        RECORD_CMD="$RECORD_CMD -e $PERF_EVENTS"
    fi

    $RECORD_CMD "$BENCHMARK_BIN" --benchmark_filter="$FILTER" || {
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
    echo "Generate annotated report:"
    echo "  perf annotate -i $PERF_DATA"
    echo ""
    echo "Generate flamegraph:"
    echo "  perf script -i $PERF_DATA | ./FlameGraph/stackcollapse-perf.pl | ./FlameGraph/flamegraph.pl > ${PERF_DATA%.data}_flamegraph.svg"
fi
