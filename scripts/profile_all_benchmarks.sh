#!/bin/bash
# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

# Script to profile all experimental benchmarks
# Usage: ./scripts/profile_all_benchmarks.sh <preset>

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
PRESET=""
OUTPUT_DIR="${PERF_OUTPUT_DIR:-./perf_output}"
CALL_GRAPH="${PERF_CALL_GRAPH:-dwarf}"

# Help function
usage() {
    cat << EOF
Usage: $0 <preset> [options]

Profile all experimental benchmarks (all configs and dimensions).

Arguments:
  preset      Build preset (experimental-perf-serial or experimental-perf-openmp)

Options:
  -o, --output DIR     Output directory (default: ./perf_output)
  -c, --call-graph TYPE Call graph mode: dwarf, fp, lbr (default: dwarf)
  -h, --help          Show this help

Examples:
  # Profile all benchmarks with serial backend
  $0 experimental-perf-serial

  # Profile all benchmarks with OpenMP
  $0 experimental-perf-openmp -o ./perf_all

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
        experimental-perf-serial|experimental-perf-openmp)
            PRESET="$1"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$PRESET" ]; then
    echo -e "${RED}Error: Missing preset argument${NC}"
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

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}=== Profiling All Benchmarks ===${NC}"
echo "Preset:        $PRESET"
echo "Binary:        $BENCHMARK_BIN"
echo "Output:        $OUTPUT_DIR"
echo "Call graph:    $CALL_GRAPH"
echo ""

# All benchmark combinations
CONFIGS=("Small" "Medium" "Large")
DIMS=("2D" "3D")

TOTAL=${#CONFIGS[@]}*${#DIMS[@]}
CURRENT=0

for CONFIG in "${CONFIGS[@]}"; do
    for DIM in "${DIMS[@]}"; do
        CURRENT=$((CURRENT + 1))
        FILTER="${CONFIG}${DIM}"
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        PERF_DATA="$OUTPUT_DIR/perf_${PRESET}_${FILTER}_${TIMESTAMP}.data"

        echo -e "${BLUE}[$CURRENT/$TOTAL]${NC} Profiling ${FILTER}..."

        perf record --call-graph $CALL_GRAPH -o "$PERF_DATA" \
            "$BENCHMARK_BIN" --benchmark_filter="$FILTER" > /dev/null 2>&1 || {
            echo -e "${RED}Failed to profile $FILTER${NC}"
            continue
        }

        echo -e "${GREEN}  ✓ $FILTER -> $PERF_DATA${NC}"
    done
done

echo ""
echo -e "${GREEN}=== All benchmarks profiled ===${NC}"
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "View individual reports:"
echo "  perf report -i $OUTPUT_DIR/perf_${PRESET}_*.data"
echo ""
echo "Generate summary:"
echo "  perf report -i $OUTPUT_DIR/perf_${PRESET}_Small2D_*.data"
echo "  perf report -i $OUTPUT_DIR/perf_${PRESET}_Large3D_*.data"
