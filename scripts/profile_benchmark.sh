#!/bin/bash
# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

#
# profile_benchmark.sh - Run experimental benchmarks with Kokkos profiling
#
# Usage:
#   ./scripts/profile_benchmark.sh <preset> <tool> <benchmark_filter> [options]
#
# Examples:
#   ./scripts/profile_benchmark.sh experimental-serial-profile kernel-timer "3D.*LargeConfig"
#   ./scripts/profile_benchmark.sh experimental-openmp-profile chrome-tracing "V2.*2D.*MediumConfig"
#   ./scripts/profile_benchmark.sh experimental-cuda-gcc14-profile space-time-stack ".*SmallConfig"
#
# Available tools: kernel-timer, chrome-tracing, space-time-stack

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PRESET=""
TOOL=""
BENCHMARK_FILTER=""
OUTPUT_DIR="${PROJECT_ROOT}/profiling_output"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
OPENMP_THREADS=22
SAMPLER_PROB=""
SAMPLER_VERBOSE=0

# Available profiling tools
declare -A TOOL_LIBS=(
    ["kernel-timer"]="profiling/simple-kernel-timer/libkp_kernel_timer.so"
    ["chrome-tracing"]="profiling/chrome-tracing/libkp_chrome_tracing.so"
    ["space-time-stack"]="profiling/space-time-stack/libkp_space_time_stack.so"
    ["memory-hwm"]="profiling/memory-hwm/libkp_hwm.so"
    ["memory-usage"]="profiling/memory-usage/libkp_memory_usage.so"
)

# Usage
usage() {
    cat << EOF
${BLUE}Usage:${NC}
    $(basename "$0") <preset> <tool> <benchmark_filter> [options]

${BLUE}Arguments:${NC}
    <preset>              CMake preset (e.g., experimental-serial-profile)
    <tool>                Profiling tool: kernel-timer, chrome-tracing, space-time-stack
    <benchmark_filter>    Google Benchmark filter (e.g., "3D.*LargeConfig")

${BLUE}Options:${NC}
    -o, --output DIR      Output directory (default: ${OUTPUT_DIR})
    -t, --threads N      OpenMP threads (default: ${OPENMP_THREADS})
    -s, --sampling-prob N  Sampling probability 1-100 (default: 100=no sampling)
    -v, --sampler-verbose Enable sampler verbose output
    -h, --help           Show this help message

${BLUE}Examples:${NC}
    # Profile 3D LargeConfig with kernel-timer (Serial)
    $(basename "$0") experimental-serial-profile kernel-timer "3D.*LargeConfig"

    # Profile 2D MediumConfig with chrome-tracing (OpenMP), 10% sampling
    $(basename "$0") experimental-openmp-profile chrome-tracing "2D.*MediumConfig" -t 22 -s 10

    # Profile SmallConfig with space-time-stack (CUDA), 5% sampling
    $(basename "$0") experimental-cuda-gcc14-profile space-time-stack ".*SmallConfig" -s 5

${BLUE}Available tools:${NC}
    kernel-timer     - Simple kernel timing (.dat output, use with kp_json_writer)
    chrome-tracing    - Chrome timeline JSON (open in chrome://tracing)
    space-time-stack - Detailed time + memory report (stdout)

${BLUE}Sampling:${NC}
    -s N                Sampling probability (1-100%, default: 100=no sampling)
    Reduces profiling overhead by only measuring N% of kernels.
    Recommended values: 1-10% for detailed tools, 10-20% for chrome-tracing.
    Use -v to see which kernels are sampled.

${BLUE}Output:${NC}
    Traces saved to: ${OUTPUT_DIR}/<run_id>/
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
        -t|--threads)
            OPENMP_THREADS="$2"
            shift 2
            ;;
        -s|--sampling-prob)
            SAMPLER_PROB="$2"
            shift 2
            ;;
        -v|--sampler-verbose)
            SAMPLER_VERBOSE=1
            shift
            ;;
        -*)
            echo "${RED}Error: Unknown option $1${NC}"
            usage
            ;;
        *)
            if [[ -z "$PRESET" ]]; then
                PRESET="$1"
            elif [[ -z "$TOOL" ]]; then
                TOOL="$1"
            elif [[ -z "$BENCHMARK_FILTER" ]]; then
                BENCHMARK_FILTER="$1"
            else
                echo "${RED}Error: Too many arguments${NC}"
                usage
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$PRESET" ]] || [[ -z "$TOOL" ]] || [[ -z "$BENCHMARK_FILTER" ]]; then
    echo "${RED}Error: Missing required arguments${NC}"
    usage
fi

# Validate tool
if [[ -z "${TOOL_LIBS[$TOOL]}" ]]; then
    echo "${RED}Error: Unknown tool '$TOOL'. Available: ${!TOOL_LIBS[@]}${NC}"
    exit 1
fi

# Create output directory
OUTPUT_DIR="${OUTPUT_DIR}/${RUN_ID}"
mkdir -p "$OUTPUT_DIR"

# Determine build directory
BUILD_DIR="${PROJECT_ROOT}/build-${PRESET}"
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "${RED}Error: Build directory not found: $BUILD_DIR${NC}"
    echo "${YELLOW}Hint: Run 'cmake --preset $PRESET' first${NC}"
    exit 1
fi

# Find profiling library
TOOL_LIB="${TOOL_LIBS[$TOOL]}"
TOOL_PATH="${BUILD_DIR}/_deps/kokkos_tools-build/${TOOL_LIB}"

if [[ ! -f "$TOOL_PATH" ]]; then
    echo "${RED}Error: Profiling library not found: $TOOL_PATH${NC}"
    echo "${YELLOW}Hint: Make sure SUBSETIX_ENABLE_KOKKOS_TOOLS=ON when configuring${NC}"
    exit 1
fi

# Benchmark executable
BENCHMARK_BIN="${BUILD_DIR}/experimental/benchmarks/experimental_comparison_benchmark"
if [[ ! -f "$BENCHMARK_BIN" ]]; then
    echo "${RED}Error: Benchmark not found: $BENCHMARK_BIN${NC}"
    echo "${YELLOW}Hint: Run 'cmake --build --preset $PRESET' first${NC}"
    exit 1
fi

# Print configuration
if [[ -n "$SAMPLER_PROB" ]]; then
    SAMPLER_INFO=" (sampling: ${SAMPLER_PROB}%)"
else
    SAMPLER_INFO=""
fi

cat << EOF
${BLUE}=== Kokkos Profiling Session ===${NC}
${GREEN}Preset:${NC}        $PRESET
${GREEN}Tool:${NC}          $TOOL $SAMPLER_INFO
${GREEN}Benchmark:${NC}     $BENCHMARK_FILTER
${GREEN}Threads:${NC}       $OPENMP_THREADS
${GREEN}Output:${NC}        $OUTPUT_DIR
${GREEN}Tool Library:${NC}  $TOOL_PATH

${BLUE}===================================${NC}
EOF

# Change to project root (so trace files are created there)
cd "$PROJECT_ROOT"

# Set environment variables
export OMP_NUM_THREADS="$OPENMP_THREADS"
export OMP_PROC_BIND="spread"

# Configure profiling library with sampler if requested
if [[ -n "$SAMPLER_PROB" ]]; then
    SAMPLER_LIB="${BUILD_DIR}/_deps/kokkos_tools-build/common/kokkos-sampler/libkp_kokkos_sampler.so"
    if [[ ! -f "$SAMPLER_LIB" ]]; then
        echo "${RED}Error: Sampler library not found: $SAMPLER_LIB${NC}"
        exit 1
    fi
    export KOKKOS_TOOLS_LIBS="${SAMPLER_LIB};${TOOL_PATH}"
    export KOKKOS_TOOLS_SAMPLER_PROB="$SAMPLER_PROB"
    if [[ $SAMPLER_VERBOSE -eq 1 ]]; then
        export KOKKOS_TOOLS_SAMPLER_VERBOSE=1
    fi
    echo "${GREEN}Using sampling: ${SAMPLER_PROB}%${NC}"
else
    export KOKKOS_PROFILE_LIBRARY="$TOOL_PATH"
fi

echo "${BLUE}Running benchmark...${NC}"

"$BENCHMARK_BIN" \
    --benchmark_filter="$BENCHMARK_FILTER" \
    --benchmark_repetitions=1 \
    --benchmark_min_time=0s \
    2>&1 | tee "$OUTPUT_DIR/benchmark_output.txt"

# Move trace files to output directory
echo ""
echo "${BLUE}Moving trace files...${NC}"

case "$TOOL" in
    kernel-timer)
        # Move .dat files
        for f in *.dat; do
            if [[ -f "$f" ]]; then
                mv "$f" "$OUTPUT_DIR/"
                echo "${GREEN}✓${NC} Moved $f"
            fi
        done
        ;;
    chrome-tracing)
        # Move .json files
        for f in *.json; do
            if [[ -f "$f" && "$f" != "CMakePresets.json" ]]; then
                mv "$f" "$OUTPUT_DIR/"
                echo "${GREEN}✓${NC} Moved $f"
            fi
        done
        ;;
    space-time-stack)
        # Report is in benchmark_output.txt, but save summary
        echo "${GREEN}✓${NC} Space-time-stack report saved to $OUTPUT_DIR/benchmark_output.txt"
        ;;
    memory-hwm)
        # Memory high water mark report is in stdout
        echo "${GREEN}✓${NC} Memory HWM report captured to $OUTPUT_DIR/benchmark_output.txt"
        ;;
    memory-usage)
        # Memory usage report is in stdout
        echo "${GREEN}✓${NC} Memory usage report captured to $OUTPUT_DIR/benchmark_output.txt"
        ;;
esac

# Create summary
cat << EOF > "$OUTPUT_DIR/summary.txt"
=== Profiling Session Summary ===
Date: $(date)
Preset: $PRESET
Tool: $TOOL
Benchmark Filter: $BENCHMARK_FILTER
OpenMP Threads: $OPENMP_THREADS
Tool Library: $TOOL_PATH
Sampling: ${SAMPLER_PROB:-100}%

Output Files:
EOF

ls -1 "$OUTPUT_DIR" | grep -v "summary.txt" | sed 's/^/  - /' >> "$OUTPUT_DIR/summary.txt"

echo ""
echo "${GREEN}=== Profiling complete! ===${NC}"
echo "${BLUE}Trace files:${NC} $OUTPUT_DIR"
echo "${BLUE}Summary:${NC}      $OUTPUT_DIR/summary.txt"
echo ""
echo "${BLUE}Next steps:${NC}"
if [[ -n "$SAMPLER_PROB" ]]; then
    echo "  ${GREEN}Sampling was enabled at ${SAMPLER_PROB}% to reduce overhead${NC}"
    echo "  Re-run without -s option for complete profiling data"
fi
case "$TOOL" in
    kernel-timer)
        echo "  Convert .dat to JSON:"
        echo "    ${BUILD_DIR}/_deps/kokkos_tools-build/profiling/simple-kernel-timer/kp_json_writer $OUTPUT_DIR/*.dat"
        ;;
    chrome-tracing)
        echo "  Open in Chrome:"
        echo "    1. Open chrome://tracing"
        echo "    2. Click 'Load' and select $OUTPUT_DIR/*.json"
        ;;
    space-time-stack)
        echo "  Report is in $OUTPUT_DIR/benchmark_output.txt"
        ;;
    memory-hwm)
        echo "  Memory HWM report is in $OUTPUT_DIR/benchmark_output.txt"
        echo "  Look for: 'KokkosP: High water mark memory consumption: XXX kB'"
        ;;
    memory-usage)
        echo "  Memory usage report is in $OUTPUT_DIR/benchmark_output.txt"
        echo "  Look for: 'Memory Usage' and 'High water mark' sections"
        ;;
esac
