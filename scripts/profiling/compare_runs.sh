#!/bin/bash
# SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
# SPDX-License-Identifier: BSD-3-Clause

#
# compare_runs.sh - Compare multiple profiling runs
#
# Usage:
#   ./scripts/profiling/compare_runs.sh <trace_dir1> <trace_dir2> ...

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
    $(basename "$0") <trace_directory1> <trace_directory2> ...

${BLUE}Description:${NC}
    Compare benchmark results from multiple profiling runs.

${BLUE}Examples:${NC}
    # Compare Serial vs OpenMP profiling runs
    $(basename "$0") profiling_output/20250122-145000-serial profiling_output/20250122-150000-openmp

    # Compare all runs from a session
    $(basename "$0") profiling_output/20250122-*/

${BLUE}Output:${NC}
    - Comparison table of benchmark times
    - Ranking of fastest/slowest runs
EOF
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

# Collect trace directories
TRACE_DIRS=()
for arg in "$@"; do
    if [[ -d "$arg" ]]; then
        TRACE_DIRS+=("$arg")
    else
        echo "${YELLOW}Warning: Skipping non-directory: $arg${NC}"
    fi
done

if [[ ${#TRACE_DIRS[@]} -eq 0 ]]; then
    echo "${RED}Error: No valid trace directories found${NC}"
    exit 1
fi

echo "${BLUE}=== Comparing ${#TRACE_DIRS[@]} profiling runs ===${NC}"
echo ""

# Function to extract benchmark results from a directory
extract_results() {
    local dir="$1"
    local name
    name=$(basename "$dir")

    # Try to parse from benchmark_output.txt
    local output_file="$dir/benchmark_output.txt"
    if [[ ! -f "$output_file" ]]; then
        return
    fi

    # Extract benchmark results using grep and awk
    grep "RandomMeshBenchmark" "$output_file" 2>/dev/null | \
        awk '{
            # Split by whitespace
            for(i=1;i<=NF;i++) {
                if($i ~ /ns$/) {
                    time_ns=$i
                    gsub(/ns/, "", time_ns)
                }
                if($i ~ /items_per_second=/) {
                    split($i, arr, "=")
                    items_per_s=arr[2]
                }
            }
            printf "%s|%s|%s\n", name, time_ns, items_per_s
        }'
}

# Collect all results
echo "${BLUE}Extracting benchmark results...${NC}"
TEMP_FILE=$(mktemp)
for dir in "${TRACE_DIRS[@]}"; do
    extract_results "$dir" >> "$TEMP_FILE"
done

# Check if we have results
if [[ ! -s "$TEMP_FILE" ]]; then
    echo "${RED}Error: No benchmark results found in trace directories${NC}"
    rm -f "$TEMP_FILE"
    exit 1
fi

echo "${GREEN}✓ Results extracted${NC}"
echo ""

# Parse and organize results
declare -A BENCHMARK_TIMES
declare -A BENCHMARK_ITEMS

while IFS='|' read -r run time_ns items_per_s; do
    # Extract benchmark name (remove run_id prefix if present)
    bench=$(echo "$run" | sed 's/^[0-9]*-[0-9]*-[0-9]*-//')
    key="${run}::${bench}"
    BENCHMARK_TIMES[$key]="$time_ns"
    BENCHMARK_ITEMS[$key]="$items_per_s"
done < "$TEMP_FILE"

rm -f "$TEMP_FILE"

# Get unique benchmarks
benchmarks=$(for key in "${!BENCHMARK_TIMES[@]}"; do
    echo "$key" | cut -d':' -f2
done | sort -u)

echo "${CYAN}=== Benchmark Comparison ===${NC}"
echo ""

# For each benchmark, create a comparison table
for bench in $benchmarks; do
    echo "${BLUE}${bench}:${NC}"
    printf "%-40s %12s %15s\n" "Run" "Time (ms)" "Items/s"
    printf "%-40s %12s %15s\n" "----" "---------" "--------"

    # Collect results for this benchmark across all runs
    declare -a run_times=()
    declare -a run_names=()

    for key in "${!BENCHMARK_TIMES[@]}"; do
        key_bench=$(echo "$key" | cut -d':' -f2)
        if [[ "$key_bench" == "$bench" ]]; then
            run_name=$(echo "$key" | cut -d':' -f1)
            time_ns=${BENCHMARK_TIMES[$key]}
            time_ms=$(echo "scale=2; $time_ns / 1000000" | bc)
            items_s=${BENCHMARK_ITEMS[$key]}
            run_names+=("$run_name")
            run_times+=("$time_ms|$items_s")
        fi
    done

    # Sort by time and display
    for i in $(seq 0 $((${#run_times[@]} - 1))); do
        for j in $(seq $i $((${#run_times[@]} - 1))); do
            time1=$(echo "${run_times[$i]}" | cut -d'|' -f1)
            time2=$(echo "${run_times[$j]}" | cut -d'|' -f1)
            if [[ $(echo "$time1 < $time2" | bc -l) -eq 1 ]]; then
                temp="${run_times[$i]}"
                run_times[$i]="${run_times[$j]}"
                run_times[$j]="$temp"
            fi
        done
    done

    for i in $(seq 0 $((${#run_times[@]} - 1))); do
        time_ms=$(echo "${run_times[$i]}" | cut -d'|' -f1)
        items_s=$(echo "${run_times[$i]}" | cut -d'|' -f2)
        run_name="${run_names[$i]}"
        printf "%-40s %12s %15s\n" "$run_name" "$time_ms" "$items_s"
    done
    echo ""
done

# Summary
echo "${CYAN}=== Summary ===${NC}"
echo ""
echo "${BLUE}Total runs compared:${NC} ${#TRACE_DIRS[@]}"
echo "${BLUE}Total benchmarks:${NC} $(echo "$benchmarks" | wc -l)"
echo ""
echo "${GREEN}Done!${NC}"
