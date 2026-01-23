#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

# Script to compare perf results between two runs
# Usage: ./scripts/compare_perf.sh <perf_data_1> <perf_data_2>

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Help function
usage() {
    cat << EOF
Usage: $0 <perf_data_1> <perf_data_2> [options]

Compare performance results between two perf data files.

Arguments:
  perf_data_1  First perf data file (before)
  perf_data_2  Second perf data file (after)

Options:
  -o, --output DIR     Output directory for reports (default: ./perf_comparison)
  -t, --top N          Show top N functions (default: 20)
  -h, --help          Show this help

Examples:
  # Compare two profiling sessions
  $0 perf_before.data perf_after.data

  # Compare with custom output
  $0 perf_v1.data perf_v2.data -o ./comparison

EOF
    exit 1
}

# Defaults
OUTPUT_DIR="./perf_comparison"
TOP_N=20

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
        -t|--top)
            TOP_N="$2"
            shift 2
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
        *)
            if [ -z "$DATA1" ]; then
                DATA1="$1"
            elif [ -z "$DATA2" ]; then
                DATA2="$1"
            else
                echo -e "${RED}Too many arguments${NC}"
                usage
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$DATA1" ] || [ -z "$DATA2" ]; then
    echo -e "${RED}Error: Missing perf data files${NC}"
    usage
fi

# Check if files exist
if [ ! -f "$DATA1" ]; then
    echo -e "${RED}Error: File not found: $DATA1${NC}"
    exit 1
fi

if [ ! -f "$DATA2" ]; then
    echo -e "${RED}Error: File not found: $DATA2${NC}"
    exit 1
fi

# Check perf availability
if ! command -v perf &> /dev/null; then
    echo -e "${RED}Error: perf is not installed${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

BASENAME1=$(basename "$DATA1" .data)
BASENAME2=$(basename "$DATA2" .data)

echo -e "${GREEN}=== Comparing Perf Results ===${NC}"
echo "Before:  $DATA1"
echo "After:   $DATA2"
echo "Output:  $OUTPUT_DIR"
echo ""

# Generate reports
REPORT1="$OUTPUT_DIR/${BASENAME1}_report.txt"
REPORT2="$OUTPUT_DIR/${BASENAME2}_report.txt"

echo -e "${BLUE}Generating reports...${NC}"

# Generate report for first file
echo "=== $DATA1 ===" > "$REPORT1"
perf report -i "$DATA1" --stdio --no-children -n | head -n $((TOP_N + 7)) >> "$REPORT1" 2>/dev/null || true

# Generate report for second file
echo "=== $DATA2 ===" > "$REPORT2"
perf report -i "$DATA2" --stdio --no-children -n | head -n $((TOP_N + 7)) >> "$REPORT2" 2>/dev/null || true

echo -e "${GREEN}  ✓ Reports generated${NC}"

# Generate side-by-side comparison
COMPARISON="$OUTPUT_DIR/comparison.txt"

echo -e "${BLUE}Generating comparison...${NC}"

cat > "$COMPARISON" << EOF
# Performance Comparison Report
# Generated: $(date)
# Before: $DATA1
# After:  $DATA2

EOF

# Add header
printf "%-50s %15s %15s %15s\n" "Function" "Before %" "After %" "Delta %" >> "$COMPARISON"
printf "%-50s %15s %15s %15s\n" "--------" "--------" "-------" "--------" >> "$COMPARISON"

# Extract top functions from both reports and compare
# This is a simple comparison - for more detailed analysis, use perf diff
{
    echo ""
    echo "=== Detailed comparison with perf diff ==="
    echo ""
    perf diff "$DATA1" "$DATA2" 2>/dev/null || echo "perf diff not available"
} >> "$COMPARISON"

echo -e "${GREEN}  ✓ Comparison generated${NC}"

# Show summary
echo ""
echo -e "${GREEN}=== Comparison Summary ===${NC}"
echo ""
echo "Reports saved:"
echo "  - $REPORT1"
echo "  - $REPORT2"
echo "  - $COMPARISON"
echo ""

# Show top of comparison
if [ -f "$COMPARISON" ]; then
    echo "=== Top Changes ==="
    head -n 30 "$COMPARISON"
fi

echo ""
echo "View full comparison:"
echo "  cat $COMPARISON"
echo ""
echo "Visualize with perf diff:"
echo "  perf diff $DATA1 $DATA2"
