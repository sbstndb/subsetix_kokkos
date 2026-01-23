#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

# Script to generate a consolidated performance report from perf data files
# Usage: ./scripts/generate_perf_report.sh <perf_data_dir>

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
PERF_DIR=""
OUTPUT_DIR="./perf_reports"
TOP_N=20

# Help function
usage() {
    cat << EOF
Usage: $0 <perf_data_dir> [options]

Generate a consolidated performance report from multiple perf data files.

Arguments:
  perf_data_dir  Directory containing perf.data files

Options:
  -o, --output DIR     Output directory for reports (default: ./perf_reports)
  -t, --top N          Show top N functions (default: 20)
  -h, --help          Show this help

Examples:
  # Generate report from perf output directory
  $0 ./perf_output

  # Generate report with custom output
  $0 ./perf_output -o ./my_report

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
        -t|--top)
            TOP_N="$2"
            shift 2
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
        *)
            if [ -z "$PERF_DIR" ]; then
                PERF_DIR="$1"
            else
                echo -e "${RED}Too many arguments${NC}"
                usage
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$PERF_DIR" ]; then
    echo -e "${RED}Error: Missing perf data directory${NC}"
    usage
fi

# Check if directory exists
if [ ! -d "$PERF_DIR" ]; then
    echo -e "${RED}Error: Directory not found: $PERF_DIR${NC}"
    exit 1
fi

# Check perf availability
if ! command -v perf &> /dev/null; then
    echo -e "${RED}Error: perf is not installed${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$OUTPUT_DIR/perf_report_${TIMESTAMP}.txt"

echo -e "${GREEN}=== Generating Performance Report ===${NC}"
echo "Source:  $PERF_DIR"
echo "Output:  $REPORT_FILE"
echo ""

# Initialize report
cat > "$REPORT_FILE" << EOF
# Consolidated Performance Report
# Generated: $(date)
# Source directory: $PERF_DIR

EOF

# Find all perf.data files
PERF_FILES=$(find "$PERF_DIR" -name "*.data" -type f 2>/dev/null || true)

if [ -z "$PERF_FILES" ]; then
    echo -e "${YELLOW}No perf.data files found in $PERF_DIR${NC}"
    exit 0
fi

FILE_COUNT=$(echo "$PERF_FILES" | wc -l)
echo -e "${BLUE}Found $FILE_COUNT perf data files${NC}"
echo ""

# Process each file
CURRENT=0
for PERF_FILE in $PERF_FILES; do
    CURRENT=$((CURRENT + 1))
    BASENAME=$(basename "$PERF_FILE" .data)

    echo -e "${BLUE}[$CURRENT/$FILE_COUNT]${NC} Processing $BASENAME..."

    # Add to report
    {
        echo ""
        echo "================================================================================"
        echo "File: $BASENAME"
        echo "Path: $PERF_FILE"
        echo "================================================================================"
        echo ""
        perf report -i "$PERF_FILE" --stdio --no-children -n 2>/dev/null | head -n $((TOP_N + 7)) || echo "Error processing file"
        echo ""
    } >> "$REPORT_FILE"

    echo -e "${GREEN}  ✓ $BASENAME${NC}"
done

# Generate summary
echo ""
echo -e "${BLUE}Generating summary...${NC}"

{
    echo ""
    echo "================================================================================"
    echo "SUMMARY"
    echo "================================================================================"
    echo ""
    echo "Total files processed: $FILE_COUNT"
    echo "Report generated: $(date)"
    echo ""
    echo "Top functions across all files:"
    echo "  (Analyze individual reports above for detailed breakdown)"
    echo ""
} >> "$REPORT_FILE"

echo -e "${GREEN}  ✓ Summary generated${NC}"

echo ""
echo -e "${GREEN}=== Report Complete ===${NC}"
echo ""
echo "Report saved to: $REPORT_FILE"
echo ""
echo "View report:"
echo "  cat $REPORT_FILE"
echo "  less $REPORT_FILE"
echo ""
echo "Generate HTML report:"
echo "  sed 's/$/<br>/' $REPORT_FILE > ${REPORT_FILE%.txt}.html"
