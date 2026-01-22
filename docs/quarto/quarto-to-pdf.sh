#!/bin/bash
# quarto-to-pdf.sh - Convert Quarto documentation to PDF

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}=== Subsetix Kokkos - Quarto to PDF Converter ===${NC}"
echo ""

# Check if quarto is installed
if ! command -v quarto &> /dev/null; then
    echo -e "${RED}Error: Quarto is not installed!${NC}"
    echo ""
    echo "Please install Quarto from: https://quarto.org/docs/get-started/"
    exit 1
fi

# Check if tinytex is available (recommended for PDF output)
if ! command -v tlmgr &> /dev/null; then
    echo -e "${YELLOW}Warning: TinyTeX not found. Quarto will use system LaTeX.${NC}"
    echo ""
    echo "For best results, install TinyTeX:"
    echo "  quarto install tinytex"
    echo ""
fi

# Change to the quarto directory
cd "$SCRIPT_DIR"

# Parse command line arguments
OUTPUT_FILE=""
INPUT_FILES=()
COMBINE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -c|--combine)
            COMBINE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [FILES...]"
            echo ""
            echo "Options:"
            echo "  -o, --output FILE    Output PDF filename"
            echo "  -c, --combine        Combine all qmd files into single PDF"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 index.qmd                    # Convert single file"
            echo "  $0 -c -o docs.pdf               # Combine all files to docs.pdf"
            echo "  $0 index.qmd quickstart.qmd     # Convert multiple files"
            exit 0
            ;;
        *.qmd)
            INPUT_FILES+=("$1")
            shift
            ;;
        *)
            echo -e "${RED}Error: Unknown option or file: $1${NC}"
            echo "Use -h for help"
            exit 1
            ;;
    esac
done

# If no files specified and not combining, convert index.qmd
if [[ ${#INPUT_FILES[@]} -eq 0 && "$COMBINE" = false ]]; then
    echo -e "${YELLOW}No files specified. Converting index.qmd by default.${NC}"
    INPUT_FILES=("index.qmd")
fi

# Function to convert a single file
convert_file() {
    local input="$1"
    local output="$2"

    echo -e "${BLUE}Converting:${NC} $input"
    quarto render "$input" --to pdf --output "$output"
}

# Function to combine all files
combine_all() {
    local output="${1:-subsetix-kokkos-docs.pdf}"

    echo -e "${YELLOW}Combining all documentation to:${NC} $output"
    echo ""

    # Create a temporary combined qmd file
    local temp_file="_combined.qmd"

    cat > "$temp_file" << 'EOF'
---
title: "Subsetix Kokkos Documentation"
author: "Subsetix Kokkos Contributors"
format:
  pdf:
    toc: true
    number-sections: true
    color-links: true
---

EOF

    # Append all qmd files in order
    for file in index.qmd quickstart.qmd user-guide/*.qmd architecture/*.qmd; do
        if [[ -f "$file" ]]; then
            echo "" >> "$temp_file"
            echo -e "${BLUE}Adding:${NC} $file"
            # Extract content (skip frontmatter)
            awk '/^---$/{in_fm++; if(in_fm==2) next; next} !in_fm' "$file" >> "$temp_file"
            echo "" >> "$temp_file"
        fi
    done

    echo ""
    echo -e "${YELLOW}Rendering combined PDF...${NC}"
    quarto render "$temp_file" --to pdf --output "$output"

    # Cleanup
    rm -f "$temp_file"
}

# Execute based on mode
if [[ "$COMBINE" = true ]]; then
    combine_all "$OUTPUT_FILE"
else
    for file in "${INPUT_FILES[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo -e "${RED}Error: File not found: $file${NC}"
            exit 1
        fi

        # Determine output filename
        if [[ -n "$OUTPUT_FILE" ]]; then
            out="$OUTPUT_FILE"
        else
            out="${file%.qmd}.pdf"
        fi

        convert_file "$file" "$out"
        echo ""
    done
fi

echo ""
echo -e "${GREEN}=== PDF generation complete! ===${NC}"
echo ""
echo "PDF files created in: $SCRIPT_DIR"
