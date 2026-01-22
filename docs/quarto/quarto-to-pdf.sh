#!/bin/bash
# quarto-to-pdf.sh - Convert Quarto documents to PDF

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}=== Subsetix Kokkos - Quarto to PDF ===${NC}"
echo ""

# Check quarto
if ! command -v quarto &> /dev/null; then
    echo -e "${RED}Error: Quarto not installed${NC}"
    echo "Install from: https://quarto.org/docs/get-started/"
    exit 1
fi

# Check LaTeX (optional but recommended)
if ! command -v pdflatex &> /dev/null; then
    echo -e "${YELLOW}Warning: No LaTeX found. Install TinyTeX:${NC}"
    echo "  quarto install tinytex"
    echo ""
fi

cd "$SCRIPT_DIR"

# Parse args
INPUT_FILES=()
OUTPUT_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [FILES...]"
            echo ""
            echo "Options:"
            echo "  -o, --output FILE    Output PDF filename"
            echo "  -h, --help           Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 index.qmd              # Convert to index.pdf"
            echo "  $0 index.qmd -o doc.pdf   # Convert to doc.pdf"
            exit 0
            ;;
        *.qmd)
            INPUT_FILES+=("$1")
            shift
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Default to index.qmd
if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
    INPUT_FILES=("index.qmd")
fi

# Convert each file
for file in "${INPUT_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}Error: File not found: $file${NC}"
        exit 1
    fi

    if [[ -n "$OUTPUT_FILE" ]]; then
        out="$OUTPUT_FILE"
    else
        out="${file%.qmd}.pdf"
    fi

    echo -e "${BLUE}Converting:${NC} $file → $out"
    quarto render "$file" --to pdf --output "$out"
    echo ""
done

echo -e "${GREEN}=== Done! ===${NC}"
echo "PDF files in: $SCRIPT_DIR"
