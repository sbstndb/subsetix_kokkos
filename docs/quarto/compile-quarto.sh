#!/bin/bash
# compile-quarto.sh - Compile Quarto documentation to HTML website

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}=== Subsetix Kokkos - Quarto Documentation Compiler ===${NC}"
echo ""

# Check if quarto is installed
if ! command -v quarto &> /dev/null; then
    echo -e "${RED}Error: Quarto is not installed!${NC}"
    echo ""
    echo "Please install Quarto from: https://quarto.org/docs/get-started/"
    exit 1
fi

echo -e "${YELLOW}Quarto version:${NC}"
quarto --version
echo ""

# Change to the quarto directory
cd "$SCRIPT_DIR"

echo -e "${YELLOW}Building documentation...${NC}"
echo ""

# Run quarto render
quarto render

echo ""
echo -e "${GREEN}=== Build complete! ===${NC}"
echo ""
echo "The documentation has been built in: $SCRIPT_DIR/_site"
echo ""
echo "To preview the documentation locally, run:"
echo "  ./preview-quarto.sh"
echo ""
echo "Or use Quarto's built-in preview:"
echo "  quarto preview"
