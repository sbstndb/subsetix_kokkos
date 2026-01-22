#!/bin/bash
# preview-quarto.sh - Preview Quarto documentation locally

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}=== Subsetix Kokkos - Quarto Documentation Preview ===${NC}"
echo ""

# Check if quarto is installed
if ! command -v quarto &> /dev/null; then
    echo -e "${RED}Error: Quarto is not installed!${NC}"
    echo ""
    echo "Please install Quarto from: https://quarto.org/docs/get-started/"
    exit 1
fi

# Change to the quarto directory
cd "$SCRIPT_DIR"

echo -e "${YELLOW}Starting preview server...${NC}"
echo ""
echo "The documentation will be available at:"
echo "  http://localhost:4000 (or available port)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run quarto preview
quarto preview
