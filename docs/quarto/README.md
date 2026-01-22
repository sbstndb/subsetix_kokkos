# Subsetix Kokkos - Quarto Documentation

This directory contains the Quarto documentation for Subsetix Kokkos.

## Setup

### Prerequisites

1. Install Quarto from https://quarto.org/docs/get-started/
2. (Optional) Install TinyTeX for PDF output: `quarto install tinytex`

### Quick Start

```bash
# Build the HTML documentation
./compile-quarto.sh

# Preview locally (with live reload)
./preview-quarto.sh
```

## Scripts

### `compile-quarto.sh`

Compiles the Quarto documentation to a static HTML website in `_site/`.

```bash
./compile-quarto.sh
```

### `preview-quarto.sh`

Starts a local web server with live reload for development.

```bash
./preview-quarto.sh
```

The documentation will be available at http://localhost:4000

### `quarto-to-pdf.sh`

Converts Quarto documents to PDF format.

```bash
# Convert single file
./quarto-to-pdf.sh index.qmd

# Convert with custom output name
./quarto-to-pdf.sh index.qmd -o my-docs.pdf

# Combine all files into single PDF
./quarto-to-pdf.sh -c -o subsetix-kokkos-docs.pdf

# Show help
./quarto-to-pdf.sh -h
```

## Directory Structure

```
docs/quarto/
├── _quarto.yml          # Quarto configuration
├── index.qmd            # Homepage
├── quickstart.qmd       # Quick start guide
├── assets/              # Custom styles
│   └── custom.scss
├── user-guide/          # User documentation
├── architecture/        # Architecture documentation
├── design/              # Design documents
└── _site/               # Generated output (not in git)
```

## Adding Content

### New Page

1. Create a `.qmd` file in the appropriate directory
2. Add it to `_quarto.yml` under the relevant section

### New Section

1. Create a new directory
2. Add content files
3. Update `_quarto.yml` to include the new section

## Deployment

The `_site/` directory can be deployed to:
- GitHub Pages
- GitLab Pages
- Netlify
- Vercel
- Any static hosting service

## Resources

- [Quarto Documentation](https://quarto.org/)
- [Quarto for Websites](https://quarto.org/docs/websites/)
