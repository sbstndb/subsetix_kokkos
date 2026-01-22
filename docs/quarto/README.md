# Subsetix Kokkos - Quarto Documentation

Quarto documentation system for Subsetix Kokkos.

## Quick Start

```bash
# Build HTML
quarto render

# Preview (live reload)
quarto preview

# Convert to PDF (requires LaTeX)
./quarto-to-pdf.sh index.qmd
```

## Scripts

| Script | Purpose |
|--------|---------|
| `compile-quarto.sh` | Build HTML website |
| `preview-quarto.sh` | Local preview server |
| `quarto-to-pdf.sh` | PDF conversion |

## Structure

```
docs/quarto/
├── _quarto.yml          # Configuration
├── index.qmd            # Homepage
├── assets/              # Custom styles
└── _site/               # Generated output (not in git)
```

## Requirements

- Quarto 1.8+ : https://quarto.org/docs/get-started/
- Optional (for PDF): `quarto install tinytex`

## Adding Content

Create `.qmd` files and add them to `_quarto.yml` under `website.sidebar.contents`.
