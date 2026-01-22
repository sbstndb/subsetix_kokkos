# Quarto Documentation

## Location

`docs/quarto/` - Quarto documentation system.

## Quick Commands

```bash
cd docs/quarto

# Build HTML website
quarto render

# Preview locally (live reload)
quarto preview

# Convert to PDF (requires LaTeX)
./quarto-to-pdf.sh index.qmd
```

## Scripts

| Script | Purpose |
|--------|---------|
| `compile-quarto.sh` | Build HTML website |
| `preview-quarto.sh` | Local preview server |
| `quarto-to-pdf.sh` | PDF conversion (requires LaTeX) |

## Adding Content

1. Create `.qmd` files in `docs/quarto/`
2. Add entries to `_quarto.yml` under `website.sidebar.contents`

## Theme

Default: `cosmo` (Bootstrap-based). Edit `_quarto.yml` to change.

## Requirements

- **Quarto** 1.8+ : https://quarto.org/docs/get-started/
- **Optional** (for PDF): LaTeX or `quarto install tinytex`

## Current State

Minimal setup with `index.qmd` placeholder. Content to be added.
