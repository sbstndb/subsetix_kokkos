# Quarto Documentation

## Location

`docs/quarto/` - Quarto documentation website and presentations.

## Quick Commands

```bash
cd docs/quarto

# Build HTML website
quarto render

# Preview locally (live reload)
quarto preview
# or
./preview-quarto.sh

# Convert to PDF
./quarto-to-pdf.sh -c -o docs.pdf  # All files combined
./quarto-to-pdf.sh index.qmd        # Single file
```

## Scripts

| Script | Purpose |
|--------|---------|
| `compile-quarto.sh` | Build HTML website |
| `preview-quarto.sh` | Local preview server |
| `quarto-to-pdf.sh` | PDF conversion |

## Adding Content

1. Create `.qmd` files in appropriate subdirectories
2. Add entries to `_quarto.yml` under `website.sidebar.contents`

## Theme

Default: `cosmo` (Bootstrap-based). Edit `_quarto.yml` to change.

## Requirements

- Quarto 1.8+ : https://quarto.org/docs/get-started/
- Optional: TinyTeX for PDF (`quarto install tinytex`)
