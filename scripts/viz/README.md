# subsetix Visualization Tool

Minimalist VTK visualization tool for subsetix - generates PNG images and MP4 animations from VTK legacy binary files.

## Features

- **Zero external VTK dependencies** - Custom VTK parser (no meshio/VTK required)
- **2D plots** - Scatter visualization for sparse CSR meshes
- **3D plots** - 3D surface visualization with field value as Z-coordinate
- **MP4 animations** - Create animations from multiple timesteps
- **Parallel processing** - Multi-core batch generation for thousands of images
- **Headless mode** - Works on servers without display (matplotlib Agg backend)
- **Configurable** - CLI arguments + YAML config file support

## Installation

```bash
# Python dependencies
pip install matplotlib numpy

# Optional: for MP4 output
sudo apt-get install ffmpeg  # Linux
brew install ffmpeg          # macOS
```

## Quick Start

```bash
# Single file
python scripts/viz/vtk_plot.py input.vtk

# Batch processing (parallel)
python scripts/viz/vtk_plot.py output/*.vtk --jobs 8

# Create MP4 animation
python scripts/viz/vtk_plot.py step_*.vtk --animate --fps 30

# With custom colormap
python scripts/viz/vtk_plot.py input.vtk --colormap plasma --field density

# Hide mesh edges
python scripts/viz/vtk_plot.py input.vtk --no-mesh

# 3D visualization
python scripts/viz/vtk_plot.py input.vtk --3d
```

## CLI Options

### Data Selection
```
--field, -f       Scalar field name (default: auto-detect)
--mesh            Show mesh edges (default: True)
--no-mesh         Hide mesh edges
```

### Visual Style
```
--colormap, -c    Matplotlib colormap (viridis, plasma, jet, coolwarm, etc.)
--vmin            Minimum value for colormap (default: auto)
--vmax            Maximum value for colormap (default: auto)
--no-cbar         Hide colorbar
```

### Dimension
```
--2d              Force 2D plot
--3d              Force 3D plot
```

### Output
```
--output, -o      Output file (single) or directory (batch)
--output-dir      Output directory for batch (default: viz_output/png)
--dpi             Resolution in DPI (default: 150)
```

### Performance
```
--jobs, -j        Number of parallel jobs (default: 4)
--cache           VTK cache size (default: 100)
```

### Animation
```
--animate         Create MP4 animation
--fps             Frames per second (default: 30)
```

### Figure
```
--figsize         Figure size as W,H in inches (default: 10,8)
```

## Examples

### Generate images for all demos
```bash
# Mach 2 cylinder demo
python scripts/viz/vtk_plot.py \
    build-serial/output/mach2_cylinder_step_*.vtk \
    --field rho \
    --colormap plasma \
    --jobs 8 \
    --output-dir viz_output/mach2_cylinder
```

### Create animation
```bash
# Create MP4 from timesteps
python scripts/viz/vtk_plot.py \
    build-serial/output/mach2_cylinder_step_*.vtk \
    --field rho \
    --animate \
    --fps 30 \
    --output viz_output/mach2_cylinder_rho.mp4
```

### Custom visualization
```bash
# High-res, custom colormap, no colorbar
python scripts/viz/vtk_plot.py input.vtk \
    --field momentum_x \
    --colormap RdYlBu \
    --no-cbar \
    --dpi 300 \
    --figsize 12,8
```

### 3D visualization
```bash
# 3D surface plot
python scripts/viz/vtk_plot.py input.vtk \
    --field pressure \
    --3d \
    --colormap viridis
```

## Output Structure

```
viz_output/
├── png/                    # PNG images
│   ├── mach2_cylinder/
│   │   ├── step_00050_rho.png
│   │   ├── step_00100_rho.png
│   │   └── ...
│   └── amr_advection/
│       └── ...
└── mp4/                    # MP4 animations
    ├── mach2_cylinder_rho.mp4
    └── ...
```

## Supported VTK Format

The tool reads **VTK Legacy Binary** format (version 3.0) with:
- `DATASET UNSTRUCTURED_GRID`
- Quad cells (type 9)
- Cell-centered scalar fields
- Big-endian binary data

This matches the format output by subsetix `vtk_export.hpp`.

## Performance

| Configuration | Speed (images/sec) |
|--------------|-------------------|
| Single core (jobs=1) | ~2-3 img/s |
| 4 cores (jobs=4) | ~8-10 img/s |
| 8 cores (jobs=8) | ~15-18 img/s |

For 1000+ images, use `--jobs 8` or higher.

## Troubleshooting

### No scalar fields found
```bash
# Check available fields
python -c "
from scripts.viz.vtk_plot import VTKReader
reader = VTKReader()
data = reader.read('your_file.vtk')
print('Fields:', list(data['cell_data'].keys()))
"
```

### MP4 generation fails
```bash
# Install ffmpeg
sudo apt-get install ffmpeg

# Or fallback to GIF (automatic)
python scripts/viz/vtk_plot.py step_*.vtk --animate  # Falls back to GIF
```

### Memory issues with large files
```bash
# Reduce cache size
python scripts/viz/vtk_plot.py *.vtk --cache 10

# Process in smaller batches
python scripts/viz/vtk_plot.py part1_*.vtk --output-dir out1
python scripts/viz/vtk_plot.py part2_*.vtk --output-dir out2
```

## Integration with CMake

Add to your `CMakeLists.txt` for automatic visualization generation:

```cmake
# Find Python
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Add visualization target
add_custom_target(viz_images
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/viz/vtk_plot.py
        ${CMAKE_CURRENT_BINARY_DIR}/output/*.vtk
        --jobs 4
        --output-dir ${CMAKE_CURRENT_BINARY_DIR}/viz_output
    DEPENDS your_executable
    COMMENT "Generating visualization images"
)
```

## License

Part of the subsetix project.
