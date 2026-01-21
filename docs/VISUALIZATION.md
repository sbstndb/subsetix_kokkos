# Visualization & Documentation Tools

subsetix includes lightweight Python tools for generating visualizations and documentation **without requiring ParaView**.

---

## 1. VTK Plot Tool (`scripts/viz/vtk_plot.py`)

Generate PNG images and MP4 animations from VTK files.

### Features
- **Zero ParaView dependency** - Custom VTK parser
- **2D/3D plots** - Scatter and surface visualizations
- **Batch processing** - Parallel generation (`--jobs N`)
- **MP4 animations** - From multiple timesteps
- **Headless mode** - Works on servers without display

### Usage

```bash
# Single file
python scripts/viz/vtk_plot.py input.vtk

# Batch processing (parallel)
python scripts/viz/vtk_plot.py output/*.vtk --jobs 8

# Animation from multiple timesteps
python scripts/viz/vtk_plot.py step_*.vtk --animate --fps 30

# Custom field and colormap
python scripts/viz/vtk_plot.py input.vtk --field density --colormap plasma
```

### Dependencies
```bash
pip install matplotlib numpy
# For MP4: sudo apt-get install ffmpeg
```

### Supported Formats
- **Input**: VTK Legacy Binary (`.vtk`)
- **Output**: PNG, MP4

---

## 2. Geometry Gallery Generator (`docs/geometry_gallery/generator.py`)

Auto-generate documentation with code + images for the Geometry2D API.

### Features
- **Compiles C++ examples** - Builds geometry demo executables
- **Runs examples** - Generates VTK files
- **Creates visualizations** - PNG via vtk_plot.py
- **Generates Markdown** - Code + images documentation

### Usage

```bash
# Build and generate all
python docs/geometry_gallery/generator.py

# Skip compilation (if already built)
python docs/geometry_gallery/generator.py --skip-build

# Clean generated files
python docs/geometry_gallery/generator.py --clean
```

### Output Structure
```
docs/geometry_gallery/
├── README.md              # Main gallery index
├── 01_box/
│   └── README.md          # Example with code + image
├── 02_box_with_cylinder/
│   └── README.md
├── .output/               # Generated VTK + PNG files
└── output/                # Symlink for access
```

### Adding New Examples

1. Create `examples/geometry_gallery/05_name.cpp`
2. Add to `examples/geometry_gallery/CMakeLists.txt`
3. Add entry to `EXAMPLES` list in `generator.py`

---

## Workflow

### Typical Workflow

```bash
# 1. Modify C++ code
vim examples/geometry_gallery/05_new_feature.cpp

# 2. Update CMakeLists.txt
vim examples/geometry_gallery/CMakeLists.txt

# 3. Regenerate gallery
python docs/geometry_gallery/generator.py

# 4. View documentation
python3 -m http.server 8080
# Open http://localhost:8080/docs/geometry_gallery/
```

### CI/CD Integration

```cmake
# Add to CMakeLists.txt for automatic visualization generation
find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_custom_target(viz_images
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/viz/vtk_plot.py
        ${CMAKE_CURRENT_BINARY_DIR}/output/*.vtk
        --jobs 4 --output-dir viz_output
    DEPENDS your_executable
)
```

---

## Technical Details

### VTK Format Support
- **VTK Legacy Binary** (version 3.0)
- `DATASET UNSTRUCTURED_GRID`
- Quad cells (type 9)
- Big-endian binary data

### Performance
- **Single file**: ~2-3 img/sec
- **4 cores**: ~8-10 img/sec
- **8 cores**: ~15-18 img/sec

For 1000+ images, use higher `--jobs` values.
