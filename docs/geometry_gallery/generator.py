#!/usr/bin/env python3
"""
geometry_gallery/generator.py

Generate documentation for subsetix Geometry2D API.

This script:
1. Compiles geometry example executables
2. Runs each example to generate VTK files
3. Converts VTK to PNG images using vtk_plot.py
4. Creates markdown documentation with code + images

Usage:
    python generator.py                    # Build and generate all
    python generator.py --skip-build       # Skip compilation, just generate docs
    python generator.py --clean            # Clean build artifacts
"""

import os
import sys
import re
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    def __init__(self):
        # Paths
        self.script_dir = Path(__file__).parent
        self.examples_dir = Path(__file__).parent.parent.parent / "examples" / "geometry_gallery"
        # Use .output (hidden) for actual files, will create symlink 'output' for access
        self.output_dir = Path(__file__).parent / ".output"
        self.build_dir = Path(__file__).parent.parent.parent / "build-serial"
        self.vtk_plot_path = Path(__file__).parent.parent.parent / "scripts" / "viz" / "vtk_plot.py"

        # Build
        self.cmake_args = []

        # Visualization
        self.dpi = 150
        self.colormap = "viridis"

# ============================================================================
# EXAMPLE METADATA
# ============================================================================

EXAMPLES = [
    {
        "name": "01_box",
        "title": "Simple Rectangular Domain",
        "description": """
        Creates a basic rectangular computational domain using `build_box()`.

        This is the starting point for any simulation - a simple box filled with fluid cells.
        """,
        "file": "01_box.cpp",
    },
    {
        "name": "02_box_with_cylinder",
        "title": "Box with Cylinder Obstacle",
        "description": """
        Demonstrates adding a circular obstacle using `add_cylinder()`.

        This is the classic "flow around cylinder" test case, widely used in CFD for
        validation of solvers. The cylinder is removed from the fluid domain.
        """,
        "file": "02_box_with_cylinder.cpp",
    },
    {
        "name": "03_multiple_cylinders",
        "title": "Multiple Cylinder Obstacles",
        "description": """
        Shows adding multiple circular obstacles in a staggered pattern.

        This configuration could represent flow through a tube bundle or heat exchanger,
        where fluid flows around an array of cylinders.
        """,
        "file": "03_multiple_cylinders.cpp",
    },
    {
        "name": "04_complex_geometry",
        "title": "Complex Geometry with Mixed Shapes",
        "description": """
        Demonstrates combining different primitive types: rectangles and cylinders.

        Shows how to build complex geometries by combining multiple obstacles
        of different types, including fluid regions (using `is_obstacle=false`).
        """,
        "file": "04_complex_geometry.cpp",
    },
]

# ============================================================================
# CODE EXTRACTION
# ============================================================================

def extract_code_blocks(filepath: Path) -> Tuple[str, str]:
    """
    Extract relevant code from example file.

    Returns: (full_code, api_code)
    """
    content = filepath.read_text()

    # Remove includes and main() wrapper, keep only the API usage
    lines = content.split('\n')

    # Find the API section (between std::cout and geometry export)
    in_api_section = False
    api_lines = []
    for line in lines:
        # Skip includes, main, empty lines at start
        if '#include' in line or 'int main()' in line or line.strip() == '{':
            continue

        # Start capturing at geometry creation
        if 'Geometry2D' in line and 'build_box' in line:
            in_api_section = True

        if in_api_section:
            # Stop at export
            if 'write_legacy_quads' in line or 'vtk::write' in line:
                break
            api_lines.append(line)

    api_code = '\n'.join(api_lines).strip()

    # Clean up the full code (remove comments, collapse)
    full_code = content
    full_code = re.sub(r'/\*\*.*?\*/', '', full_code, flags=re.DOTALL)  # Remove block comments
    full_code = re.sub(r'//.*$', '', full_code, flags=re.MULTILINE)    # Remove line comments

    return full_code, api_code

# ============================================================================
# BUILD
# ============================================================================

def build_examples(config: Config, skip_build: bool = False) -> bool:
    """Build the example executables using CMake."""

    if skip_build:
        print("Skipping build (--skip-build)")
        return True

    print("=" * 60)
    print("Building geometry gallery examples...")
    print("=" * 60)

    # Create build directory if it doesn't exist
    if not config.build_dir.exists():
        print(f"Error: Build directory not found: {config.build_dir}")
        print("Please run: cmake -B build-serial -DCMAKE_BUILD_TYPE=Release")
        return False

    # Run CMake configure for examples
    examples_dir = config.script_dir / "examples"
    cmake_cmd = [
        "cmake",
        "--build", str(config.build_dir),
        "--target",
        "01_box",
        "02_box_with_cylinder",
        "03_multiple_cylinders",
        "04_complex_geometry",
        "--",
        "-j4",  # Parallel build
    ]

    print(f"Running: {' '.join(cmake_cmd)}")
    result = subprocess.run(cmake_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Build failed!")
        print(result.stderr)
        return False

    print("Build succeeded!")
    return True

# ============================================================================
# RUN EXAMPLES
# ============================================================================

def run_example(example: dict, config: Config) -> bool:
    """Run a single example executable."""

    # Try different possible paths for the executable
    possible_paths = [
        config.build_dir / "examples" / "geometry_gallery" / example['name'],
        config.build_dir / example['name'] / example['name'],
        config.build_dir / example['name'],
    ]

    exe_path = None
    for path in possible_paths:
        if path.exists():
            exe_path = path
            break

    if not exe_path:
        print(f"Warning: Executable not found for {example['name']}")
        print(f"  Tried: {[str(p) for p in possible_paths]}")
        return False

    print(f"Running {example['name']}...")

    # Run from output directory
    result = subprocess.run(
        [str(exe_path)],
        cwd=config.output_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return False

    print(f"  {result.stdout.strip()}")
    return True

# ============================================================================
# GENERATE IMAGES
# ============================================================================

def generate_image(example: dict, config: Config) -> bool:
    """Generate PNG image from VTK file using vtk_plot.py."""

    vtk_file = config.output_dir / f"{example['name']}.vtk"
    png_file = config.output_dir / f"{example['name']}.png"

    if not vtk_file.exists():
        print(f"Warning: VTK file not found: {vtk_file}")
        return False

    print(f"Generating image: {png_file.name}")

    # Run vtk_plot.py
    cmd = [
        sys.executable,
        str(config.vtk_plot_path),
        str(vtk_file),
        "--output", str(png_file),
        "--dpi", str(config.dpi),
        "--colormap", config.colormap,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return False

    print(f"  Created: {png_file.name}")
    return True

# ============================================================================
# GENERATE MARKDOWN
# ============================================================================

def generate_example_markdown(example: dict, config: Config) -> str:
    """Generate markdown page for a single example."""

    example_file = config.examples_dir / example['file']
    full_code, api_code = extract_code_blocks(example_file)

    # Count lines
    api_lines = len(api_code.split('\n'))

    md = f"""# {example['title']}

{example['description'].strip()}

## API Usage

```cpp
{api_code}
```

## Result

![{example['title']}](output/{example['name']}.png)

*Figure: {example['title']}*



## Full Example Code

<details>
<summary>Click to expand full source code</summary>

```cpp
{full_code}
```

</details>

---
"""

    return md

def generate_gallery_markdown(config: Config) -> str:
    """Generate the main gallery index markdown."""

    md = """# Geometry Gallery

This gallery demonstrates the **Geometry2D** API for creating computational domains in subsetix.

Each example shows:
- The API code used to create the geometry
- A visual representation of the resulting mesh
- Full source code

---

## Examples

"""

    for example in EXAMPLES:
        md += f"""### [{example['title']}]({example['name']}/README.md)

{example['description'].strip().split('.')[0]}.

[View example →]({example['name']}/README.md)

![{example['name']}](output/{example['name']}.png)

---

"""

    return md

# ============================================================================
# MAIN GENERATION
# ============================================================================

def generate_all(config: Config, skip_build: bool = False) -> int:
    """Main generation function."""

    print("\n" + "=" * 60)
    print("Geometry Gallery Generator")
    print("=" * 60 + "\n")

    # 1. Build
    if not build_examples(config, skip_build):
        return 1

    # 2. Ensure output directory exists
    if not config.output_dir.is_symlink() and not config.output_dir.exists():
        config.output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Run examples and generate images
    print("\n" + "-" * 60)
    print("Running examples and generating images...")
    print("-" * 60)

    for example in EXAMPLES:
        if not run_example(example, config):
            continue
        generate_image(example, config)

    # 4. Generate markdown files
    print("\n" + "-" * 60)
    print("Generating markdown documentation...")
    print("-" * 60)

    # Main gallery README
    gallery_md = generate_gallery_markdown(config)
    (config.script_dir / "README.md").write_text(gallery_md)
    print(f"Created: {config.script_dir / 'README.md'}")

    # Individual example READMEs
    for example in EXAMPLES:
        example_dir = config.script_dir / example['name']
        example_dir.mkdir(exist_ok=True)

        md = generate_example_markdown(example, config)
        (example_dir / "README.md").write_text(md)
        print(f"Created: {example_dir / 'README.md'}")

    # Copy output files to script dir for viewing
    output_link = config.script_dir / "output"
    if output_link.exists():
        if output_link.is_symlink() or output_link.is_dir():
            if output_link.is_symlink():
                output_link.unlink()
            else:
                shutil.rmtree(output_link)

    # Create symlink or copy for easy access
    try:
        # Create relative symlink
        rel_path = os.path.relpath(config.output_dir, config.script_dir)
        output_link.symlink_to(rel_path)
    except (OSError, NotImplementedError):
        # Symlink not supported, copy a few key files instead
        output_link.mkdir(exist_ok=True)
        for vtk_file in config.output_dir.glob("*.vtk"):
            shutil.copy2(vtk_file, output_link / vtk_file.name)
        for png_file in config.output_dir.glob("*.png"):
            shutil.copy2(png_file, output_link / png_file.name)
        print(f"Copied output files to: {output_link}")

    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)
    print(f"\nGallery: {config.script_dir / 'README.md'}")
    print(f"Output:  {config.output_dir}")

    return 0

# ============================================================================
# CLEAN
# ============================================================================

def clean(config: Config):
    """Clean build artifacts and generated files."""

    print("Cleaning generated files...")

    # Remove output directory
    if config.output_dir.exists():
        shutil.rmtree(config.output_dir)
        print(f"Removed: {config.output_dir}")

    # Remove example READMEs
    for example in EXAMPLES:
        example_dir = config.script_dir / example['name']
        if example_dir.exists():
            shutil.rmtree(example_dir)
            print(f"Removed: {example_dir}")

    # Remove gallery README
    gallery_readme = config.script_dir / "README.md"
    if gallery_readme.exists():
        gallery_readme.unlink()
        print(f"Removed: {gallery_readme}")

    # Remove output symlink
    output_link = config.script_dir / "output"
    if output_link.exists():
        if output_link.is_symlink():
            output_link.unlink()
        elif output_link.is_dir():
            shutil.rmtree(output_link)
        print(f"Removed: {output_link}")

    print("Clean complete!")

# ============================================================================
# CLI
# ============================================================================

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate geometry gallery documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generator.py              # Build and generate all
  python generator.py --skip-build  # Just generate docs (assume built)
  python generator.py --clean       # Clean generated files
        """
    )

    parser.add_argument("--skip-build", action="store_true",
                       help="Skip compilation, just generate documentation")
    parser.add_argument("--clean", action="store_true",
                       help="Clean generated files")

    return parser.parse_args()

def main():
    args = parse_args()
    config = Config()

    if args.clean:
        clean(config)
        return 0

    return generate_all(config, skip_build=args.skip_build)

if __name__ == "__main__":
    sys.exit(main())
