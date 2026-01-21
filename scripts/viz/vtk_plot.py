#!/usr/bin/env python3
"""
vtk_plot.py - Minimalist VTK visualization tool for subsetix

Features:
- Read VTK legacy binary files (custom parser, no meshio required)
- Generate 2D (tripcolor) and 3D (scatter) plots
- Export to PNG or MP4 (animation)
- Parallel batch processing
- CLI configuration

Usage:
    python vtk_plot.py input.vtk                    # Basic usage
    python vtk_plot.py *.vtk --jobs 8               # Parallel batch
    python vtk_plot.py step_*.vtk --animate         # Create MP4
    python vtk_plot.py input.vtk --field density --colormap viridis
"""

import argparse
import sys
import os
from pathlib import Path
from functools import lru_cache
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field

# Headless mode - must be before matplotlib imports
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import numpy as np

# Optional: numpy is required (usually available with matplotlib)
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PlotConfig:
    """Configuration for plotting"""
    # Data selection
    field: Optional[str] = None  # None = auto-detect first scalar
    mesh: bool = True            # Show mesh edges

    # Visual style
    colormap: str = 'viridis'
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    cbar: bool = True            # Show colorbar

    # Dimension
    dim: Optional[int] = None    # None = auto (2D for 2D mesh, 3D for 3D)

    # Output
    dpi: int = 150
    output_dir: str = 'viz_output/png'

    # Performance
    cache_size: int = 100
    jobs: int = 4

    # Animation
    fps: int = 30
    animate: bool = False

    # Figure size
    figsize: Tuple[float, float] = (10, 8)


# ============================================================================
# VTK READER WITH CACHE
# ============================================================================

class VTKReader:
    """Read VTK legacy binary files with caching"""

    def __init__(self, cache_size: int = 100):
        self.cache_size = cache_size

    @lru_cache(maxsize=100)
    def read(self, filepath: str) -> Dict[str, Any]:
        """
        Read VTK legacy binary file and return structured data

        Returns dict with:
            - points: (N, 3) array of vertex coordinates
            - cells: list of cell connectivity (for quads)
            - cell_data: dict of cell-centered scalar fields
            - point_data: dict of point-centered scalar fields
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"VTK file not found: {filepath}")

        with open(filepath, 'rb') as f:
            # Read header
            header = f.readline().decode('ascii').strip()
            if not header.startswith('# vtk DataFile'):
                raise ValueError(f"Not a VTK file: {filepath}")

            # Skip title line
            f.readline()

            # Read format (should be BINARY)
            format_str = f.readline().decode('ascii').strip()
            is_binary = (format_str == 'BINARY')

            # Read dataset type
            dataset_line = f.readline().decode('ascii').strip()

            result = {
                'points': None,
                'cells': [],
                'cell_data': {},
                'point_data': {},
            }

            # Parse sections
            while True:
                line = f.readline()
                if not line:
                    break

                line_str = line.decode('ascii').strip()

                # POINTS section
                if line_str.startswith('POINTS '):
                    parts = line_str.split()
                    n_points = int(parts[1])
                    data_type = parts[2]

                    if is_binary:
                        # Read binary points (big-endian float32)
                        n_floats = n_points * 3
                        data = f.read(n_floats * 4)
                        points = np.frombuffer(data, dtype='>f4').astype(np.float64)
                        points = points.reshape((n_points, 3))
                        result['points'] = points
                    else:
                        # ASCII points
                        points = []
                        for _ in range(n_points):
                            vals = f.readline().decode('ascii').strip().split()
                            points.append([float(v) for v in vals])
                        result['points'] = np.array(points)

                # CELLS section
                elif line_str.startswith('CELLS '):
                    parts = line_str.split()
                    n_cells = int(parts[1])
                    size = int(parts[2])

                    if is_binary:
                        # Read binary cell connectivity (big-endian int32)
                        data = f.read(size * 4)
                        cell_data = np.frombuffer(data, dtype='>i4')

                        # Parse into cells
                        idx = 0
                        cells = []
                        for _ in range(n_cells):
                            n_verts = int(cell_data[idx])
                            idx += 1
                            verts = [int(cell_data[idx + i]) for i in range(n_verts)]
                            idx += n_verts
                            cells.append(('quad', verts))
                        result['cells'] = cells
                    else:
                        # ASCII cells
                        cells = []
                        for _ in range(n_cells):
                            vals = f.readline().decode('ascii').strip().split()
                            n_verts = int(vals[0])
                            verts = [int(v) for v in vals[1:]]
                            cells.append(('quad', verts))
                        result['cells'] = cells

                # CELL_TYPES section
                elif line_str.startswith('CELL_TYPES '):
                    parts = line_str.split()
                    n_cells = int(parts[1])

                    if is_binary:
                        f.read(n_cells * 4)  # Skip cell types
                    else:
                        for _ in range(n_cells):
                            f.readline()

                # CELL_DATA section
                elif line_str.startswith('CELL_DATA '):
                    parts = line_str.split()
                    n_cells = int(parts[1])

                    # Read SCALARS subsection
                    while True:
                        scalar_line = f.readline().decode('ascii').strip()
                        if not scalar_line:
                            break

                        if scalar_line.startswith('SCALARS '):
                            parts = scalar_line.split()
                            field_name = parts[1]
                            data_type = parts[2]

                            # Skip LOOKUP_TABLE line
                            f.readline()

                            # Read scalar values
                            if is_binary:
                                data = f.read(n_cells * 4)
                                values = np.frombuffer(data, dtype='>f4').astype(np.float64)
                            else:
                                values = []
                                for _ in range(n_cells):
                                    val = float(f.readline().decode('ascii').strip())
                                    values.append(val)
                                values = np.array(values)

                            result['cell_data'][field_name] = values
                        elif scalar_line.startswith('VECTORS ') or scalar_line.startswith('NORMALS '):
                            # Skip lookup table line and data
                            f.readline()
                            f.read(n_cells * 3 * 4) if is_binary else [f.readline() for _ in range(n_cells)]
                        else:
                            break

                # POINT_DATA section
                elif line_str.startswith('POINT_DATA '):
                    parts = line_str.split()
                    n_points = int(parts[1])

                    # Read SCALARS subsection
                    while True:
                        scalar_line = f.readline()
                        if not scalar_line:
                            break
                        scalar_line = scalar_line.decode('ascii').strip()

                        if scalar_line.startswith('SCALARS '):
                            parts = scalar_line.split()
                            field_name = parts[1]
                            data_type = parts[2]

                            # Skip LOOKUP_TABLE line
                            f.readline()

                            # Read scalar values
                            if is_binary:
                                data = f.read(n_points * 4)
                                values = np.frombuffer(data, dtype='>f4').astype(np.float64)
                            else:
                                values = []
                                for _ in range(n_points):
                                    val = float(f.readline().decode('ascii').strip())
                                    values.append(val)
                                values = np.array(values)

                            result['point_data'][field_name] = values
                        else:
                            break

        return result

    def get_scalar_fields(self, filepath: str) -> List[str]:
        """Get list of available scalar field names"""
        data = self.read(filepath)
        fields = []

        # Check cell data
        for key in data['cell_data'].keys():
            fields.append(key)

        # Check point data
        for key in data['point_data'].keys():
            fields.append(f"point_{key}")

        return fields

    def get_field_data(self, filepath: str, field_name: str) -> np.ndarray:
        """Extract scalar field values"""
        data = self.read(filepath)

        # Try cell data first
        if field_name in data['cell_data']:
            return data['cell_data'][field_name]

        # Try point data
        if field_name in data['point_data']:
            return data['point_data'][field_name]

        # Try with point_ prefix
        if field_name.startswith('point_'):
            base_name = field_name[6:]
            if base_name in data['point_data']:
                return data['point_data'][base_name]

        raise ValueError(f"Field '{field_name}' not found in {filepath}. Available: {list(data['cell_data'].keys())}, {[f'point_{k}' for k in data['point_data'].keys()]}")

    def detect_dimension(self, filepath: str) -> int:
        """Detect if mesh is 2D or 3D based on points"""
        data = self.read(filepath)
        points = data['points']

        if points is None or len(points) == 0:
            return 2

        # Check z-coordinate variation
        z_range = points[:, 2].max() - points[:, 2].min()
        return 2 if z_range < 1e-10 else 3


# ============================================================================
# PLOTTER
# ============================================================================

class VTKPlotter:
    """Plot VTK data with matplotlib"""

    def __init__(self, config: PlotConfig):
        self.config = config
        self.reader = VTKReader(cache_size=config.cache_size)

    def _extract_quad_mesh(self, vtk_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract quad mesh data for plotting

        For VTK quad cells:
        - Each quad has 4 vertices
        - We need to build (x, y) grid and values

        Returns:
            x, y: Arrays of cell center coordinates
            values: Array of scalar values
        """
        points = vtk_data['points']
        cells = vtk_data['cells']

        if not cells:
            raise ValueError("No cells found in VTK file")

        # Cells are stored as list of tuples: [('quad', [verts]), ...]
        # or potentially other formats
        n_cells = len(cells)

        # Get cell centers
        cell_centers = np.zeros((n_cells, 2))
        for i, cell in enumerate(cells):
            if isinstance(cell, tuple) and len(cell) == 2:
                # Format: ('quad', [v0, v1, v2, v3])
                verts = cell[1]
            elif isinstance(cell, list) or isinstance(cell, np.ndarray):
                verts = cell
            else:
                continue

            # Get vertices and compute center
            vertices = points[verts]
            cell_centers[i] = vertices[:, :2].mean(axis=0)

        # Return cell center coordinates
        x = cell_centers[:, 0]
        y = cell_centers[:, 1]

        return x, y

    def plot_2d(self, vtk_file: str, field_name: str, output_path: str) -> None:
        """Create 2D plot using scatter for unstructured quad mesh"""

        vtk_data = self.reader.read(vtk_file)
        values = self.reader.get_field_data(vtk_file, field_name)

        x, y = self._extract_quad_mesh(vtk_data)

        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)

        # Normalize values
        if self.config.vmin is None:
            vmin = values.min()
        else:
            vmin = self.config.vmin
        if self.config.vmax is None:
            vmax = values.max()
        else:
            vmax = self.config.vmax
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Plot as scatter for unstructured sparse meshes
        # This is the most robust approach for CSR sparse geometry
        scatter = ax.scatter(x, y, c=values,
                           cmap=self.config.colormap,
                           norm=norm,
                           s=10,  # marker size
                           alpha=0.8,
                           edgecolors='none')

        # Add mesh overlay (optional)
        if self.config.mesh:
            # Don't draw mesh edges for sparse CSR - too many cells
            pass

        # Colorbar
        if self.config.cbar:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(field_name, rotation=270, labelpad=15)

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.set_title(f'{Path(vtk_file).stem} - {field_name}')

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)

    def plot_3d(self, vtk_file: str, field_name: str, output_path: str) -> None:
        """Create 3D surface plot"""

        vtk_data = self.reader.read(vtk_file)
        values = self.reader.get_field_data(vtk_file, field_name)

        x, y = self._extract_quad_mesh(vtk_data)

        # For 3D visualization of 2D data, extrude in Z direction
        # Use the scalar value as Z coordinate
        z = values

        # Create figure
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection='3d')

        # Normalize for colormap
        if self.config.vmin is None:
            vmin = values.min()
        else:
            vmin = self.config.vmin
        if self.config.vmax is None:
            vmax = values.max()
        else:
            vmax = self.config.vmax
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Plot surface (using scatter for unstructured data)
        scatter = ax.scatter(x, y, z,
                           c=values,
                           cmap=self.config.colormap,
                           norm=norm,
                           s=1,
                           alpha=0.6)

        # Colorbar
        if self.config.cbar:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(field_name, rotation=270, labelpad=15)

        # Labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel(field_name)
        ax.set_title(f'{Path(vtk_file).stem} - {field_name} (3D)')

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)


# ============================================================================
# ANIMATION
# ============================================================================

class VTKAnimator:
    """Create MP4 animation from multiple VTK files"""

    def __init__(self, config: PlotConfig):
        self.config = config
        self.plotter = VTKPlotter(config)

    def create_animation(self, vtk_files: List[str], field_name: str, output_path: str) -> None:
        """Create MP4 animation from VTK files"""

        vtk_files = sorted(vtk_files)

        # Setup figure
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)

        # Read first file to setup plot
        reader = VTKReader(cache_size=self.config.cache_size)
        vtk_data = reader.read(vtk_files[0])
        values = reader.get_field_data(vtk_files[0], field_name)
        x, y = VTKPlotter(self.config)._extract_quad_mesh(vtk_data)

        # Auto-detect limits
        if self.config.vmin is None:
            vmin = np.min([reader.get_field_data(f, field_name).min() for f in vtk_files])
        else:
            vmin = self.config.vmin
        if self.config.vmax is None:
            vmax = np.max([reader.get_field_data(f, field_name).max() for f in vtk_files])
        else:
            vmax = self.config.vmax
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Initial plot - use scatter (not tripcolor) to preserve holes
        scatter = ax.scatter(x, y, c=values,
                           cmap=self.config.colormap,
                           norm=norm,
                           s=10,
                           alpha=0.8,
                           edgecolors='none')

        if self.config.cbar:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(field_name, rotation=270, labelpad=15)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        title = ax.set_title('')

        def update(frame_idx):
            # Update data
            vtk_data = reader.read(vtk_files[frame_idx])
            values = reader.get_field_data(vtk_files[frame_idx], field_name)

            # Update scatter plot colors
            scatter.set_array(values)
            title.set_text(f'{Path(vtk_files[frame_idx]).stem} - {field_name}')

            return scatter, title

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(vtk_files),
                                      interval=1000/self.config.fps, blit=False)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use ffmpeg for MP4
        try:
            ani.save(output_path, writer='ffmpeg', fps=self.config.fps, dpi=self.config.dpi)
        except Exception as e:
            # Fallback to pillow
            print(f"Warning: ffmpeg not available, using pillow: {e}")
            ani.save(output_path.with_suffix('.gif'), writer='pillow', fps=self.config.fps)

        plt.close(fig)


# ============================================================================
# PARALLEL BATCH PROCESSING
# ============================================================================

def process_single_vtk(args: Tuple[str, PlotConfig, str, int]) -> str:
    """Process a single VTK file - for parallel execution"""
    vtk_file, config, field_name, dim = args

    plotter = VTKPlotter(config)

    # Auto-detect field if not specified
    if field_name is None:
        fields = plotter.reader.get_scalar_fields(vtk_file)
        if not fields:
            raise ValueError(f"No scalar fields found in {vtk_file}")
        field_name = fields[0]

    # Generate output filename
    input_path = Path(vtk_file)
    if config.output_dir:
        output_dir = Path(config.output_dir)
    else:
        output_dir = input_path.parent / 'viz_output'

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{input_path.stem}_{field_name}.png"

    # Plot
    if dim == 2:
        plotter.plot_2d(vtk_file, field_name, str(output_file))
    else:
        plotter.plot_3d(vtk_file, field_name, str(output_file))

    return str(output_file)


def batch_process(vtk_files: List[str], config: PlotConfig) -> List[str]:
    """Process multiple VTK files in parallel"""

    if config.jobs <= 1:
        # Sequential processing
        results = []
        for vtk_file in vtk_files:
            # Auto-detect field and dimension
            plotter = VTKPlotter(config)
            field_name = config.field
            dim = config.dim

            if field_name is None:
                fields = plotter.reader.get_scalar_fields(vtk_file)
                if fields:
                    field_name = fields[0]
                else:
                    print(f"Warning: No scalar fields in {vtk_file}, skipping")
                    continue

            if dim is None:
                dim = plotter.reader.detect_dimension(vtk_file)

            output_file = process_single_vtk((vtk_file, config, field_name, dim))
            results.append(output_file)
            print(f"  Generated: {output_file}")

        return results

    else:
        # Parallel processing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Prepare arguments
        args_list = []
        for vtk_file in vtk_files:
            # Auto-detect for each file
            plotter = VTKPlotter(config)
            field_name = config.field
            dim = config.dim

            if field_name is None:
                fields = plotter.reader.get_scalar_fields(vtk_file)
                if fields:
                    field_name = fields[0]
                else:
                    continue

            if dim is None:
                dim = plotter.reader.detect_dimension(vtk_file)

            args_list.append((vtk_file, config, field_name, dim))

        # Process in parallel
        results = []
        with ProcessPoolExecutor(max_workers=config.jobs) as executor:
            futures = {executor.submit(process_single_vtk, args): args[0]
                      for args in args_list}

            for future in as_completed(futures):
                vtk_file = futures[future]
                try:
                    output_file = future.result()
                    results.append(output_file)
                    print(f"  Generated: {output_file}")
                except Exception as e:
                    print(f"  Error processing {vtk_file}: {e}", file=sys.stderr)

        return results


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(
        description='Visualize VTK files - Generate PNG or MP4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.vtk                                # Basic usage
  %(prog)s *.vtk --jobs 8                           # Parallel batch
  %(prog)s step_*.vtk --animate                     # Create MP4
  %(prog)s input.vtk --field density --colormap jet --no-mesh
        """
    )

    # Input files
    parser.add_argument('input', nargs='+', help='VTK file(s) to process')

    # Data selection
    parser.add_argument('--field', '-f', type=str, default=None,
                       help='Scalar field name (default: auto-detect first)')
    parser.add_argument('--mesh', action='store_true', default=True,
                       help='Show mesh edges (default: True)')
    parser.add_argument('--no-mesh', dest='mesh', action='store_false',
                       help='Hide mesh edges')

    # Visual style
    parser.add_argument('--colormap', '-c', type=str, default='viridis',
                       choices=['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                               'rainbow', 'jet', 'coolwarm', 'RdYlBu', 'RdYlGn'],
                       help='Matplotlib colormap (default: viridis)')
    parser.add_argument('--vmin', type=float, default=None,
                       help='Minimum value for colormap (default: auto)')
    parser.add_argument('--vmax', type=float, default=None,
                       help='Maximum value for colormap (default: auto)')
    parser.add_argument('--no-cbar', dest='cbar', action='store_false',
                       help='Hide colorbar')

    # Dimension
    parser.add_argument('--2d', dest='dim', action='store_const', const=2,
                       help='Force 2D plot')
    parser.add_argument('--3d', dest='dim', action='store_const', const=3,
                       help='Force 3D plot')

    # Output
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file (for single input) or directory')
    parser.add_argument('--output-dir', type=str, default='viz_output/png',
                       help='Output directory for batch processing (default: viz_output/png)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Output resolution in DPI (default: 150)')

    # Performance
    parser.add_argument('--jobs', '-j', type=int, default=4,
                       help='Number of parallel jobs (default: 4)')
    parser.add_argument('--cache', type=int, default=100,
                       help='VTK cache size (default: 100)')

    # Animation
    parser.add_argument('--animate', action='store_true',
                       help='Create MP4 animation from multiple files')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for animation (default: 30)')

    # Figure size
    parser.add_argument('--figsize', type=str, default='10,8',
                       help='Figure size as W,H (default: 10,8)')

    return parser.parse_args()


def main():
    """Main entry point"""

    args = parse_args()

    # Parse figure size
    try:
        figsize = tuple(float(x) for x in args.figsize.split(','))
        if len(figsize) != 2:
            raise ValueError
    except:
        print("Error: --figsize must be two comma-separated numbers, e.g., '10,8'", file=sys.stderr)
        sys.exit(1)

    # Create config
    config = PlotConfig(
        field=args.field,
        mesh=args.mesh,
        colormap=args.colormap,
        vmin=args.vmin,
        vmax=args.vmax,
        cbar=args.cbar,
        dim=args.dim,
        dpi=args.dpi,
        output_dir=args.output_dir,
        cache_size=args.cache,
        jobs=args.jobs,
        fps=args.fps,
        animate=args.animate,
        figsize=figsize,
    )

    # Get input files
    vtk_files = []
    for inp in args.input:
        path = Path(inp)
        if path.exists():
            vtk_files.append(str(path))
        else:
            print(f"Warning: File not found: {inp}", file=sys.stderr)

    if not vtk_files:
        print("Error: No valid VTK files found", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(vtk_files)} VTK file(s)...")

    # Animation mode
    if args.animate:
        if len(vtk_files) < 2:
            print("Error: Animation requires at least 2 VTK files", file=sys.stderr)
            sys.exit(1)

        # Auto-detect field
        plotter = VTKPlotter(config)
        field_name = config.field
        if field_name is None:
            fields = plotter.reader.get_scalar_fields(vtk_files[0])
            if not fields:
                print("Error: No scalar fields found for animation", file=sys.stderr)
                sys.exit(1)
            field_name = fields[0]

        # Generate output path
        if args.output:
            output_path = args.output
        else:
            output_path = Path(config.output_dir) / f"{Path(vtk_files[0]).stem}_{field_name}.mp4"

        print(f"Creating animation: {output_path}")
        animator = VTKAnimator(config)
        animator.create_animation(vtk_files, field_name, str(output_path))
        print(f"  Saved: {output_path}")

    # Single file mode
    elif len(vtk_files) == 1:
        vtk_file = vtk_files[0]

        # Auto-detect field and dimension
        plotter = VTKPlotter(config)
        field_name = config.field
        dim = config.dim

        if field_name is None:
            fields = plotter.reader.get_scalar_fields(vtk_file)
            if not fields:
                print("Error: No scalar fields found", file=sys.stderr)
                sys.exit(1)
            field_name = fields[0]
            print(f"  Auto-detected field: {field_name}")

        if dim is None:
            dim = plotter.reader.detect_dimension(vtk_file)
            print(f"  Auto-detected dimension: {dim}D")

        # Generate output path
        if args.output:
            output_path = args.output
        else:
            output_path = Path(config.output_dir) / f"{Path(vtk_file).stem}_{field_name}.png"

        # Plot
        if dim == 2:
            plotter.plot_2d(vtk_file, field_name, str(output_path))
        else:
            plotter.plot_3d(vtk_file, field_name, str(output_path))

        print(f"  Saved: {output_path}")

    # Batch mode
    else:
        print(f"Using {config.jobs} parallel job(s)...")
        results = batch_process(vtk_files, config)
        print(f"\nGenerated {len(results)} image(s)")


if __name__ == '__main__':
    main()
