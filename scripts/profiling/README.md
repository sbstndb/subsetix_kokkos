# Profiling Scripts

This directory contains scripts for profiling the experimental benchmarks using NVIDIA profiling tools.

## Prerequisites

1. **CUDA Toolkit**: Ensure profiling tools are installed
   ```bash
   # Nsight Compute (recommended for GPU kernel analysis)
   find /usr -name "ncu" 2>/dev/null  # May be in /usr/local/cuda-12.8/bin/ncu

   # Nsight Systems (for timeline analysis)
   which nsys  # Should show /usr/local/bin/nsys or similar
   ```

2. **GPU Access**: You need access to an NVIDIA GPU

3. **Build**: Use the profiling presets for best results

## Quick Start

### Nsight Compute (Recommended for GPU Kernel Analysis)

```bash
# Profile 3D LargeConfig with detailed GPU metrics
./run_ncu.sh --benchmark "3D_LargeConfig"

# Quick profiling with basic metrics
./run_ncu.sh --benchmark "SmallConfig" --section-set basic
```

### Nsight Systems (Timeline Analysis)

```bash
# Quick profiling on 3D SmallConfig (fast)
./run_nsys_quick.sh

# Profile specific benchmark
./run_nsys.sh --benchmark "3D_LargeConfig"

# View results
nsys-ui profiling_output/*.nsys-rep
```

### Quick profiling (SmallConfig, 3D)
```bash
./run_nvprof_quick.sh
```

### Profile specific benchmark (e.g., 3D LargeConfig)
```bash
./run_nvprof.sh --benchmark "3D_LargeConfig"
```

## Scripts

### Nsight Compute Scripts (Recommended for GPU Kernel Analysis)

#### `run_ncu.sh` - Main Nsight Compute profiling script

Deep GPU kernel profiling and analysis with Nsight Compute.

**Usage:**
```bash
./run_ncu.sh [OPTIONS]
```

**Options:**
- `--preset PRESET` - CMake preset (default: `profiling-cuda-gcc12`)
- `--benchmark FILTER` - Benchmark filter (default: `3D_LargeConfig`)
- `--output-dir DIR` - Output directory (default: `profiling_output_ncu`)
- `--section-set SET` - Section set (basic, full, or custom)
- `--ncu-opts OPTS` - Extra ncu options
- `--build-only` - Only build, don't run profiling
- `--kernel-only` - Skip build, run profiling only
- `--help` - Show help message

**Examples:**
```bash
# Profile 3D LargeConfig
./run_ncu.sh --benchmark "3D_LargeConfig"

# Profile with detailed metrics
./run_ncu.sh --benchmark "LargeConfig" --section-set full

# Profile specific version
./run_ncu.sh --benchmark "V2_3D_MediumConfig"
```

### Nsight Systems Scripts (Timeline Analysis)

#### `run_nsys.sh` - Main Nsight Systems profiling script

Flexible script for profiling specific benchmarks with Nsight Systems.

**Usage:**
```bash
./run_nsys.sh [OPTIONS]
```

**Options:**
- `--preset PRESET` - CMake preset (default: `profiling-cuda-gcc12`)
- `--benchmark FILTER` - Benchmark filter (default: `LargeConfig`)
- `--output-dir DIR` - Output directory (default: `profiling_output`)
- `--nsys-opts OPTS` - Extra nsys options
- `--build-only` - Only build, don't run profiling
- `--trace-only` - Skip build, run profiling only
- `--help` - Show help message

**Examples:**
```bash
# Profile 3D LargeConfig
./run_nsys.sh --benchmark "3D_LargeConfig"

# Profile with CPU sampling
./run_nsys.sh --benchmark "LargeConfig" --nsys-opts "--sample=cpu"

# Profile specific version
./run_nsys.sh --benchmark "V2_3D_MediumConfig"
```

#### `run_nsys_quick.sh` - Quick Nsight Systems profiling

Fast profiling for development iterations using SmallConfig.

**Usage:**
```bash
./run_nsys_quick.sh [OPTIONS]
```

**Options:**
- `--dimension 2D|3D|both` - Which dimension to profile (default: `3D`)
- `--version V1|V2|V3|all` - Which version to profile (default: `all`)
- `--output-dir DIR` - Output directory (default: `profiling_output_quick`)
- `--skip-build` - Skip the build step
- `--detailed` - Use detailed profiling options

**Examples:**
```bash
# Quick 3D profiling, all versions
./run_nsys_quick.sh

# Quick 2D profiling, v2 only
./run_nsys_quick.sh --dimension 2D --version V2
```

### `analyze_results.sh` - Analyze profiling results

Extract and summarize key information from profiling results.

**Usage:**
```bash
./analyze_results.sh [PROFILING_DIR]
```

**Example:**
```bash
# Analyze most recent results
./analyze_results.sh

# Analyze specific directory
./analyze_results.sh profiling_output/3d_large
```

## CMake Presets

The project includes profiling-specific presets:

- **`profiling-cuda-gcc12`**: RelWithDebInfo build with profiling enabled (default)
- **`profiling-cuda-gcc12-release`**: Release build with debug symbols for profiling

**To use:**
```bash
cmake --preset profiling-cuda-gcc12
cmake --build --preset profiling-cuda-gcc12
```

## Benchmark Filters

The benchmark executables support filtering with `--benchmark_filter`:

| Pattern | Matches |
|---------|---------|
| `SmallConfig` | All SmallConfig benchmarks |
| `MediumConfig` | All MediumConfig benchmarks |
| `LargeConfig` | All LargeConfig benchmarks |
| `3D` | All 3D benchmarks |
| `2D` | All 2D benchmarks |
| `V1_3D_LargeConfig` | V1 3D LargeConfig only |
| `V2_SmallConfig` | V2 SmallConfig only |
| `3D_LargeConfig` | All 3D LargeConfig benchmarks |

## Viewing Results

### Using Nsight Compute (Recommended for GPU Analysis)

```bash
# Find ncu location (may vary)
NCU_BIN=/usr/local/cuda-12.8/bin/ncu

# View detailed results
$NCU_BIN --import profiling_output_ncu/*.ncu-rep --page=details

# View specific pages
$NCU_BIN --import profiling_output_ncu/*.ncu-rep --page=raw      # Raw metrics
$NCU_BIN --import profiling_output_ncu/*.ncu-rep --page=source   # Source code
$NCU_BIN --import profiling_output_ncu/*.ncu-rep --page=session  # Session info
```

### Using Nsight Systems GUI (Timeline Analysis)
```bash
# Open the GUI with your trace file
nsys-ui profiling_output/*.nsys-rep

# Or on some systems
nvsight-sys profiling_output/*.nsys-rep
```

### Using Nsight Systems CLI
```bash
# View statistics summary
nsys stats profiling_output/*.nsys-rep

# Export to CSV
nsys stats profiling_output/*.nsys-rep --format csv --output stats.csv

# View GPU memory info
nsys stats profiling_output/*.nsys-rep --report gpumemtimesum
```

### Using the analyze script
```bash
./analyze_results.sh profiling_output
```

## Common Profiling Options

### ncu Options

| Option | Description |
|--------|-------------|
| `--set basic` | Basic metrics (fast) |
| `--set full` | All available metrics (detailed, slow) |
| `-k <kernel>` | Profile specific kernel |
| `-c <count>` | Limit number of kernel launches |
| `-s <skip>` | Skip first N kernel launches |

## Output Files

Profiling generates several files depending on the tool used:

**ncu:**
- `*.ncu-rep` - Binary profile data for Nsight Compute
- `*_stdout.log` - Benchmark stdout

**nsys:**
- `*.nsys-rep` - Binary trace data for Nsight Systems
- `*.sqlite` - Exported SQLite database
- `*_stdout.log` - Benchmark stdout

## Troubleshooting

### "ncu not found"
The script searches multiple locations. Ensure ncu is installed:
```bash
# Search for ncu
find /usr -name "ncu" 2>/dev/null

# Add to PATH if found
export PATH=/usr/local/cuda-12.8/bin:$PATH
```

### "nsys not found"
Install Nsight Systems from:
https://developer.nvidia.com/tools-overview

Or add to PATH:
```bash
export PATH=/opt/nvidia/nsight-systems/2024_6/bin:$PATH
```

### "No CUDA-capable device detected"
Ensure:
1. NVIDIA GPU is installed
2. NVIDIA drivers are loaded
3. CUDA toolkit is properly installed

### Build errors
Make sure you're using a profiling preset:
```bash
cmake --preset profiling-cuda-gcc12
cmake --build --preset profiling-cuda-gcc12
```

### Profiling overhead
For initial testing, use SmallConfig or `--section-set basic` with ncu to reduce overhead.

## Tips

1. **Start with ncu for GPU analysis** - Use `run_ncu.sh --section-set basic` for fast iterations
2. **Use specific filters** - Target specific benchmarks to reduce profiling time
3. **Check GPU temperature** - Long profiling runs can heat up the GPU
4. **Use --kernel-only** - Once built, skip rebuild for subsequent runs
5. **Analyze with ncu CLI** - Use `ncu --import ... --page=details` for detailed analysis
