# Performance Profiling Guide

This guide explains the principles and commands for profiling the Subsetix Kokkos library using the Linux `perf` tool.

## Principles of Performance Profiling

### Why Profile?

Performance profiling helps identify:
- **Hot paths**: Code sections consuming the most CPU time
- **Memory bottlenecks**: Cache misses, memory bandwidth limitations
- **Branch prediction issues**: Mispredicted branches causing pipeline stalls
- **Inefficient algorithms**: Functions that could be optimized
- **Parallelization overhead**: Thread synchronization costs in OpenMP/CUDA

### Profiling Methodology

1. **Baseline first**: Profile before making changes to establish a baseline
2. **Profile representative workloads**: Use realistic data sizes and patterns
3. **Focus on hot functions**: 80% of execution time is typically in 20% of code
4. **Iterate**: Profile → Optimize → Verify → Profile again
5. **Compare multiple backends**: Serial vs OpenMP vs CUDA may have different bottlenecks

### Understanding perf Output

Key concepts:
- **Samples**: perf periodically interrupts the CPU to record what's running
- **Overhead**: The cost of profiling (dwarf call graph has higher overhead)
- **User vs Kernel time**: Distinguish between application and system time
- **Symbols**: Function names from debug information

## Build Configuration for Profiling

### CMake Presets

The profiling presets automatically:
- Add debug symbols (`-g`) for symbol resolution
- Use `RelWithDebInfo` build type (optimization with debug info)
- Enable `SUBSETIX_ENABLE_PERF` flag

```bash
# Serial backend
cmake --preset experimental-perf-serial
cmake --build --preset experimental-perf-serial

# OpenMP backend
cmake --preset experimental-perf-openmp
cmake --build --preset experimental-perf-openmp
```

### Verify Binary Has Debug Symbols

```bash
file ./build-experimental-perf-serial/experimental/tests/experimental_v1_unitary_test
# Should show: "with debug_info, not stripped"

readelf -S ./build-experimental-perf-serial/experimental/tests/experimental_v1_unitary_test | grep debug
# Should show .debug_info, .debug_line, etc.
```

## Convenience Scripts

The `scripts/` directory provides several scripts to simplify profiling workflows:

### Script Overview

| Script | Purpose |
|--------|---------|
| `perf_profile.sh` | Generic profiling for any executable |
| `profile_benchmark.sh` | Profile specific benchmark configurations |
| `profile_all_benchmarks.sh` | Profile all benchmark configurations |
| `compare_perf.sh` | Compare two perf data files |
| `generate_perf_report.sh` | Generate consolidated report from multiple files |

### Profile a Specific Benchmark

```bash
# Profile Small 2D benchmarks with serial backend
./scripts/profile_benchmark.sh experimental-perf-serial Small 2D

# Profile Medium 2D benchmarks with OpenMP
./scripts/profile_benchmark.sh experimental-perf-openmp Medium 2D

# Profile Large 3D benchmarks with custom output directory
./scripts/profile_benchmark.sh experimental-perf-serial Large 3D -o ./my_perf

# Use perf stat for real-time statistics
./scripts/profile_benchmark.sh experimental-perf-serial Small 2D --stat
```

### Profile All Benchmarks

```bash
# Profile all benchmarks (Small/Medium/Large × 2D/3D = 6 combinations)
./scripts/profile_all_benchmarks.sh experimental-perf-serial

# Profile all with OpenMP
./scripts/profile_all_benchmarks.sh experimental-perf-openmp -o ./perf_all
```

### Compare Performance Results

```bash
# Compare before/after optimization
./scripts/compare_perf.sh perf_before.data perf_after.data

# Compare different algorithm versions
./scripts/compare_perf.sh perf_v1.data perf_v2.data -o ./comparison
```

### Generate Consolidated Report

```bash
# Generate summary report from all perf data in a directory
./scripts/generate_perf_report.sh ./perf_output

# Generate report with custom output
./scripts/generate_perf_report.sh ./perf_output -o ./my_report
```

## Perf Commands

### Basic Commands

#### Record performance data

```bash
# Basic recording with call graph
perf record --call-graph dwarf <executable>

# Output to specific file
perf record --call-graph dwarf -o my_perf.data <executable>
```

#### View recorded data

```bash
# Interactive report
perf report

# From specific file
perf report -i my_perf.data
```

#### Real-time statistics

```bash
# Basic statistics
perf stat <executable>

# Specific events
perf stat -e cycles,instructions,cache-misses <executable>

# More detailed
perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses <executable>
```

### Advanced Commands

#### Profile specific events

```bash
# CPU cycles and instructions
perf record -e cycles,instructions --call-graph dwarf <executable>

# Cache performance
perf record -e cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses --call-graph dwarf <executable>

# Memory operations
perf record -e mem-loads,mem-stores --call-graph dwarf <executable>
```

#### Filter by function

```bash
# Only record specific function
perf record --filter='filter_function*' --call-graph dwarf <executable>

# Exclude kernel
perf record --exclude-perf --call-graph dwarf <executable>
```

#### Annotate source code

```bash
# Interactive annotation
perf annotate

# Annotate specific function
perf annotate <function_name>

# From specific file
perf annotate -i my_perf.data
```

#### Script output for further analysis

```bash
# Generate script output
perf script > perf_script.txt

# From specific file
perf script -i my_perf.data > perf_script.txt
```

## Profiling Workflows

### Workflow 1: Identify Hot Functions

```bash
# 1. Record with call graph
perf record --call-graph dwarf ./build-experimental-perf-serial/experimental/tests/experimental_v1_unitary_test

# 2. View report sorted by overhead
perf report --sort=overhead --percent-limit=1

# 3. Focus on top functions
# In perf report:
#   - Use Enter to expand
#   - Use + to zoom into function
#   - Use - to zoom out
```

### Workflow 2: Compare Before/After Optimization

```bash
# Before optimization
perf record --call-graph dwarf -o before.data ./build-experimental-perf-serial/experimental/tests/experimental_v1_unitary_test
perf report -i before.data > before_report.txt

# After optimization
perf record --call-graph dwarf -o after.data ./build-experimental-perf-serial/experimental/tests/experimental_v1_unitary_test
perf report -i after.data > after_report.txt

# Compare
diff before_report.txt after_report.txt
```

### Workflow 3: Generate Flame Graph

```bash
# Install FlameGraph tools (one time)
git clone https://github.com/brendangregg/FlameGraph.git

# Generate flamegraph
perf script | ./FlameGraph/stackcollapse-perf.pl | ./FlameGraph/flamegraph.pl > flamegraph.svg

# View in browser
firefox flamegraph.svg  # or your preferred browser
```

### Workflow 4: Profile Benchmarks

Benchmark naming convention:
- **2D**: `V1_SmallConfig`, `V2_SmallConfig`, `V3_SmallConfig`, etc.
- **3D**: `V1_3D_SmallConfig`, `V2_3D_SmallConfig`, `V3_3D_SmallConfig`, etc.

```bash
# Profile specific 2D benchmark size (matches V1_SmallConfig, V2_SmallConfig, V3_SmallConfig)
perf record --call-graph dwarf \
  ./build-experimental-perf-serial/experimental/benchmarks/experimental_comparison_benchmark \
  --benchmark_filter=SmallConfig

# Profile specific 3D benchmark size (matches V1_3D_MediumConfig, V2_3D_MediumConfig, V3_3D_MediumConfig)
perf record --call-graph dwarf \
  ./build-experimental-perf-serial/experimental/benchmarks/experimental_comparison_benchmark \
  --benchmark_filter=3D_MediumConfig

# Or use the convenience script
./scripts/profile_benchmark.sh experimental-perf-serial Small 2D
./scripts/profile_benchmark.sh experimental-perf-serial Medium 3D

# Profile all benchmarks and save
perf record --call-graph dwarf -o benchmark_perf.data \
  ./build-experimental-perf-serial/experimental/benchmarks/experimental_comparison_benchmark
```

### Workflow 5: OpenMP-Specific Profiling

```bash
# Control thread count
OMP_NUM_THREADS=4 perf record --call-graph dwarf \
  ./build-experimental-perf-openmp/experimental/tests/experimental_v1_unitary_test

# Profile per-thread
perf record --call-graph dwarf --threads \
  ./build-experimental-perf-openmp/experimental/tests/experimental_v1_unitary_test

# Check OpenMP overhead
perf record --call-graph dwarf -e omp_* \
  ./build-experimental-perf-openmp/experimental/tests/experimental_v1_unitary_test
```

### Workflow 6: Memory Access Patterns

```bash
# Profile cache behavior
perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses \
  ./build-experimental-perf-serial/experimental/tests/experimental_v1_unitary_test

# Profile memory bandwidth
perf stat -e mem-loads,mem-stores,cycles,instructions \
  ./build-experimental-perf-serial/experimental/tests/experimental_v1_unitary_test
```

## Perf Report Navigation

When running `perf report`:

- **Enter**: Expand function to see callees
- **+**: Zoom into selected function
- **-**: Zoom out to previous level
- **h**: Toggle call graph mode
- **s**: Switch to symbol view
- **t**: Toggle sorting
- **q**: Quit

## Common Performance Events

| Event | Description |
|-------|-------------|
| `cycles` | CPU cycles (overall time) |
| `instructions` | Retired instructions |
| `cache-references` | Total cache accesses |
| `cache-misses` | Cache misses |
| `branches` | Branch instructions |
| `branch-misses` | Mispredicted branches |
| `L1-dcache-load-misses` | L1 data cache misses |
| `LLC-load-misses` | Last level cache misses |
| `mem-loads` | Memory loads |
| `mem-stores` | Memory stores |
| `cpu-clock` | CPU clock time |
| `task-clock` | Task clock time |

## Call Graph Recording Methods

| Method | Accuracy | Overhead | Use Case |
|--------|----------|----------|----------|
| `dwarf` | High | High | Detailed profiling, when accuracy matters |
| `fp` (frame pointer) | Medium | Low | Production profiling, lower overhead |
| `lbr` (last branch record) | Medium | Medium | Hardware-supported, good for call stacks |

Example:
```bash
perf record --call-graph dwarf <executable>   # Most accurate
perf record --call-graph fp <executable>      # Lower overhead
perf record --call-graph lbr <executable>     # Hardware-supported
```

## Interpreting Results

### Key Metrics

1. **Overhead %**: Percentage of time spent in function
   - Focus on functions with >5% overhead

2. **IPC (Instructions Per Cycle)**: `instructions / cycles`
   - Higher is better (typically 1-4 for modern CPUs)
   - Low IPC indicates memory bottlenecks

3. **Cache Miss Rate**: `cache-misses / cache-references`
   - Lower is better
   - >10% may indicate optimization opportunities

4. **Branch Misprediction Rate**: `branch-misses / branches`
   - Lower is better
   - >5% may indicate opportunities to simplify conditionals

### For Set Algebra Operations

Key functions to examine in the experimental module:
- `set_union`: Union operation implementations
- `set_intersection`: Intersection operation implementations
- `set_difference`: Difference operation implementations
- Parallel scan operations (prefix sums)
- Memory allocation patterns in CSR operations

### For OpenMP Code

Look for:
- Load imbalance across threads
- Synchronization overhead
- False sharing (cache line bouncing)
- Sequential sections in parallel regions

## Troubleshooting

### Permission Denied Errors

```bash
# Check current setting
cat /proc/sys/kernel/perf_event_paranoid

# Relax restriction (requires sudo)
sudo sysctl -w kernel.perf_event_paranoid=1

# Or run with sudo
sudo perf record --call-graph dwarf <executable>
```

### No Symbols in Report

```bash
# Verify debug symbols are present
readelf -S <executable> | grep debug

# If missing, rebuild with debug symbols
cmake --preset experimental-perf-serial
cmake --build --preset experimental-perf-serial
```

### Inaccurate Call Graphs

Try different call graph methods:
```bash
perf record --call-graph dwarf <executable>   # Most accurate
perf record --call-graph fp <executable>      # If available
perf record --call-graph lbr <executable>     # If hardware supports
```

## Best Practices

1. **Profile representative workloads**: Use realistic data sizes
2. **Multiple runs**: Profile multiple runs to account for variance
3. **Warm-up**: Let benchmarks warm up before recording
4. **Focus on hot paths**: Don't optimize rarely-called code
5. **Compare backends**: Profile serial, OpenMP, and CUDA separately
6. **Document findings**: Keep notes of profiling sessions
7. **Verify optimizations**: Re-profile after changes to confirm improvements

## Additional Resources

- [perf Tutorial](https://perf.wiki.kernel.org/index.php/Tutorial)
- [Brendan Gregg's perf Examples](http://www.brendangregg.com/perf.html)
- [Linux perf wiki](https://perf.wiki.kernel.org/)
- [FlameGraph Tools](https://github.com/brendangregg/FlameGraph)
