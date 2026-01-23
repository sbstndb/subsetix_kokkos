<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique
-->
# Performance Profiling with perf

This document describes how to profile the experimental module using the Linux `perf` tool.

## Prerequisites

### System Permissions

The `perf` tool requires appropriate permissions. Check the current setting:

```bash
cat /proc/sys/kernel/perf_event_paranoid
```

Values:
- `-1`: Allow all events for all users (no restrictions)
- `0`: Disallow raw and ftrace tracepoint access
- `1`: Disallow CPU event access
- `2`: Disallow kernel profiling
- `>= 2`: More restrictive

If you get permission errors, you can temporarily relax the restriction:

```bash
sudo sysctl -w kernel.perf_event_paranoid=1
```

To make it permanent, add to `/etc/sysctl.conf`:

```
kernel.perf_event_paranoid = 1
```

### Install perf

On Debian/Ubuntu:

```bash
sudo apt-get install linux-tools-common linux-tools-generic linux-tools-$(uname -r)
```

## Build with Profiling Support

Two CMake presets are available for profiling:

### Serial Backend

```bash
cmake --preset experimental-perf-serial
cmake --build --preset experimental-perf-serial
```

### OpenMP Backend

```bash
cmake --preset experimental-perf-openmp
cmake --build --preset experimental-perf-openmp
```

These presets:
- Enable `SUBSETIX_ENABLE_PERF`
- Add debug symbols (`-g`) for symbol resolution
- Use `RelWithDebInfo` build type (optimization with debug info)

## Profiling Workflows

### Quick Profiling with Scripts

Several convenience scripts are provided for different profiling tasks:

#### Generic profiling

```bash
./scripts/perf_profile.sh experimental-perf-serial \
  ./build-experimental-perf-serial/experimental/tests/experimental_v1_unitary_test
```

#### Profile specific benchmark configurations

```bash
# Profile Small 2D benchmarks
./scripts/profile_benchmark_perf.sh experimental-perf-serial Small 2D

# Profile Medium 2D benchmarks with OpenMP
./scripts/profile_benchmark_perf.sh experimental-perf-openmp Medium 2D

# Use perf stat for real-time statistics
./scripts/profile_benchmark_perf.sh experimental-perf-serial Small 2D --stat
```

#### Profile all benchmark configurations

```bash
# Profile all benchmarks (Small/Medium/Large × 2D/3D)
./scripts/profile_all_benchmarks.sh experimental-perf-serial
```

#### Compare performance results

```bash
# Compare before/after optimization
./scripts/compare_perf.sh perf_before.data perf_after.data
```

#### Generate consolidated reports

```bash
# Generate summary from multiple perf data files
./scripts/generate_perf_report.sh ./perf_output
```

All scripts create output in `./perf_output/` with timestamps by default.

### Manual Profiling

#### 1. Basic profiling with call graph

```bash
perf record --call-graph dwarf \
  ./build-experimental-perf-serial/experimental/tests/experimental_v1_unitary_test
```

#### 2. View the report

```bash
perf report
```

Navigation:
- `Enter`: Expand function
- `+`: Zoom into function
- `-`: Zoom out
- `q`: Quit

#### 3. Annotate specific functions

```bash
perf annotate
```

#### 4. Stat counters (real-time monitoring)

```bash
perf stat -e cycles,instructions,cache-misses,branches,branch-misses \
  ./build-experimental-perf-serial/experimental/tests/experimental_v1_unitary_test
```

## Profiling Benchmarks

Benchmark naming convention:
- **2D**: `V1_SmallConfig`, `V2_SmallConfig`, `V3_SmallConfig`, etc.
- **3D**: `V1_3D_SmallConfig`, `V2_3D_SmallConfig`, `V3_3D_SmallConfig`, etc.

### Run a specific benchmark size

```bash
# Filter for 2D Small benchmarks (matches V1_SmallConfig, V2_SmallConfig, V3_SmallConfig)
perf record --call-graph dwarf \
  ./build-experimental-perf-serial/experimental/benchmarks/experimental_comparison_benchmark \
  --benchmark_filter=SmallConfig

# Filter for 3D Medium benchmarks
perf record --call-graph dwarf \
  ./build-experimental-perf-serial/experimental/benchmarks/experimental_comparison_benchmark \
  --benchmark_filter=3D_MediumConfig
```

Or use the convenience script:
```bash
./scripts/profile_benchmark_perf.sh experimental-perf-serial Small 2D
./scripts/profile_benchmark_perf.sh experimental-perf-serial Medium 3D
```

### Profile all benchmarks

```bash
perf record --call-graph dwarf \
  ./build-experimental-perf-serial/experimental/benchmarks/experimental_comparison_benchmark
```

## Advanced Usage

### Specific Events

```bash
perf record -e cycles,instructions,L1-dcache-load-misses,LLC-load-misses \
  --call-graph dwarf ./build-experimental-perf-serial/experimental/tests/experimental_v1_unitary_test
```

Common events:
- `cycles`: CPU cycles
- `instructions`: Retired instructions
- `cache-misses`: Cache misses
- `branches`: Branch instructions
- `branch-misses`: Mispredicted branches
- `L1-dcache-load-misses`: L1 data cache misses
- `LLC-load-misses`: Last level cache misses

### Flame Graph Generation

Install FlameGraph tools:

```bash
git clone https://github.com/brendangregg/FlameGraph.git
```

Generate flamegraph:

```bash
perf script | ./FlameGraph/stackcollapse-perf.pl | \
  ./FlameGraph/flamegraph.pl > flamegraph.svg
```

Open `flamegraph.svg` in a browser.

### Compare Two Runs

```bash
# First run
perf record --call-graph dwarf ./build-experimental-perf-serial/experimental/tests/experimental_v1_unitary_test
perf report > report1.txt

# Second run (after changes)
perf record --call-graph dwarf ./build-experimental-perf-serial/experimental/tests/experimental_v1_unitary_test
perf report > report2.txt

# Compare
diff report1.txt report2.txt
```

## Tips for OpenMP Profiling

When profiling OpenMP code, you may want to:

1. Control the number of threads:
   ```bash
   OMP_NUM_THREADS=4 perf record --call-graph dwarf \
     ./build-experimental-perf-openmp/experimental/tests/experimental_v1_unitary_test
   ```

2. Profile per-thread:
   ```bash
   perf record --call-graph dwarf --threads \
     ./build-experimental-perf-openmp/experimental/tests/experimental_v1_unitary_test
   ```

## Interpreting Results

Key metrics to look for:

1. **Hot functions**: Functions consuming the most CPU time
2. **Cache misses**: High cache-miss ratios indicate memory access issues
3. **Branch mispredictions**: Conditional branch optimization opportunities
4. **Instruction mix**: Balance between compute and memory operations

For the experimental module, pay attention to:
- `set_union`, `set_intersection`, `set_difference` implementations
- Memory allocation patterns in CSR operations
- Scan operations (parallel prefix sums)

## Troubleshooting

### "Permission denied" or perf_event_paranoid errors

See "System Permissions" section above.

### No symbols in report

Ensure the binary was built with debug symbols:
```bash
file ./build-experimental-perf-serial/experimental/tests/experimental_v1_unitary_test
# Should show "with debug_info, not stripped"
```

### Inaccurate call graphs

Try different call graph recording methods:
- `--call-graph dwarf`: Most accurate, higher overhead
- `--call-graph fp`: Frame pointer based, lower overhead
- `--call-graph lbr`: Last branch record, hardware-supported

## Additional Resources

- [perf wiki](https://perf.wiki.kernel.org/)
- [Brendan Gregg's perf examples](http://www.brendangregg.com/perf.html)
- [FlameGraph documentation](https://github.com/brendangregg/FlameGraph)
