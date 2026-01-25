<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique
-->
# Row Mapping Alternatives Development Worktree

This worktree contains experimental implementations of alternative row mapping strategies
for the intersection algorithms in Subsetix-Kokkos.

## Purpose

The goal is to explore and benchmark different approaches to row mapping in the
`compute_row_ptr` operation, which is a critical performance component of the
intersection algorithms.

## Branch

- **Branch**: `feature/row-mapping-optimizations`
- **Based on**: `main` (commit 3a63d0a)
- **Worktree location**: `/home/sbstndbs/subsetix_kokkos_rowmap_dev`

## Alternative Implementations

This worktree will contain the following alternative implementations:

1. **v4_hash.hpp** - Hash-based row mapping
   - Uses unordered_map for O(1) average lookup
   - Memory overhead: hash table storage
   - Good for sparse, irregular row distributions

2. **v5_parallel_merge.hpp** - Parallel merge row mapping
   - Parallel merge-based approach for sorted inputs
   - Better GPU utilization through coalesced patterns
   - Suitable for large-scale parallel execution

3. **v6_direct_index.hpp** - Direct index row mapping
   - Index-based computation when applicable
   - Minimal memory overhead
   - Fastest for regular/structured patterns

4. **v7_soa_optimized.hpp** - Structure of Arrays + GPU optimized
   - SOA layout for better memory access patterns
   - GPU-specific optimizations (warp-level, shared memory)
   - Designed for CUDA/HIP backends

5. **v8_hybrid_cpu_gpu.hpp** - Hybrid CPU-GPU row mapping
   - Device-host cooperative computation
   - Overlaps computation and data transfer
   - Optimal for heterogeneous systems

6. **v9_adaptive.hpp** - Adaptive runtime strategy selection
   - Runtime profiling and strategy selection
   - Heuristics based on input characteristics
   - Automatic optimization for diverse workloads

## Testing

All implementations will be tested with:
- Cross-version comparison tests (verify output correctness)
- Overlap pattern tests (various geometric configurations)
- Large mesh tests (performance and scalability)

## Benchmarking

Comprehensive benchmarks will compare:
- Different input sizes (small, medium, large)
- Different row distributions (regular, sparse, irregular)
- Different backends (Serial, OpenMP, CUDA)
- Memory usage and allocation patterns

## Build Configuration

This worktree uses the standard playground build presets:

```bash
cmake --preset playground-serial
cmake --build --preset playground-serial
```

## Documentation

- `docs/HASH_ROW_MAPPING_ANALYSIS.md` - Analysis of hash-based approach
- `docs/HYBRID_ROW_MAPPER_DESIGN.md` - Design for hybrid CPU-GPU approach
- `docs/ROW_MAPPING_OPTIMIZATION_GUIDE.md` - General optimization guide
- `docs/row_mapping_optimization_patterns.hpp` - Common optimization patterns

## Next Steps

1. Implement v4_hash.hpp
2. Implement v5_parallel_merge.hpp
3. Implement v6_direct_index.hpp
4. Implement v7_soa_optimized.hpp
5. Implement v8_hybrid_cpu_gpu.hpp
6. Implement v9_adaptive.hpp
7. Create cross-version tests
8. Create comprehensive benchmarks
9. Analyze results and document recommendations
