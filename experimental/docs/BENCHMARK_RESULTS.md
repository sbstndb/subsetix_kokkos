# Successive Intersection Benchmark Results

## Summary

Benchmark comparison of **Naive** vs **Workspace** approaches for successive mesh intersection on CPU (Serial backend).

**Note**: Graph DAG approach was not benchmarked due to implementation incompatibility issues.

## Configuration

- **Backend**: Serial (CPU)
- **Benchmark Framework**: Google Benchmark v1.9.4
- **Min Time**: 3 seconds per benchmark
- **Date**: 2025-01-23

## Results

### 2 Meshes (Baseline)

| Config | Naive (ns) | Workspace (ns) | Speedup |
|--------|-----------|----------------|---------|
| Small  | 11,221    | 11,254         | 0.997× |
| Medium | -         | -              | -       |
| Large  | -         | -              | -       |

**Observation**: For 2 meshes, there's minimal difference. Workspace overhead slightly exceeds benefits.

### 4 Meshes

| Config | Naive (ns) | Workspace (ns) | Speedup |
|--------|-----------|----------------|---------|
| Small  | 13,343    | 13,781         | 0.968× |
| Medium | 21,689    | 21,174         | **1.024×** |
| Large  | 90,605    | 90,215         | **1.004×** |

**Observation**: Slight improvement for Medium/Large configs, but negligible (<3%).

### 8 Meshes

| Config | Naive (ns) | Workspace (ns) | Speedup |
|--------|-----------|----------------|---------|
| Small  | 14,411    | 13,618         | **1.058×** |
| Medium | 21,495    | 21,983         | 0.978× |
| Large  | 89,546    | 90,518         | 0.989× |

**Observation**: Mixed results. Small config shows 5.8% improvement, but others are within noise.

## Analysis

### Why Limited Speedup on CPU Serial?

1. **Memory allocation is fast** on CPU compared to GPU
2. **No kernel launch overhead** - all operations are in-memory
3. **Small problem sizes** - benchmarks use modest mesh sizes
4. **Cache effects** - both approaches benefit from CPU caching

### Expected Behavior on GPU

On GPU with CUDA backend, we expect:
- **Naive**: 4(N-1) kernel launches + synchronizations
- **Workspace**: 4(N-1) kernel launches (same), but fewer allocations
- **Graph DAG**: ~1 synchronization (significant speedup expected)

The real benefit of workspace reuse and graph execution would be visible on GPU where:
- Kernel launch overhead is significant (~10-20μs each)
- Memory allocation is expensive
- Synchronization points kill performance

## Test Results

**Passing Tests (10/18)**:
- EmptyVector_ReturnsEmpty ✓
- SingleMesh_ReturnsInput ✓
- AllEmptyMeshes_ReturnsEmpty ✓
- DisjointMeshes_ReturnsEmpty ✓
- FirstMeshEmpty_ReturnsEmpty ✓
- MiddleMeshEmpty_ReturnsEmpty ✓
- Idempotence_IdenticalMeshes ✓
- Associativity_DifferentOrders ✓
- UnifiedAPI_DefaultConfig ✓
- Workspace_AutoGrowth ✓

**Failing Tests (8/18)**: All involve Graph implementation comparison
- Root cause: Graph implementation uses different mesh size allocation
- Error: `Kokkos::deep_copy extents of views don't match`

## Recommendations

1. **For CPU Serial**: Workspace reuse provides minimal benefit for small N
2. **For GPU**: Workspace and Graph DAG would show significant improvement
3. **Graph Implementation**: Needs fixing to match v3::Mesh output format
4. **Future Work**: Run benchmarks on CUDA backend for meaningful comparison

## Files Created

- `experimental/include/experimental/subsetix/csr/set_algebra/successive_intersection.hpp` - Unified API
- `experimental/include/experimental/subsetix/csr/set_algebra/detail/successive_intersection_naive.hpp` - Naive implementation
- `experimental/include/experimental/subsetix/csr/set_algebra/detail/successive_intersection_workspace.hpp` - Workspace implementation
- `experimental/include/experimental/subsetix/csr/set_algebra/detail/successive_intersection_graph.hpp` - Graph implementation (incomplete)
- `experimental/tests/set_algebra/test_successive_intersection.cpp` - Cross-validation tests
- `experimental/benchmarks/set_algebra/successive_intersection_benchmark.cpp` - Performance benchmarks
