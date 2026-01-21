# Hybrid/Adaptive Strategy

## Overview

**Runtime strategy selection** based on mesh characteristics - no single approach is optimal for all scenarios.

## Core Idea

Different mesh sizes, densities, and access patterns benefit from different strategies:

```cpp
enum class Strategy {
  Classic_Binary,   // Current RowKey3D + binary search
  Morton_Merge,     // Morton codes + merge-based set ops
  Packed_Hash,      // Packed keys + hash lookup
  Bitmap,           // Bitmap for bounded dense domains
  Tiled_CSR,        // Tiled layout for stencil ops
  Octree_Morton,    // Octree for AMR
  Auto              // Automatic selection
};

Strategy select_strategy(const MeshMetrics& metrics);
```

## Decision Matrix

| Scenario | Row Key | Lookup | Layout | Best Strategy |
|----------|---------|--------|--------|---------------|
| **Small sparse** (< 10K rows, < 10% density) | RowKey3D | Binary search | CSR | `Classic_Binary` |
| **Large sparse** (> 1M rows, < 10% density) | Morton | Merge | Flat CSR | `Morton_Merge` |
| **Small dense** (< 10K rows, > 50% density) | Packed32 | Hash | CSR | `Packed_Hash` |
| **Large dense** (> 1M rows, > 50% density) | Packed32 | Perfect hash | **Bitmap** | `Bitmap` |
| **AMR refinement** | Morton | Tree traversal | Octree | `Octree_Morton` |
| **Stencil heavy** | Morton | Direct index | **Tiled** | `Tiled_CSR` |

## Mesh Metrics

```cpp
struct MeshMetrics {
  // Size
  std::size_t num_rows = 0;
  std::size_t num_intervals = 0;
  std::size_t total_cells = 0;

  // Domain
  Coord y_min = 0, y_max = 0;
  Coord z_min = 0, z_max = 0;
  std::size_t domain_size = 0;

  // Density
  double row_density = 0.0;      // num_rows / domain_size
  double cell_density = 0.0;

  // Access pattern hints
  bool is_static = true;
  bool needs_amr = false;
  bool stencil_heavy = false;
  bool set_operation_heavy = false;

  static MeshMetrics compute(const Mesh<3, MemorySpace>& mesh);
};
```

## Selection Logic

```cpp
class StrategySelector {
public:
  struct Config {
    std::size_t small_threshold = 10000;
    std::size_t large_threshold = 1000000;
    double sparse_density = 0.1;
    double dense_density = 0.5;
    std::size_t bitmap_domain_limit = 1 << 20;  // 1M
  };

  static Strategy select(const MeshMetrics& m, const Config& cfg = {}) {
    // Explicit overrides
    if (m.needs_amr) return Strategy::Octree_Morton;
    if (m.stencil_heavy && m.num_rows > cfg.small_threshold) {
      return Strategy::Tiled_CSR;
    }

    // Domain-based
    if (m.domain_size <= cfg.bitmap_domain_limit &&
        m.cell_density > cfg.dense_density) {
      return Strategy::Bitmap;
    }

    // Size + density matrix
    if (m.num_rows < cfg.small_threshold) {
      // Small meshes
      if (m.row_density > cfg.dense_density) {
        return Strategy::Packed_Hash;
      }
      return Strategy::Classic_Binary;
    }

    if (m.num_rows >= cfg.large_threshold) {
      // Large meshes
      if (m.row_density > cfg.dense_density) {
        if (m.domain_size <= cfg.bitmap_domain_limit) {
          return Strategy::Bitmap;
        }
        return Strategy::Tiled_CSR;
      }

      if (m.set_operation_heavy) {
        return Strategy::Morton_Merge;
      }
      return Strategy::Classic_Binary;
    }

    // Medium meshes (10K - 1M)
    if (m.set_operation_heavy) {
      return Strategy::Morton_Merge;
    }
    if (m.row_density > 0.3) {
      return Strategy::Packed_Hash;
    }
    return Strategy::Classic_Binary;
  }
};
```

## Performance Summary by Strategy

| Strategy | Memory | Lookup (5M rows) | Set Ops | Best For | Worst For |
|----------|--------|------------------|---------|----------|-----------|
| **Classic** | 16×R | 46 comp | Baseline | Small meshes | Large sparse |
| **Morton** | 16×R | 23 comp | 2× faster | Large sparse | Small meshes |
| **Hash** | 18×R | 1-3 probe | 3× faster | Dynamic data | Static sorted |
| **Bitmap** | 2 MB | O(1) | **100×** | Dense bounded | Sparse/unbounded |
| **Tiled** | 17×R | O(1) in tile | 1.2× | **Stencils** | Set ops |
| **Octree** | 25×R | O(log L) | 0.5× | **AMR** | Static meshes |

**Legend:** R = rows, L = levels (tree depth), comp = comparisons

## Hybrid Mesh Type

```cpp
template <class MemorySpace>
class HybridMesh3D {
public:
  Strategy strategy = Strategy::Auto;

  union {
    // Classic CSR (default)
    struct {
      Kokkos::View<RowKey3D*, MemorySpace> row_keys;
      Kokkos::View<std::size_t*, MemorySpace> row_ptr;
    } classic;

    // Morton-encoded
    struct {
      Kokkos::View<uint64_t*, MemorySpace> morton_codes;
      Kokkos::View<std::size_t*, MemorySpace> row_ptr;
    } morton;

    // Bitmap
    struct {
      Kokkos::View<uint64_t*, MemorySpace> bitmap;
      int y_max, z_max;
    } bitmap;

    // Tiled
    struct {
      Kokkos::View<TileMeta*, MemorySpace> tiles;
      Kokkos::View<Interval*, MemorySpace> intervals;
    } tiled;

    // Octree
    struct {
      Kokkos::View<Node*, MemorySpace> nodes;
    } octree;
  };

  // Automatic operations delegate to appropriate implementation
  template <typename Func>
  void for_each_row(Func&& func) const;

  Mesh<3, MemorySpace> intersect(const Mesh<3, MemorySpace>& other) const;
};
```

## Implementation Sketch

```cpp
// Factory function
template <class MemorySpace>
Mesh<3, MemorySpace>
create_optimized_mesh(const Mesh<3, MemorySpace>& input,
                      Strategy strategy = Strategy::Auto) {
  if (strategy == Strategy::Auto) {
    MeshMetrics metrics = MeshMetrics::compute(input);
    strategy = StrategySelector::select(metrics);
  }

  switch (strategy) {
    case Strategy::Classic_Binary:
      return input;  // No transformation

    case Strategy::Morton_Merge:
      return morton::transform(input);

    case Strategy::Packed_Hash:
      return hash::transform(input);

    case Strategy::Bitmap:
      return bitmap::from_csr(input, /*inferred domain*/);

    case Strategy::Tiled_CSR:
      return tiled::from_csr(input);

    case Strategy::Octree_Morton:
      return octree::from_csr(input);

    default:
      return input;
  }
}
```

## Pros and Cons

### Pros

1. **Optimal for all scenarios** - Automatic selection
2. **Future-proof** - Easy to add new strategies
3. **User-friendly** - No manual tuning needed
4. **Benchmarked** - Can validate choices empirically

### Cons

1. **Complex implementation** - Many code paths
2. **Metrics computation** - O(R) overhead at startup
3. **Debugging difficulty** - Which strategy is active?
4. **Type bloat** - Templates for all strategies

## When to Use

| Scenario | Use Hybrid? |
|----------|------------|
| Single well-understood workload | **No** - Pick one strategy |
| Variable workloads | **Yes** - Automatic adaptation |
| Library code | **Yes** - Users have different needs |
| Production | **Yes** - Can tune thresholds |

## Recommended Default Configuration

```cpp
StrategySelector::Config default_config() {
  return {
    .small_threshold = 10000,      // 10K rows
    .large_threshold = 1000000,    // 1M rows
    .sparse_density = 0.1,         // 10%
    .dense_density = 0.5,          // 50%
    .bitmap_domain_limit = 1 << 20  // 1M cells
  };
}
```

## Quick Reference

| If you have... | Use this strategy |
|----------------|------------------|
| < 10K rows | Classic Binary |
| 10K-1M rows, sparse | Morton Merge |
| > 1M rows, dense | Bitmap |
| AMR refinement | Octree |
| Stencil operations | Tiled |
| Dynamic insertions | Hash |
| Don't know / variable | **Auto** |

## References

- See individual strategy documents for details
- 01_morton_encoding.md
- 02_hash_based_lookup.md
- 03_merge_based_set_algebra.md
- 04_bitmap_representation.md
- 05_tiled_memory_layout.md
- 06_octree_hierarchical.md
