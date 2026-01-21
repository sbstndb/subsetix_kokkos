# Tiled Memory Layout Strategy

## Overview

Group cells into fixed-size **tiles** (e.g., 16×16×16 blocks) for better spatial locality and GPU coalescing.

## Current Problem

```cpp
// Current: CSR with individual (y, z) rows
struct Mesh<3, MemorySpace> {
  Kokkos::View<RowKey3D*, MemorySpace> row_keys;  // Scattered (y,z) pairs
  Kokkos::View<std::size_t*, MemorySpace> row_ptr;
  Kokkos::View<Interval*, MemorySpace> intervals;
};

// Problem: Adjacent cells may be in different rows → poor cache locality
```

**Issues:**
- Poor spatial locality for stencil operations
- Memory transactions are scattered
- No cache blocking for GPU

## Proposed Solution

Organize cells into **fixed-size tiles** in the YZ plane:

```
Tile at (tile_y, tile_z):
  - Contains cells for y in [tile_y × TILE_Y, (tile_y + 1) × TILE_Y)
  - Contains cells for z in [tile_z × TILE_Z, (tile_z + 1) × TILE_Z)
  - Stores X-intervals contiguously

Benefits:
- Cells within tile are contiguous in memory
- Thread block processes one tile
- Natural fit for GPU memory hierarchy
```

### Tile Structure

```cpp
struct Tile {
  int tile_y, tile_z;           // Tile position
  std::size_t offset;            // Offset into intervals array
  std::size_t count;             // Number of intervals in tile
  std::size_t cells;             // Total cells in tile
};

template <class MemorySpace>
class TiledMesh3D {
  Kokkos::View<Tile*, MemorySpace> tiles;           // [num_tiles]
  Kokkos::View<TileKey*, MemorySpace> tile_keys;     // [num_tiles] - Sorted
  Kokkos::View<Interval*, MemorySpace> intervals;    // [total_intervals]
};
```

## API Design

### Core Tiled Structure

```cpp
namespace experimental::subsetix::csr::tiled {

/**
 * @brief Tile key identifying a tile in the YZ plane.
 */
struct TileKey {
  int ty = 0;  // Tile index in Y direction
  int tz = 0;  // Tile index in Z direction

  KOKKOS_INLINE_FUNCTION
  bool operator==(const TileKey& other) const {
    return ty == other.ty && tz == other.tz;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<(const TileKey& other) const {
    if (ty != other.ty) return ty < other.ty;
    return tz < other.tz;
  }
};

/**
 * @brief Per-tile storage metadata.
 */
struct TileMeta {
  TileKey key;           // Tile position (ty, tz)
  std::size_t offset;    // Offset into intervals array
  std::size_t count;     // Number of intervals in this tile
  std::size_t cells;     // Total cells in this tile
  int16_t num_rows;      // Number of non-empty (y,z) rows in tile
};

/**
 * @brief Tiled 3D mesh representation.
 */
template <class MemorySpace>
class Mesh3D {
public:
  static constexpr int DIM = 3;
  static constexpr int DEFAULT_TILE_SIZE_Y = 16;
  static constexpr int DEFAULT_TILE_SIZE_Z = 16;

  using TileMetaView = Kokkos::View<TileMeta*, MemorySpace>;
  using TileKeyView = Kokkos::View<TileKey*, MemorySpace>;
  using IntervalView = Kokkos::View<Interval*, MemorySpace>;

  // Tile dimensions
  int tile_size_y = DEFAULT_TILE_SIZE_Y;
  int tile_size_z = DEFAULT_TILE_SIZE_Z;

  // Tile metadata (sorted by tile key)
  TileMetaView tiles;
  TileKeyView tile_keys;

  // Interval storage (grouped by tile)
  IntervalView intervals;

  std::size_t num_tiles = 0;
  std::size_t num_intervals = 0;
  std::size_t total_cells = 0;

  /**
   * @brief Get tile key for a given (y, z) coordinate.
   */
  KOKKOS_INLINE_FUNCTION
  TileKey get_tile_key(Coord y, Coord z) const {
    return TileKey{
      y / tile_size_y,
      z / tile_size_z
    };
  }

  /**
   * @brief Find tile by binary search.
   */
  KOKKOS_INLINE_FUNCTION
  int find_tile(const TileKey& key) const {
    std::size_t lo = 0, hi = num_tiles;
    while (lo < hi) {
      const std::size_t mid = (lo + hi) / 2;
      if (tile_keys(mid) < key) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return (lo < num_tiles && tile_keys(lo) == key) ? static_cast<int>(lo) : -1;
  }
};

} // namespace experimental::subsetix::csr::tiled
```

### Stencil Operations

```cpp
/**
 * @brief Stencil computation with tile-aware halo loading.
 *
 * For interior tiles, all neighbors are within the same tile.
 * For boundary tiles, load adjacent tiles.
 */
template <typename T, class MemorySpace, typename StencilFunc>
void apply_stencil(const Mesh3D<MemorySpace>& mesh,
                   Kokkos::View<T*, MemorySpace> field_in,
                   Kokkos::View<T*, MemorySpace> field_out,
                   StencilFunc&& stencil) {

  const int ty_min = mesh.tile_keys(0).ty;
  const int tz_min = mesh.tile_keys(0).tz;

  Kokkos::parallel_for(
    "tiled_stencil",
    Kokkos::RangePolicy<ExecSpace>(0, mesh.num_tiles),
    KOKKOS_LAMBDA(const std::size_t tile_idx) {
      const auto& tile = mesh.tiles(tile_idx);
      const TileKey key = mesh.tile_keys(tile_idx);

      // Compute Y-Z bounds for this tile
      const Coord y_begin = key.ty * mesh.tile_size_y;
      const Coord y_end = y_begin + mesh.tile_size_y;
      const Coord z_begin = key.tz * mesh.tile_size_z;
      const Coord z_end = z_begin + mesh.tile_size_z;

      // Process intervals in this tile
      for (std::size_t iv_idx = 0; iv_idx < tile.count; ++iv_idx) {
        const std::size_t global_iv = tile.offset + iv_idx;
        const auto& iv = mesh.intervals(global_iv);

        for (Coord x = iv.begin; x < iv.end; ++x) {
          for (Coord y = y_begin; y < y_end; ++y) {
            for (Coord z = z_begin; z < z_end; ++z) {
              // Check if neighbors are within same tile
              const bool y_same_tile = (y > y_begin && y < y_end - 1);
              const bool z_same_tile = (z > z_begin && z < z_end - 1);

              T result = 0;

              if (y_same_tile && z_same_tile) {
                // All neighbors in same tile - no boundary checks!
                result = stencil(x, y, z, field_in);
              } else {
                // Need boundary checks or halo loading
                // ...
              }

              // Write output
              const std::size_t out_idx = compute_index(x, y, z);
              field_out(out_idx) = result;
            }
          }
        }
      }
    });
}
```

### Set Operations

```cpp
/**
 * @brief Tile-level set intersection.
 *
 * 1. Find common tiles via merge
 * 2. For each common tile, intersect intervals
 * 3. Merge results
 */
template <class MemorySpace>
Mesh3D<MemorySpace>
intersect_meshes(const Mesh3D<MemorySpace>& A,
                const Mesh3D<MemorySpace>& B) {
  using ExecSpace = typename MemorySpace::execution_space;

  Mesh3D<MemorySpace> result;
  result.tile_size_y = A.tile_size_y;
  result.tile_size_z = A.tile_size_z;

  if (A.num_tiles == 0 || B.num_tiles == 0) {
    return result;
  }

  // Phase 1: Find common tiles via two-pointer merge
  const std::size_t max_common_tiles = std::min(A.num_tiles, B.num_tiles);

  Kokkos::View<TileKey*, MemorySpace> common_tile_keys("common_keys", max_common_tiles);
  Kokkos::View<int*, MemorySpace> tile_idx_a("idx_a", max_common_tiles);
  Kokkos::View<int*, MemorySpace> tile_idx_b("idx_b", max_common_tiles);
  Kokkos::View<int*, MemorySpace> common_flags("flags", max_common_tiles);

  std::size_t num_common = 0;

  // Merge to find common tiles (can be done in parallel with merge-path)
  // For now, serial on host:
  auto host_keys_a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, A.tile_keys);
  auto host_keys_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, B.tile_keys);

  std::size_t ia = 0, ib = 0, out_idx = 0;
  while (ia < A.num_tiles && ib < B.num_tiles) {
    if (host_keys_a(ia) < host_keys_b(ib)) {
      ++ia;
    } else if (host_keys_b(ib) < host_keys_a(ia)) {
      ++ib;
    } else {
      // Common tile found!
      common_tile_keys(out_idx) = host_keys_a(ia);
      tile_idx_a(out_idx) = static_cast<int>(ia);
      tile_idx_b(out_idx) = static_cast<int>(ib);
      common_flags(out_idx) = 1;
      ++out_idx;
      ++ia;
      ++ib;
    }
  }
  num_common = out_idx;

  if (num_common == 0) {
    return result;
  }

  // Phase 2: Count intervals per common tile
  // Phase 3: Allocate and intersect intervals
  // Phase 4: Compact empty tiles
  // ... (similar to CSR but at tile level)

  return result;
}
```

## Performance Analysis

### Memory Overhead

| Component | CSR | Tiled | Overhead |
|-----------|-----|-------|----------|
| Row/Tile keys | 8 × R bytes | 8 × T bytes | T < R typically |
| Metadata | 8 × R bytes (row_ptr) | 32 × T bytes (TileMeta) | |
| Intervals | Same | Same | 0 |

Where R = number of rows, T = number of tiles

For 16×16 tiles and 5M rows:
- T ≈ R / 256 ≈ 20K tiles
- CSR: 16 × 5M = 80 MB
- Tiled: 8 × 20K + 32 × 20K = 0.8 MB (tile metadata) + intervals
- **Tile metadata overhead: ~1%**

### Spatial Locality

```
CSR layout (by row):
  [(y0,z0,x0..xN), (y0,z1,x0..xM), (y1,z0,x0..xK), ...]
  → Adjacent (y,z) may be far apart in memory

Tiled layout (by tile):
  [Tile(0,0): all intervals, Tile(0,1): all intervals, ...]
  → Adjacent (y,z) within same tile are contiguous

For 7-point 3D stencil accessing (x±1, y±1, z±1):
CSR: 6 potential row lookups (cache misses)
Tiled: 0-1 tile lookups (most neighbors in same tile)

Cache hit rate improvement: ~40%
```

### GPU Coalescing

```
Thread i accesses (y, z) within same tile:
  → Single 128-byte cache line covers multiple cells

With 16×16 tiles and 32-thread warp:
  - Warp can process 2 adjacent z-rows simultaneously
  - Coalesced memory access pattern

CSR: Scattered accesses, poor coalescing
Tiled: 80%+ warp efficiency for stencil operations
```

### Estimated Performance by Backend

#### Serial (CPU)

```
For 5M rows, 16×16 tiles (20K tiles):
Stencil operation (7-point 3D):
CSR: ~100 ms (cache misses on row lookups)
Tiled: ~60 ms (better locality)

Speedup: 1.5-2× for stencil operations
Set operations: ~1.2× (tile-level merge vs row-level)
```

#### OpenMP (CPU)

```
Benefits:
- Tiles map well to threads
- Reduced false sharing (tile boundaries)
- Better L3 cache utilization

For 5M rows, 16 threads:
CSR: ~100 ms / 16 = ~6 ms per thread (with contention)
Tiled: ~60 ms / 16 = ~3.75 ms per thread (independent tiles)

Speedup: 1.6× for stencil
Scalability: Near-linear (tiles are independent)
```

#### CUDA (GPU)

```
Benefits:
- Tile size matches GPU block size (16×16)
- Shared memory can hold entire tile
- Coalesced global memory access
- Reduced global memory traffic

For 5M rows, 16×16 tiles:
CSR: ~20 ms (scattered row access)
Tiled: ~8 ms (coalesced tile access)

Warp efficiency:
CSR: ~60% (divergent row lookups)
Tiled: ~85% (contiguous within tile)

Speedup: 2-3× for stencil operations
```

### Overhead for Small Meshes

| Mesh Size | CSR | Tiled (16×16) | Notes |
|-----------|-----|---------------|-------|
| 1K rows | 0.05 ms | 0.06 ms | Tile overhead |
| 10K rows | 0.5 ms | 0.4 ms | Break-even |
| 100K rows | 6 ms | 3 ms | 2× faster |
| 1M+ rows | 80 ms | 30 ms | 2.5× faster |

**Break-even point:** ~10K rows

## Kokkos Implementation

### Conversion from CSR

```cpp
// experimental/include/experimental/subsetix/csr/tiled/builder.hpp

namespace experimental::subsetix::csr::tiled {

class TiledMeshBuilder {
public:
  struct Config {
    int tile_size_y = 16;
    int tile_size_z = 16;
    bool use_morton_order = true;  // Sort tiles by Morton code
  };

  /**
   * @brief Build tiled mesh from CSR mesh.
   */
  template <class MemorySpace>
  static Mesh3D<MemorySpace>
  from_csr(const Mesh<3, MemorySpace>& csr_mesh,
           const Config& cfg = {}) {
    using ExecSpace = typename MemorySpace::execution_space;

    Mesh3D<MemorySpace> result;
    result.tile_size_y = cfg.tile_size_y;
    result.tile_size_z = cfg.tile_size_z;

    if (csr_mesh.num_rows == 0) {
      return result;
    }

    // Phase 1: Assign rows to tiles
    Kokkos::View<TileKey*, MemorySpace> row_tile_keys("row_tiles", csr_mesh.num_rows);

    auto csr_keys = csr_mesh.row_keys;

    Kokkos::parallel_for(
      "assign_rows_to_tiles",
      Kokkos::RangePolicy<ExecSpace>(0, csr_mesh.num_rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const TileKey tile{
          csr_keys(i).y / cfg.tile_size_y,
          csr_keys(i).z / cfg.tile_size_z
        };
        row_tile_keys(i) = tile;
      });

    ExecSpace().fence();

    // Phase 2: Sort rows by tile key
    Kokkos::View<std::size_t*, MemorySpace> permutation("perm", csr_mesh.num_rows);
    // ... sort implementation

    // Phase 3: Count unique tiles and intervals per tile
    // Phase 4: Allocate tile metadata and intervals
    // Phase 5: Fill tile data
    // Phase 6: Sort tiles by Morton code (optional)

    return result;
  }
};

} // namespace experimental::subsetix::csr::tiled
```

### Tile Statistics

```cpp
/**
 * @brief Statistics for tuning tile size.
 */
struct TileStatistics {
  std::size_t total_tiles = 0;
  std::size_t empty_tiles = 0;
  std::size_t partially_filled = 0;
  std::size_t full_tiles = 0;

  double average_fill_ratio = 0.0;
  double average_cells_per_tile = 0.0;

  template <class MemorySpace>
  static TileStatistics compute(const Mesh3D<MemorySpace>& mesh);
};
```

## Implementation Roadmap

### Phase 1: Core Structure (1 week)

- [ ] Implement `Mesh3D` tiled structure
- [ ] Add `TileKey` and `TileMeta` types
- [ ] Implement `get_tile_key` and `find_tile`
- [ ] Unit tests

### Phase 2: Conversion (1 week)

- [ ] Implement `from_csr` builder
- [ ] Sort rows by tile assignment
- [ ] Count tiles and intervals
- [ ] Fill tile metadata

### Phase 3: Stencil Operations (2 weeks)

- [ ] Implement tile-aware stencil
- [ ] Add halo loading for boundary tiles
- [ ] Optimize shared memory usage (GPU)
- [ ] Benchmark stencil performance

### Phase 4: Set Operations (2 weeks)

- [ ] Tile-level intersection
- [ ] Tile-level union
- [ ] Interval intersection within tiles
- [ ] Compact empty tiles

## Pros and Cons

### Pros

1. **Excellent spatial locality** - Contiguous cells within tile
2. **GPU-friendly** - Natural block size matching
3. **Stencil optimization** - Most neighbors in same tile
4. **Memory coalescing** - Sequential access patterns
5. **Parallel granularity** - Each tile is independent work unit

### Cons

1. **Tile size tuning** - Need to choose optimal size
2. **Partial tile overhead** - Boundary tiles are partially filled
3. **Implementation complexity** - More complex than CSR
4. **Small mesh overhead** - Tile metadata cost for tiny meshes
5. **Set operation complexity** - Two-level merge (tiles + intervals)

## When to Use

| Scenario | Recommended? |
|----------|--------------|
| Small meshes (< 10K rows) | **No** - Tile overhead |
| Large meshes | **Yes** - Better locality |
| Stencil-heavy | **Yes** - Major benefit |
| Set-operation heavy | **Maybe** - Small benefit |
| Memory-constrained | **No** - Slight overhead |
| GPU execution | **Yes** - Good fit |

## Tile Size Selection

| Tile Size | Best For | GPU Fit |
|-----------|----------|---------|
| 8×8 | Very sparse | Small warps |
| 16×16 | **General purpose** | **32-thread warp** |
| 32×32 | Dense domains | 2 warps per tile |
| 64×64 | Very dense | Larger blocks |

**Default recommendation: 16×16 tiles**

## References

- ExaBricks (NVIDIA): AMR volume rendering with bricks
- AMReX: Block-structured AMR with tiles
- Tiling GPU programming patterns
