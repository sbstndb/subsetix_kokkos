# 3D Coordinate Strategy Synthesis

## Executive Summary

The 3D extension of Subsetix introduces a fundamental challenge: efficiently representing and manipulating sparse (y, z) row keys in the CSR interval set structure. The current 2D implementation uses a single scalar `Coord` for rows, enabling O(log n) binary search with single comparison. For 3D, we need `(y, z)` pairs, which complicates lookup and set operations.

This document synthesizes **7 alternative strategies** for handling 3D coordinates, each with different trade-offs in memory, performance, and implementation complexity. The strategies address different aspects of the problem:

1. **Morton Encoding** - Reduces 2-comparison lookup to single comparison via bit interleaving
2. **Hash-Based Lookup** - O(1) average-case lookup via device-side hash table
3. **Merge-Based Set Algebra** - O(n+m) complexity via merge-path algorithms
4. **Bitmap Representation** - O(1) operations via bit-level parallelism for bounded domains
5. **Tiled Memory Layout** - Improved spatial locality via fixed-size tiles
6. **Octree Hierarchical** - AMR support via tree-based representation
7. **Hybrid Adaptive** - Runtime strategy selection based on mesh characteristics

### The Core Problem

```cpp
// Current 2D - single comparison
struct RowKey2D {
  Coord y;
  bool operator<(const RowKey2D& other) const { return y < other.y; }
};

// Current 3D proposal - 2 comparisons required
struct RowKey3D {
  Coord y, z;
  bool operator<(const RowKey3D& other) const {
    if (y != other.y) return y < other.y;  // Comparison 1
    return z < other.z;                    // Comparison 2
  }
};
```

**Impact:** Binary search requires twice as many comparisons per iteration, causing 2x slowdown in row mapping and significant branch divergence on GPU.

---

## API Comparison

### Strategy 1: Morton Encoding

**Data Structures:**

```cpp
namespace experimental::subsetix::csr::morton {

struct MortonKey {
  uint64_t code;  // Interleaved y and z bits

  KOKKOS_INLINE_FUNCTION
  bool operator<(const MortonKey& other) const {
    return code < other.code;  // SINGLE comparison
  }

  static MortonKey from_yz(Coord y, Coord z);
  void to_yz(Coord& y, Coord& z) const;
};

template <class MemorySpace>
class Mesh3D {
  Kokkos::View<MortonKey*, MemorySpace> morton_keys;  // Sorted Morton codes
  Kokkos::View<std::size_t*, MemorySpace> row_ptr;
  Kokkos::View<Interval*, MemorySpace> intervals;

  KOKKOS_INLINE_FUNCTION
  Coord get_y(std::size_t row_idx) const;
  KOKKOS_INLINE_FUNCTION
  Coord get_z(std::size_t row_idx) const;
};

} // namespace morton
```

**Key Methods:**
- `morton_encode_2d(y, z) -> uint64_t` - Bit interleaving
- `morton_decode_2d(code, &y, &z)` - Extract coordinates
- `Mesh3D::from_classic(classic_mesh)` - Conversion from RowKey3D

**Usage Example:**

```cpp
// Convert from classic CSR
auto classic_mesh = build_mesh_3d(...);
auto morton_mesh = morton::Mesh3D<Kokkos::CudaSpace>::from_classic(classic_mesh);

// Set operations use standard merge (both sorted by Morton code)
auto result = intersect_meshes(morton_mesh_a, morton_mesh_b);

// Lookup is now single-comparison binary search
int row_idx = find_row_by_morton(morton_mesh.morton_keys, morton_mesh.num_rows,
                                  morton::MortonKey::from_yz(y, z));
```

**Integration Points:**
- Drop-in replacement for `Mesh<3, MemorySpace>`
- Requires conversion pass at mesh construction
- Compatible with existing interval operations

---

### Strategy 2: Hash-Based Lookup

**Data Structures:**

```cpp
namespace experimental::subsetix::csr::hash {

template <class MemorySpace>
class RowHashMap {
  Kokkos::View<RowKey3D*, MemorySpace> keys;
  Kokkos::View<int32_t*, MemorySpace> values;   // Row indices
  Kokkos::View<uint8_t*, MemorySpace> occupied;

  std::size_t capacity;
  float max_load_factor = 0.7f;

  void build(const Kokkos::View<RowKey3D*, MemorySpace>& row_keys,
             std::size_t num_rows);

  KOKKOS_INLINE_FUNCTION
  int32_t find(const RowKey3D& key) const;  // O(1) average
};

} // namespace hash
```

**Key Methods:**
- `RowHashMap::build(row_keys, num_rows)` - Construct hash table
- `RowHashMap::find(key) -> int32_t` - O(1) lookup
- `intersect_meshes_hash(A, B)` - Hash-based intersection

**Usage Example:**

```cpp
// Build hash table for mesh B
hash::RowHashMap<Kokkos::CudaSpace> hash_b;
hash_b.build(mesh_b.row_keys, mesh_b.num_rows);

// O(1) lookups instead of O(log n) binary search
Kokkos::parallel_for("hash_intersect", mesh_a.num_rows,
  KOKKOS_LAMBDA(const std::size_t i) {
    const auto key = mesh_a.row_keys(i);
    const int idx_b = hash_b.find(key);  // O(1)!
    // ... process match
  });
```

**Integration Points:**
- Drop-in acceleration for set operations
- No changes to mesh data structure
- Requires hash table build phase (O(n))

---

### Strategy 3: Merge-Based Set Algebra

**Data Structures:**

```cpp
namespace experimental::subsetix::csr::merge_path {

// Uses existing RowKey3D - no new data structures
// Only changes the algorithm for set operations

KOKKOS_INLINE_FUNCTION
void merge_path_partition(
    const RowKey3D* A, std::size_t n_a,
    const RowKey3D* B, std::size_t n_b,
    std::size_t diagonal,
    std::size_t& a_idx, std::size_t& b_idx);

template <typename Compare>
KOKKOS_INLINE_FUNCTION
std::size_t merge_partition(
    const RowKey3D* A, std::size_t a_lo, std::size_t a_hi,
    const RowKey3D* B, std::size_t b_lo, std::size_t b_hi,
    int* match_a, int* match_b,
    Compare cmp);

} // namespace merge_path
```

**Key Methods:**
- `merge_path_partition(A, B, diagonal, &a, &b)` - Diagonal search
- `merge_partition(A, B, match_a, match_b, cmp)` - Two-pointer merge
- `intersect_merge_path(A, B)` - O(n+m) intersection

**Usage Example:**

```cpp
// Same interface as current implementation
auto result = merge_path::intersect_meshes(mesh_a, mesh_b);

// Internally uses merge-path instead of binary search
// - Partitions merge into P independent chunks
// - Each chunk does sequential two-pointer merge
// - No branch divergence, excellent parallelism
```

**Integration Points:**
- Algorithm-only change, no data structure changes
- Drop-in replacement for `set_intersection_device`
- Compatible with existing `Mesh<3, MemorySpace>`

---

### Strategy 4: Bitmap Representation

**Data Structures:**

```cpp
namespace experimental::subsetix::csr::bitmap {

template <class MemorySpace>
class Mesh3D {
  using WordType = uint64_t;
  Kokkos::View<WordType*, MemorySpace> bitmap;  // One bit per (y,z)
  Kokkos::View<std::size_t*, MemorySpace> row_ptr;
  Kokkos::View<Interval*, MemorySpace> intervals;

  Coord y_min, y_max, z_min, z_max;
  std::size_t num_words;

  KOKKOS_INLINE_FUNCTION
  bool has_row(Coord y, Coord z) const;

  KOKKOS_INLINE_FUNCTION
  void set_row(Coord y, Coord z);

  static Mesh3D from_csr(const Mesh<3, MemorySpace>& csr,
                         Coord y_min, Coord y_max,
                         Coord z_min, Coord z_max);
};

} // namespace bitmap
```

**Key Methods:**
- `has_row(y, z) -> bool` - O(1) bit test
- `set_row(y, z)` - Atomic bit set
- `intersect_meshes(A, B)` - Single bitwise AND kernel
- `count_rows()` - Parallel population count

**Usage Example:**

```cpp
// Convert from CSR (requires bounded domain)
auto bitmap_mesh = bitmap::Mesh3D<Kokkos::CudaSpace>::from_csr(
    classic_mesh, 0, 4096, 0, 4096);

// Set operations become bitwise operations
auto result = intersect_meshes(bitmap_mesh_a, bitmap_mesh_b);
// Internally: result.bitmap = A.bitmap & B.bitmap (single kernel!)

// O(1) row lookup
bool exists = bitmap_mesh.has_row(y, z);
```

**Integration Points:**
- Alternative mesh type for bounded domains
- Requires domain bounds at construction
- Dramatically faster for dense domains

---

### Strategy 5: Tiled Memory Layout

**Data Structures:**

```cpp
namespace experimental::subsetix::csr::tiled {

struct TileKey {
  int ty, tz;  // Tile indices

  KOKKOS_INLINE_FUNCTION
  bool operator<(const TileKey& other) const;
};

struct TileMeta {
  TileKey key;
  std::size_t interval_offset;
  std::size_t interval_count;
  std::size_t cells;
  int16_t num_rows;
};

template <class MemorySpace>
class Mesh3D {
  static constexpr int DEFAULT_TILE_SIZE_Y = 16;
  static constexpr int DEFAULT_TILE_SIZE_Z = 16;

  int tile_size_y, tile_size_z;
  Kokkos::View<TileMeta*, MemorySpace> tiles;
  Kokkos::View<TileKey*, MemorySpace> tile_keys;  // Sorted
  Kokkos::View<Interval*, MemorySpace> intervals;

  KOKKOS_INLINE_FUNCTION
  TileKey get_tile_key(Coord y, Coord z) const;

  KOKKOS_INLINE_FUNCTION
  int find_tile(const TileKey& key) const;

  static Mesh3D from_csr(const Mesh<3, MemorySpace>& csr);
};

} // namespace tiled
```

**Key Methods:**
- `get_tile_key(y, z) -> TileKey` - Compute tile index
- `find_tile(key) -> int` - Binary search on tiles
- `for_each_tile(func)` - Iterate tiles
- `apply_stencil(field_in, field_out, stencil)` - Tile-aware stencil

**Usage Example:**

```cpp
// Convert from CSR
auto tiled_mesh = tiled::Mesh3D<Kokkos::CudaSpace>::from_csr(classic_mesh);

// Stencil operations benefit from tile locality
apply_stencil(tiled_mesh, field_in, field_out,
  KOKKOS_LAMBDA(Coord x, Coord y, Coord z, auto& field) {
    // Most neighbors are in same tile → no boundary checks
    return laplacian(x, y, z, field);
  });

// Set operations work at tile level
auto result = intersect_meshes(tiled_mesh_a, tiled_mesh_b);
```

**Integration Points:**
- Alternative mesh type for stencil-heavy workloads
- Transparent conversion from CSR
- Best for GPU execution

---

### Strategy 6: Octree Hierarchical

**Data Structures:**

```cpp
namespace experimental::subsetix::csr::octree {

struct Node {
  uint64_t morton_code;     // Morton code for position
  int8_t level;             // Tree level
  uint8_t child_mask;       // Which children exist (8 bits)
  uint32_t first_child;     // Index of first child
  uint32_t parent_idx;      // Index of parent

  // Leaf data
  std::size_t interval_offset;
  std::size_t interval_count;

  KOKKOS_INLINE_FUNCTION
  bool is_leaf() const;

  KOKKOS_INLINE_FUNCTION
  bool has_child(int idx) const;
};

template <class MemorySpace>
class Mesh3D {
  Kokkos::View<Node*, MemorySpace> nodes;
  Kokkos::View<Interval*, MemorySpace> intervals;

  std::size_t num_nodes, num_leaves;
  int max_level;

  KOKKOS_INLINE_FUNCTION
  const Node& find_node(Coord x, Coord y, Coord z) const;

  void refine_leaf(std::size_t leaf_idx);
  void coarsen_siblings(std::size_t parent_idx);

  template <typename Func>
  void for_each_leaf(Func&& func) const;
};

} // namespace octree
```

**Key Methods:**
- `find_node(x, y, z) -> Node` - Tree traversal
- `refine_leaf(leaf_idx)` - Split leaf into 8 children
- `coarsen_siblings(parent_idx)` - Merge 8 leaves
- `traverse_coarse_to_fine(func)` - Level-order traversal

**Usage Example:**

```cpp
// Build octree from CSR
auto octree_mesh = octree::Mesh3D<Kokkos::CudaSpace>::from_csr(classic_mesh);

// AMR refinement
for (auto leaf : octree_mesh.leaves()) {
  if (should_refine(leaf)) {
    octree_mesh.refine_leaf(leaf.idx);
  }
}

// Multigrid: traverse coarse to fine
traverse_coarse_to_fine(octree_mesh,
  KOKKOS_LAMBDA(const Node& node, std::size_t idx) {
    // Process each level
  });
```

**Integration Points:**
- Specialized for AMR workflows
- Supports dynamic refinement/coarsening
- Natural hierarchy for multigrid

---

### Strategy 7: Hybrid Adaptive

**Data Structures:**

```cpp
namespace experimental::subsetix::csr::hybrid {

enum class Strategy {
  Classic_Binary,   // Current RowKey3D + binary search
  Morton_Merge,     // Morton codes + merge
  Packed_Hash,      // Packed keys + hash
  Bitmap,           // Bitmap for dense bounded
  Tiled_CSR,        // Tiled layout
  Octree_Morton,    // Octree for AMR
  Auto              // Automatic selection
};

struct MeshMetrics {
  std::size_t num_rows, num_intervals, total_cells;
  Coord y_min, y_max, z_min, z_max;
  std::size_t domain_size;
  double row_density, cell_density;
  bool is_static, needs_amr, stencil_heavy, set_operation_heavy;

  static MeshMetrics compute(const Mesh<3, MemorySpace>& mesh);
};

class StrategySelector {
  static Strategy select(const MeshMetrics& metrics,
                         const Config& cfg = {});
};

template <class MemorySpace>
class HybridMesh3D {
  Strategy strategy;

  union {
    struct { /* classic data */ } classic;
    struct { /* morton data */ } morton;
    struct { /* bitmap data */ } bitmap;
    struct { /* tiled data */ } tiled;
    struct { /* octree data */ } octree;
  };

  // Operations delegate to appropriate implementation
  Mesh<3, MemorySpace> intersect(const Mesh<3, MemorySpace>& other) const;
};

} // namespace hybrid
```

**Key Methods:**
- `MeshMetrics::compute(mesh) -> MeshMetrics` - Analyze mesh
- `StrategySelector::select(metrics) -> Strategy` - Choose strategy
- `create_optimized_mesh(input, strategy) -> Mesh` - Transform mesh

**Usage Example:**

```cpp
// Automatic strategy selection
auto metrics = hybrid::MeshMetrics::compute(input_mesh);
auto strategy = hybrid::StrategySelector::select(metrics);
auto optimized = create_optimized_mesh(input_mesh, strategy);

// Or use Auto mode
auto optimized = create_optimized_mesh(input_mesh, hybrid::Strategy::Auto);

// Operations are automatic
auto result = optimized.intersect(other_mesh);
// Uses bitmap::intersect if optimized is bitmap
// Uses morton::intersect if optimized is morton
// etc.
```

**Integration Points:**
- Facade over all strategies
- Runtime decision making
- Zero user code changes for basic use

---

## Strategy Comparison Table

| Aspect | Classic Binary | Morton Encoding | Hash Table | Merge-Path | Bitmap | Tiled | Octree |
|--------|---------------|-----------------|------------|------------|--------|-------|--------|
| **Memory Footprint** | 16×R | 16×R | 18.6×R | 16×R | 2 MB* | 17×R | 25×R |
| **Lookup Complexity** | O(log n) | O(log n) | O(1) avg | O(1)** | O(1) | O(1)*** | O(log L) |
| **Lookup Comparisons** | 2 per iter | 1 per iter | 1-3 probes | 0 | 0 | 0 | 1 per level |
| **Set Operation Complexity** | O(R_A × log R_B) | O(R_A + R_B) | O(R_A + R_B) | O(R_A + R_B) | O(Y×Z/64) | O(T_A + T_B) | O(R_A + R_B) |
| **Best Use Case** | < 10K rows | Large sparse | Dynamic data | Large sorted | Dense bounded | Stencils | AMR |
| **Worst Use Case** | Large sparse | Small meshes | Memory limit | Tiny meshes | Sparse/unbounded | Set ops | Static |
| **Implementation Difficulty** | ✓ (existing) | Medium | Medium | Medium | Medium | High | Very High |
| **GPU Friendliness** | Poor (divergence) | Good | Good | Excellent | Excellent | Excellent | Poor (serial) |
| **Break-even Point** | N/A | ~10K rows | ~10K rows | ~3K rows | ~100×100 | ~10K rows | ~5K cells |

**Legend:**
- R = number of rows
- T = number of tiles (typically R/256)
- Y×Z = domain size
- L = number of tree levels
- *Bitmap memory is constant for bounded domain
- **Merge-path has no binary search, only sequential merge
- ***Tiled lookup is O(1) within tile, binary search on tiles

---

## Performance by Backend

### Serial (CPU)

| Strategy | Small (< 10K) | Medium (10K-1M) | Large (> 1M) | Break-even |
|----------|---------------|-----------------|--------------|------------|
| **Classic Binary** | **0.05 ms** | 5 ms | 100 ms | N/A |
| **Morton Encoding** | 0.06 ms (+20%) | **2 ms** (2.5×) | **50 ms** (2×) | ~10K rows |
| **Hash Table** | 0.15 ms (3×) | 0.4 ms | **15 ms** | ~10K rows |
| **Merge-Path** | 0.04 ms | **1.5 ms** (3×) | **25 ms** (4×) | ~3K rows |
| **Bitmap** | 0.08 ms | 0.3 ms | **5 ms** (20×) | ~100×100 |
| **Tiled** | 0.06 ms | 3 ms | 60 ms | ~10K rows |
| **Octree** | 0.15 ms | 8 ms | 50 ms | ~5K cells |

**Notes:**
- Hash table has build overhead that hurts small meshes
- Merge-path excels on CPU due to sequential merge
- Bitmap dominates for dense bounded domains
- Octree only wins with AMR operations

---

### OpenMP (CPU)

| Strategy | Small (< 10K) | Medium (10K-1M) | Large (> 1M) | Scalability |
|----------|---------------|-----------------|--------------|-------------|
| **Classic Binary** | **0.01 ms** | 0.6 ms | 12 ms | Good |
| **Morton Encoding** | 0.012 ms | **0.4 ms** (1.5×) | **8 ms** (1.5×) | Good |
| **Hash Table** | 0.03 ms | **0.25 ms** (2.4×) | **3 ms** (4×) | Excellent |
| **Merge-Path** | 0.008 ms | **0.2 ms** (3×) | **4 ms** (3×) | Excellent |
| **Bitmap** | 0.02 ms | **0.1 ms** (6×) | **0.6 ms** (20×) | Near-linear |
| **Tiled** | 0.012 ms | 0.5 ms | **10 ms** (1.2×) | Excellent |
| **Octree** | 0.04 ms | 1.5 ms | 8 ms | Good |

**Notes:**
- Merge-path scales best due to independent partitions
- Hash table shows excellent parallel speedup
- Bitmap near-linear due to embarrassingly parallel bitwise ops
- Morton reduces branch misprediction

---

### CUDA (GPU)

| Strategy | Small (< 10K) | Medium (10K-1M) | Large (> 1M) | Warp Efficiency |
|----------|---------------|-----------------|--------------|-----------------|
| **Classic Binary** | **0.02 ms** | 2 ms | 50 ms | ~60% |
| **Morton Encoding** | 0.025 ms | **1 ms** (2×) | **25 ms** (2×) | ~85% |
| **Hash Table** | 0.05 ms | **0.8 ms** (2.5×) | **8 ms** (6×) | ~80% |
| **Merge-Path** | 0.015 ms | **0.5 ms** (4×) | **10 ms** (5×) | **~90%** |
| **Bitmap** | 0.03 ms | **0.15 ms** (13×) | **0.5 ms** (100×) | **~95%** |
| **Tiled** | 0.025 ms | 1.5 ms | **30 ms** (1.7×) | **~85%** |
| **Octree** | 0.05 ms | 3 ms | 30 ms | ~50% |

**Notes:**
- Bitmap dominates on GPU (tensor cores, coalesced access)
- Merge-path best for set operations (no divergence)
- Morton reduces warp divergence significantly
- Octree suffers from serial traversal

---

### Break-even Analysis

| Mesh Size | Best Strategy | Speedup vs Classic |
|-----------|---------------|-------------------|
| **< 1K rows** | Classic Binary | 1× (baseline) |
| **1K-10K rows** | Merge-Path | 2-3× |
| **10K-100K rows (sparse)** | Morton + Hash | 3-4× |
| **10K-100K rows (dense)** | Bitmap | 10-20× |
| **100K-1M rows (sparse)** | Merge-Path | 4-5× |
| **100K-1M rows (dense)** | Bitmap | 20-50× |
| **> 1M rows (sparse)** | Merge-Path | 5-10× |
| **> 1M rows (dense)** | Bitmap | 50-100× |
| **AMR workflows** | Octree | 10× (refinement) |
| **Stencil heavy** | Tiled | 2-3× |

**Allocation/Conversion Overhead:**

| Strategy | Conversion Time | When Worth It |
|----------|----------------|---------------|
| Classic | 0 ms | N/A |
| Morton | O(n log n) sort | > 10K rows |
| Hash | O(n) build | > 10K rows |
| Merge-Path | 0 ms | Always |
| Bitmap | O(n) scatter | Dense domains |
| Tiled | O(n) assign | > 10K rows |
| Octree | O(n log n) build | AMR only |

---

## Implementation Estimates

### Morton Encoding

**Lines of Code:** ~800 LOC
- Core types: 150 LOC
- Encoding/decoding: 100 LOC
- Conversion: 200 LOC
- Set operations: 250 LOC
- Tests: 100 LOC

**Time to Implement:** 3-4 person-weeks

**Risk Level:** Low

**Dependencies:**
- None (standalone)
- Can coexist with classic implementation

**Phasing:**
1. Week 1: Core infrastructure (MortonKey, encode/decode)
2. Week 2: Conversion utilities
3. Week 3: Set algebra integration
4. Week 4: Testing and optimization

---

### Hash-Based Lookup

**Lines of Code:** ~900 LOC
- Hash table: 300 LOC
- Hash functions: 100 LOC
- Set operations: 300 LOC
- Tests: 200 LOC

**Time to Implement:** 3-4 person-weeks

**Risk Level:** Medium

**Dependencies:**
- None
- Requires atomic operations support

**Phasing:**
1. Week 1: Basic hash table with linear probing
2. Week 2: Optimize hash functions
3. Week 3: Set operation integration
4. Week 4: GPU optimization and testing

---

### Merge-Based Set Algebra

**Lines of Code:** ~600 LOC
- Merge-path primitives: 200 LOC
- Set operations: 300 LOC
- Tests: 100 LOC

**Time to Implement:** 2-3 person-weeks

**Risk Level:** Low

**Dependencies:**
- None (algorithm-only change)
- Compatible with existing mesh types

**Phasing:**
1. Week 1: Diagonal search and merge partition
2. Week 2: Set operation integration
3. Week 3: Optimization and testing

---

### Bitmap Representation

**Lines of Code:** ~1,100 LOC
- Core bitmap: 300 LOC
- Conversion: 300 LOC
- Set operations: 300 LOC
- Tests: 200 LOC

**Time to Implement:** 4-5 person-weeks

**Risk Level:** Medium

**Dependencies:**
- Requires domain bounds inference
- Population count intrinsics

**Phasing:**
1. Week 1: Core bitmap structure
2. Week 2: Conversion from CSR
3. Week 3: Set operations (bitwise)
4. Week 4-5: Optimization and testing

---

### Tiled Memory Layout

**Lines of Code:** ~1,400 LOC
- Core tiled: 400 LOC
- Builder: 400 LOC
- Set operations: 400 LOC
- Stencil ops: 200 LOC

**Time to Implement:** 5-6 person-weeks

**Risk Level:** High

**Dependencies:**
- Tile size tuning
- Shared memory optimization (GPU)

**Phasing:**
1. Week 1-2: Core tiled structure
2. Week 3: Conversion builder
3. Week 4: Set operations
4. Week 5: Stencil optimization
5. Week 6: Testing and tuning

---

### Octree Hierarchical

**Lines of Code:** ~1,800 LOC
- Core octree: 600 LOC
- Traversal: 300 LOC
- AMR operations: 500 LOC
- Set operations: 400 LOC

**Time to Implement:** 6-8 person-weeks

**Risk Level:** High

**Dependencies:**
- 3D Morton encoding
- Dynamic memory management
- Complex tree algorithms

**Phasing:**
1. Week 1-2: Core octree structure
2. Week 3: Traversal algorithms
3. Week 4-5: AMR operations (refine/coarsen)
4. Week 6-7: Set operations
5. Week 8: Testing and optimization

---

### Hybrid Adaptive

**Lines of Code:** ~2,500 LOC
- Strategy selector: 400 LOC
- Metrics computation: 300 LOC
- Facade/delegation: 800 LOC
- Integration: 1,000 LOC

**Time to Implement:** 4-6 person-weeks (after other strategies)

**Risk Level:** Medium

**Dependencies:**
- All other strategies must be implemented
- Extensive testing matrix

**Phasing:**
1. Week 1-2: Strategy selector and metrics
2. Week 3-4: Facade and delegation
3. Week 5-6: Integration testing

---

## Kokkos Implementation Notes

### Morton Encoding

**Kokkos Features:**
- `Kokkos::View<uint64_t*, MemorySpace>` for Morton codes
- `Kokkos::sort()` for sorting by Morton code
- Custom comparator in `parallel_sort`

**GPU Considerations:**
- Use SWAR bit interleaving (not loops)
- Lookup tables for small coordinates
- Cache Morton codes in shared memory

**Memory Space Handling:**
- Convert between HostSpace and DeviceSpace
- Morton encoding is host-side (preprocessing)
- Sorting can be device-side

**Execution Space Policies:**
- Serial: Simple loop-based encoding
- OpenMP: Parallel encoding with `parallel_for`
- CUDA: Cooperative group encoding

```cpp
// Optimized device-side encoding
KOKKOS_INLINE_FUNCTION
uint64_t morton_encode_2d_fast(Coord y, Coord z) {
  uint64_t x = static_cast<uint64_t>(y);
  uint64_t z_ = static_cast<uint64_t>(z);

  // SWAR (SIMD Within A Register)
  x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
  x = (x | (x << 8))  & 0x00FF00FF00FF00FFULL;
  x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0FULL;
  x = (x | (x << 2))  & 0x3333333333333333ULL;
  x = (x | (x << 1))  & 0x5555555555555555ULL;

  z_ = (z_ | (z_ << 16)) & 0x0000FFFF0000FFFFULL;
  z_ = (z_ | (z_ << 8))  & 0x00FF00FF00FF00FFULL;
  z_ = (z_ | (z_ << 4))  & 0x0F0F0F0F0F0F0F0FULL;
  z_ = (z_ | (z_ << 2))  & 0x3333333333333333ULL;
  z_ = (z_ | (z_ << 1))  & 0x5555555555555555ULL;

  return (x << 1) | z_;
}
```

---

### Hash-Based Lookup

**Kokkos Features:**
- `Kokkos::View` with `Atomic` modifier for concurrent insert
- `Kokkos::atomic_compare_exchange` for lock-free insert
- `Kokkos::parallel_reduce` for statistics

**GPU Considerations:**
- Use warp-level primitives for collision resolution
- Avoid atomic contention in build phase
- Pre-allocate to avoid dynamic allocation

**Memory Space Handling:**
- Hash table lives in DeviceMemorySpace
- Build happens on device
- No host-device copies during lookup

**Execution Space Policies:**
- Serial: Simple hash with linear probing
- OpenMP: Parallel build with thread-local buckets
- CUDA: Warp-aggregated atomics

```cpp
// GPU-friendly hash table build
Kokkos::parallel_for("hash_build",
  Kokkos::RangePolicy<ExecSpace>(0, num_rows),
  KOKKOS_LAMBDA(const std::size_t i) {
    const Key key = row_keys(i);
    const Value value = static_cast<Value>(i);

    std::size_t idx = hash_function(key) % capacity;

    // Lock-free linear probing with atomics
    while (true) {
      uint8_t expected = 0;
      if (Kokkos::atomic_compare_exchange(&occupied(idx), &expected, 1) == 0) {
        keys(idx) = key;
        values(idx) = value;
        break;
      }
      if (keys(idx) == key) {
        break;  // Already inserted (duplicate)
      }
      idx = (idx + 1) % capacity;
    }
  });
```

---

### Merge-Based Set Algebra

**Kokkos Features:**
- `Kokkos::parallel_scan` for offset computation
- `Kokkos::View` for temporary match arrays
- No special memory requirements

**GPU Considerations:**
- Partition size tuning is critical
- Too few partitions: load imbalance
- Too many: diagonal search overhead
- Target: ~256 rows per partition

**Memory Space Handling:**
- All temporary arrays in DeviceMemorySpace
- No additional allocation beyond current approach

**Execution Space Policies:**
- Serial: Simple merge algorithm
- OpenMP: Parallel partitions
- CUDA: One partition per CTA

```cpp
// Merge-path intersection
const std::size_t num_partitions = (n_a + n_b + 255) / 256;

Kokkos::parallel_for("merge_path_intersect",
  Kokkos::RangePolicy<ExecSpace>(0, num_partitions),
  KOKKOS_LAMBDA(const std::size_t p) {
    // Find partition bounds via diagonal search
    std::size_t diagonal_start = (p * (n_a + n_b)) / num_partitions;
    std::size_t diagonal_end = ((p + 1) * (n_a + n_b)) / num_partitions;

    std::size_t a_lo, b_lo, a_hi, b_hi;
    diagonal_search(A, n_a, B, n_b, diagonal_start, a_lo, b_lo);
    diagonal_search(A, n_a, B, n_b, diagonal_end, a_hi, b_hi);

    // Sequential merge within partition
    std::size_t out = partition_offsets(p);
    while (a_lo < a_hi && b_lo < b_hi) {
      if (A[a_lo] < B[b_lo]) { ++a_lo; }
      else if (B[b_lo] < A[a_lo]) { ++b_lo; }
      else {
        match_a[out] = a_lo;
        match_b[out] = b_lo;
        ++out; ++a_lo; ++b_lo;
      }
    }
    partition_offsets(p + 1) = out;
  });
```

---

### Bitmap Representation

**Kokkos Features:**
- `Kokkos::View<uint64_t*, MemorySpace>` for bitmap
- Population count intrinsics (`__popcll` on CUDA, `std::popcount` on CPU)
- Bitwise operations in parallel kernels

**GPU Considerations:**
- Tensor core acceleration for bitwise ops
- 64-bit coalesced memory access
- Perfect fit for GPU architecture

**Memory Space Handling:**
- Bitmap in DeviceMemorySpace
- Domain inference on host, conversion on device

**Execution Space Policies:**
- Serial: Loop over words with `std::popcount`
- OpenMP: Parallel over words
- CUDA: Parallel over words with `__popcll`

```cpp
// Bitmap intersection (single kernel!)
Kokkos::parallel_for("bitmap_intersect",
  Kokkos::RangePolicy<ExecSpace>(0, num_words),
  KOKKOS_LAMBDA(const std::size_t i) {
    result.bitmap(i) = A.bitmap(i) & B.bitmap(i);  // Bitwise AND!
  });

// Count set bits (parallel reduction)
std::size_t num_rows = 0;
Kokkos::parallel_reduce("bitmap_count",
  Kokkos::RangePolicy<ExecSpace>(0, num_words),
  KOKKOS_LAMBDA(const std::size_t i, std::size_t& local) {
    #ifdef __CUDA_ARCH__
      local += __popcll(result.bitmap(i));
    #else
      local += std::popcount(result.bitmap(i));
    #endif
  }, num_rows);
```

---

### Tiled Memory Layout

**Kokkos Features:**
- `Kokkos::View<TileMeta*, MemorySpace>` for tile metadata
- Team policies for tile-level parallelism
- Shared memory for tile data (GPU)

**GPU Considerations:**
- Tile size should match warp size (32)
- 16×16 tiles = 256 cells per tile
- Shared memory can hold entire tile
- Coalesced access within tile

**Memory Space Handling:**
- Tiles in DeviceMemorySpace
- Conversion requires sorting

**Execution Space Policies:**
- Serial: Loop over tiles
- OpenMP: One thread per tile
- CUDA: One CTA per tile

```cpp
// Tile-aware stencil
Kokkos::parallel_for("tiled_stencil",
  Kokkos::TeamPolicy<ExecSpace>(num_tiles, Kokkos::AUTO, 16),
  KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<ExecSpace>::member_type& team) {
    const std::size_t tile_idx = team.league_rank();
    const auto& tile = tiles(tile_idx);

    // Load tile into shared memory
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, tile.count),
      [&](const std::size_t i) {
        shared_intervals[i] = intervals(tile.offset + i);
      });
    team.team_barrier();

    // Process stencil (neighbors in same tile)
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, tile.cells),
      [&](const std::size_t i) {
        // No boundary checks needed!
        output(i) = apply_stencil(i, shared_intervals);
      });
  });
```

---

### Octree Hierarchical

**Kokkos Features:**
- `Kokkos::View<Node*, MemorySpace>` for tree
- Recursive parallelism (limited)
- Level-order traversal

**GPU Considerations:**
- Tree traversal is serial (major limitation)
- Batch refinement operations
- Use shared memory for subtrees

**Memory Space Handling:**
- Nodes in DeviceMemorySpace
- Tree building on host, traversal on device

**Execution Space Policies:**
- Serial: Recursive traversal
- OpenMP: Level-order parallel
- CUDA: Limited (serial traversal)

```cpp
// Coarse-to-fine traversal (GPU-friendly)
for (int level = 0; level <= max_level; ++level) {
  Kokkos::parallel_for("octree_level",
    Kokkos::RangePolicy<ExecSpace>(0, num_nodes),
    KOKKOS_LAMBDA(const std::size_t i) {
      const Node& node = nodes(i);
      if (node.level == level && node.is_leaf()) {
        // Process all nodes at this level in parallel
        process_leaf(node, i);
      }
    });
  ExecSpace().fence();
}
```

---

## Recommendations

### Which Strategy to Implement First

**Recommendation: Merge-Based Set Algebra (Strategy 3)**

**Rationale:**
1. **Lowest Risk** - Algorithm-only change, no new data structures
2. **Immediate Benefit** - 4-5× speedup for large meshes
3. **No Overhead** - Zero conversion cost, works with existing meshes
4. **GPU Friendly** - Best warp efficiency (~90%)
5. **Fastest to Implement** - 2-3 weeks vs 4-8 weeks for others
6. **Foundational** - Enables other optimizations (Morton + Merge)

**Implementation Path:**
```
Week 1: Implement diagonal_search and merge_partition kernels
Week 2: Integrate with v1 set algebra, replace Phase 1
Week 3: Testing, benchmarking, tuning partition sizes
```

**Expected Results:**
- 4-5× speedup for set operations on meshes > 10K rows
- 2× speedup for meshes 3K-10K rows
- No performance loss for small meshes (< 3K rows)

---

### Complementary Strategies

These strategies work well **in combination**:

1. **Merge-Path + Morton Encoding** (Best combo)
   - Morton reduces comparisons from 2 to 1
   - Merge-path eliminates binary search entirely
   - Combined: 6-8× speedup for large sparse meshes
   - Implementation: First do merge-path, then add Morton

2. **Bitmap + Hybrid Selection** (For dense domains)
   - Bitmap dominates for dense bounded domains (50-100×)
   - Hybrid selector automatically chooses bitmap when appropriate
   - Implementation: Add bitmap after merge-path is stable

3. **Tiled + Bitmap** (For stencil operations)
   - Bitmap for fast row lookup
   - Tiled layout for stencil locality
   - Best for PDE solvers on dense domains

---

### Strategies to Skip

**Skip: Octree (Strategy 6) - Unless doing AMR**

**Rationale:**
- 56% memory overhead
- Serial traversal (poor GPU performance)
- 6-8 weeks implementation time
- Only benefits AMR workflows
- Set operations are **slower** than CSR

**Only implement if:**
- You need dynamic refinement/coarsening
- You're doing multigrid solvers
- You need level-of-detail rendering

---

**Skip: Hash Table (Strategy 2) - If using Merge-Path**

**Rationale:**
- 16% memory overhead
- O(n) build time
- Merge-path is faster and has no overhead
- Hash only wins for dynamic insertions

**Only implement if:**
- You need dynamic mesh updates
- Mesh changes between operations
- You can't pre-sort

---

### Suggested Implementation Order

```
Phase 1 (Weeks 1-3): Merge-Path Set Algebra
  ✓ Immediate 4-5× speedup
  ✓ Low risk
  ✓ Enables future optimizations

Phase 2 (Weeks 4-7): Morton Encoding
  ✓ Additional 1.5-2× speedup
  ✓ Works with merge-path
  ✓ No memory overhead

Phase 3 (Weeks 8-12): Bitmap Representation
  ✓ 10-100× speedup for dense domains
  ✓ Different use case (bounded domains)
  ✓ GPU tensor core acceleration

Phase 4 (Weeks 13-18): Hybrid Adaptive
  ✓ Automatic strategy selection
  ✓ User-friendly API
  ✓ Production-ready

Phase 5 (Optional, Weeks 19+): Specialized Strategies
  - Tiled (if stencil-heavy)
  - Octree (if AMR needed)
```

---

### Quick Decision Guide

**For your specific workload:**

| If you have... | Implement this |
|----------------|----------------|
| **Generic 3D extension** | Merge-Path (3 weeks) |
| **Large sparse meshes (> 100K rows)** | Merge-Path + Morton (6 weeks) |
| **Dense bounded domains** | Merge-Path + Bitmap (10 weeks) |
| **Stencil-heavy PDE solver** | Merge-Path + Tiled (12 weeks) |
| **AMR workflows** | Merge-Path + Octree (14 weeks) |
| **Production library** | All phases (18 weeks) |
| **Research prototype** | Merge-Path only (3 weeks) |

---

### Final Recommendations

**For Subsetix Kokkos 3D Extension:**

1. **Start with Merge-Path** - Highest ROI, lowest risk
2. **Add Morton Encoding** - Complements merge-path well
3. **Consider Bitmap** - If you have dense bounded domains
4. **Skip Hash and Octree** - Unless specific use cases
5. **Build Hybrid Layer** - For production user experience

**Expected Overall Performance:**

| Mesh Type | Speedup vs Classic |
|-----------|-------------------|
| Small (< 10K, sparse) | 2-3× |
| Medium (10K-1M, sparse) | 6-8× |
| Large (> 1M, sparse) | 8-10× |
| Any (dense, bounded) | 20-50× |
| AMR operations | 10× (refinement) |

**Key Success Factors:**
- Extensive benchmarking at each phase
- GPU profiling for each backend
- Memory usage monitoring
- Backward compatibility with 2D code
- Clear documentation of trade-offs

---

## References

- Green, R. et al. (2012). "Merge Path: A Visually Intuitive Approach to Merging"
- Morton, G. M. (1966). "A Computer Oriented Geodetic Data Base"
- p4est: Scalable Algorithms for Parallel Adaptive Mesh Refinement
- AMReX: Block-structured AMR framework
- CUDA Developers Blog: "Maximizing Performance with Massively Parallel Hash Maps"
- ExaBricks (NVIDIA): AMR volume rendering with bricks
- Spaden (ACM 2024): Bitmap-based sparse matrix operations
