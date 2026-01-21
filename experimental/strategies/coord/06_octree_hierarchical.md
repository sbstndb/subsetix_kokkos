# Octree/Hierarchical Strategy (AMR)

## Overview

**Octree-based hierarchical representation** for Adaptive Mesh Refinement (AMR) workflows.

## Current Problem

```cpp
// Current: Flat CSR representation
struct Mesh<3, MemorySpace> {
  Kokkos::View<RowKey3D*, MemorySpace> row_keys;
  // ... flat structure
};

// Problems for AMR:
// 1. No parent-child relationships
// 2. Refinement requires rebuilding entire mesh
// 3. Coarsening is expensive
// 4. No level-of-detail support
```

**Issues:**
- Expensive refinement/coarsening operations
- No natural hierarchy for multi-resolution
- Coarse-to-fine traversal is inefficient

## Proposed Solution

**Octree with Morton-encoded coordinates:**

```
Level 0: Root node (entire domain)
  - Morton code: 0

Level 1: 8 children (2×2×2 subdivision)
  - Morton codes: 0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111

Level 2: 64 grandchildren (4×4×4)
  - Morton codes: 0b000000, 0b000001, ...

Each node:
  - Stores Morton code prefix
  - Stores X-intervals for cells at this level
  - Pointers to children (if refined)
```

### Octree Node Structure

```cpp
struct OctreeNode {
  uint64_t morton_prefix;    // Morton code prefix for this node
  int8_t level;              // Tree level (0 = root)
  uint8_t child_mask;        // 8 bits: which children exist
  uint32_t first_child;      // Index of first child in node array
  uint32_t parent;           // Index of parent node

  // Leaf nodes store intervals
  std::size_t interval_offset;
  std::size_t interval_count;
};
```

### Morton Encoding for Hierarchy

```
Level L Morton code = prefix for all descendants

Example for 2D (simpler):
  Level 0: 0b______ (any 6-bit code)
  Level 1: 0b00____ (first quadrant), 0b01____ (second), ...
  Level 2: 0b0000__ (first sub-quadrant), ...

3D with 3 bits per level:
  Level 0: 0b_______________ (21 bits max)
  Level 1: 0b000_____________ (3 bits for octant)
  Level 2: 0b000000__________ (6 bits)
  Level 3: 0b000000000______ (9 bits)
  ...
```

## API Design

### Core Octree Structure

```cpp
namespace experimental::subsetix::csr::octree {

/**
 * @brief Octree node for 3D sparse hierarchical mesh.
 */
struct Node {
  uint64_t morton_code;     // Morton code for this node's position
  int8_t level;             // Tree level (0 = coarsest)
  uint8_t child_mask;       // Bitmask of existing children (0-7)
  uint32_t first_child;     // Index into nodes array for first child
  uint32_t parent_idx;      // Index of parent node (or -1 for root)

  // Leaf data (only for leaf nodes)
  std::size_t interval_offset;  // Offset into intervals array
  std::size_t interval_count;   // Number of intervals

  KOKKOS_INLINE_FUNCTION
  bool is_leaf() const {
    return child_mask == 0;
  }

  KOKKOS_INLINE_FUNCTION
  bool has_child(int child_index) const {
    return (child_mask >> child_index) & 1;
  }
};

/**
 * @brief Octree-based 3D mesh for AMR.
 */
template <class MemorySpace>
class Mesh3D {
public:
  using NodeView = Kokkos::View<Node*, MemorySpace>;
  using IntervalView = Kokkos::View<Interval*, MemorySpace>;

  NodeView nodes;           // [num_nodes] - All nodes in tree
  IntervalView intervals;   // [num_intervals] - X intervals for leaves

  std::size_t num_nodes = 0;
  std::size_t num_leaves = 0;
  std::size_t num_intervals = 0;
  int max_level = 0;

  /**
   * @brief Find node containing a given (x, y, z) coordinate.
   *
   * Traverses from root, selecting child based on Morton code bits.
   */
  KOKKOS_INLINE_FUNCTION
  const Node& find_node(Coord x, Coord y, Coord z) const {
    uint64_t target_morton = morton_encode_3d(x, y, z);
    std::size_t node_idx = 0;  // Start at root

    while (true) {
      const Node& node = nodes(node_idx);

      if (node.is_leaf()) {
        return node;
      }

      // Determine which child to traverse
      const int level = node.level;
      const int child_bits = (target_morton >> (3 * level)) & 0x7;

      if (!node.has_child(child_bits)) {
        return node;  // No child at this level
      }

      node_idx = node.first_child + count_children_before(node, child_bits);
    }
  }

  /**
   * @brief Refine a leaf node (split into 8 children).
   */
  void refine_leaf(std::size_t leaf_idx);

  /**
   * @brief Coarsen nodes (merge 8 siblings into parent).
   */
  void coarsen_siblings(std::size_t parent_idx);
};

} // namespace experimental::subsetix::csr::octree
```

### Morton Encoding for 3D

```cpp
/**
 * @brief Encode 3D coordinates to Morton code.
 */
KOKKOS_INLINE_FUNCTION
uint64_t morton_encode_3d(Coord x, Coord y, Coord z) {
  uint64_t result = 0;

  for (int i = 0; i < 21; ++i) {  // 21 bits per coordinate
    result |= ((static_cast<uint64_t>(x) & (1ULL << i)) << (2 * i)) |     // x at pos 0,3,6,...
              ((static_cast<uint64_t>(y) & (1ULL << i)) << (2 * i + 1)) |  // y at pos 1,4,7,...
              ((static_cast<uint64_t>(z) & (1ULL << i)) << (2 * i + 2));    // z at pos 2,5,8,...
  }

  return result;
}

/**
 * @brief Decode Morton code to 3D coordinates.
 */
KOKKOS_INLINE_FUNCTION
void morton_decode_3d(uint64_t code, Coord& x, Coord& y, Coord& z) {
  uint64_t x_bits = 0, y_bits = 0, z_bits = 0;

  for (int i = 0; i < 21; ++i) {
    x_bits |= ((code >> (2 * i)) & 1ULL) << i;
    y_bits |= ((code >> (2 * i + 1)) & 1ULL) << i;
    z_bits |= ((code >> (2 * i + 2)) & 1ULL) << i;
  }

  x = static_cast<Coord>(x_bits);
  y = static_cast<Coord>(y_bits);
  z = static_cast<Coord>(z_bits);
}
```

### Refinement Operation

```cpp
/**
 * @brief Refine a leaf node by splitting it into 8 children.
 *
 * Creates 8 new leaf nodes at level+1, each covering 1/8 of the parent's volume.
 */
template <class MemorySpace>
void Mesh3D<MemorySpace>::refine_leaf(std::size_t leaf_idx) {
  using ExecSpace = typename MemorySpace::execution_space;

  Node& parent = nodes(leaf_idx);

  if (!parent.is_leaf()) {
    return;  // Already refined
  }

  // Allocate 8 new nodes
  const std::size_t old_num_nodes = num_nodes;
  num_nodes += 8;

  NodeView new_nodes("nodes", num_nodes);
  Kokkos::deep_copy(new_nodes, nodes);  // Copy existing

  nodes = new_nodes;

  // Initialize children
  const uint64_t parent_morton = parent.morton_code;
  const int new_level = parent.level + 1;
  const std::size_t first_child = old_num_nodes;

  parent.first_child = static_cast<uint32_t>(first_child);
  parent.child_mask = 0xFF;  // All 8 children exist

  max_level = std::max(max_level, new_level);

  // Each child covers an octant of the parent
  for (int child = 0; child < 8; ++child) {
    Node& child_node = nodes(first_child + child);

    // Morton code: parent code + child index at next level
    child_node.morton_code = parent_morton | (static_cast<uint64_t>(child) << (3 * new_level));
    child_node.level = new_level;
    child_node.parent_idx = static_cast<uint32_t>(leaf_idx);
    child_node.child_mask = 0;  // Children start as leaves
    child_node.interval_offset = 0;
    child_node.interval_count = 0;
  }

  // Distribute parent intervals to children
  // ... (implementation based on interval positions)

  num_leaves += 7;  // 1 parent leaf → 8 child leaves
}
```

### Coarsening Operation

```cpp
/**
 * @brief Coarsen 8 sibling nodes into a single parent.
 */
template <class MemorySpace>
void Mesh3D<MemorySpace>::coarsen_siblings(std::size_t parent_idx) {
  Node& parent = nodes(parent_idx);

  if (parent.is_leaf()) {
    return;  // Already a leaf
  }

  // Check if all 8 children are leaves
  bool all_leaves = true;
  for (int c = 0; c < 8; ++c) {
    if (!parent.has_child(c)) {
      all_leaves = false;
      break;
    }
    const Node& child = nodes(parent.first_child + c);
    if (!child.is_leaf()) {
      all_leaves = false;
      break;
    }
  }

  if (!all_leaves) {
    return;  // Cannot coarsen
  }

  // Merge intervals from all children
  std::size_t total_intervals = 0;
  for (int c = 0; c < 8; ++c) {
    const Node& child = nodes(parent.first_child + c);
    total_intervals += child.interval_count;
  }

  // Allocate new interval array
  // ... (copy and merge intervals from children)

  // Mark parent as leaf
  parent.child_mask = 0;
  parent.interval_count = total_intervals;
  // ... set interval_offset

  // Children will be garbage-collected
  num_leaves -= 7;  // 8 child leaves → 1 parent leaf
}
```

## Performance Analysis

### Memory Overhead

| Component | CSR | Octree | Notes |
|-----------|-----|--------|-------|
| Row keys | 8 × R bytes | 8 × R bytes | Same (Node replaces RowKey) |
| Tree structure | - | ~32 × N bytes | N = num_nodes ≥ R |
| Interval storage | Same | Same | 0 |

Where R = number of leaf rows (actual cells), N = total nodes including internal

For balanced octree with R leaves:
- N ≈ R × (8/7) ≈ 1.14 × R
- Tree overhead: 32 × 0.14 × R ≈ 4.5 × R bytes
- **Total overhead: ~56% more memory** than CSR

### Traversal Complexity

| Operation | CSR | Octree |
|-----------|-----|--------|
| Point lookup | O(log R) | O(log R) |
| Refine cell | O(R) (rebuild) | O(1) (just add children) |
| Coarsen cells | O(R) (rebuild) | O(1) (merge siblings) |
| Coarse-to-fine traversal | N/A | O(R) (natural) |
| Fine-to-coarse traversal | N/A | O(R) (natural) |

**Octree enables O(1) refinement/coarsening** - critical for AMR!

### AMR-Specific Operations

```
Refinement workload (refine 10% of cells):
CSR:
- Copy entire mesh: O(R)
- Rebuild CSR structure: O(R log R)
- Update row_keys, row_ptr: O(R)

Octree:
- For each refined cell: O(1)
- Add 8 children, redistribute intervals
- Total: O(0.1 × R)

Speedup: ~10× for refinement

Coarsening workload (coarsen 10% of cells):
CSR: Same as refinement - O(R)
Octree: O(0.1 × R)

Speedup: ~10× for coarsening

Multi-level traversal (for multigrid):
CSR: Not supported - need separate meshes
Octree: O(R) - walk tree naturally

Speedup: Infinite (CSR can't do this!)
```

### Estimated Performance by Backend

#### Serial (CPU)

```
For 5M cells, 5 AMR levels:
Construction:
CSR: ~200 ms (build from scratch)
Octree: ~250 ms (build tree)

Refinement (refine 10% of cells):
CSR: ~150 ms (rebuild)
Octree: ~20 ms (add children)

Speedup: 7× for refinement
Break-even: ~2 refinement operations
```

#### OpenMP (CPU)

```
Benefits:
- Independent subtrees can be processed in parallel
- Tree traversal is mostly read-only
- Refinement of independent regions is parallel

For 5M cells, 16 threads:
Refinement (10% of cells):
CSR: ~150 ms / 16 = ~9 ms (with contention)
Octree: ~20 ms / 16 = ~1.25 ms (independent refinements)

Speedup: 7× for refinement
Scalability: Good (minimal contention)
```

#### CUDA (GPU)

```
Challenges:
- Tree traversal is serial (can't parallelize easily)
- Irregular memory access patterns
- Dynamic node allocation

Solutions:
- Batch refinement operations
- Use shared memory for subtrees
- Allocate nodes in blocks

For 5M cells:
Refinement (10% of cells):
CSR: ~50 ms (GPU rebuild)
Octree: ~30 ms (GPU refinement with batching)

Speedup: ~1.7× (less than CPU due to overhead)
Break-even: Multiple refinement operations

Note: Octree benefits diminish on GPU due to serial traversal
```

### Overhead for Small Meshes

| Mesh Size | CSR | Octree | Notes |
|-----------|-----|--------|-------|
| 1K cells | 0.1 ms | 0.15 ms | Tree overhead |
| 10K cells | 1 ms | 1.2 ms | Break-even at ~5K |
| 100K cells | 12 ms | 8 ms | Benefits start |
| 1M cells | 150 ms | 50 ms | 3× faster for AMR |

**Break-even point:** ~5K cells **with AMR operations**

Without AMR, octree has overhead due to tree structure.

## Kokkos Implementation

### Core Octree Types

```cpp
// experimental/include/experimental/subsetix/csr/octree/mesh.hpp

#pragma once

#include <Kokkos_Core.hpp>
#include <cstdint>

namespace experimental::subsetix::csr::octree {

using Coord = int32_t;

// Forward declarations
template <class MemorySpace> class Mesh3D;

struct Node {
  uint64_t morton_code = 0;
  int8_t level = 0;
  uint8_t child_mask = 0;
  uint32_t first_child = 0;
  uint32_t parent_idx = 0;  // Use UINT32_MAX for root

  // Leaf data
  std::size_t interval_offset = 0;
  std::size_t interval_count = 0;

  KOKKOS_INLINE_FUNCTION
  bool is_leaf() const { return child_mask == 0; }

  KOKKOS_INLINE_FUNCTION
  bool has_child(int idx) const {
    return (child_mask >> idx) & 1;
  }

  KOKKOS_INLINE_FUNCTION
  int child_index(std::size_t child_node_idx) const;
};

template <class MemorySpace>
class Mesh3D {
public:
  using NodeView = Kokkos::View<Node*, MemorySpace>;
  using IntervalView = Kokkos::View<Interval*, MemorySpace>;

  NodeView nodes;
  IntervalView intervals;

  std::size_t num_nodes = 0;
  std::size_t num_leaves = 0;
  std::size_t num_intervals = 0;
  int max_level = 0;

  KOKKOS_INLINE_FUNCTION
  Mesh3D() = default;

  // Core operations
  void build_root(int x_max, int y_max, int z_max);

  KOKKOS_INLINE_FUNCTION
  const Node& find_node(Coord x, Coord y, Coord z) const;

  void refine_leaf(std::size_t leaf_idx);
  void coarsen_siblings(std::size_t parent_idx);

  // Iteration
  template <typename Func>
  void for_each_leaf(Func&& func) const;
};

} // namespace experimental::subsetix::csr::octree
```

### Traversal Kernels

```cpp
/**
 * @brief Parallel traversal of all leaf nodes.
 */
template <class MemorySpace, typename Func>
KOKKOS_INLINE_FUNCTION
void for_each_leaf(const Mesh3D<MemorySpace>& mesh, Func&& func) {
  for (std::size_t i = 0; i < mesh.num_nodes; ++i) {
    const Node& node = mesh.nodes(i);
    if (node.is_leaf()) {
      func(node, i);
    }
  }
}

/**
 * @brief Coarse-to-fine traversal (for multigrid).
 */
template <class MemorySpace, typename Func>
void traverse_coarse_to_fine(const Mesh3D<MemorySpace>& mesh,
                             Func&& func) {
  // Visit nodes in level order
  for (int level = 0; level <= mesh.max_level; ++level) {
    Kokkos::parallel_for(
      "octree_level_traversal",
      Kokkos::RangePolicy<ExecSpace>(0, mesh.num_nodes),
      KOKKOS_LAMBDA(const std::size_t i) {
        const Node& node = mesh.nodes(i);
        if (node.level == level) {
          func(node, i);
        }
      });
    ExecSpace().fence();
  }
}
```

## Implementation Roadmap

### Phase 1: Core Octree (2 weeks)

- [ ] Implement `Node` structure
- [ ] Add `Mesh3D` octree class
- [ ] Implement Morton encoding/decoding for 3D
- [ ] Add basic tree building
- [ ] Unit tests

### Phase 2: Traversal (1 week)

- [ ] Implement `find_node`
- [ ] Add leaf iteration
- [ ] Add level-order traversal
- [ ] Test correctness

### Phase 3: AMR Operations (2 weeks)

- [ ] Implement `refine_leaf`
- [ ] Implement `coarsen_siblings`
- [ ] Add interval redistribution
- [ ] Test AMR workflows

### Phase 4: Set Operations (2 weeks)

- [ ] Octree-aware intersection
- [ ] Octree-aware union
- [ ] Multi-level operations
- [ ] Benchmark vs CSR

## Pros and Cons

### Pros

1. **O(1) refinement** - Just add child nodes
2. **O(1) coarsening** - Merge siblings
3. **Natural hierarchy** - Perfect for AMR
4. **Multi-level traversal** - Built-in level-of-detail
5. **Spatial locality** - Morton code ordering
6. **Octant queries** - Fast neighborhood searches

### Cons

1. **56% memory overhead** - Tree structure
2. **Complex implementation** - Hard to debug
3. **Serial traversal** - Not GPU-friendly
4. **Small mesh penalty** - Tree overhead
5. **Set operations slower** - Two-level merge

## When to Use

| Scenario | Recommended? |
|----------|--------------|
| AMR workflows | **Yes** - Primary use case |
| Static meshes | **No** - Overhead |
| Multi-level solvers | **Yes** - Natural hierarchy |
| Memory-constrained | **No** - 56% overhead |
| GPU execution | **Maybe** - Serial traversal hurts |
| Frequent refinement | **Yes** - 10× faster |

## Comparison with CSR

| Aspect | CSR | Octree |
|--------|-----|--------|
| **Memory** | 16 × R | 25 × R |
| **Refinement** | O(R) rebuild | O(1) per cell |
| **Coarsening** | O(R) rebuild | O(1) per group |
| **Set operations** | Fast | Slower (two-level) |
| **AMR support** | Poor | **Excellent** |
| **GPU friendly** | **Yes** | No (serial) |

**Conclusion:** Octree is specialized for AMR - use CSR for static meshes!

## References

- p4est: Scalable Algorithms for Parallel Adaptive Mesh Refinement
- NeuralVDB: Hierarchical sparse volumes
- Sparse Voxel Octrees (NVIDIA)
- AMReX: Block-structured AMR framework
