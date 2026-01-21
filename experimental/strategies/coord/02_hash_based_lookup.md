# Hash-Based Lookup Strategy

## Overview

Replace binary search with **O(1) hash table lookup** for row mapping in set operations.

## Current Problem

```cpp
// Current Phase 1: Binary search O(R_A * log R_B)
Kokkos::parallel_for(..., KOKKOS_LAMBDA(const std::size_t i) {
  const RowKey key = rows_a(i);
  const int idx_b = csr::detail::find_row_by_yz(rows_b, num_rows_b, key.y, key.z);
  // O(log R_B) per row - 2 comparisons per iteration
});
```

**Issues:**
- O(R_A × log R_B) complexity for row mapping
- Branch divergence from binary search on GPU
- Poor cache behavior (random access pattern)

## Proposed Solution

Use a **device-side hash table** for O(1) average-case row lookup:

```cpp
// Build hash table from B's rows
RowHashMap<MemorySpace> hash_b;
hash_b.build(rows_b, num_rows_b);

// Phase 1: O(R_A) lookups instead of O(R_A * log R_B)
Kokkos::parallel_for(..., KOKKOS_LAMBDA(const std::size_t i) {
  const RowKey key = rows_a(i);
  const int idx_b = hash_b.find(key);  // O(1) average!
});
```

## API Design

### Hash Table Structure

```cpp
namespace experimental::subsetix::csr::hash {

/**
 * @brief Device-friendly hash table for RowKey3D -> row index mapping.
 *
 * Uses open addressing with linear probing for GPU compatibility.
 */
template <class MemorySpace>
class RowHashMap {
public:
  using Key = RowKey3D;
  using Value = int32_t;  // Row index

  // Hash table storage
  Kokkos::View<Key*, MemorySpace> keys;      // [capacity]
  Kokkos::View<Value*, MemorySpace> values;   // [capacity]
  Kokkos::View<uint8_t*, MemorySpace> occupied;  // [capacity] - boolean

  std::size_t capacity = 0;
  std::size_t size = 0;
  float max_load_factor = 0.7f;

  // Build from row keys array
  void build(const Kokkos::View<Key*, MemorySpace>& row_keys,
             std::size_t num_rows);

  // Lookup (device-side)
  KOKKOS_INLINE_FUNCTION
  Value find(const Key& key) const {
    std::size_t idx = hash_function(key) % capacity;

    // Linear probing
    while (occupied(idx) && keys(idx) != key) {
      idx = (idx + 1) % capacity;
      if (idx == hash_function(key) % capacity) {
        return -1;  // Not found (table full)
      }
    }

    return occupied(idx) ? values(idx) : -1;
  }
};

} // namespace experimental::subsetix::csr::hash
```

### Hash Function Options

```cpp
// Option 1: Simple linear hash
KOKKOS_INLINE_FUNCTION
std::size_t hash_simple(const RowKey3D& key) {
  // Combine y and z into single value
  uint64_t combined = (static_cast<uint64_t>(key.y) << 32) |
                      (static_cast<uint64_t>(key.z) & 0xFFFFFFFF);
  return combined % TABLE_SIZE;
}

// Option 2: Morton-based hash (better spatial locality)
KOKKOS_INLINE_FUNCTION
std::size_t hash_morton(const RowKey3D& key) {
  uint64_t morton = morton_encode_2d(key.y, key.z);
  return morton % TABLE_SIZE;
}

// Option 3: Multiplicative hash (better distribution)
KOKKOS_INLINE_FUNCTION
std::size_t hash_multiply(const RowKey3D& key) {
  const uint64_t GOLDEN_RATIO = 0x9E3779B97F4A7C15ULL;
  uint64_t combined = (static_cast<uint64_t>(key.y) << 32) | key.z;
  return (combined * GOLDEN_RATIO) % TABLE_SIZE;
}

// Option 4: FNV-1a hash
KOKKOS_INLINE_FUNCTION
std::size_t hash_fnv1a(const RowKey3D& key) {
  const uint64_t FNV_OFFSET = 14695981039346656037ULL;
  const uint64_t FNV_PRIME = 1099511628211ULL;

  uint64_t hash = FNV_OFFSET;
  hash ^= static_cast<uint64_t>(key.y);
  hash *= FNV_PRIME;
  hash ^= static_cast<uint64_t>(key.z);
  hash *= FNV_PRIME;

  return hash % TABLE_SIZE;
}
```

### Collision Resolution

```cpp
// Linear probing (GPU-friendly, cache-coherent)
KOKKOS_INLINE_FUNCTION
int find_linear_probe(const RowKey3D* keys, const uint8_t* occupied,
                      const int* values, std::size_t capacity,
                      const RowKey3D& key) {
  std::size_t idx = hash_multiply(key) % capacity;
  std::size_t start_idx = idx;

  do {
    if (!occupied[idx]) {
      return -1;  // Not found
    }
    if (keys[idx] == key) {
      return values[idx];  // Found
    }
    idx = (idx + 1) % capacity;
  } while (idx != start_idx);

  return -1;  // Table searched, not found
}

// Cuckoo hashing (worst-case O(1), but more complex)
template <std::size_t NUM_TABLES = 2>
class CuckooHashMap {
  // Each key has NUM_TABLES possible positions
  // On collision, kick existing entry to its alternate position
};
```

### Set Intersection with Hash

```cpp
template <class MemorySpace>
Mesh<3, MemorySpace>
intersect_hash(const Mesh<3, MemorySpace>& A,
               const Mesh<3, MemorySpace>& B) {
  using ExecSpace = typename MemorySpace::execution_space;

  // Phase 1: Build hash table from B (O(R_B))
  RowHashMap<MemorySpace> hash_b;
  hash_b.build(B.row_keys, B.num_rows);

  // Phase 2: Find matching rows (O(R_A) with O(1) lookups)
  Kokkos::View<int*, MemorySpace> matches_a("matches_a", A.num_rows);
  Kokkos::View<int*, MemorySpace> matches_b("matches_b", A.num_rows);

  Kokkos::parallel_for(
    "hash_intersection_phase1",
    Kokkos::RangePolicy<ExecSpace>(0, A.num_rows),
    KOKKOS_LAMBDA(const std::size_t i) {
      const auto key = A.row_keys(i);
      const int idx_b = hash_b.find(key);
      matches_a(i) = idx_b >= 0 ? static_cast<int>(i) : -1;
      matches_b(i) = idx_b;
    });

  // Phase 3: Compact matches
  // ... (same as current implementation)

  // Phase 4-5: Interval intersection (unchanged)
  // ...
}
```

## Performance Analysis

### Memory Overhead

| Component | Current CSR | Hash Table | Overhead |
|-----------|------------|------------|----------|
| row_keys | 8 × R bytes | 8 × H bytes | |
| row_ptr | 8 × R bytes | - | |
| Hash table overhead | - | 8 × H + 1 × H + 4 × H | 13 × H bytes |
| **Total** | 16 × R bytes | 13 × H bytes | |

Where R = number of rows, H = hash table capacity = R / load_factor

For load_factor = 0.7:
- H = R / 0.7 ≈ 1.43 × R
- Hash memory: 13 × 1.43 × R ≈ 18.6 × R bytes
- **Overhead: ~16% more memory** than CSR

### Time Complexity

| Operation | Current (Binary Search) | Hash Table | Speedup |
|-----------|------------------------|------------|---------|
| Build (from B) | O(1) - already sorted | O(R_B) | - |
| Row mapping | O(R_A × log R_B) | O(R_A) | **log R_B×** |
| Lookup (single) | O(log R_B) | O(1) avg | ~20× for 5M rows |
| Worst case | O(log R_B) | O(R_B) (degenerate) | - |

For 5M rows:
- Binary search: 23 comparisons per lookup
- Hash: 1-3 probes per lookup (average)

### Estimated Performance by Backend

#### Serial (CPU)

```
Benefits:
- O(R_A) vs O(R_A × log R_B) for row mapping
- Better cache behavior for hash table access

Costs:
- Hash table build time: O(R_B)
- Memory overhead

For 5M rows, mesh intersection:
Current: ~200 ms
Hash: ~150 ms (25% faster, assuming 50ms build time)

Break-even: ~50K rows
```

#### OpenMP (CPU)

```
Benefits:
- Parallel hash table building
- O(1) lookups scale linearly
- Less branch prediction penalty

For 5M rows:
Current: ~50 ms
Hash: ~30 ms (40% faster)

Scalability: Near-linear with thread count
```

#### CUDA (GPU)

```
Benefits:
- Reduced warp divergence (no binary search)
- Coalesced memory access to hash table
- O(R_A) parallel work vs O(R_A × log R_B)

Costs:
- Atomic operations during build
- Potential load imbalance (some threads probe more)

For 5M rows:
Current: ~20 ms
Hash: ~8 ms (60% faster)

Warp efficiency:
Current: ~60% (binary search divergence)
Hash: ~80% (short probe sequences)
```

### Overhead for Small Meshes

| Mesh Size | Current | Hash | Notes |
|-----------|---------|------|-------|
| 1K rows | 0.05 ms | 0.15 ms | 3× slower (build dominates) |
| 5K rows | 0.2 ms | 0.25 ms | 1.25× slower |
| 10K rows | 0.5 ms | 0.4 ms | **Break-even** |
| 50K rows | 4 ms | 2 ms | 2× faster |
| 500K+ rows | 50 ms | 15 ms | 3× faster |

**Break-even point:** ~10K rows

For small meshes (< 10K rows), hash table build time exceeds the benefit.

## Kokkos Implementation

### Core Hash Table

```cpp
// experimental/include/experimental/subsetix/csr/hash/row_hash_map.hpp

#pragma once

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <experimental/subsetix/csr/mesh.hpp>

namespace experimental::subsetix::csr::hash {

template <class MemorySpace>
class RowHashMap {
public:
  using Key = RowKey3D;
  using Value = int32_t;
  using KeyView = Kokkos::View<Key*, MemorySpace>;
  using ValueView = Kokkos::View<Value*, MemorySpace>;
  using OccupiedView = Kokkos::View<uint8_t*, MemorySpace>;

  KeyView keys;
  ValueView values;
  OccupiedView occupied;

  std::size_t capacity = 0;
  std::size_t size = 0;
  float max_load_factor = 0.7f;

  KOKKOS_INLINE_FUNCTION
  RowHashMap() = default;

  /**
   * @brief Build hash table from row keys array.
   */
  void build(const KeyView& row_keys, std::size_t num_rows);

  /**
   * @brief Find row index for given key (O(1) average).
   */
  KOKKOS_INLINE_FUNCTION
  Value find(const Key& key) const {
    if (capacity == 0) return -1;

    std::size_t idx = hash_function(key) % capacity;
    const std::size_t start_idx = idx;

    do {
      if (!occupied(idx)) {
        return -1;  // Not found
      }
      if (keys(idx) == key) {
        return values(idx);  // Found
      }
      idx = (idx + 1) % capacity;
    } while (idx != start_idx);

    return -1;  // Searched entire table
  }

private:
  KOKKOS_INLINE_FUNCTION
  std::size_t hash_function(const Key& key) const {
    // Multiplicative hash with golden ratio
    const uint64_t GOLDEN_RATIO = 0x9E3779B97F4A7C15ULL;
    uint64_t combined = (static_cast<uint64_t>(key.y) << 32) |
                        (static_cast<uint64_t>(key.z) & 0xFFFFFFFFULL);
    return static_cast<std::size_t>((combined * GOLDEN_RATIO) % capacity);
  }
};

// Build implementation (host-side)
template <class MemorySpace>
void RowHashMap<MemorySpace>::build(const KeyView& row_keys,
                                     std::size_t num_rows) {
  using ExecSpace = typename MemorySpace::execution_space;

  size = num_rows;
  capacity = static_cast<std::size_t>(num_rows / max_load_factor) + 1;

  // Allocate storage
  keys = KeyView("hash_keys", capacity);
  values = ValueView("hash_values", capacity);
  occupied = OccupiedView("hash_occupied", capacity);

  // Initialize to empty
  Kokkos::deep_copy(occupied, uint8_t(0));

  if (num_rows == 0) return;

  // Insert each key
  Kokkos::parallel_for(
    "hash_build",
    Kokkos::RangePolicy<ExecSpace>(0, num_rows),
    KOKKOS_LAMBDA(const std::size_t i) {
      const Key key = row_keys(i);
      const Value value = static_cast<Value>(i);

      std::size_t idx = 0;
      // Compute hash (need access to capacity)
      // ... (simplified - actual implementation needs care)

      // Linear probing to find empty slot
      while (Kokkos::atomic_compare_exchange(&occupied(idx), 0, 1) != 0) {
        idx = (idx + 1) % capacity;
      }

      keys(idx) = key;
      values(idx) = value;
    });

  ExecSpace().fence();
}

} // namespace experimental::subsetix::csr::hash
```

### Set Intersection with Hash

```cpp
// experimental/include/experimental/subsetix/csr/hash/set_algebra.hpp

template <class MemorySpace>
Mesh<3, MemorySpace>
intersect_meshes_hash(const Mesh<3, MemorySpace>& A,
                      const Mesh<3, MemorySpace>& B) {
  using ExecSpace = typename MemorySpace::execution_space;
  using HashMap = RowHashMap<MemorySpace>;

  if (A.num_rows == 0 || B.num_rows == 0) {
    return Mesh<3, MemorySpace>{};
  }

  // Phase 1: Build hash table from B
  HashMap hash_b;
  hash_b.build(B.row_keys, B.num_rows);

  // Phase 2: Find matching rows via hash lookup
  constexpr int INVALID = -1;
  Kokkos::View<int*, MemorySpace> match_a("match_a", A.num_rows);
  Kokkos::View<int*, MemorySpace> match_b("match_b", A.num_rows);
  Kokkos::View<int*, MemorySpace> flags("flags", A.num_rows);

  Kokkos::parallel_for(
    "hash_intersection_row_map",
    Kokkos::RangePolicy<ExecSpace>(0, A.num_rows),
    KOKKOS_LAMBDA(const std::size_t i) {
      const auto key = A.row_keys(i);
      const int idx_b = hash_b.find(key);
      match_a(i) = (idx_b >= 0) ? static_cast<int>(i) : INVALID;
      match_b(i) = idx_b;
      flags(i) = (idx_b >= 0) ? 1 : 0;
    });

  ExecSpace().fence();

  // Phase 3: Scan to compact matches
  // ... (same as current implementation)

  // Phase 4-5: Interval intersection
  // ... (unchanged)
}
```

## Implementation Roadmap

### Phase 1: Basic Hash Table (1-2 weeks)

- [ ] Implement `RowHashMap` with linear probing
- [ ] Add multiplicative hash function
- [ ] Implement `build()` method with atomics
- [ ] Add unit tests for correctness

### Phase 2: Set Integration (1 week)

- [ ] Modify `intersect_meshes` to use hash
- [ ] Handle edge cases (empty meshes, duplicates)
- [ ] Benchmark against binary search

### Phase 3: Optimization (2 weeks)

- [ ] Try cuckoo hashing for better worst-case
- [ ] Optimize hash function (Morton-based?)
- [ ] Resize/rehash strategy
- [ ] GPU warp-level primitives

### Phase 4: Advanced Features (optional)

- [ ] Concurrent insert support (AMR)
- [ ] Delete operations
- [ ] Rehash support for dynamic growth
- [ ] Perfect hash for static geometries

## Pros and Cons

### Pros

1. **O(1) average lookup** - Much faster than binary search
2. **No branch divergence** - Linear probing is predictable
3. **Scales well** - Linear in number of rows, not log-linear
4. **GPU-friendly** - Short probe sequences, coalesced access
5. **Flexible** - Works with unsorted data

### Cons

1. **Memory overhead** - ~16% more memory
2. **Build time** - O(R) upfront cost
3. **Worst-case O(R)** - Can degenerate with bad hash
4. **Not ordered** - Can't iterate in sorted order
5. **Small mesh penalty** - Build overhead for small datasets

## When to Use

| Scenario | Recommended? |
|----------|--------------|
| Small meshes (< 10K rows) | **No** - build overhead |
| Large sparse meshes | **Yes** - 2-3× speedup |
| Set-operation heavy | **Yes** - O(1) lookups |
| Memory-constrained | **No** - 16% overhead |
| AMR with frequent updates | **Maybe** - need rehash |
| One-time operations | **Maybe** - consider build cost |

## Comparison with Other Strategies

| Strategy | Lookup | Memory | Build | Best For |
|----------|--------|--------|-------|----------|
| **Current (Binary Search)** | O(log n) | Low | None | Small meshes |
| **Morton Encoding** | O(log n) | Low | O(n log n) | Large sparse |
| **Hash Table** | O(1) | Medium | O(n) | Set operations |
| **Bitmap** | O(1) | High* | O(1) | Dense bounded |

*Bitmap is memory-efficient for dense, high-overhead for sparse

## References

- Knuth, T. (1998). "The Art of Computer Programming, Vol. 3"
- CUDA Developers Blog: "Maximizing Performance with Massively Parallel Hash Maps"
- CUB: CUDA Unbound library - device-wide primitives
- arXiv:2406.09255 (2024) - Lockless GPU hash table with cuckoo hashing
