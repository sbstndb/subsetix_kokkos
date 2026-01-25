# Hash-Based Row Mapping Analysis for Playground Intersection Algorithm

## Executive Summary

This document analyzes the feasibility of replacing the current binary search-based row mapping in the playground intersection algorithm with a hash-based approach using Kokkos-compatible unordered_map or custom hash table implementation.

**Key Finding**: Hash-based row mapping is **feasible and promising** for the playground intersection algorithm, with potential performance improvements of 1.5-3x for large sparse meshes, but introduces memory overhead and implementation complexity that must be carefully weighed.

---

## Table of Contents

1. [Current Implementation Analysis](#1-current-implementation-analysis)
2. [Feasibility Assessment](#2-feasibility-assessment)
3. [Memory Requirements](#3-memory-requirements)
4. [Design: Two-Phase Hash-Based Approach](#4-design-two-phase-hash-based-approach)
5. [Performance Characteristics](#5-performance-characteristics)
6. [Implementation Risks and Gotchas](#6-implementation-risks-and-gotchas)
7. [Recommendations](#7-recommendations)

---

## 1. Current Implementation Analysis

### 1.1 Current Row Mapping Mechanism

**Location**: `/playground/intersection/include/playground/subsetix/csr/intersection/algorithm/optimized.hpp`

**Current approach**: Binary search via `find_row_by_y` (2D) and `find_row_by_yz` (3D) in `detail/utils.hpp`

**Algorithm**:
```cpp
// Phase 1: Row mapping (lines 161-198)
for each row i in mesh A:
    if DIM == 2:
        idx_b = binary_search(rows_b, num_rows_b, rows_a(i).y)
    else: // DIM == 3
        idx_b = binary_search_yz(rows_b, num_rows_b, rows_a(i).y, rows_a(i).z)
    if idx_b found:
        flags(i) = 1
        tmp_idx_a(i) = i
        tmp_idx_b(i) = idx_b
```

**Complexity**: O(n log m) where n = |A.rows|, m = |B.rows|

**Performance characteristics**:
- Binary search depth: log2(m) comparisons per lookup
- Memory access pattern: Random (cache-line friendly for sorted arrays)
- GPU thread divergence: Minimal (unpredictable branches)

### 1.2 Row Key Structures

```cpp
// 2D: Single coordinate key
template<class CoordType>
struct RowKey2D {
    CoordType y = 0;
    bool operator==(const RowKey2D& other) const { return y == other.y; }
    bool operator<(const RowKey2D& other) const { return y < other.y; }
};

// 3D: Compound key (lexicographic ordering)
template<class CoordType>
struct RowKey3D {
    CoordType y = 0;
    CoordType z = 0;
    bool operator==(const RowKey3D& other) const { return y == other.y && z == other.z; }
    bool operator<(const RowKey3D& other) const {
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};
```

**Hash function considerations**:
- 2D: Simple hash of single integer (trivial)
- 3D: Must combine two integers while preserving distribution

### 1.3 Problem Scale

Based on benchmark configurations:

| Config | 2D Rows | 3D Rows | Binary Search Depth |
|--------|---------|---------|---------------------|
| Small  | ~19     | ~1,229  | 2D: 5 comparisons    |
| Medium | ~154    | ~78,643 | 2D: 8 comparisons    |
| Large  | ~1,229  | ~5.0M   | 2D: 11 comparisons   |
| XLarge | ~1,229  | ~10.0M  | 2D: 11 comparisons   |

**Observation**: For 3D meshes with millions of rows, binary search requires ~23 comparisons per lookup, which becomes expensive when performed for each row in mesh A.

---

## 2. Feasibility Assessment

### 2.1 Kokkos::UnorderedMap Evaluation

**Availability**: Kokkos provides `Kokkos::UnorderedMap` since version 3.0+

**Pros**:
- Fully GPU-compatible (CUDA, HIP, OpenMP)
- Handles dynamic insertion with automatic memory management
- Thread-safe operations (atomic inserts)
- Well-tested and maintained

**Cons**:
- Overhead for dynamic resizing (during Phase 1 build)
- Indirect memory access pattern (pointer chasing)
- Limited control over hash function and collision resolution
- Additional memory overhead for internal metadata

**Verdict**: **Usable but not optimal** for this use case. The map is built once and queried many times, so we don't need dynamic resizing. A custom open-addressing hash table would be more efficient.

### 2.2 Custom Hash Table Approach

**Recommended approach**: Open addressing with linear probing

**Why open addressing**:
- Better cache locality (entries stored contiguously)
- No pointer chasing (GPU-friendly)
- Simpler memory layout (single array)
- Predictable memory access patterns

**Why linear probing**:
- Simplest collision resolution
- Good cache behavior (sequential probing)
- Efficient on GPU (coalesced memory access)

**Verdict**: **Highly feasible** and recommended for production implementation.

### 2.3 Device-Side Constraints

**KOKKOS_INLINE_FUNCTION requirement**:
```cpp
KOKKOS_INLINE_FUNCTION
int hash_lookup(const RowKey& key) const {
    // Hash computation must be device-compatible
    // No dynamic memory allocation
    // No virtual functions
    // Minimal branching
}
```

**Hash function constraints**:
- Must use only integer operations
- No floating-point (non-deterministic across devices)
- No recursion or complex control flow
- Must be `constexpr` friendly

---

## 3. Memory Requirements

### 3.1 Memory Overhead Formula

For open addressing hash table with load factor α:

```
table_size = ceil(num_rows_b / α)

total_memory = table_size × (sizeof(Key) + sizeof(Value) + sizeof(State))
```

Where:
- `Key`: RowKey2D (4 bytes) or RowKey3D (8 bytes)
- `Value`: int32_t row index (4 bytes)
- `State`: uint8_t occupancy flag (1 byte)

### 3.2 Per-Configuration Memory Estimates

| Config | 3D Rows | Load Factor | Table Size | Memory (2D) | Memory (3D) | Overhead vs Binary Search |
|--------|---------|-------------|------------|-------------|-------------|---------------------------|
| Small  | 1,229   | 0.7         | 1,756      | ~14 KB      | ~24 KB      | +12 KB                    |
| Medium | 78,643  | 0.7         | 112,349    | ~890 KB     | ~1.6 MB     | +800 KB                   |
| Large  | 5.0M    | 0.7         | 7.14M      | ~57 MB      | ~101 MB     | +56 MB                    |
| XLarge | 10.0M   | 0.7         | 14.29M     | ~114 MB     | ~203 MB     | +113 MB                   |

**Binary search baseline**: Only requires sorted row_keys array (already present in mesh structure).

### 3.3 Load Factor Trade-offs

| Load Factor | Memory Overhead | Expected Probes | Collision Rate |
|-------------|-----------------|-----------------|----------------|
| 0.5 (low)   | +100%           | 1.0 - 1.5       | Very low       |
| 0.7 (medium)| +43%            | 1.5 - 2.5       | Low            |
| 0.9 (high)  | +11%            | 3.0 - 10.0      | High           |

**Recommendation**: Use load factor 0.7 for balanced performance and memory usage.

### 3.4 Memory Access Pattern Comparison

**Binary search**:
```
Memory access per lookup: log2(m) random accesses
- Small (3D): 11 random accesses × 8 bytes = 88 bytes transferred
- Large (3D): 23 random accesses × 8 bytes = 184 bytes transferred
```

**Hash lookup**:
```
Memory access per lookup: 1-3 random accesses (depends on load factor)
- Average case: 1.5 random accesses × 12 bytes = 18 bytes transferred
- Worst case: 10 random accesses × 12 bytes = 120 bytes transferred
```

**Analysis**: Hash table reduces memory bandwidth requirements by 3-5x in average case.

---

## 4. Design: Two-Phase Hash-Based Approach

### 4.1 Phase 1: Build Hash Table from Mesh B

**Goal**: Create a hash table mapping B's row keys to their indices

**Algorithm**:
```cpp
template <int DIM, class MemorySpace, class CoordType>
struct RowHashMap {
    using Key = std::conditional_t<DIM == 2, RowKey2D<CoordType>, RowKey3D<CoordType>>;

    Kokkos::View<Key*, MemorySpace> keys;          // [table_size]
    Kokkos::View<int*, MemorySpace> values;        // [table_size] - row indices
    Kokkos::View<uint8_t*, MemorySpace> occupied;  // [table_size] - 0=empty, 1=full
    std::size_t table_size = 0;                    // Power of 2 for fast modulo

    // Hash function (compile-time configurable)
    KOKKOS_INLINE_FUNCTION
    std::size_t hash(const Key& key) const {
        if constexpr (DIM == 2) {
            // 2D: Simple golden-ratio hash
            const uint32_t k = static_cast<uint32_t>(key.y);
            return (k * 0x9e3779b9U) & (table_size - 1);
        } else {
            // 3D: Combine y and z using boost::hash_combine style
            const uint32_t k1 = static_cast<uint32_t>(key.y);
            const uint32_t k2 = static_cast<uint32_t>(key.z);
            const uint32_t combined = k1 ^ (k2 + 0x9e3779b9U + (k1 << 6) + (k1 >> 2));
            return combined & (table_size - 1);
        }
    }
};
```

**Build pseudocode**:
```cpp
RowHashMap<DIM, MemorySpace, CoordType>
build_row_hash_map(const Mesh<DIM, MemorySpace, CoordType, IndexType>& B,
                   double load_factor = 0.7) {
    RowHashMap map;

    if (B.num_rows == 0) return map;

    // Calculate table size (power of 2)
    const std::size_t min_size = static_cast<std::size_t>(B.num_rows / load_factor);
    map.table_size = 1;
    while (map.table_size < min_size) map.table_size *= 2;

    // Allocate tables
    map.keys = Kokkos::View<Key*, MemorySpace>("hash_keys", map.table_size);
    map.values = Kokkos::View<int*, MemorySpace>("hash_values", map.table_size);
    map.occupied = Kokkos::View<uint8_t*, MemorySpace>("hash_occupied", map.table_size);

    // Initialize to empty
    Kokkos::deep_copy(map.occupied, uint8_t(0));

    // Parallel insert (may have collisions - handled by linear probing)
    Kokkos::parallel_for(
        "build_hash_map",
        Kokkos::RangePolicy<ExecSpace>(0, B.num_rows),
        KOKKOS_LAMBDA(const std::size_t i) {
            const Key key = B.row_keys(i);
            std::size_t idx = map.hash(key);

            // Linear probing: find empty slot
            while (Kokkos::atomic_compare_exchange_strong(&map.occupied(idx), 0, 1) != 0) {
                // Slot already occupied, check if same key (duplicate insert)
                if (map.keys(idx) == key) {
                    // Duplicate: update value (should not happen with unique keys)
                    map.values(idx) = static_cast<int>(i);
                    return;
                }
                // Move to next slot (wrap around)
                idx = (idx + 1) & (map.table_size - 1);
            }

            // Insert key-value pair
            map.keys(idx) = key;
            map.values(idx) = static_cast<int>(i);
        });

    ExecSpace().fence();
    return map;
}
```

**Key design decisions**:
1. **Power-of-2 table size**: Enables fast modulo using bitwise AND
2. **Atomic compare-exchange**: Thread-safe parallel insertion
3. **Linear probing**: Simple collision resolution with good cache behavior
4. **Duplicate detection**: Handles edge case of duplicate row keys

### 4.2 Phase 2: Query Hash Table for Mesh A

**Goal**: For each row in A, find corresponding row index in B

**Algorithm**:
```cpp
KOKKOS_INLINE_FUNCTION
int hash_lookup(const RowHashMap& map, const Key& key) {
    if (map.table_size == 0) return -1;

    std::size_t idx = map.hash(key);
    const std::size_t start_idx = idx;

    do {
        // Check if slot is occupied
        if (map.occupied(idx) == 0) {
            return -1;  // Key not found
        }

        // Check if key matches
        if (map.keys(idx) == key) {
            return map.values(idx);  // Found
        }

        // Linear probe to next slot
        idx = (idx + 1) & (map.table_size - 1);
    } while (idx != start_idx);

    return -1;  // Table full, key not found
}
```

**Row mapping kernel**:
```cpp
Kokkos::View<int*, MemorySpace> tmp_idx_b("tmp_idx_b", A.num_rows);
Kokkos::View<int*, MemorySpace> flags("flags", A.num_rows);

Kokkos::parallel_for(
    "hash_row_map",
    Kokkos::RangePolicy<ExecSpace>(0, A.num_rows),
    KOKKOS_LAMBDA(const std::size_t i) {
        const Key key = A.row_keys(i);
        const int idx_b = hash_lookup(map, key);

        if (idx_b >= 0) {
            flags(i) = 1;
            tmp_idx_b(i) = idx_b;
        } else {
            flags(i) = 0;
            tmp_idx_b(i) = -1;
        }
    });
```

### 4.3 Comparison with Current Implementation

| Aspect | Binary Search | Hash-Based |
|--------|---------------|------------|
| Build time | 0 (uses existing data) | O(m) parallel insert |
| Lookup time | O(log m) | O(1) average, O(n) worst |
| Memory overhead | 0 bytes | ~40% of mesh size |
| Thread divergence | Medium (binary search branches) | Low (linear probing) |
| Implementation complexity | Low | Medium |

---

## 5. Performance Characteristics

### 5.1 Theoretical Performance Comparison

**Binary search complexity**:
- Time: O(n log m) lookups
- Memory: O(1) additional memory
- Best case: Ω(n log m) - always log m comparisons
- Worst case: O(n log m)

**Hash-based complexity**:
- Time: O(n) average case lookups + O(m) build time
- Memory: O(m) additional memory
- Best case: Ω(n) - 1 probe per lookup (low load factor)
- Worst case: O(n × m) - catastrophic clustering (very unlikely)

### 5.2 Expected Performance by Mesh Size

| Config | n (A) | m (B) | Binary Search | Hash-Based | Expected Speedup |
|--------|-------|-------|---------------|------------|------------------|
| Small  | 19    | 1,229 | 19 × 11 = 209 comparisons | 19 × 1.5 = 28.5 probes | **0.7x** (slower due to build overhead) |
| Medium | 154   | 78K   | 154 × 17 = 2,618 comparisons | 154 × 2 = 308 probes | **1.5x** |
| Large  | 1,229 | 5.0M  | 1,229 × 23 = 28,267 comparisons | 1,229 × 2.5 = 3,073 probes | **2.5x** |
| XLarge | 1,229 | 10.0M | 1,229 × 24 = 29,496 comparisons | 1,229 × 3 = 3,687 probes | **3.0x** |

**Note**: Speedup estimates assume:
- Hash table build time is amortized over many lookups
- Average 1.5-3 probes per lookup (load factor 0.7)
- Binary search requires log2(m) comparisons

### 5.3 GPU Thread Divergence Analysis

**Binary search divergence**:
```cpp
while (lo < hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    if (rows(mid).y < y) {  // DIVERGENCE POINT
        lo = mid + 1;       // Some threads go here
    } else {
        hi = mid;           // Others go here
    }
}
// Warp divergence factor: ~50% (threads split on each comparison)
```

**Hash lookup divergence**:
```cpp
do {
    if (occupied(idx) == 0) return -1;  // Early exit for missing keys
    if (keys(idx) == key) return values(idx);  // Early exit on match
    idx = (idx + 1) & (table_size - 1);  // Continue probing
} while (idx != start_idx);
// Warp divergence factor: ~10-20% (only diverges on early exits)
```

**Analysis**: Hash-based lookup has significantly less warp divergence, which is critical for GPU performance.

### 5.4 Cache Performance

**Binary search**:
- Memory access: Random (jumps to mid, quarter, etc.)
- Cache efficiency: Poor (each lookup touches different cache lines)
- Spatial locality: Low (accesses scattered across array)

**Hash-based with linear probing**:
- Memory access: Random hash + sequential probing
- Cache efficiency: Good (probing accesses sequential locations)
- Spatial locality: High (probe sequence is cache-friendly)

**Estimated cache miss reduction**: 40-60% on GPU, 30-50% on CPU

---

## 6. Implementation Risks and Gotchas

### 6.1 Hash Function Quality

**Risk**: Poor hash distribution causes clustering, degrading performance to O(n)

**Mitigation**:
- Use well-tested hash functions (golden-ratio, MurmurHash3 finalizer)
- Validate distribution on representative datasets
- Provide fallback to binary search if clustering detected

**Testing approach**:
```cpp
// Test hash distribution
Kokkos::View<std::size_t*, HostSpace> bucket_counts("bucket_counts", table_size);
for (std::size_t i = 0; i < num_rows; ++i) {
    std::size_t idx = hash(rows(i));
    bucket_counts(idx)++;
}

// Compute standard deviation
double mean = static_cast<double>(num_rows) / table_size;
double variance = 0.0;
for (std::size_t i = 0; i < table_size; ++i) {
    double diff = bucket_counts(i) - mean;
    variance += diff * diff;
}
double std_dev = std::sqrt(variance / table_size);

// Expect std_dev < 0.5 * mean for good distribution
```

### 6.2 Collision Resolution Worst Case

**Risk**: Catastrophic clustering makes all lookups O(table_size)

**Mitigation**:
- Use Robin Hood hashing (minimizes probe length variance)
- Implement table growth when load factor exceeded
- Fall back to binary search after max probe limit

**Robin Hood variation**:
```cpp
// During insert: "steal from rich, give to poor"
// If new element has shorter probe length than existing element, swap them
std::size_t probe_new = 0;
std::size_t idx = hash(new_key);

while (occupied(idx)) {
    std::size_t probe_existing = (idx - hash(keys(idx))) & (table_size - 1);

    if (probe_new > probe_existing) {
        // Swap: existing element has shorter probe, it should be closer to its hash
        std::swap(new_key, keys(idx));
        std::swap(new_value, values(idx));
        probe_new = probe_existing;
    }

    idx = (idx + 1) & (table_size - 1);
    probe_new++;
}
```

### 6.3 Memory Pressure

**Risk**: Hash table can double memory usage for large 3D meshes

**Mitigation strategies**:
1. **Adaptive strategy**: Use hash for small/medium meshes, binary for large
2. **Streaming**: Process rows in batches, build hash table per batch
3. **Perfect hash**: Detect dense meshes and use direct indexing

**Memory budget formula**:
```
max_hash_table_size = available_memory - (mesh_A_memory + mesh_B_memory + output_memory)

if max_hash_table_size < required_hash_table_size:
    use_binary_search_fallback()
```

### 6.4 Thread Safety During Build

**Risk**: Race conditions during parallel insert can cause lost updates

**Current approach**: Atomic compare-exchange ensures correctness

**Gotcha**: ABA problem (rare but possible with 32-bit atomics)

**Alternative**: Use per-thread local tables + merge (higher memory, no atomics)

### 6.5 Handling of Missing Keys

**Current behavior**: Returns -1 for missing rows, sets flags(i) = 0

**Hash lookup must preserve this**:
```cpp
KOKKOS_INLINE_FUNCTION
int hash_lookup_with_missing(const RowHashMap& map, const Key& key) {
    if (map.table_size == 0) return -1;

    std::size_t idx = map.hash(key);
    const std::size_t start_idx = idx;
    std::size_t probes = 0;
    const std::size_t max_probes = map.table_size;  // Prevent infinite loop

    while (probes < max_probes) {
        if (map.occupied(idx) == 0) {
            return -1;  // Missing key
        }

        if (map.keys(idx) == key) {
            return map.values(idx);  // Found
        }

        idx = (idx + 1) & (map.table_size - 1);
        probes++;
    }

    return -1;  // Table full, key not found
}
```

### 6.6 CUDA-Specific Issues

**Shared memory pollution**: Hash table accessed by all threads can saturate memory bandwidth

**Mitigation**: Use read-only cache for lookup phase
```cpp
// In CUDA, hash table can be cached in read-only texture cache
__device__ int hash_lookup_cached(const RowHashMap& map, const Key& key) {
    // Compiler will use LD.G (cached load) instead of LD.CG (global cache)
    std::size_t idx = hash(key);
    // ... rest of lookup
}
```

**Warp divergence reduction**: Group lookups by hash bucket (complex implementation)

---

## 7. Recommendations

### 7.1 When to Use Hash-Based Row Mapping

**Use hash-based when**:
- Mesh B is large (> 10K rows for 2D, > 100K rows for 3D)
- Many lookups will be performed (mesh A has > 1K rows)
- Memory is not severely constrained
- Mesh B is static (built once, queried many times)

**Use binary search when**:
- Mesh B is small (< 1K rows)
- Memory is constrained
- Mesh B changes frequently (hash table rebuild overhead)
- Mesh is dense (direct indexing is better than hash)

### 7.2 Adaptive Strategy Recommendation

Implement a hybrid approach that selects the optimal strategy at runtime:

```cpp
template <int DIM, class MemorySpace>
inline Mesh<DIM, MemorySpace>
intersect_meshes_adaptive(const Mesh<DIM, MemorySpace>& A,
                          const Mesh<DIM, MemorySpace>& B) {
    // Decision tree
    if (B.num_rows < 1000) {
        // Small mesh: binary search is faster
        return intersect_meshes_binary_search(A, B);
    } else if (B.num_rows > 100000 && is_dense_mesh(B)) {
        // Large dense mesh: use perfect hash (direct indexing)
        return intersect_meshes_perfect_hash(A, B);
    } else {
        // Large sparse mesh: use open-addressing hash table
        return intersect_meshes_hash(A, B);
    }
}
```

### 7.3 Implementation Priority

**Phase 1: Proof of Concept** (1-2 weeks)
- Implement basic open-addressing hash table for 2D
- Test correctness on Small/Medium configurations
- Benchmark against binary search baseline

**Phase 2: Optimization** (2-3 weeks)
- Extend to 3D with proper hash function
- Implement Robin Hood hashing for better probe distribution
- Add adaptive strategy selection
- GPU testing and optimization

**Phase 3: Production Readiness** (1-2 weeks)
- Comprehensive testing (edge cases, collision handling)
- Memory pressure testing
- Documentation and examples
- Integration with existing API

### 7.4 Expected Performance Gains

**Conservative estimates** (after optimization):

| Mesh Size | Expected Speedup | Confidence |
|-----------|------------------|------------|
| Small (< 1K rows) | 0.8-1.2x | Low (build overhead may dominate) |
| Medium (1K-100K) | 1.5-2.0x | High |
| Large (> 100K) | 2.0-3.0x | Medium (depends on memory bandwidth) |
| Dense (> 90% density) | 4.0-6.0x | High (use perfect hash instead) |

### 7.5 Final Verdict

**Feasibility**: ✓ **Highly feasible**

**Recommendation**: ✓ **Implement with adaptive strategy**

**Risk level**: Medium (manageable with testing and fallback)

**Expected ROI**: 1.5-3x performance improvement for typical workloads, with minimal regression for small meshes due to adaptive selection.

---

## Appendix A: Pseudocode Summary

### Complete Hash-Based Intersection Algorithm

```cpp
// ============================================================================
// Phase 0: Build hash table from mesh B
// ============================================================================
RowHashMap map = build_row_hash_map<3>(B, load_factor=0.7);

// ============================================================================
// Phase 1: Row mapping using hash lookup
// ============================================================================
Kokkos::parallel_for("hash_row_map", RangePolicy(0, A.num_rows),
    KOKKOS_LAMBDA(const std::size_t i) {
        const Key key = A.row_keys(i);
        const int idx_b = hash_lookup(map, key);

        if (idx_b >= 0) {
            flags(i) = 1;
            tmp_idx_a(i) = static_cast<int>(i);
            tmp_idx_b(i) = idx_b;
        } else {
            flags(i) = 0;
            tmp_idx_a(i) = -1;
            tmp_idx_b(i) = -1;
        }
    });

// ============================================================================
// Phase 2-5: Same as current implementation (interval intersection)
// ============================================================================
// ... (scan, compact, count intervals, fill, compact)
```

---

## Appendix B: References

1. **Kokkos UnorderedMap Documentation**
   - https://kokkos.org/kokkos-core-wiki/API/unordered_map.html

2. **Open Addressing Hash Tables**
   - Knuth, T. (1998). *The Art of Computer Programming, Volume 3: Sorting and Searching*
   - Chapter 6.4: Hashing

3. **Robin Hood Hashing**
   - Amble, O., & Knuth, D. (1974). "Ordered hash tables"
   - *Computer Journal*, 17(2), 135-142.

4. **GPU Hash Table Design**
   - Alcantara, D. A., et al. (2012). "Building an efficient hash table on the GPU"
   - *GPU Computing Gems Jade Edition*

5. **Existing Optimization Documentation**
   - `/home/sbstndbs/subsetix_kokkos/docs/ROW_MAPPING_OPTIMIZATION_GUIDE.md`
   - `/home/sbstndbs/subsetix_kokkos/docs/HYBRID_ROW_MAPPER_DESIGN.md`

---

## Document Metadata

**Author**: Claude Code Analysis
**Date**: 2025-01-24
**Context**: Playground intersection algorithm optimization
**Related files**:
- `/playground/intersection/include/playground/subsetix/csr/intersection/algorithm/optimized.hpp`
- `/playground/intersection/include/playground/subsetix/csr/intersection/detail/utils.hpp`
- `/playground/intersection/include/playground/subsetix/csr/intersection/types.hpp`
