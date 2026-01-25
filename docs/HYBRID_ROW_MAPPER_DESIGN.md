<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique
-->
# Hybrid Row Mapping Strategy Design Document

## Executive Summary

This document describes a comprehensive **hybrid row mapping strategy** for subsetix_kokkos that adapts to mesh characteristics for optimal performance across different sizes and patterns. The design maintains GPU compatibility (Kokkos/CUDA), supports both 2D and 3D meshes, and remains backward compatible with the existing API.

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Tier 1: Always-On Optimizations](#3-tier-1-always-on-optimizations)
4. [Tier 2: Size-Based Strategies](#4-tier-2-size-based-strategies)
5. [Tier 3: Advanced Techniques](#5-tier-3-advanced-techniques)
6. [Implementation Details](#6-implementation-details)
7. [Integration Points](#7-integration-points)
8. [Testing Strategy](#8-testing-strategy)
9. [Performance Expectations](#9-performance-expectations)

---

## 1. Overview

### 1.1 Current State

The current row mapping implementation in `/include/subsetix/geometry/csr_mapping.hpp` uses:
- **Binary search** (`find_row_by_y`) for row lookups
- **Linear scan** for interval matching
- **O(log R)** per-row lookup where R is the number of rows
- **No caching** or reuse of previously computed mappings

### 1.2 Motivation

Row mapping is a critical operation in:
- Field masking operations (`build_field_mask_mapping`)
- Set algebra operations (union, intersection, difference)
- AMR refinement and projection operations
- Field remapping between geometries

Performance varies significantly based on:
- **Row count**: Small (< 100), Medium (100-10K), Large (> 10K)
- **Distribution**: Dense (consecutive rows), Sparse (gapped), Clustered
- **Backend**: Serial, OpenMP, CUDA (warp parallelism available)
- **Dimensionality**: 2D (single key), 3D (compound key)

### 1.3 Design Philosophy

The hybrid mapper follows a **tiered approach**:

```
Tier 1: Always-on optimizations (zero overhead when disabled)
   ├─ Structure-of-Arrays (SoA) layout
   ├─ Warp-level primitives (CUDA)
   └─ Cache-friendly access patterns

Tier 2: Size-based strategies (selected at runtime)
   ├─ Small meshes (< 100 rows)
   │   └─ Linear search (branchless, vectorizable)
   ├─ Medium meshes (100-10K rows)
   │   └─ Binary search + LRU cache (32 entries)
   └─ Large meshes (> 10K rows)
       ├─ Binary search + LRU cache (128 entries)
       └─ Sorted row optimization (direct indexing)

Tier 3: Advanced techniques (opt-in, situation-dependent)
   ├─ Perfect hash table (uniform grids)
   ├─ Hierarchical indexing (AMR octree meshes)
   └─ GPU shared memory caching (CUDA only)
```

---

## 2. Architecture

### 2.1 Class Hierarchy

```cpp
namespace subsetix::csr::detail {

// Base interface (polymorphic-free for GPU compatibility)
template<int Dim, class MemorySpace>
class HybridRowMapperBase {
public:
    struct LookupResult {
        int row_index = -1;
        bool cache_hit = false;
    };

    virtual ~HybridRowMapperBase() = default;

    // Core operations (must be KOKKOS_INLINE_FUNCTION)
    KOKKOS_INLINE_FUNCTION
    virtual LookupResult find_row(const RowKey<Dim>& key) const = 0;

    KOKKOS_INLINE_FUNCTION
    virtual void build_cache(const Mesh<Dim, MemorySpace>& mesh) = 0;
};

// Tier 1: SoA mapper (always active)
template<int Dim, class MemorySpace>
class SoARowMapper : public HybridRowMapperBase<Dim, MemorySpace> {
    // Structure-of-Arrays layout for better vectorization
    using CoordView = Kokkos::View<typename RowKey<Dim>::coord_type*, MemorySpace>;
    CoordView y_coords;      // Separate array for Y
    CoordView z_coords;      // Separate array for Z (3D only)
    // ... implementation
};

// Tier 2: Size-based mappers
template<int Dim, class MemorySpace>
class LinearScanMapper : public HybridRowMapperBase<Dim, MemorySpace> {
    // Branchless linear search for small meshes
};

template<int Dim, class MemorySpace>
class BinarySearchMapper : public HybridRowMapperBase<Dim, MemorySpace> {
    // Binary search + LRU cache for medium/large meshes
};

// Tier 3: Advanced mappers
template<int Dim, class MemorySpace>
class PerfectHashMapper : public HybridRowMapperBase<Dim, MemorySpace> {
    // Direct hash for uniform grids
};

template<int Dim, class MemorySpace>
class HierarchicalMapper : public HybridRowMapperBase<Dim, MemorySpace> {
    // Multi-level index for AMR meshes
};

// Unified facade (runtime dispatch)
template<int Dim, class MemorySpace>
class HybridRowMapper {
private:
    enum class Strategy {
        LINEAR_SCAN,        // < 100 rows
        BINARY_SEARCH,      // 100-10K rows
        BINARY_SEARCH_CACHE, // > 10K rows
        PERFECT_HASH,       // uniform grid
        HIERARCHICAL        // AMR mesh
    };

    Strategy strategy_;
    std::unique_ptr<HybridRowMapperBase<Dim, MemorySpace>> impl_;

    // Decision tree for strategy selection
    Strategy select_strategy(const Mesh<Dim, MemorySpace>& mesh);

public:
    // Build mapper from mesh (auto-detects optimal strategy)
    static HybridRowMapper build(const Mesh<Dim, MemorySpace>& mesh);

    // Explicit strategy override
    static HybridRowMapper build_with_strategy(
        const Mesh<Dim, MemorySpace>& mesh,
        Strategy strategy);

    // Core API (forward to implementation)
    KOKKOS_INLINE_FUNCTION
    typename HybridRowMapperBase<Dim, MemorySpace>::LookupResult
    find_row(const RowKey<Dim>& key) const;
};

} // namespace subsetix::csr::detail
```

### 2.2 Mesh Characteristics Analysis

```cpp
template<int Dim, class MemorySpace>
struct MeshCharacteristics {
    std::size_t num_rows = 0;
    std::size_t num_intervals = 0;

    // Distribution metrics
    double avg_rows_per_y = 0.0;      // For 3D: avg rows with same Y
    double row_spacing_variance = 0.0; // 0.0 = uniform, > 0.0 = gapped
    double clustering_factor = 0.0;    // 0.0 = scattered, 1.0 = clustered

    // Pattern detection
    bool is_uniform_grid = false;      // Consecutive Y/Z coordinates
    bool is_amr_mesh = false;          // Hierarchical refinement pattern
    bool is_sorted = true;             // Rows are sorted (should always be true)

    // Computed during analysis (host-side only)
    static MeshCharacteristics analyze(const Mesh<Dim, MemorySpace>& mesh);
};
```

---

## 3. Tier 1: Always-On Optimizations

### 3.1 Structure-of-Arrays (SoA) Layout

**Problem**: The current `RowKey2D`/`RowKey3D` structures use Array-of-Structs (AoS) layout, which causes unnecessary memory loads on GPU.

**Solution**: Separate coordinate arrays for better vectorization and cache utilization.

```cpp
// Current AoS (inefficient)
struct RowKey3D {
    Coord y = 0;
    Coord z = 0;
};
Kokkos::View<RowKey3D*, MemorySpace> row_keys;  // 2x memory transactions per load

// Optimized SoA
template<int Dim, class MemorySpace>
class SoARowKeys {
public:
    using CoordView = Kokkos::View<Coord*, MemorySpace>;

    CoordView y_coords;  // [num_rows]
    CoordView z_coords;  // [num_rows] (only for Dim=3)

    std::size_t size() const { return y_coords.extent(0); }

    // Accessors
    KOKKOS_INLINE_FUNCTION
    RowKey<Dim> operator[](std::size_t i) const {
        if constexpr (Dim == 2) {
            return RowKey2D<Coord>{y_coords(i)};
        } else {
            return RowKey3D<Coord>{y_coords(i), z_coords(i)};
        }
    }
};
```

**Benefits**:
- **2x reduction** in memory transactions on GPU
- Better SIMD vectorization on CPU
- Allows independent loading of Y/Z coordinates

### 3.2 Warp-Level Primitives (CUDA)

**Problem**: Binary search in warp causes divergence.

**Solution**: Use warp-level primitives for cooperative search.

```cpp
#ifdef KOKKOS_ENABLE_CUDA

template<class RowKeyView>
KOKKOS_INLINE_FUNCTION
int find_row_warp_binary(const RowKeyView& rows, std::size_t num_rows, Coord y) {
    // Warp-level cooperative binary search
    // All threads in warp participate, reducing divergence

    const int warp_id = Kokkos::Impl::cuda_internal::warp_id();
    const int lane_id = Kokkos::Impl::cuda_internal::lane_id();

    // Early exit: have one thread per warp perform the search
    int result = -1;

    if (lane_id == 0) {
        // Single thread performs binary search
        std::size_t lo = 0;
        std::size_t hi = num_rows;

        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (rows(mid).y < y) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        if (lo < num_rows && rows(lo).y == y) {
            result = static_cast<int>(lo);
        }
    }

    // Broadcast result to all lanes in warp
    return __shfl_sync(0xFFFFFFFF, result, 0);
}

#endif // KOKKOS_ENABLE_CUDA
```

### 3.3 Cache-Friendly Access Patterns

```cpp
// Precompute frequently accessed data
template<int Dim, class MemorySpace>
struct RowMappingCache {
    // Small LRU cache (host-side for CPU, device-side for GPU)
    static constexpr std::size_t CACHE_SIZE = 32;

    struct CacheEntry {
        RowKey<Dim> key;
        int row_index;
        bool valid;
    };

    Kokkos::View<CacheEntry*, MemorySpace> entries;
    std::size_t next_slot = 0;

    KOKKOS_INLINE_FUNCTION
    int lookup(const RowKey<Dim>& key) const {
        // Linear search in cache (fast for small CACHE_SIZE)
        for (std::size_t i = 0; i < CACHE_SIZE; ++i) {
            if (entries(i).valid && entries(i).key == key) {
                return entries(i).row_index;
            }
        }
        return -1;  // Cache miss
    }

    KOKKOS_INLINE_FUNCTION
    void insert(const RowKey<Dim>& key, int row_index) {
        // Simple round-robin replacement
        const std::size_t slot = next_slot % CACHE_SIZE;
        entries(slot) = CacheEntry{key, row_index, true};
        ++next_slot;
    }
};
```

---

## 4. Tier 2: Size-Based Strategies

### 4.1 Linear Scan Mapper (Small Meshes: < 100 rows)

**Use Case**: Small meshes where binary search overhead dominates.

```cpp
template<int Dim, class MemorySpace>
class LinearScanMapper : public HybridRowMapperBase<Dim, MemorySpace> {
public:
    using RowKeyView = typename Mesh<Dim, MemorySpace>::RowKeyView;

    RowKeyView row_keys;
    std::size_t num_rows;

    // Branchless linear search using predication
    KOKKOS_INLINE_FUNCTION
    LookupResult find_row(const RowKey<Dim>& key) const {
        int found_index = -1;

        // Fully unrolled loop for small N
        #pragma unroll
        for (std::size_t i = 0; i < num_rows; ++i) {
            // Predicated comparison (no branch)
            const bool match = (row_keys(i) == key);
            found_index = match ? static_cast<int>(i) : found_index;
        }

        return LookupResult{found_index, false};
    }

    void build_cache(const Mesh<Dim, MemorySpace>& mesh) override {
        row_keys = mesh.row_keys;
        num_rows = mesh.num_rows;
    }
};
```

**Performance**: For N < 100, linear scan is ~2x faster than binary search due to:
- No branch mispredictions
- Better instruction-level parallelism
- Prefetcher effectiveness

### 4.2 Binary Search Mapper (Medium Meshes: 100-10K rows)

**Use Case**: Standard binary search with optional LRU cache.

```cpp
template<int Dim, class MemorySpace>
class BinarySearchMapper : public HybridRowMapperBase<Dim, MemorySpace> {
public:
    using RowKeyView = typename Mesh<Dim, MemorySpace>::RowKeyView;

    RowKeyView row_keys;
    std::size_t num_rows;
    RowMappingCache<Dim, MemorySpace> cache;

    KOKKOS_INLINE_FUNCTION
    LookupResult find_row(const RowKey<Dim>& key) const {
        // Check cache first
        const int cached = cache.lookup(key);
        if (cached >= 0) {
            return LookupResult{cached, true};
        }

        // Binary search
        int result = -1;
        std::size_t lo = 0;
        std::size_t hi = num_rows;

        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (row_keys(mid) < key) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        if (lo < num_rows && row_keys(lo) == key) {
            result = static_cast<int>(lo);
            // Insert into cache (const_cast - safe in this context)
            const_cast<RowMappingCache<Dim, MemorySpace>&>(cache).insert(key, result);
        }

        return LookupResult{result, false};
    }

    void build_cache(const Mesh<Dim, MemorySpace>& mesh) override {
        row_keys = mesh.row_keys;
        num_rows = mesh.num_rows;
        // Initialize cache
    }
};
```

### 4.3 Sorted Row Optimization (Large Meshes: > 10K rows)

**Use Case**: When row keys form a contiguous range, use direct indexing.

```cpp
template<int Dim, class MemorySpace>
class SortedRowMapper : public HybridRowMapperBase<Dim, MemorySpace> {
public:
    using RowKeyView = typename Mesh<Dim, MemorySpace>::RowKeyView;

    RowKeyView row_keys;
    std::size_t num_rows;

    Coord min_y = 0;  // Minimum Y coordinate
    Coord y_range = 0;  // max_y - min_y

    bool is_compact = false;  // True if rows are contiguous

    KOKKOS_INLINE_FUNCTION
    LookupResult find_row(const RowKey<Dim>& key) const {
        if (is_compact) {
            // Direct indexing: O(1)
            const Coord y = key.y;
            const Coord offset = y - min_y;

            if (offset >= 0 && offset < y_range) {
                const std::size_t idx = static_cast<std::size_t>(offset);
                if (idx < num_rows && row_keys(idx).y == y) {
                    return LookupResult{static_cast<int>(idx), false};
                }
            }
            return LookupResult{-1, false};
        } else {
            // Fallback to binary search
            return binary_search_impl(key);
        }
    }

    void build_cache(const Mesh<Dim, MemorySpace>& mesh) override {
        row_keys = mesh.row_keys;
        num_rows = mesh.num_rows;

        // Check if rows form a contiguous range
        if (num_rows > 0) {
            min_y = row_keys(0).y;
            const Coord max_y = row_keys(num_rows - 1).y;
            y_range = max_y - min_y + 1;

            // Verify contiguity (O(N) check during setup)
            is_compact = (y_range == static_cast<Coord>(num_rows));
        }
    }

private:
    KOKKOS_INLINE_FUNCTION
    LookupResult binary_search_impl(const RowKey<Dim>& key) const {
        std::size_t lo = 0;
        std::size_t hi = num_rows;

        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (row_keys(mid) < key) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        if (lo < num_rows && row_keys(lo) == key) {
            return LookupResult{static_cast<int>(lo), false};
        }
        return LookupResult{-1, false};
    }
};
```

---

## 5. Tier 3: Advanced Techniques

### 5.1 Perfect Hash Mapper (Uniform Grids)

**Use Case**: Meshes with uniformly spaced row keys (e.g., structured grids).

```cpp
template<int Dim, class MemorySpace>
class PerfectHashMapper : public HybridRowMapperBase<Dim, MemorySpace> {
public:
    using RowKeyView = typename Mesh<Dim, MemorySpace>::RowKeyView;

    Coord y_min = 0;
    Coord y_stride = 1;  // Spacing between consecutive Y values
    std::size_t num_rows = 0;

    KOKKOS_INLINE_FUNCTION
    LookupResult find_row(const RowKey<Dim>& key) const {
        // Perfect hash function
        const Coord offset = key.y - y_min;

        if (offset >= 0 && offset % y_stride == 0) {
            const std::size_t idx = static_cast<std::size_t>(offset / y_stride);
            if (idx < num_rows) {
                return LookupResult{static_cast<int>(idx), false};
            }
        }

        return LookupResult{-1, false};
    }

    void build_cache(const Mesh<Dim, MemorySpace>& mesh) override {
        // Detect uniform spacing
        if (mesh.num_rows >= 2) {
            y_min = mesh.row_keys(0).y;
            y_stride = mesh.row_keys(1).y - mesh.row_keys(0).y;
            num_rows = mesh.num_rows;

            // Verify uniform spacing
            bool is_uniform = true;
            for (std::size_t i = 2; i < mesh.num_rows; ++i) {
                const Coord stride = mesh.row_keys(i).y - mesh.row_keys(i-1).y;
                if (stride != y_stride) {
                    is_uniform = false;
                    break;
                }
            }

            if (!is_uniform) {
                y_stride = 0;  // Disable perfect hash
            }
        }
    }
};
```

**Performance**: O(1) lookup with minimal computation.

### 5.2 Hierarchical Mapper (AMR Meshes)

**Use Case**: Octree/quadtree-based AMR meshes with hierarchical refinement.

```cpp
template<int Dim, class MemorySpace>
class HierarchicalMapper : public HybridRowMapperBase<Dim, MemorySpace> {
public:
    // Two-level index: coarse level + fine level
    struct LevelIndex {
        Coord coarse_key;  // Coarse-level Y/Z
        std::size_t offset;  // Offset in row_keys array
        std::size_t count;   // Number of rows in this coarse cell
    };

    Kokkos::View<LevelIndex*, MemorySpace> level0_index;  // Coarse level
    std::size_t level0_size = 0;

    typename Mesh<Dim, MemorySpace>::RowKeyView row_keys;

    KOKKOS_INLINE_FUNCTION
    LookupResult find_row(const RowKey<Dim>& key) const {
        // Level 0: Find coarse cell
        const Coord coarse = key.y >> 4;  // Example: 16-cell coarsening

        // Binary search in level0_index
        std::size_t lo = 0;
        std::size_t hi = level0_size;

        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (level0_index(mid).coarse_key < coarse) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        if (lo < level0_size && level0_index(lo).coarse_key == coarse) {
            // Level 1: Binary search within coarse cell
            const std::size_t offset = level0_index(lo).offset;
            const std::size_t count = level0_index(lo).count;

            for (std::size_t i = 0; i < count; ++i) {
                const std::size_t idx = offset + i;
                if (row_keys(idx) == key) {
                    return LookupResult{static_cast<int>(idx), false};
                }
            }
        }

        return LookupResult{-1, false};
    }

    void build_cache(const Mesh<Dim, MemorySpace>& mesh) override {
        // Build two-level index (host-side operation)
        // ... implementation
    }
};
```

**Benefits**:
- Reduces binary search depth from O(log N) to O(log N/M + log M)
- Where M is the coarsening factor (e.g., 16)
- Effective for AMR meshes with clustered refinement

### 5.3 GPU Shared Memory Caching (CUDA Only)

```cpp
#ifdef KOKKOS_ENABLE_CUDA

template<class RowKeyView>
__device__ int find_row_shared_cache(const RowKeyView& row_keys,
                                     std::size_t num_rows,
                                     Coord y,
                                     int* shared_cache,
                                     int cache_size) {
    // Per-block shared memory cache
    const int tid = threadIdx.x;
    extern __shared__ int cache[];

    // Load cache entries cooperatively
    if (tid < cache_size) {
        cache[tid] = -1;
    }
    __syncthreads();

    // Try to find in cache
    for (int i = 0; i < cache_size; ++i) {
        if (cache[i] >= 0) {
            const Coord cached_y = row_keys(cache[i]).y;
            if (cached_y == y) {
                return cache[i];
            }
        }
    }

    // Cache miss: perform search
    int result = binary_search_impl(row_keys, num_rows, y);

    // Insert into cache (race condition OK - LRU semantics)
    const int slot = tid % cache_size;
    cache[slot] = result;

    return result;
}

#endif // KOKKOS_ENABLE_CUDA
```

---

## 6. Implementation Details

### 6.1 Strategy Selection Logic

```cpp
template<int Dim, class MemorySpace>
auto HybridRowMapper<Dim, MemorySpace>::select_strategy(
    const Mesh<Dim, MemorySpace>& mesh) -> Strategy {

    const MeshCharacteristics chars = MeshCharacteristics::analyze(mesh);

    // Priority 1: Perfect hash for uniform grids
    if (chars.is_uniform_grid && chars.row_spacing_variance < 0.01) {
        return Strategy::PERFECT_HASH;
    }

    // Priority 2: Hierarchical for AMR meshes
    if (chars.is_amr_mesh && chars.clustering_factor > 0.7) {
        return Strategy::HIERARCHICAL;
    }

    // Priority 3: Size-based selection
    if (chars.num_rows < 100) {
        return Strategy::LINEAR_SCAN;
    } else if (chars.num_rows < 10000) {
        return Strategy::BINARY_SEARCH;
    } else {
        return Strategy::BINARY_SEARCH_CACHE;
    }
}
```

### 6.2 Mesh Characteristics Analysis

```cpp
template<int Dim, class MemorySpace>
MeshCharacteristics<Dim>
MeshCharacteristics<Dim>::analyze(const Mesh<Dim, MemorySpace>& mesh) {
    MeshCharacteristics<Dim> chars;

    chars.num_rows = mesh.num_rows;
    chars.num_intervals = mesh.num_intervals;

    if (mesh.num_rows == 0) {
        return chars;
    }

    // Copy row_keys to host for analysis
    const auto host_keys = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, mesh.row_keys);

    // Compute statistics
    Coord prev_y = host_keys(0).y;
    std::vector<Coord> gaps;
    std::size_t uniform_count = 0;

    for (std::size_t i = 1; i < mesh.num_rows; ++i) {
        const Coord curr_y = host_keys(i).y;
        const Coord gap = curr_y - prev_y;

        gaps.push_back(gap);

        if (gap == 1) {
            ++uniform_count;
        }

        prev_y = curr_y;
    }

    // Uniform grid detection
    chars.is_uniform_grid = (uniform_count == mesh.num_rows - 1);

    // Gap variance
    if (!gaps.empty()) {
        const double mean_gap = std::accumulate(gaps.begin(), gaps.end(), 0.0) / gaps.size();
        double variance = 0.0;
        for (Coord gap : gaps) {
            variance += (gap - mean_gap) * (gap - mean_gap);
        }
        chars.row_spacing_variance = variance / gaps.size();
    }

    // AMR detection (heuristic: clustered rows with similar gaps)
    // ... implementation

    // Verify sorted property
    bool is_sorted = true;
    for (std::size_t i = 1; i < mesh.num_rows; ++i) {
        if (!(host_keys(i-1) < host_keys(i))) {
            is_sorted = false;
            break;
        }
    }
    chars.is_sorted = is_sorted;

    return chars;
}
```

### 6.3 KOKKOS_INLINE_FUNCTION Compatibility

All device-accessible functions must be marked with `KOKKOS_INLINE_FUNCTION`:

```cpp
// Correct: Device-compatible
KOKKOS_INLINE_FUNCTION
int find_row(const RowKey<Dim>& key) const {
    // Implementation
}

// Incorrect: Not accessible from device
int find_row(const RowKey<Dim>& key) const {
    // Implementation
}
```

**Critical**: Virtual functions cannot be used in device code. The design uses:
- Compile-time polymorphism (templates)
- Explicit strategy selection on host
- Type erasure through `std::unique_ptr` (host-side only)

---

## 7. Integration Points

### 7.1 Replace Existing Row Mapping

```cpp
// Before: /include/subsetix/geometry/csr_mapping.hpp
inline Kokkos::View<int*, DeviceMemorySpace>
build_row_map_y(const IntervalSet2DDevice::RowKeyView& mask_rows,
                const IntervalSet2DDevice::RowKeyView& parent_rows,
                std::size_t num_parent_rows) {
    Kokkos::View<int*, DeviceMemorySpace> mapping(
        "subsetix_row_map_y", mask_rows.extent(0));

    Kokkos::parallel_for(
        "subsetix_row_map_y_kernel",
        Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(mask_rows.extent(0))),
        KOKKOS_LAMBDA(const int i) {
            mapping(i) = detail::find_row_by_y(parent_rows, num_parent_rows,
                                                mask_rows(i).y);
        });

    ExecSpace().fence();
    return mapping;
}

// After: Use hybrid mapper
inline Kokkos::View<int*, DeviceMemorySpace>
build_row_map_y(const IntervalSet2DDevice::RowKeyView& mask_rows,
                const IntervalSet2DDevice::RowKeyView& parent_rows,
                std::size_t num_parent_rows) {

    // Build parent mesh wrapper
    IntervalSet2DDevice parent_mesh;
    parent_mesh.row_keys = parent_rows;
    parent_mesh.num_rows = num_parent_rows;

    // Create hybrid mapper (auto-detects optimal strategy)
    auto mapper = detail::HybridRowMapper<2, DeviceMemorySpace>::build(parent_mesh);

    // Extract mapper device data (for use in kernel)
    const auto mapper_device = mapper.get_device_view();

    Kokkos::View<int*, DeviceMemorySpace> mapping(
        "subsetix_row_map_y", mask_rows.extent(0));

    Kokkos::parallel_for(
        "subsetix_row_map_y_kernel",
        Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(mask_rows.extent(0))),
        KOKKOS_LAMBDA(const int i) {
            auto result = mapper_device.find_row(mask_rows(i));
            mapping(i) = result.row_index;
        });

    ExecSpace().fence();
    return mapping;
}
```

### 7.2 Integration with Set Algebra

```cpp
// /include/subsetix/geometry/csr_set_ops.hpp

// Row lookup in set operations
template <class RowKeyViewA, class RowKeyViewB>
KOKKOS_INLINE_FUNCTION
RowRanges extract_row_ranges_hybrid(
    int ia, int ib,
    const RowKeyViewA& row_ptr_a,
    const RowKeyViewB& row_ptr_b,
    const auto& mapper_a,  // Hybrid mapper for mesh A
    const auto& mapper_b)  // Hybrid mapper for mesh B
{
    RowRanges r;
    if (ia >= 0) {
        const std::size_t row_a = static_cast<std::size_t>(ia);
        r.begin_a = row_ptr_a(row_a);
        r.end_a = row_ptr_a(row_a + 1);
    }
    if (ib >= 0) {
        const std::size_t row_b = static_cast<std::size_t>(ib);
        r.begin_b = row_ptr_b(row_b);
        r.end_b = row_ptr_b(row_b + 1);
    }
    return r;
}
```

### 7.3 Integration with Field Masking

```cpp
// /include/subsetix/csr_ops/field_mapping.hpp

inline FieldMaskMapping
build_field_mask_mapping_hybrid(const IntervalSet2DDevice& mask,
                                const IntervalSet2DDevice& geom) {
    FieldMaskMapping mapping;

    if (mask.num_rows == 0 || mask.num_intervals == 0 ||
        geom.num_rows == 0 || geom.num_intervals == 0) {
        return mapping;
    }

    // Create hybrid mapper for mask -> geom mapping
    auto mapper = detail::HybridRowMapper<2, DeviceMemorySpace>::build(geom);
    const auto mapper_device = mapper.get_device_view();

    // Build row map using hybrid mapper
    mapping.row_map = Kokkos::View<int*, DeviceMemorySpace>(
        "subsetix_row_map", mask.num_rows);

    Kokkos::parallel_for(
        "subsetix_hybrid_row_map",
        Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(mask.num_rows)),
        KOKKOS_LAMBDA(const int i) {
            auto result = mapper_device.find_row(mask.row_keys(i));
            mapping.row_map(i) = result.row_index;
        });

    ExecSpace().fence();

    // Rest of the implementation remains the same
    // ...
}
```

### 7.4 Backward Compatibility

The hybrid mapper is **opt-in** by default. Existing code continues to work:

```cpp
// Legacy API still works
inline Kokkos::View<int*, DeviceMemorySpace>
build_row_map_y_legacy(...) {
    // Original implementation
}

// New API with suffix _hybrid
inline Kokkos::View<int*, DeviceMemorySpace>
build_row_map_y_hybrid(...) {
    // Hybrid mapper implementation
}

// Or use CMake option
#ifdef SUBSETIX_USE_HYBRID_ROW_MAPPER
  #define build_row_map_y build_row_map_y_hybrid
#else
  #define build_row_map_y build_row_map_y_legacy
#endif
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

```cpp
// /tests/hybrid_row_mapper_test.cpp

template<int Dim>
class HybridRowMapperTest : public ::testing::Test {
protected:
    using MemorySpace = Kokkos::HostSpace;  // Start with host for testing
    using Mesh = playground::subsetix::csr::intersection::Mesh<Dim, MemorySpace>;
};

TYPED_TEST_SUITE_P(HybridRowMapperTest);

TYPED_TEST_P(HybridRowMapperTest, CorrectnessLinearScan) {
    // Test linear scan mapper on small mesh
    Mesh mesh = create_small_mesh<Dim>(50);  // 50 rows

    auto mapper = detail::HybridRowMapper<Dim, MemorySpace>::build_with_strategy(
        mesh, detail::HybridRowMapper<Dim, MemorySpace>::Strategy::LINEAR_SCAN);

    // Test all rows
    for (std::size_t i = 0; i < mesh.num_rows; ++i) {
        auto result = mapper.find_row(mesh.row_keys(i));
        EXPECT_EQ(result.row_index, static_cast<int>(i));
    }

    // Test non-existent row
    typename Mesh::RowKey missing_key;
    missing_key.y = -9999;
    auto result = mapper.find_row(missing_key);
    EXPECT_EQ(result.row_index, -1);
}

TYPED_TEST_P(HybridRowMapperTest, CorrectnessBinarySearch) {
    // Test binary search mapper on medium mesh
    Mesh mesh = create_medium_mesh<Dim>(5000);  // 5K rows

    auto mapper = detail::HybridRowMapper<Dim, MemorySpace>::build_with_strategy(
        mesh, detail::HybridRowMapper<Dim, MemorySpace>::Strategy::BINARY_SEARCH);

    // Test correctness
    for (std::size_t i = 0; i < mesh.num_rows; i += 100) {
        auto result = mapper.find_row(mesh.row_keys(i));
        EXPECT_EQ(result.row_index, static_cast<int>(i));
    }
}

TYPED_TEST_P(HybridRowMapperTest, PerfectHashUniformGrid) {
    // Test perfect hash on uniform grid
    Mesh mesh = create_uniform_grid<Dim>(1000);  // Consecutive rows

    auto mapper = detail::HybridRowMapper<Dim, MemorySpace>::build(mesh);

    // Verify perfect hash was selected
    // (add introspection API to HybridRowMapper)

    // Test O(1) lookup
    for (std::size_t i = 0; i < mesh.num_rows; i += 10) {
        auto result = mapper.find_row(mesh.row_keys(i));
        EXPECT_EQ(result.row_index, static_cast<int>(i));
    }
}

REGISTER_TYPED_TEST_SUITE_P(HybridRowMapperTest,
    CorrectnessLinearScan,
    CorrectnessBinarySearch,
    PerfectHashUniformGrid
);

using Dimensions = ::testing::Types<2, 3>;
INSTANTIATE_TYPED_TEST_SUITE_P(Dimensions, HybridRowMapperTest, Dimensions);
```

### 8.2 Performance Benchmarks

```cpp
// /benchmarks/hybrid_row_mapper_benchmark.cpp

template<int Dim>
void BM_LinearScan(benchmark::State& state) {
    const std::size_t num_rows = state.range(0);
    auto mesh = create_random_mesh<Dim>(num_rows);

    auto mapper = detail::HybridRowMapper<Dim, DeviceMemorySpace>::build_with_strategy(
        mesh, detail::HybridRowMapper<Dim, DeviceMemorySpace>::Strategy::LINEAR_SCAN);

    for (auto _ : state) {
        for (std::size_t i = 0; i < num_rows; ++i) {
            auto result = mapper.find_row(mesh.row_keys(i));
            benchmark::DoNotOptimize(result);
        }
    }

    state.SetItemsProcessed(state.iterations() * num_rows);
}

template<int Dim>
void BM_BinarySearch(benchmark::State& state) {
    const std::size_t num_rows = state.range(0);
    auto mesh = create_random_mesh<Dim>(num_rows);

    auto mapper = detail::HybridRowMapper<Dim, DeviceMemorySpace>::build_with_strategy(
        mesh, detail::HybridRowMapper<Dim, DeviceMemorySpace>::Strategy::BINARY_SEARCH);

    for (auto _ : state) {
        for (std::size_t i = 0; i < num_rows; ++i) {
            auto result = mapper.find_row(mesh.row_keys(i));
            benchmark::DoNotOptimize(result);
        }
    }

    state.SetItemsProcessed(state.iterations() * num_rows);
}

template<int Dim>
void BM_PerfectHash(benchmark::State& state) {
    const std::size_t num_rows = state.range(0);
    auto mesh = create_uniform_grid<Dim>(num_rows);

    auto mapper = detail::HybridRowMapper<Dim, DeviceMemorySpace>::build(mesh);

    for (auto _ : state) {
        for (std::size_t i = 0; i < num_rows; ++i) {
            auto result = mapper.find_row(mesh.row_keys(i));
            benchmark::DoNotOptimize(result);
        }
    }

    state.SetItemsProcessed(state.iterations() * num_rows);
}

// Register benchmarks
BENCHMARK_TEMPLATE(BM_LinearScan, 2)->Range(10, 100);
BENCHMARK_TEMPLATE(BM_BinarySearch, 2)->Range(100, 100000);
BENCHMARK_TEMPLATE(BM_PerfectHash, 2)->Range(1000, 100000);
```

### 8.3 Cross-Strategy Validation

```cpp
// Verify all strategies produce identical results
template<int Dim>
void cross_strategy_validation_test() {
    std::vector<std::size_t> sizes = {50, 500, 5000, 50000};

    for (std::size_t size : sizes) {
        auto mesh = create_random_mesh<Dim>(size);

        std::vector<detail::HybridRowMapper<Dim, HostMemorySpace>::Strategy> strategies = {
            detail::HybridRowMapper<Dim, HostMemorySpace>::Strategy::LINEAR_SCAN,
            detail::HybridRowMapper<Dim, HostMemorySpace>::Strategy::BINARY_SEARCH,
            detail::HybridRowMapper<Dim, HostMemorySpace>::Strategy::BINARY_SEARCH_CACHE
        };

        std::vector<std::vector<int>> results;

        for (auto strategy : strategies) {
            auto mapper = detail::HybridRowMapper<Dim, HostMemorySpace>::build_with_strategy(
                mesh, strategy);

            std::vector<int> strategy_results(size);
            for (std::size_t i = 0; i < size; ++i) {
                auto result = mapper.find_row(mesh.row_keys(i));
                strategy_results[i] = result.row_index;
            }
            results.push_back(strategy_results);
        }

        // Compare all results
        for (std::size_t i = 1; i < results.size(); ++i) {
            EXPECT_EQ(results[0], results[i])
                << "Strategies produced different results for size " << size;
        }
    }
}
```

### 8.4 GPU Compatibility Tests

```cpp
// /tests/hybrid_row_mapper_cuda_test.cpp

#ifdef KOKKOS_ENABLE_CUDA

TEST(HybridRowMapperCuda, DeviceExecution) {
    using MemorySpace = Kokkos::CudaSpace;
    constexpr int Dim = 2;

    // Create mesh on device
    auto host_mesh = create_random_mesh<Dim>(1000);
    auto device_mesh = to_device<MemorySpace>(host_mesh);

    // Build mapper on host
    auto mapper = detail::HybridRowMapper<Dim, MemorySpace>::build(device_mesh);

    // Test lookup in device kernel
    Kokkos::View<int*, MemorySpace> results("results", host_mesh.num_rows);

    Kokkos::parallel_for(
        "test_hybrid_mapper_device",
        Kokkos::RangePolicy<Kokkos::Cuda>(0, host_mesh.num_rows),
        KOKKOS_LAMBDA(const int i) {
            auto mapper_device = mapper.get_device_view();
            auto result = mapper_device.find_row(device_mesh.row_keys(i));
            results(i) = result.row_index;
        });

    Kokkos::fence();

    // Verify results on host
    auto host_results = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, results);

    for (std::size_t i = 0; i < host_mesh.num_rows; ++i) {
        EXPECT_EQ(host_results(i), static_cast<int>(i));
    }
}

#endif // KOKKOS_ENABLE_CUDA
```

### 8.5 Integration Tests

```cpp
// /tests/hybrid_row_mapper_integration_test.cpp

TEST(HybridRowMapperIntegration, FieldMaskMapping) {
    // Test hybrid mapper in actual field masking operation
    auto mask = make_disk_device({0, 0}, 100);
    auto geom = make_box_device({-200, -200, 200, 200});

    // Use hybrid mapper for field mask mapping
    auto mapping = build_field_mask_mapping_hybrid(mask, geom);

    // Verify correctness
    auto host_mapping = to_host(mapping);

    // Check that all mask rows have valid mappings
    for (std::size_t i = 0; i < mask.num_rows; ++i) {
        EXPECT_GE(host_mapping.row_map(i), 0);
    }
}

TEST(HybridRowMapperIntegration, SetAlgebra) {
    // Test hybrid mapper in set operations
    auto A = make_random_device(domain, 0.3, 42);
    auto B = make_random_device(domain, 0.3, 43);

    CsrSetAlgebraContext ctx;

    // Use hybrid mapper for row lookup in union
    auto result = set_union_hybrid(A, B, ctx);

    // Verify result properties
    EXPECT_GE(result.num_rows, std::min(A.num_rows, B.num_rows));
}
```

---

## 9. Performance Expectations

### 9.1 Expected Speedups

| Mesh Size | Strategy | Baseline (ns/lookup) | Expected (ns/lookup) | Speedup |
|-----------|----------|---------------------|---------------------|---------|
| 50 rows | Linear scan | 150 | 75 | 2.0x |
| 500 rows | Binary search | 200 | 180 | 1.1x |
| 5K rows | Binary + cache | 250 | 220 | 1.1x |
| 50K rows | Binary + cache | 300 | 200 | 1.5x |
| 50K rows (uniform) | Perfect hash | 300 | 50 | 6.0x |
| 50K rows (AMR) | Hierarchical | 300 | 150 | 2.0x |

### 9.2 Memory Overhead

| Strategy | Additional Memory (per mapper) |
|----------|-------------------------------|
| Linear scan | 0 bytes |
| Binary search | 0 bytes |
| Binary + LRU cache (32 entries) | 32 * (4 + 4) = 256 bytes |
| Binary + LRU cache (128 entries) | 128 * (4 + 4) = 1 KB |
| Perfect hash | 16 bytes (metadata) |
| Hierarchical | ~10% of original row_keys size |

### 9.3 Build Time Overhead

Strategy selection and mapper construction add **negligible overhead**:
- Mesh characteristics analysis: O(N) host-side operation
- Strategy selection: O(1)
- Mapper construction: O(1) for most strategies

Total overhead: **< 1ms** for meshes up to 1M rows.

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Implement `MeshCharacteristics` analysis
- [ ] Implement `LinearScanMapper` (2D and 3D)
- [ ] Implement `BinarySearchMapper` (2D and 3D)
- [ ] Add host-side unit tests

### Phase 2: Integration (Week 3-4)
- [ ] Implement `HybridRowMapper` facade
- [ ] Add strategy selection logic
- [ ] Integrate with `build_row_map_y`
- [ ] Add integration tests

### Phase 3: Optimizations (Week 5-6)
- [ ] Implement `PerfectHashMapper`
- [ ] Implement `HierarchicalMapper`
- [ ] Add SoA layout optimization
- [ ] Add warp-level primitives (CUDA)

### Phase 4: Validation (Week 7-8)
- [ ] Performance benchmarks
- [ ] GPU compatibility tests
- [ ] Cross-validation tests
- [ ] Documentation and examples

---

## 11. Conclusion

The hybrid row mapping strategy provides:

1. **Automatic optimization**: Selects the best strategy based on mesh characteristics
2. **GPU compatibility**: All implementations are `KOKKOS_INLINE_FUNCTION` compatible
3. **2D/3D support**: Template-based design handles both dimensions
4. **Backward compatibility**: Can be enabled/disabled via CMake option
5. **Extensibility**: New strategies can be added without changing existing code

The tiered approach ensures that optimizations are applied only when beneficial, maintaining zero-overhead for small meshes while providing significant speedups for large, structured, or hierarchical meshes.

### Key Files to Modify/Create

- **New**: `/include/subsetix/geometry/hybrid_row_mapper.hpp`
- **New**: `/include/subsetix/geometry/mesh_characteristics.hpp`
- **Modify**: `/include/subsetix/geometry/csr_mapping.hpp`
- **Modify**: `/include/subsetix/csr_ops/field_mapping.hpp`
- **New**: `/tests/hybrid_row_mapper_test.cpp`
- **New**: `/benchmarks/hybrid_row_mapper_benchmark.cpp`
