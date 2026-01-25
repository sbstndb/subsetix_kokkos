# Row Mapping Optimization - Design Notes

## Design Philosophy

All row mapping alternatives share these core principles:

1. **Bit-identical results**: Every version produces exactly the same output
2. **Identical phases 2-5**: Only row mapping (phase 1) differs
3. **Namespace isolation**: Each version has its own namespace
4. **Kokkos compatibility**: All work with Serial, OpenMP, CUDA backends
5. **Header-only**: No separate .cpp files needed

## Common Patterns

### Mesh Type

Most versions use the same Mesh type as baseline:

```cpp
template <int DIM, class MemorySpace, class CoordType, class IndexType>
class Mesh {
  Kokkos::View<RowKey*, MemorySpace> row_keys;
  Kokkos::View<IndexType*, MemorySpace> row_ptr;
  Kokkos::View<Interval<CoordType>*, MemorySpace> intervals;
  std::size_t num_rows;
  std::size_t num_intervals;
};
```

**RowKey types:**
```cpp
// 2D: single coordinate
using RowKey2D = CoordType;

// 3D: struct with y and z
template <class T>
struct RowKey3D {
  T y, z;
  // Comparison operators for sorting
};
```

**Exception**: v7 uses SoA layout for better 3D performance:
```cpp
template <class MemorySpace, class CoordType>
struct SoAMesh3D {
  Kokkos::View<CoordType*, MemorySpace> y_coords;  // Separate arrays
  Kokkos::View<CoordType*, MemorySpace> z_coords;
  Kokkos::View<IndexType*, MemorySpace> row_ptr;
  Kokkos::View<Interval<CoordType>*, MemorySpace> intervals;
  std::size_t num_rows;
  std::size_t num_intervals;
};
```

### Entry Point Signature

All versions follow the same signature pattern:

```cpp
template <int DIM, class CoordType, class IndexType>
inline Mesh<DIM, DeviceMemorySpace, CoordType, IndexType>
intersect_meshes(const Mesh<DIM, DeviceMemorySpace, CoordType, IndexType>& A,
                const Mesh<DIM, DeviceMemorySpace, CoordType, IndexType>& B);
```

**Return value:** New mesh containing the intersection

**Constraints:**
- Input meshes must have sorted `row_keys`
- Input meshes must be in device memory space
- Output mesh is allocated in device memory space

### Shared Utilities

All versions use utilities from `detail/utils.hpp`:

#### Row Intersection (Phases 4-5)

```cpp
template <bool CountOnly>
KOKKOS_INLINE_FUNCTION
void row_intersection_impl(
    const Interval<CoordType>* A_intervals, IndexType A_begin, IndexType A_end,
    const Interval<CoordType>* B_intervals, IndexType B_begin, IndexType B_end,
    IndexType* out_counts,           // CountOnly=true: only count
    Interval<CoordType>* out_intervals, IndexType* out_idx  // CountOnly=false: fill
);
```

**Algorithm:** Two-pointer merge for sorted interval arrays
- **CountOnly=true**: Only counts overlapping intervals
- **CountOnly=false**: Fills intersection intervals

#### Row Range Extraction

```cpp
KOKKOS_INLINE_FUNCTION
std::pair<IndexType, IndexType>
extract_row_ranges(const IndexType* row_ptr, IndexType row_idx);
```

**Returns:** `[begin, end)` interval range for a row

#### Memory Management

```cpp
template <int DIM, class MemorySpace, class CoordType, class IndexType>
Mesh<DIM, MemorySpace, CoordType, IndexType>
allocate_mesh(std::size_t num_rows, std::size_t num_intervals);
```

**Allocates:** Mesh with specified capacity

## Algorithm-Specific Notes

### v4: Hash-Based

**Hash Table Structure:**
```cpp
struct HashEntry {
  RowKey key;
  IndexType value;
  bool occupied;
  bool is_tombstone;  // Marks deleted entries
};
```

**Growth Strategy:** Double capacity when load factor > 0.7

**Collision Resolution:** Linear probing with tombstone handling

**Key Considerations:**
- Need good hash function for uniformity
- Tombstone accumulation requires periodic rehash
- Memory overhead: ~40% for load factor 0.7

### v5: Parallel Merge

**Work Distribution:**
```cpp
// Each thread gets equal chunk of expected result size
IndexType chunk_size = (expected_matches + num_threads - 1) / num_threads;
IndexType my_begin = thread_id * chunk_size;
IndexType my_end = std::min(my_begin + chunk_size, expected_matches);
```

**Binary Search for Start Position:**
```cpp
// Find position in A and B for merge start
IndexType pos_A = binary_search_to_position(A_keys, my_begin);
IndexType pos_B = binary_search_to_position(B_keys, my_begin);
```

**Key Considerations:**
- Work imbalance if merges finish at different rates
- Binary search overhead per thread
- Excellent cache locality within chunks

### v6: Direct Index

**Pattern Detection:**
```cpp
enum class Pattern {
  Dense,           // [min, min+1, min+2, ..., max]
  UniformStride,   // [min, min+stride, min+2*stride, ...]
  SmallRange,      // max-min < threshold (use lookup table)
  Fallback         // Use binary search
};
```

**Detection Logic:**
```cpp
Pattern detect_pattern(const RowKey* keys, std::size_t n) {
  CoordType min_key = keys[0];
  CoordType max_key = keys[n-1];

  // Check for dense
  if (max_key - min_key == n - 1 && is_unique(keys, n)) {
    return Pattern::Dense;
  }

  // Check for uniform stride
  CoordType stride = keys[1] - keys[0];
  if (has_uniform_stride(keys, n, stride)) {
    return Pattern::UniformStride;
  }

  // Check for small range
  if (max_key - min_key < SMALL_RANGE_THRESHOLD) {
    return Pattern::SmallRange;
  }

  return Pattern::Fallback;
}
```

**Key Considerations:**
- Detection overhead must be amortized
- Lookup table memory for small ranges
- Stride must fit in CoordType range

### v7: SoA Optimized

**AoS vs SoA Layout:**

Original (AoS):
```cpp
struct RowKey3D { CoordType y, z; };
Kokkos::View<RowKey3D*, MemorySpace> row_keys;  // [y0,z0], [y1,z1], ...
```

Optimized (SoA):
```cpp
Kokkos::View<CoordType*, MemorySpace> y_coords;  // [y0,y1,y2,...]
Kokkos::View<CoordType*, MemorySpace> z_coords;  // [z0,z1,z2,...]
```

**Access Pattern:**
```cpp
// Coalesced access on GPU
Kokkos::parallel_for(num_rows, KOKKOS_LAMBDA(const int i) {
  CoordType y = y_coords[i];  // Warp loads contiguous y's
  CoordType z = z_coords[i];  // Warp loads contiguous z's
  // ...
});
```

**Key Considerations:**
- Only beneficial for 3D (2D has single coordinate)
- Requires separate views
- More complex indexing
- Must convert to/from standard Mesh type

### v8: Hybrid CPU-GPU

**Two-Phase Approach:**

Phase 1 (CPU - Serial):
```cpp
// Serial merge to find matching rows
std::vector<std::pair<IndexType, IndexType>> matching_rows;
IndexType i = 0, j = 0;
while (i < A.num_rows && j < B.num_rows) {
  if (A.row_keys[i] == B.row_keys[j]) {
    matching_rows.push_back({i, j});
    ++i; ++j;
  } else if (A.row_keys[i] < B.row_keys[j]) {
    ++i;
  } else {
    ++j;
  }
}
```

Phase 2 (GPU - Parallel):
```cpp
// Copy only matching rows to GPU
auto d_matching_A = Kokkos::create_mirror_view_and_copy(
    DeviceMemorySpace(), Kokkos::View<IndexType*, ...>(matching_A.data(), N));
auto d_matching_B = Kokkos::create_mirror_view_and_copy(
    DeviceMemorySpace(), Kokkos::View<IndexType*, ...>(matching_B.data(), N));

// Process intervals on GPU
Kokkos::parallel_for(matching_rows.size(), KOKKOS_LAMBDA(const int k) {
  IndexType row_A = d_matching_A[k];
  IndexType row_B = d_matching_B[k];
  // Process intervals...
});
```

**Key Considerations:**
- Only works with CUDA backend
- Host-device synchronization overhead
- Only beneficial if N_match << min(N_A, N_B)
- Requires dual code paths

### v9: Adaptive

**Decision Tree:**
```cpp
struct MeshCharacteristics {
  std::size_t size_A, size_B;
  double balance_ratio;           // min(size_A, size_B) / max(size_A, size_B)
  double coord_density;           // size / (max_coord - min_coord)
  bool uniform_stride;
  Pattern pattern;
};

Algorithm select_algorithm(const MeshCharacteristics& chars) {
  // Unbalanced meshes -> Hybrid (if CUDA) or Hash
  if (chars.balance_ratio < 0.1) {
    #ifdef KOKKOS_ENABLE_CUDA
      return Algorithm::Hybrid;
    #else
      return Algorithm::Hash;
    #endif
  }

  // Dense patterns -> Direct Index
  if (chars.coord_density > 0.8) {
    return Algorithm::DirectIndex;
  }

  // Uniform stride -> Direct Index
  if (chars.uniform_stride) {
    return Algorithm::DirectIndex;
  }

  // Large sparse -> Hash or Parallel Merge
  if (chars.size_A > 10000 && chars.coord_density < 0.1) {
    return chars.balance_ratio > 0.5 ? Algorithm::ParallelMerge : Algorithm::Hash;
  }

  // 3D meshes -> SoA
  if (DIM == 3) {
    return Algorithm::SoAOptimized;
  }

  // Default -> Baseline
  return Algorithm::Baseline;
}
```

**Key Considerations:**
- Analysis overhead (O(N) pass through data)
- Decision thresholds need tuning
- Must include all algorithm implementations
- Larger binary size

## Performance Considerations

### Memory Bandwidth

**v2 (Baseline):** Binary search = random access = poor cache locality
**v5 (Merge):** Sequential access = excellent cache locality
**v6 (Direct):** Direct indexing = excellent cache locality
**v7 (SoA):** Coalesced access = excellent GPU performance

### Parallel Efficiency

**v2 (Baseline):** Embarrassingly parallel (no synchronization)
**v4 (Hash):** Minimal contention with good hash function
**v5 (Merge):** Work imbalance possible
**v6 (Direct):** Embarrassingly parallel
**v8 (Hybrid):** CPU serial + GPU parallel = hybrid scaling

### Preprocessing Overhead

| Version | Preprocessing | Overhead |
|---------|--------------|----------|
| v2      | None         | 0%       |
| v4      | Build hash   | O(N_B)   |
| v5      | None         | 0%       |
| v6      | Detect pattern | O(N)   |
| v7      | Pack/unpack  | O(N)     |
| v8      | CPU merge    | O(N_A + N_B) |
| v9      | Analyze + select | O(N) |

### Memory Overhead

| Version | Extra Memory | Notes |
|---------|--------------|-------|
| v2      | 0%           | Baseline |
| v4      | ~40%         | Hash table (load factor 0.7) |
| v5      | 0%           | In-place merge |
| v6      | 0-100KB      | Lookup table for small range |
| v7      | 0%           | Different layout, same size |
| v8      | O(N_match)   | Matching row indices |
| v9      | Variable     | Depends on selection |

## Testing Strategy

### Correctness Verification

All versions must pass:
1. **Unit tests**: Test-specific functionality
2. **Cross-version tests**: Compare all outputs
3. **Overlap pattern tests**: Geometric edge cases
4. **Large mesh tests**: Stress test with realistic sizes

### Performance Validation

Benchmarks measure:
1. **Small configs** (<100 rows): Preprocessing overhead matters
2. **Dense configs**: Direct index should dominate
3. **Sparse configs**: Hash and merge should dominate
4. **3D configs**: SoA should show benefit
5. **Unbalanced configs**: Hybrid should shine

### Regression Testing

Run full suite on every change:
```bash
ctest --preset playground-serial -R intersection
```

## Future Enhancements

### Short Term

1. **SIMD Optimization**: Add explicit SIMD for CPU backends
2. **GPU Shared Memory**: Cache row keys in shared memory
3. **Adaptive Tuning**: Learn optimal thresholds at runtime
4. **Memory Profiling**: Track actual memory usage per version

### Medium Term

1. **Hierarchical Merge**: Multi-level parallel merge
2. **Perfect Hash**: Static perfect hash for dense meshes
3. **Batch Tuning**: Optimize batch sizes for hybrid
4. **Sparse Hash**: Sparse hash table for very large meshes

### Long Term

1. **Machine Learning**: Learn algorithm selection from data
2. **Auto-tuning**: Runtime optimization with feedback
3. **Custom Allocators**: Pool allocators for reduced overhead
4. **Compression**: Compress row keys for memory efficiency

## Implementation Checklist

When adding a new algorithm version:

- [ ] Implement in `algorithm/v<N>_name.hpp`
- [ ] Use unique namespace: `playground::subsetix::csr::intersection::name`
- [ ] Follow entry point signature
- [ ] Use shared utilities from `detail/utils.hpp`
- [ ] Add unit test in `tests/intersection/test_v<N>_unitary.cpp`
- [ ] Add to cross-version test
- [ ] Add to comparison benchmark
- [ ] Document in `ROW_MAPPING_ALTERNATIVES.md`
- [ ] Update CMakeLists.txt if needed
- [ ] Verify bit-identical results
- [ ] Test on all backends (Serial, OpenMP, CUDA)

## References

### Related Work

1. **Parallel Merge**: Blelloch 1990, "Prefix Sums and Their Applications"
2. **Hash Tables**: Knuth 1998, "The Art of Computer Programming, Vol. 3"
3. **SoA Layout**: NVIDIA 2020, "CUDA C Best Practices Guide"
4. **Hybrid CPU-GPU**: Gelado et al. 2010, "An Asymmetric Distributed Shared Memory Model"

### Internal Documentation

- `ROW_MAPPING_ALTERNATIVES.md` - User-facing algorithm comparison
- `PROFILING_STRATEGY.md` - Performance profiling techniques
- `PROFILING_QUICKREF.md` - Quick profiling reference

## License

```
SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
SPDX-License-Identifier: BSD-3-Clause
```
