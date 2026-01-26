# Overhead Analysis - Playground Intersection Module

**Date:** 2026-01-26
**Branch:** `reduce-overhead`
**Target:** Reduce ~500μs overhead on small meshes (19 rows)

---

## Executive Summary

Current small mesh performance: **~500μs per intersection**
**Actual computation time: ~50-100μs (10-20%)**
**Overhead: ~400-450μs (80-90%)** ← This is what we need to fix!

---

## Problem Breakdown

### 1. Memory Allocation Overhead (33 allocations per call)

**Current behavior:**
- 18 allocations for mesh conversion (9 per mesh × 2 meshes)
- 19-22 allocations for intersection algorithm
- Total: **~37-40 Kokkos::View allocations** per intersection

**Estimated cost: ~280-450μs (56-90% of total time)**

**Sources:**
| Phase | Allocations | Cost |
|-------|-------------|------|
| Conversion CommonMesh→Device | 18 | 50-100μs |
| Intersection - temp buffers | 8 | 100-150μs |
| Intersection - output mesh (first) | 3 | 20-30μs |
| Intersection - output mesh (compacted) | 3 | 100-150μs |
| Scalar views | 3 | 10-20μs |

**Root cause:** No workspace/pool pattern. Every intersection allocates from scratch.

---

### 2. Synchronization Overhead (7 sync points per call)

**Current behavior:**
- 5 explicit `Kokkos::fence()` calls
- 4 `Kokkos::deep_copy()` for scalars
- **Total: 9 synchronization points** per intersection

**Estimated cost: ~65-165μs (13-33%)**

| Sync Type | Count | Cost per | Total |
|-----------|-------|----------|-------|
| Explicit fence | 5 | 10-20μs | 50-100μs |
| deep_copy scalar | 4 | 5-15μs | 15-65μs |

**Root cause:** Over-synchronization. Many fences are redundant because:
- `parallel_scan` has implicit synchronization
- `deep_copy` already synchronizes
- Subsequent kernels don't need previous completion

---

### 3. Kernel Launch Overhead (13-18 launches per call)

**Current behavior:**
- Row mapping: 3 launches
- Row compaction: 2 launches
- Interval counting: 1 launch
- Scan: 2 launches
- Fill intervals: 1 launch
- Final compaction: 4-5 launches
- **Total: 13-18 kernel launches** per intersection

**Estimated cost: ~65μs (13%)**

**Special case:** Line 451 in baseline.hpp launches a kernel for **1 iteration**:
```cpp
Kokkos::parallel_for("intersection_compact_final_ptr",
    Kokkos::RangePolicy<ExecSpace>(0, 1),  // ONLY 1 ITERATION!
    KOKKOS_LAMBDA(const std::size_t) {
      compacted.row_ptr(final_num_rows) = out.row_ptr(num_rows_out);
    });
```

This is pure overhead (~5-10μs) and should be a host-side assignment.

---

### 4. Conversion Overhead (Double allocation pattern)

**Current behavior in `from_common()`:**
1. Allocate 3 device views (row_keys, row_ptr, intervals)
2. Allocate 3 MORE mirror views (even when same memory space!)
3. Fill data into mirrors
4. deep_copy mirrors→device (even when both are HostSpace!)

**Estimated cost: ~350-700μs per mesh conversion**

**Problem:** The code doesn't specialize for HostSpace:
- When `MemorySpace == HostSpace` (Serial backend), the mirror/deep_copy is **completely unnecessary**
- Still allocates 6 views instead of 3
- Still performs 3 deep_copy operations within the same memory space

---

### 5. Benchmark Measurement Overhead

**Current benchmark loop:**
```cpp
for (auto _ : state) {
    auto result = baseline::intersect_meshes_2d(mesh_a_, mesh_b_);
    benchmark::DoNotOptimize(result.num_rows);
    Kokkos::fence();  // Fence #1
    benchmark::DoNotOptimize(result.num_intervals);
    Kokkos::fence();  // Fence #2 - REDUNDANT!
}
```

**Issues:**
- **Redundant second fence**: adds ~10-50μs
- **No warm-up iterations**: First iteration includes CUDA context init
- **DoNotOptimize split**: Prevents some optimizations

---

## Priority Action Plan

### 🔴 Priority 1: Quick Wins (100-200μs reduction)

#### 1.1 Remove Redundant Benchmark Fence
**File:** `playground/intersection/benchmarks/intersection/regular_mesh_benchmark.cpp`

**Change:**
```cpp
// Before (lines 163-167):
benchmark::DoNotOptimize(result.num_rows);
Kokkos::fence();
benchmark::DoNotOptimize(result.num_intervals);
Kokkos::fence();

// After:
benchmark::DoNotOptimize(result.num_rows);
benchmark::DoNotOptimize(result.num_intervals);
Kokkos::fence();  // Single fence at end
```

**Expected gain: ~10-50μs per iteration**

#### 1.2 Remove Single-Iteration Kernel
**File:** `playground/intersection/include/playground/subsetix/csr/intersection/algorithm/baseline.hpp`

**Replace lines 451-456:**
```cpp
// Before:
Kokkos::parallel_for("intersection_compact_final_ptr",
    Kokkos::RangePolicy<ExecSpace>(0, 1),
    KOKKOS_LAMBDA(const std::size_t) {
      compacted.row_ptr(final_num_rows) = out.row_ptr(num_rows_out);
    });

// After:
auto row_ptr_host = Kokkos::create_mirror_view(Kokkos::HostSpace{}, out.row_ptr);
Kokkos::deep_copy(row_ptr_host, out.row_ptr);
compacted.row_ptr(final_num_rows) = row_ptr_host(num_rows_out);
Kokkos::deep_copy(compacted.row_ptr, row_ptr_host);
```

**Expected gain: ~5-10μs per iteration**

#### 1.3 Remove Unnecessary Explicit Fences
**Files:** `baseline.hpp`, `optimized.hpp`

**Remove lines 249, 267, 296** - These fences are redundant because:
- `parallel_scan` synchronizes implicitly
- `deep_copy` synchronizes automatically
- Next kernel doesn't need previous completion

**Expected gain: ~30-60μs per iteration**

**Total Priority 1 gain: ~45-120μs (9-24% reduction)**

---

### 🟡 Priority 2: Algorithm Improvements (150-300μs reduction)

#### 2.1 Implement Workspace Pattern
**File:** Create `playground/intersection/include/playground/subsetix/csr/intersection/workspace.hpp`

**Reference:** `include/subsetix/csr_ops/workspace.hpp` (already exists in stable codebase)

**Implementation:**
```cpp
template <typename ExecSpace>
struct IntersectionWorkspace {
    // Reusable buffers for common operations
    Kokkos::View<int*, typename ExecSpace::memory_space> int_buf_;
    Kokkos::View<std::size_t*, typename ExecSpace::memory_space> size_buf_;
    std::size_t capacity_ = 0;

    void ensure_capacity(std::size_t required) {
        if (required > capacity_) {
            int_buf_ = Kokkos::View<int*, ...>("int_buf", required);
            size_buf_ = Kokkos::View<std::size_t*, ...>("size_buf", required);
            capacity_ = required;
        }
    }
};
```

**Usage:** Pre-allocate in benchmark `SetUp()`, reuse across iterations.

**Expected gain: ~150-300μs per iteration** (eliminates most allocation overhead)

#### 2.2 Specialize Conversion for HostSpace
**File:** `playground/intersection/tests/intersection/test_common_format.hpp`

**Add specialization in `from_common()`:**
```cpp
static DeviceMesh from_common(const CommonMesh& common) {
    DeviceMesh mesh;
    mesh.num_rows = common.rows.size();
    mesh.num_intervals = /* count */;

    // Allocate views
    mesh.row_keys = Kokkos::View<...>("row_keys", mesh.num_rows);
    mesh.row_ptr = Kokkos::View<...>("row_ptr", mesh.num_rows + 1);
    mesh.intervals = Kokkos::View<...>("intervals", mesh.num_intervals);

    if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
        // FAST PATH: Fill directly without mirror/deep_copy
        for (std::size_t i = 0; i < common.rows.size(); ++i) {
            mesh.row_keys(i) = /* ... */;
            mesh.row_ptr(i) = /* ... */;
        }
    } else {
        // CUDA PATH: Use mirror pattern (current code)
        auto keys_h = Kokkos::create_mirror_view(mesh.row_keys);
        // ... fill mirrors ...
        Kokkos::deep_copy(mesh.row_keys, keys_h);
    }

    return mesh;
}
```

**Expected gain: ~200-400μs per mesh conversion** (only affects Serial backend)

**Total Priority 2 gain: ~350-700μs (70-140% reduction!)**

---

### 🟢 Priority 3: Advanced Optimizations (50-150μs reduction)

#### 3.1 Replace Scalar deep_copy with Kernel-Based Reduction
**Current:** Use device scalar view + deep_copy
**Better:** Use `Kokkos::parallel_reduce` for single scalar result

**Example:**
```cpp
// Before (line 252):
Kokkos::View<std::size_t, DeviceMemorySpace> num_rows_out_view("num_rows_out");
Kokkos::parallel_scan(..., num_rows_out_view);
Kokkos::deep_copy(num_rows_out_host, num_rows_out_view);

// After:
std::size_t num_rows_out_host = 0;
Kokkos::parallel_scan(...,
    KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final) {
        // ... scan logic ...
        if (final) update = num_rows;
    }, num_rows_out_host);  // Direct reduction!
```

**Expected gain: ~15-45μs** (eliminates 3 deep_copy calls)

#### 3.2 Lazy Compaction
**Idea:** Skip compaction phase, return mesh with possible empty rows.
**Benefit:** Eliminates 4-5 kernel launches when compaction isn't needed.
**Trade-off:** Caller must handle sparse meshes.

**Expected gain: ~40-100μs** (when compaction skippable)

**Total Priority 3 gain: ~55-145μs (11-29% reduction)**

---

## Expected Results

### Before Optimization
| Metric | Value |
|--------|-------|
| Small mesh (19 rows) | ~500μs |
| Actual computation | ~50-100μs (10-20%) |
| Overhead | ~400-450μs (80-90%) |

### After All Optimizations
| Priority | Time Reduction | Expected Time |
|----------|---------------|---------------|
| Baseline | 0% | ~500μs |
| Priority 1 | 9-24% | ~380-455μs |
| Priority 1+2 | 79-164% | **~130-190μs** |
| Priority 1+2+3 | 90-193% | **~85-155μs** |

**Target: Reduce overhead from 400-450μs to 50-100μs**
**Result: Overhead becomes 30-50% of total time instead of 80-90%**

---

## File Changes Summary

| Priority | File | Lines Changed | Type |
|----------|------|---------------|------|
| 1.1 | `regular_mesh_benchmark.cpp` | 163-167 | Remove fence |
| 1.2 | `baseline.hpp`, `optimized.hpp` | 451-456 | Remove kernel |
| 1.3 | `baseline.hpp`, `optimized.hpp` | 249, 267, 296 | Remove fences |
| 2.1 | NEW: `workspace.hpp` | - | Add file |
| 2.1 | `baseline.hpp`, `optimized.hpp` | - | Use workspace |
| 2.2 | `test_common_format.hpp` | 169-206 | Add HostSpace specialize |
| 3.1 | `baseline.hpp`, `optimized.hpp` | 252, 345, 404 | Replace deep_copy |
| 3.2 | `baseline.hpp`, `optimized.hpp` | 396-464 | Make optional |

---

## Next Steps

1. **Commit this analysis** to `reduce-overhead` branch
2. **Implement Priority 1** changes (quick wins)
3. **Benchmark** to measure improvement
4. **Implement Priority 2** (workspace pattern)
5. **Benchmark** again
6. **Consider Priority 3** based on results

---

## References

- Agent reports stored in Claude session:
  - Conversion analysis: `agentId: ae2141a`
  - GPU kernel analysis: `agentId: a663478`
  - Memory analysis: `agentId: ae4cbde`
  - Benchmark analysis: `agentId: a70404a`

- Key files:
  - `playground/intersection/include/playground/subsetix/csr/intersection/algorithm/baseline.hpp`
  - `playground/intersection/include/playground/subsetix/csr/intersection/algorithm/optimized.hpp`
  - `playground/intersection/tests/intersection/test_common_format.hpp`
  - `playground/intersection/benchmarks/intersection/regular_mesh_benchmark.cpp`
  - `include/subsetix/csr_ops/workspace.hpp` (reference implementation)
