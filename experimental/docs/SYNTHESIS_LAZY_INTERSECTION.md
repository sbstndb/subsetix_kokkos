# Synthesis: Lazy Intersection Experimental Module

## Executive Summary

This document synthesizes research findings for implementing a lazy intersection system in subsetix_kokkos, comparing three approaches:
1. **Naive**: Repeated allocations (baseline)
2. **Workspace Reuse**: Pre-allocated buffers reused
3. **Graph DAG**: Lazy evaluation with Kokkos::Experimental::Graph

---

## File Structure

```
experimental/include/experimental/subsetix/csr/set_algebra/
├── successive_intersection.hpp          # Master header, unified API
└── detail/
    ├── successive_intersection_naive.hpp      # Naive implementation
    ├── successive_intersection_workspace.hpp  # Workspace implementation
    └── successive_intersection_graph.hpp      # Graph DAG implementation

experimental/tests/set_algebra/
└── test_successive_intersection.cpp          # Cross-validation tests

experimental/benchmarks/set_algebra/
└── successive_intersection_benchmark.cpp     # Performance comparison
```

---

## Key API Signatures

```cpp
namespace experimental::subsetix::csr::successive {

// Unified API - strategy-based dispatch
enum class Strategy { Naive, Workspace, Graph };

template <int DIM>
struct Config {
    Strategy strategy = Strategy::Naive;
    struct {
        std::size_t max_rows = 0;
        std::size_t max_intervals = 0;
        double growth_factor = 1.5;
    } workspace;
};

template <int DIM>
Mesh<DIM> intersect_successive(
    const std::vector<Mesh<DIM>>& meshes,
    const Config<DIM>& config = {});

// Per-strategy APIs
namespace naive {
    template <int DIM>
    Mesh<DIM> intersect(const std::vector<Mesh<DIM>>& meshes);
}

namespace workspace {
    struct Workspace { /* pre-allocated buffers */ };
    template <int DIM>
    Mesh<DIM> intersect(const std::vector<Mesh<DIM>>& meshes, Workspace& ws);
}

namespace graph {
    template <int DIM>
    Mesh<DIM> intersect(const std::vector<Mesh<DIM>>& meshes);
}

} // namespace experimental::subsetix::csr::successive
```

---

## Implementation Priority

### Phase 1: Naive (1 day)
- Wrap existing `v3::intersect_meshes` in a loop
- Add timing/allocation tracking
- Baseline for correctness

### Phase 2: Workspace (2 days)
- Implement `IntersectionWorkspace` with ping-pong buffers
- Port `UnifiedCsrWorkspace` pattern from stable version
- Add `ensure_capacity` for growth

### Phase 3: Graph (2-3 days)
- Build `Kokkos::Experimental::Graph` for entire chain
- Pre-allocate temporary storage
- Single graph submission

### Phase 4: Tests & Benchmarks (2 days)
- Cross-validation (all approaches must match)
- Benchmarks with MinTime(3.0)
- Generate comparison report

---

## Testing Strategy

### Correctness Tests
```cpp
// All three approaches MUST produce identical results
TEST(CrossValidation, FourMeshes) {
    auto r_naive = naive::intersect({A,B,C,D});
    auto r_ws = workspace::intersect({A,B,C,D}, ws);
    auto r_graph = graph::intersect({A,B,C,D});
    EXPECT_EQ(r_naive, r_ws);
    EXPECT_EQ(r_naive, r_graph);
}
```

### Benchmark Configurations
- **Small**: y_max=64, ~19 rows (fast iterations)
- **Medium**: y_max=512, ~154 rows (standard)
- **Large**: y_max=4096, ~1229 rows (stress test)

### Chain Lengths
- 2 meshes (baseline)
- 4 meshes (typical AMR)
- 8 meshes (deep refinement)

---

## Expected Outcomes

| Approach | Allocations | Sync Points | Expected Speedup |
|----------|-------------|-------------|------------------|
| Naive | N-1 | 4(N-1) | 1.0x (baseline) |
| Workspace | 1 | 4(N-1) | 1.2-1.5x |
| Graph | 1 | 1 | 2-5x (GPU) |

---

## Critical Files

1. `successive_intersection.hpp` - Unified API
2. `detail/successive_intersection_naive.hpp` - Baseline
3. `detail/successive_intersection_workspace.hpp` - Memory optimization
4. `test_successive_intersection.cpp` - Correctness validation
5. `successive_intersection_benchmark.cpp` - Performance comparison

---

## Next Steps

1. Create file structure
2. Implement Naive (use existing v3)
3. Implement Workspace (reuse UnifiedCsrWorkspace)
4. Implement Graph (Kokkos::Experimental::Graph)
5. Add tests and benchmarks
6. Generate final report
