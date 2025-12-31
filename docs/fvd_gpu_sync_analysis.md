# FVD GPU Synchronization Analysis & Kokkos Graph Solution

## Problem Statement: Excessive Host-Device Synchronizations

### Current Architecture Synchronization Points

Let me analyze the **current FVD high-level API** to identify ALL synchronization points:

---

## 1. TIME INTEGRATION (RK2/RK3/RK4)

### Current Implementation (BAD for GPU)

```cpp
// time/time_integrators.hpp - rk_step function
template<System, TimeIntegrator>
KOKKOS_FUNCTION
void rk_step(const View<Conserved**>& U, Real dt, Real t, auto&& rhs, ...)
{
    // This function is HOST-side (KOKKOS_FUNCTION = can be called from host or device)
    // But the loop below is on HOST!

    for (int s = 0; s < stages; ++s) {  // ← HOST loop
        Real stage_time = t + Integrator::c[s] * dt;

        if (s == 0) {
            rhs(U, stage_storage, s, stage_time);  // ← Device kernel
            // IMPLICIT SYNC HERE!
        } else {
            detail::compute_stage_solution(...);   // ← Device kernel
            // IMPLICIT SYNC HERE!
            rhs(stage_solution, stage_storage, s, stage_time);  // ← Device kernel
            // IMPLICIT SYNC HERE!
        }
    }

    detail::combine_stages(U, dt, stage_storage);  // ← Device kernel
    // FINAL SYNC HERE!
}
```

### Problem Analysis

**For RK3 with 1000 timesteps:**
- 3 stages per step × 1000 steps = **3000 kernel launches**
- Each kernel launch = **implicit sync** when it completes
- Between stages, host must compute next stage time
- **Result: 3000+ host-device syncs per simulation**

**Overhead per sync (CUDA):**
- ~5-50 µs for kernel launch completion
- Host-device communication overhead
- Driver overhead

**Total overhead**: 3000 × 10 µs = **30 ms minimum** just in sync overhead!

For a simulation that should take 100 ms, that's **30% overhead**.

---

## 2. OBSERVER CALLBACKS

### Current Implementation

```cpp
// solver/observer.hpp
void ObserverManager::notify(SolverEvent event, const SolverState<Real>& state)
{
    // This is HOST code
    for (const auto& [id, callback] : it->second) {
        callback(event, state);  // Host-side lambda!
    }
}

// In solver (pseudocode):
for (int s = 0; s < stages; ++s) {
    // Compute stage...
    observers_.notify(SubStepEnd, state);  // ← SYNC + host callback!
}
```

### Problem Analysis

If we notify observers for every RK substep:
- **3 substeps × 1000 timesteps = 3000 notifications**
- Each notification requires:
  1. Device → Host sync (to get state for callback)
  2. Host callback execution
  3. Host → Device sync (to continue)

**Result: 6000+ additional syncs** if observers are used per substep!

---

## 3. TIME-DEPENDENT BOUNDARY CONDITIONS

### Current Implementation

```cpp
// boundary/time_dependent_bc.hpp
class BcManager {
    // Host-side storage
    std::vector<Descriptor> descriptors_;  // HOST

    // Device-side registry
    BcRegistry<System> device_registry_;   // DEVICE

    void sync_to_device() {
        registry_.rebuild_lookup();  // Rebuild on device
        // Requires deep_copy or similar
    }
};

// User usage:
solver.boundary_manager().update_bc("left", 0, new_type, new_value);
solver.boundary_manager().sync_to_device();  // ← SYNC!
```

### Problem Analysis

Every time a BC changes:
1. Host modifies BC data
2. `sync_to_device()` copies to device
3. **Sync occurs**

For frequently changing BCs (e.g., every timestep):
- **1000 timesteps = 1000 syncs**

---

## 4. ADAPTIVE TIME STEPPING

### Current Implementation

```cpp
Real compute_max_cfl() {
    // Device reduction kernel
    Real max_cfl;
    Kokkos::parallel_reduce(..., max_cfl);  // ← Device kernel
    // IMPLICIT SYNC HERE - must wait for reduction result!
    return max_cfl;
}

void step_adaptive() {
    Real current_cfl = compute_max_cfl();  // ← SYNC!
    if (dt_controller_.should_adapt(current_cfl, cfg)) {
        dt_ = dt_controller_.compute_dt(dt_, current_cfl, cfg);  // Host computation
    }
    step(dt_);
}
```

### Problem Analysis

For adaptive dt with CFL checking every timestep:
- **1000 timesteps = 1000 reduction syncs**

Reductions are particularly expensive because:
- All threads must complete
- Result must be copied back to host
- Host makes decision, then new kernel launched

---

## 5. AMR REMESHING

### Current Implementation

```cpp
void check_remesh() {
    // Evaluate criteria on device
    auto refinement_flags = evaluate_refinement(...);  // Device kernel

    // Copy flags to host
    auto host_flags = Kokkos::create_mirror_view(refinement_flags);
    Kokkos::deep_copy(host_flags, refinement_flags);  // ← SYNC!

    // Decide on host
    bool should_remesh = analyze_flags(host_flags);

    if (should_remesh) {
        perform_remesh();  // Many syncs
    }
}
```

### Problem Analysis

For AMR with stride = 100:
- 1000 timesteps / 100 = **10 remesh checks**
- Each remesh can involve **10-100 syncs** (tree analysis, data redistribution)
- Total: **100-1000 syncs** for AMR alone

---

## 6. CHECKPOINT/OUTPUT

### Current Implementation

```cpp
void write_checkpoint(const std::string& filename) {
    // Copy data to host
    auto host_U = Kokkos::create_mirror_view(U_);
    Kokkos::deep_copy(host_U, U_);  // ← SYNC!

    // Write on host
    write_to_file(host_U, filename);
}
```

For checkpoints every 100 steps:
- 1000 / 100 = **10 large syncs**

---

## TOTAL SYNCHRONIZATION COUNT

| Feature | Syncs per 1000 steps | Overhead (approx) |
|---------|---------------------|-------------------|
| RK stages (RK3) | 3000+ | 30-50 ms |
| Observer callbacks | 3000+ (if per substep) | 20-40 ms |
| BC updates | 100-1000 (if frequent) | 1-10 ms |
| Adaptive dt | 1000 (reductions) | 10-20 ms |
| AMR remeshing | 100-1000 | 10-50 ms |
| Checkpoints | 10 | 5-10 ms |
| **TOTAL** | **~7000 syncs** | **75-180 ms** |

For a simulation that should take ~100 ms on GPU:
- **75-180% overhead** due to syncs!

---

## SOLUTION: Kokkos Graphs

### What are Kokkos Graphs?

Kokkos Graphs (Kokkos::Experimental::fork_graph, etc.) allow:
- **Batching multiple kernels into a single DAG**
- **Reducing kernel launch overhead**
- **Enabling CUDA Graphs** (when available)
- **Overlapping host work with device execution**

### Key Benefits

1. **Single launch for entire timestep**
2. **No intermediate syncs** between stages
3. **Automatic dependency management**
4. **CUDA Graph support** (even better performance)

---

## Proposed Graph-Based Architecture

### Option A: Timestep Graph (Recommended)

```cpp
template<System, TimeIntegrator>
class TimestepGraph {
public:
    using Graph = Kokkos::Experimental::TypeErasedGraph;
    using GraphNode = Graph::node_type;

    void build(const SolverState& state) {
        // Create graph for ONE complete timestep
        graph_ = Graph();

        // Stage 0
        auto rhs0_node = graph_.create_node(
            [&](Graph& g) {
                // RHS kernel for stage 0
                return g.create_kernel(
                    "rhs_stage0",
                    compute_rhs_policy,
                    KOKKOS_LAMBDA(int j, int i) {
                        // RHS computation
                    }
                );
            }
        );

        // Stage 1
        auto solution1_node = graph_.create_node(
            [&](Graph& g) {
                return g.create_kernel(
                    "compute_stage1",
                    stage_solution_policy,
                    KOKKOS_LAMBDA(int j, int i) {
                        // U_1 = U_0 + dt * a[1][0] * k_0
                    }
                );
            }
        );

        auto rhs1_node = graph_.create_node(
            [&](Graph& g) {
                return g.create_kernel(
                    "rhs_stage1",
                    compute_rhs_policy,
                    KOKKOS_LAMBDA(int j, int i) {
                        // RHS for stage 1
                    }
                );
            }
        );

        // ... add all stages

        // Combine stages (final update)
        auto combine_node = graph_.create_node(
            [&](Graph& g) {
                return g.create_kernel(
                    "combine_stages",
                    combine_policy,
                    KOKKOS_LAMBDA(int j, int i) {
                        // U_{n+1} = U_n + dt * sum(b[i] * k_i)
                    }
                );
            }
        );

        // Build dependency chain
        rhs0_node >> solution1_node >> rhs1_node >> ... >> combine_node;

        // Compile graph
        graph_.compile();
    }

    void execute() {
        // SINGLE LAUNCH for entire timestep!
        graph_.execute();
    }

private:
    Graph graph_;
};
```

### Usage Pattern

```cpp
// Initialization (once)
TimestepGraph<System, Kutta3<float>> timestep_graph;
timestep_graph.build(initial_state);

// Main loop
for (int step = 0; step < nsteps; ++step) {
    // Update graph parameters (dt, t) if needed
    timestep_graph.update_parameters(dt, t);

    // SINGLE GRAPH EXECUTION = NO INTERMEDIATE SYNCS!
    timestep_graph.execute();

    // Optional: sync after full timestep for observers
    if (step % observer_interval == 0) {
        // Copy state to host ONCE per N steps
        Kokkos::deep_copy(host_state, device_state);
        observers_.notify(StepEnd, host_state);
    }
}
```

### Benefits

- **1 sync per timestep** instead of 3-7 syncs
- **Graph execution is batched** by CUDA
- **Kernel launch overhead reduced** by 10-100x
- **CUDA Graphs** can be enabled for even better performance

---

## Option B: Full Simulation Graph (More Complex)

Graph that includes:
- Multiple timesteps
- Remeshing
- Checkpointing
- Observer notifications (batched)

```cpp
class SimulationGraph {
public:
    void build() {
        // Create graph for 100 timesteps at a time
        for (int i = 0; i < 100; ++i) {
            auto step_node = add_timestep_node();
            if (i % 10 == 0) {
                auto remesh_node = add_remesh_node();
                auto checkpoint_node = add_checkpoint_node();
                step_node >> remesh_node >> checkpoint_node >> next_step_node;
            } else {
                step_node >> next_step_node;
            }
        }

        // Add observer batch at end
        auto observer_node = add_observer_node();
        last_step_node >> observer_node;

        graph_.compile();
    }

    void execute() {
        // Executes 100 timesteps with ONLY 1 sync at the end!
        graph_.execute();
    }
};
```

### Benefits

- **1 sync per 100 timesteps** instead of 300+
- **Maximum GPU utilization**
- **Host only involved** every 100 steps

### Drawbacks

- **Complex to implement**
- **Less flexible** (can't easily change things mid-simulation)
- **Memory intensive** (graph can be large)

---

## Option C: Hybrid Approach (Recommended for FVD)

Combine graphs for performance with flexibility:

```cpp
class HybridSolver {
public:
    void step() {
        // Use graph for timestep (fast path)
        if (!needs_remesh_ && !observers_need_notification()) {
            timestep_graph_.execute();  // NO SYNC
        } else {
            // Fall back to step-by-step for flexibility
            step_by_step();
        }
    }

private:
    TimestepGraph timestep_graph_;

    bool needs_remesh() const {
        return step_count_ % remesh_stride_ == 0;
    }

    bool observers_need_notification() const {
        return step_count_ % observer_stride_ == 0;
    }
};
```

### Benefits

- **Fast path** (graph) for most steps
- **Flexible path** (step-by-step) when needed
- **Balance of performance and flexibility**

---

## IMPLEMENTATION STRATEGY

### Phase 1: Minimal Graph Integration

1. **Create `TimestepGraph` class** that wraps RK stages
2. **Modify AdaptiveSolver** to use graph when possible
3. **Benchmark** vs current implementation

### Phase 2: Optimize Synchronization Points

1. **Batch observer notifications** (notify every N steps, not every substep)
2. **Reduce BC syncs** (only sync when actually changed)
3. **Fuse reductions** (CFL computation within graph)

### Phase 3: Advanced Features

1. **Add remeshing to graph**
2. **Add checkpointing to graph**
3. **Enable CUDA Graphs explicitly**

---

## Kokkos Graph API Summary

### Key Functions

```cpp
#include <Kokkos_Core.hpp>
#include <Kokkos_Experimental_Graph.hpp>

// Create a graph
Kokkos::Experimental::TypeErasedGraph graph;

// Create nodes
auto node1 = graph.create_node([&](auto& g) {
    return g.create_kernel("kernel1", policy, kernel1_lambda);
});

auto node2 = graph.create_node([&](auto& g) {
    return g.create_kernel("kernel2", policy, kernel2_lambda);
});

// Create dependency: node1 → node2
node1 >> node2;

// Compile graph
graph.compile();

// Execute graph (single launch!)
graph.execute();
```

### Requirements

- Kokkos 4.0+
- CUDA 11.0+ (for CUDA Graphs)
- HIP 5.0+ (for ROCm graphs)

---

## RECOMMENDATION

### YES, use Kokkos Graphs for FVD!

**Reasons:**

1. **Massive performance gain**: 10-100x reduction in kernel launch overhead
2. **Better GPU utilization**: No idle time between stages
3. **Scalability**: Scales better with problem size
4. **Future-proof**: CUDA Graphs are the direction of GPU computing

### Implementation Priority

1. **High**: Timestep graph (RK stages)
2. **Medium**: Reduction fusion (CFL computation)
3. **Low**: Full simulation graph (too complex for now)

### Expected Performance Improvement

| Metric | Current | With Graphs | Improvement |
|--------|---------|-------------|-------------|
| Kernel launches/step | 3-7 | 1 | 3-7x |
| Syncs/step | 3-7 | 0-1 | 3-7x |
| Timestep time | 100 µs | 30-50 µs | 2-3x |
| Total simulation | 100 ms | 30-50 ms | 2-3x |

---

## ALTERNATIVE: Kernel Fusion

If graphs are not available, fuse operations into single kernels:

```cpp
// Instead of 3 kernels for RK3:
KOKKOS_LAMBDA(int j, int i) {
    // Stage 0
    Conserved k0 = compute_rhs(U, j, i);

    // Stage 1
    Conserved U1 = U(j,i) + dt * Real(0.5) * k0;
    Conserved k1 = compute_rhs_at(U1, j, i);

    // Stage 2
    Conserved U2 = U(j,i) + dt * Real(-1.0) * k0 + dt * Real(2.0) * k1;
    Conserved k2 = compute_rhs_at(U2, j, i);

    // Combine
    U(j,i) += dt * (Real(1.0/6.0) * k0 + Real(2.0/3.0) * k1 + Real(1.0/6.0) * k2);
}
```

**Pros**: Single kernel, no syncs
**Cons**: Less flexible, harder to maintain, higher register pressure

---

## CONCLUSION

The current FVD API has **7000+ host-device syncs** for a typical simulation.
Using Kokkos Graphs can reduce this to **~100 syncs**, a **70x reduction**.

**Recommendation**: Implement Phase 1 (TimestepGraph) immediately, then benchmark.
