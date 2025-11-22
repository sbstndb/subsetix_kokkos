# Towards a Kokkos‑First Design

This document proposes a roadmap to make the whole subsetix core "Kokkos‑first":  
Kokkos data structures become the primary representation (both on device and host),
and `std::vector`/ad‑hoc host structs are pushed to the edges (I/O, tests, quick
construction utilities).

The goal is:
- Minimize Host↔Device transitions in hot code paths.
- Have a single conceptual data model for geometry and fields across backends.
- Keep the library usable from "plain C++" (VTK export, tests, examples) without
  forcing Kokkos everywhere in user code.

This is a **plan**, not an implementation. It is organized into phases that can
be executed incrementally.

---

## 0. Current State (Short Diagnosis)

### Geometry

- Device geometry is represented by `IntervalSet2DView<MemorySpace>`:
  - Type aliases: `IntervalSet2DDevice`, `IntervalSet2DHostView`.
  - Fields: `row_keys`, `row_ptr`, `intervals`, `cell_offsets`,
    plus counters `num_rows`, `num_intervals`, `total_cells`.
  - All set‑algebra kernels (`set_union_device`, `set_intersection_device`,
    `set_difference_device`, `set_symmetric_difference_device`) are written
    against `IntervalSet2DDevice`.

- Host geometry has an additional owning struct:
  - `IntervalSet2DHost` with `std::vector<RowKey2D> row_keys`, etc.,
    plus `rebuild_mapping()` to maintain `cell_offsets` and `total_cells`.
  - Conversions:
    - `build_device_from_host(const IntervalSet2DHost&) -> IntervalSet2DDevice`.
    - `build_host_from_device(const IntervalSet2DDevice&) -> IntervalSet2DHost`.
  - Tests and examples use `IntervalSet2DHost` heavily to build geometries on CPU.

### Fields

- Host fields:
  - `IntervalField2DHost<T>` (std::vector‑based CSR of `FieldInterval` + `values`).
  - Used pervasively in tests and VTK export.

- Device fields:
  - `Field2D<T, MemorySpace>` with aliases:
    - `Field2DDevice<T> = Field2D<T, DeviceMemorySpace>`.
    - `Field2DHost<T>   = Field2D<T, HostMemorySpace>` (rarely used).
  - Composition:
    - `geometry` is an `IntervalSet2DView<MemorySpace>`.
    - `values` is a `Kokkos::View<T*, MemorySpace>`.

- Conversions:
  - `build_device_field_from_host(const IntervalField2DHost<T>&) -> Field2DDevice<T>`.
  - `build_host_field_from_device(const Field2DDevice<T>&) -> IntervalField2DHost<T>`.
  - `make_field_like_geometry(const IntervalSet2DHost&, const T& init_value)`.

### Multilevel

- `MultilevelGeo<MemorySpace>` and `MultilevelField<T, MemorySpace>` are already
  Kokkos‑first: they store `IntervalSet2DView<MemorySpace>` and `Field2D<T, MemorySpace>`.
- Deep copies device→host are implemented in `multilevel.hpp` using
  `create_mirror_view_and_copy`.

### Ops / Workspaces / Examples

- Set algebra, AMR, remap, stencil, threshold, morphology, transform:
  - All operate on `IntervalSet2DDevice` and `Field2DDevice<T>`.
  - Use `CsrSetAlgebraContext` + `UnifiedCsrWorkspace` for scratch buffers.

- Host‑side logic:
  - Examples and tests often:
    1. Build `IntervalSet2DHost` / `IntervalField2DHost<T>` on CPU.
    2. Convert to device (`build_device_from_host` / `build_device_field_from_host`).
    3. Run device ops.
    4. Convert back to host for asserts or VTK.

This architecture is already device‑centric for heavy computations, but:
- Host structs duplicate information and conversion logic.
- Host and device worlds are conceptually separate.
- There is no single abstraction for "owning Kokkos geometry/fields" that can live
  both on host and device.

---

## 1. Target Architecture (Kokkos‑First)

We want:

1. **A single logical representation** of geometry and fields:
   - Geometry = CSR (rows, intervals, offsets, counters) implemented via Kokkos views.
   - Fields   = geometry + `values` view.

2. **Owning wrappers** in terms of Kokkos:
   - Remove most of the ad‑hoc `std::vector` owners.
   - Provide Kokkos‑aware owning types that can live in `HostSpace` and `DeviceMemorySpace`.

3. **Conversions via Kokkos only**:
   - `HostSpace` ↔ `DeviceMemorySpace` conversions use `create_mirror_view_and_copy`
     or `deep_copy` between views.
   - Where needed, adapt to `std::vector` only at I/O boundaries (VTK, user APIs
     for non‑Kokkos code).

4. **Public API streamlined**:
   - Expose a small, consistent set of types:
     - `IntervalSet2DHostKokkos`, `IntervalSet2DDevice`.
     - `Field2DHost<T>`, `Field2DDevice<T>`.
     - Multilevel variants on top of those.
   - Provide explicit "legacy" helpers for `IntervalSet2DHost` / `IntervalField2DHost<T>`
     only for compatibility.

The rest of this document breaks the migration into concrete phases.

---

## 2. Phase A – Core Utilities for View/Host Interop

**Goal:** Centralize all host↔device copy logic through a small set of reusable
helpers, and stop writing manual loops.

### A.1. Add generic copy helpers

Add a header `include/subsetix/detail/view_copy_utils.hpp` (or extend
`memory_utils.hpp`) with:

- `vector_to_view` (host vector → View on given memory space):

```cpp
template <class T, class MemorySpace>
Kokkos::View<T*, MemorySpace>
vector_to_view(const std::vector<T>& v, const std::string& label);
```

Semantics:
- Allocates `Kokkos::View<T*, MemorySpace>(label, v.size())`.
- Creates a `HostSpace` mirror, fills it with `std::copy(v.begin(), v.end(), mirror.data())`.
- `deep_copy` to the target view and returns it.

- `view_to_vector` (any 1D View → `std::vector`):

```cpp
template <class ViewType>
std::vector<typename ViewType::non_const_value_type>
view_to_vector(const ViewType& view);
```

Semantics:
- Uses `create_mirror_view_and_copy` to get a host mirror.
- Constructs a `std::vector` sized to `view.extent(0)` and `std::copy`s from the mirror.

### A.2. Use these helpers for IntervalSet conversions

In `csr_interval_set.hpp`:

- Replace manual loops in `build_device_from_host`:
  - Use `vector_to_view` to create `row_keys`, `row_ptr`, `intervals`, `cell_offsets`
    on device.
  - Preserve the existing logic that checks/rebuilds `cell_offsets` and `total_cells`
    on the host struct before copy.

- Replace manual loops in `build_host_from_device`:
  - Use `view_to_vector` to populate `row_keys`, `row_ptr`, `intervals`.
  - Call `rebuild_mapping()` to recompute `cell_offsets` and `total_cells`.

### A.3. Use these helpers for Field conversions

In `csr_field.hpp`:

- `build_device_field_from_host`:
  - Keep the geometry path: constructing `IntervalSet2DHost` then
    `build_device_from_host`.
  - Replace the per‑element copy from `host.values` into a host view by
    `vector_to_view<T, DeviceMemorySpace>(host.values, label_for_values)`.

- `build_host_field_from_device`:
  - Keep the geometry reconstruction via `build_host_from_device`.
  - Replace the manual loop from device `values` to `host.values` by `view_to_vector`.

Result: all host↔device copies are handled by two generic helpers, which will
be reused when we introduce Kokkos‑owning host types.

---

## 3. Phase B – Introduce Kokkos‑Owning Host Types

**Goal:** Provide Kokkos‑based owning types on host (`HostSpace`) for geometry
and fields, and reduce reliance on `std::vector` host structs inside the library.

### B.1. Geometry: IntervalSet2DHostView as primary host representation

Today:
- Device: `IntervalSet2DDevice` (`IntervalSet2DView<DeviceMemorySpace>`).
- Host for kernels: `IntervalSet2DHostView` (`IntervalSet2DView<HostMemorySpace>`).
- Host owning: `IntervalSet2DHost` (`std::vector`).

Target:
- Introduce an "owning" host type built on top of `IntervalSet2DHostView`:

```cpp
template <class MemorySpace>
struct IntervalSet2DOwning {
  IntervalSet2DView<MemorySpace> view;
  // Possibly: label, allocators, capacity info.
};

using IntervalSet2DHostKokkos = IntervalSet2DOwning<HostMemorySpace>;
using IntervalSet2DDeviceKokkos = IntervalSet2DOwning<DeviceMemorySpace>;
```

However, to keep the migration incremental and small:

1. **First step**: re‑use `IntervalSet2DHostView` as the canonical Kokkos host
   representation and provide helpers:

   - `IntervalSet2DHostView make_host_view_from_host(const IntervalSet2DHost&)`.
   - `IntervalSet2DHost make_host_from_host_view(const IntervalSet2DHostView&)`.

2. Internally, prefer `IntervalSet2DHostView` in new components instead of
   `IntervalSet2DHost`, and keep `IntervalSet2DHost` as a legacy construction/IO type.

3. In a second step (optional, more intrusive), evolve `IntervalSet2DHostView`
   into a true owning type `IntervalSet2DOwning<HostMemorySpace>` by:
   - Adding constructors that allocate and fill the host views directly
     (bypassing `std::vector`).
   - Redirecting existing building functions to this type.

### B.2. Fields: Field2DHost<T> as primary host representation

Today:
- Host owning: `IntervalField2DHost<T>` + `Field2DDevice<T>` conversions.
- `Field2DHost<T>` exists but is not widely used.

Target:

1. Define helper constructors for `Field2DHost<T>`:

```cpp
template <typename T>
Field2DHost<T> make_field_host_like(const IntervalSet2DHostView& geom,
                                    const std::string& label);
```

2. Implement `make_field_like_geometry` in terms of `Field2DHost<T>`:
   - Option A (minimal change): keep `IntervalField2DHost<T>` as return type
     but construct a `Field2DHost<T>` internally and then convert it to the
     legacy struct for now.
   - Option B (API change): introduce a new `make_field_like_geometry_kokkos`
     that returns `Field2DHost<T>` and gradually migrate examples/tests to it.

3. Add explicit conversions:
   - `IntervalField2DHost<T> make_legacy_from_field_host(const Field2DHost<T>&)`.
   - `Field2DHost<T> make_field_host_from_legacy(const IntervalField2DHost<T>&)`.

Once these exist, internal library components that currently operate on
`IntervalField2DHost<T>` can be progressively ported to `Field2DHost<T>`
without impacting the external API.

---

## 4. Phase C – Refactor Internal APIs Module by Module

**Goal:** Make internal code paths Kokkos‑first while preserving public behavior.

### C.1. Interval geometry module (`csr_interval_set.hpp`)

Refactors:
- Move all host logic that does not strictly need `std::vector` (e.g. computing
  `cell_offsets` and `total_cells`, simple transformations) to operate directly
  on Kokkos views (`IntervalSet2DHostView`) using host execution space.
- Provide a minimal set of host builders:
  - Pure device builders (already exist): `make_box_device`, `make_disk_device`,
    `make_random_device`, `make_checkerboard_device`.
  - Optional host builders: `make_box_host_view`, etc., implemented via
    `IntervalSet2DHostView` to allow CPU‑side tests without going through
    `IntervalSet2DHost`.

Externally:
- Keep `IntervalSet2DHost` and `build_device_from_host`/`build_host_from_device`
  as compatibility functions, but mark them as "host‑side construction /
  debugging helpers" in documentation.

### C.2. Field core / algebra / remap / AMR / stencil

These modules are already Kokkos‑first on the device side. The main changes:

1. **Consolidate host use**:
   - Replace any ad‑hoc use of `IntervalField2DHost<T>` in these headers
     (if present) by `Field2DHost<T>` or `IntervalSet2DHostView` as appropriate.
   - If host code is only used in tests, move it to test helpers instead.

2. **Use `Field2DHost<T>` in multilevel utilities**:
   - In `multilevel.hpp`, the deep copy from device to host is already
     Kokkos‑based; keep it as is, but note that once `Field2DHost<T>` is
     the main host field type, those host multilevel fields will be ready
     to be consumed directly by host‑side Kokkos kernels.

3. **Reduce host roundtrips in examples**:
   - For example, in `field_subview_workflow.cpp`, today:
     - `make_box_device` → `build_host_from_device` → `make_field_like_geometry`
       (host) → `build_device_field_from_host`.
   - Target:
     - Use a `make_field_like_geometry_device` helper that builds
       `Field2DDevice<double>` directly from `IntervalSet2DDevice`, using
       a simple Kokkos kernel to initialize values.
   - Provide generic `fill_by_lambda_device(Field2DDevice<T>& field, Functor f)`
     helpers where needed.

### C.3. Multilevel (`multilevel.hpp`)

This header is already heavily Kokkos‑based. Adjustments:

- Where host code is required (e.g. for VTK export), prefer:
  - Device→host conversion via `create_mirror_view_and_copy` to `HostSpace`
    views (`Field2DHost<T>`, `IntervalSet2DHostView`).
  - Legacy `IntervalSet2DHost` / `IntervalField2DHost<T>` only at the
    boundary with `vtk_export.hpp`.

### C.4. VTK export (`vtk_export.hpp`)

Currently:
- Works with `IntervalSet2DHost` and `IntervalField2DHost<T>`.

Target:
- Add overloads that accept Kokkos‑based host types:

```cpp
void write_legacy_quads(const IntervalSet2DHostView& geom, const std::string& filename);

template <typename T>
void write_legacy_quads(const Field2DHost<T>& field,
                        const std::string& filename,
                        const std::string& scalar_name);
```

Implementation detail:
- Internally, these overloads can:
  - Either operate directly on the host views (`row_keys(i)`, `intervals(i)`),
    iterating with standard for loops (HostSpace Kokkos views are trivially
    indexable).
  - Or construct a temporary `IntervalSet2DHost` / `IntervalField2DHost<T>`
    via the conversion helpers and re‑use the existing code.  
    (This is a low‑risk migration path.)

Once these overloads are in place:
- Examples and tests can migrate to Kokkos‑based host types.
- The legacy `write_legacy_quads` for `IntervalSet2DHost` /
  `IntervalField2DHost<T>` remain for backward compatibility.

---

## 5. Phase D – Tests and Examples Migration

**Goal:** Use Kokkos‑based host types in tests/examples to exercise the new
Kokkos‑first paths and reduce reliance on legacy host structs.

### D.1. Test helpers (`csr_test_utils.hpp`, field test utils)

- Introduce Kokkos variants:

```cpp
IntervalSet2DHostView make_host_view_csr(
    const std::vector<std::pair<Coord, std::vector<Interval>>>& rows);

void expect_equal_csr(const IntervalSet2DHostView& a,
                      const IntervalSet2DHostView& b);
```

- Implement them directly using host Kokkos views.
- Gradually:
  - Add new tests using the Kokkos versions.
  - Optionally migrate existing tests if/when convenient, keeping the legacy
    helpers for reference.

### D.2. Geometry tests (`csr_interval_set_smoke_test.cpp`, set algebra tests)

- Add test cases that:
  - Build geometries directly on device (`make_box_device`, `make_disk_device`,
    `make_random_device`, `make_checkerboard_device`).
  - Deep copy to host views (`IntervalSet2DHostView`) via `create_mirror_view_and_copy`
    and validate invariants/cardinalities.

- Keep existing `IntervalSet2DHost`‑based tests as "legacy / regression" checks.

### D.3. Field tests (`csr_field_*`, AMR, remap, stencil, subview)

- For new tests, prefer:
  - Geometry built via device builders.
  - Fields allocated directly as `Field2DDevice<T>` (`Field2D<T, DeviceMemorySpace>`),
    initialized with device kernels.
  - Host validation via `Field2DHost<T>` with `create_mirror_view_and_copy`.

- For existing tests:
  - Where the logic is more about field operations than host construction
    patterns, consider:
    - Keeping the IntervalField2DHost<T> construction for clarity.
    - But assert results on `Field2DHost<T>` copies to exercise the
      Kokkos host path.

### D.4. Examples

Systematically review the examples in `examples/`:

- Replace patterns:
  - `make_box_device` → `build_host_from_device` → host init loops → `build_device_field_from_host`.
- By:
  - Device geometry (`make_box_device`).
  - `Field2DDevice<T>` construction using geometry.
  - A Kokkos kernel to initialize values on the device.
  - Host copies only at the end for VTK (`Field2DHost<T>` or legacy host field).

Where VTK export is used:
- Switch to the Kokkos‑based `write_legacy_quads` overloads as soon as
  they exist (see C.4).

---

## 6. Phase E – Optional Cleanups and API Polishing

After the main migration is done and stable:

1. **Document the "canonical" types**:
   - In public headers and README, clearly state:
     - Geometry on device: `IntervalSet2DDevice`.
     - Geometry on host (Kokkos): `IntervalSet2DHostView` / future owning variant.
     - Legacy host geometry: `IntervalSet2DHost` (for construction and IO).
     - Fields on device: `Field2DDevice<T>`.
     - Fields on host (Kokkos): `Field2DHost<T>`.
     - Legacy host fields: `IntervalField2DHost<T>` (VTK, simple CPU workflows).

2. **Deprecation strategy (if desired)**:
   - Mark legacy host types and helper functions with documentation notes
     (and possibly `[[deprecated]]` in a later release).
   - Provide a migration guide from `IntervalSet2DHost` / `IntervalField2DHost<T>`
     to Kokkos‑based host types.

3. **Unify small patterns**:
   - Factor out repeated row/interval lookups (binary searches) into shared
     helpers usable on both host and device.
   - This is orthogonal to Kokkos‑first, but becomes easier once all code
     paths share the same Kokkos geometry types.

---

## 7. Execution Strategy and Risk Management

- **Incremental, per‑phase**:
  - Start with Phase A (helpers and conversions) – low risk, mechanical.
  - Then Phase C.4 + D.4 (VTK + selected examples) to validate the approach
    on end‑to‑end workflows.
  - Expand to tests (Phase D.1‑D.3) when the helper API is stable.
  - Introduce optional host‑owning Kokkos types only after there is clear
    value and tests/examples use them.

- **Preserve tests at every step**:
  - Do not remove existing tests tied to `IntervalSet2DHost` /
    `IntervalField2DHost<T>` until there are equivalent Kokkos‑first tests.

- **Performance validation**:
  - Benchmarks already exist in `benchmarks/`. Extend them to:
    - Compare flows that convert host→device once vs. multiple times.
    - Measure any overhead introduced by new helpers (should be negligible
      compared to kernels, but worth checking).

By following these phases, the codebase progressively becomes Kokkos‑first:
most logic operates on Kokkos views on both host and device, while legacy
`std::vector` structs are confined to small, well‑documented compatibility
layers. This aligns well with a mono‑GPU, performance‑oriented workflow while
keeping the public API approachable for C++ users who are not fully invested
in Kokkos.

