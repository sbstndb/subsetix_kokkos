# Field & Set Operations Plan

This document describes how we will introduce device‑side operations on fields filtered by CSR interval sets, plus stencils and AMR‑related extensions. It is intentionally incremental so we can implement and validate each step separately.

## 1. Scope & Constraints

- Work on `IntervalField2DDevice<T>` and `IntervalSet2DDevice` only (device‑first).
- Mask sets are assumed to be **aligned sub‑geometries** of the field (no implicit intersection; if they “overflow”, behaviour is undefined and tests must not rely on it).
- API should expose both:
  - Simple, ready‑to‑use operations (`fill`, `copy`, `scale`).
  - A **generic masked apply** that takes a `KOKKOS_LAMBDA` functor with access to geometry and value.

## 2. Phase 1 – Core Masked Local Operations

- Add `include/subsetix/csr_field_ops.hpp` (`subsetix::csr`).
- Define a generic primitive:
  - `apply_on_set_device(field, mask, Functor)`, where `Functor` supports  
    `KOKKOS_INLINE_FUNCTION void operator()(Coord x, Coord y, ValueReference v) const;`
- Implementation strategy:
  - Iterate over the **mask geometry** (rows + intervals + x cells).
  - For each cell `(x, y)`, compute the corresponding index into `field.values` under the assumption that geometries are aligned.
  - Call the functor from a `KOKKOS_LAMBDA` kernel.
- Build convenience wrappers on top:
  - `fill_on_set_device(field, mask, value)`
  - `copy_on_set_device(dst, src, mask)`
  - `scale_on_set_device(field, mask, alpha)`
- Testing:
  - GTest fixtures that generate `(geometry, field, mask)` using existing builders (box, checkerboard, random).
  - Verify behaviour for small hand‑crafted cases and a few random cases.

## 3. Phase 2 – Stencils Restricted to a Set

- Design an API:
  - `apply_stencil_on_set_device(field_out, field_in, mask, StencilFunctor)`
  - `StencilFunctor` sees `(x, y, accessor_in, accessor_out)` where accessor wraps CSR lookup.
- Start with simple stencils (e.g. 5‑points / 1D neighbour sum), assuming that all required neighbours exist in memory and belong to the same level.
- Use existing translation/geometry ops when useful, but keep a direct CSR lookup path for performance.
- Tests:
  - Validate on simple patterns (ramps, constant fields) and check expected discrete derivatives or averages.

## 4. Phase 3 – AMR‑Aware Field Operations (Later)

- Reuse existing AMR geometry ops:
  - `refine_level_up_device`, `project_level_down_device`.
- Add masked versions:
  - `restrict_field_on_set_device(coarse, fine, mask)`
  - `prolong_field_on_set_device(fine, coarse, mask)`
- Focus on correctness first (matching AMR rules), then add property tests (e.g. conservation of sums in simple cases).

## 5. Phase 4 – Benchmarks & Extensions

- Add Google Benchmark suites for:
  - `apply_on_set_device` on various mask sizes and shapes.
  - Simple stencils on sets.
- Metric: `ns_per_cell` (analogous à `ns_per_interval`), measured on serial / OpenMP / CUDA presets.
- As a later extension, introduce more complex operators (limiters, flux computations) built on top of the same primitives.

