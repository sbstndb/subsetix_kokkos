# CSR Field Operations Upgrade Summary

## Overview

This document summarizes the major upgrade to the CSR Field operations module, bringing it to the same level of maturity as the CSR IntervalSet operations.

## Changes Implemented

### 1. Infrastructure Memory (Workspaces)

**File**: `include/subsetix/csr_ops/workspace.hpp`

Added support for field value buffers in `UnifiedCsrWorkspace`:
- `double_buf_0`, `double_buf_1`: Double-precision value buffers
- `float_buf_0`, `float_buf_1`: Single-precision value buffers
- Accessors: `get_double_buf_0()`, `get_double_buf_1()`, `get_float_buf_0()`, `get_float_buf_1()`

This enables zero-allocation field operations by reusing preallocated GPU memory.

### 2. Modular Refactoring

The monolithic `csr_field_ops.hpp` (993 lines) has been split into specialized modules:

#### `include/subsetix/csr_ops/field_core.hpp`
Core field operations:
- `apply_on_set_device()`: Apply custom functor on masked field
- `fill_on_set_device()`: Fill field values on a mask
- `scale_on_set_device()`: Scale field values on a mask
- `copy_on_set_device()`: Copy field values on a mask
- Mask-to-field mapping utilities

#### `include/subsetix/csr_ops/field_stencil.hpp`
Stencil operations:
- `apply_stencil_on_set_device()`: Apply stencil functor on masked region
- `FieldStencilContext`: Context for accessing neighbor values
- Vertical interval mapping for structured grids

#### `include/subsetix/csr_ops/field_amr.hpp`
AMR (Adaptive Mesh Refinement) operations:
- `restrict_field_on_set_device()`: Coarsen field from fine to coarse grid
- `prolong_field_on_set_device()`: Refine field from coarse to fine grid
- AMR interval mapping utilities

#### `include/subsetix/csr_field_ops.hpp`
Now a facade header that includes all specialized modules for backward compatibility.

### 3. Field Algebra (New Module)

**File**: `include/subsetix/csr_ops/field_algebra.hpp`

New arithmetic operations for fields with identical geometry:
- `field_add_device()`: Element-wise addition (result = a + b)
- `field_sub_device()`: Element-wise subtraction (result = a - b)
- `field_mul_device()`: Element-wise multiplication (result = a * b)
- `field_div_device()`: Element-wise division (result = a / b)
- `field_abs_device()`: Element-wise absolute value (result = |a|)
- `field_axpby_device()`: Linear combination (result = alpha*a + beta*b)
- `field_dot_device()`: Dot product (returns scalar)
- `field_norm_l2_device()`: L2 norm (returns scalar)

All operations are fully parallel and work on Serial, OpenMP, and CUDA backends.

### 4. Field Remapping (New Module)

**File**: `include/subsetix/csr_ops/field_remap.hpp`

Operations for transferring field values between different geometries:

- `remap_field_device()`: Transfer values from source to destination geometry
  - Copies values where geometries overlap
  - Fills with default value where no overlap exists
  - Essential after set algebra operations that change geometry

- `accumulate_field_device()`: Accumulate values from source into destination
  - Adds source values to destination where geometries overlap
  - Leaves destination unchanged where no overlap
  - Useful for multi-source contributions

These operations enable the critical link between set algebra (geometry changes) and field operations (value transfers).

### 5. Testing

Added comprehensive test coverage:

#### `tests/csr_field_algebra_smoke_test.cpp`
Tests for all arithmetic operations:
- Basic operations (add, sub, mul, div)
- Unary operations (abs)
- Linear combinations (axpby)
- Reductions (dot, norm_l2)
- Multi-row fields

#### `tests/csr_field_remap_smoke_test.cpp`
Tests for remapping operations:
- Full overlap scenarios
- Partial overlap scenarios
- No overlap scenarios
- Multiple intervals
- Accumulation operations

All tests pass on Serial, OpenMP, and CUDA backends.

## Benefits

### Performance
- Zero-allocation field operations via workspace reuse
- Reduced GPU memory allocation overhead
- Better cache locality through modular design

### Maintainability
- Clear separation of concerns (core, stencil, AMR, algebra, remap)
- Easier to locate and modify specific functionality
- Consistent with IntervalSet module organization

### Functionality
- Complete arithmetic operations for fields
- Field transfer between different geometries
- Essential for complex AMR workflows

## Usage Examples

### Field Algebra
```cpp
// Create three fields with identical geometry
auto field_a = build_device_field_from_host(host_a);
auto field_b = build_device_field_from_host(host_b);
auto result = build_device_field_from_host(host_result);

// Compute: result = 2*a + 3*b
field_axpby_device(result, 2.0, field_a, 3.0, field_b);

// Compute dot product
double dot = field_dot_device(field_a, field_b);
```

### Field Remapping
```cpp
// After set union: C = A ∪ B
IntervalSet2DDevice geom_c = set_union(geom_a, geom_b);

// Create field on new geometry
auto field_c = make_field_like_geometry(geom_c, 0.0);

// Transfer values from field_a to field_c
remap_field_device(field_c, field_a, 0.0);

// Accumulate values from field_b
accumulate_field_device(field_c, field_b);
```

## Compatibility

All changes maintain backward compatibility:
- Existing code using `csr_field_ops.hpp` continues to work
- No API changes to existing functions
- New functionality is additive only

## Testing Status

✅ All tests pass on:
- Serial backend
- OpenMP backend
- CUDA backend (GCC 12)

Total: 2 new test files, 22 new test cases.

