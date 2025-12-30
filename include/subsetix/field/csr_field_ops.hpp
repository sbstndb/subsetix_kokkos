#pragma once

/**
 * @file csr_field_ops.hpp
 * @brief Facade header for CSR field operations.
 *
 * This header includes all field operation modules for backward compatibility.
 * The implementation has been split into specialized modules:
 *   - csr_ops/field_mapping.hpp: Mask-to-field mapping utilities
 *   - csr_ops/field_subset.hpp: Basic operations on subsets (fill, copy, scale)
 *   - csr_ops/field_stencil.hpp: Stencil operations
 *   - csr_ops/field_amr.hpp: AMR operations (restrict, prolong)
 *   - csr_ops/field_arith.hpp: Arithmetic operations (add, sub, mul, div)
 *   - csr_ops/field_remap.hpp: Projection and remapping between geometries
 *   - csr_ops/field_subview.hpp: High-level subview utilities
 */

#include <subsetix/csr_ops/field_mapping.hpp>
#include <subsetix/csr_ops/field_stencil.hpp>
#include <subsetix/csr_ops/field_amr.hpp>
#include <subsetix/csr_ops/field_arith.hpp>
#include <subsetix/csr_ops/field_remap.hpp>
#include <subsetix/csr_ops/field_subview.hpp>
#include <subsetix/csr_ops/field_subset.hpp>
