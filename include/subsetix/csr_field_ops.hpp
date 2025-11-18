#pragma once

/**
 * @file csr_field_ops.hpp
 * @brief Facade header for CSR field operations.
 *
 * This header includes all field operation modules for backward compatibility.
 * The implementation has been split into specialized modules:
 *   - csr_ops/field_core.hpp: Basic operations (fill, copy, scale)
 *   - csr_ops/field_stencil.hpp: Stencil operations
 *   - csr_ops/field_amr.hpp: AMR operations (restrict, prolong)
 *   - csr_ops/field_algebra.hpp: Arithmetic operations (add, sub, mul, div)
 *   - csr_ops/field_remap.hpp: Projection and remapping between geometries
 */

#include <subsetix/csr_ops/field_core.hpp>
#include <subsetix/csr_ops/field_stencil.hpp>
#include <subsetix/csr_ops/field_amr.hpp>
#include <subsetix/csr_ops/field_algebra.hpp>
#include <subsetix/csr_ops/field_remap.hpp>
