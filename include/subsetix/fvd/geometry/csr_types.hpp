#pragma once

// ============================================================================
// FVD GEOMETRY CSR TYPES
// ============================================================================
//
// This file provides using declarations that redirect to the real
// subsetix::csr types. The stub implementations have been removed
// in favor of using the complete, GPU-native CSR implementation.
//
// ============================================================================

#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/csr_ops/set_algebra.hpp>
#include <subsetix/csr_ops/workspace.hpp>

namespace subsetix::fvd::csr {

// ============================================================================
// CORE TYPES - Using subsetix::csr types
// ============================================================================

using IntervalSet2DDevice = subsetix::csr::IntervalSet2D<subsetix::csr::DeviceMemorySpace>;
using IntervalSet2DHost = subsetix::csr::IntervalSet2D<subsetix::csr::HostMemorySpace>;

using Box2D = subsetix::csr::Box2D;
using Disk2D = subsetix::csr::Disk2D;
using Domain2D = subsetix::csr::Domain2D;

// Type aliases for convenience
template <class MemorySpace>
using IntervalSet2D = subsetix::csr::IntervalSet2D<MemorySpace>;

// ============================================================================
// CONTEXT TYPES
// ============================================================================

using CsrSetAlgebraContext = subsetix::csr::CsrSetAlgebraContext;

// ============================================================================
// HELPER FUNCTIONS - Re-export subsetix::csr functions
// ============================================================================

using subsetix::csr::make_box_device;
using subsetix::csr::make_disk_device;
using subsetix::csr::make_random_device;
using subsetix::csr::make_bitmap_device;
using subsetix::csr::make_checkerboard_device;

using subsetix::csr::allocate_interval_set_device;
using subsetix::csr::compute_cell_offsets_device;
using subsetix::csr::compute_cell_offsets_host;

using subsetix::csr::to;

// CSG operations
using subsetix::csr::set_union_device;
using subsetix::csr::set_difference_device;
using subsetix::csr::set_intersection_device;
using subsetix::csr::set_symmetric_difference_device;

} // namespace subsetix::fvd::csr
