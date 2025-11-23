#pragma once

#include <Kokkos_Core.hpp>

namespace subsetix {
namespace csr {

#if defined(SUBSETIX_EXECSPACE_CUDA)
using ExecSpace = Kokkos::Cuda;
#elif defined(SUBSETIX_EXECSPACE_OPENMP)
using ExecSpace = Kokkos::OpenMP;
#elif defined(SUBSETIX_EXECSPACE_SERIAL)
using ExecSpace = Kokkos::Serial;
#else
using ExecSpace = Kokkos::DefaultExecutionSpace;
#endif

#if defined(SUBSETIX_MEMORYSPACE_FORCE_UVM)
using DeviceMemorySpace = Kokkos::CudaUVMSpace;
#elif defined(SUBSETIX_MEMORYSPACE_FORCE_HOSTPINNED)
using DeviceMemorySpace = Kokkos::HostPinnedSpace;
#else
using DeviceMemorySpace = typename ExecSpace::memory_space;
#endif

using HostMemorySpace = Kokkos::HostSpace;

} // namespace csr
} // namespace subsetix
