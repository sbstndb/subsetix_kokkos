#pragma once

#include <Kokkos_Core.hpp>
#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_field.hpp>

namespace subsetix {

// Maximum number of levels supported (compile-time constant)
constexpr int MAX_LEVELS = 16;

/**
 * @brief Multilevel geometry container.
 *
 * Contains physical metadata and a static array of interval sets for each level.
 * This structure is designed to be lightweight and TriviallyCopyable, so it can
 * be passed by value to Kokkos kernels (lambda captures).
 *
 * Copying this structure copies the Views (handles), not the deep data.
 */
template <class MemorySpace>
struct MultilevelGeo {
  using GeoView = csr::IntervalSet2DView<MemorySpace>;

  // Physical domain metadata
  double origin_x = 0.0;
  double origin_y = 0.0;
  double root_dx = 1.0; // Cell size at level 0
  double root_dy = 1.0; // Cell size at level 0 (usually same as dx)

  // Number of levels currently populated/valid
  int num_active_levels = 0;

  // Static array of levels.
  // Using Kokkos::Array ensures it's compatible with device kernels.
  Kokkos::Array<GeoView, MAX_LEVELS> levels;

  KOKKOS_INLINE_FUNCTION
  MultilevelGeo() {
    for (int i = 0; i < MAX_LEVELS; ++i) {
      levels[i] = GeoView();
    }
  }

  /**
   * @brief Get the cell size for a specific level.
   */
  KOKKOS_INLINE_FUNCTION
  double dx_at(int level) const {
    // dx_L = root_dx / 2^L
    // 1 << level is 2^level (integer arithmetic)
    return root_dx / static_cast<double>(1 << level);
  }

  KOKKOS_INLINE_FUNCTION
  double dy_at(int level) const {
    return root_dy / static_cast<double>(1 << level);
  }
};

/**
 * @brief Multilevel field container.
 */
template <typename T, class MemorySpace>
struct MultilevelField {
  using FieldView = csr::IntervalField2DView<T, MemorySpace>;

  int num_active_levels = 0;
  Kokkos::Array<FieldView, MAX_LEVELS> levels;

  KOKKOS_INLINE_FUNCTION
  MultilevelField() {
    for (int i = 0; i < MAX_LEVELS; ++i) {
      levels[i] = FieldView();
    }
  }
};

// Type aliases for convenience
using MultilevelGeoDevice = MultilevelGeo<csr::DeviceMemorySpace>;
using MultilevelGeoHost = MultilevelGeo<csr::HostMemorySpace>;

template <typename T>
using MultilevelFieldDevice = MultilevelField<T, csr::DeviceMemorySpace>;

template <typename T>
using MultilevelFieldHost = MultilevelField<T, csr::HostMemorySpace>;

/**
 * @brief Deep copy a MultilevelGeo from Device to Host.
 *
 * This is necessary for IO operations (VTK export) or inspection.
 */
inline MultilevelGeoHost deep_copy_to_host(const MultilevelGeoDevice& dev_geo) {
  MultilevelGeoHost host_geo;
  host_geo.origin_x = dev_geo.origin_x;
  host_geo.origin_y = dev_geo.origin_y;
  host_geo.root_dx = dev_geo.root_dx;
  host_geo.root_dy = dev_geo.root_dy;
  host_geo.num_active_levels = dev_geo.num_active_levels;

  for (int i = 0; i < MAX_LEVELS; ++i) {
    // We use the existing helper to deep copy the CSR structure
    // Note: build_host_from_device returns a Host structure (std::vectors),
    // but our MultilevelGeoHost expects Host Views.
    // We need a helper that copies View<Device> to View<Host>.
    
    // csr::build_host_from_device returns IntervalSet2DHost (std::vector based).
    // We need to convert that back to IntervalSet2DHostView OR implement a direct View-to-View copy helper.
    // Let's implement a direct deep_copy helper for IntervalSet2DView.
    
    const auto& d_view = dev_geo.levels[i];
    auto& h_view = host_geo.levels[i];

    if (d_view.num_rows == 0) {
      continue;
    }

    h_view.num_rows = d_view.num_rows;
    h_view.num_intervals = d_view.num_intervals;

    h_view.row_keys = Kokkos::create_mirror_view_and_copy(csr::HostMemorySpace{}, d_view.row_keys);
    h_view.row_ptr = Kokkos::create_mirror_view_and_copy(csr::HostMemorySpace{}, d_view.row_ptr);
    h_view.intervals = Kokkos::create_mirror_view_and_copy(csr::HostMemorySpace{}, d_view.intervals);
  }

  return host_geo;
}

/**
 * @brief Deep copy a MultilevelField from Device to Host.
 */
template <typename T>
inline MultilevelFieldHost<T> deep_copy_to_host(const MultilevelFieldDevice<T>& dev_field) {
  MultilevelFieldHost<T> host_field;
  host_field.num_active_levels = dev_field.num_active_levels;

  for (int i = 0; i < MAX_LEVELS; ++i) {
    const auto& d_view = dev_field.levels[i];
    auto& h_view = host_field.levels[i];

    if (d_view.num_rows == 0) {
      continue;
    }

    h_view.num_rows = d_view.num_rows;
    h_view.num_intervals = d_view.num_intervals;
    h_view.value_count = d_view.value_count;

    h_view.row_keys = Kokkos::create_mirror_view_and_copy(csr::HostMemorySpace{}, d_view.row_keys);
    h_view.row_ptr = Kokkos::create_mirror_view_and_copy(csr::HostMemorySpace{}, d_view.row_ptr);
    h_view.intervals = Kokkos::create_mirror_view_and_copy(csr::HostMemorySpace{}, d_view.intervals);
    h_view.values = Kokkos::create_mirror_view_and_copy(csr::HostMemorySpace{}, d_view.values);
  }

  return host_field;
}

} // namespace subsetix

