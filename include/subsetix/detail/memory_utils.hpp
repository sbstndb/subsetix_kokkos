#pragma once

#include <string>
#include <Kokkos_Core.hpp>

namespace subsetix {
namespace csr {
namespace detail {

/**
 * @brief Ensure a Kokkos View has at least the required capacity.
 *
 * If the current capacity (extent 0) is less than required_size, the view is
 * reallocated with the new size. This function does NOT preserve content
 * (unlike Kokkos::resize) because it's intended for scratch buffers.
 *
 * @tparam ViewType The type of the Kokkos::View.
 * @param view The view to check/resize (passed by reference).
 * @param required_size The minimum required extent.
 * @param label The label to use if reallocation occurs.
 */
template <class ViewType>
inline void ensure_view_capacity(ViewType& view,
                                 std::size_t required_size,
                                 const std::string& label) {
  if (view.extent(0) < required_size) {
    // Optionally add a growth factor here if frequent resizes are expected,
    // but for scratch buffers, exact fit or worst-case fit is usually sufficient.
    view = ViewType(label, required_size);
  }
}

} // namespace detail
} // namespace csr
} // namespace subsetix
