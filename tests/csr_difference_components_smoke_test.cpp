#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>

using namespace subsetix::csr;

TEST(CSRDifferenceComponentsSmokeTest, RowDifferenceMapping) {
  auto hostA = make_interval_set_host(
      {{0}, {2}, {4}},  // row_keys
      {0, 0, 0, 0},     // row_ptr (empty rows)
      {}                // no intervals
  );

  auto hostB = make_interval_set_host(
      {{1}, {2}, {4}},  // row_keys
      {0, 0, 0, 0},     // row_ptr (empty rows)
      {}                // no intervals
  );

  auto A = to<DeviceMemorySpace>(hostA);
  auto B = to<DeviceMemorySpace>(hostB);

  detail::RowMergeResult diff =
      detail::build_row_difference_mapping(A, B);

  ASSERT_EQ(diff.num_rows, 3u);

  auto h_rows = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, diff.row_keys);
  auto h_idx_a = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, diff.row_index_a);
  auto h_idx_b = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, diff.row_index_b);

  const Coord expected_y[3] = {0, 2, 4};
  const int expected_ia[3] = {0, 1, 2};  // identity mapping
  const int expected_ib[3] = {-1, 1, 2};

  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(h_rows(i).y, expected_y[i]);
    EXPECT_EQ(h_idx_a(i), expected_ia[i]);
    EXPECT_EQ(h_idx_b(i), expected_ib[i]);
  }
}
