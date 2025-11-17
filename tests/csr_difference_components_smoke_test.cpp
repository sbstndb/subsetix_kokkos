#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_set_ops.hpp>

using namespace subsetix::csr;

TEST(CSRDifferenceComponentsSmokeTest, RowDifferenceMapping) {
  IntervalSet2DHost hostA;
  IntervalSet2DHost hostB;

  hostA.row_keys.push_back(RowKey2D{0});
  hostA.row_keys.push_back(RowKey2D{2});
  hostA.row_keys.push_back(RowKey2D{4});
  hostA.row_ptr = {0, 0, 0, 0};

  hostB.row_keys.push_back(RowKey2D{1});
  hostB.row_keys.push_back(RowKey2D{2});
  hostB.row_keys.push_back(RowKey2D{4});
  hostB.row_ptr = {0, 0, 0, 0};

  auto A = build_device_from_host(hostA);
  auto B = build_device_from_host(hostB);

  detail::RowDifferenceResult diff =
      detail::build_row_difference_mapping(A, B);

  ASSERT_EQ(diff.num_rows, 3u);

  auto h_rows = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, diff.row_keys);
  auto h_idx_b = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, diff.row_index_b);

  const Coord expected_y[3] = {0, 2, 4};
  const int expected_ib[3] = {-1, 1, 2};

  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(h_rows(i).y, expected_y[i]);
    EXPECT_EQ(h_idx_b(i), expected_ib[i]);
  }
}
