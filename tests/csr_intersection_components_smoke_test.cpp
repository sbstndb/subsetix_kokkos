#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_set_ops.hpp>

using namespace subsetix::csr;

TEST(CSRIntersectionComponentsSmokeTest, RowIntersectionMapping) {
  IntervalSet2DHost hostA;
  IntervalSet2DHost hostB;

  hostA.row_keys.push_back(RowKey2D{0});
  hostA.row_keys.push_back(RowKey2D{2});
  hostA.row_keys.push_back(RowKey2D{3});
  hostA.row_ptr = {0, 0, 0, 0};

  hostB.row_keys.push_back(RowKey2D{1});
  hostB.row_keys.push_back(RowKey2D{2});
  hostB.row_keys.push_back(RowKey2D{3});
  hostB.row_keys.push_back(RowKey2D{5});
  hostB.row_ptr = {0, 0, 0, 0, 0};

  auto A = build_device_from_host(hostA);
  auto B = build_device_from_host(hostB);

  detail::RowMergeResult merge =
      detail::build_row_intersection_mapping(A, B);

  ASSERT_EQ(merge.num_rows, 2u);

  auto h_rows = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, merge.row_keys);
  auto h_idx_a = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, merge.row_index_a);
  auto h_idx_b = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, merge.row_index_b);

  const Coord expected_y[2] = {2, 3};
  const int expected_ia[2] = {1, 2};
  const int expected_ib[2] = {1, 2};

  for (std::size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(h_rows(i).y, expected_y[i]);
    EXPECT_EQ(h_idx_a(i), expected_ia[i]);
    EXPECT_EQ(h_idx_b(i), expected_ib[i]);
  }
}
