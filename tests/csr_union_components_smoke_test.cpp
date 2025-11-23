#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <subsetix/geometry.hpp>

using namespace subsetix::csr;

TEST(CSRUnionComponentsSmokeTest, RowUnionMapping) {
  IntervalSet2DHost hostA;
  IntervalSet2DHost hostB;

  hostA.row_keys.push_back(RowKey2D{0});
  hostA.row_keys.push_back(RowKey2D{2});
  hostA.row_ptr.push_back(0);
  hostA.row_ptr.push_back(0);
  hostA.row_ptr.push_back(0);

  hostB.row_keys.push_back(RowKey2D{1});
  hostB.row_keys.push_back(RowKey2D{2});
  hostB.row_keys.push_back(RowKey2D{4});
  hostB.row_ptr.push_back(0);
  hostB.row_ptr.push_back(0);
  hostB.row_ptr.push_back(0);
  hostB.row_ptr.push_back(0);

  auto A = build_device_from_host(hostA);
  auto B = build_device_from_host(hostB);

  detail::RowMergeResult merge =
      detail::build_row_union_mapping(A, B);

  ASSERT_EQ(merge.num_rows, 4u);

  auto h_rows = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, merge.row_keys);
  auto h_idx_a = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, merge.row_index_a);
  auto h_idx_b = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, merge.row_index_b);

  const Coord expected_y[4] = {0, 1, 2, 4};
  const int expected_ia[4] = {0, -1, 1, -1};
  const int expected_ib[4] = {-1, 0, 1, 2};

  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(h_rows(i).y, expected_y[i]);
    EXPECT_EQ(h_idx_a(i), expected_ia[i]);
    EXPECT_EQ(h_idx_b(i), expected_ib[i]);
  }
}
