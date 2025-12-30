#include <gtest/gtest.h>

#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>

using namespace subsetix::csr;

namespace {

IntervalSet2DDevice make_basic_box(Coord size) {
  Box2D box;
  box.x_min = 0;
  box.x_max = size;
  box.y_min = 0;
  box.y_max = size;
  return make_box_device(box);
}

IntervalSet2DDevice make_interior_mask(Coord size) {
  Box2D box;
  box.x_min = 1;
  box.x_max = size - 1;
  box.y_min = 1;
  box.y_max = size - 1;
  return make_box_device(box);
}

} // namespace

TEST(IntervalSubSet2DTest, MaskMatchesGeometry) {
  auto geom = make_basic_box(8);
  IntervalSubSet2DDevice subset;
  build_interval_subset_device(geom, geom, subset);

  ASSERT_TRUE(subset.valid());
  EXPECT_EQ(subset.num_entries, geom.num_intervals);
  EXPECT_EQ(subset.total_cells, geom.total_cells);

  auto host_begin = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, subset.x_begin);
  auto host_end = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, subset.x_end);
  auto host_indices = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, subset.interval_indices);

  auto host_geom_intervals = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, geom.intervals);
  for (std::size_t i = 0; i < subset.num_entries; ++i) {
    const std::size_t idx = host_indices(i);
    const auto iv = host_geom_intervals(idx);
    EXPECT_EQ(host_begin(i), iv.begin);
    EXPECT_EQ(host_end(i), iv.end);
  }
}

TEST(IntervalSubSet2DTest, MaskInteriorProducesPartialEntries) {
  auto geom = make_basic_box(10);
  auto mask = make_interior_mask(10);
  IntervalSubSet2DDevice subset;
  build_interval_subset_device(geom, mask, subset);

  ASSERT_TRUE(subset.valid());
  EXPECT_EQ(subset.total_cells,
            static_cast<std::size_t>((10 - 2) * (10 - 2)));

  auto host_rows = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, subset.row_indices);
  auto host_begin = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, subset.x_begin);
  auto host_end = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, subset.x_end);
  auto geom_rows = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, geom.row_keys);

  for (std::size_t i = 0; i < subset.num_entries; ++i) {
    const int row_idx = host_rows(i);
    const Coord y = geom_rows(row_idx).y;
    EXPECT_GE(y, 1);
    EXPECT_LE(y, 8);
    EXPECT_EQ(host_begin(i), 1);
    EXPECT_EQ(host_end(i), 9);
  }
}

TEST(IntervalSubSet2DTest, EmptyMaskProducesInvalidSubset) {
  auto geom = make_basic_box(6);
  IntervalSet2DDevice mask;
  IntervalSubSet2DDevice subset;
  build_interval_subset_device(geom, mask, subset);
  EXPECT_FALSE(subset.valid());
  EXPECT_EQ(subset.num_entries, 0u);
  EXPECT_EQ(subset.total_cells, 0u);
}

TEST(IntervalSubSet2DTest, MaskRowOutsideGeometryThrows) {
  auto geom = make_basic_box(4);
  IntervalSet2DHost mask_host = make_interval_set_host(
      {RowKey2D{5}},
      {0, 1},
      {Interval{0, 2}}
  );
  auto mask = to<DeviceMemorySpace>(mask_host);

  IntervalSubSet2DDevice subset;
  EXPECT_THROW(build_interval_subset_device(geom, mask, subset),
               std::runtime_error);
}
