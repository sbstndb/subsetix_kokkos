#include <Kokkos_Core.hpp>
#include <cstdint>
#include <gtest/gtest.h>

#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>

namespace {

using namespace subsetix::csr;

bool check_csr(const IntervalSet2DHost& host,
               Coord x_min,
               Coord x_max,
               Coord y_min,
               Coord y_max) {
  const std::size_t num_rows_expected =
      static_cast<std::size_t>(y_max - y_min);

  if (host.row_keys.extent(0) != num_rows_expected) {
    return false;
  }
  if (host.row_ptr.extent(0) != num_rows_expected + 1) {
    return false;
  }
  if (host.num_rows == 0) {
    return host.num_intervals == 0;
  }
  if (host.row_ptr(0) != 0) {
    return false;
  }

  for (std::size_t i = 0; i < num_rows_expected; ++i) {
    const Coord y = host.row_keys(i).y;
    if (y < y_min || y >= y_max) {
      return false;
    }
    const std::size_t begin = host.row_ptr(i);
    const std::size_t end = host.row_ptr(i + 1);
    if (end < begin || end > host.intervals.extent(0)) {
      return false;
    }
    for (std::size_t k = begin; k < end; ++k) {
      const auto& iv = host.intervals(k);
      if (!(iv.begin < iv.end)) {
        return false;
      }
      if (iv.begin < x_min || iv.end > x_max) {
        return false;
      }
    }
  }

  if (host.row_ptr(host.num_rows) != host.intervals.extent(0)) {
    return false;
  }

  return true;
}

} // namespace

TEST(CSRBuildersSmokeTest, BoxBuilder) {
  Box2D box;
  box.x_min = 0;
  box.x_max = 10;
  box.y_min = 0;
  box.y_max = 3;

  auto rect_dev = make_box_device(box);
  auto rect_host = to<HostMemorySpace>(rect_dev);

  ASSERT_TRUE(check_csr(rect_host, box.x_min, box.x_max,
                        box.y_min, box.y_max));

  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(rect_host.row_keys(i).y, static_cast<Coord>(i));
    EXPECT_EQ(rect_host.intervals(i).begin, 0);
    EXPECT_EQ(rect_host.intervals(i).end, 10);
  }
}

TEST(CSRBuildersSmokeTest, DiskBuilder) {
  Disk2D disk;
  disk.cx = 0;
  disk.cy = 0;
  disk.radius = 1;

  auto disk_dev = make_disk_device(disk);
  auto disk_host = to<HostMemorySpace>(disk_dev);

  ASSERT_FALSE(disk_host.num_rows == 0);

  ASSERT_TRUE(check_csr(disk_host,
                        static_cast<Coord>(disk.cx - disk.radius),
                        static_cast<Coord>(disk.cx + disk.radius + 1),
                        static_cast<Coord>(disk.cy - disk.radius),
                        static_cast<Coord>(disk.cy + disk.radius + 1)));

  bool ok_middle = false;
  for (std::size_t i = 0; i < disk_host.num_rows; ++i) {
    if (disk_host.row_keys(i).y == 0) {
      std::size_t begin = disk_host.row_ptr(i);
      std::size_t end = disk_host.row_ptr(i + 1);
      if (begin < end) {
        const auto& iv = disk_host.intervals(begin);
        if (iv.begin <= 0 && iv.end > 0) {
          ok_middle = true;
        }
      }
      break;
    }
  }
  EXPECT_TRUE(ok_middle);
}

TEST(CSRBuildersSmokeTest, RandomBuilder) {
  Domain2D dom;
  dom.x_min = 0;
  dom.x_max = 20;
  dom.y_min = 0;
  dom.y_max = 10;

  auto rand_dev = make_random_device(dom, 0.3, 12345);
  auto rand_host = to<HostMemorySpace>(rand_dev);

  EXPECT_TRUE(check_csr(rand_host, dom.x_min, dom.x_max,
                        dom.y_min, dom.y_max));
}

TEST(CSRBuildersSmokeTest, CheckerboardBuilder) {
  Domain2D dom_cb;
  dom_cb.x_min = 0;
  dom_cb.x_max = 8;
  dom_cb.y_min = 0;
  dom_cb.y_max = 4;

  const Coord square_size = 2;

  auto cb_dev = make_checkerboard_device(dom_cb, square_size);
  auto cb_host = to<HostMemorySpace>(cb_dev);

  ASSERT_TRUE(check_csr(cb_host, dom_cb.x_min, dom_cb.x_max,
                        dom_cb.y_min, dom_cb.y_max));

  const std::size_t num_rows = cb_host.num_rows;
  const Coord width = dom_cb.x_max - dom_cb.x_min;
  const Coord height = dom_cb.y_max - dom_cb.y_min;

  for (std::size_t i = 0; i < num_rows; ++i) {
    const Coord y = cb_host.row_keys(i).y;
    const Coord local_y = static_cast<Coord>(y - dom_cb.y_min);
    const std::size_t block_y =
        static_cast<std::size_t>(local_y / square_size);
    const std::size_t begin = cb_host.row_ptr(i);
    const std::size_t end = cb_host.row_ptr(i + 1);
    const std::size_t count = end - begin;

    const std::size_t num_blocks_x =
        static_cast<std::size_t>(
            (static_cast<long long>(width) + square_size - 1) /
            square_size);

    std::size_t expected = 0;
    for (std::size_t bx = 0; bx < num_blocks_x; ++bx) {
      const bool filled = (((bx + block_y) & 1U) == 0U);
      if (filled) {
        ++expected;
      }
    }

    EXPECT_EQ(count, expected);

    std::size_t written = 0;
    for (std::size_t bx = 0; bx < num_blocks_x; ++bx) {
      const bool filled = (((bx + block_y) & 1U) == 0U);
      const Coord x0 = static_cast<Coord>(dom_cb.x_min + bx * square_size);
      const Coord x1_candidate =
          static_cast<Coord>(x0 + square_size);
      const Coord x1 = (x1_candidate > dom_cb.x_max)
                           ? dom_cb.x_max
                           : x1_candidate;

      if (!filled) {
        continue;
      }

      ASSERT_LT(begin + written, end);

      const auto& iv = cb_host.intervals(begin + written);
      EXPECT_EQ(iv.begin, x0);
      EXPECT_EQ(iv.end, x1);

      ++written;
    }

    EXPECT_EQ(written, count);
    (void)height;
  }
}

TEST(CSRBuildersSmokeTest, BitmapBuilder) {
  const std::size_t height = 3;
  const std::size_t width = 5;
  Kokkos::View<std::uint8_t**, DeviceMemorySpace> d_mask(
      "bitmap_dev", height, width);
  auto h_mask = Kokkos::create_mirror_view(d_mask);

  // Pattern:
  // 1 1 0 0 0
  // 0 1 1 0 1
  // 0 0 0 0 0
  h_mask(0, 0) = 1; h_mask(0, 1) = 1;
  h_mask(0, 2) = 0; h_mask(0, 3) = 0; h_mask(0, 4) = 0;

  h_mask(1, 0) = 0; h_mask(1, 1) = 1; h_mask(1, 2) = 1;
  h_mask(1, 3) = 0; h_mask(1, 4) = 1;

  for (std::size_t j = 0; j < width; ++j) {
    h_mask(2, j) = 0;
  }
  Kokkos::deep_copy(d_mask, h_mask);

  const Coord x_min = -2;
  const Coord y_min = 5;
  auto dev = make_bitmap_device(d_mask, x_min, y_min, 1);
  auto host = to<HostMemorySpace>(dev);

  ASSERT_TRUE(check_csr(host, x_min, x_min + static_cast<Coord>(width),
                        y_min, y_min + static_cast<Coord>(height)));

  ASSERT_EQ(host.row_keys.extent(0), height);
  ASSERT_EQ(host.row_ptr.extent(0), height + 1);

  // Row 0: one run [x_min, x_min+2)
  {
    const std::size_t begin = host.row_ptr(0);
    const std::size_t end = host.row_ptr(1);
    ASSERT_EQ(end - begin, 1U);
    EXPECT_EQ(host.intervals(begin).begin, x_min);
    EXPECT_EQ(host.intervals(begin).end, x_min + 2);
  }

  // Row 1: two runs [x_min+1, x_min+3) and [x_min+4, x_min+5)
  {
    const std::size_t begin = host.row_ptr(1);
    const std::size_t end = host.row_ptr(2);
    ASSERT_EQ(end - begin, 2U);
    EXPECT_EQ(host.intervals(begin).begin, x_min + 1);
    EXPECT_EQ(host.intervals(begin).end, x_min + 3);
    EXPECT_EQ(host.intervals(begin + 1).begin, x_min + 4);
    EXPECT_EQ(host.intervals(begin + 1).end, x_min + 5);
  }

  // Row 2: empty.
  EXPECT_EQ(host.row_ptr(3) - host.row_ptr(2), 0U);
}
