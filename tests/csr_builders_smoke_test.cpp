#include <Kokkos_Core.hpp>
#include <cstdint>
#include <gtest/gtest.h>

#include <subsetix/csr_interval_set.hpp>

namespace {

using namespace subsetix::csr;

bool check_csr(const IntervalSet2DHost& host,
               Coord x_min,
               Coord x_max,
               Coord y_min,
               Coord y_max) {
  const std::size_t num_rows_expected =
      static_cast<std::size_t>(y_max - y_min);

  if (host.row_keys.size() != num_rows_expected) {
    return false;
  }
  if (host.row_ptr.size() != num_rows_expected + 1) {
    return false;
  }
  if (host.row_ptr.empty()) {
    return host.intervals.empty();
  }
  if (host.row_ptr.front() != 0) {
    return false;
  }

  for (std::size_t i = 0; i < num_rows_expected; ++i) {
    const Coord y = host.row_keys[i].y;
    if (y < y_min || y >= y_max) {
      return false;
    }
    const std::size_t begin = host.row_ptr[i];
    const std::size_t end = host.row_ptr[i + 1];
    if (end < begin || end > host.intervals.size()) {
      return false;
    }
    for (std::size_t k = begin; k < end; ++k) {
      const auto& iv = host.intervals[k];
      if (!(iv.begin < iv.end)) {
        return false;
      }
      if (iv.begin < x_min || iv.end > x_max) {
        return false;
      }
    }
  }

  if (host.row_ptr.back() != host.intervals.size()) {
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
  auto rect_host = build_host_from_device(rect_dev);

  ASSERT_TRUE(check_csr(rect_host, box.x_min, box.x_max,
                        box.y_min, box.y_max));

  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(rect_host.row_keys[i].y, static_cast<Coord>(i));
    EXPECT_EQ(rect_host.intervals[i].begin, 0);
    EXPECT_EQ(rect_host.intervals[i].end, 10);
  }
}

TEST(CSRBuildersSmokeTest, DiskBuilder) {
  Disk2D disk;
  disk.cx = 0;
  disk.cy = 0;
  disk.radius = 1;

  auto disk_dev = make_disk_device(disk);
  auto disk_host = build_host_from_device(disk_dev);

  ASSERT_FALSE(disk_host.row_keys.empty());

  ASSERT_TRUE(check_csr(disk_host,
                        static_cast<Coord>(disk.cx - disk.radius),
                        static_cast<Coord>(disk.cx + disk.radius + 1),
                        static_cast<Coord>(disk.cy - disk.radius),
                        static_cast<Coord>(disk.cy + disk.radius + 1)));

  bool ok_middle = false;
  for (std::size_t i = 0; i < disk_host.row_keys.size(); ++i) {
    if (disk_host.row_keys[i].y == 0) {
      std::size_t begin = disk_host.row_ptr[i];
      std::size_t end = disk_host.row_ptr[i + 1];
      if (begin < end) {
        const auto& iv = disk_host.intervals[begin];
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
  auto rand_host = build_host_from_device(rand_dev);

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
  auto cb_host = build_host_from_device(cb_dev);

  ASSERT_TRUE(check_csr(cb_host, dom_cb.x_min, dom_cb.x_max,
                        dom_cb.y_min, dom_cb.y_max));

  const std::size_t num_rows = cb_host.row_keys.size();
  const Coord width = dom_cb.x_max - dom_cb.x_min;
  const Coord height = dom_cb.y_max - dom_cb.y_min;

  for (std::size_t i = 0; i < num_rows; ++i) {
    const Coord y = cb_host.row_keys[i].y;
    const Coord local_y = static_cast<Coord>(y - dom_cb.y_min);
    const std::size_t block_y =
        static_cast<std::size_t>(local_y / square_size);
    const std::size_t begin = cb_host.row_ptr[i];
    const std::size_t end = cb_host.row_ptr[i + 1];
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

      const auto& iv = cb_host.intervals[begin + written];
      EXPECT_EQ(iv.begin, x0);
      EXPECT_EQ(iv.end, x1);

      ++written;
    }

    EXPECT_EQ(written, count);
    (void)height;
  }
}

TEST(CSRBuildersSmokeTest, HollowBoxBuilder) {
  HollowBox2D frame;
  frame.outer = Box2D{0, 6, 0, 5};
  frame.inner = Box2D{2, 4, 1, 4};

  auto dev = make_hollow_box_device(frame);
  auto host = build_host_from_device(dev);

  ASSERT_TRUE(check_csr(host,
                        frame.outer.x_min, frame.outer.x_max,
                        frame.outer.y_min, frame.outer.y_max));

  const std::size_t num_rows = host.row_keys.size();
  ASSERT_EQ(num_rows, 5U);

  for (std::size_t i = 0; i < num_rows; ++i) {
    const Coord y = host.row_keys[i].y;
    const std::size_t begin = host.row_ptr[i];
    const std::size_t end = host.row_ptr[i + 1];
    const std::size_t count = end - begin;

    if (y < frame.inner.y_min || y >= frame.inner.y_max) {
      ASSERT_EQ(count, 1U);
      const auto& iv = host.intervals[begin];
      EXPECT_EQ(iv.begin, frame.outer.x_min);
      EXPECT_EQ(iv.end, frame.outer.x_max);
      continue;
    }

    ASSERT_EQ(count, 2U);
    const auto& left = host.intervals[begin];
    const auto& right = host.intervals[begin + 1];
    EXPECT_EQ(left.begin, frame.outer.x_min);
    EXPECT_EQ(left.end, frame.inner.x_min);
    EXPECT_EQ(right.begin, frame.inner.x_max);
    EXPECT_EQ(right.end, frame.outer.x_max);
  }
}

TEST(CSRBuildersSmokeTest, HollowDiskBuilder) {
  HollowDisk2D disk;
  disk.cx = 0;
  disk.cy = 0;
  disk.outer_radius = 2;
  disk.inner_radius = 1;

  auto dev = make_hollow_disk_device(disk);
  auto host = build_host_from_device(dev);

  ASSERT_TRUE(check_csr(host,
                        disk.cx - disk.outer_radius,
                        disk.cx + disk.outer_radius + 1,
                        disk.cy - disk.outer_radius,
                        disk.cy + disk.outer_radius + 1));

  // Row y = 0 should have a hole: two disjoint intervals.
  for (std::size_t i = 0; i < host.row_keys.size(); ++i) {
    if (host.row_keys[i].y != 0) {
      continue;
    }
    const std::size_t begin = host.row_ptr[i];
    const std::size_t end = host.row_ptr[i + 1];
    ASSERT_EQ(end - begin, 2U);
    const auto& left = host.intervals[begin];
    const auto& right = host.intervals[begin + 1];
    const Coord expected_left_end =
        disk.cx - disk.inner_radius;
    const Coord expected_right_begin =
        disk.cx + disk.inner_radius + 1;

    EXPECT_LT(left.begin, disk.cx);
    EXPECT_EQ(left.end, expected_left_end);
    EXPECT_EQ(right.begin, expected_right_begin);
    EXPECT_GT(right.end, expected_right_begin);
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
  auto host = build_host_from_device(dev);

  ASSERT_TRUE(check_csr(host, x_min, x_min + static_cast<Coord>(width),
                        y_min, y_min + static_cast<Coord>(height)));

  ASSERT_EQ(host.row_keys.size(), height);
  ASSERT_EQ(host.row_ptr.size(), height + 1);

  // Row 0: one run [x_min, x_min+2)
  {
    const std::size_t begin = host.row_ptr[0];
    const std::size_t end = host.row_ptr[1];
    ASSERT_EQ(end - begin, 1U);
    EXPECT_EQ(host.intervals[begin].begin, x_min);
    EXPECT_EQ(host.intervals[begin].end, x_min + 2);
  }

  // Row 1: two runs [x_min+1, x_min+3) and [x_min+4, x_min+5)
  {
    const std::size_t begin = host.row_ptr[1];
    const std::size_t end = host.row_ptr[2];
    ASSERT_EQ(end - begin, 2U);
    EXPECT_EQ(host.intervals[begin].begin, x_min + 1);
    EXPECT_EQ(host.intervals[begin].end, x_min + 3);
    EXPECT_EQ(host.intervals[begin + 1].begin, x_min + 4);
    EXPECT_EQ(host.intervals[begin + 1].end, x_min + 5);
  }

  // Row 2: empty.
  EXPECT_EQ(host.row_ptr[3] - host.row_ptr[2], 0U);
}
