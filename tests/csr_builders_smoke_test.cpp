#include <Kokkos_Core.hpp>
#include <subsetix/csr_interval_set.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  int result = 0;
  {
    using namespace subsetix::csr;

    auto check_csr = [](const IntervalSet2DHost& host,
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
    };

    // 1) Rectangle builder: box [0,10) x [0,3)
    Box2D box;
    box.x_min = 0;
    box.x_max = 10;
    box.y_min = 0;
    box.y_max = 3;

    auto rect_dev = make_box_device(box);
    auto rect_host = build_host_from_device(rect_dev);

    if (!check_csr(rect_host, box.x_min, box.x_max, box.y_min, box.y_max)) {
      result = 1;
    } else {
      for (std::size_t i = 0; i < 3; ++i) {
        if (rect_host.row_keys[i].y != static_cast<Coord>(i)) {
          result = 1;
          break;
        }
        if (rect_host.intervals[i].begin != 0 ||
            rect_host.intervals[i].end != 10) {
          result = 1;
          break;
        }
      }
    }

    if (result != 0) {
      Kokkos::finalize();
      return result;
    }

    // 2) Disk builder: disk centred at (0,0) radius 1 -> rows y=-1,0,1.
    Disk2D disk;
    disk.cx = 0;
    disk.cy = 0;
    disk.radius = 1;

    auto disk_dev = make_disk_device(disk);
    auto disk_host = build_host_from_device(disk_dev);

    if (disk_host.row_keys.size() == 0) {
      result = 1;
    } else if (check_csr(disk_host,
                         static_cast<Coord>(disk.cx - disk.radius),
                         static_cast<Coord>(disk.cx + disk.radius + 1),
                         static_cast<Coord>(disk.cy - disk.radius),
                         static_cast<Coord>(disk.cy + disk.radius + 1))) {
      // At least the middle row y=0 should have an interval containing 0.
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
      if (!ok_middle) {
        result = 1;
      }
    } else {
      result = 1;
    }

    if (result != 0) {
      Kokkos::finalize();
      return result;
    }

    // 3) Random builder: domain [0,20) x [0,10), moderate fill prob.
    Domain2D dom;
    dom.x_min = 0;
    dom.x_max = 20;
    dom.y_min = 0;
    dom.y_max = 10;

    auto rand_dev = make_random_device(dom, 0.3, 12345);
    auto rand_host = build_host_from_device(rand_dev);

    if (!check_csr(rand_host, dom.x_min, dom.x_max, dom.y_min, dom.y_max)) {
      result = 1;
    }

    if (result != 0) {
      Kokkos::finalize();
      return result;
    }

    // 4) Checkerboard builder: domain [0,8) x [0,4), square size 2.
    Domain2D dom_cb;
    dom_cb.x_min = 0;
    dom_cb.x_max = 8;
    dom_cb.y_min = 0;
    dom_cb.y_max = 4;

    const Coord square_size = 2;

    auto cb_dev = make_checkerboard_device(dom_cb, square_size);
    auto cb_host = build_host_from_device(cb_dev);

    if (!check_csr(cb_host, dom_cb.x_min, dom_cb.x_max,
                   dom_cb.y_min, dom_cb.y_max)) {
      result = 1;
    } else {
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

        // Expected number of filled blocks along X on this row.
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

        if (count != expected) {
          result = 1;
          break;
        }

        Coord x_expected = dom_cb.x_min;
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

          if (begin + written >= end) {
            result = 1;
            break;
          }

          const auto& iv = cb_host.intervals[begin + written];
          if (iv.begin != x0 || iv.end != x1) {
            result = 1;
            break;
          }

          ++written;
        }

        if (result != 0) {
          break;
        }

        if (written != count) {
          result = 1;
          break;
        }

        (void)height;
      }
    }
  }

  Kokkos::finalize();
  return result;
}
