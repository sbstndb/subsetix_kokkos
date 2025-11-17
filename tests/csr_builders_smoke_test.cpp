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

    // 4) Checkerboard builder: domain [0,8) x [0,2).
    Domain2D dom_cb;
    dom_cb.x_min = 0;
    dom_cb.x_max = 8;
    dom_cb.y_min = 0;
    dom_cb.y_max = 2;

    auto cb_dev = make_checkerboard_device(dom_cb);
    auto cb_host = build_host_from_device(cb_dev);

    if (!check_csr(cb_host, dom_cb.x_min, dom_cb.x_max,
                   dom_cb.y_min, dom_cb.y_max)) {
      result = 1;
    } else {
      const std::size_t num_rows = cb_host.row_keys.size();
      const Coord width = dom_cb.x_max - dom_cb.x_min;

      for (std::size_t i = 0; i < num_rows; ++i) {
        const Coord y = cb_host.row_keys[i].y;
        const bool even_parity = (((dom_cb.x_min + y) & 1) == 0);
        const std::size_t begin = cb_host.row_ptr[i];
        const std::size_t end = cb_host.row_ptr[i + 1];
        const std::size_t count = end - begin;

        const std::size_t expected =
            even_parity ? static_cast<std::size_t>((width + 1) / 2)
                        : static_cast<std::size_t>(width / 2);

        if (count != expected) {
          result = 1;
          break;
        }

        Coord x_expected =
            even_parity ? dom_cb.x_min : static_cast<Coord>(dom_cb.x_min + 1);
        for (std::size_t k = begin; k < end; ++k) {
          const auto& iv = cb_host.intervals[k];
          if (iv.begin != x_expected || iv.end != x_expected + 1) {
            result = 1;
            break;
          }
          x_expected = static_cast<Coord>(x_expected + 2);
        }

        if (result != 0) {
          break;
        }
      }
    }
  }

  Kokkos::finalize();
  return result;
}
