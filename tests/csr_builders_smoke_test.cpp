#include <Kokkos_Core.hpp>
#include <subsetix/csr_interval_set.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  int result = 0;
  {
    using namespace subsetix::csr;

    // 1) Rectangle builder: box [0,10) x [0,3)
    Box2D box;
    box.x_min = 0;
    box.x_max = 10;
    box.y_min = 0;
    box.y_max = 3;

    auto rect_dev = make_box_device(box);
    auto rect_host = build_host_from_device(rect_dev);

    if (rect_host.row_keys.size() != 3 ||
        rect_host.row_ptr.size() != 4 ||
        rect_host.intervals.size() != 3) {
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
    } else {
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

    if (rand_host.row_keys.size() != 10 ||
        rand_host.row_ptr.size() != 11) {
      result = 1;
    } else {
      // Check CSR invariants and bounds.
      for (std::size_t i = 0; i < rand_host.row_keys.size(); ++i) {
        const auto y = rand_host.row_keys[i].y;
        if (y < dom.y_min || y >= dom.y_max) {
          result = 1;
          break;
        }
        const std::size_t begin = rand_host.row_ptr[i];
        const std::size_t end = rand_host.row_ptr[i + 1];
        if (end < begin || end > rand_host.intervals.size()) {
          result = 1;
          break;
        }
        for (std::size_t k = begin; k < end; ++k) {
          const auto& iv = rand_host.intervals[k];
          if (!(iv.begin < iv.end)) {
            result = 1;
            break;
          }
          if (iv.begin < dom.x_min || iv.end > dom.x_max) {
            result = 1;
            break;
          }
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

