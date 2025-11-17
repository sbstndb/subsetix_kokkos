#include <Kokkos_Core.hpp>
#include <subsetix/csr_interval_set.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  int result = 0;
  {
    using namespace subsetix::csr;

    IntervalSet2DHost host;
    host.row_keys.push_back(RowKey2D{0});
    host.row_keys.push_back(RowKey2D{5});

    host.row_ptr.push_back(0);
    host.intervals.push_back(Interval{0, 10}); // row 0
    host.row_ptr.push_back(host.intervals.size());
    host.intervals.push_back(Interval{5, 8});  // row 1
    host.row_ptr.push_back(host.intervals.size());

    auto dev = build_device_from_host(host);
    auto host_roundtrip = build_host_from_device(dev);

    if (host_roundtrip.row_keys.size() != host.row_keys.size()) {
      result = 1;
    } else if (host_roundtrip.row_ptr.size() != host.row_ptr.size()) {
      result = 1;
    } else if (host_roundtrip.intervals.size() != host.intervals.size()) {
      result = 1;
    } else {
      for (std::size_t i = 0; i < host.row_keys.size(); ++i) {
        if (host_roundtrip.row_keys[i].y != host.row_keys[i].y) {
          result = 1;
          break;
        }
      }
    }
  }

  Kokkos::finalize();
  return result;
}

