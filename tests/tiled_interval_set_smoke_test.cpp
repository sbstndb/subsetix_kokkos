#include <Kokkos_Core.hpp>
#include <subsetix/tiled_interval_set.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  int result = 0;
  {
    using namespace subsetix::tiled;

    TiledIntervalSet2DHost host;
    host.min_y = 0;
    host.rows_per_tile = 4;
    host.tiles.resize(2);

    // Tile 0: one row at y = 0 with interval [0, 10).
    {
      auto& tile0 = host.tiles[0];
      tile0.row_keys.push_back(RowKey2D{0});
      tile0.row_ptr.push_back(0);
      tile0.intervals.push_back(Interval{0, 10});
      tile0.row_ptr.push_back(tile0.intervals.size());
    }

    // Tile 1: one row at y = 5 with interval [5, 8).
    {
      auto& tile1 = host.tiles[1];
      tile1.row_keys.push_back(RowKey2D{5});
      tile1.row_ptr.push_back(0);
      tile1.intervals.push_back(Interval{5, 8});
      tile1.row_ptr.push_back(tile1.intervals.size());
    }

    // Build device representation and back to host.
    auto dev = build_device_from_host(host);
    auto host_roundtrip = build_host_from_device(dev);

    if (host_roundtrip.tiles.size() != host.tiles.size()) {
      result = 1;
    } else if (host_roundtrip.tiles[0].row_keys.size() != 1 ||
               host_roundtrip.tiles[1].row_keys.size() != 1) {
      result = 1;
    } else if (host_roundtrip.tiles[0].row_keys[0].y != 0 ||
               host_roundtrip.tiles[1].row_keys[0].y != 5) {
      result = 1;
    }
  }

  Kokkos::finalize();
  return result;
}
