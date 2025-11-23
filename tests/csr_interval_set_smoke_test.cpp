#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>

TEST(CSRIntervalSetSmokeTest, HostDeviceRoundtrip) {
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

  ASSERT_EQ(host_roundtrip.row_keys.size(), host.row_keys.size());
  ASSERT_EQ(host_roundtrip.row_ptr.size(), host.row_ptr.size());
  ASSERT_EQ(host_roundtrip.intervals.size(), host.intervals.size());

  for (std::size_t i = 0; i < host.row_keys.size(); ++i) {
    EXPECT_EQ(host_roundtrip.row_keys[i].y, host.row_keys[i].y);
  }
}

