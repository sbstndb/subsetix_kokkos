#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>

TEST(CSRIntervalSetSmokeTest, HostDeviceRoundtrip) {
  using namespace subsetix::csr;

  // Build host using the new helper
  auto host = make_interval_set_host(
      {{0}, {5}},           // row_keys (y=0, y=5)
      {0, 1, 2},            // row_ptr
      {{0, 10}, {5, 8}}     // intervals
  );

  auto dev = to<DeviceMemorySpace>(host);
  auto host_roundtrip = to<HostMemorySpace>(dev);

  ASSERT_EQ(host_roundtrip.num_rows, host.num_rows);
  ASSERT_EQ(host_roundtrip.num_intervals, host.num_intervals);

  for (std::size_t i = 0; i < host.num_rows; ++i) {
    EXPECT_EQ(host_roundtrip.row_keys(i).y, host.row_keys(i).y);
  }
}
