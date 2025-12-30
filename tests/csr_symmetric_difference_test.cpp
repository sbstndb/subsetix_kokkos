#include <gtest/gtest.h>
#include <vector>

#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>
#include "csr_test_utils.hpp"

using namespace subsetix::csr;

namespace {

IntervalSet2DHost make_set_a() {
  // Row 1: [0, 10)
  // Row 2: [0, 5), [10, 15)
  return subsetix::csr_test::make_host_csr({
    {1, {Interval{0, 10}}},
    {2, {Interval{0, 5}, Interval{10, 15}}}
  });
}

IntervalSet2DHost make_set_b() {
  // Row 1: [5, 15)  (Overlaps A)
  // Row 3: [0, 10)  (New row)
  return subsetix::csr_test::make_host_csr({
    {1, {Interval{5, 15}}},
    {3, {Interval{0, 10}}}
  });
}

// Expected XOR:
// Row 1: [0, 10) XOR [5, 15) = [0, 5) U [10, 15)
// Row 2: [0, 5), [10, 15) (Only in A)
// Row 3: [0, 10) (Only in B)
void verify_xor_result(const IntervalSet2DHost& result) {
  ASSERT_EQ(result.row_keys.extent(0), 3);

  // Row 1
  EXPECT_EQ(result.row_keys(0).y, 1);
  std::size_t count1 = result.row_ptr(1) - result.row_ptr(0);
  ASSERT_EQ(count1, 2);
  EXPECT_EQ(result.intervals(result.row_ptr(0)).begin, 0);
  EXPECT_EQ(result.intervals(result.row_ptr(0)).end, 5);
  EXPECT_EQ(result.intervals(result.row_ptr(0)+1).begin, 10);
  EXPECT_EQ(result.intervals(result.row_ptr(0)+1).end, 15);

  // Row 2
  EXPECT_EQ(result.row_keys(1).y, 2);
  std::size_t count2 = result.row_ptr(2) - result.row_ptr(1);
  ASSERT_EQ(count2, 2);
  EXPECT_EQ(result.intervals(result.row_ptr(1)).begin, 0);
  EXPECT_EQ(result.intervals(result.row_ptr(1)).end, 5);
  EXPECT_EQ(result.intervals(result.row_ptr(1)+1).begin, 10);
  EXPECT_EQ(result.intervals(result.row_ptr(1)+1).end, 15);

  // Row 3
  EXPECT_EQ(result.row_keys(2).y, 3);
  std::size_t count3 = result.row_ptr(3) - result.row_ptr(2);
  ASSERT_EQ(count3, 1);
  EXPECT_EQ(result.intervals(result.row_ptr(2)).begin, 0);
  EXPECT_EQ(result.intervals(result.row_ptr(2)).end, 10);
}

} // namespace

TEST(CSRSetOpsSmokeTest, SymmetricDifference) {
  auto host_a = make_set_a();
  auto host_b = make_set_b();

  auto dev_a = to<DeviceMemorySpace>(host_a);
  auto dev_b = to<DeviceMemorySpace>(host_b);

  // Reuse workspace
  CsrSetAlgebraContext ctx;
  
  // Allocate conservatively (union size is safe upper bound)
  auto dev_out = allocate_interval_set_device(
      dev_a.num_rows + dev_b.num_rows,
      dev_a.num_intervals + dev_b.num_intervals);
  
  set_symmetric_difference_device(dev_a, dev_b, dev_out, ctx);

  auto host_out = to<HostMemorySpace>(dev_out);
  verify_xor_result(host_out);
}

TEST(CSRSetOpsSmokeTest, SymmetricDifferenceEmpty) {
  IntervalSet2DHost empty;
  auto host_a = make_set_a();
  
  auto dev_empty = to<DeviceMemorySpace>(empty);
  auto dev_a = to<DeviceMemorySpace>(host_a);
  
  CsrSetAlgebraContext ctx;
  auto dev_out = allocate_interval_set_device(
      dev_a.num_rows + dev_empty.num_rows,
      dev_a.num_intervals + dev_empty.num_intervals);
  
  // A XOR Empty = A
  set_symmetric_difference_device(dev_a, dev_empty, dev_out, ctx);
  auto host_out_1 = to<HostMemorySpace>(dev_out);
  EXPECT_EQ(host_out_1.row_keys.extent(0), 2);
  EXPECT_EQ(host_out_1.intervals.extent(0), 3);

  // Empty XOR A = A
  set_symmetric_difference_device(dev_empty, dev_a, dev_out, ctx);
  auto host_out_2 = to<HostMemorySpace>(dev_out);
  EXPECT_EQ(host_out_2.row_keys.extent(0), 2);
  EXPECT_EQ(host_out_2.intervals.extent(0), 3);
}
