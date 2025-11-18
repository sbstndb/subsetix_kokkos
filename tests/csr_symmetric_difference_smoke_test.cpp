#include <gtest/gtest.h>
#include <vector>

#include <subsetix/csr_set_ops.hpp>
#include "csr_test_utils.hpp"

using namespace subsetix::csr;

namespace {

IntervalSet2DHost make_set_a() {
  // Row 1: [0, 10)
  // Row 2: [0, 5), [10, 15)
  IntervalSet2DHost host;
  host.row_keys = {RowKey2D{1}, RowKey2D{2}};
  host.row_ptr = {0, 1, 3};
  host.intervals = {
    Interval{0, 10},
    Interval{0, 5}, Interval{10, 15}
  };
  return host;
}

IntervalSet2DHost make_set_b() {
  // Row 1: [5, 15)  (Overlaps A)
  // Row 3: [0, 10)  (New row)
  IntervalSet2DHost host;
  host.row_keys = {RowKey2D{1}, RowKey2D{3}};
  host.row_ptr = {0, 1, 2};
  host.intervals = {
    Interval{5, 15},
    Interval{0, 10}
  };
  return host;
}

// Expected XOR:
// Row 1: [0, 10) XOR [5, 15) = [0, 5) U [10, 15)
// Row 2: [0, 5), [10, 15) (Only in A)
// Row 3: [0, 10) (Only in B)
void verify_xor_result(const IntervalSet2DHost& result) {
  ASSERT_EQ(result.row_keys.size(), 3);
  
  // Row 1
  EXPECT_EQ(result.row_keys[0].y, 1);
  std::size_t count1 = result.row_ptr[1] - result.row_ptr[0];
  ASSERT_EQ(count1, 2);
  EXPECT_EQ(result.intervals[result.row_ptr[0]].begin, 0);
  EXPECT_EQ(result.intervals[result.row_ptr[0]].end, 5);
  EXPECT_EQ(result.intervals[result.row_ptr[0]+1].begin, 10);
  EXPECT_EQ(result.intervals[result.row_ptr[0]+1].end, 15);

  // Row 2
  EXPECT_EQ(result.row_keys[1].y, 2);
  std::size_t count2 = result.row_ptr[2] - result.row_ptr[1];
  ASSERT_EQ(count2, 2);
  EXPECT_EQ(result.intervals[result.row_ptr[1]].begin, 0);
  EXPECT_EQ(result.intervals[result.row_ptr[1]].end, 5);
  EXPECT_EQ(result.intervals[result.row_ptr[1]+1].begin, 10);
  EXPECT_EQ(result.intervals[result.row_ptr[1]+1].end, 15);

  // Row 3
  EXPECT_EQ(result.row_keys[2].y, 3);
  std::size_t count3 = result.row_ptr[3] - result.row_ptr[2];
  ASSERT_EQ(count3, 1);
  EXPECT_EQ(result.intervals[result.row_ptr[2]].begin, 0);
  EXPECT_EQ(result.intervals[result.row_ptr[2]].end, 10);
}

} // namespace

TEST(CSRSetOpsSmokeTest, SymmetricDifference) {
  auto host_a = make_set_a();
  auto host_b = make_set_b();

  auto dev_a = build_device_from_host(host_a);
  auto dev_b = build_device_from_host(host_b);

  // Reuse workspace
  CsrSetAlgebraContext ctx;
  
  // Allocate conservatively (union size is safe upper bound)
  auto dev_out = allocate_union_output_buffer(dev_a, dev_b);
  
  set_symmetric_difference_device(dev_a, dev_b, dev_out, ctx);

  auto host_out = build_host_from_device(dev_out);
  verify_xor_result(host_out);
}

TEST(CSRSetOpsSmokeTest, SymmetricDifferenceEmpty) {
  IntervalSet2DHost empty;
  auto host_a = make_set_a();
  
  auto dev_empty = build_device_from_host(empty);
  auto dev_a = build_device_from_host(host_a);
  
  CsrSetAlgebraContext ctx;
  auto dev_out = allocate_union_output_buffer(dev_a, dev_empty);
  
  // A XOR Empty = A
  set_symmetric_difference_device(dev_a, dev_empty, dev_out, ctx);
  auto host_out_1 = build_host_from_device(dev_out);
  EXPECT_EQ(host_out_1.row_keys.size(), 2);
  EXPECT_EQ(host_out_1.intervals.size(), 3);
  
  // Empty XOR A = A
  set_symmetric_difference_device(dev_empty, dev_a, dev_out, ctx);
  auto host_out_2 = build_host_from_device(dev_out);
  EXPECT_EQ(host_out_2.row_keys.size(), 2);
  EXPECT_EQ(host_out_2.intervals.size(), 3);
}
