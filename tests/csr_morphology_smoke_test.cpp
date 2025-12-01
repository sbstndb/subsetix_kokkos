#include <gtest/gtest.h>
#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>
#include <subsetix/field/csr_field.hpp>
#include <subsetix/field/csr_field_ops.hpp>

namespace {

using namespace subsetix::csr;

class CsrMorphologyTest : public ::testing::Test {
protected:
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice A, B;

  void SetUp() override {
    // Warmup
  }

  void TearDown() override {
    // Cleanup handled by views
  }
};

TEST_F(CsrMorphologyTest, ExpandBox) {
  Box2D box{10, 20, 10, 20};
  A = make_box_device(box);
  
  IntervalSet2DDevice out = allocate_interval_set_device(1000, 1000); // Sufficient capacity
  
  // Expand by 1 in X and 1 in Y
  expand_device(A, 1, 1, out, ctx);
  
  auto h_out = to<HostMemorySpace>(out);
  
  // Expected: Box 9..21, 9..21
  EXPECT_EQ(h_out.num_rows, 12u); // 21 - 9

  for(size_t i=0; i<h_out.num_rows; ++i) {
    EXPECT_EQ(h_out.row_keys(i).y, 9 + (int)i);
    EXPECT_EQ(h_out.intervals(h_out.row_ptr(i)).begin, 9);
    EXPECT_EQ(h_out.intervals(h_out.row_ptr(i)).end, 21);
  }
}

TEST_F(CsrMorphologyTest, ShrinkBox) {
  Box2D box{10, 20, 10, 20}; // 10x10 box
  A = make_box_device(box);
  
  IntervalSet2DDevice out = allocate_interval_set_device(1000, 1000);
  
  // Shrink by 2 in X and 1 in Y
  // Expected: Box 12..18 (X), 11..19 (Y)
  // Y range: 10+1 .. 20-1 = 11..19 (size 8)
  // X range: 10+2 .. 20-2 = 12..18 (size 6)
  shrink_device(A, 2, 1, out, ctx);
  
  auto h_out = to<HostMemorySpace>(out);

  EXPECT_EQ(h_out.num_rows, 8u);

  if (h_out.num_rows > 0) {
    EXPECT_EQ(h_out.row_keys(0).y, 11);
    EXPECT_EQ(h_out.row_keys(h_out.num_rows - 1).y, 18);

    for(size_t i=0; i<h_out.num_rows; ++i) {
        EXPECT_EQ(h_out.intervals(h_out.row_ptr(i)).begin, 12);
        EXPECT_EQ(h_out.intervals(h_out.row_ptr(i)).end, 18);
    }
  }
}

TEST_F(CsrMorphologyTest, ExpandEmpty) {
    IntervalSet2DDevice empty_set;
    IntervalSet2DDevice out = allocate_interval_set_device(100, 100);
    expand_device(empty_set, 2, 2, out, ctx);
    EXPECT_EQ(out.num_rows, 0);
}

TEST_F(CsrMorphologyTest, ShrinkToEmpty) {
    Box2D box{10, 12, 10, 12}; // 2x2 box
    A = make_box_device(box);
    IntervalSet2DDevice out = allocate_interval_set_device(100, 100);
    
    shrink_device(A, 1, 1, out, ctx); // Should exactly disappear
    EXPECT_EQ(out.num_rows, 0);
}

TEST_F(CsrMorphologyTest, ExpandWithGaps) {
    // Row 10: [10, 20]
    // Row 12: [10, 20]
    // Gap at 11
    
    // Expand Y=1 should bridge the gap
    
    // Construct manual set
    IntervalSet2DHost h = make_interval_set_host(
        {{10}, {12}},
        {0, 1, 2},
        {{10, 20}, {10, 20}}
    );

    A = to<DeviceMemorySpace>(h);
    IntervalSet2DDevice out = allocate_interval_set_device(100, 100);
    
    expand_device(A, 0, 1, out, ctx);
    
    auto h_out = to<HostMemorySpace>(out);

    // 10 expands to 9,10,11
    // 12 expands to 11,12,13
    // Union: 9, 10, 11, 12, 13

    EXPECT_EQ(h_out.num_rows, 5u);
    EXPECT_EQ(h_out.row_keys(0).y, 9);
    EXPECT_EQ(h_out.row_keys(4).y, 13);
}

TEST_F(CsrMorphologyTest, ShrinkRequiresContiguity) {
    // Row 10: [10, 20]
    // Row 12: [10, 20]
    // Gap at 11
    
    // Shrink Y=1 needs neighbors +1 and -1.
    // For 11 to exist, we need 10, 11, 12 in input. 11 missing.
    // For 10 to exist, need 9, 10, 11. 9, 11 missing.
    // Result should be empty.
    
    IntervalSet2DHost h = make_interval_set_host(
        {{10}, {12}},
        {0, 1, 2},
        {{10, 20}, {10, 20}}
    );

    A = to<DeviceMemorySpace>(h);
    IntervalSet2DDevice out = allocate_interval_set_device(100, 100);
    
    shrink_device(A, 0, 1, out, ctx);
    
    EXPECT_EQ(out.num_rows, 0);
}

} // namespace
