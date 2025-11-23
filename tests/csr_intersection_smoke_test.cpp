#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>

using namespace subsetix::csr;

namespace {

IntervalSet2DDevice run_intersection(const IntervalSet2DDevice& lhs,
                                     const IntervalSet2DDevice& rhs) {
  CsrSetAlgebraContext ctx;
  auto out = allocate_interval_set_device(
      lhs.num_rows + rhs.num_rows, lhs.num_intervals + rhs.num_intervals);
  set_intersection_device(lhs, rhs, out, ctx);
  return out;
}

} // namespace

TEST(CSRIntersectionSmokeTest, EmptyIntersection) {
  IntervalSet2DDevice empty_a;
  IntervalSet2DDevice empty_b;

  auto I = run_intersection(empty_a, empty_b);
  EXPECT_EQ(I.num_rows, 0u);
  EXPECT_EQ(I.num_intervals, 0u);

  Box2D box;
  box.x_min = 0;
  box.x_max = 4;
  box.y_min = 0;
  box.y_max = 2;

  IntervalSet2DDevice A = make_box_device(box);

  auto I2 = run_intersection(A, empty_b);
  auto I3 = run_intersection(empty_b, A);

  EXPECT_EQ(I2.num_rows, 0u);
  EXPECT_EQ(I2.num_intervals, 0u);
  EXPECT_EQ(I3.num_rows, 0u);
  EXPECT_EQ(I3.num_intervals, 0u);
}

TEST(CSRIntersectionSmokeTest, OverlappingBoxesSameRows) {
  Box2D boxA;
  boxA.x_min = 0;
  boxA.x_max = 6;
  boxA.y_min = 0;
  boxA.y_max = 2;

  Box2D boxB;
  boxB.x_min = 2;
  boxB.x_max = 4;
  boxB.y_min = 0;
  boxB.y_max = 2;

  IntervalSet2DDevice A = make_box_device(boxA);
  IntervalSet2DDevice B = make_box_device(boxB);

  auto I = run_intersection(A, B);
  auto host_I = build_host_from_device(I);

  ASSERT_EQ(host_I.row_keys.size(), 2u);
  for (std::size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(host_I.row_keys[i].y, static_cast<Coord>(i));
  }

  ASSERT_EQ(host_I.row_ptr.size(), 3u);
  EXPECT_EQ(host_I.row_ptr[0], 0u);
  EXPECT_EQ(host_I.row_ptr[1], 1u);
  EXPECT_EQ(host_I.row_ptr[2], 2u);

  ASSERT_EQ(host_I.intervals.size(), 2u);
  for (const auto& iv : host_I.intervals) {
    EXPECT_EQ(iv.begin, 2);
    EXPECT_EQ(iv.end, 4);
  }
}

TEST(CSRIntersectionSmokeTest, BoxesOnDisjointRows) {
  Box2D boxA;
  boxA.x_min = 0;
  boxA.x_max = 4;
  boxA.y_min = 0;
  boxA.y_max = 2; // rows 0,1

  Box2D boxB;
  boxB.x_min = 0;
  boxB.x_max = 4;
  boxB.y_min = 3;
  boxB.y_max = 5; // rows 3,4

  IntervalSet2DDevice A = make_box_device(boxA);
  IntervalSet2DDevice B = make_box_device(boxB);

  auto I = run_intersection(A, B);

  EXPECT_EQ(I.num_rows, 0u);
  EXPECT_EQ(I.num_intervals, 0u);
}
