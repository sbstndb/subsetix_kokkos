#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_set_ops.hpp>

#include "csr_csr_test_utils.hpp"

using namespace subsetix::csr;
using namespace subsetix::csr_test;

namespace {

IntervalSet2DDevice run_union(const IntervalSet2DDevice& lhs,
                              const IntervalSet2DDevice& rhs) {
  CsrSetAlgebraContext ctx;
  auto out = allocate_union_output_buffer(lhs, rhs);
  set_union_device(lhs, rhs, out, ctx);
  return out;
}

} // namespace

TEST(CSRUnionSmokeTest, EmptyUnion) {
  IntervalSet2DDevice empty_a;
  IntervalSet2DDevice empty_b;

  auto u = run_union(empty_a, empty_b);
  EXPECT_EQ(u.num_rows, 0u);
  EXPECT_EQ(u.num_intervals, 0u);
}

TEST(CSRUnionSmokeTest, EmptyAndNonEmpty) {
  Box2D box;
  box.x_min = 0;
  box.x_max = 4;
  box.y_min = 0;
  box.y_max = 2;

  IntervalSet2DDevice A = make_box_device(box);
  IntervalSet2DDevice B; // empty

  auto u1 = run_union(A, B);
  auto u2 = run_union(B, A);

  auto host_A = build_host_from_device(A);
  auto host_u1 = build_host_from_device(u1);
  auto host_u2 = build_host_from_device(u2);

  expect_equal_csr(host_A, host_u1);
  expect_equal_csr(host_A, host_u2);
}

TEST(CSRUnionSmokeTest, OverlappingBoxesSameRows) {
  Box2D boxA;
  boxA.x_min = 0;
  boxA.x_max = 4;
  boxA.y_min = 0;
  boxA.y_max = 2;

  Box2D boxB;
  boxB.x_min = 2;
  boxB.x_max = 6;
  boxB.y_min = 0;
  boxB.y_max = 2;

  IntervalSet2DDevice A = make_box_device(boxA);
  IntervalSet2DDevice B = make_box_device(boxB);

  auto U = run_union(A, B);
  auto host_U = build_host_from_device(U);

  ASSERT_EQ(host_U.row_keys.size(), 2u);
  for (std::size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(host_U.row_keys[i].y, static_cast<Coord>(i));
  }

  ASSERT_EQ(host_U.row_ptr.size(), 3u);
  EXPECT_EQ(host_U.row_ptr[0], 0u);
  EXPECT_EQ(host_U.row_ptr[1], 1u);
  EXPECT_EQ(host_U.row_ptr[2], 2u);

  ASSERT_EQ(host_U.intervals.size(), 2u);
  for (std::size_t i = 0; i < 2; ++i) {
    const auto& iv = host_U.intervals[i];
    EXPECT_EQ(iv.begin, 0);
    EXPECT_EQ(iv.end, 6);
  }
}

TEST(CSRUnionSmokeTest, BoxesOnDisjointRows) {
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

  auto U = run_union(A, B);
  auto host_U = build_host_from_device(U);

  ASSERT_EQ(host_U.row_keys.size(), 4u);
  const Coord expected_y[4] = {0, 1, 3, 4};
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(host_U.row_keys[i].y, expected_y[i]);
  }

  ASSERT_EQ(host_U.row_ptr.size(), 5u);
  EXPECT_EQ(host_U.row_ptr[0], 0u);
  EXPECT_EQ(host_U.row_ptr[1], 1u);
  EXPECT_EQ(host_U.row_ptr[2], 2u);
  EXPECT_EQ(host_U.row_ptr[3], 3u);
  EXPECT_EQ(host_U.row_ptr[4], 4u);

  ASSERT_EQ(host_U.intervals.size(), 4u);
  for (std::size_t i = 0; i < 4; ++i) {
    const auto& iv = host_U.intervals[i];
    EXPECT_EQ(iv.begin, 0);
    EXPECT_EQ(iv.end, 4);
  }
}

TEST(CSRUnionSmokeTest, SameRowMultipleDisjointIntervals) {
  IntervalSet2DHost hostA;
  IntervalSet2DHost hostB;

  hostA = make_host_csr({
      {0, {Interval{0, 2}, Interval{8, 10}}},
  });

  hostB = make_host_csr({
      {0, {Interval{2, 5}}},
  });

  auto A = build_device_from_host(hostA);
  auto B = build_device_from_host(hostB);

  auto U = run_union(A, B);
  auto host_U = build_host_from_device(U);

  ASSERT_EQ(host_U.row_keys.size(), 1u);
  EXPECT_EQ(host_U.row_keys[0].y, 0);

  ASSERT_EQ(host_U.row_ptr.size(), 2u);
  EXPECT_EQ(host_U.row_ptr[0], 0u);
  EXPECT_EQ(host_U.row_ptr[1], 2u);

  ASSERT_EQ(host_U.intervals.size(), 2u);
  const auto& iv0 = host_U.intervals[0];
  const auto& iv1 = host_U.intervals[1];
  EXPECT_EQ(iv0.begin, 0);
  EXPECT_EQ(iv0.end, 5);
  EXPECT_EQ(iv1.begin, 8);
  EXPECT_EQ(iv1.end, 10);
}
