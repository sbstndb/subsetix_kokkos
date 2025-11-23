#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>

#include "csr_test_utils.hpp"

using namespace subsetix::csr;
using namespace subsetix::csr_test;

namespace {

IntervalSet2DDevice run_difference(const IntervalSet2DDevice& lhs,
                                   const IntervalSet2DDevice& rhs) {
  CsrSetAlgebraContext ctx;
  auto out = allocate_interval_set_device(lhs.num_rows,
                                          lhs.num_intervals + rhs.num_intervals);
  set_difference_device(lhs, rhs, out, ctx);
  return out;
}

} // namespace

TEST(CSRDifferenceSmokeTest, EmptyAndNonEmpty) {
  IntervalSet2DDevice empty_a;
  IntervalSet2DDevice empty_b;

  auto D0 = run_difference(empty_a, empty_b);
  EXPECT_EQ(D0.num_rows, 0u);
  EXPECT_EQ(D0.num_intervals, 0u);

  Box2D box;
  box.x_min = 0;
  box.x_max = 4;
  box.y_min = 0;
  box.y_max = 2;

  IntervalSet2DDevice A = make_box_device(box);

  // A \ empty = A
  auto D1 = run_difference(A, empty_b);
  auto host_A = build_host_from_device(A);
  auto host_D1 = build_host_from_device(D1);

  ASSERT_EQ(host_A.row_keys.size(), host_D1.row_keys.size());
  ASSERT_EQ(host_A.row_ptr.size(), host_D1.row_ptr.size());
  ASSERT_EQ(host_A.intervals.size(), host_D1.intervals.size());

  for (std::size_t i = 0; i < host_A.row_keys.size(); ++i) {
    EXPECT_EQ(host_A.row_keys[i].y, host_D1.row_keys[i].y);
  }
  for (std::size_t i = 0; i < host_A.row_ptr.size(); ++i) {
    EXPECT_EQ(host_A.row_ptr[i], host_D1.row_ptr[i]);
  }
  for (std::size_t i = 0; i < host_A.intervals.size(); ++i) {
    EXPECT_EQ(host_A.intervals[i].begin, host_D1.intervals[i].begin);
    EXPECT_EQ(host_A.intervals[i].end, host_D1.intervals[i].end);
  }

  // empty \ A = empty
  auto D2 = run_difference(empty_b, A);
  EXPECT_EQ(D2.num_rows, 0u);
  EXPECT_EQ(D2.num_intervals, 0u);
}

TEST(CSRDifferenceSmokeTest, EqualBoxesSameRows) {
  Box2D box;
  box.x_min = 0;
  box.x_max = 4;
  box.y_min = 0;
  box.y_max = 2;

  IntervalSet2DDevice A = make_box_device(box);
  IntervalSet2DDevice B = make_box_device(box);

  auto D = run_difference(A, B);

  EXPECT_EQ(D.num_rows, 2u);
  EXPECT_EQ(D.num_intervals, 0u);
}

TEST(CSRDifferenceSmokeTest, OverlappingBoxesSameRows) {
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

  auto D = run_difference(A, B);
  auto host_D = build_host_from_device(D);

  ASSERT_EQ(host_D.row_keys.size(), 2u);
  for (std::size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(host_D.row_keys[i].y, static_cast<Coord>(i));
  }

  ASSERT_EQ(host_D.row_ptr.size(), 3u);
  EXPECT_EQ(host_D.row_ptr[0], 0u);
  EXPECT_EQ(host_D.row_ptr[1], 2u);
  EXPECT_EQ(host_D.row_ptr[2], 4u);

  ASSERT_EQ(host_D.intervals.size(), 4u);

  const auto& iv0 = host_D.intervals[0];
  const auto& iv1 = host_D.intervals[1];
  const auto& iv2 = host_D.intervals[2];
  const auto& iv3 = host_D.intervals[3];

  EXPECT_EQ(iv0.begin, 0);
  EXPECT_EQ(iv0.end, 2);
  EXPECT_EQ(iv1.begin, 4);
  EXPECT_EQ(iv1.end, 6);
  EXPECT_EQ(iv2.begin, 0);
  EXPECT_EQ(iv2.end, 2);
  EXPECT_EQ(iv3.begin, 4);
  EXPECT_EQ(iv3.end, 6);
}

TEST(CSRDifferenceSmokeTest, BoxesOnDisjointRows) {
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

  auto D = run_difference(A, B);
  auto host_A = build_host_from_device(A);
  auto host_D = build_host_from_device(D);

  expect_equal_csr(host_A, host_D);
}
