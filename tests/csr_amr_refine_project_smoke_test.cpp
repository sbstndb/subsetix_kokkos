#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_set_ops.hpp>

#include "csr_test_utils.hpp"

using namespace subsetix::csr;
using namespace subsetix::csr_test;

TEST(CSRAmrRefineProjectTest, RefineThenProjectBoxIsIdentity) {
  Box2D box;
  box.x_min = 0;
  box.x_max = 8;
  box.y_min = 0;
  box.y_max = 4;

  auto coarse = make_box_device(box);
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice fine, coarse_back;
  refine_level_up_device(coarse, fine, ctx);
  project_level_down_device(fine, coarse_back, ctx);

  auto host_coarse = build_host_from_device(coarse);
  auto host_back = build_host_from_device(coarse_back);

  expect_equal_csr(host_back, host_coarse);
}

TEST(CSRAmrRefineProjectTest, RefineDuplicatesRowsAndIntervals) {
  IntervalSet2DHost coarse_host =
      make_host_csr({
          {0, {Interval{0, 1}, Interval{2, 3}}},
          {2, {Interval{1, 4}}},
      });

  auto coarse = build_device_from_host(coarse_host);
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice fine;
  refine_level_up_device(coarse, fine, ctx);
  auto fine_host = build_host_from_device(fine);

  ASSERT_EQ(fine_host.row_keys.size(),
            coarse_host.row_keys.size() * 2);
  ASSERT_EQ(fine_host.intervals.size(),
            coarse_host.intervals.size() * 2);

  std::vector<Coord> expected_rows = {0, 1, 4, 5};
  for (std::size_t i = 0; i < expected_rows.size(); ++i) {
    EXPECT_EQ(fine_host.row_keys[i].y, expected_rows[i]);
  }

  std::vector<Interval> expected_intervals = {
      Interval{0, 2}, Interval{4, 6},
      Interval{0, 2}, Interval{4, 6},
      Interval{2, 8}, Interval{2, 8},
  };
  ASSERT_EQ(fine_host.intervals.size(),
            expected_intervals.size());
  for (std::size_t i = 0; i < expected_intervals.size(); ++i) {
    EXPECT_EQ(fine_host.intervals[i].begin,
              expected_intervals[i].begin);
    EXPECT_EQ(fine_host.intervals[i].end,
              expected_intervals[i].end);
  }
}

TEST(CSRAmrRefineProjectTest, RefineBoxCardinalityFactorFour) {
  Box2D box;
  box.x_min = 0;
  box.x_max = 10;
  box.y_min = 0;
  box.y_max = 5;

  auto coarse = make_box_device(box);
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice fine;
  refine_level_up_device(coarse, fine, ctx);

  auto host_coarse = build_host_from_device(coarse);
  auto host_fine = build_host_from_device(fine);

  const std::size_t card_coarse = cardinality(host_coarse);
  const std::size_t card_fine = cardinality(host_fine);

  // La cardinalité additionne les largeurs des intervalles :
  // doublage en X et duplication en Y => facteur 4.
  EXPECT_EQ(card_fine, 4 * card_coarse);
}

TEST(CSRAmrRefineProjectTest, Project1DExamplesOnSingleRow) {
  // level1 [0,1) -> level0 [0,1)
  {
    IntervalSet2DHost fine_host =
        make_host_csr({
            {0, {Interval{0, 1}}},
        });

    auto fine = build_device_from_host(fine_host);
    CsrSetAlgebraContext ctx;
    IntervalSet2DDevice coarse;
    project_level_down_device(fine, coarse, ctx);
    auto coarse_host = build_host_from_device(coarse);

    IntervalSet2DHost expected =
        make_host_csr({
            {0, {Interval{0, 1}}},
        });

    expect_equal_csr(coarse_host, expected);
  }

  // level1 [0,3) -> level0 [0,2)
  {
    IntervalSet2DHost fine_host =
        make_host_csr({
            {0, {Interval{0, 3}}},
        });

    auto fine = build_device_from_host(fine_host);
    CsrSetAlgebraContext ctx;
    IntervalSet2DDevice coarse;
    project_level_down_device(fine, coarse, ctx);
    auto coarse_host = build_host_from_device(coarse);

    IntervalSet2DHost expected =
        make_host_csr({
            {0, {Interval{0, 2}}},
        });

    expect_equal_csr(coarse_host, expected);
  }
}

TEST(CSRAmrRefineProjectTest, ProjectMergesNeighbourRows) {
  // Deux lignes fines y = 0 et y = 1, même intervalle [0,1).
  IntervalSet2DHost fine_host =
      make_host_csr({
          {0, {Interval{0, 1}}},
          {1, {Interval{0, 1}}},
      });

  auto fine = build_device_from_host(fine_host);
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice coarse;
  project_level_down_device(fine, coarse, ctx);
  auto coarse_host = build_host_from_device(coarse);

  IntervalSet2DHost expected =
      make_host_csr({
          {0, {Interval{0, 1}}},
      });

  expect_equal_csr(coarse_host, expected);
}

TEST(CSRAmrRefineProjectTest, ProjectIdempotent) {
  // Multiple fine level rows that project to the same coarse rows:
  // (0,1) -> 0, (2,3) -> 1, (4,5) -> 2.
  IntervalSet2DHost fine_host =
      make_host_csr({
          {0, {Interval{0, 1}}},
          {1, {Interval{0, 1}}},
          {2, {Interval{2, 3}}},
          {3, {Interval{2, 3}}},
          {4, {Interval{4, 5}}},
          {5, {Interval{4, 5}}},
      });

  auto fine = build_device_from_host(fine_host);

  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice coarse;
  project_level_down_device(fine, coarse, ctx);

  auto host_coarse = build_host_from_device(coarse);

  // Verify the expected projection result
  IntervalSet2DHost expected_result =
      make_host_csr({
          {0, {Interval{0, 1}}},
          {1, {Interval{1, 2}}},
          {2, {Interval{2, 3}}},
      });

  expect_equal_csr(host_coarse, expected_result);
}
