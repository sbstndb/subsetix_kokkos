#include <gtest/gtest.h>

#include <subsetix/geometry.hpp>

#include "csr_test_utils.hpp"

using namespace subsetix::csr;
using namespace subsetix::csr_test;

namespace {

IntervalSet2DDevice make_disk_in_box(const Box2D& box,
                                     Coord cx,
                                     Coord cy,
                                     Coord radius) {
  Disk2D disk;
  disk.cx = cx;
  disk.cy = cy;
  disk.radius = radius;
  return make_disk_device(disk);
}

IntervalSet2DDevice make_checkerboard_in_domain(const Box2D& box,
                                                Coord cell_size) {
  Domain2D dom;
  dom.x_min = box.x_min;
  dom.x_max = box.x_max;
  dom.y_min = box.y_min;
  dom.y_max = box.y_max;
  return make_checkerboard_device(dom, cell_size);
}

IntervalSet2DDevice run_union(const IntervalSet2DDevice& lhs,
                              const IntervalSet2DDevice& rhs,
                              CsrSetAlgebraContext& ctx) {
  auto out = allocate_interval_set_device(
      lhs.num_rows + rhs.num_rows, lhs.num_intervals + rhs.num_intervals);
  set_union_device(lhs, rhs, out, ctx);
  return out;
}

IntervalSet2DDevice run_intersection(const IntervalSet2DDevice& lhs,
                                     const IntervalSet2DDevice& rhs,
                                     CsrSetAlgebraContext& ctx) {
  auto out = allocate_interval_set_device(
      lhs.num_rows + rhs.num_rows, lhs.num_intervals + rhs.num_intervals);
  set_intersection_device(lhs, rhs, out, ctx);
  return out;
}

IntervalSet2DDevice run_difference(const IntervalSet2DDevice& lhs,
                                   const IntervalSet2DDevice& rhs,
                                   CsrSetAlgebraContext& ctx) {
  auto out = allocate_interval_set_device(
      lhs.num_rows, lhs.num_intervals + rhs.num_intervals);
  set_difference_device(lhs, rhs, out, ctx);
  return out;
}

} // namespace

TEST(CSRSetAlgebraPropertiesTest, PartitionOfAByB) {
  // A = disque, B = rectangle partiellement recouvrant A.
  Box2D domain;
  domain.x_min = 0;
  domain.x_max = 64;
  domain.y_min = 0;
  domain.y_max = 32;

  auto A = make_disk_in_box(domain, 32, 16, 10);

  Box2D boxB;
  boxB.x_min = 16;
  boxB.x_max = 48;
  boxB.y_min = 8;
  boxB.y_max = 24;
  auto B = make_box_device(boxB);

  CsrSetAlgebraContext ctx;

  auto I = run_intersection(A, B, ctx);
  auto D1 = run_difference(A, B, ctx); // A \ B
  auto U = run_union(I, D1, ctx);

  auto host_A = build_host_from_device(A);
  auto host_I = build_host_from_device(I);
  auto host_D1 = build_host_from_device(D1);
  auto host_U = build_host_from_device(U);

  // A = (A ∩ B) ∪ (A \ B)
  expect_equal_csr(host_A, host_U);

  // Intersection nulle entre (A ∩ B) et (A \ B) en termes de cellules.
  auto I_cap_D1 = run_intersection(I, D1, ctx);
  auto host_I_cap_D1 = build_host_from_device(I_cap_D1);
  EXPECT_EQ(I_cap_D1.num_intervals, 0u);
  EXPECT_EQ(cardinality(host_I_cap_D1), 0u);

  // |A| = |A ∩ B| + |A \ B|
  const std::size_t card_A = cardinality(host_A);
  const std::size_t card_I = cardinality(host_I);
  const std::size_t card_D1 = cardinality(host_D1);
  EXPECT_EQ(card_A, card_I + card_D1);
}

TEST(CSRSetAlgebraPropertiesTest, InclusionExclusionCardinality) {
  Box2D domain;
  domain.x_min = 0;
  domain.x_max = 64;
  domain.y_min = 0;
  domain.y_max = 32;

  auto A = make_disk_in_box(domain, 20, 16, 10);
  auto B = make_disk_in_box(domain, 40, 16, 10);

  CsrSetAlgebraContext ctx;

  auto U = run_union(A, B, ctx);
  auto I = run_intersection(A, B, ctx);

  auto host_A = build_host_from_device(A);
  auto host_B = build_host_from_device(B);
  auto host_U = build_host_from_device(U);
  auto host_I = build_host_from_device(I);

  const std::size_t card_A = cardinality(host_A);
  const std::size_t card_B = cardinality(host_B);
  const std::size_t card_U = cardinality(host_U);
  const std::size_t card_I = cardinality(host_I);

  // Inclusion–exclusion: |A ∪ B| + |A ∩ B| = |A| + |B|
  EXPECT_EQ(card_U + card_I, card_A + card_B);
}

TEST(CSRSetAlgebraPropertiesTest, DeMorganRelativeToBoxDomain) {
  Box2D domain;
  domain.x_min = 0;
  domain.x_max = 64;
  domain.y_min = 0;
  domain.y_max = 32;

  auto D_full = make_box_device(domain);
  auto A = make_disk_in_box(domain, 20, 16, 10);
  auto B = make_checkerboard_in_domain(domain, 4);

  CsrSetAlgebraContext ctx;

  // Compléments relatifs dans D_full.
  auto Ac = run_difference(D_full, A, ctx);
  auto Bc = run_difference(D_full, B, ctx);

  // 1) D \ (A ∪ B) == (D \ A) ∩ (D \ B)
  auto A_union_B = run_union(A, B, ctx);
  auto left1 = run_difference(D_full, A_union_B, ctx);
  auto right1 = run_intersection(Ac, Bc, ctx);

  auto host_left1 = build_host_from_device(left1);
  auto host_right1 = build_host_from_device(right1);
  expect_equal_csr(host_left1, host_right1);

  // 2) D \ (A ∩ B) == (D \ A) ∪ (D \ B)
  auto A_inter_B = run_intersection(A, B, ctx);
  auto left2 = run_difference(D_full, A_inter_B, ctx);
  auto right2 = run_union(Ac, Bc, ctx);

  auto host_left2 = build_host_from_device(left2);
  auto host_right2 = build_host_from_device(right2);
  expect_equal_csr(host_left2, host_right2);
}
