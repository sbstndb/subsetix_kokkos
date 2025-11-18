#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_set_ops.hpp>

#include "csr_test_utils.hpp"

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

TEST(CSRTranslationSmokeTest, TranslationByZeroIsIdentity) {
  Box2D box;
  box.x_min = 0;
  box.x_max = 16;
  box.y_min = 0;
  box.y_max = 8;

  auto A = make_box_device(box);
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice T;
  translate_x_device(A, 0, T, ctx);

  auto host_A = build_host_from_device(A);
  auto host_T = build_host_from_device(T);

  expect_equal_csr(host_A, host_T);
}

TEST(CSRTranslationSmokeTest, SimplePositiveTranslation) {
  IntervalSet2DHost host_in =
      make_host_csr({
          {0, {Interval{0, 2}, Interval{4, 5}}},
          {2, {Interval{1, 3}}},
      });

  auto dev_in = build_device_from_host(host_in);
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice dev_out;
  translate_x_device(dev_in, 3, dev_out, ctx);
  auto host_out = build_host_from_device(dev_out);

  IntervalSet2DHost expected =
      make_host_csr({
          {0, {Interval{3, 5}, Interval{7, 8}}},
          {2, {Interval{4, 6}}},
      });

  expect_equal_csr(host_out, expected);
}

TEST(CSRTranslationSmokeTest, SimplePositiveTranslationY) {
  IntervalSet2DHost host_in =
      make_host_csr({
          {0, {Interval{0, 2}, Interval{4, 5}}},
          {3, {Interval{1, 3}}},
      });

  auto dev_in = build_device_from_host(host_in);
  const Coord dy = 2;
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice dev_out;
  translate_y_device(dev_in, dy, dev_out, ctx);
  auto host_out = build_host_from_device(dev_out);

  IntervalSet2DHost expected =
      make_host_csr({
          {0 + dy, {Interval{0, 2}, Interval{4, 5}}},
          {3 + dy, {Interval{1, 3}}},
      });

  expect_equal_csr(host_out, expected);
}

TEST(CSRTranslationSmokeTest, CardinalityInvariantUnderTranslation) {
  Box2D domain;
  domain.x_min = 0;
  domain.x_max = 64;
  domain.y_min = 0;
  domain.y_max = 32;

  // Construire un set un peu riche : union d'un disque et d'un checkerboard.
  Disk2D disk;
  disk.cx = 20;
  disk.cy = 16;
  disk.radius = 8;

  auto A = make_disk_device(disk);

  Domain2D dom;
  dom.x_min = domain.x_min;
  dom.x_max = domain.x_max;
  dom.y_min = domain.y_min;
  dom.y_max = domain.y_max;

  auto B = make_checkerboard_device(dom, 4);

  auto U = run_union(A, B);
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice T;
  translate_x_device(U, -5, T, ctx);

  auto host_U = build_host_from_device(U);
  auto host_T = build_host_from_device(T);

  const std::size_t card_U = cardinality(host_U);
  const std::size_t card_T = cardinality(host_T);

  EXPECT_EQ(card_U, card_T);
}

TEST(CSRTranslationSmokeTest, CardinalityInvariantUnderTranslationY) {
  Box2D domain;
  domain.x_min = 0;
  domain.x_max = 64;
  domain.y_min = 0;
  domain.y_max = 32;

  Disk2D disk;
  disk.cx = 20;
  disk.cy = 16;
  disk.radius = 8;

  auto A = make_disk_device(disk);

  Domain2D dom;
  dom.x_min = domain.x_min;
  dom.x_max = domain.x_max;
  dom.y_min = domain.y_min;
  dom.y_max = domain.y_max;

  auto B = make_checkerboard_device(dom, 4);

  auto U = run_union(A, B);
  const Coord dy = -3;
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice T;
  translate_y_device(U, dy, T, ctx);

  auto host_U = build_host_from_device(U);
  auto host_T = build_host_from_device(T);

  const std::size_t card_U = cardinality(host_U);
  const std::size_t card_T = cardinality(host_T);

  EXPECT_EQ(card_U, card_T);
}

TEST(CSRTranslationSmokeTest, UnionCommutesWithTranslation) {
  Box2D domain;
  domain.x_min = 0;
  domain.x_max = 64;
  domain.y_min = 0;
  domain.y_max = 32;

  // A = disque, B = checkerboard sur le même domaine.
  Disk2D disk;
  disk.cx = 24;
  disk.cy = 16;
  disk.radius = 9;

  auto A = make_disk_device(disk);

  Domain2D dom;
  dom.x_min = domain.x_min;
  dom.x_max = domain.x_max;
  dom.y_min = domain.y_min;
  dom.y_max = domain.y_max;

  auto B = make_checkerboard_device(dom, 4);

  const Coord dx = 5;

  // (A ∪ B) translaté.
  auto U = run_union(A, B);
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice U_shift, A_shift, B_shift;
  translate_x_device(U, dx, U_shift, ctx);

  // A et B translatés puis union.
  translate_x_device(A, dx, A_shift, ctx);
  translate_x_device(B, dx, B_shift, ctx);
  auto U_shift_alt = run_union(A_shift, B_shift);

  auto host_U_shift = build_host_from_device(U_shift);
  auto host_U_shift_alt = build_host_from_device(U_shift_alt);

  expect_equal_csr(host_U_shift, host_U_shift_alt);
}

TEST(CSRTranslationSmokeTest, UnionCommutesWithTranslationY) {
  Box2D domain;
  domain.x_min = 0;
  domain.x_max = 64;
  domain.y_min = 0;
  domain.y_max = 32;

  Disk2D disk;
  disk.cx = 24;
  disk.cy = 16;
  disk.radius = 9;

  auto A = make_disk_device(disk);

  Domain2D dom;
  dom.x_min = domain.x_min;
  dom.x_max = domain.x_max;
  dom.y_min = domain.y_min;
  dom.y_max = domain.y_max;

  auto B = make_checkerboard_device(dom, 4);

  const Coord dy = 5;

  auto U = run_union(A, B);
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice U_shift, A_shift, B_shift;
  translate_y_device(U, dy, U_shift, ctx);

  translate_y_device(A, dy, A_shift, ctx);
  translate_y_device(B, dy, B_shift, ctx);
  auto U_shift_alt = run_union(A_shift, B_shift);

  auto host_U_shift = build_host_from_device(U_shift);
  auto host_U_shift_alt = build_host_from_device(U_shift_alt);

  expect_equal_csr(host_U_shift, host_U_shift_alt);
}
