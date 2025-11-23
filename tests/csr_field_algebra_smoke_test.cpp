#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <subsetix/field/csr_field.hpp>
#include <subsetix/field/csr_field_ops.hpp>
#include <subsetix/csr_ops/field_algebra.hpp>

using namespace subsetix::csr;

TEST(CsrFieldAlgebraTest, FieldAddBasic) {
  // Create a simple geometry: single row with one interval
  IntervalField2DHost<double> host_a;
  host_a.append_interval(0, 0, {1.0, 2.0, 3.0});

  IntervalField2DHost<double> host_b;
  host_b.append_interval(0, 0, {4.0, 5.0, 6.0});

  IntervalField2DHost<double> host_result;
  host_result.append_interval(0, 0, {0.0, 0.0, 0.0});

  auto dev_a = build_device_field_from_host(host_a);
  auto dev_b = build_device_field_from_host(host_b);
  auto dev_result = build_device_field_from_host(host_result);

  field_add_device(dev_result, dev_a, dev_b);

  auto result = build_host_field_from_device(dev_result);

  ASSERT_EQ(result.values.size(), 3);
  EXPECT_DOUBLE_EQ(result.values[0], 5.0);
  EXPECT_DOUBLE_EQ(result.values[1], 7.0);
  EXPECT_DOUBLE_EQ(result.values[2], 9.0);
}

TEST(CsrFieldAlgebraTest, FieldSubBasic) {
  IntervalField2DHost<double> host_a;
  host_a.append_interval(0, 0, {10.0, 20.0, 30.0});

  IntervalField2DHost<double> host_b;
  host_b.append_interval(0, 0, {1.0, 2.0, 3.0});

  IntervalField2DHost<double> host_result;
  host_result.append_interval(0, 0, {0.0, 0.0, 0.0});

  auto dev_a = build_device_field_from_host(host_a);
  auto dev_b = build_device_field_from_host(host_b);
  auto dev_result = build_device_field_from_host(host_result);

  field_sub_device(dev_result, dev_a, dev_b);

  auto result = build_host_field_from_device(dev_result);

  ASSERT_EQ(result.values.size(), 3);
  EXPECT_DOUBLE_EQ(result.values[0], 9.0);
  EXPECT_DOUBLE_EQ(result.values[1], 18.0);
  EXPECT_DOUBLE_EQ(result.values[2], 27.0);
}

TEST(CsrFieldAlgebraTest, FieldMulBasic) {
  IntervalField2DHost<double> host_a;
  host_a.append_interval(0, 0, {2.0, 3.0, 4.0});

  IntervalField2DHost<double> host_b;
  host_b.append_interval(0, 0, {5.0, 6.0, 7.0});

  IntervalField2DHost<double> host_result;
  host_result.append_interval(0, 0, {0.0, 0.0, 0.0});

  auto dev_a = build_device_field_from_host(host_a);
  auto dev_b = build_device_field_from_host(host_b);
  auto dev_result = build_device_field_from_host(host_result);

  field_mul_device(dev_result, dev_a, dev_b);

  auto result = build_host_field_from_device(dev_result);

  ASSERT_EQ(result.values.size(), 3);
  EXPECT_DOUBLE_EQ(result.values[0], 10.0);
  EXPECT_DOUBLE_EQ(result.values[1], 18.0);
  EXPECT_DOUBLE_EQ(result.values[2], 28.0);
}

TEST(CsrFieldAlgebraTest, FieldDivBasic) {
  IntervalField2DHost<double> host_a;
  host_a.append_interval(0, 0, {10.0, 20.0, 30.0});

  IntervalField2DHost<double> host_b;
  host_b.append_interval(0, 0, {2.0, 4.0, 5.0});

  IntervalField2DHost<double> host_result;
  host_result.append_interval(0, 0, {0.0, 0.0, 0.0});

  auto dev_a = build_device_field_from_host(host_a);
  auto dev_b = build_device_field_from_host(host_b);
  auto dev_result = build_device_field_from_host(host_result);

  field_div_device(dev_result, dev_a, dev_b);

  auto result = build_host_field_from_device(dev_result);

  ASSERT_EQ(result.values.size(), 3);
  EXPECT_DOUBLE_EQ(result.values[0], 5.0);
  EXPECT_DOUBLE_EQ(result.values[1], 5.0);
  EXPECT_DOUBLE_EQ(result.values[2], 6.0);
}

TEST(CsrFieldAlgebraTest, FieldAbsBasic) {
  IntervalField2DHost<double> host_a;
  host_a.append_interval(0, 0, {-1.0, 2.0, -3.0, 4.0});

  IntervalField2DHost<double> host_result;
  host_result.append_interval(0, 0, {0.0, 0.0, 0.0, 0.0});

  auto dev_a = build_device_field_from_host(host_a);
  auto dev_result = build_device_field_from_host(host_result);

  field_abs_device(dev_result, dev_a);

  auto result = build_host_field_from_device(dev_result);

  ASSERT_EQ(result.values.size(), 4);
  EXPECT_DOUBLE_EQ(result.values[0], 1.0);
  EXPECT_DOUBLE_EQ(result.values[1], 2.0);
  EXPECT_DOUBLE_EQ(result.values[2], 3.0);
  EXPECT_DOUBLE_EQ(result.values[3], 4.0);
}

TEST(CsrFieldAlgebraTest, FieldAxpbyBasic) {
  IntervalField2DHost<double> host_a;
  host_a.append_interval(0, 0, {1.0, 2.0, 3.0});

  IntervalField2DHost<double> host_b;
  host_b.append_interval(0, 0, {4.0, 5.0, 6.0});

  IntervalField2DHost<double> host_result;
  host_result.append_interval(0, 0, {0.0, 0.0, 0.0});

  auto dev_a = build_device_field_from_host(host_a);
  auto dev_b = build_device_field_from_host(host_b);
  auto dev_result = build_device_field_from_host(host_result);

  // result = 2.0 * a + 3.0 * b
  field_axpby_device(dev_result, 2.0, dev_a, 3.0, dev_b);

  auto result = build_host_field_from_device(dev_result);

  ASSERT_EQ(result.values.size(), 3);
  EXPECT_DOUBLE_EQ(result.values[0], 2.0 * 1.0 + 3.0 * 4.0);  // 14.0
  EXPECT_DOUBLE_EQ(result.values[1], 2.0 * 2.0 + 3.0 * 5.0);  // 19.0
  EXPECT_DOUBLE_EQ(result.values[2], 2.0 * 3.0 + 3.0 * 6.0);  // 24.0
}

TEST(CsrFieldAlgebraTest, FieldDotBasic) {
  IntervalField2DHost<double> host_a;
  host_a.append_interval(0, 0, {1.0, 2.0, 3.0});

  IntervalField2DHost<double> host_b;
  host_b.append_interval(0, 0, {4.0, 5.0, 6.0});

  auto dev_a = build_device_field_from_host(host_a);
  auto dev_b = build_device_field_from_host(host_b);

  double dot = field_dot_device(dev_a, dev_b);

  // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
  EXPECT_DOUBLE_EQ(dot, 32.0);
}

TEST(CsrFieldAlgebraTest, FieldNormL2Basic) {
  IntervalField2DHost<double> host_a;
  host_a.append_interval(0, 0, {3.0, 4.0});

  auto dev_a = build_device_field_from_host(host_a);

  double norm = field_norm_l2_device(dev_a);

  // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
  EXPECT_DOUBLE_EQ(norm, 5.0);
}

TEST(CsrFieldAlgebraTest, MultiRowField) {
  // Test with multiple rows
  IntervalField2DHost<double> host_a;
  host_a.append_interval(0, 0, {1.0, 2.0});
  host_a.append_interval(1, 0, {3.0, 4.0});

  IntervalField2DHost<double> host_b;
  host_b.append_interval(0, 0, {5.0, 6.0});
  host_b.append_interval(1, 0, {7.0, 8.0});

  IntervalField2DHost<double> host_result;
  host_result.append_interval(0, 0, {0.0, 0.0});
  host_result.append_interval(1, 0, {0.0, 0.0});

  auto dev_a = build_device_field_from_host(host_a);
  auto dev_b = build_device_field_from_host(host_b);
  auto dev_result = build_device_field_from_host(host_result);

  field_add_device(dev_result, dev_a, dev_b);

  auto result = build_host_field_from_device(dev_result);

  ASSERT_EQ(result.values.size(), 4);
  EXPECT_DOUBLE_EQ(result.values[0], 6.0);
  EXPECT_DOUBLE_EQ(result.values[1], 8.0);
  EXPECT_DOUBLE_EQ(result.values[2], 10.0);
  EXPECT_DOUBLE_EQ(result.values[3], 12.0);
}

