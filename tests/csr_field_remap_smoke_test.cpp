#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <subsetix/csr_field.hpp>
#include <subsetix/csr_ops/field_remap.hpp>

using namespace subsetix::csr;

TEST(CsrFieldRemapTest, RemapFullOverlap) {
  // Source and destination have identical geometry
  IntervalField2DHost<double> host_src;
  host_src.append_interval(0, 0, {1.0, 2.0, 3.0});

  IntervalField2DHost<double> host_dst;
  host_dst.append_interval(0, 0, {0.0, 0.0, 0.0});

  auto dev_src = build_device_field_from_host(host_src);
  auto dev_dst = build_device_field_from_host(host_dst);

  remap_field_device(dev_dst, dev_src, -999.0);

  auto result = build_host_field_from_device(dev_dst);

  ASSERT_EQ(result.values.size(), 3);
  EXPECT_DOUBLE_EQ(result.values[0], 1.0);
  EXPECT_DOUBLE_EQ(result.values[1], 2.0);
  EXPECT_DOUBLE_EQ(result.values[2], 3.0);
}

TEST(CsrFieldRemapTest, RemapPartialOverlap) {
  // Source: [0, 3) with values {1, 2, 3}
  IntervalField2DHost<double> host_src;
  host_src.append_interval(0, 0, {1.0, 2.0, 3.0});

  // Destination: [1, 5) with values {0, 0, 0, 0}
  IntervalField2DHost<double> host_dst;
  host_dst.append_interval(0, 1, {0.0, 0.0, 0.0, 0.0});

  auto dev_src = build_device_field_from_host(host_src);
  auto dev_dst = build_device_field_from_host(host_dst);

  remap_field_device(dev_dst, dev_src, -999.0);

  auto result = build_host_field_from_device(dev_dst);

  ASSERT_EQ(result.values.size(), 4);
  // dst[1] = src[1] = 2.0
  // dst[2] = src[2] = 3.0
  // dst[3] = default = -999.0
  // dst[4] = default = -999.0
  EXPECT_DOUBLE_EQ(result.values[0], 2.0);
  EXPECT_DOUBLE_EQ(result.values[1], 3.0);
  EXPECT_DOUBLE_EQ(result.values[2], -999.0);
  EXPECT_DOUBLE_EQ(result.values[3], -999.0);
}

TEST(CsrFieldRemapTest, RemapNoOverlap) {
  // Source: row 0, [0, 3)
  IntervalField2DHost<double> host_src;
  host_src.append_interval(0, 0, {1.0, 2.0, 3.0});

  // Destination: row 1, [0, 3) (different row)
  IntervalField2DHost<double> host_dst;
  host_dst.append_interval(1, 0, {0.0, 0.0, 0.0});

  auto dev_src = build_device_field_from_host(host_src);
  auto dev_dst = build_device_field_from_host(host_dst);

  remap_field_device(dev_dst, dev_src, -1.0);

  auto result = build_host_field_from_device(dev_dst);

  ASSERT_EQ(result.values.size(), 3);
  // No overlap: all should be default
  EXPECT_DOUBLE_EQ(result.values[0], -1.0);
  EXPECT_DOUBLE_EQ(result.values[1], -1.0);
  EXPECT_DOUBLE_EQ(result.values[2], -1.0);
}

TEST(CsrFieldRemapTest, RemapMultipleIntervals) {
  // Source: two intervals [0, 2) and [5, 7) with values
  IntervalField2DHost<double> host_src;
  host_src.append_interval(0, 0, {10.0, 20.0});
  host_src.append_interval(0, 5, {50.0, 60.0});

  // Destination: one large interval [0, 8)
  IntervalField2DHost<double> host_dst;
  host_dst.append_interval(0, 0, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

  auto dev_src = build_device_field_from_host(host_src);
  auto dev_dst = build_device_field_from_host(host_dst);

  remap_field_device(dev_dst, dev_src, 0.0);

  auto result = build_host_field_from_device(dev_dst);

  ASSERT_EQ(result.values.size(), 8);
  EXPECT_DOUBLE_EQ(result.values[0], 10.0);  // from src
  EXPECT_DOUBLE_EQ(result.values[1], 20.0);  // from src
  EXPECT_DOUBLE_EQ(result.values[2], 0.0);   // default
  EXPECT_DOUBLE_EQ(result.values[3], 0.0);   // default
  EXPECT_DOUBLE_EQ(result.values[4], 0.0);   // default
  EXPECT_DOUBLE_EQ(result.values[5], 50.0);  // from src
  EXPECT_DOUBLE_EQ(result.values[6], 60.0);  // from src
  EXPECT_DOUBLE_EQ(result.values[7], 0.0);   // default
}

TEST(CsrFieldRemapTest, AccumulateFullOverlap) {
  // Source and destination have identical geometry
  IntervalField2DHost<double> host_src;
  host_src.append_interval(0, 0, {1.0, 2.0, 3.0});

  IntervalField2DHost<double> host_dst;
  host_dst.append_interval(0, 0, {10.0, 20.0, 30.0});

  auto dev_src = build_device_field_from_host(host_src);
  auto dev_dst = build_device_field_from_host(host_dst);

  accumulate_field_device(dev_dst, dev_src);

  auto result = build_host_field_from_device(dev_dst);

  ASSERT_EQ(result.values.size(), 3);
  EXPECT_DOUBLE_EQ(result.values[0], 11.0);  // 10 + 1
  EXPECT_DOUBLE_EQ(result.values[1], 22.0);  // 20 + 2
  EXPECT_DOUBLE_EQ(result.values[2], 33.0);  // 30 + 3
}

TEST(CsrFieldRemapTest, AccumulatePartialOverlap) {
  // Source: [0, 3) with values {1, 2, 3}
  IntervalField2DHost<double> host_src;
  host_src.append_interval(0, 0, {1.0, 2.0, 3.0});

  // Destination: [1, 5) with values {10, 20, 30, 40}
  IntervalField2DHost<double> host_dst;
  host_dst.append_interval(0, 1, {10.0, 20.0, 30.0, 40.0});

  auto dev_src = build_device_field_from_host(host_src);
  auto dev_dst = build_device_field_from_host(host_dst);

  accumulate_field_device(dev_dst, dev_src);

  auto result = build_host_field_from_device(dev_dst);

  ASSERT_EQ(result.values.size(), 4);
  // dst[1] += src[1] => 10 + 2 = 12
  // dst[2] += src[2] => 20 + 3 = 23
  // dst[3] unchanged => 30
  // dst[4] unchanged => 40
  EXPECT_DOUBLE_EQ(result.values[0], 12.0);
  EXPECT_DOUBLE_EQ(result.values[1], 23.0);
  EXPECT_DOUBLE_EQ(result.values[2], 30.0);
  EXPECT_DOUBLE_EQ(result.values[3], 40.0);
}

TEST(CsrFieldRemapTest, AccumulateNoOverlap) {
  // Source: row 0
  IntervalField2DHost<double> host_src;
  host_src.append_interval(0, 0, {1.0, 2.0, 3.0});

  // Destination: row 1 (different row)
  IntervalField2DHost<double> host_dst;
  host_dst.append_interval(1, 0, {10.0, 20.0, 30.0});

  auto dev_src = build_device_field_from_host(host_src);
  auto dev_dst = build_device_field_from_host(host_dst);

  accumulate_field_device(dev_dst, dev_src);

  auto result = build_host_field_from_device(dev_dst);

  ASSERT_EQ(result.values.size(), 3);
  // No overlap: dst unchanged
  EXPECT_DOUBLE_EQ(result.values[0], 10.0);
  EXPECT_DOUBLE_EQ(result.values[1], 20.0);
  EXPECT_DOUBLE_EQ(result.values[2], 30.0);
}

