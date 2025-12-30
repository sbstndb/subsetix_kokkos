#include <gtest/gtest.h>
#include <vector>
#include <Kokkos_Core.hpp>

#include <subsetix/field/csr_field.hpp>
#include <subsetix/field/csr_field_ops.hpp>
#include <subsetix/csr_ops/threshold.hpp>
#include "csr_test_utils.hpp"

namespace {

using namespace subsetix::csr;

TEST(CSRThresholdSmokeTest, BasicThresholding) {
  // Construct a field with values:
  // Row 0: [0, 5) values: 0, 2, 3, 0, 4 -> (epsilon=1) -> [1,3), [4,5)
  // Row 2: [10, 15) values: 5, 5, 5, 1, 1 -> (epsilon=2) -> [10,13)
  
  IntervalField2DHost<float> host_field;
  host_field.append_interval(0, 0, {0.f, 2.f, 3.f, 0.f, 4.f});
  host_field.append_interval(2, 10, {5.f, 5.f, 5.f, 1.f, 1.f});

  auto dev_field = build_device_field_from_host(host_field);
  
  // Threshold with epsilon = 1.0
  // Row 0: values > 1.0 are indices 1, 2 (values 2,3) and 4 (value 4).
  //        Intervals: [0+1, 0+3) = [1, 3), [0+4, 0+5) = [4, 5)
  // Row 2: values > 1.0 is 5,5,5 (indices 0,1,2).
  //        Intervals: [10, 13)
  
  auto result_set_dev = threshold_field(dev_field, 1.0);
  auto result_set_host = to<HostMemorySpace>(result_set_dev);

  subsetix::csr_test::expect_equal_csr(
      result_set_host,
      subsetix::csr_test::make_host_csr({
          {0, {{1, 3}, {4, 5}}},
          {2, {{10, 13}}}
      }));
}

TEST(CSRThresholdSmokeTest, EmptyResult) {
  IntervalField2DHost<float> host_field;
  host_field.append_interval(0, 0, {1.f, 2.f, 3.f});
  
  auto dev_field = build_device_field_from_host(host_field);
  
  // Epsilon higher than any value
  auto result_set_dev = threshold_field(dev_field, 10.0);
  auto result_set_host = to<HostMemorySpace>(result_set_dev);

  // Expect same rows but empty intervals (or no intervals at all if implementation optimizes empty rows?
  // The implementation copies row keys, so rows exist but row_ptr will indicate empty ranges).
  // make_host_csr with empty vector makes empty rows, but here we expect rows to exist.

  ASSERT_EQ(result_set_host.num_rows, 1u);
  ASSERT_EQ(result_set_host.num_intervals, 0u);
  ASSERT_EQ(result_set_host.row_keys(0).y, 0);
}

TEST(CSRThresholdSmokeTest, FullResult) {
  IntervalField2DHost<float> host_field;
  host_field.append_interval(0, 0, {1.f, 2.f, 3.f});
  
  auto dev_field = build_device_field_from_host(host_field);
  
  // Epsilon 0, all values > 0
  auto result_set_dev = threshold_field(dev_field, 0.0);
  auto result_set_host = to<HostMemorySpace>(result_set_dev);

  subsetix::csr_test::expect_equal_csr(
      result_set_host,
      subsetix::csr_test::make_host_csr({
          {0, {{0, 3}}}
      }));
}

TEST(CSRThresholdSmokeTest, NegativeValues) {
  IntervalField2DHost<float> host_field;
  // Values: -5, -2, 0, 2, 5
  host_field.append_interval(0, 0, {-5.f, -2.f, 0.f, 2.f, 5.f});
  
  auto dev_field = build_device_field_from_host(host_field);
  
  // Epsilon 3. Should pick |-5| and |5|.
  auto result_set_dev = threshold_field(dev_field, 3.0);
  auto result_set_host = to<HostMemorySpace>(result_set_dev);

  subsetix::csr_test::expect_equal_csr(
      result_set_host,
      subsetix::csr_test::make_host_csr({
          {0, {{0, 1}, {4, 5}}}
      }));
}

TEST(CSRThresholdSmokeTest, GapsHandling) {
  IntervalField2DHost<float> host_field;
  // Row 0: interval [0,2) vals {5,5}, interval [3,5) vals {5,5}. Gap at 2.
  host_field.append_interval(0, 0, {5.f, 5.f});
  host_field.append_interval(0, 3, {5.f, 5.f});
  
  auto dev_field = build_device_field_from_host(host_field);
  
  // Epsilon 1. All values pass. But gap at 2 should break the interval.
  auto result_set_dev = threshold_field(dev_field, 1.0);
  auto result_set_host = to<HostMemorySpace>(result_set_dev);

  subsetix::csr_test::expect_equal_csr(
      result_set_host,
      subsetix::csr_test::make_host_csr({
          {0, {{0, 2}, {3, 5}}}
      }));
}

TEST(CSRThresholdSmokeTest, MergeAcrossFieldIntervals) {
  IntervalField2DHost<float> host_field;
  // Row 0: interval [0,2) vals {5,5}, interval [2,4) vals {5,5}. Contiguous at 2.
  host_field.append_interval(0, 0, {5.f, 5.f});
  host_field.append_interval(0, 2, {5.f, 5.f});
  
  auto dev_field = build_device_field_from_host(host_field);
  
  // Epsilon 1. All values pass. They should merge into [0, 4).
  auto result_set_dev = threshold_field(dev_field, 1.0);
  auto result_set_host = to<HostMemorySpace>(result_set_dev);

  subsetix::csr_test::expect_equal_csr(
      result_set_host,
      subsetix::csr_test::make_host_csr({
          {0, {{0, 4}}}
      }));
}

} // namespace

