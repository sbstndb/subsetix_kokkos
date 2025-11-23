#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <subsetix/field/csr_field.hpp>
#include <subsetix/field/csr_field_ops.hpp>
#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>
#include <subsetix/csr_ops/field_stencil.hpp>

using namespace subsetix::csr;

namespace {

IntervalField2DHost<double> make_split_field() {
  IntervalField2DHost<double> host;
  // y = 0: two disjoint intervals
  host.append_interval(0, 0, std::vector<double>{0.0, 1.0});
  host.append_interval(0, 4, std::vector<double>{4.0, 5.0});
  // y = 1: single interval (interior mask will live here)
  host.append_interval(
      1, 0,
      std::vector<double>{10.0, 11.0, 12.0, 13.0, 14.0, 15.0});
  // y = 2: two disjoint intervals, shifted values
  host.append_interval(2, 0, std::vector<double>{20.0, 21.0});
  host.append_interval(2, 4, std::vector<double>{24.0, 25.0});
  return host;
}

IntervalField2DHost<double> make_multi_interval_field() {
  IntervalField2DHost<double> host;
  // y = 0: single wide interval
  host.append_interval(
      0, 0, std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  // y = 1: two disjoint intervals
  host.append_interval(1, 0, std::vector<double>{10, 11, 12});
  host.append_interval(1, 5, std::vector<double>{15, 16, 17});
  // y = 2: two disjoint intervals covering the same x-spans
  host.append_interval(2, 0, std::vector<double>{20, 21, 22, 23});
  host.append_interval(2, 5, std::vector<double>{25, 26, 27, 28});
  return host;
}

IntervalSet2DHost make_multi_interval_mask() {
  IntervalSet2DHost mask;
  mask.row_keys = {RowKey2D{1}};
  mask.row_ptr = {0, 2};
  mask.intervals = {Interval{1, 2}, Interval{6, 7}}; // interior points of each interval
  return mask;
}

IntervalField2DHost<double> make_zero_like(
    const IntervalField2DHost<double>& ref) {
  IntervalField2DHost<double> host;
  for (std::size_t i = 0; i < ref.row_keys.size(); ++i) {
    const Coord y = ref.row_keys[i].y;
    const std::size_t begin = ref.row_ptr[i];
    const std::size_t end = ref.row_ptr[i + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto iv = ref.intervals[k];
      host.append_interval(y, iv.begin,
                           std::vector<double>(static_cast<std::size_t>(iv.end - iv.begin),
                                               0.0));
    }
  }
  return host;
}

IntervalSet2DHost make_interior_mask_split() {
  IntervalSet2DHost mask;
  mask.row_keys = {RowKey2D{1}};
  mask.row_ptr = {0, 2};
  mask.intervals = {Interval{1, 2}, Interval{4, 5}};
  return mask;
}

double host_value_at(const IntervalField2DHost<double>& field,
                     Coord x, Coord y) {
  for (std::size_t i = 0; i < field.row_keys.size(); ++i) {
    if (field.row_keys[i].y != y) {
      continue;
    }
    const std::size_t begin = field.row_ptr[i];
    const std::size_t end = field.row_ptr[i + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto iv = field.intervals[k];
      if (x >= iv.begin && x < iv.end) {
        const std::size_t idx =
            iv.value_offset + static_cast<std::size_t>(x - iv.begin);
        return field.values[idx];
      }
    }
  }
  return std::numeric_limits<double>::quiet_NaN();
}

struct FivePointAverageCsr {
  KOKKOS_INLINE_FUNCTION
  double operator()(Coord x,
                    Coord y,
                    const CsrStencilPoint<double>& p) const {
    (void)x;
    (void)y;
    return (p.center() + p.east() + p.west() + p.north() + p.south()) / 5.0;
  }
};

} // namespace

TEST(CSRFieldStencilSubsetTest, FivePointAverageAcrossSplitRows) {
  auto input_host = make_split_field();
  auto output_host = make_zero_like(input_host);
  auto mask_host = make_interior_mask_split();

  auto input_dev = build_device_field_from_host(input_host);
  auto output_dev = build_device_field_from_host(output_host);
  auto mask_dev = build_device_from_host(mask_host);

  apply_csr_stencil_on_set_device(output_dev, input_dev,
                                  mask_dev, FivePointAverageCsr{});

  auto result = build_host_field_from_device(output_dev);

  const double expected_x1y1 =
      (11.0 + 12.0 + 10.0 + 21.0 + 1.0) / 5.0; // (center+E+W+N+S)
  const double expected_x4y1 =
      (14.0 + 15.0 + 13.0 + 24.0 + 4.0) / 5.0;

  EXPECT_DOUBLE_EQ(host_value_at(result, 1, 1), expected_x1y1);
  EXPECT_DOUBLE_EQ(host_value_at(result, 4, 1), expected_x4y1);

  // Outside the mask remains zero.
  EXPECT_DOUBLE_EQ(host_value_at(result, 0, 0), 0.0);
  EXPECT_DOUBLE_EQ(host_value_at(result, 5, 2), 0.0);
}

TEST(CSRFieldStencilSubsetTest, FivePointAverageWithVaryingIntervalCounts) {
  auto input_host = make_multi_interval_field();
  auto output_host = make_zero_like(input_host);
  auto mask_host = make_multi_interval_mask();

  auto input_dev = build_device_field_from_host(input_host);
  auto output_dev = build_device_field_from_host(output_host);
  auto mask_dev = build_device_from_host(mask_host);

  apply_csr_stencil_on_set_device(output_dev, input_dev,
                                  mask_dev, FivePointAverageCsr{});

  auto result = build_host_field_from_device(output_dev);

  // x=1, y=1 lives in interval [0,3) with neighbours existing in y=0 and y=2
  const double expected_x1y1 = (11.0 + 12.0 + 10.0 + 1.0 + 21.0) / 5.0;
  // x=6, y=1 lives in interval [5,8)
  const double expected_x6y1 = (16.0 + 17.0 + 15.0 + 6.0 + 26.0) / 5.0;

  EXPECT_DOUBLE_EQ(host_value_at(result, 1, 1), expected_x1y1);
  EXPECT_DOUBLE_EQ(host_value_at(result, 6, 1), expected_x6y1);

  // Outside the mask remains zero.
  EXPECT_DOUBLE_EQ(host_value_at(result, 0, 0), 0.0);
  EXPECT_DOUBLE_EQ(host_value_at(result, 9, 0), 0.0);
  EXPECT_DOUBLE_EQ(host_value_at(result, 2, 2), 0.0);
}
