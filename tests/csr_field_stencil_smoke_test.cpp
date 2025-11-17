#include <vector>

#include <gtest/gtest.h>

#include <subsetix/csr_field.hpp>
#include <subsetix/csr_field_ops.hpp>
#include <subsetix/csr_interval_set.hpp>

using namespace subsetix::csr;

namespace {

IntervalField2DHost<double> make_stencil_input_field() {
  IntervalField2DHost<double> host;
  for (Coord y = 0; y < 3; ++y) {
    std::vector<double> row_values;
    for (Coord x = 0; x < 4; ++x) {
      row_values.push_back(static_cast<double>(x + 10 * y));
    }
    host.append_interval(y, 0, row_values);
  }
  return host;
}

IntervalField2DHost<double> make_zero_field_like(
    const IntervalField2DHost<double>& ref) {
  IntervalField2DHost<double> host;
  for (std::size_t i = 0; i < ref.row_keys.size(); ++i) {
    const Coord y = ref.row_keys[i].y;
    std::vector<double> zeros;
    const std::size_t begin = ref.row_ptr[i];
    const std::size_t end = ref.row_ptr[i + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = ref.intervals[k];
      zeros.assign(static_cast<std::size_t>(fi.end - fi.begin), 0.0);
      host.append_interval(y, fi.begin, zeros);
      zeros.clear();
    }
  }
  return host;
}

IntervalSet2DHost make_interior_mask() {
  IntervalSet2DHost mask;
  mask.row_keys = {RowKey2D{1}};
  mask.row_ptr = {0, 1};
  mask.intervals = {Interval{1, 3}};
  return mask;
}

struct FivePointAverage {
  KOKKOS_INLINE_FUNCTION
  double operator()(Coord x,
                    Coord y,
                    std::size_t linear_index,
                    int interval_index,
                    const detail::FieldStencilContext<double>& ctx) const {
    const double center = ctx.center(linear_index);
    const double east = ctx.right(linear_index);
    const double west = ctx.left(linear_index);
    const double north = ctx.north(x, interval_index);
    const double south = ctx.south(x, interval_index);
    return (center + east + west + north + south) / 5.0;
  }
};

} // namespace

TEST(CSRFieldStencilSmokeTest, FivePointAverageOnInterior) {
  auto input_host = make_stencil_input_field();
  auto output_host = make_zero_field_like(input_host);
  auto mask_host = make_interior_mask();

  auto input_dev = build_device_field_from_host(input_host);
  auto output_dev = build_device_field_from_host(output_host);
  auto mask_dev = build_device_from_host(mask_host);

  apply_stencil_on_set_device(output_dev, input_dev,
                              mask_dev, FivePointAverage{});

  auto result = build_host_field_from_device(output_dev);

  // Only two cells should be updated (y=1, x=1 and 2)
  const double expected_cell_1 =
      (11.0 + 12.0 + 10.0 + 1.0 + 21.0) / 5.0;
  const double expected_cell_2 =
      (12.0 + 13.0 + 11.0 + 2.0 + 22.0) / 5.0;

  EXPECT_DOUBLE_EQ(result.values[5], expected_cell_1);
  EXPECT_DOUBLE_EQ(result.values[6], expected_cell_2);

  // Values outside the mask remain zero.
  EXPECT_DOUBLE_EQ(result.values[0], 0.0);
  EXPECT_DOUBLE_EQ(result.values[1], 0.0);
  EXPECT_DOUBLE_EQ(result.values[2], 0.0);
  EXPECT_DOUBLE_EQ(result.values[3], 0.0);
}
