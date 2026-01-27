// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#ifdef SUBSETIX_ENABLE_PLAYGROUND

#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <playground/subsetix/field/csr_field.hpp>
#include <playground/subsetix/field/csr_stencil.hpp>

using namespace playground::subsetix::csr;

namespace {

using Coord = int32_t;

IntervalFieldHost<2, float> make_uniform_field_3x4() {
  IntervalFieldHost<2, float> host;
  for (Coord y = 0; y < 3; ++y) {
    std::vector<float> row_values;
    row_values.reserve(4);
    for (Coord x = 0; x < 4; ++x) {
      row_values.push_back(static_cast<float>(x + 10 * y));
    }
    host.append_interval(y, 0, row_values);
  }
  return host;
}

IntervalFieldHost<2, float> make_zero_like(const IntervalFieldHost<2, float>& ref) {
  IntervalFieldHost<2, float> host;
  for (std::size_t row = 0; row < ref.row_keys.size(); ++row) {
    const Coord y = ref.row_keys[row].y;
    const std::size_t begin = static_cast<std::size_t>(ref.row_ptr[row]);
    const std::size_t end = static_cast<std::size_t>(ref.row_ptr[row + 1]);
    for (std::size_t k = begin; k < end; ++k) {
      const auto iv = ref.intervals[k];
      host.append_interval(y, iv.begin,
                           std::vector<float>(static_cast<std::size_t>(iv.end - iv.begin),
                                              0.0f));
    }
  }
  return host;
}

CsrMeshHost<2, Coord> make_interior_mask_uniform_3x4() {
  CsrMeshHost<2, Coord> host;
  host.append_interval(1, 1, 3);
  return host;
}

IntervalFieldHost<2, float> make_split_field() {
  IntervalFieldHost<2, float> host;
  host.append_interval(0, 0, std::vector<float>{0.0f, 1.0f});
  host.append_interval(0, 4, std::vector<float>{4.0f, 5.0f});
  host.append_interval(1, 0, std::vector<float>{10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f});
  host.append_interval(2, 0, std::vector<float>{20.0f, 21.0f});
  host.append_interval(2, 4, std::vector<float>{24.0f, 25.0f});
  return host;
}

CsrMeshHost<2, Coord> make_interior_mask_split() {
  CsrMeshHost<2, Coord> host;
  host.append_interval(1, 1, 2);
  host.append_interval(1, 4, 5);
  return host;
}

IntervalFieldHost<2, float> make_multi_interval_field() {
  IntervalFieldHost<2, float> host;
  host.append_interval(0, 0,
                       std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  host.append_interval(1, 0, std::vector<float>{10, 11, 12});
  host.append_interval(1, 5, std::vector<float>{15, 16, 17});
  host.append_interval(2, 0, std::vector<float>{20, 21, 22, 23});
  host.append_interval(2, 5, std::vector<float>{25, 26, 27, 28});
  return host;
}

CsrMeshHost<2, Coord> make_multi_interval_mask() {
  CsrMeshHost<2, Coord> host;
  host.append_interval(1, 1, 2);
  host.append_interval(1, 6, 7);
  return host;
}

float host_value_at(const IntervalFieldHost<2, float>& field, Coord x, Coord y) {
  for (std::size_t r = 0; r < field.row_keys.size(); ++r) {
    if (field.row_keys[r].y != y) continue;
    const std::size_t begin = static_cast<std::size_t>(field.row_ptr[r]);
    const std::size_t end = static_cast<std::size_t>(field.row_ptr[r + 1]);
    for (std::size_t k = begin; k < end; ++k) {
      const auto iv = field.intervals[k];
      if (x >= iv.begin && x < iv.end) {
        const std::size_t idx =
            static_cast<std::size_t>(iv.value_offset) +
            static_cast<std::size_t>(x - iv.begin);
        return field.values[idx];
      }
    }
  }
  return std::numeric_limits<float>::quiet_NaN();
}

struct FivePointAverage {
  KOKKOS_INLINE_FUNCTION
  float operator()(Coord x, Coord y, const CsrStencilPoint5<float>& p) const {
    (void)x;
    (void)y;
    return (p.center() + p.east() + p.west() + p.north() + p.south()) / 5.0f;
  }
};

}  // namespace

TEST(PlaygroundFieldStencil2DTest, FivePointAverageOnInteriorUniform) {
  auto input_host = make_uniform_field_3x4();
  auto output_host = make_zero_like(input_host);
  auto mask_host = make_interior_mask_uniform_3x4();

  auto input_dev = build_device_field_from_host(input_host, "in");
  auto output_dev = build_device_field_from_host(output_host, "out");
  auto mask_dev = build_device_mesh_from_host(mask_host, "mask");

  apply_csr_stencil_5pt_on_mask_device(output_dev, input_dev, mask_dev,
                                       FivePointAverage{});

  const auto result = build_host_field_from_device(output_dev);

  EXPECT_FLOAT_EQ(host_value_at(result, 1, 1), 11.0f);
  EXPECT_FLOAT_EQ(host_value_at(result, 2, 1), 12.0f);

  EXPECT_FLOAT_EQ(host_value_at(result, 0, 0), 0.0f);
  EXPECT_FLOAT_EQ(host_value_at(result, 3, 2), 0.0f);
}

TEST(PlaygroundFieldStencil2DTest, FivePointAverageAcrossSplitRows) {
  auto input_host = make_split_field();
  auto output_host = make_zero_like(input_host);
  auto mask_host = make_interior_mask_split();

  auto input_dev = build_device_field_from_host(input_host, "in");
  auto output_dev = build_device_field_from_host(output_host, "out");
  auto mask_dev = build_device_mesh_from_host(mask_host, "mask");

  apply_csr_stencil_5pt_on_mask_device(output_dev, input_dev, mask_dev,
                                       FivePointAverage{});

  const auto result = build_host_field_from_device(output_dev);

  EXPECT_FLOAT_EQ(host_value_at(result, 1, 1), 11.0f);
  EXPECT_FLOAT_EQ(host_value_at(result, 4, 1), 14.0f);

  EXPECT_FLOAT_EQ(host_value_at(result, 0, 0), 0.0f);
  EXPECT_FLOAT_EQ(host_value_at(result, 5, 2), 0.0f);
}

TEST(PlaygroundFieldStencil2DTest, FivePointAverageWithVaryingIntervalCounts) {
  auto input_host = make_multi_interval_field();
  auto output_host = make_zero_like(input_host);
  auto mask_host = make_multi_interval_mask();

  auto input_dev = build_device_field_from_host(input_host, "in");
  auto output_dev = build_device_field_from_host(output_host, "out");
  auto mask_dev = build_device_mesh_from_host(mask_host, "mask");

  apply_csr_stencil_5pt_on_mask_device(output_dev, input_dev, mask_dev,
                                       FivePointAverage{});

  const auto result = build_host_field_from_device(output_dev);

  EXPECT_FLOAT_EQ(host_value_at(result, 1, 1), 11.0f);
  EXPECT_FLOAT_EQ(host_value_at(result, 6, 1), 16.0f);

  EXPECT_FLOAT_EQ(host_value_at(result, 0, 0), 0.0f);
  EXPECT_FLOAT_EQ(host_value_at(result, 9, 0), 0.0f);
  EXPECT_FLOAT_EQ(host_value_at(result, 2, 2), 0.0f);
}

#endif  // SUBSETIX_ENABLE_PLAYGROUND

