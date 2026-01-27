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

IntervalFieldHost<3, float> make_uniform_field_3x3x3() {
  IntervalFieldHost<3, float> host;
  for (Coord y = 0; y < 3; ++y) {
    for (Coord z = 0; z < 3; ++z) {
      std::vector<float> vals;
      vals.reserve(3);
      for (Coord x = 0; x < 3; ++x) {
        vals.push_back(static_cast<float>(x + 10 * y + 100 * z));
      }
      host.append_interval(y, z, 0, vals);
    }
  }
  return host;
}

IntervalFieldHost<3, float> make_zero_like(const IntervalFieldHost<3, float>& ref) {
  IntervalFieldHost<3, float> host;
  for (std::size_t row = 0; row < ref.row_keys.size(); ++row) {
    const Coord y = ref.row_keys[row].y;
    const Coord z = ref.row_keys[row].z;
    const std::size_t begin = static_cast<std::size_t>(ref.row_ptr[row]);
    const std::size_t end = static_cast<std::size_t>(ref.row_ptr[row + 1]);
    for (std::size_t k = begin; k < end; ++k) {
      const auto iv = ref.intervals[k];
      host.append_interval(y, z, iv.begin,
                           std::vector<float>(static_cast<std::size_t>(iv.end - iv.begin),
                                              0.0f));
    }
  }
  return host;
}

CsrMeshHost<3, Coord> make_center_cell_mask_3x3x3() {
  CsrMeshHost<3, Coord> host;
  host.append_interval(1, 1, 1, 2);
  return host;
}

float host_value_at(const IntervalFieldHost<3, float>& field, Coord x, Coord y, Coord z) {
  for (std::size_t r = 0; r < field.row_keys.size(); ++r) {
    const auto key = field.row_keys[r];
    if (key.y != y || key.z != z) continue;
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

struct SevenPointAverage {
  KOKKOS_INLINE_FUNCTION
  float operator()(Coord x, Coord y, Coord z, const CsrStencilPoint7<float>& p) const {
    (void)x;
    (void)y;
    (void)z;
    return (p.center() + p.east() + p.west() +
            p.north() + p.south() + p.up() + p.down()) /
           7.0f;
  }
};

}  // namespace

TEST(PlaygroundFieldStencil3DTest, SevenPointAverageOnSingleInteriorCell) {
  auto input_host = make_uniform_field_3x3x3();
  auto output_host = make_zero_like(input_host);
  auto mask_host = make_center_cell_mask_3x3x3();

  auto input_dev = build_device_field_from_host(input_host, "in");
  auto output_dev = build_device_field_from_host(output_host, "out");
  auto mask_dev = build_device_mesh_from_host(mask_host, "mask");

  apply_csr_stencil_7pt_on_mask_device(output_dev, input_dev, mask_dev,
                                       SevenPointAverage{});

  const auto result = build_host_field_from_device(output_dev);

  EXPECT_FLOAT_EQ(host_value_at(result, 1, 1, 1), 111.0f);
  EXPECT_FLOAT_EQ(host_value_at(result, 0, 0, 0), 0.0f);
  EXPECT_FLOAT_EQ(host_value_at(result, 2, 2, 2), 0.0f);
}

#endif  // SUBSETIX_ENABLE_PLAYGROUND

