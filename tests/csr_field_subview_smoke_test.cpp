#include <functional>
#include <vector>

#include <gtest/gtest.h>

#include <subsetix/csr_field.hpp>
#include <subsetix/csr_field_ops.hpp>
#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_ops/amr.hpp>
#include <subsetix/csr_set_ops.hpp>

using namespace subsetix::csr;

namespace {

IntervalField2DHost<int> make_test_field() {
  IntervalField2DHost<int> host;
  host.append_interval(0, 0, std::vector<int>{1, 2, 3, 4});
  host.append_interval(1, 0, std::vector<int>{5, 6, 7, 8});
  return host;
}

IntervalSet2DHost make_mask_host() {
  IntervalSet2DHost mask;
  mask.row_keys = {RowKey2D{0}, RowKey2D{1}};
  mask.row_ptr = {0, 1, 2};
  mask.intervals = {Interval{1, 3}, Interval{0, 2}};
  return mask;
}

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
    const std::size_t begin = ref.row_ptr[i];
    const std::size_t end = ref.row_ptr[i + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = ref.intervals[k];
      const std::size_t len =
          static_cast<std::size_t>(fi.end - fi.begin);
      std::vector<double> zeros(len, 0.0);
      host.append_interval(y, fi.begin, zeros);
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

IntervalSet2DHost split_geometry_intervals(const IntervalSet2DHost& base) {
  IntervalSet2DHost out;
  out.row_keys = base.row_keys;
  out.row_ptr.resize(base.row_ptr.size());
  out.intervals.clear();

  for (std::size_t row = 0; row < base.row_keys.size(); ++row) {
    const std::size_t begin = base.row_ptr[row];
    const std::size_t end = base.row_ptr[row + 1];
    out.row_ptr[row] = out.intervals.size();
    for (std::size_t k = begin; k < end; ++k) {
      const Interval iv = base.intervals[k];
      const Coord len = static_cast<Coord>(iv.end - iv.begin);
      if (len <= 1) {
        out.intervals.push_back(iv);
        continue;
      }
      const Coord mid = iv.begin + len / 2;
      out.intervals.push_back(Interval{iv.begin, mid});
      out.intervals.push_back(Interval{mid, iv.end});
    }
    out.row_ptr[row + 1] = out.intervals.size();
  }

  out.rebuild_mapping();
  return out;
}

struct FivePointAverage {
  KOKKOS_INLINE_FUNCTION
  double operator()(Coord x, Coord y, std::size_t linear_index,
                    int interval_index,
                    const subsetix::csr::detail::FieldStencilContext<double>& ctx) const {
    const double center = ctx.center(linear_index);
    const double east = ctx.right(linear_index);
    const double west = ctx.left(linear_index);
    const double north = ctx.north(x, interval_index);
    const double south = ctx.south(x, interval_index);
    return (center + east + west + north + south) / 5.0;
  }
};

template <typename T>
void fill_field_with_pattern(IntervalField2DHost<T>& field,
                             const std::function<T(Coord, Coord)>& pattern) {
  for (std::size_t row = 0; row < field.row_keys.size(); ++row) {
    const Coord y = field.row_keys[row].y;
    const std::size_t begin = field.row_ptr[row];
    const std::size_t end = field.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = field.intervals[k];
      for (Coord x = fi.begin; x < fi.end; ++x) {
        const std::size_t offset =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        field.values[offset] = pattern(x, y);
      }
    }
  }
}

template <typename T>
bool host_try_get(const IntervalField2DHost<T>& field,
                  Coord x, Coord y, T& out) {
  for (std::size_t row = 0; row < field.row_keys.size(); ++row) {
    if (field.row_keys[row].y != y) {
      continue;
    }
    const std::size_t begin = field.row_ptr[row];
    const std::size_t end = field.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = field.intervals[k];
      if (x >= fi.begin && x < fi.end) {
        const std::size_t offset =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        out = field.values[offset];
        return true;
      }
    }
  }
  return false;
}

struct LinearCoordinatePattern {
  KOKKOS_INLINE_FUNCTION
  void operator()(Coord x, Coord y,
                  Field2DDevice<int>::ValueView::reference_type value,
                  std::size_t /*idx*/) const {
    value = static_cast<int>(x + 10 * y);
  }
};

} // namespace

TEST(CsrFieldSubViewTest, FillOnSubregion) {
  auto field_host = make_test_field();
  auto mask_host = make_mask_host();

  auto field_dev = build_device_field_from_host(field_host);
  auto mask_dev = build_device_from_host(mask_host);

  auto sub = make_subview(field_dev, mask_dev, "fill_subview");
  fill_subview_device(sub, 99);

  auto updated = build_host_field_from_device(field_dev);

  EXPECT_EQ(updated.values[0], 1);
  EXPECT_EQ(updated.values[1], 99);
  EXPECT_EQ(updated.values[2], 99);
  EXPECT_EQ(updated.values[3], 4);

  EXPECT_EQ(updated.values[4], 99);
  EXPECT_EQ(updated.values[5], 99);
  EXPECT_EQ(updated.values[6], 7);
  EXPECT_EQ(updated.values[7], 8);
}

TEST(CsrFieldSubViewTest, FillOnSubregionWithSubsetCaching) {
  auto field_host = make_test_field();
  auto mask_host = make_mask_host();

  auto field_dev = build_device_field_from_host(field_host);
  auto mask_dev = build_device_from_host(mask_host);

  auto sub = make_subview(field_dev, mask_dev, "subset_fill");
  EXPECT_FALSE(sub.has_subset());

  CsrSetAlgebraContext ctx;
  fill_subview_device(sub, 77, ctx);
  EXPECT_TRUE(sub.has_subset());

  fill_subview_device(sub, 88); // reuse cached subset

  auto updated = build_host_field_from_device(field_dev);
  EXPECT_EQ(updated.values[1], 88);
  EXPECT_EQ(updated.values[2], 88);
  EXPECT_EQ(updated.values[4], 88);
  EXPECT_EQ(updated.values[5], 88);
}

TEST(CsrFieldSubViewTest, CustomLambdaOnSubregion) {
  auto field_host = make_test_field();
  auto mask_host = make_mask_host();

  auto field_dev = build_device_field_from_host(field_host);
  auto mask_dev = build_device_from_host(mask_host);

  auto sub = make_subview(field_dev, mask_dev, "lambda_subview");

  apply_on_subview_device(sub, LinearCoordinatePattern{});

  auto updated = build_host_field_from_device(field_dev);

  EXPECT_EQ(updated.values[0], 1);
  EXPECT_EQ(updated.values[1], 1 + 0 * 10);
  EXPECT_EQ(updated.values[2], 2 + 0 * 10);
  EXPECT_EQ(updated.values[3], 4);

  EXPECT_EQ(updated.values[4], 0 + 10);
  EXPECT_EQ(updated.values[5], 1 + 10);
  EXPECT_EQ(updated.values[6], 7);
  EXPECT_EQ(updated.values[7], 8);
}

TEST(CsrFieldSubViewTest, StencilOnSubregion) {
  auto input_host = make_stencil_input_field();
  auto output_host = make_zero_field_like(input_host);
  auto mask_host = make_interior_mask();

  auto input_dev = build_device_field_from_host(input_host);
  auto output_dev = build_device_field_from_host(output_host);
  auto mask_dev = build_device_from_host(mask_host);

  auto input_sub = make_subview(input_dev, mask_dev, "stencil_in");
  auto output_sub = make_subview(output_dev, mask_dev, "stencil_out");

  apply_stencil_on_subview_device(output_sub, input_sub, FivePointAverage{});

  auto result = build_host_field_from_device(output_dev);
  const double expected_cell_1 =
      (11.0 + 12.0 + 10.0 + 1.0 + 21.0) / 5.0;
  const double expected_cell_2 =
      (12.0 + 13.0 + 11.0 + 2.0 + 22.0) / 5.0;

  EXPECT_DOUBLE_EQ(result.values[5], expected_cell_1);
  EXPECT_DOUBLE_EQ(result.values[6], expected_cell_2);
}

TEST(CsrFieldSubViewTest, StencilOnSubregionWithSubset) {
  auto input_host = make_stencil_input_field();
  auto output_host = make_zero_field_like(input_host);
  auto mask_host = make_interior_mask();

  auto input_dev = build_device_field_from_host(input_host);
  auto output_dev = build_device_field_from_host(output_host);
  auto mask_dev = build_device_from_host(mask_host);

  auto input_sub = make_subview(input_dev, mask_dev, "subset_stencil_in");
  auto output_sub = make_subview(output_dev, mask_dev, "subset_stencil_out");

  CsrSetAlgebraContext ctx;
  apply_stencil_on_subview_device(output_sub, input_sub, FivePointAverage{},
                                  ctx);
  EXPECT_TRUE(output_sub.has_subset());

  auto result = build_host_field_from_device(output_dev);
  const double expected_cell_1 =
      (11.0 + 12.0 + 10.0 + 1.0 + 21.0) / 5.0;
  const double expected_cell_2 =
      (12.0 + 13.0 + 11.0 + 2.0 + 22.0) / 5.0;

  EXPECT_DOUBLE_EQ(result.values[5], expected_cell_1);
  EXPECT_DOUBLE_EQ(result.values[6], expected_cell_2);
}

TEST(CsrFieldSubViewTest, CopySubViewHandlesDifferentLayoutsWithSubset) {
  Box2D domain{0, 8, 0, 4};
  auto dense_geom_dev = make_box_device(domain);
  auto dense_geom_host = build_host_from_device(dense_geom_dev);
  auto split_geom_host = split_geometry_intervals(dense_geom_host);

  auto src_host =
      make_field_like_geometry<double>(dense_geom_host, 0.0);
  fill_field_with_pattern<double>(
      src_host, [](Coord x, Coord y) {
        return static_cast<double>(x + 5 * y);
      });
  auto dst_host =
      make_field_like_geometry<double>(split_geom_host, -1.0);

  auto src_dev = build_device_field_from_host(src_host);
  auto dst_dev = build_device_field_from_host(dst_host);

  Box2D region = domain;
  region.x_min = 1;
  region.x_max = domain.x_max - 1;
  region.y_min = 1;
  region.y_max = domain.y_max - 1;
  auto mask_dev = make_box_device(region);

  auto src_sub = make_subview(src_dev, mask_dev, "copy_src");
  auto dst_sub = make_subview(dst_dev, mask_dev, "copy_dst");

  CsrSetAlgebraContext ctx;
  copy_subview_device(dst_sub, src_sub, ctx);

  auto copied = build_host_field_from_device(dst_dev);

  double value = 0.0;
  ASSERT_TRUE(host_try_get(copied, 2, 2, value));
  EXPECT_DOUBLE_EQ(value, static_cast<double>(2 + 5 * 2));
  ASSERT_TRUE(host_try_get(copied, 5, 1, value));
  EXPECT_DOUBLE_EQ(value, static_cast<double>(5 + 5 * 1));
}

TEST(CsrFieldSubViewTest, StencilOnSubregionHandlesDifferentLayouts) {
  Box2D domain{0, 8, 0, 4};
  auto dense_geom_dev = make_box_device(domain);
  auto dense_geom_host = build_host_from_device(dense_geom_dev);
  auto split_geom_host = split_geometry_intervals(dense_geom_host);

  auto src_host =
      make_field_like_geometry<double>(split_geom_host, 0.0);
  fill_field_with_pattern<double>(
      src_host, [](Coord x, Coord y) {
        return static_cast<double>(x + 10 * y);
      });
  auto dst_host =
      make_field_like_geometry<double>(dense_geom_host, 0.0);

  auto src_dev = build_device_field_from_host(src_host);
  auto dst_dev = build_device_field_from_host(dst_host);

  Box2D region{1, 7, 1, 3};
  auto mask_dev = make_box_device(region);

  auto src_sub = make_subview(src_dev, mask_dev, "split_src");
  auto dst_sub = make_subview(dst_dev, mask_dev, "dense_dst");

  CsrSetAlgebraContext ctx;
  apply_stencil_on_subview_device(dst_sub, src_sub,
                                  FivePointAverage{}, ctx);

  auto result = build_host_field_from_device(dst_dev);
  double value = 0.0;
  ASSERT_TRUE(host_try_get(result, 3, 2, value));
  const double center = static_cast<double>(3 + 10 * 2);
  const double east = static_cast<double>(4 + 10 * 2);
  const double west = static_cast<double>(2 + 10 * 2);
  const double north = static_cast<double>(3 + 10 * 3);
  const double south = static_cast<double>(3 + 10 * 1);
  const double expected =
      (center + east + west + north + south) / 5.0;
  EXPECT_DOUBLE_EQ(value, expected);
}

TEST(CsrFieldSubViewTest, RestrictSubViewAveragesFineValues) {
  auto coarse_geom = make_box_device(Box2D{0, 2, 0, 2});
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice fine_geom;
  refine_level_up_device(coarse_geom, fine_geom, ctx);

  auto coarse_geom_host = build_host_from_device(coarse_geom);
  auto fine_geom_host = build_host_from_device(fine_geom);

  auto coarse_field_host =
      make_field_like_geometry<double>(coarse_geom_host, 0.0);
  auto fine_field_host =
      make_field_like_geometry<double>(fine_geom_host, 0.0);

  fill_field_with_pattern<double>(
      fine_field_host,
      [](Coord x, Coord y) {
        return static_cast<double>(x + 10 * y);
      });

  auto coarse_field_dev =
      build_device_field_from_host(coarse_field_host);
  auto fine_field_dev =
      build_device_field_from_host(fine_field_host);

  auto coarse_sub = make_subview(coarse_field_dev, coarse_geom, "coarse_sub");
  restrict_field_subview_device(coarse_sub, fine_field_dev);

  auto restricted = build_host_field_from_device(coarse_field_dev);

  for (std::size_t row = 0; row < restricted.row_keys.size(); ++row) {
    const Coord y = restricted.row_keys[row].y;
    const std::size_t begin = restricted.row_ptr[row];
    const std::size_t end = restricted.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = restricted.intervals[k];
      for (Coord x = fi.begin; x < fi.end; ++x) {
        const std::size_t offset =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        Coord fine_x0 = static_cast<Coord>(2 * x);
        Coord fine_y0 = static_cast<Coord>(2 * y);
        Coord fine_coords[4][2] = {
            {fine_x0, fine_y0},
            {static_cast<Coord>(fine_x0 + 1), fine_y0},
            {fine_x0, static_cast<Coord>(fine_y0 + 1)},
            {static_cast<Coord>(fine_x0 + 1),
             static_cast<Coord>(fine_y0 + 1)}};

        double sum = 0.0;
        int count = 0;
        for (int idx = 0; idx < 4; ++idx) {
          double value = 0.0;
          if (host_try_get(fine_field_host,
                           fine_coords[idx][0],
                           fine_coords[idx][1], value)) {
            sum += value;
            ++count;
          }
        }

        const double expected =
            (count > 0) ? sum / static_cast<double>(count)
                        : 0.0;
        EXPECT_DOUBLE_EQ(restricted.values[offset],
                         expected);
      }
    }
  }
}

TEST(CsrFieldSubViewTest, RestrictSubViewSubsetMatchesAverages) {
  auto coarse_geom = make_box_device(Box2D{0, 2, 0, 2});
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice fine_geom;
  refine_level_up_device(coarse_geom, fine_geom, ctx);

  auto coarse_geom_host = build_host_from_device(coarse_geom);
  auto fine_geom_host = build_host_from_device(fine_geom);

  auto coarse_field_host =
      make_field_like_geometry<double>(coarse_geom_host, 0.0);
  auto fine_field_host =
      make_field_like_geometry<double>(fine_geom_host, 0.0);

  fill_field_with_pattern<double>(
      fine_field_host,
      [](Coord x, Coord y) {
        return static_cast<double>(x + 10 * y);
      });

  auto coarse_field_dev =
      build_device_field_from_host(coarse_field_host);
  auto fine_field_dev =
      build_device_field_from_host(fine_field_host);

  auto coarse_sub = make_subview(coarse_field_dev, coarse_geom, "coarse_subset");
  restrict_field_subview_device(coarse_sub, fine_field_dev, ctx);
  EXPECT_TRUE(coarse_sub.has_subset());

  auto restricted = build_host_field_from_device(coarse_field_dev);
  for (std::size_t row = 0; row < restricted.row_keys.size(); ++row) {
    const Coord y = restricted.row_keys[row].y;
    const std::size_t begin = restricted.row_ptr[row];
    const std::size_t end = restricted.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = restricted.intervals[k];
      for (Coord x = fi.begin; x < fi.end; ++x) {
        const std::size_t offset =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        Coord fine_x0 = static_cast<Coord>(2 * x);
        Coord fine_y0 = static_cast<Coord>(2 * y);
        Coord fine_coords[4][2] = {
            {fine_x0, fine_y0},
            {static_cast<Coord>(fine_x0 + 1), fine_y0},
            {fine_x0, static_cast<Coord>(fine_y0 + 1)},
            {static_cast<Coord>(fine_x0 + 1),
             static_cast<Coord>(fine_y0 + 1)}};

        double sum = 0.0;
        int count = 0;
        for (int idx = 0; idx < 4; ++idx) {
          double value = 0.0;
          if (host_try_get(fine_field_host,
                           fine_coords[idx][0],
                           fine_coords[idx][1], value)) {
            sum += value;
            ++count;
          }
        }
        const double expected =
            (count > 0) ? sum / static_cast<double>(count)
                        : 0.0;
        EXPECT_DOUBLE_EQ(restricted.values[offset],
                         expected);
      }
    }
  }
}

TEST(CsrFieldSubViewTest, ProlongSubViewCopiesCoarseValues) {
  auto coarse_geom = make_box_device(Box2D{0, 3, 0, 2});
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice fine_geom;
  refine_level_up_device(coarse_geom, fine_geom, ctx);

  auto coarse_geom_host = build_host_from_device(coarse_geom);
  auto fine_geom_host = build_host_from_device(fine_geom);

  auto coarse_field_host =
      make_field_like_geometry<double>(coarse_geom_host, 0.0);
  auto fine_field_host =
      make_field_like_geometry<double>(fine_geom_host, 0.0);

  fill_field_with_pattern<double>(
      coarse_field_host,
      [](Coord x, Coord y) {
        return static_cast<double>(100 + 3 * x + 5 * y);
      });

  auto coarse_field_dev =
      build_device_field_from_host(coarse_field_host);
  auto fine_field_dev =
      build_device_field_from_host(fine_field_host);

  auto fine_sub = make_subview(fine_field_dev, fine_geom, "fine_sub");
  prolong_field_subview_device(fine_sub, coarse_field_dev);

  auto prolonged = build_host_field_from_device(fine_field_dev);

  for (std::size_t row = 0; row < prolonged.row_keys.size(); ++row) {
    const Coord y = prolonged.row_keys[row].y;
    const std::size_t begin = prolonged.row_ptr[row];
    const std::size_t end = prolonged.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = prolonged.intervals[k];
      for (Coord x = fi.begin; x < fi.end; ++x) {
        const std::size_t offset =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        const Coord coarse_x = detail::floor_div2(x);
        const Coord coarse_y = detail::floor_div2(y);
        const double expected =
            100.0 + 3.0 * static_cast<double>(coarse_x) +
            5.0 * static_cast<double>(coarse_y);
        EXPECT_DOUBLE_EQ(prolonged.values[offset],
                         expected);
      }
    }
  }
}

TEST(CsrFieldSubViewTest, ProlongSubViewSubsetCopiesCoarseValues) {
  auto coarse_geom = make_box_device(Box2D{0, 3, 0, 2});
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice fine_geom;
  refine_level_up_device(coarse_geom, fine_geom, ctx);

  auto coarse_geom_host = build_host_from_device(coarse_geom);
  auto fine_geom_host = build_host_from_device(fine_geom);

  auto coarse_field_host =
      make_field_like_geometry<double>(coarse_geom_host, 0.0);
  auto fine_field_host =
      make_field_like_geometry<double>(fine_geom_host, 0.0);

  fill_field_with_pattern<double>(
      coarse_field_host,
      [](Coord x, Coord y) {
        return static_cast<double>(100 + 3 * x + 5 * y);
      });

  auto coarse_field_dev =
      build_device_field_from_host(coarse_field_host);
  auto fine_field_dev =
      build_device_field_from_host(fine_field_host);

  auto fine_sub = make_subview(fine_field_dev, fine_geom, "fine_subset");
  prolong_field_subview_device(fine_sub, coarse_field_dev, ctx);
  EXPECT_TRUE(fine_sub.has_subset());

  auto prolonged = build_host_field_from_device(fine_field_dev);
  for (std::size_t row = 0; row < prolonged.row_keys.size(); ++row) {
    const Coord y = prolonged.row_keys[row].y;
    const std::size_t begin = prolonged.row_ptr[row];
    const std::size_t end = prolonged.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = prolonged.intervals[k];
      for (Coord x = fi.begin; x < fi.end; ++x) {
        const std::size_t offset =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        const Coord coarse_x = detail::floor_div2(x);
        const Coord coarse_y = detail::floor_div2(y);
        const double expected =
            100.0 + 3.0 * static_cast<double>(coarse_x) +
            5.0 * static_cast<double>(coarse_y);
        EXPECT_DOUBLE_EQ(prolonged.values[offset], expected);
      }
    }
  }
}

TEST(CsrFieldSubViewTest, ProlongPredictionSubViewMatchesLinearField) {
  auto coarse_geom = make_box_device(Box2D{0, 4, 0, 4});
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice fine_geom;
  refine_level_up_device(coarse_geom, fine_geom, ctx);

  auto coarse_geom_host = build_host_from_device(coarse_geom);
  auto fine_geom_host = build_host_from_device(fine_geom);

  auto coarse_field_host =
      make_field_like_geometry<double>(coarse_geom_host, 0.0);

  fill_field_with_pattern<double>(
      coarse_field_host,
      [](Coord x, Coord y) {
        return static_cast<double>(2 * x + 4 * y);
      });

  auto coarse_field_dev =
      build_device_field_from_host(coarse_field_host);
  auto fine_field_host =
      make_field_like_geometry<double>(fine_geom_host, 0.0);
  auto fine_field_dev =
      build_device_field_from_host(fine_field_host);

  auto fine_sub = make_subview(fine_field_dev, fine_geom, "fine_pred_sub");
  prolong_field_prediction_subview_device(fine_sub, coarse_field_dev);

  auto prolonged = build_host_field_from_device(fine_field_dev);

  for (std::size_t row = 0; row < prolonged.row_keys.size(); ++row) {
    const Coord y = prolonged.row_keys[row].y;
    if (y < 2 || y > 5) continue;

    const std::size_t begin = prolonged.row_ptr[row];
    const std::size_t end = prolonged.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = prolonged.intervals[k];
      for (Coord x = fi.begin; x < fi.end; ++x) {
        if (x < 2 || x > 5) continue;
        const std::size_t offset =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        const double expected =
            static_cast<double>(x + 2 * y) - 1.5;
        EXPECT_NEAR(prolonged.values[offset],
                    expected, 1e-10)
            << "at fine x=" << x << " y=" << y;
      }
    }
  }
}
