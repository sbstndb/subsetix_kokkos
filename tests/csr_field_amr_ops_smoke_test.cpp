#include <functional>

#include <gtest/gtest.h>

#include <subsetix/csr_field.hpp>
#include <subsetix/csr_field_ops.hpp>
#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_set_ops.hpp>

using namespace subsetix::csr;

namespace {

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

IntervalSet2DDevice make_box_mask(Coord x_min, Coord x_max,
                                  Coord y_min, Coord y_max) {
  Box2D box;
  box.x_min = x_min;
  box.x_max = x_max;
  box.y_min = y_min;
  box.y_max = y_max;
  return make_box_device(box);
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

} // namespace

TEST(CSRFieldAmrOpsSmokeTest, RestrictAveragesFineValues) {
  auto coarse_geom = make_box_mask(0, 2, 0, 2);
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
  auto mask_dev = coarse_geom;

  restrict_field_on_set_device(coarse_field_dev,
                               fine_field_dev, mask_dev);

  auto restricted =
      build_host_field_from_device(coarse_field_dev);

  // For coarse cell (x, y), expected average:
  // avg = (2x + 20y + 5.5)
  for (std::size_t row = 0; row < restricted.row_keys.size();
       ++row) {
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

TEST(CSRFieldAmrOpsSmokeTest, ProlongCopiesCoarseValues) {
  auto coarse_geom = make_box_mask(0, 3, 0, 2);
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
  auto mask_dev = fine_geom;

  prolong_field_on_set_device(fine_field_dev,
                              coarse_field_dev, mask_dev);

  auto prolonged =
      build_host_field_from_device(fine_field_dev);

  for (std::size_t row = 0; row < prolonged.row_keys.size();
       ++row) {
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

TEST(CSRFieldAmrOpsSmokeTest, ProlongWithPredictionLinearReconstruction) {
  // Coarse 4x4 domain
  auto coarse_geom = make_box_mask(0, 4, 0, 4);
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice fine_geom;
  refine_level_up_device(coarse_geom, fine_geom, ctx);

  auto coarse_geom_host = build_host_from_device(coarse_geom);
  auto fine_geom_host = build_host_from_device(fine_geom);

  auto coarse_field_host =
      make_field_like_geometry<double>(coarse_geom_host, 0.0);
  
  // P(x,y) = 2x + 4y
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
  
  prolong_field_prediction_device(fine_field_dev, coarse_field_dev, fine_geom);

  auto prolonged = build_host_field_from_device(fine_field_dev);

  // Check internal cells where we expect exact reconstruction
  // Coarse x in [1, 2], y in [1, 2]
  // Fine x in [2, 5], y in [2, 5] (approx)
  
  for (std::size_t row = 0; row < prolonged.row_keys.size(); ++row) {
    const Coord y = prolonged.row_keys[row].y;
    if (y < 2 || y > 5) continue;
    
    const std::size_t begin = prolonged.row_ptr[row];
    const std::size_t end = prolonged.row_ptr[row + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const auto fi = prolonged.intervals[k];
      for (Coord x = fi.begin; x < fi.end; ++x) {
        if (x < 2 || x > 5) continue;
        
        // Expected fine value for linear field: x + 2y - 1.5
        double expected = static_cast<double>(x + 2 * y) - 1.5;
        
        const std::size_t offset =
            fi.value_offset +
            static_cast<std::size_t>(x - fi.begin);
        EXPECT_NEAR(prolonged.values[offset], expected, 1e-10) 
             << "at fine x=" << x << " y=" << y;
      }
    }
  }
}

TEST(CSRFieldAmrOpsSmokeTest,
     RestrictCoordsMatchesStandardOnFullRefinement) {
  auto coarse_geom = make_box_mask(0, 3, 0, 3);
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

  auto fine_field_dev =
      build_device_field_from_host(fine_field_host);

  auto coarse_field_standard =
      build_device_field_from_host(coarse_field_host);
  auto coarse_field_coords =
      build_device_field_from_host(coarse_field_host);

  auto mask_dev = coarse_geom;

  restrict_field_on_set_device(coarse_field_standard,
                               fine_field_dev, mask_dev);
  restrict_field_on_set_coords_device(coarse_field_coords,
                                      fine_field_dev, mask_dev);

  auto restricted_standard =
      build_host_field_from_device(coarse_field_standard);
  auto restricted_coords =
      build_host_field_from_device(coarse_field_coords);

  ASSERT_EQ(restricted_standard.row_keys.size(),
            restricted_coords.row_keys.size());
  ASSERT_EQ(restricted_standard.intervals.size(),
            restricted_coords.intervals.size());
  ASSERT_EQ(restricted_standard.values.size(),
            restricted_coords.values.size());

  for (std::size_t i = 0; i < restricted_standard.values.size(); ++i) {
    EXPECT_DOUBLE_EQ(restricted_standard.values[i],
                     restricted_coords.values[i]);
  }
}

TEST(CSRFieldAmrOpsSmokeTest,
     ProlongPredictionCoordsMatchesStandardOnFullRefinement) {
  auto coarse_geom = make_box_mask(0, 4, 0, 4);
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice fine_geom;
  refine_level_up_device(coarse_geom, fine_geom, ctx);

  auto coarse_geom_host = build_host_from_device(coarse_geom);
  auto fine_geom_host = build_host_from_device(fine_geom);

  auto coarse_field_host =
      make_field_like_geometry<double>(coarse_geom_host, 0.0);

  // Linear polynomial P(x,y) = 2x + 4y
  fill_field_with_pattern<double>(
      coarse_field_host,
      [](Coord x, Coord y) {
        return static_cast<double>(2 * x + 4 * y);
      });

  auto coarse_field_dev =
      build_device_field_from_host(coarse_field_host);

  auto fine_field_host_standard =
      make_field_like_geometry<double>(fine_geom_host, 0.0);
  auto fine_field_host_coords =
      make_field_like_geometry<double>(fine_geom_host, 0.0);

  auto fine_field_standard =
      build_device_field_from_host(fine_field_host_standard);
  auto fine_field_coords =
      build_device_field_from_host(fine_field_host_coords);

  prolong_field_prediction_device(fine_field_standard,
                                  coarse_field_dev, fine_geom);
  prolong_field_prediction_coords_device(fine_field_coords,
                                         coarse_field_dev, fine_geom);

  auto prolonged_standard =
      build_host_field_from_device(fine_field_standard);
  auto prolonged_coords =
      build_host_field_from_device(fine_field_coords);

  ASSERT_EQ(prolonged_standard.row_keys.size(),
            prolonged_coords.row_keys.size());
  ASSERT_EQ(prolonged_standard.intervals.size(),
            prolonged_coords.intervals.size());
  ASSERT_EQ(prolonged_standard.values.size(),
            prolonged_coords.values.size());

  for (std::size_t i = 0; i < prolonged_standard.values.size(); ++i) {
    EXPECT_NEAR(prolonged_standard.values[i],
                prolonged_coords.values[i],
                1e-12);
  }
}
