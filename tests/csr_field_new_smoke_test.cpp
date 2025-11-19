#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <subsetix/csr_field.hpp>
#include <subsetix/csr_ops/field_algebra.hpp>
#include <subsetix/csr_ops/field_core.hpp>

using namespace subsetix::csr;

namespace {

IntervalSet2DHost make_test_geometry() {
  IntervalSet2DHost geom;
  geom.row_keys = {RowKey2D{0}, RowKey2D{1}};
  geom.row_ptr = {0, 1, 2};
  geom.intervals = {Interval{0, 3}, Interval{1, 4}};
  geom.rebuild_mapping();
  return geom;
}

std::vector<double> make_values(std::size_t count, double start) {
  std::vector<double> vals(count);
  for (std::size_t i = 0; i < count; ++i) {
    vals[i] = start + static_cast<double>(i);
  }
  return vals;
}

void assign_field(Field2DDevice<double>& field,
                  const std::vector<double>& values) {
  auto mirror = Kokkos::create_mirror_view(field.values);
  for (std::size_t i = 0; i < values.size(); ++i) {
    mirror(i) = values[i];
  }
  Kokkos::deep_copy(field.values, mirror);
}

void assign_legacy(IntervalField2DHost<double>& field,
                   const std::vector<double>& values) {
  field.values = values;
}

} // namespace

TEST(Field2DTest, FieldAddMatchesLegacy) {
  auto geom_host = make_test_geometry();
  auto geom_dev = build_device_from_host(geom_host);

  Field2DDevice<double> field_a(geom_dev, "a");
  Field2DDevice<double> field_b(geom_dev, "b");
  Field2DDevice<double> field_out(geom_dev, "out");

  auto legacy_a_host = make_field_like_geometry<double>(geom_host, 0.0);
  auto legacy_b_host = make_field_like_geometry<double>(geom_host, 0.0);
  auto legacy_out_host = make_field_like_geometry<double>(geom_host, 0.0);

  const auto values_a = make_values(geom_host.total_cells, 1.0);
  const auto values_b = make_values(geom_host.total_cells, 10.0);

  assign_field(field_a, values_a);
  assign_field(field_b, values_b);
  assign_legacy(legacy_a_host, values_a);
  assign_legacy(legacy_b_host, values_b);

  auto legacy_a_dev = build_device_field_from_host(legacy_a_host);
  auto legacy_b_dev = build_device_field_from_host(legacy_b_host);
  auto legacy_out_dev = build_device_field_from_host(legacy_out_host);

  field_add_device(field_out, field_a, field_b);
  field_add_device(legacy_out_dev, legacy_a_dev, legacy_b_dev);

  auto field_result =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, field_out.values);
  auto legacy_result_host = build_host_field_from_device(legacy_out_dev);

  ASSERT_EQ(legacy_result_host.values.size(), geom_host.total_cells);
  for (std::size_t i = 0; i < geom_host.total_cells; ++i) {
    EXPECT_DOUBLE_EQ(field_result(i), legacy_result_host.values[i]);
  }
}

TEST(Field2DTest, FillOnMask) {
  auto geom_host = make_test_geometry();
  auto geom_dev = build_device_from_host(geom_host);

  Field2DDevice<double> field(geom_dev, "mask_field");
  auto initial_values = make_values(geom_host.total_cells, 0.0);
  assign_field(field, initial_values);

  IntervalSet2DHost mask_host;
  mask_host.row_keys = {RowKey2D{0}};
  mask_host.row_ptr = {0, 1};
  mask_host.intervals = {Interval{1, 3}};
  mask_host.rebuild_mapping();
  auto mask_dev = build_device_from_host(mask_host);

  fill_on_set_device(field, mask_dev, 5.0);

  auto result =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, field.values);

  // Geometry layout: row0 [0,3), row1 [1,4)
  // Total cells = 6, ordering: (0,0)(1,0)(2,0)(1,1)(2,1)(3,1)
  std::vector<double> expected = {0.0, 5.0, 5.0, 3.0, 4.0, 5.0};

  ASSERT_EQ(expected.size(), geom_host.total_cells);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_DOUBLE_EQ(result(i), expected[i]);
  }
}

