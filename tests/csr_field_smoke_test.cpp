#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_field.hpp>
#include <subsetix/vtk_export.hpp>

namespace {

using namespace subsetix::csr;

IntervalField2DHost<float> build_scaled_field_roundtrip() {
  IntervalField2DHost<float> host_field;
  host_field.append_interval(0, 0, std::vector<float>{1.f, 2.f, 3.f});
  host_field.append_interval(2, 5, std::vector<float>{4.f, 5.f});

  auto dev_field = build_device_field_from_host(host_field);

  using ExecSpace = Kokkos::DefaultExecutionSpace;
  Kokkos::parallel_for(
      "subsetix_csr_field_scale_values",
      Kokkos::RangePolicy<ExecSpace>(0, dev_field.size()),
      KOKKOS_LAMBDA(const std::size_t i) {
        dev_field.values(i) *= 2.0f;
      });
  ExecSpace().fence();

  auto roundtrip = build_host_field_from_device(dev_field);

  return roundtrip;
}

} // namespace

TEST(CSRFieldSmokeTest, HostDeviceRoundtripAndScale) {
  using namespace subsetix::csr;

  IntervalField2DHost<float> host_field;
  host_field.append_interval(0, 0, std::vector<float>{1.f, 2.f, 3.f});
  host_field.append_interval(2, 5, std::vector<float>{4.f, 5.f});

  ASSERT_EQ(host_field.num_rows(), 2u);
  ASSERT_EQ(host_field.num_intervals(), 2u);
  ASSERT_EQ(host_field.value_count(), 5u);

  auto roundtrip = build_scaled_field_roundtrip();

  ASSERT_EQ(roundtrip.num_rows(), host_field.num_rows());
  ASSERT_EQ(roundtrip.num_intervals(), host_field.num_intervals());
  ASSERT_EQ(roundtrip.value_count(), host_field.value_count());

  for (std::size_t i = 0; i < host_field.value_count(); ++i) {
    const float expected = host_field.values[i] * 2.0f;
    EXPECT_EQ(roundtrip.values[i], expected);
  }
}

TEST(CSRFieldSmokeTest, VtkExportFromBoxField) {
  using namespace subsetix::csr;

  Box2D box;
  box.x_min = 0;
  box.x_max = 4;
  box.y_min = 0;
  box.y_max = 2;

  auto geom_dev = make_box_device(box);
  auto geom_host = build_host_from_device(geom_dev);

  auto field_host =
      make_field_like_geometry<float>(geom_host, 1.0f);

  subsetix::vtk::write_legacy_quads(field_host,
                                    "csr_field_box.vtk",
                                    "value");

  // We don't parse the file here, but the call should not throw or crash.
  SUCCEED();
}
