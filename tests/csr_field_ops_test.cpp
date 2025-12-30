#include <vector>

#include <gtest/gtest.h>

#include <subsetix/field/csr_field.hpp>
#include <subsetix/field/csr_field_ops.hpp>
#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>

using namespace subsetix::csr;

namespace {

IntervalField2DHost<int> make_test_field() {
  IntervalField2DHost<int> host;
  host.append_interval(0, 0, std::vector<int>{1, 2, 3, 4});
  host.append_interval(1, 0, std::vector<int>{5, 6, 7, 8});
  return host;
}

IntervalField2DHost<int> make_source_field() {
  IntervalField2DHost<int> host;
  host.append_interval(0, 0, std::vector<int>{10, 11, 12, 13});
  host.append_interval(1, 0, std::vector<int>{20, 21, 22, 23});
  return host;
}

IntervalSet2DHost make_mask_host() {
  return make_interval_set_host(
      {{0}, {1}},           // row_keys
      {0, 1, 2},            // row_ptr
      {{1, 3}, {0, 2}}      // intervals
  );
}

void apply_custom_pattern(Field2DDevice<int>& field,
                          const IntervalSet2DDevice& mask) {
  apply_on_set_device(
      field, mask,
      KOKKOS_LAMBDA(
          Coord x, Coord y,
          Field2DDevice<int>::ValueView::reference_type value,
          std::size_t /*idx*/) {
        value = static_cast<int>(x + 10 * y);
      });
}

} // namespace

TEST(CSRFieldOpsSmokeTest, FillOnMask) {
  auto field_host = make_test_field();
  auto mask_host = make_mask_host();

  auto field_dev = build_device_field_from_host(field_host);
  auto mask_dev = to<DeviceMemorySpace>(mask_host);

  fill_on_set_device(field_dev, mask_dev, 99);

  auto updated = build_host_field_from_device(field_dev);

  ASSERT_EQ(updated.values.size(), field_host.values.size());

  // row 0: indices 0..3
  EXPECT_EQ(updated.values[0], 1);
  EXPECT_EQ(updated.values[1], 99);
  EXPECT_EQ(updated.values[2], 99);
  EXPECT_EQ(updated.values[3], 4);

  // row 1: indices 4..7
  EXPECT_EQ(updated.values[4], 99);
  EXPECT_EQ(updated.values[5], 99);
  EXPECT_EQ(updated.values[6], 7);
  EXPECT_EQ(updated.values[7], 8);
}

TEST(CSRFieldOpsSmokeTest, CopyOnMask) {
  auto dst_host = make_test_field();
  auto src_host = make_source_field();
  auto mask_host = make_mask_host();

  auto dst_dev = build_device_field_from_host(dst_host);
  auto src_dev = build_device_field_from_host(src_host);
  auto mask_dev = to<DeviceMemorySpace>(mask_host);

  copy_on_set_device(dst_dev, src_dev, mask_dev);

  auto updated = build_host_field_from_device(dst_dev);

  EXPECT_EQ(updated.values[0], 1);
  EXPECT_EQ(updated.values[1], 10 + 1);
  EXPECT_EQ(updated.values[2], 10 + 2);
  EXPECT_EQ(updated.values[3], 4);

  EXPECT_EQ(updated.values[4], 20);
  EXPECT_EQ(updated.values[5], 21);
  EXPECT_EQ(updated.values[6], 7);
  EXPECT_EQ(updated.values[7], 8);
}

TEST(CSRFieldOpsSmokeTest, CustomLambdaOnMask) {
  auto field_host = make_test_field();
  auto mask_host = make_mask_host();

  auto field_dev = build_device_field_from_host(field_host);
  auto mask_dev = to<DeviceMemorySpace>(mask_host);

  apply_custom_pattern(field_dev, mask_dev);

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
