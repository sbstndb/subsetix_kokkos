#include <Kokkos_Core.hpp>

#include <cstdio>

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_set_ops.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  int result = 0;
  {
    using namespace subsetix::csr;

    using ExecSpace = Kokkos::DefaultExecutionSpace;

    // -----------------------------------------------------------------------
    // Test 1: row_union_count / row_union_fill on overlapping intervals.
    // A: [0,2), [4,6)
    // B: [1,3), [5,7)
    // Union: [0,3), [4,7)
    // -----------------------------------------------------------------------
    {
      const std::size_t nA = 2;
      const std::size_t nB = 2;

      IntervalSet2DDevice::IntervalView intervals_a(
          "subsetix_csr_union_components_a", nA);
      IntervalSet2DDevice::IntervalView intervals_b(
          "subsetix_csr_union_components_b", nB);

      auto h_a =
          Kokkos::create_mirror_view(intervals_a);
      auto h_b =
          Kokkos::create_mirror_view(intervals_b);

      h_a(0) = Interval{0, 2};
      h_a(1) = Interval{4, 6};

      h_b(0) = Interval{1, 3};
      h_b(1) = Interval{5, 7};

      Kokkos::deep_copy(intervals_a, h_a);
      Kokkos::deep_copy(intervals_b, h_b);

      Kokkos::View<std::size_t, DeviceMemorySpace> d_count(
          "subsetix_csr_union_components_count");

      Kokkos::parallel_for(
          "subsetix_csr_union_components_count_overlap",
          Kokkos::RangePolicy<ExecSpace>(0, 1),
          KOKKOS_LAMBDA(const int) {
            const std::size_t count =
                detail::row_union_count(intervals_a, 0, nA,
                                        intervals_b, 0, nB);
            d_count() = count;
          });

      ExecSpace().fence();

      std::size_t count = 0;
      Kokkos::deep_copy(count, d_count);

      if (count != 2) {
        std::printf(
            "CSRUnionComponentsSmokeTest: Test 1 count failed "
            "(expected 2, got %zu)\n",
            count);
        result = 1;
      }

      if (result == 0) {
        IntervalSet2DDevice::IntervalView intervals_out(
            "subsetix_csr_union_components_out", count);

        Kokkos::parallel_for(
            "subsetix_csr_union_components_fill_overlap",
            Kokkos::RangePolicy<ExecSpace>(0, 1),
            KOKKOS_LAMBDA(const int) {
              detail::row_union_fill(
                  intervals_a, 0, nA,
                  intervals_b, 0, nB,
                  intervals_out, 0);
            });

        ExecSpace().fence();

        auto h_out = Kokkos::create_mirror_view_and_copy(
            HostMemorySpace{}, intervals_out);

        if (!(h_out(0).begin == 0 && h_out(0).end == 3 &&
              h_out(1).begin == 4 && h_out(1).end == 7)) {
          std::printf(
              "CSRUnionComponentsSmokeTest: Test 1 fill failed "
              "([%d,%d), [%d,%d))\n",
              static_cast<int>(h_out(0).begin),
              static_cast<int>(h_out(0).end),
              static_cast<int>(h_out(1).begin),
              static_cast<int>(h_out(1).end));
          result = 1;
        }
      }

      if (result != 0) {
        Kokkos::finalize();
        return result;
      }
    }

    // -----------------------------------------------------------------------
    // Test 2: row_union_count / row_union_fill on touching intervals.
    // A: [0,2)
    // B: [2,4)
    // Union should merge into [0,4) (touching).
    // -----------------------------------------------------------------------
    {
      const std::size_t nA = 1;
      const std::size_t nB = 1;

      IntervalSet2DDevice::IntervalView intervals_a(
          "subsetix_csr_union_components_touch_a", nA);
      IntervalSet2DDevice::IntervalView intervals_b(
          "subsetix_csr_union_components_touch_b", nB);

      auto h_a =
          Kokkos::create_mirror_view(intervals_a);
      auto h_b =
          Kokkos::create_mirror_view(intervals_b);

      h_a(0) = Interval{0, 2};
      h_b(0) = Interval{2, 4};

      Kokkos::deep_copy(intervals_a, h_a);
      Kokkos::deep_copy(intervals_b, h_b);

      Kokkos::View<std::size_t, DeviceMemorySpace> d_count(
          "subsetix_csr_union_components_touch_count");

      Kokkos::parallel_for(
          "subsetix_csr_union_components_count_touch",
          Kokkos::RangePolicy<ExecSpace>(0, 1),
          KOKKOS_LAMBDA(const int) {
            const std::size_t count =
                detail::row_union_count(intervals_a, 0, nA,
                                        intervals_b, 0, nB);
            d_count() = count;
          });

      ExecSpace().fence();

      std::size_t count = 0;
      Kokkos::deep_copy(count, d_count);

      if (count != 1) {
        std::printf(
            "CSRUnionComponentsSmokeTest: Test 2 count failed "
            "(expected 1, got %zu)\n",
            count);
        result = 1;
      }

      if (result == 0) {
        IntervalSet2DDevice::IntervalView intervals_out(
            "subsetix_csr_union_components_touch_out", count);

        Kokkos::parallel_for(
            "subsetix_csr_union_components_fill_touch",
            Kokkos::RangePolicy<ExecSpace>(0, 1),
            KOKKOS_LAMBDA(const int) {
              detail::row_union_fill(
                  intervals_a, 0, nA,
                  intervals_b, 0, nB,
                  intervals_out, 0);
            });

        ExecSpace().fence();

        auto h_out = Kokkos::create_mirror_view_and_copy(
            HostMemorySpace{}, intervals_out);

        if (!(h_out(0).begin == 0 && h_out(0).end == 4)) {
          std::printf(
              "CSRUnionComponentsSmokeTest: Test 2 fill failed "
              "([%d,%d))\n",
              static_cast<int>(h_out(0).begin),
              static_cast<int>(h_out(0).end));
          result = 1;
        }
      }

      if (result != 0) {
        Kokkos::finalize();
        return result;
      }
    }

    // -----------------------------------------------------------------------
    // Test 3: build_row_union_mapping on interleaved rows.
    //
    // A rows: y = {0, 2}
    // B rows: y = {1, 2, 4}
    //
    // Expected merged rows: {0, 1, 2, 4}
    // Mapping:
    //   y=0 -> ia=0, ib=-1
    //   y=1 -> ia=-1, ib=0
    //   y=2 -> ia=1, ib=1
    //   y=4 -> ia=-1, ib=2
    // -----------------------------------------------------------------------
    {
      IntervalSet2DHost hostA;
      IntervalSet2DHost hostB;

      hostA.row_keys.push_back(RowKey2D{0});
      hostA.row_keys.push_back(RowKey2D{2});
      hostA.row_ptr.push_back(0);
      hostA.row_ptr.push_back(0);
      hostA.row_ptr.push_back(0);

      hostB.row_keys.push_back(RowKey2D{1});
      hostB.row_keys.push_back(RowKey2D{2});
      hostB.row_keys.push_back(RowKey2D{4});
      hostB.row_ptr.push_back(0);
      hostB.row_ptr.push_back(0);
      hostB.row_ptr.push_back(0);
      hostB.row_ptr.push_back(0);

      auto A = build_device_from_host(hostA);
      auto B = build_device_from_host(hostB);

      detail::RowMergeResult merge =
          detail::build_row_union_mapping(A, B);

      if (merge.num_rows != 4) {
        std::printf(
            "CSRUnionComponentsSmokeTest: Test 3 num_rows failed "
            "(expected 4, got %zu)\n",
            merge.num_rows);
        result = 1;
      }

      if (result == 0) {
        auto h_rows = Kokkos::create_mirror_view_and_copy(
            HostMemorySpace{}, merge.row_keys);
        auto h_idx_a = Kokkos::create_mirror_view_and_copy(
            HostMemorySpace{}, merge.row_index_a);
        auto h_idx_b = Kokkos::create_mirror_view_and_copy(
            HostMemorySpace{}, merge.row_index_b);

        const Coord expected_y[4] = {0, 1, 2, 4};
        const int expected_ia[4] = {0, -1, 1, -1};
        const int expected_ib[4] = {-1, 0, 1, 2};

        for (std::size_t i = 0; i < 4; ++i) {
          if (h_rows(i).y != expected_y[i] ||
              h_idx_a(i) != expected_ia[i] ||
              h_idx_b(i) != expected_ib[i]) {
            std::printf(
                "CSRUnionComponentsSmokeTest: Test 3 mapping "
                "failed at i=%zu (y=%d, ia=%d, ib=%d)\n",
                i,
                static_cast<int>(h_rows(i).y),
                h_idx_a(i),
                h_idx_b(i));
            result = 1;
            break;
          }
        }
      }
    }
  }

  Kokkos::finalize();
  return result;
}

