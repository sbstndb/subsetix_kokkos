#pragma once

#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_set_ops.hpp>

// Utilitaires de tests pour les opérateurs de base sur une ligne
// (union, intersection, différence A \ B).

namespace subsetix {
namespace csr_test {

using namespace subsetix::csr;

struct RowOpCase {
  std::vector<Interval> intervals_a;
  std::vector<Interval> intervals_b;
  std::vector<Interval> expected_union;
  std::vector<Interval> expected_intersection;
  std::vector<Interval> expected_difference_a_minus_b;
};

inline std::vector<RowOpCase> build_row_op_cases() {
  std::vector<RowOpCase> cases;

  // 0) A = {}, B = {}.
  {
    RowOpCase c;
    cases.push_back(c);
  }

  // 1) A non vide, B vide.
  {
    RowOpCase c;
    c.intervals_a.push_back(Interval{0, 2});
    c.expected_union = c.intervals_a;
    c.expected_difference_a_minus_b = c.intervals_a;
    cases.push_back(c);
  }

  // 2) A vide, B non vide.
  {
    RowOpCase c;
    c.intervals_b.push_back(Interval{0, 2});
    c.expected_union = c.intervals_b;
    // intersection et différence A\B vides.
    cases.push_back(c);
  }

  // 3) Recouvrement partiel : A = [0,3), B = [1,4).
  {
    RowOpCase c;
    c.intervals_a.push_back(Interval{0, 3});
    c.intervals_b.push_back(Interval{1, 4});
    c.expected_union.push_back(Interval{0, 4});
    c.expected_intersection.push_back(Interval{1, 3});
    c.expected_difference_a_minus_b.push_back(Interval{0, 1});
    cases.push_back(c);
  }

  // 4) B strictement inclus dans A, multiples trous.
  {
    RowOpCase c;
    c.intervals_a.push_back(Interval{0, 10});
    c.intervals_b.push_back(Interval{2, 4});
    c.intervals_b.push_back(Interval{6, 8});
    c.expected_union.push_back(Interval{0, 10});
    c.expected_intersection.push_back(Interval{2, 4});
    c.expected_intersection.push_back(Interval{6, 8});
    c.expected_difference_a_minus_b.push_back(Interval{0, 2});
    c.expected_difference_a_minus_b.push_back(Interval{4, 6});
    c.expected_difference_a_minus_b.push_back(Interval{8, 10});
    cases.push_back(c);
  }

  // 5) Plusieurs intervalles dans A, B chevauche les deux.
  //    A = [0,2), [4,6); B = [1,5)
  {
    RowOpCase c;
    c.intervals_a.push_back(Interval{0, 2});
    c.intervals_a.push_back(Interval{4, 6});
    c.intervals_b.push_back(Interval{1, 5});

    c.expected_union.push_back(Interval{0, 6});

    c.expected_intersection.push_back(Interval{1, 2});
    c.expected_intersection.push_back(Interval{4, 5});

    c.expected_difference_a_minus_b.push_back(Interval{0, 1});
    c.expected_difference_a_minus_b.push_back(Interval{5, 6});
    cases.push_back(c);
  }

  // 6) A et B disjoints.
  //    A = [0,2), [4,6); B = [10,12)
  {
    RowOpCase c;
    c.intervals_a.push_back(Interval{0, 2});
    c.intervals_a.push_back(Interval{4, 6});
    c.intervals_b.push_back(Interval{10, 12});

    c.expected_union.push_back(Interval{0, 2});
    c.expected_union.push_back(Interval{4, 6});
    c.expected_union.push_back(Interval{10, 12});

    c.expected_difference_a_minus_b = c.intervals_a;
    cases.push_back(c);
  }

  // 7) Intervalles qui se touchent : A = [0,2), B = [2,4).
  {
    RowOpCase c;
    c.intervals_a.push_back(Interval{0, 2});
    c.intervals_b.push_back(Interval{2, 4});

    c.expected_union.push_back(Interval{0, 4});
    // intersection vide
    c.expected_difference_a_minus_b.push_back(Interval{0, 2});
    cases.push_back(c);
  }

  // 8) B couvre plusieurs intervalles de A : A = [0,2), [4,6); B = [0,6).
  {
    RowOpCase c;
    c.intervals_a.push_back(Interval{0, 2});
    c.intervals_a.push_back(Interval{4, 6});
    c.intervals_b.push_back(Interval{0, 6});

    c.expected_union.push_back(Interval{0, 6});

    c.expected_intersection.push_back(Interval{0, 2});
    c.expected_intersection.push_back(Interval{4, 6});

    // A \ B vide
    cases.push_back(c);
  }

  return cases;
}

inline void run_row_union_case(const RowOpCase& c,
                               std::size_t case_id) {
  using ExecSpace = Kokkos::DefaultExecutionSpace;

  const std::size_t nA = c.intervals_a.size();
  const std::size_t nB = c.intervals_b.size();

  IntervalSet2DDevice::IntervalView intervals_a(
      "subsetix_csr_row_union_a", nA);
  IntervalSet2DDevice::IntervalView intervals_b(
      "subsetix_csr_row_union_b", nB);

  auto h_a = Kokkos::create_mirror_view(intervals_a);
  auto h_b = Kokkos::create_mirror_view(intervals_b);

  for (std::size_t i = 0; i < nA; ++i) {
    h_a(i) = c.intervals_a[i];
  }
  for (std::size_t i = 0; i < nB; ++i) {
    h_b(i) = c.intervals_b[i];
  }

  Kokkos::deep_copy(intervals_a, h_a);
  Kokkos::deep_copy(intervals_b, h_b);

  Kokkos::View<std::size_t, DeviceMemorySpace> d_count(
      "subsetix_csr_row_union_count");

  Kokkos::parallel_for(
      "subsetix_csr_row_union_count_kernel",
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
  EXPECT_EQ(count, c.expected_union.size())
      << "row_union_count mismatch in case " << case_id;

  IntervalSet2DDevice::IntervalView intervals_out(
      "subsetix_csr_row_union_out", count);

  Kokkos::parallel_for(
      "subsetix_csr_row_union_fill_kernel",
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
  ASSERT_EQ(h_out.extent(0), c.expected_union.size())
      << "row_union_fill size mismatch in case " << case_id;

  for (std::size_t i = 0; i < c.expected_union.size(); ++i) {
    const auto& expected = c.expected_union[i];
    const auto got = h_out(i);
    EXPECT_EQ(got.begin, expected.begin)
        << "row_union_fill begin mismatch in case " << case_id
        << " interval " << i;
    EXPECT_EQ(got.end, expected.end)
        << "row_union_fill end mismatch in case " << case_id
        << " interval " << i;
  }
}

inline void run_row_intersection_case(const RowOpCase& c,
                                      std::size_t case_id) {
  using ExecSpace = Kokkos::DefaultExecutionSpace;

  const std::size_t nA = c.intervals_a.size();
  const std::size_t nB = c.intervals_b.size();

  IntervalSet2DDevice::IntervalView intervals_a(
      "subsetix_csr_row_intersection_a", nA);
  IntervalSet2DDevice::IntervalView intervals_b(
      "subsetix_csr_row_intersection_b", nB);

  auto h_a = Kokkos::create_mirror_view(intervals_a);
  auto h_b = Kokkos::create_mirror_view(intervals_b);

  for (std::size_t i = 0; i < nA; ++i) {
    h_a(i) = c.intervals_a[i];
  }
  for (std::size_t i = 0; i < nB; ++i) {
    h_b(i) = c.intervals_b[i];
  }

  Kokkos::deep_copy(intervals_a, h_a);
  Kokkos::deep_copy(intervals_b, h_b);

  Kokkos::View<std::size_t, DeviceMemorySpace> d_count(
      "subsetix_csr_row_intersection_count");

  Kokkos::parallel_for(
      "subsetix_csr_row_intersection_count_kernel",
      Kokkos::RangePolicy<ExecSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        const std::size_t count =
            detail::row_intersection_count(intervals_a, 0, nA,
                                           intervals_b, 0, nB);
        d_count() = count;
      });

  ExecSpace().fence();

  std::size_t count = 0;
  Kokkos::deep_copy(count, d_count);
  EXPECT_EQ(count, c.expected_intersection.size())
      << "row_intersection_count mismatch in case " << case_id;

  IntervalSet2DDevice::IntervalView intervals_out(
      "subsetix_csr_row_intersection_out", count);

  Kokkos::parallel_for(
      "subsetix_csr_row_intersection_fill_kernel",
      Kokkos::RangePolicy<ExecSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        detail::row_intersection_fill(
            intervals_a, 0, nA,
            intervals_b, 0, nB,
            intervals_out, 0);
      });

  ExecSpace().fence();

  auto h_out = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, intervals_out);
  ASSERT_EQ(h_out.extent(0), c.expected_intersection.size())
      << "row_intersection_fill size mismatch in case "
      << case_id;

  for (std::size_t i = 0; i < c.expected_intersection.size(); ++i) {
    const auto& expected = c.expected_intersection[i];
    const auto got = h_out(i);
    EXPECT_EQ(got.begin, expected.begin)
        << "row_intersection_fill begin mismatch in case "
        << case_id << " interval " << i;
    EXPECT_EQ(got.end, expected.end)
        << "row_intersection_fill end mismatch in case "
        << case_id << " interval " << i;
  }
}

inline void run_row_difference_case(const RowOpCase& c,
                                    std::size_t case_id) {
  using ExecSpace = Kokkos::DefaultExecutionSpace;

  const std::size_t nA = c.intervals_a.size();
  const std::size_t nB = c.intervals_b.size();

  IntervalSet2DDevice::IntervalView intervals_a(
      "subsetix_csr_row_difference_a", nA);
  IntervalSet2DDevice::IntervalView intervals_b(
      "subsetix_csr_row_difference_b", nB);

  auto h_a = Kokkos::create_mirror_view(intervals_a);
  auto h_b = Kokkos::create_mirror_view(intervals_b);

  for (std::size_t i = 0; i < nA; ++i) {
    h_a(i) = c.intervals_a[i];
  }
  for (std::size_t i = 0; i < nB; ++i) {
    h_b(i) = c.intervals_b[i];
  }

  Kokkos::deep_copy(intervals_a, h_a);
  Kokkos::deep_copy(intervals_b, h_b);

  Kokkos::View<std::size_t, DeviceMemorySpace> d_count(
      "subsetix_csr_row_difference_count");

  Kokkos::parallel_for(
      "subsetix_csr_row_difference_count_kernel",
      Kokkos::RangePolicy<ExecSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        const std::size_t count =
            detail::row_difference_count(intervals_a, 0, nA,
                                         intervals_b, 0, nB);
        d_count() = count;
      });

  ExecSpace().fence();

  std::size_t count = 0;
  Kokkos::deep_copy(count, d_count);
  EXPECT_EQ(count, c.expected_difference_a_minus_b.size())
      << "row_difference_count mismatch in case " << case_id;

  IntervalSet2DDevice::IntervalView intervals_out(
      "subsetix_csr_row_difference_out", count);

  Kokkos::parallel_for(
      "subsetix_csr_row_difference_fill_kernel",
      Kokkos::RangePolicy<ExecSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        detail::row_difference_fill(
            intervals_a, 0, nA,
            intervals_b, 0, nB,
            intervals_out, 0);
      });

  ExecSpace().fence();

  auto h_out = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, intervals_out);
  ASSERT_EQ(h_out.extent(0),
            c.expected_difference_a_minus_b.size())
      << "row_difference_fill size mismatch in case "
      << case_id;

  for (std::size_t i = 0;
       i < c.expected_difference_a_minus_b.size();
       ++i) {
    const auto& expected = c.expected_difference_a_minus_b[i];
    const auto got = h_out(i);
    EXPECT_EQ(got.begin, expected.begin)
        << "row_difference_fill begin mismatch in case "
        << case_id << " interval " << i;
    EXPECT_EQ(got.end, expected.end)
        << "row_difference_fill end mismatch in case "
        << case_id << " interval " << i;
  }
}

} // namespace csr_test
} // namespace subsetix
