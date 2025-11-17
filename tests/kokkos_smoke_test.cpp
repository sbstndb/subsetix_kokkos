#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace {

int compute_parallel_sum_on_host_mirror() {
  constexpr int N = 10;
  Kokkos::View<int*> data("data", N);

  Kokkos::parallel_for(
      "InitData",
      N,
      KOKKOS_LAMBDA(const int i) { data(i) = i; });

  auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);

  int sum = 0;
  for (int i = 0; i < N; ++i) {
    sum += mirror(i);
  }

  return sum;
}

} // namespace

TEST(KokkosSmokeTest, ParallelSumOnHostMirror) {
  EXPECT_EQ(compute_parallel_sum_on_host_mirror(), 45);
}
