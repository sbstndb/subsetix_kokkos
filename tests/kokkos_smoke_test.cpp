#include <Kokkos_Core.hpp>
#include <cstdio>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        constexpr int N = 10;
        Kokkos::View<int*> data("data", N);

        Kokkos::parallel_for(
            "InitData",
            N,
            KOKKOS_LAMBDA(const int i) { data(i) = i; });

        // Simple host mirror to touch the data and ensure basic API usage.
        auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);

        int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += mirror(i);
        }

        if (sum == 45) {
            std::printf("Kokkos smoke test passed (sum=%d)\n", sum);
        } else {
            std::printf("Kokkos smoke test failed (sum=%d)\n", sum);
            return 1;
        }
    }
    Kokkos::finalize();
    return 0;
}

