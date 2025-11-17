#include <Kokkos_Core.hpp>
#include <cstdio>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        constexpr int N = 1 << 20;
        Kokkos::View<double*> data("data", N);

        Kokkos::parallel_for(
            "FillOnes",
            N,
            KOKKOS_LAMBDA(const int i) { data(i) = 1.0; });

        // Just to ensure the kernel is launched; we don't check performance.
        auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);

        double sum = 0.0;
        for (int i = 0; i < N; ++i) {
            sum += mirror(i);
        }

        std::printf("Kokkos dummy benchmark sum=%f\n", sum);
    }
    Kokkos::finalize();
    return 0;
}

