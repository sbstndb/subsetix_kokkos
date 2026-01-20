/// @file compile_test.cpp
/// @brief Simple compilation test for Phase 0 headers
///
/// This file tests that all the new Phase 0 headers compile correctly

#include <Kokkos_Core.hpp>

// Include all Phase 0 headers
#include "../examples/mach2_cylinder/mach2_config.hpp"
#include "../examples/mach2_cylinder/mach2_fvd_bridge.hpp"
#include "../examples/mach2_cylinder/mach2_utils.hpp"
#include "field_comparator.hpp"

int main() {
    Kokkos::ScopeGuard guard(0, nullptr);

    // Test 1: Config compiles
    mach2::RunConfig cfg;
    cfg.nx = 400;
    cfg.ny = 160;
    cfg.normalize();

    // Test 2: Bridge types compile
    using namespace mach2::bridge;
    Conserved U{1.0f, 2.0f, 0.0f, 10.0f};
    Primitive q = cons_to_prim(U, 1.4f);
    Conserved U2 = prim_to_cons(q, 1.4f);
    float a = sound_speed(q, 1.4f);

    // Test 3: Field comparator compiles
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> v1("v1", 10);
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace> v2("v2", 10);
    // auto result = mach2::validation::compare_views(v1, v2);

    std::cout << "Phase 0 compilation test: PASSED\n";
    std::cout << "  Config: " << cfg.nx << "x" << cfg.ny << "\n";
    std::cout << "  Conserved: " << U.rho << ", " << U.rhou << ", " << U.rhov << ", " << U.E << "\n";
    std::cout << "  Primitive: " << q.rho << ", " << q.u << ", " << q.v << ", " << q.p << "\n";
    std::cout << "  Sound speed: " << a << "\n";

    return 0;
}
