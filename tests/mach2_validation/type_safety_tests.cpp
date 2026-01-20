/// @file type_safety_tests.cpp
/// @brief Type safety and compatibility tests for Phase 0.5
///
/// Phase 0.5: Type Safety Validation
/// - Compile-time type compatibility checks
/// - GPU safety verification
/// - Binary equivalence validation
/// - Function signature matching

#include <Kokkos_Core.hpp>
#include <iostream>
#include <iomanip>
#include <type_traits>

// Use the bridge types (don't depend on FVD yet)
#include "../../examples/mach2_cylinder/mach2_fvd_bridge.hpp"

namespace mach2::validation::type_safety {

using namespace bridge;

// ============================================================================
// 1. COMPILE-TIME TYPE CHECKS
// ============================================================================

struct TypeCheckResults {
    int passed = 0;
    int failed = 0;
    int warnings = 0;

    void pass(const char* test, const char* details = "") {
        std::cout << "[PASS] " << test;
        if (details[0] != '\0') {
            std::cout << ": " << details;
        }
        std::cout << "\n";
        ++passed;
    }

    void fail(const char* test, const char* reason) {
        std::cout << "[FAIL] " << test << ": " << reason << "\n";
        ++failed;
    }

    void warn(const char* test, const char* reason) {
        std::cout << "[WARN] " << test << ": " << reason << "\n";
        ++warnings;
    }

    void print_summary() const {
        std::cout << "\n=== Type Safety Summary ===\n";
        std::cout << "  Passed:  " << passed << "\n";
        std::cout << "  Failed:  " << failed << "\n";
        std::cout << "  Warnings: " << warnings << "\n";
        if (failed == 0) {
            std::cout << "  Result:   ✅ ALL CHECKS PASSED\n";
        } else {
            std::cout << "  Result:   ❌ SOME CHECKS FAILED\n";
        }
    }

    bool all_passed() const { return failed == 0; }
};

// ============================================================================
// TESTS
// ============================================================================

/// Test 1: Structure size and alignment
void test_structure_sizes(TypeCheckResults& results) {
    std::cout << "\n--- Test 1: Structure Sizes and Alignment ---\n";

    // Conserved structure
    constexpr size_t expected_conserved_size = 4 * sizeof(Real);  // 4 floats
    constexpr size_t expected_conserved_align = alignof(Real);

    if (sizeof(Conserved) == expected_conserved_size) {
        auto msg = std::to_string(sizeof(Conserved)) + " bytes";
        results.pass("Conserved size", msg.c_str());
    } else {
        auto msg = std::to_string(sizeof(Conserved)) + " vs expected " +
                   std::to_string(expected_conserved_size);
        results.fail("Conserved size", msg.c_str());
    }

    if (alignof(Conserved) == expected_conserved_align) {
        auto msg = std::to_string(alignof(Conserved)) + " bytes";
        results.pass("Conserved alignment", msg.c_str());
    } else {
        auto msg = std::to_string(alignof(Conserved)) + " vs expected " +
                   std::to_string(expected_conserved_align);
        results.fail("Conserved alignment", msg.c_str());
    }

    // Primitive structure
    constexpr size_t expected_primitive_size = 4 * sizeof(Real);  // 4 floats
    constexpr size_t expected_primitive_align = alignof(Real);

    if (sizeof(Primitive) == expected_primitive_size) {
        auto msg = std::to_string(sizeof(Primitive)) + " bytes";
        results.pass("Primitive size", msg.c_str());
    } else {
        auto msg = std::to_string(sizeof(Primitive)) + " vs expected " +
                   std::to_string(expected_primitive_size);
        results.fail("Primitive size", msg.c_str());
    }

    if (alignof(Primitive) == expected_primitive_align) {
        auto msg = std::to_string(alignof(Primitive)) + " bytes";
        results.pass("Primitive alignment", msg.c_str());
    } else {
        auto msg = std::to_string(alignof(Primitive)) + " vs expected " +
                   std::to_string(expected_primitive_align);
        results.fail("Primitive alignment", msg.c_str());
    }

    // Layout verification (offset checks)
    Conserved c;
    auto offset = [&](const auto& member) {
        return reinterpret_cast<const char*>(&member) -
               reinterpret_cast<const char*>(&c);
    };

    size_t rho_offset = offset(c.rho);
    size_t rhou_offset = offset(c.rhou);
    size_t rhov_offset = offset(c.rhov);
    size_t E_offset = offset(c.E);

    // Expected offsets for float (4 bytes each)
    if (rho_offset == 0 && rhou_offset == 4 && rhov_offset == 8 && E_offset == 12) {
        results.pass("Conserved layout", "rho:0 rhou:4 rhov:8 E:12");
    } else {
        results.fail("Conserved layout",
                    ("rho:" + std::to_string(rho_offset) +
                     " rhou:" + std::to_string(rhou_offset) +
                     " rhov:" + std::to_string(rhov_offset) +
                     " E:" + std::to_string(E_offset)).c_str());
    }
}

/// Test 2: Type properties for GPU usage
void test_type_properties(TypeCheckResults& results) {
    std::cout << "\n--- Test 2: Type Properties (GPU Safety) ---\n";

    // Trivially copyable (required for GPU transfer)
    if constexpr (std::is_trivially_copyable_v<Conserved>) {
        results.pass("Conserved is trivially copyable");
    } else {
        results.fail("Conserved", "NOT trivially copyable - GPU transfer may fail");
    }

    if constexpr (std::is_trivially_copyable_v<Primitive>) {
        results.pass("Primitive is trivially copyable");
    } else {
        results.fail("Primitive", "NOT trivially copyable - GPU transfer may fail");
    }

    // Standard layout (required for offsetof)
    if constexpr (std::is_standard_layout_v<Conserved>) {
        results.pass("Conserved is standard layout");
    } else {
        results.fail("Conserved", "NOT standard layout");
    }

    if constexpr (std::is_standard_layout_v<Primitive>) {
        results.pass("Primitive is standard layout");
    } else {
        results.fail("Primitive", "NOT standard layout");
    }

    // Trivial default constructor
    if constexpr (std::is_trivially_default_constructible_v<Conserved>) {
        results.pass("Conserved has trivial default constructor");
    } else {
        results.warn("Conserved", "default constructor is not trivial (OK if members have default init)");
    }

    if constexpr (std::is_trivially_default_constructible_v<Primitive>) {
        results.pass("Primitive has trivial default constructor");
    } else {
        results.warn("Primitive", "default constructor is not trivial (OK if members have default init)");
    }
}

/// Test 3: Function equivalence (bridge vs FVD)
#ifdef SUBSETIX_FVD_ENABLED
    #include <subsetix/fvd/system/euler2d.hpp>

    void test_fvd_compatibility(TypeCheckResults& results) {
        std::cout << "\n--- Test 3: FVD Compatibility ---\n";

        using FVDSystem = subsetix::fvd::Euler2D<Real>;
        using FVDConserved = typename FVDSystem::Conserved;
        using FVDPrimitive = typename FVDSystem::Primitive;

        // Binary compatibility
        if constexpr (sizeof(FVDConserved) == sizeof(Conserved)) {
            results.pass("FVD::Conserved size matches mach2::Conserved");
        } else {
            results.fail("FVD::Conserved size",
                        (std::to_string(sizeof(FVDConserved)) + " vs " +
                         std::to_string(sizeof(Conserved))).c_str());
        }

        if constexpr (sizeof(FVDPrimitive) == sizeof(Primitive)) {
            results.pass("FVD::Primitive size matches mach2::Primitive");
        } else {
            results.fail("FVD::Primitive size",
                        (std::to_string(sizeof(FVDPrimitive)) + " vs " +
                         std::to_string(sizeof(Primitive))).c_str());
        }

        // Alignment compatibility
        if constexpr (alignof(FVDConserved) == alignof(Conserved)) {
            results.pass("FVD::Conserved alignment matches mach2::Conserved");
        } else {
            results.fail("FVD::Conserved alignment",
                        (std::to_string(alignof(FVDConserved)) + " vs " +
                         std::to_string(alignof(Conserved))).c_str());
        }

        // GPU safety of FVD types
        if constexpr (std::is_trivially_copyable_v<FVDConserved>) {
            results.pass("FVD::Conserved is GPU-safe");
        } else {
            results.fail("FVD::Conserved", "NOT GPU-safe");
        }

        if constexpr (std::is_trivially_copyable_v<FVDPrimitive>) {
            results.pass("FVD::Primitive is GPU-safe");
        } else {
            results.fail("FVD::Primitive", "NOT GPU-safe");
        }

        // Check if KOKKOS_INLINE_FUNCTION is properly defined
        #ifdef KOKKOS_INLINE_FUNCTION
            results.pass("KOKKOS_INLINE_FUNCTION is defined");
        #else
            results.fail("KOKKOS_INLINE_FUNCTION", "NOT defined - device code won't compile");
        #endif
    }
#else
    void test_fvd_compatibility(TypeCheckResults& results) {
        std::cout << "\n--- Test 3: FVD Compatibility ---\n";
        results.warn("FVD layer", "SUBSETIX_FVD_ENABLED not set - skipping FVD tests");
        results.warn("FVD layer", "Enable with: -DSUBSETIX_FVD_ENABLED=ON");
    }
#endif

/// Test 4: Runtime numerical equivalence
void test_numerical_equivalence(TypeCheckResults& results) {
    std::cout << "\n--- Test 4: Numerical Equivalence ---\n";

    // Test round-trip conversion
    const Real gamma = 1.4f;

    Conserved U_original{1.5f, 3.0f, 0.5f, 15.0f};

    // Convert using bridge functions
    Primitive q = cons_to_prim(U_original, gamma);
    Conserved U_converted = prim_to_cons(q, gamma);

    // Check equivalence
    const Real tol = 1e-6f;
    bool rho_match = std::abs(U_original.rho - U_converted.rho) < tol;
    bool rhou_match = std::abs(U_original.rhou - U_converted.rhou) < tol;
    bool rhov_match = std::abs(U_original.rhov - U_converted.rhov) < tol;
    bool E_match = std::abs(U_original.E - U_converted.E) < tol;

    if (rho_match && rhou_match && rhov_match && E_match) {
        results.pass("Round-trip conversion", "Conserved → Primitive → Conserved");
    } else {
        results.fail("Round-trip conversion",
                    ("mismatch detected: rho=" + std::to_string(U_converted.rho) +
                     " rhou=" + std::to_string(U_converted.rhou) +
                     " rhov=" + std::to_string(U_converted.rhov) +
                     " E=" + std::to_string(U_converted.E)).c_str());
    }

    // Test sound speed
    Real a = sound_speed(q, gamma);
    const Real expected_a = std::sqrt(gamma * q.p / q.rho);
    if (std::abs(a - expected_a) < tol) {
        auto msg = std::to_string(a);
        results.pass("Sound speed calculation", msg.c_str());
    } else {
        auto msg = std::to_string(a) + " vs " + std::to_string(expected_a);
        results.fail("Sound speed", msg.c_str());
    }

    // Test known values (standard atmosphere at sea level)
    Primitive sea_level{1.225f, 100.0f, 0.0f, 101325.0f};
    Conserved U_sea = prim_to_cons(sea_level, 1.4f);
    Real a_sea = sound_speed(sea_level, 1.4f);

    // Expected sound speed at sea level: sqrt(1.4 * 101325 / 1.225) ≈ 340.3 m/s
    const Real expected_a_sea = 340.3f;
    if (std::abs(a_sea - expected_a_sea) < 1.0f) {
        auto msg = std::to_string(a_sea) + " m/s";
        results.pass("Sea level sound speed", msg.c_str());
    } else {
        auto msg = std::to_string(a_sea) + " m/s vs expected " +
                   std::to_string(expected_a_sea) + " m/s";
        results.fail("Sea level sound speed", msg.c_str());
    }
}

/// Test 5: Kokkos execution space compatibility
void test_kokkos_compatibility(TypeCheckResults& results) {
    std::cout << "\n--- Test 5: Kokkos Compatibility ---\n";

    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using DeviceSpace = Kokkos::DefaultExecutionSpace::device_type;

    std::cout << "  Execution Space: " << Kokkos::DefaultExecutionSpace::name() << "\n";
    // Note: memory_space_name() is not available in all Kokkos versions
    // std::cout << "  Memory Space: " << Kokkos::DefaultExecutionSpace::memory_space_name() << "\n";

    // Test that types can be used in Kokkos views
    using ConservedView = Kokkos::View<Conserved*, ExecSpace>;
    using PrimitiveView = Kokkos::View<Primitive*, ExecSpace>;

    results.pass("Kokkos::View<Conserved*> compiles");
    results.pass("Kokkos::View<Primitive*> compiles");

    // Test parallel_for with Conserved
    ConservedView test_view("test", 10);
    Kokkos::parallel_for("test_kernel",
        Kokkos::RangePolicy<ExecSpace>(0, 1),
        KOKKOS_LAMBDA(const int i) {
            Conserved c{1.0f, 2.0f, 3.0f, 4.0f};
            test_view(i) = c;
        });
    Kokkos::fence();
    results.pass("Kokkos::parallel_for with Conserved");

    // Test parallel_reduce with Real
    Kokkos::View<Real*, ExecSpace> test_real("real", 10);
    Real sum = 0.0f;
    Kokkos::parallel_reduce("test_reduce",
        Kokkos::RangePolicy<ExecSpace>(0, 10),
        KOKKOS_LAMBDA(const int i, Real& local_sum) {
            local_sum += test_real(i);
        }, sum);
    results.pass("Kokkos::parallel_reduce with Real");
}

/// Test 6: Memory layout verification
void test_memory_layout(TypeCheckResults& results) {
    std::cout << "\n--- Test 6: Memory Layout Verification ---\n";

    // Verify SoA (Structure of Arrays) vs AoS (Array of Structures)
    // Our CSR fields use SoA: separate arrays for rho, rhou, rhov, E
    // Conserved/Primitive structs use AoS

    // Test that memcpy between equivalent types works
    Conserved src{1.0f, 2.0f, 3.0f, 4.0f};
    Conserved dst;
    std::memcpy(&dst, &src, sizeof(Conserved));

    if (dst.rho == 1.0f && dst.rhou == 2.0f && dst.rhov == 3.0f && dst.E == 4.0f) {
        results.pass("memcpy Conserved", "binary copy successful");
    } else {
        results.fail("memcpy Conserved", "binary copy failed");
    }

    // Test padding (no implicit padding between members)
    static_assert(sizeof(Conserved) == 4 * sizeof(Real),
                  "Conserved must have no padding");
    results.pass("Conserved has no padding", "sizeof == 4 * sizeof(Real)");

    // Test that array access works correctly
    Conserved array[2] = {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}};

    if (array[0].rhou == 2.0f && array[1].rhov == 7.0f) {
        results.pass("Array indexing works", "array[0].rhou=2.0, array[1].rhov=7.0");
    } else {
        results.fail("Array indexing", "unexpected values");
    }
}

// ============================================================================
// MAIN
// ============================================================================

} // namespace mach2::validation::type_safety

int main(int argc, char* argv[]) {
    using namespace mach2::validation::type_safety;

    Kokkos::ScopeGuard guard(argc, argv);

    std::cout << "==============================================\n";
    std::cout << "  MACH2 VALIDATION: PHASE 0.5 - TYPE SAFETY\n";
    std::cout << "==============================================\n";

    TypeCheckResults results;

    test_structure_sizes(results);
    test_type_properties(results);
    test_fvd_compatibility(results);
    test_numerical_equivalence(results);
    test_kokkos_compatibility(results);
    test_memory_layout(results);

    results.print_summary();

    return results.all_passed() ? 0 : 1;
}
