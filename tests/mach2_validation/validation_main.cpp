/// @file validation_main.cpp
/// @brief Main entry point for mach2 validation framework
///
/// Phase 0: Validation Infrastructure
///
/// This program provides the testing framework for validating the gradual
/// migration from mach2_cylinder.cpp to the FVD abstraction layer.
///
/// Usage:
///   ./mach2_validation [--baseline] [--compare] [--help]
///
/// Modes:
///   --baseline   : Run original mach2 and save reference data
///   --compare    : Compare current implementation against baseline
///   --type-safety: Run compile-time type safety checks (Phase 0.5)

#include <Kokkos_Core.hpp>

#include "field_comparator.hpp"

#include "../examples/mach2_cylinder/mach2_config.hpp"
#include "../examples/mach2_cylinder/mach2_fvd_bridge.hpp"
#include "../examples/mach2_cylinder/mach2_utils.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <cstdlib>

namespace mach2::validation {

using namespace bridge;

// ============================================================================
// VALIDATION MODES
// ============================================================================

enum class Mode {
    Baseline,   // Generate baseline data from original implementation
    Compare,    // Compare against baseline
    TypeSafety, // Run type safety checks (Phase 0.5)
    Diagnostics // Run diagnostic tests
};

// ============================================================================
// BASELINE GENERATION
// ============================================================================

/// @brief Run original mach2 and save baseline data
/// @param cfg Simulation configuration
/// @param output_dir Output directory for baseline data
int generate_baseline(const RunConfig& cfg,
                      const std::filesystem::path& output_dir) {
    std::cout << "=== MACH2 VALIDATION: BASELINE GENERATION ===\n";
    std::cout << "Configuration:\n";
    std::cout << "  Grid: " << cfg.nx << "x" << cfg.ny << "\n";
    std::cout << "  Cylinder: (" << cfg.cx << ", " << cfg.cy
              << ") radius=" << cfg.radius << "\n";
    std::cout << "  Mach inlet: " << cfg.mach_inlet << "\n";
    std::cout << "  CFL: " << cfg.cfl << "\n";
    std::cout << "  t_final: " << cfg.t_final << "\n";
    std::cout << "  AMR: " << (cfg.enable_amr ? "enabled" : "disabled") << "\n";
    std::cout << "Output directory: " << output_dir << "\n\n";

    // TODO: Call original mach2 implementation
    // For now, this is a stub that will be implemented when we
    // extract the main loop from mach2_cylinder.cpp

    std::cout << "[STUB] Baseline generation not yet implemented.\n";
    std::cout << "[TODO] Integrate with original mach2_cylinder.cpp main loop\n";

    return 0;
}

// ============================================================================
// COMPARISON MODE
// ============================================================================

/// @brief Compare current implementation against baseline
/// @param cfg Simulation configuration
/// @param baseline_dir Directory containing baseline data
int compare_baseline(const RunConfig& cfg,
                     const std::filesystem::path& baseline_dir) {
    std::cout << "=== MACH2 VALIDATION: COMPARISON MODE ===\n";
    std::cout << "Baseline directory: " << baseline_dir << "\n\n";

    // TODO: Load baseline data
    // TODO: Run current implementation
    // TODO: Compare fields using field_comparator.hpp

    std::cout << "[STUB] Comparison mode not yet implemented.\n";
    std::cout << "[TODO] Implement baseline data loading and comparison\n";

    return 0;
}

// ============================================================================
// TYPE SAFETY CHECKS (Phase 0.5)
// ============================================================================

/// @brief Run compile-time type safety checks
int run_type_safety_checks() {
    std::cout << "=== MACH2 VALIDATION: TYPE SAFETY CHECKS ===\n\n";

    #ifdef SUBSETIX_FVD_ENABLED
        std::cout << "FVD layer is enabled.\n";

        // Check binary compatibility (compile-time)
        std::cout << "Checking binary compatibility...\n";
        std::cout << "  sizeof(Conserved):     "
                  << sizeof(Conserved) << " bytes\n";
        std::cout << "  sizeof(Primitive):    "
                  << sizeof(Primitive) << " bytes\n";

        #ifdef SUBSETIX_FVD_TYPES
            using FVDSystem = fvd_types::System;
            using FVDConserved = typename FVDSystem::Conserved;
            using FVDPrimitive = typename FVDSystem::Primitive;

            std::cout << "  sizeof(FVD::Conserved): "
                      << sizeof(FVDConserved) << " bytes\n";
            std::cout << "  sizeof(FVD::Primitive): "
                      << sizeof(FVDPrimitive) << " bytes\n";

            // Compile-time checks are in mach2_fvd_bridge.hpp
            std::cout << "\n[PASS] Type safety static_asserts passed.\n";
        #endif

        // Check GPU safety
        std::cout << "\nChecking GPU safety...\n";
        std::cout << "  Conserved is trivially copyable: "
                  << (std::is_trivially_copyable_v<Conserved> ? "yes" : "NO") << "\n";
        std::cout << "  Primitive is trivially copyable: "
                  << (std::is_trivially_copyable_v<Primitive> ? "yes" : "NO") << "\n";

        if (!std::is_trivially_copyable_v<Conserved> ||
            !std::is_trivially_copyable_v<Primitive>) {
            std::cout << "[FAIL] Types are not GPU-safe!\n";
            return 1;
        }

        std::cout << "[PASS] GPU safety checks passed.\n";

    #else
        std::cout << "FVD layer is NOT enabled (SUBSETIX_FVD_CHANNELS=OFF).\n";
        std::cout << "[INFO] Type safety checks require FVD layer.\n";
    #endif

    std::cout << "\n=== TYPE SAFETY CHECKS COMPLETE ===\n";
    return 0;
}

// ============================================================================
// DIAGNOSTIC TESTS
// ============================================================================

/// @brief Run diagnostic tests on the bridge implementation
int run_diagnostic_tests() {
    std::cout << "=== MACH2 VALIDATION: DIAGNOSTIC TESTS ===\n\n";

    // Test 1: Basic structure sizes
    std::cout << "Test 1: Structure sizes\n";
    std::cout << "  Conserved: " << sizeof(Conserved) << " bytes\n";
    std::cout << "  Primitive: " << sizeof(Primitive) << " bytes\n";
    std::cout << "  Expected: 16 bytes (4 floats)\n";

    if (sizeof(Conserved) != 16 || sizeof(Primitive) != 16) {
        std::cout << "[FAIL] Structure size mismatch!\n";
        return 1;
    }
    std::cout << "[PASS]\n\n";

    // Test 2: Function availability
    std::cout << "Test 2: Bridge function availability\n";
    std::cout << "  cons_to_prim: ";
    // Check if function compiles
    Conserved U{1.0f, 2.0f, 0.0f, 10.0f};
    Primitive q = cons_to_prim(U, 1.4f);
    std::cout << "OK (q.p=" << q.p << ")\n";

    std::cout << "  prim_to_cons: ";
    Conserved U2 = prim_to_cons(q, 1.4f);
    std::cout << "OK (U2.E=" << U2.E << ")\n";

    std::cout << "  sound_speed: ";
    Real a = sound_speed(q, 1.4f);
    std::cout << "OK (a=" << a << ")\n";

    std::cout << "[PASS]\n\n";

    // Test 3: Round-trip conversion
    std::cout << "Test 3: Round-trip conversion\n";
    Conserved original{1.5f, 3.0f, 0.5f, 15.0f};
    Primitive p = cons_to_prim(original, 1.4f);
    Conserved converted = prim_to_cons(p, 1.4f);

    const Real tol = 1e-6f;
    bool match = std::abs(original.rho - converted.rho) < tol &&
                 std::abs(original.rhou - converted.rhou) < tol &&
                 std::abs(original.rhov - converted.rhov) < tol &&
                 std::abs(original.E - converted.E) < tol;

    if (match) {
        std::cout << "[PASS] Round-trip successful\n";
    } else {
        std::cout << "[FAIL] Round-trip mismatch!\n";
        std::cout << "  Original:   " << original.rho << ", " << original.rhou
                  << ", " << original.rhov << ", " << original.E << "\n";
        std::cout << "  Converted:  " << converted.rho << ", " << converted.rhou
                  << ", " << converted.rhov << ", " << converted.E << "\n";
        return 1;
    }

    std::cout << "\n=== DIAGNOSTIC TESTS COMPLETE ===\n";
    return 0;
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

} // namespace mach2::validation

int main(int argc, char* argv[]) {
    using namespace mach2::validation;

    // Initialize Kokkos
    Kokkos::ScopeGuard guard(argc, argv);

    // Parse command line (simple parsing)
    Mode mode = Mode::Diagnostics;
    std::filesystem::path data_dir = ".";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--baseline") {
            mode = Mode::Baseline;
        } else if (arg == "--compare") {
            mode = Mode::Compare;
        } else if (arg == "--type-safety") {
            mode = Mode::TypeSafety;
        } else if (arg == "--diagnostics") {
            mode = Mode::Diagnostics;
        } else if (arg == "--data-dir" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "MACH2 Validation Framework - Phase 0\n\n"
                      << "Usage: " << argv[0] << " [OPTIONS]\n\n"
                      << "Options:\n"
                      << "  --baseline       Generate baseline data\n"
                      << "  --compare        Compare against baseline\n"
                      << "  --type-safety    Run type safety checks\n"
                      << "  --diagnostics    Run diagnostic tests (default)\n"
                      << "  --data-dir DIR   Data directory\n"
                      << "  --help, -h       Show this message\n";
            return 0;
        }
    }

    // Run selected mode
    switch (mode) {
        case Mode::Baseline:
            return generate_baseline(mach2::RunConfig{}, data_dir);
        case Mode::Compare:
            return compare_baseline(mach2::RunConfig{}, data_dir);
        case Mode::TypeSafety:
            return run_type_safety_checks();
        case Mode::Diagnostics:
            return run_diagnostic_tests();
        default:
            std::cerr << "Unknown mode\n";
            return 1;
    }

    return 0;
}
