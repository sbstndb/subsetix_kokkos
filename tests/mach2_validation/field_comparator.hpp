#pragma once

#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
#include <sstream>

namespace mach2::validation {

/// Norms for field comparison
struct NormResult {
    double L1 = 0.0;   // Sum of absolute differences
    double L2 = 0.0;   // Square root of sum of squared differences
    double Linf = 0.0; // Maximum absolute difference
    std::size_t n = 0; // Number of elements compared

    /// Relative L1 norm (L1 / n)
    double relative_L1() const { return (n > 0) ? L1 / static_cast<double>(n) : 0.0; }

    /// Relative L2 norm (sqrt(L2^2 / n))
    double relative_L2() const { return (n > 0) ? std::sqrt(L2 * L2 / static_cast<double>(n)) : 0.0; }

    /// Check if all norms are below tolerance
    bool below(double tolerance) const {
        return Linf < tolerance && relative_L2() < tolerance;
    }

    /// String representation
    std::string to_string() const {
        std::ostringstream oss;
        oss << "L1=" << L1 << " L2=" << L2 << " Linf=" << Linf
            << " (rel_L2=" << relative_L2() << " n=" << n << ")";
        return oss.str();
    }
};

/// Compare two Kokkos views and compute L1, L2, Linf norms
template<typename ViewType>
NormResult compare_views(const ViewType& reference,
                        const ViewType& candidate,
                        const std::string& name = "field") {
    using ExecSpace = typename ViewType::execution_space;
    using ValueType = typename ViewType::value_type;

    static_assert(std::is_floating_point_v<ValueType>,
                  "compare_views only works with floating point types");

    if (reference.extent(0) != candidate.extent(0)) {
        std::cerr << "Error: size mismatch for '" << name
                  << "': reference=" << reference.extent(0)
                  << ", candidate=" << candidate.extent(0) << "\n";
        return NormResult{};
    }

    const std::size_t n = reference.extent(0);

    // Compute differences on device
    Kokkos::View<ValueType, ExecSpace> L1_result("L1");
    Kokkos::View<ValueType, ExecSpace> L2_result("L2");
    Kokkos::View<ValueType, ExecSpace> Linf_result("Linf");

    Kokkos::parallel_reduce(
        "compare_views_" + name,
        Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const std::size_t i,
                      ValueType& local_L1,
                      ValueType& local_L2,
                      ValueType& local_Linf) {
            const ValueType ref = reference(i);
            const ValueType cand = candidate(i);
            const ValueType diff = std::fabs(ref - cand);

            local_L1 += diff;
            local_L2 += diff * diff;
            local_Linf = Kokkos::fmax(local_Linf, diff);
        },
        Kokkos::Sum<ValueType, ExecSpace>(L1_result),
        Kokkos::Sum<ValueType, ExecSpace>(L2_result),
        Kokkos::Max<ValueType, ExecSpace>(Linf_result));

    // Copy results to host
    auto L1_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, L1_result);
    auto L2_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, L2_result);
    auto Linf_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Linf_result);

    // L2 needs square root
    NormResult result;
    result.L1 = static_cast<double>(L1_host());
    result.L2 = std::sqrt(static_cast<double>(L2_host()));
    result.Linf = static_cast<double>(Linf_host());
    result.n = n;

    return result;
}

/// Compare multiple views and check if all pass tolerance
struct MultiFieldComparison {
    std::string name;
    NormResult rho;
    NormResult rhou;
    NormResult rhov;
    NormResult E;

    /// Check if all fields are below tolerance
    bool all_below(double tolerance) const {
        return rho.below(tolerance) &&
               rhou.below(tolerance) &&
               rhov.below(tolerance) &&
               E.below(tolerance);
    }

    /// Get maximum Linf norm across all fields
    double max_Linf() const {
        return std::max({rho.Linf, rhou.Linf, rhov.Linf, E.Linf});
    }

    /// Print comparison report
    void print_report() const {
        std::cout << "Field comparison: " << name << "\n";
        std::cout << "  rho:   " << rho.to_string() << "\n";
        std::cout << "  rhou:  " << rhou.to_string() << "\n";
        std::cout << "  rhov:  " << rhov.to_string() << "\n";
        std::cout << "  E:     " << E.to_string() << "\n";
        std::cout << "  Max Linf: " << max_Linf() << "\n";
    }

    /// Print validation result
    void print_validation(double tolerance) const {
        const bool pass = all_below(tolerance);
        std::cout << (pass ? "[PASS] " : "[FAIL] ") << name << " (tolerance=" << tolerance << ")\n";
        if (!pass) {
            print_report();
        }
    }
};

} // namespace mach2::validation
