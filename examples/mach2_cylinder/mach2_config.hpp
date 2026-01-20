#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace mach2 {

/// Configuration for Mach 2 cylinder simulation
/// This is a common configuration structure that can be shared
/// between the original mach2_cylinder.cpp and the FVD refactored version.
struct RunConfig {
    // Grid resolution
    int nx = 400;
    int ny = 160;
    int cx = -1;  // Cylinder center x (set from nx if negative)
    int cy = -1;  // Cylinder center y (set from ny if negative)
    int radius = 20;

    // Physics parameters
    float mach_inlet = 2.0f;
    float rho = 1.0f;
    float p = 1.0f;
    float gamma = 1.4f;

    // Time stepping
    float cfl = 0.45f;
    float t_final = 0.01f;
    int max_steps = 5000;

    // Output
    int output_stride = 50;
    bool enable_output = true;

    // AMR parameters
    int max_amr_levels = 4;      // Includes coarse level
    bool enable_amr = true;
    float amr_fraction = 0.5f;  // Fraction of domain refined in each direction
    int amr_guard = 2;           // Coarse-cell guard radius around refined zone
    int amr_remesh_stride = 0;   // 0 = static AMR, >0 = remesh every N steps

    // Boundary conditions
    bool no_slip = false;  // If true, no-slip walls; if false, slip walls

    // Custom obstacle geometry (PBM bitmap)
    std::string pbm_path;

    /// Parse command line arguments into config
    static RunConfig parse_args(int argc, char* argv[]);

    /// Validate and normalize config parameters
    void normalize();

    /// Get domain bounds from config
    struct DomainBounds {
        int x_min, x_max, y_min, y_max;
    };
    DomainBounds get_domain_bounds() const {
        return {0, nx, 0, ny};
    }
};

} // namespace mach2
