#pragma once

/// @file mach2_utils.hpp
/// @brief Utility functions extracted from mach2_cylinder.cpp
///
/// These functions are shared between the original implementation
/// and the FVD-refactored version. They provide diagnostics, output,
/// and other helper functionality.

#include "mach2_fvd_bridge.hpp"
#include "mach2_config.hpp"

#include <subsetix/csr_ops/field_stencil.hpp>
#include <subsetix/multilevel/multilevel.hpp>
#include <subsetix/io/vtk_export.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>

namespace mach2::utils {

using subsetix::csr::apply_on_set_device;
using subsetix::csr::DeviceMemorySpace;
using subsetix::csr::IntervalSet2DDevice;
using subsetix::csr::Field2DDevice;
using subsetix::MultilevelGeoDevice;
using subsetix::MultilevelFieldDevice;

// ============================================================================
// DIAGNOSTICS
// ============================================================================

/// @brief Compute derived quantities (density, pressure, Mach number)
///
/// This function computes primitive variables and diagnostics from
/// conserved variables. It is used for both analysis and output.
///
/// @param U Conserved variables (input)
/// @param density Density field output (can reuse U.rho.geometry)
/// @param pressure Pressure field output (can reuse U.rho.geometry)
/// @param mach Mach number field output (can reuse U.rho.geometry)
/// @param gamma Specific heat ratio
inline void compute_diagnostics(const bridge::ConservedFields& U,
                                Field2DDevice<Real>& density,
                                Field2DDevice<Real>& pressure,
                                Field2DDevice<Real>& mach,
                                Real gamma) {
    using namespace bridge;
    auto rho = U.rho.values;
    auto rhou = U.rhou.values;
    auto rhov = U.rhov.values;
    auto E = U.E.values;
    auto p_out = pressure.values;
    auto m_out = mach.values;

    apply_on_set_device(
        density, density.geometry,
        KOKKOS_LAMBDA(Coord /*x*/, Coord /*y*/,
                      Real& rho_out, std::size_t idx) {
            Conserved s;
            s.rho = rho(idx);
            s.rhou = rhou(idx);
            s.rhov = rhov(idx);
            s.E = E(idx);

            const Primitive q = cons_to_prim(s, gamma);
            const Real a = sound_speed(q, gamma);
            const Real vel = Kokkos::sqrt(q.u * q.u + q.v * q.v);

            rho_out = q.rho;
            p_out(idx) = q.p;
            m_out(idx) = (a > static_cast<Real>(1e-12)) ? (vel / a)
                                                         : static_cast<Real>(0.0);
        });
}

/// @brief Compute total mass in the system (for conservation checking)
/// @param U Conserved variables
/// @return Total mass (sum of rho over all active cells)
inline Real compute_total_mass(const bridge::ConservedFields& U) {
    Real total = static_cast<Real>(0.0);
    auto rho = U.rho.values;

    Kokkos::parallel_reduce(
        "mach2_total_mass",
        Kokkos::RangePolicy<DeviceMemorySpace>(0, static_cast<int>(U.size())),
        KOKKOS_LAMBDA(const int idx, Real& sum) { sum += rho(idx); },
        total);

    return total;
}

// ============================================================================
// OUTPUT
// ============================================================================

/// @brief Generate VTK filename for a given step and suffix
/// @param output_dir Output directory path
/// @param step Time step number
/// @param suffix Field name suffix (e.g., "density", "pressure")
/// @return Full path to VTK file
inline std::string vtk_filename(const std::filesystem::path& output_dir,
                                int step,
                                std::string_view suffix) {
    std::ostringstream oss;
    oss << "step_" << std::setw(5) << std::setfill('0') << step << "_" << suffix << ".vtk";

    // Use example output helper if available, otherwise just join paths
    // For now, simple path join
    return (output_dir / oss.str()).string();
}

/// @brief Write multilevel output to VTK files
///
/// This function writes the simulation results for all AMR levels
/// to VTK format for visualization and analysis.
///
/// @tparam MaxLevels Maximum number of AMR levels
/// @param geoms Array of geometries for each level
/// @param density Array of density fields for each level
/// @param pressure Array of pressure fields for each level
/// @param mach Array of Mach number fields for each level
/// @param U_active Array of conserved variables for each level
/// @param has_level Array indicating which levels are active
/// @param max_active_level Highest active level index
/// @param gamma Specific heat ratio
/// @param output_dir Output directory path
/// @param step Current time step
template<int MaxLevels>
inline void write_multilevel_outputs(
    const std::array<IntervalSet2DDevice, MaxLevels>& geoms,
    const std::array<Field2DDevice<Real>, MaxLevels>& density,
    const std::array<Field2DDevice<Real>, MaxLevels>& pressure,
    const std::array<Field2DDevice<Real>, MaxLevels>& mach,
    const std::array<bridge::ConservedFields, MaxLevels>& U_active,
    const std::array<bool, MaxLevels>& has_level,
    int max_active_level,
    Real gamma,
    const std::filesystem::path& output_dir,
    int step) {

    MultilevelGeoDevice geo;
    geo.origin_x = 0.0;
    geo.origin_y = 0.0;
    geo.root_dx = 1.0;
    geo.root_dy = 1.0;
    geo.num_active_levels = max_active_level + 1;

    MultilevelFieldDevice<Real> f_density;
    MultilevelFieldDevice<Real> f_pressure;
    MultilevelFieldDevice<Real> f_mach;
    f_density.num_active_levels = geo.num_active_levels;
    f_pressure.num_active_levels = geo.num_active_levels;
    f_mach.num_active_levels = geo.num_active_levels;

    // Fill multilevel containers
    for (int lvl = 0; lvl <= max_active_level; ++lvl) {
        if (!has_level[lvl]) {
            continue;
        }
        geo.levels[lvl] = geoms[lvl];
        f_density.levels[lvl] = density[lvl];
        f_pressure.levels[lvl] = pressure[lvl];
        f_mach.levels[lvl] = mach[lvl];

        // Compute diagnostics for this level
        compute_diagnostics(U_active[lvl],
                            f_density.levels[lvl],
                            f_pressure.levels[lvl],
                            f_mach.levels[lvl],
                            gamma);
    }

    // Write to VTK
    const auto host_geo = subsetix::deep_copy_to_host(geo);
    const auto host_rho = subsetix::deep_copy_to_host(f_density);

    subsetix::vtk::write_multilevel_field_vtk(
        host_rho, host_geo,
        vtk_filename(output_dir, step, "density"),
        "rho");
}

// ============================================================================
// CONFIGURATION
// ============================================================================

/// @brief Parse command line arguments into RunConfig
/// @param argc Argument count
/// @param argv Argument values
/// @return Parsed and normalized configuration
inline RunConfig parse_args(int argc, char* argv[]) {
    using namespace bridge;
    RunConfig cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];

        auto read_int = [&](int& out) {
            if (i + 1 < argc) {
                out = std::stoi(argv[++i]);
            }
        };

        auto read_float = [&](float& out) {
            if (i + 1 < argc) {
                out = std::stof(argv[++i]);
            }
        };

        if (arg == "--nx") read_int(cfg.nx);
        else if (arg == "--ny") read_int(cfg.ny);
        else if (arg == "--cx") read_int(cfg.cx);
        else if (arg == "--cy") read_int(cfg.cy);
        else if (arg == "--radius") read_int(cfg.radius);
        else if (arg == "--mach-inlet") read_float(cfg.mach_inlet);
        else if (arg == "--rho") read_float(cfg.rho);
        else if (arg == "--p") read_float(cfg.p);
        else if (arg == "--gamma") read_float(cfg.gamma);
        else if (arg == "--cfl") read_float(cfg.cfl);
        else if (arg == "--t-final") read_float(cfg.t_final);
        else if (arg == "--max-steps") read_int(cfg.max_steps);
        else if (arg == "--output-stride") read_int(cfg.output_stride);
        else if (arg == "--no-slip") cfg.no_slip = true;
        else if (arg == "--no-output") cfg.enable_output = false;
        else if (arg == "--no-amr") cfg.enable_amr = false;
        else if (arg == "--amr") cfg.enable_amr = true;
        else if (arg == "--amr-fraction") read_float(cfg.amr_fraction);
        else if (arg == "--amr-guard") read_int(cfg.amr_guard);
        else if (arg == "--amr-levels") read_int(cfg.max_amr_levels);
        else if (arg == "--amr-remesh-stride") read_int(cfg.amr_remesh_stride);
        else if (arg == "--pbm" && i + 1 < argc) {
            cfg.pbm_path = argv[++i];
        }
    }

    cfg.normalize();
    return cfg;
}

/// @brief Normalize configuration parameters
/// This applies default values and constraints to ensure valid config
inline void RunConfig::normalize() {
    // Set cylinder center to domain center if not specified
    if (cx < 0) cx = nx / 4;
    if (cy < 0) cy = ny / 2;

    // Clamp AMR parameters
    amr_fraction = std::clamp(amr_fraction, 1e-8f, 0.95f);
    if (amr_guard < 1) amr_guard = 1;
    if (amr_remesh_stride < 0) amr_remesh_stride = 0;

    // Clamp AMR levels
    constexpr int MAX_AMR_LEVELS = 6;
    max_amr_levels = std::clamp(max_amr_levels, 1, MAX_AMR_LEVELS);
}

/// @brief Build inflow state from configuration
/// @param cfg Configuration
/// @return Conserved variables representing the inflow state
inline bridge::Conserved build_inflow_state(const RunConfig& cfg) {
    using namespace bridge;
    Primitive q;
    q.rho = cfg.rho;
    q.p = cfg.p;
    const Real a = std::sqrt(cfg.gamma * q.p / q.rho);
    q.u = cfg.mach_inlet * a;
    q.v = 0.0f;
    return prim_to_cons(q, cfg.gamma);
}

} // namespace mach2::utils
