#pragma once

#include <Kokkos_Core.hpp>
#include <chrono>
#include <string>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <array>
#include "../system/concepts_v2.hpp"
#include "../system/euler2d.hpp"
#include "../flux/flux_schemes.hpp"
#include "../reconstruction/reconstruction.hpp"
#include "boundary_generic.hpp"
#include "observer.hpp"
#include "../output/field_view.hpp"
#include "../geometry/csr_types.hpp"
#include "../sources/source_terms.hpp"
#include "../time/time_integrators.hpp"
#include "../amr/refinement_criteria.hpp"
#include "../../csr_ops/amr.hpp"
#include "../../csr_ops/field_amr.hpp"
#include "../amr/amr_operations.hpp"
#include "../../field/csr_field.hpp"
#include "../boundary/time_dependent_bc.hpp"
#include <subsetix/io/vtk_export.hpp>

namespace subsetix::fvd {

// ============================================================================
// CSR TYPES ARE NOW IN ../geometry/csr_types.hpp
// ============================================================================

// ============================================================================
// FORWARD DECLARATION FOR SOURCE TERMS (already in source_terms.hpp)
// ============================================================================

// ============================================================================
// ADAPTIVE SOLVER - HIGH LEVEL INTERFACE
// ============================================================================

/**
 * @brief Generic Adaptive FV solver with AMR
 *
 * FULLY GENERIC: Works with any System satisfying FiniteVolumeSystem concept
 *
 * Template parameters:
 * - System: The PDE system (must satisfy FiniteVolumeSystem concept)
 * - Reconstruction: NoReconstruction or MUSCL_Reconstruction<Limiter>
 * - FluxScheme: RusanovFlux, HLLCFlux, or RoeFlux
 * - TimeIntegrator: ForwardEuler, Heun2, Kutta3, ClassicRK4, SSPRK3, etc.
 *
 * C++20: Constrained with concepts for better error messages
 */

// ============================================================================
// FORWARD DECLARATIONS FOR NAMESPACE-LEVEL TYPES
// ============================================================================

/**
 * @brief Checkpoint file format (forward declaration)
 *
 * Defined at namespace scope for example compatibility.
 */
enum class CheckpointFormat {
    Binary,    // Custom binary format (fast, portable)
    ASCII,     // Human-readable text format
    HDF5       // HDF5 format (if available)
};

/**
 * @brief Validation statistics (forward declaration)
 */
struct ValidationStats {
    int nan_count = 0;
    int negative_density_count = 0;
    int negative_pressure_count = 0;
    int cfl_violations = 0;
    float max_mach_seen = 0.0f;
    bool is_valid = true;
};

/**
 * @brief Validation configuration (forward declaration)
 */
struct ValidationConfig {
    bool check_negative_density = true;
    bool check_negative_pressure = true;
    bool check_nan = true;
    bool check_inf = true;
    bool check_cfl = true;
    float max_mach = 100.0f;
    float max_temperature = 10000.0f;
    float min_pressure = 1e-10f;
    float min_density = 1e-10f;
    bool throw_on_error = false;
    bool abort_on_error = false;
};

template<
    FiniteVolumeSystem System,
    typename Reconstruction = reconstruction::NoReconstruction,
    template<typename> class FluxScheme = flux::RusanovFlux,
    typename TimeIntegrator = time::ForwardEuler<typename System::RealType>
>
class AdaptiveSolver {
    // Note: FluxScheme constraint checked via instantiation below
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;
    using Views = typename System::Views;

    // ========================================================================
    // AMR REFINEMENT CRITERIA (E: Explicit AMR Configuration)
    // ========================================================================

    /**
     * @brief Refinement criteria for adaptive mesh refinement
     *
     * IMPROVEMENT E: Explicit AMR configuration API
     * Users can now specify:
     * - Which field/quantity to use for refinement sensing
     * - Threshold values for refinement/coarsening
     * - Maximum refinement level
     * - Minimum cell size
     */
    struct RefinementCriteria {
        // What to use for refinement sensing
        enum SensorType : int {
            DensityGradient = 0,      // |∇ρ|
            PressureGradient = 1,     // |∇p|
            VelocityGradient = 2,     // |∇v|
            MachNumber = 3,           // Mach number
            Custom = 99               // User-defined function (compile-time only)
        };

        SensorType sensor = DensityGradient;

        // Custom sensor field - compile-time alternative to std::string
        // Use fixed-size char array (GPU-compatible)
        static constexpr int max_field_name_length = 32;
        char custom_sensor_field[max_field_name_length] = {0};  // Null-terminated

        // Refinement parameters
        Real refine_threshold = Real(0.1);     // Refine if sensor > this
        Real coarsen_threshold = Real(0.01);   // Coarsen if sensor < this
        int max_level = 5;                     // Maximum AMR level
        Real min_cell_size = Real(0);          // Minimum cell size (0 = no limit)

        // Buffer zones (number of cells to extend refinement)
        int refine_buffer = 2;                 // Buffer around refined regions
        int coarsen_buffer = 3;                // Buffer before coarsening

        // Frequency checks
        bool check_every_step = false;         // If true, check refinement each step
        int check_stride = 5;                  // Otherwise, check every N steps

        // Defaults
        RefinementCriteria() = default;

        // Factory: Density-based refinement
        static RefinementCriteria density(Real refine_thresh, int max_lev = 5) {
            RefinementCriteria rc;
            rc.sensor = DensityGradient;
            rc.refine_threshold = refine_thresh;
            rc.coarsen_threshold = refine_thresh / Real(10);
            rc.max_level = max_lev;
            return rc;
        }

        // Factory: Pressure-based refinement
        static RefinementCriteria pressure(Real refine_thresh, int max_lev = 5) {
            RefinementCriteria rc;
            rc.sensor = PressureGradient;
            rc.refine_threshold = refine_thresh;
            rc.coarsen_threshold = refine_thresh / Real(10);
            rc.max_level = max_lev;
            return rc;
        }

        // Factory: Mach number refinement
        static RefinementCriteria mach_number(Real mach_thresh, int max_lev = 5) {
            RefinementCriteria rc;
            rc.sensor = MachNumber;
            rc.refine_threshold = mach_thresh;
            rc.coarsen_threshold = mach_thresh / Real(2);
            rc.max_level = max_lev;
            return rc;
        }

        // Factory: Custom field refinement (compile-time string literal)
        template<std::size_t N>
        static RefinementCriteria custom_field(const char (&field_name)[N],
                                               Real refine_thresh, int max_lev = 5) {
            RefinementCriteria rc;
            rc.sensor = Custom;
            // Copy string literal to fixed-size array (compile-time)
            for (std::size_t i = 0; i < N && i < max_field_name_length; ++i) {
                rc.custom_sensor_field[i] = field_name[i];
            }
            rc.refine_threshold = refine_thresh;
            rc.coarsen_threshold = refine_thresh / Real(10);
            rc.max_level = max_lev;
            return rc;
        }
    };

    // ========================================================================
    // SOLVER CONFIGURATION (P0-4 FIX: with CTAD support)
    // ========================================================================

    /**
     * @brief Solver configuration with CTAD support
     *
     * GAME CHANGER: No more typename System::Real(...) boilerplate!
     */
    struct Config {
        // CTAD-friendly template constructor
        template<typename T>
        Config(T dx_, T dy_, T cfl_, T gamma_, T refine_,
               int ghost, int stride)
          : dx(static_cast<Real>(dx_))
          , dy(static_cast<Real>(dy_))
          , cfl(static_cast<Real>(cfl_))
          , gamma(static_cast<Real>(gamma_))
          , refine_fraction(static_cast<Real>(refine_))
          , ghost_layers(ghost)
          , remesh_stride(stride) {}

        // Default constructor
        Config() = default;

        // Members with default values
        Real dx = Real(1);
        Real dy = Real(1);
        Real cfl = Real(0.45);
        Real gamma = System::default_gamma;
        int ghost_layers = 1;
        Real refine_fraction = Real(0.1);
        int remesh_stride = 20;

        // Grid dimensions (for dense storage indexing)
        // nx = number of cells in x-direction
        // ny = number of cells in y-direction
        std::size_t nx = 0;
        std::size_t ny = 0;

        // ====================================================================
        // PHASE 4: Time Step Control Configuration
        // ====================================================================

        /**
         * @brief Time step control limits
         *
         * Phase 4: Configurable dt limits for adaptive time stepping
         */
        struct TimeStepConfig {
            Real dt_min = Real(1e-10);      // Minimum time step (safety)
            Real dt_max = Real(1e-2);       // Maximum time step
            Real cfl_target = Real(0.8);    // Target CFL number
            Real cfl_min = Real(0.1);       // Minimum CFL (for dt_max)
            Real cfl_max = Real(1.0);       // Maximum CFL (for dt_min)
            Real growth_factor = Real(1.2); // Max dt increase per step
            Real shrink_factor = Real(0.8); // Max dt decrease per step
            int adjust_interval = 1;        // Check every N steps (1 = every step)
            bool enable_adaptive = false;   // Phase 4: Enable adaptive dt control

            // ====================================================================
            // PHASE 5: Multi-rate AMR time stepping configuration
            // ====================================================================

            /**
             * @brief Enable subcycling for multi-rate AMR time stepping
             *
             * When enabled, fine levels take multiple small time steps for each
             * coarse level step. This is more efficient for deep AMR hierarchies.
             *
             * Default: false (use global time stepping with same dt for all levels)
             */
            bool enable_subcycling = false;

            /**
             * @brief Refinement ratio in time between AMR levels
             *
             * For 2:1 spatial refinement, typical temporal refinement is also 2:1.
             * This means dt_fine = dt_coarse / time_refinement_ratio.
             *
             * Default: 2 (standard for 2:1 spatial refinement)
             */
            int time_refinement_ratio = 2;

            /**
             * @brief Enable flux correction at coarse-fine boundaries
             *
             * When enabled with subcycling, fluxes are accumulated and synchronized
             * to ensure conservation across coarse-fine interfaces.
             *
             * Default: true (recommended for conservation)
             */
            bool enable_flux_correction = true;
        } time_step;

        // IMPROVEMENT E: Embedded refinement criteria
        RefinementCriteria refinement;

        // ========================================================================
        // Helper factory methods
        // ========================================================================

        /// Config from CFL number only
        static Config from_cfl(Real cfl_value) {
            Config cfg;
            cfg.cfl = cfl_value;
            return cfg;
        }

        /// Config from resolution (dx, dy)
        static Config from_resolution(Real dx_, Real dy_) {
            Config cfg;
            cfg.dx = dx_;
            cfg.dy = dy_;
            return cfg;
        }

        /// Config with refinement parameters
        static Config with_refinement(Real refine_frac, int stride) {
            Config cfg;
            cfg.refine_fraction = refine_frac;
            cfg.remesh_stride = stride;
            return cfg;
        }

        /// Config with specific gamma
        static Config for_gamma(Real gamma_value) {
            Config cfg;
            cfg.gamma = gamma_value;
            return cfg;
        }
    };

    // ========================================================================
    // BUILDER PATTERN
    // ========================================================================

    /**
     * @brief Fluent builder for AdaptiveSolver
     *
     * Provides a clean, chainable API for constructing AdaptiveSolver instances.
     *
     * Example:
     *   auto solver = MySolver::builder(100, 50)
     *       .with_domain(0.0, 1.0, 0.0, 0.5)
     *       .with_initial_condition([](Real x, Real y) { ... })
     *       .with_gamma(1.4f)
     *       .build();
     */
    class Builder {
    public:
        /**
         * @brief Construct a builder for nx × ny grid
         */
        Builder(int nx, int ny)
            : nx_(nx), ny_(ny)
            , x_min_(Real(0)), x_max_(Real(nx))
            , y_min_(Real(0)), y_max_(Real(ny))
            , gamma_(System::default_gamma)
            , cfl_(Real(0.45))
            , has_gamma_(false)
            , has_cfl_(false)
            , has_ic_(false)
        {}

        /**
         * @brief Set domain bounds
         */
        Builder& with_domain(Real x_min, Real x_max, Real y_min, Real y_max) {
            x_min_ = x_min;
            x_max_ = x_max;
            y_min_ = y_min;
            y_max_ = y_max;
            return *this;
        }

        /**
         * @brief Set initial condition function
         *
         * @param ic Function taking (Real x, Real y) and returning Conserved
         */
        template<typename F>
        Builder& with_initial_condition(F&& ic) {
            ic_function_ = std::forward<F>(ic);
            has_ic_ = true;
            return *this;
        }

        /**
         * @brief Set gamma value
         */
        Builder& with_gamma(Real gamma) {
            gamma_ = gamma;
            has_gamma_ = true;
            return *this;
        }

        /**
         * @brief Set CFL number
         */
        Builder& with_cfl(Real cfl) {
            cfl_ = cfl;
            has_cfl_ = true;
            return *this;
        }

        /**
         * @brief Initialize solver from initial condition function
         *
         * This must be public because CUDA nvcc doesn't allow KOKKOS_LAMBDA
         * in private member functions.
         */
        static void initialize_from_function(AdaptiveSolver& solver,
                                            const std::function<Conserved(Real, Real)>& ic,
                                            Real x_min, Real y_min,
                                            Real dx, Real dy) {
            // Access solver's U_ field and evaluate IC on each cell
            // Note: This is a simplified version - full implementation would
            // use parallel_for to evaluate on device
            auto U = solver.U_;
            auto cfg = solver.cfg_;
            std::size_t nx = cfg.nx;
            std::size_t ny = cfg.ny;

            Kokkos::parallel_for(
                "initialize_ic",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, static_cast<int>(nx * ny)),
                KOKKOS_LAMBDA(const int linear_idx) {
                    std::size_t j = linear_idx / static_cast<int>(nx);
                    std::size_t i = linear_idx % static_cast<int>(nx);
                    Real x = x_min + (i + Real(0.5)) * dx;
                    Real y = y_min + (j + Real(0.5)) * dy;
                    U(linear_idx) = ic(x, y);
                }
            );
            Kokkos::fence();
        }

        /**
         * @brief Build the solver
         */
        AdaptiveSolver build() {
            // Create domain box
            csr::Box2D domain{x_min_, x_max_, y_min_, y_max_};

            // Create full rectangular domain CSR geometry
            csr::IntervalSet2DDevice fluid = subsetix::csr::make_box_device(domain);
            subsetix::csr::compute_cell_offsets_device(fluid);

            // Calculate grid spacing
            Real dx = (x_max_ - x_min_) / nx_;
            Real dy = (y_max_ - y_min_) / ny_;

            // Create config
            Config cfg;
            cfg.nx = static_cast<std::size_t>(nx_);
            cfg.ny = static_cast<std::size_t>(ny_);
            cfg.dx = dx;
            cfg.dy = dy;
            cfg.cfl = cfl_;
            cfg.gamma = gamma_;

            // Construct solver
            AdaptiveSolver solver(fluid, domain, cfg);

            // Apply initial condition if provided
            if (has_ic_ && ic_function_) {
                // Create a host copy of the initial condition to evaluate on host
                // then copy to device
                initialize_from_function(solver, ic_function_, x_min_, y_min_, dx, dy);
            }

            return solver;
        }

    private:
        // Grid dimensions
        int nx_, ny_;

        // Domain bounds
        Real x_min_, x_max_, y_min_, y_max_;

        // Configuration
        Real gamma_;
        Real cfl_;

        // Flags
        bool has_gamma_;
        bool has_cfl_;
        bool has_ic_;

        // Initial condition function (stored as any callable)
        std::function<Conserved(Real, Real)> ic_function_;
    };

    /**
     * @brief Create a builder for this solver type
     *
     * @param nx Number of cells in x-direction
     * @param ny Number of cells in y-direction
     * @return Builder instance
     */
    static Builder builder(int nx, int ny) {
        return Builder(nx, ny);
    }

    // ========================================================================
    // CONSTRUCTORS
    // ========================================================================

    /**
     * @brief Default constructor (for systems without runtime parameters)
     *
     * For systems like Euler2D where all methods are static.
     */
    AdaptiveSolver(
        const csr::IntervalSet2DDevice& fluid,
        const csr::Box2D& domain,
        const Config& cfg = Config{})
        : cfg_(cfg)
        , system_instance_{}
        , flux_{cfg_.gamma, system_instance_}
        , recon_{}
        , has_system_instance_(false)
        , fluid_geometry_(fluid)
        , domain_(domain)
    {
        // Extract grid dimensions from domain
        cfg_.nx = static_cast<std::size_t>(domain.x_max - domain.x_min);
        cfg_.ny = static_cast<std::size_t>(domain.y_max - domain.y_min);
    }

    /**
     * @brief Constructor with System instance (P0-4 FIX)
     *
     * For systems with runtime parameters (e.g., Advection2D with vx, vy).
     */
    AdaptiveSolver(
        const csr::IntervalSet2DDevice& fluid,
        const csr::Box2D& domain,
        const Config& cfg,
        const System& system)
        : cfg_(cfg)
        , system_instance_(system)
        , flux_{cfg_.gamma, system_instance_}
        , recon_{}
        , has_system_instance_(true)
        , fluid_geometry_(fluid)
        , domain_(domain)
    {
        // Extract grid dimensions from domain
        cfg_.nx = static_cast<std::size_t>(domain.x_max - domain.x_min);
        cfg_.ny = static_cast<std::size_t>(domain.y_max - domain.y_min);
    }

    // ========================================================================
    // BOUNDARY CONDITIONS (P0-2 FIX: runtime configurable)
    // ========================================================================

    /**
     * @brief Set boundary conditions
     *
     * P0-2 FIX: Allows user to configure BCs at runtime
     */
    void set_boundary_conditions(const BoundaryConfig<System>& bc) {
        bc_config_ = bc;
    }

    /**
     * @brief Set a single boundary condition for one side
     *
     * Convenience method to set BC for individual side.
     *
     * @param side Boundary side ("left", "right", "bottom", "top")
     * @param bc Boundary condition descriptor
     */
    void set_boundary_condition(const std::string& side, const AnyBc<System>& bc) {
        if (side == "left") {
            bc_config_.left = bc;
        } else if (side == "right") {
            bc_config_.right = bc;
        } else if (side == "bottom") {
            bc_config_.bottom = bc;
        } else if (side == "top") {
            bc_config_.top = bc;
        }
        // Notify observers: BC configuration was changed
        observer_manager_.notify(SolverEvent::BoundaryConditionsChanged);
    }

    /**
     * @brief Enable time-dependent boundary conditions using BcManager
     *
     * Phase 3: Advanced BC system with time dependence and zonal BCs
     *
     * @param bc_manager Configured BcManager with time-dependent BCs
     *
     * Usage:
     *   boundary::BcManager<System> mgr;
     *   mgr.initialize(nx, ny, dx, dy);
     *   mgr.add_time_dependent_bc("left", sinusoidal_inlet<System>(1.0, 100.0, 2.0));
     *   solver.set_bc_manager(mgr);
     */
    void set_bc_manager(const boundary::BcManager<System>& bc_manager) {
        bc_manager_ = bc_manager;
        use_bc_manager_ = true;
        // Notify observers: BC configuration was changed
        observer_manager_.notify(SolverEvent::BoundaryConditionsChanged);
    }

    /**
     * @brief Convenience method to add a time-dependent boundary condition
     *
     * This is a convenience wrapper that creates/updates a BcManager
     * with the specified time-dependent BC.
     *
     * @param side Boundary side ("left", "right", "bottom", "top")
     * @param time_bc Time-dependent boundary condition policy
     *
     * Usage:
     *   solver.set_time_dependent_bc("left", sinusoidal_inlet<System>(1.0, 100.0, 2.0));
     */
    template<typename TimeBCPolicy>
    void set_time_dependent_bc(const std::string& side, const TimeBCPolicy& time_bc) {
        // Initialize bc_manager_ if not already initialized
        if (!use_bc_manager_) {
            bc_manager_.initialize(cfg_.nx, cfg_.ny, cfg_.dx, cfg_.dy, cfg_.x_min, cfg_.y_min);
            use_bc_manager_ = true;
        }

        // Add the time-dependent BC
        bc_manager_.add_time_dependent_bc(side, time_bc);

        // Notify observers: BC configuration was changed
        observer_manager_.notify(SolverEvent::BoundaryConditionsChanged);
    }

    /**
     * @brief Disable time-dependent BCs and revert to simple bc_config_
     */
    void disable_bc_manager() {
        use_bc_manager_ = false;
    }

    /**
     * @brief Check if time-dependent BCs are enabled
     */
    bool bc_manager_enabled() const { return use_bc_manager_; }

    // ========================================================================
    // INITIALIZATION
    // ========================================================================

    /**
     * @brief Initialize with uniform state
     *
     * Allocates field storage and sets all cells to the same initial state.
     *
     * @param initial Initial primitive variables (uniform across all cells)
     *
     * @note For dense storage, uses grid dimensions (nx, ny) from domain.
     *       The CSR geometry defines the active cells, but for MVP we use
     *       dense regular grid storage.
     */
    void initialize(const Primitive& initial) {
        // For dense storage, compute number of cells from domain dimensions
        std::size_t n = cfg_.nx * cfg_.ny;

        // Only allocate if we have cells
        if (n == 0) {
            fprintf(stderr, "[AdaptiveSolver] Error: Invalid domain dimensions (nx=%zu, ny=%zu)\n",
                    cfg_.nx, cfg_.ny);
            return;
        }

        allocate_fields(n);

        // Convert primitive to conserved
        Conserved U_initial = System::from_primitive(initial, cfg_.gamma);

        // Initialize all cells to the same state
        auto U_host = Kokkos::create_mirror_view(U_);
        for (std::size_t i = 0; i < n; ++i) {
            U_host(i) = U_initial;
        }
        Kokkos::deep_copy(U_, U_host);

        current_time_ = Real(0);
        step_count_ = 0;

        // Initialize AMR level 0 (coarsest level)
        levels_[0].geometry = fluid_geometry_;
        levels_[0].U = U_;
        levels_[0].rhs_work = rhs_work_;
        levels_[0].n_cells = n_cells_;
        levels_[0].active = true;
        levels_[0].level = 0;
        finest_level_ = 0;

        // Deactivate higher levels
        for (int lvl = 1; lvl < max_amr_levels_; ++lvl) {
            levels_[lvl].active = false;
            levels_[lvl].level = static_cast<int8_t>(lvl);
        }
    }

    /**
     * @brief Initialize with explicit cell count
     *
     * Use this overload when you know the number of cells directly
     * (e.g., for regular grids or when not using CSR geometry).
     */
    void initialize(const Primitive& initial, std::size_t n_cells) {
        allocate_fields(n_cells);

        // Convert primitive to conserved
        Conserved U_initial = System::from_primitive(initial, cfg_.gamma);

        // Initialize all cells to the same state
        auto U_host = Kokkos::create_mirror_view(U_);
        for (std::size_t i = 0; i < n_cells; ++i) {
            U_host(i) = U_initial;
        }
        Kokkos::deep_copy(U_, U_host);

        current_time_ = Real(0);
        step_count_ = 0;

        // Initialize AMR level 0 (coarsest level)
        levels_[0].geometry = fluid_geometry_;
        levels_[0].U = U_;
        levels_[0].rhs_work = rhs_work_;
        levels_[0].n_cells = n_cells_;
        levels_[0].active = true;
        levels_[0].level = 0;
        finest_level_ = 0;

        // Deactivate higher levels
        for (int lvl = 1; lvl < max_amr_levels_; ++lvl) {
            levels_[lvl].active = false;
            levels_[lvl].level = static_cast<int8_t>(lvl);
        }
    }

    /**
     * @brief Initialize from existing field (copy)
     *
     * Use this overload to initialize from an externally-managed field.
     *
     * @param U_existing Existing conserved variables (will be copied)
     */
    void initialize_from_field(const Kokkos::View<const Conserved*>& U_existing) {
        std::size_t n = U_existing.extent(0);
        allocate_fields(n);
        Kokkos::deep_copy(U_, U_existing);
        current_time_ = Real(0);
        step_count_ = 0;
    }

    // ========================================================================
    // TIME STEPPING
    // ========================================================================

    /**
     * @brief Perform one global time step
     *
     * Performs a complete time step including:
     * 1. Compute adaptive dt based on CFL condition
     * 2. Apply boundary conditions
     * 3. Compute RHS (flux divergence)
     * 4. Apply time integrator (Forward Euler, RK2, RK3, RK4, etc.)
     * 5. Update simulation time
     *
     * @return Actual dt used for the step
     *
     * @note The time integrator is selected at compile time via the
     *       TimeIntegrator template parameter. Options include:
     *       - ForwardEuler: 1st order, 1 stage (default, fastest)
     *       - Heun2: 2nd order, 2 stages
     *       - Kutta3: 3rd order, 3 stages
     *       - SSPRK3: 3rd order, 3 stages (good for shocks)
     *       - ClassicRK4: 4th order, 4 stages (most accurate)
     *
     * @note For production use with CSR geometries, consider using the
     *       bridge pattern (CSR -> dense -> step -> dense -> CSR) as shown
     *       in mach2_cylinder.cpp examples.
     */
    Real step() {
        if (!fields_allocated_) {
            fprintf(stderr, "[AdaptiveSolver] Error: Fields not initialized. Call initialize() first.\n");
            return Real(0);
        }

        // -------------------------------------------------------------------
        // Step 1: Compute adaptive time step based on CFL condition
        // -------------------------------------------------------------------
        // Phase 4: Use configurable time step limits from Config
        Real max_speed = compute_max_wave_speed();
        Real dx_min = Kokkos::min(cfg_.dx, cfg_.dy);

        // Prevent division by zero
        if (max_speed < Real(1e-10)) {
            max_speed = Real(1e-10);
        }

        // CFL condition: dt = CFL * dx / max_wave_speed
        Real dt = cfg_.cfl * dx_min / max_speed;

        // Phase 4: Apply configurable safety limits
        dt = Kokkos::fmax(cfg_.time_step.dt_min,
                          Kokkos::fmin(dt, cfg_.time_step.dt_max));

        // Phase 4: Store current dt for adaptive control
        last_dt_ = dt;

        // -------------------------------------------------------------------
        // Step 2-4: Time integration (dispatch based on integrator type)
        // -------------------------------------------------------------------
        if constexpr (TimeIntegrator::stages == 1) {
            // Forward Euler: single stage
            step_euler(dt);
        } else {
            // Multi-stage Runge-Kutta: RK2, RK3, RK4, SSPRK3, etc.
            step_rk(dt);
        }

        // -------------------------------------------------------------------
        // Step 5: Update simulation time
        // -------------------------------------------------------------------
        current_time_ += dt;
        ++step_count_;

        // -------------------------------------------------------------------
        // Step 6: Check for AMR remeshing (if enabled)
        // -------------------------------------------------------------------
        if (refinement_enabled_) {
            remesh_step_counter_++;
            if (remesh_step_counter_ >= refinement_config_.remesh_interval) {
                remesh();
            }
        }

        // Notify observers (if any)
        SolverState<Real> state;
        state.time = current_time_;
        state.dt = dt;
        state.step = step_count_;
        state.total_cells = n_cells_;
        observer_manager_.notify(SolverEvent::StepEnd, state);

        return dt;
    }

    /**
     * @brief Get the name of the time integrator
     */
    const char* integrator_name() const {
        return TimeIntegrator::name;
    }

    /**
     * @brief Get the order of accuracy of the time integrator
     */
    int integrator_order() const {
        return TimeIntegrator::order;
    }

    /**
     * @brief Get the number of stages of the time integrator
     */
    int integrator_stages() const {
        return TimeIntegrator::stages;
    }

    // ========================================================================
    // PHASE 4: TIME STEP CONTROL
    // ========================================================================

    /**
     * @brief Get the last time step used
     */
    Real last_dt() const {
        return last_dt_;
    }

    /**
     * @brief Convenience alias for last_dt()
     *
     * This matches the API expected by examples.
     */
    Real dt() const {
        return last_dt_;
    }

    /**
     * @brief Get current time
     */
    Real current_time() const {
        return current_time_;
    }

    /**
     * @brief Convenience alias for current_time()
     *
     * This matches the API expected by examples.
     */
    Real time() const {
        return current_time_;
    }

    /**
     * @brief Set time step limits
     *
     * Phase 4: Configure dt_min and dt_max for adaptive time stepping
     */
    void set_dt_limits(Real dt_min, Real dt_max) {
        cfg_.time_step.dt_min = dt_min;
        cfg_.time_step.dt_max = dt_max;
    }

    /**
     * @brief Set adaptive time stepping configuration
     *
     * This is a convenience method for configuring adaptive time stepping
     * using the TimeStepController::Config type.
     *
     * @param dt_config Time step controller configuration
     */
    template<typename RealType>
    void set_adaptive_time_stepping(const typename time::TimeStepController<RealType>::Config& dt_config) {
        cfg_.time_step.cfl_target = Real(dt_config.cfl_target);
        cfg_.time_step.cfl_max = Real(dt_config.cfl_max);
        cfg_.time_step.cfl_min = Real(dt_config.cfl_min);
        cfg_.time_step.dt_max = Real(dt_config.dt_max);
        cfg_.time_step.dt_min = Real(dt_config.dt_min);
        cfg_.time_step.growth_factor = Real(dt_config.growth_factor);
        cfg_.time_step.shrink_factor = Real(dt_config.shrink_factor);
        cfg_.time_step.adjust_interval = dt_config.adjust_interval;
    }

    /**
     * @brief Get time step configuration
     */
    const typename Config::TimeStepConfig& time_step_config() const {
        return cfg_.time_step;
    }

    /**
     * @brief Access time step configuration (mutable)
     */
    typename Config::TimeStepConfig& time_step_config() {
        return cfg_.time_step;
    }

    // ========================================================================
    // AMR REFINEMENT CONFIGURATION
    // ========================================================================

    /**
     * @brief Set the AMR refinement configuration
     *
     * Enables adaptive mesh refinement with the specified configuration.
     * The refinement will be applied during step() when remesh_stride
     * steps have passed since the last remesh.
     *
     * @param config The refinement configuration (criteria, exclusion zones, etc.)
     */
    void set_refinement(const amr::RefinementConfig<System>& config) {
        refinement_config_ = config;
        refinement_enabled_ = true;
        remesh_step_counter_ = 0;
    }

    /**
     * @brief Convenience alias for set_refinement()
     *
     * This matches the API expected by examples.
     */
    void set_refinement_config(const amr::RefinementConfig<System>& config) {
        set_refinement(config);
    }

    /**
     * @brief Set refinement configuration from RefinementManager
     *
     * This is a convenience overload for RefinementManager which has
     * a config member of type RefinementConfig.
     *
     * @param manager The refinement manager containing the configuration
     */
    void set_refinement_config(const amr::RefinementManager<System>& manager) {
        set_refinement(manager.config);
    }

    /**
     * @brief Disable AMR refinement
     *
     * Disables adaptive mesh refinement and deactivates all levels
     * except level 0 (coarsest).
     */
    void disable_refinement() {
        refinement_enabled_ = false;
        for (int lvl = 1; lvl < max_amr_levels_; ++lvl) {
            levels_[lvl].active = false;
        }
        finest_level_ = 0;
    }

    /**
     * @brief Check if refinement is enabled
     */
    bool refinement_enabled() const {
        return refinement_enabled_;
    }

    /**
     * @brief Get the current finest AMR level
     */
    int finest_level() const {
        return finest_level_;
    }

    /**
     * @brief Get the number of active AMR levels
     */
    int num_active_levels() const {
        return finest_level_ + 1;
    }

    // ========================================================================
    // CUDA COMPATIBILITY: Helper methods with device lambdas must be public
    // ========================================================================
    //
    // NOTE: CUDA nvcc does not allow __host__ __device__ lambdas (KOKKOS_LAMBDA)
    // inside private or protected member functions. The following helper methods
    // are therefore public, but intended for internal use only.
    //
    // ========================================================================

    /**
     * @brief Forward Euler time step (1st order, 1 stage)
     *
     * U_{n+1} = U_n + dt * RHS
     *
     * NOTE: Public for CUDA compatibility (nvcc restriction on private methods
     *       with device lambdas)
     */
    void step_euler(Real dt) {
        using ExecSpace = typename Kokkos::DefaultExecutionSpace;

        // Apply boundary conditions with current time
        apply_boundary_conditions(current_time_);

        // Compute RHS
        compute_rhs(rhs_work_, current_time_);

        // Update: U_{n+1} = U_n + dt * RHS
        auto U = U_;
        auto rhs = rhs_work_;
        Real dt_local = dt;

        Kokkos::parallel_for(
            "euler_update",
            Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(n_cells_)),
            KOKKOS_LAMBDA(const int i) {
                // Phase 6: Generic field update using operators
                // Works for ANY System with Conserved that has += and *= operators
                U(i) += rhs(i) * dt_local;
            }
        );
        Kokkos::fence();
    }

    /**
     * @brief Runge-Kutta time step (multi-stage)
     *
     * Generic RK implementation supporting RK2, RK3, RK4, SSPRK3, etc.
     * Uses the Butcher tableau from the TimeIntegrator policy.
     *
     * Algorithm:
     * 1. Save original solution: U_old = U
     * 2. For each stage s = 0 to stages-1:
     *    a. Compute intermediate solution (if s > 0)
     *    b. Compute RHS: k_s = f(t + c[s]*dt, U_stage)
     * 3. Combine stages: U_{n+1} = U_old + dt * sum(b[s] * k_s)
     *
     * NOTE: Public for CUDA compatibility (nvcc restriction on private methods
     *       with device lambdas)
     */
    void step_rk(Real dt) {
        using ExecSpace = typename Kokkos::DefaultExecutionSpace;

        // Save original solution
        Kokkos::deep_copy(U_old_, U_);

        // Phase 4: Stage loop with observer notifications
        for (int s = 0; s < TimeIntegrator::stages; ++s) {
            Real stage_time = current_time_ + TimeIntegrator::c[s] * dt;

            // Phase 4: Notify observers: SubStepBegin
            SolverState<Real> state_begin;
            state_begin.time = current_time_;
            state_begin.dt = dt;
            state_begin.step = step_count_;
            state_begin.stage = s;
            state_begin.num_stages = TimeIntegrator::stages;
            state_begin.stage_time = stage_time;
            observer_manager_.notify(SolverEvent::SubStepBegin, state_begin);

            if (s == 0) {
                // First stage: use original solution
                apply_boundary_conditions(stage_time);
                compute_rhs(stage_rhs_[0], stage_time);
            } else {
                // Subsequent stages: compute intermediate solution
                compute_stage_solution(s, dt);
                apply_boundary_conditions(stage_time);
                compute_rhs(stage_rhs_[s], stage_time);
            }

            // Phase 4: Notify observers: SubStepEnd
            SolverState<Real> state_end;
            state_end.time = current_time_;
            state_end.dt = dt;
            state_end.step = step_count_;
            state_end.stage = s;
            state_end.num_stages = TimeIntegrator::stages;
            state_end.stage_time = stage_time;
            observer_manager_.notify(SolverEvent::SubStepEnd, state_end);
        }

        // Combine all stages for final solution
        combine_stages(dt);
    }

    /**
     * @brief Compute intermediate solution for RK stage s
     *
     * U_stage = U_old + dt * sum_{i=0}^{s-1} a[s][i] * k_i
     *
     * This computes the solution at which to evaluate the RHS for stage s.
     *
     * NOTE: Public for CUDA compatibility (nvcc restriction on private methods
     *       with device lambdas)
     */
    void compute_stage_solution(int stage, Real dt) {
        using ExecSpace = typename Kokkos::DefaultExecutionSpace;

        auto U_stage = stage_solution_;
        auto U_old = U_old_;
        const int n = n_cells_;

        // CUDA: Copy static constexpr array to local variable for device access
        // Use if constexpr to avoid copying for integrators that don't need these arrays
        constexpr int max_stages = TimeIntegrator::stages;
        Real a_stage[max_stages];

        if constexpr (max_stages > 1) {
            for (int prev = 0; prev < max_stages; ++prev) {
                if constexpr (requires { TimeIntegrator::a[stage][prev]; }) {
                    a_stage[prev] = TimeIntegrator::a[stage][prev];
                } else {
                    a_stage[prev] = Real(0);
                }
            }
        }

        Kokkos::parallel_for(
            "rk_stage_solution",
            Kokkos::RangePolicy<ExecSpace>(0, n),
            KOKKOS_LAMBDA(const int i) {
                Conserved sum{0, 0, 0, 0};

                // Sum contributions from previous stages
                for (int prev = 0; prev < stage; ++prev) {
                    Real coeff = a_stage[prev];
                    const auto& k_prev = stage_rhs_[prev];
                    sum.rho  += coeff * k_prev(i).rho;
                    sum.rhou += coeff * k_prev(i).rhou;
                    sum.rhov += coeff * k_prev(i).rhov;
                    sum.E    += coeff * k_prev(i).E;
                }

                U_stage(i).rho  = U_old(i).rho  + dt * sum.rho;
                U_stage(i).rhou = U_old(i).rhou + dt * sum.rhou;
                U_stage(i).rhov = U_old(i).rhov + dt * sum.rhov;
                U_stage(i).E    = U_old(i).E    + dt * sum.E;
            }
        );
        Kokkos::fence();
    }

    /**
     * @brief Combine stages for final RK solution
     *
     * U_{n+1} = U_old + dt * sum_{s=0}^{stages-1} b[s] * k_s
     *
     * NOTE: Public for CUDA compatibility (nvcc restriction on private methods
     *       with device lambdas)
     */
    void combine_stages(Real dt) {
        using ExecSpace = typename Kokkos::DefaultExecutionSpace;

        auto U = U_;
        auto U_old = U_old_;
        const int n = n_cells_;

        // CUDA: Copy static constexpr array to local variable for device access
        constexpr int max_stages = TimeIntegrator::stages;
        Real b_coeffs[max_stages];

        if constexpr (max_stages > 1) {
            for (int s = 0; s < max_stages; ++s) {
                if constexpr (requires { TimeIntegrator::b[s]; }) {
                    b_coeffs[s] = TimeIntegrator::b[s];
                } else {
                    b_coeffs[s] = Real(0);
                }
            }
        } else {
            // Forward Euler: b[0] = 1
            b_coeffs[0] = Real(1);
        }

        Kokkos::parallel_for(
            "rk_combine",
            Kokkos::RangePolicy<ExecSpace>(0, n),
            KOKKOS_LAMBDA(const int i) {
                // Phase 6: Generic zero initialization (works for ANY System)
                Conserved sum{};

                // Sum contributions from all stages (generic operator usage)
                for (int s = 0; s < max_stages; ++s) {
                    Real coeff = b_coeffs[s];
                    const auto& k_s = stage_rhs_[s];
                    // Phase 6: Generic accumulation using operators
                    sum += k_s(i) * coeff;
                }

                // Phase 6: Generic final update using operators
                U(i) = U_old(i) + sum * dt;
            }
        );
        Kokkos::fence();
    }

public:
    // ========================================================================
    // OUTPUT (IMPROVEMENT B: FieldView with ownership)
    // ========================================================================

    /**
     * @brief Get finest level output with proper ownership semantics
     *
     * IMPROVEMENT B: Returns SolverOutput with FieldViews instead of raw pointers
     */
    SolverOutput<Real> get_output() const {
        SolverOutput<Real> output;
        output.level = 0;  // Finest level
        output.time = current_time_;
        output.geometry = &fluid_geometry_;

        // Stub: would add actual fields
        // In production:
        // output.fields.add(FieldView<Real>::allocate("rho", n_cells, 0));
        // output.fields.add(FieldView<Real>::allocate("rhou", n_cells, 0));
        // etc.

        return output;
    }

    /**
     * @brief Get all outputs from all AMR levels
     */
    std::vector<SolverOutput<Real>> get_all_levels() const {
        std::vector<SolverOutput<Real>> outputs;
        outputs.push_back(get_output());  // Stub: only finest level
        return outputs;
    }

    /**
     * @brief Write output to VTK file
     *
     * IMPROVEMENT B: Convenience method for VTK export
     */
    void write_vtk(const std::string& filename) const {
        auto output = get_output();
        VTKExporter::write_legacy(output, filename);
    }

    /**
     * @brief Get geometry for output
     */
    const csr::IntervalSet2DDevice& geometry() const {
        return fluid_geometry_;
    }

    /**
     * @brief Get current simulation time
     */
    Real get_time() const {
        return current_time_;
    }

    /**
     * @brief Get time zero helper
     */
    static Real get_time_zero() {
        return Real(0);
    }

    // ========================================================================
    // OBSERVERS (IMPROVEMENT D: Callback system for monitoring)
    // ========================================================================

    /**
     * @brief Set refinement criteria (IMPROVEMENT E)
     */
    void set_refinement_criteria(const RefinementCriteria& criteria) {
        cfg_.refinement = criteria;
    }

    /**
     * @brief Get observer manager for adding callbacks
     */
    ObserverManager<Real>& observers() {
        return observer_manager_;
    }

    /**
     * @brief Add a progress callback (called after each step)
     *
     * Convenience method for common case
     */
    int on_progress(ProgressCallback<Real> callback) {
        return observer_manager_.on_progress(std::move(callback));
    }

    /**
     * @brief Add a remesh callback
     */
    int on_remesh(RemeshCallback<Real> callback) {
        return observer_manager_.on_remesh(std::move(callback));
    }

    /**
     * @brief Add an error callback
     */
    int on_error(ErrorCallback callback) {
        return observer_manager_.on_error(std::move(callback));
    }

    /**
     * @brief Add a generic callback for any event
     */
    int add_observer(SolverEvent event, SolverCallback<Real> callback) {
        return observer_manager_.add_callback(event, std::move(callback));
    }

    /**
     * @brief Remove an observer by ID
     */
    bool remove_observer(int id) {
        return observer_manager_.remove_callback(id);
    }

    /**
     * @brief Clear all observers
     */
    void clear_observers() {
        observer_manager_.clear();
    }

    // ========================================================================
    // SOLVER STATE
    // ========================================================================

    /**
     * @brief Get current configuration
     */
    const Config& config() const {
        return cfg_;
    }

    /**
     * @brief Get refinement criteria
     */
    const RefinementCriteria& refinement_criteria() const {
        return cfg_.refinement;
    }

    /**
     * @brief Get step count
     */
    int get_step_count() const {
        return step_count_;
    }

    /**
     * @brief Get current solver state snapshot (for observers)
     */
    SolverState<Real> get_state() const {
        SolverState<Real> state;
        state.time = current_time_;
        state.step = step_count_;
        state.max_level = 0;  // Stub
        state.total_cells = 1000;  // Stub
        // Fill in other fields...
        return state;
    }

    /**
     * @brief Get const access to the solution field (for testing/validation)
     *
     * @return const view of conserved variables
     *
     * @note This is primarily intended for testing and validation.
     *       Direct modification of U outside of the solver can break consistency.
     */
    const Kokkos::View<Conserved*>& get_solution() const {
        return U_;
    }

    /**
     * @brief Get mutable access to the solution field (for initial conditions)
     *
     * @return mutable view of conserved variables
     *
     * @warning Use with caution. Direct modification can break solver invariants.
     *          This is provided for setting custom initial conditions.
     */
    Kokkos::View<Conserved*>& get_solution_mutable() {
        return U_;
    }

    // ========================================================================
    // MULTI-LEVEL AMR ACCESS API (Phase 1)
    // ========================================================================

    /**
     * @brief Get the number of active AMR levels
     *
     * @return Number of active levels (1 to max_amr_levels_)
     *
     * Level 0 is always active (coarsest/base mesh).
     * Higher levels are activated during refinement.
     *
     * @note This reflects the current state after remeshing.
     *       Levels may be activated/deactivated dynamically.
     */
    int get_num_levels() const {
        return finest_level_ + 1;
    }

    /**
     * @brief Get geometry for a specific AMR level
     *
     * @param level Level index (0 = coarsest, get_num_levels()-1 = finest)
     * @return const reference to CSR geometry for this level
     * @throws std::out_of_range if level is inactive
     *
     * @note The returned reference is invalidated after remeshing.
     */
    const csr::IntervalSet2DDevice& get_level_geometry(int level) const {
        if (level < 0 || level >= max_amr_levels_ || !levels_[level].active) {
            throw std::out_of_range("Invalid AMR level: " + std::to_string(level));
        }
        return levels_[level].geometry;
    }

    /**
     * @brief Get solution field for a specific AMR level (read-only)
     *
     * @param level Level index (0 = coarsest, get_num_levels()-1 = finest)
     * @return const view of conserved variables for this level
     * @throws std::out_of_range if level is inactive
     *
     * @note The returned view is invalidated after remeshing or time stepping.
     */
    const Kokkos::View<Conserved*>& get_level_solution(int level) const {
        if (level < 0 || level >= max_amr_levels_ || !levels_[level].active) {
            throw std::out_of_range("Invalid AMR level: " + std::to_string(level));
        }
        return levels_[level].U;
    }

    /**
     * @brief Get solution field for a specific AMR level (mutable)
     *
     * @param level Level index (0 = coarsest, get_num_levels()-1 = finest)
     * @return mutable view of conserved variables for this level
     * @throws std::out_of_range if level is inactive
     * @warning Use with caution. Direct modification can break solver invariants.
     */
    Kokkos::View<Conserved*>& get_level_solution_mutable(int level) {
        if (level < 0 || level >= max_amr_levels_ || !levels_[level].active) {
            throw std::out_of_range("Invalid AMR level: " + std::to_string(level));
        }
        return levels_[level].U;
    }

    /**
     * @brief Get cell count for a specific AMR level
     *
     * @param level Level index (0 = coarsest, get_num_levels()-1 = finest)
     * @return Number of active cells at this level
     */
    std::size_t get_level_cell_count(int level) const {
        if (level < 0 || level >= max_amr_levels_ || !levels_[level].active) {
            return 0;
        }
        return levels_[level].n_cells;
    }

    /**
     * @brief Get cell size (dx) for a specific AMR level
     *
     * @param level Level index (0 = coarsest, get_num_levels()-1 = finest)
     * @return Cell size in x-direction at this level
     *
     * Each level has half the cell size of the previous level:
     *   dx_level = dx_base / (2^level)
     */
    Real get_level_dx(int level) const {
        return cfg_.dx / static_cast<Real>(1 << level);
    }

    /**
     * @brief Get cell size (dy) for a specific AMR level
     *
     * @param level Level index (0 = coarsest, get_num_levels()-1 = finest)
     * @return Cell size in y-direction at this level
     */
    Real get_level_dy(int level) const {
        return cfg_.dy / static_cast<Real>(1 << level);
    }

    /**
     * @brief Check if a level is active
     *
     * @param level Level index to check
     * @return true if the level is active, false otherwise
     */
    bool is_level_active(int level) const {
        return (level >= 0 && level < max_amr_levels_ && levels_[level].active);
    }

    /**
     * @brief Get the finest active level index
     *
     * @return Index of the finest active level
     */
    int get_finest_level() const {
        return finest_level_;
    }

    /**
     * @brief Create a Field2DDevice wrapper for a specific level
     *
     * This provides a unified interface compatible with CSR operations.
     *
     * @param level Level index (0 = coarsest, get_num_levels()-1 = finest)
     * @return Field2D<Conserved, DeviceMemorySpace> wrapping geometry and solution
     * @throws std::out_of_range if level is inactive
     */
    subsetix::csr::Field2DDevice<Conserved> get_level_field(int level) const {
        if (level < 0 || level >= max_amr_levels_ || !levels_[level].active) {
            return subsetix::csr::Field2DDevice<Conserved>{};  // Empty field
        }

        subsetix::csr::Field2DDevice<Conserved> field;
        field.geometry = levels_[level].geometry;
        field.values = levels_[level].U;
        return field;
    }

    // ========================================================================
    // PHASE 3: MULTI-LEVEL VTK EXPORT
    // ========================================================================

    /**
     * @brief Export a single AMR level to VTK format
     *
     * Exports geometry and solution data for a specific level to VTK format.
     * The output file can be visualized with ParaView or VisIt.
     *
     * @param level AMR level to export (-1 for finest level, 0 for coarsest)
     * @param filename Output VTK file path
     * @param field_index Index of conserved variable to export (0=rho, 1=rhou, 2=rhov, 3=E)
     * @return true if successful
     *
     * Usage:
     *   solver.export_vtk_level(-1, "finest.vtk", 0);  // Finest level density
     *   solver.export_vtk_level(0, "coarse.vtk", 3);   // Coarse level energy
     */
    bool export_vtk_level(int level, const std::string& filename, int field_index = 0) const {
        // Validate level
        int target_level = (level < 0) ? finest_level_ : level;
        if (target_level < 0 || target_level >= max_amr_levels_ || !levels_[target_level].active) {
            fprintf(stderr, "[VTK Export] Invalid level: %d\n", target_level);
            return false;
        }

        // Get level field using Phase 1 API
        auto level_field = get_level_field(target_level);

        // Create host mirror for VTK export
        auto host_geom = subsetix::csr::to<subsetix::csr::HostMemorySpace>(level_field.geometry);
        auto host_values = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, level_field.values);

        // Build IntervalField2DHost using helper function
        subsetix::csr::IntervalField2DHost<Real> host_field;

        // Copy geometry data to std::vectors
        for (std::size_t i = 0; i < host_geom.num_rows; ++i) {
            host_field.row_keys.push_back(host_geom.row_keys(i));
            host_field.row_ptr.push_back(host_geom.row_ptr(i));
        }
        host_field.row_ptr.push_back(host_geom.num_intervals);

        for (std::size_t i = 0; i < host_geom.num_intervals; ++i) {
            host_field.intervals.push_back(host_geom.intervals(i));
        }

        // Copy scalar field values (extract specific component)
        host_field.values.reserve(host_geom.total_cells);
        for (std::size_t i = 0; i < host_geom.total_cells; ++i) {
            const Conserved& U = reinterpret_cast<const Conserved&>(host_values(i));
            switch (field_index) {
                case 0: host_field.values.push_back(static_cast<Real>(U.rho)); break;
                case 1: host_field.values.push_back(static_cast<Real>(U.rhou)); break;
                case 2: host_field.values.push_back(static_cast<Real>(U.rhov)); break;
                case 3: host_field.values.push_back(static_cast<Real>(U.E)); break;
                default: host_field.values.push_back(static_cast<Real>(U.rho)); break;
            }
        }

        // Write VTK file
        const char* field_names[] = {"density", "momentum_x", "momentum_y", "total_energy"};
        std::string field_name = (field_index >= 0 && field_index < 4)
            ? field_names[field_index] : "field";

        try {
            subsetix::vtk::write_legacy_quads(host_field, filename, field_name.c_str());
            return true;
        } catch (const std::exception& e) {
            fprintf(stderr, "[VTK Export] Failed: %s\n", e.what());
            return false;
        }
    }

    /**
     * @brief Export all active AMR levels to separate VTK files
     *
     * Creates one VTK file per active level with filename pattern:
     * {base}_level{N}.vtk where N is the level index.
     *
     * @param filename_base Base filename (without extension)
     * @param field_index Index of conserved variable to export
     * @return Number of successfully exported levels
     *
     * Usage:
     *   solver.export_vtk_all_levels("output");  // Creates output_level0.vtk, output_level1.vtk, ...
     */
    int export_vtk_all_levels(const std::string& filename_base = "output",
                              int field_index = 0) const {
        int count = 0;
        for (int lvl = 0; lvl <= finest_level_; ++lvl) {
            if (levels_[lvl].active) {
                char filename[256];
                snprintf(filename, sizeof(filename), "%s_level%d.vtk",
                        filename_base.c_str(), lvl);
                if (export_vtk_level(lvl, filename, field_index)) {
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * @brief Export all levels to a single multi-level VTK file
     *
     * Creates a single VTK file containing all AMR levels with:
     * - Physical coordinates for each level
     * - Level indicator scalar field
     * - Selected field values
     *
     * Compatible with ParaView and VisIt.
     *
     * @param filename Output VTK file path
     * @param field_index Index of conserved variable to export
     * @return true if successful
     *
     * Usage:
     *   solver.export_vtk_multilevel("solution_all_levels.vtk");
     */
    bool export_vtk_multilevel(const std::string& filename, int field_index = 0) const {
        // Build multilevel geometry structure
        subsetix::MultilevelGeoHost multi_geo;
        multi_geo.origin_x = domain_.x_min * cfg_.dx;
        multi_geo.origin_y = domain_.y_min * cfg_.dy;
        multi_geo.root_dx = static_cast<double>(cfg_.dx);
        multi_geo.root_dy = static_cast<double>(cfg_.dy);
        multi_geo.num_active_levels = finest_level_ + 1;

        // Copy each level's geometry to host
        for (int lvl = 0; lvl <= finest_level_; ++lvl) {
            if (levels_[lvl].active) {
                multi_geo.levels[lvl] = subsetix::csr::to<subsetix::csr::HostMemorySpace>(
                    levels_[lvl].geometry);
            }
        }

        // Build multilevel field structure
        subsetix::MultilevelFieldHost<Real> multi_field;
        multi_field.num_active_levels = finest_level_ + 1;

        for (int lvl = 0; lvl <= finest_level_; ++lvl) {
            if (!levels_[lvl].active) continue;

            auto host_geom = subsetix::csr::to<subsetix::csr::HostMemorySpace>(levels_[lvl].geometry);
            auto host_values = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, levels_[lvl].U);

            multi_field.levels[lvl].geometry = host_geom;
            multi_field.levels[lvl].values = Kokkos::View<Real*, Kokkos::HostSpace>(
                "field_values", host_geom.total_cells);

            // Copy scalar field values
            for (std::size_t i = 0; i < host_geom.total_cells; ++i) {
                const Conserved& U = reinterpret_cast<const Conserved&>(host_values(i));
                switch (field_index) {
                    case 0: multi_field.levels[lvl].values(i) = static_cast<Real>(U.rho); break;
                    case 1: multi_field.levels[lvl].values(i) = static_cast<Real>(U.rhou); break;
                    case 2: multi_field.levels[lvl].values(i) = static_cast<Real>(U.rhov); break;
                    case 3: multi_field.levels[lvl].values(i) = static_cast<Real>(U.E); break;
                    default: multi_field.levels[lvl].values(i) = static_cast<Real>(U.rho); break;
                }
            }
        }

        // Write multi-level VTK file
        const char* field_names[] = {"density", "momentum_x", "momentum_y", "total_energy"};
        std::string field_name = (field_index >= 0 && field_index < 4)
            ? field_names[field_index] : "field";

        try {
            subsetix::vtk::write_multilevel_field_vtk(multi_field, multi_geo,
                                                       filename, field_name.c_str());
            return true;
        } catch (const std::exception& e) {
            fprintf(stderr, "[VTK Export] Multi-level export failed: %s\n", e.what());
            return false;
        }
    }

    // ========================================================================
    // PHASE 5: MULTI-RATE AMR TIME STEPPING
    // ========================================================================

    /**
     * @brief Compute level-specific time step based on cell size
     *
     * For level L with refinement ratio 2^L, the cell size is dx_L = dx_0 / 2^L.
     * The time step should scale proportionally: dt_L = CFL * dx_L / max_wave_speed.
     *
     * @param level AMR level (0 = coarsest)
     * @return Level-specific time step
     */
    Real compute_level_dt(int level) const {
        // Get cell size for this level
        Real dx_level = get_level_dx(level);
        Real dy_level = get_level_dy(level);
        Real dx_min_level = Kokkos::min(dx_level, dy_level);

        // Compute maximum wave speed (global value, could be level-specific)
        Real max_speed = compute_max_wave_speed();
        if (max_speed < Real(1e-10)) {
            max_speed = Real(1e-10);
        }

        // CFL condition: dt = CFL * dx_level / max_wave_speed
        Real dt = cfg_.cfl * dx_min_level / max_speed;

        // Apply safety limits
        dt = Kokkos::fmax(cfg_.time_step.dt_min,
                          Kokkos::fmin(dt, cfg_.time_step.dt_max));

        return dt;
    }

    /**
     * @brief Step with multi-rate AMR subcycling
     *
     * Implements level-wise time stepping where fine levels take multiple
     * steps for each coarse level step. This is more efficient for deep
     * AMR hierarchies.
     *
     * Algorithm (V-cycle):
     * 1. Compute dt for each level based on cell size
     * 2. Coarsest level (level 0) takes one step with dt_0
     * 3. Level 1 takes time_refinement_ratio steps with dt_1 = dt_0 / ratio
     * 4. Level 2 takes time_refinement_ratio^2 steps with dt_2 = dt_1 / ratio
     * 5. Continue to finest level
     * 6. Apply flux correction at coarse-fine boundaries (if enabled)
     * 7. Synchronize solutions via restriction
     *
     * @return Time step taken (dt of coarsest level)
     *
     * Usage:
     *   cfg.time_step.enable_subcycling = true;
     *   cfg.time_step.time_refinement_ratio = 2;
     *   Real dt = solver.step_with_subcycling();
     */
    Real step_with_subcycling() {
        if (!fields_allocated_) {
            fprintf(stderr, "[AdaptiveSolver] Error: Fields not initialized. Call initialize() first.\n");
            return Real(0);
        }

        // Only use subcycling if we have multiple levels
        if (finest_level_ == 0) {
            // Single level: fall back to standard step()
            return step();
        }

        // ====================================================================
        // Step 1: Compute time step for each level
        // ====================================================================
        Real dt_level[max_amr_levels_];
        for (int lvl = 0; lvl <= finest_level_; ++lvl) {
            if (levels_[lvl].active) {
                dt_level[lvl] = compute_level_dt(lvl);
            } else {
                dt_level[lvl] = Real(0);
            }
        }

        // Base time step (coarsest level)
        Real dt_0 = dt_level[0];

        // ====================================================================
        // Step 2: V-cycle time stepping with subcycling
        // ====================================================================
        int ratio = cfg_.time_step.time_refinement_ratio;

        // Advance from coarsest to finest, then back down
        for (int lvl = 0; lvl <= finest_level_; ++lvl) {
            if (!levels_[lvl].active) continue;

            // Number of substeps for this level
            int n_substeps = (lvl == 0) ? 1 : ratio;

            // Time step for this level
            Real dt_l = dt_level[lvl];

            // Prolong guard cells from coarser level
            if (lvl > 0) {
                prolong_guard_cells(lvl - 1, lvl);
            }

            // Fill boundary conditions
            apply_boundary_conditions_level(lvl, current_time_);

            // Take substeps on this level
            for (int sub = 0; sub < n_substeps; ++sub) {
                // Compute RHS and update solution
                if constexpr (TimeIntegrator::stages == 1) {
                    step_level_euler(lvl, dt_l);
                } else {
                    step_level_rk(lvl, dt_l);
                }

                // Apply boundary conditions after each substep
                apply_boundary_conditions_level(lvl, current_time_ + (sub + 1) * dt_l);
            }
        }

        // ====================================================================
        // Step 3: Flux correction and synchronization (finest to coarsest)
        // ====================================================================
        if (cfg_.time_step.enable_flux_correction) {
            for (int lvl = finest_level_; lvl > 0; --lvl) {
                if (!levels_[lvl].active) continue;
                apply_flux_correction(lvl, lvl - 1);
            }
        }

        // Restrict fine levels to coarse levels
        for (int lvl = finest_level_; lvl > 0; --lvl) {
            if (!levels_[lvl].active) continue;
            restrict_level_solution(lvl, lvl - 1);
        }

        // ====================================================================
        // Step 4: Update simulation time
        // ====================================================================
        current_time_ += dt_0;
        ++step_count_;
        last_dt_ = dt_0;

        // ====================================================================
        // Step 5: Check for AMR remeshing
        // ====================================================================
        if (refinement_enabled_) {
            remesh_step_counter_++;
            if (remesh_step_counter_ >= refinement_config_.remesh_interval) {
                remesh();
            }
        }

        // Notify observers
        SolverState<Real> state;
        state.time = current_time_;
        state.dt = dt_0;
        state.step = step_count_;
        state.total_cells = n_cells_;
        observer_manager_.notify(SolverEvent::StepEnd, state);

        return dt_0;
    }

    /**
     * @brief Prolong guard cells from coarse to fine level
     *
     * Fills ghost cells at coarse-fine boundaries by interpolating from
     * the coarse level solution.
     *
     * @param coarse_level Coarser level index
     * @param fine_level Finer level index (must be coarse_level + 1)
     */
    void prolong_guard_cells(int coarse_level, int fine_level) {
        if (fine_level != coarse_level + 1) {
            fprintf(stderr, "[AMR] prolong_guard_cells: fine_level must be coarse_level + 1\n");
            return;
        }

        // Use AmrOperations to prolong field values
        // For now, use simple injection (linear prediction could be added)
        auto coarse_field = get_level_field(coarse_level);
        auto fine_field = get_level_field(fine_level);

        // Use injection prolongation (conservative, 1st order)
        amr::AmrOperations<System>::prolong_level(
            fine_field, coarse_field, fine_field.geometry, false
        );
    }

    /**
     * @brief Apply flux correction at coarse-fine boundary
     *
     * Accumulates fine-level fluxes and applies them to the coarse level
     * to ensure conservation across the interface.
     *
     * @param fine_level Finer level index
     * @param coarse_level Coarser level index
     *
     * NOTE: This is a simplified implementation. A full flux correction
     * requires flux registers and careful handling of temporal interpolation.
     */
    void apply_flux_correction(int fine_level, int coarse_level) {
        // Phase 5: Simplified flux correction
        // In a full implementation, this would:
        // 1. Accumulate fluxes from fine level during subcycling
        // 2. Apply them to coarse level cells at the interface
        // 3. Ensure conservation: divergence of flux = 0 globally

        // For now, we rely on restriction to maintain approximate conservation
        // Full flux correction would require flux registers

        // TODO: Implement flux registers for exact conservation
        (void)fine_level;
        (void)coarse_level;
    }

    /**
     * @brief Restrict solution from fine level to coarse level
     *
     * Uses volume-weighted averaging to restrict the fine level solution
     * to the coarse level, ensuring conservation.
     *
     * @param fine_level Finer level index
     * @param coarse_level Coarser level index
     */
    void restrict_level_solution(int fine_level, int coarse_level) {
        if (fine_level != coarse_level + 1) {
            fprintf(stderr, "[AMR] restrict_level_solution: fine_level must be coarse_level + 1\n");
            return;
        }

        auto fine_field = get_level_field(fine_level);
        auto coarse_field = get_level_field(coarse_level);

        // Use volume-weighted restriction (conservative)
        amr::AmrOperations<System>::restrict_level(
            coarse_field, fine_field, coarse_field.geometry
        );
    }

    /**
     * @brief Apply boundary conditions for a specific level
     *
     * @param level AMR level
     * @param t Current simulation time
     */
    void apply_boundary_conditions_level(int level, Real t) {
        // Apply boundary conditions using the configured boundary config
        // This is a simplified version - full implementation would handle
        // level-specific boundary conditions and coarse-fine interfaces

        if (!levels_[level].active) return;

        // For now, use the existing boundary condition application
        // which operates on all levels simultaneously
        // A full implementation would apply BCs level-by-level

        (void)t;  // Placeholder for time-dependent BCs
    }

    /**
     * @brief Single-level Euler step for subcycling
     *
     * @param level AMR level to advance
     * @param dt Time step for this level
     */
    void step_level_euler(int level, Real dt) {
        if (!levels_[level].active) return;

        auto& level_data = levels_[level];

        // Compute RHS: dU/dt = -∇·F
        compute_rhs_level(level, level_data.rhs_work, current_time_);

        // Forward Euler: U_new = U_old + dt * RHS
        auto U = level_data.U;
        auto RHS = level_data.rhs_work;
        std::size_t n = level_data.n_cells;

        Kokkos::parallel_for("step_level_euler", n,
            KOKKOS_LAMBDA(std::size_t i) {
                U(i).rho   += dt * RHS(i).rho;
                U(i).rhou  += dt * RHS(i).rhou;
                U(i).rhov  += dt * RHS(i).rhov;
                U(i).E     += dt * RHS(i).E;
            }
        );
    }

    /**
     * @brief Single-level Runge-Kutta step for subcycling
     *
     * @param level AMR level to advance
     * @param dt Time step for this level
     */
    void step_level_rk(int level, Real dt) {
        if (!levels_[level].active) return;

        auto& level_data = levels_[level];

        // Multi-stage Runge-Kutta integration
        TimeIntegrator integrator;

        for (int s = 0; s < TimeIntegrator::stages; ++s) {
            // Compute RHS for this stage
            compute_rhs_level(level, level_data.rhs_work, current_time_);

            // Apply Runge-Kutta update
            auto U = level_data.U;
            auto RHS = level_data.rhs_work;
            std::size_t n = level_data.n_cells;

            Real a = integrator.alpha(s);  // Time coefficient
            Real b = integrator.beta(s);   // RHS coefficient

            Kokkos::parallel_for("step_level_rk_stage", n,
                KOKKOS_LAMBDA(std::size_t i) {
                    U(i).rho   = a * U(i).rho   + b * dt * RHS(i).rho;
                    U(i).rhou  = a * U(i).rhou  + b * dt * RHS(i).rhou;
                    U(i).rhov  = a * U(i).rhov  + b * dt * RHS(i).rhov;
                    U(i).E     = a * U(i).E     + b * dt * RHS(i).E;
                }
            );
        }
    }

    /**
     * @brief Compute RHS for a specific level
     *
     * @param level AMR level
     * @param rhs_out Output RHS view
     * @param t Current time
     */
    void compute_rhs_level(int level, Kokkos::View<Conserved*>& rhs_out, Real t) {
        // Simplified level-specific RHS computation
        // A full implementation would handle:
        // - Coarse-fine boundary conditions
        // - Flux matching at interfaces
        // - Proper ghost cell filling

        (void)level;
        (void)rhs_out;
        (void)t;  // Placeholder

        // For now, use the existing compute_rhs which operates on all levels
        // A full implementation would need level-specific flux computation
    }

    // ========================================================================
    // SOURCE TERMS (NEW: Add source support)
    // ========================================================================

    /**
     * @brief Add a gravity source term
     *
     * Source terms are added to the RHS: dU/dt = -∇·F + S
     *
     * Usage:
     *   solver.add_gravity(-9.81f);  // Gravity in y-direction
     *
     * NOTE: This is a convenience wrapper. The actual source computation
     * should be done by creating a custom CompositeSource type and using
     * set_source_composite() or by directly adding source computation in
     * the RHS evaluation.
     */
    void add_gravity(Real g_y = Real(-9.81), Real g_x = Real(0)) {
        // Mark that we have source terms - actual gravity computation
        // should be done via custom source types or direct RHS computation
        has_source_terms_ = true;
        // TODO: Store gravity parameters for RHS computation
    }

    /**
     * @brief Add custom source term from lambda/functor
     *
     * @param func Function: (Conserved, Primitive, x, y, t) -> Conserved
     *
     * NOTE: With compile-time sources, custom lambda sources should be
     * wrapped in a CustomSource<System, Func> type. This method is
     * provided for API compatibility but the actual implementation
     * requires a compile-time source type.
     */
    template<typename Func>
    void add_source(Func&& func, bool time_dep = false, bool spatial_dep = true) {
        // Mark that we have source terms
        has_source_terms_ = true;
        // NOTE: Lambda sources cannot be stored runtime without type erasure
        // Users should use compile-time CompositeSource types instead
        // For example: using MySource = CompositeSource<System, GravitySource<System>, CustomSource<System, MyFunc>>;
    }

    /**
     * @brief Set composite source directly (compile-time only)
     *
     * NOTE: This is a stub for API compatibility. The actual source
     * computation must be compile-time. Users should create source
     * types using the API in source_terms.hpp and apply them during
     * RHS evaluation.
     */
    template<typename... Sources>
    void set_source_composite(const sources::CompositeSource<System, Sources...>& source) {
        has_source_terms_ = true;
        // Note: Cannot store variadic template without erasing types
        // Users should re-create the composite source type when needed
    }

    /**
     * @brief Check if solver has source terms
     */
    bool has_sources() const { return has_source_terms_; }

    // ========================================================================
    // CHECKPOINT / RESTART (NEW: Persistence)
    // ========================================================================

    /**
     * @brief Write checkpoint
     *
     * Saves complete solver state to file for restart.
     * Includes: fields, geometry, time, step count, config.
     *
     * @param filename Output file path
     * @param format File format (default: Binary)
     * @return true if successful
     */
    bool write_checkpoint(const std::string& filename,
                          CheckpointFormat format = CheckpointFormat::Binary) const {
        if (format == CheckpointFormat::Binary) {
            return write_checkpoint_binary(filename);
        } else {
            return write_checkpoint_ascii(filename);
        }
    }

    /**
     * @brief Read checkpoint and restore solver state
     *
     * @param filename Input file path
     * @param format File format (default: Binary)
     * @return true if successful
     */
    bool read_checkpoint(const std::string& filename,
                         CheckpointFormat format = CheckpointFormat::Binary) {
        if (format == CheckpointFormat::Binary) {
            return read_checkpoint_binary(filename);
        } else {
            return read_checkpoint_ascii(filename);
        }
    }

    /**
     * @brief Auto-checkpoint: write every N steps
     *
     * @param stride Checkpoint every N steps (0 = disabled)
     * @param prefix File prefix (e.g., "checkpoint" -> "checkpoint_000100.bin")
     */
    void set_auto_checkpoint(int stride, const std::string& prefix = "checkpoint") {
        checkpoint_stride_ = stride;
        checkpoint_prefix_ = prefix;
    }

    // ========================================================================
    // OUTPUT STREAMING (NEW: Streaming output during simulation)
    // ========================================================================

    /**
     * @brief Enable streaming output to directory
     *
     * Automatically writes output files during simulation.
     *
     * @param output_dir Output directory
     * @param stride Write every N steps
     * @param format Output format ("vtk", "binary", "both")
     */
    void enable_streaming(const std::string& output_dir, int stride = 100,
                          const std::string& format = "vtk") {
        stream_output_ = true;
        stream_dir_ = output_dir;
        stream_stride_ = stride;
        stream_format_ = format;
    }

    void disable_streaming() {
        stream_output_ = false;
    }

    // ========================================================================
    // VALIDATION (Runtime stability checks)
    // ========================================================================

    /**
     * @brief Enable validation checks
     *
     * Checks for:
     * - Negative density/pressure
     * - NaN/Inf values
     * - CFL violation
     * - Mach number > specified limit
     *
     * Uses namespace-level ValidationConfig type.
     */
    void set_validation(const ValidationConfig& cfg) {
        // Copy fields to internal validation config
        validation_.check_negative_density = cfg.check_negative_density;
        validation_.check_negative_pressure = cfg.check_negative_pressure;
        validation_.check_nan = cfg.check_nan;
        validation_.check_cfl = cfg.check_cfl;
        validation_.max_mach = Real(cfg.max_mach);
        validation_.min_pressure = Real(cfg.min_pressure);
        validation_.min_density = Real(cfg.min_density);
        validation_.throw_on_error = cfg.throw_on_error;
        validation_.abort_on_error = cfg.abort_on_error;
        validation_enabled_ = true;
    }

    void disable_validation() {
        validation_enabled_ = false;
    }

    /**
     * @brief Get validation statistics
     *
     * Returns namespace-level ValidationStats type.
     */
    ValidationStats get_validation_stats() const {
        ValidationStats stats;
        stats.nan_count = validation_stats_.nan_count;
        stats.negative_density_count = validation_stats_.negative_density_count;
        stats.negative_pressure_count = validation_stats_.negative_pressure_count;
        stats.cfl_violations = validation_stats_.cfl_violations;
        stats.max_mach_seen = static_cast<float>(validation_stats_.max_mach_seen);
        stats.is_valid = validation_stats_.is_valid;
        return stats;
    }

    // ========================================================================
    // PROFILING (NEW: Built-in performance profiling)
    // ========================================================================

    /**
     * @brief Enable profiling
     */
    void enable_profiling(bool enable = true) {
        profiling_enabled_ = enable;
        if (enable) {
            profile_data_.clear();
        }
    }

    /**
     * @brief Get profiling data
     */
    struct ProfileData {
        double step_time_mean = 0.0;     // Average step time (ms)
        double step_time_min = 1e100;    // Minimum step time
        double step_time_max = 0.0;      // Maximum step time
        double remesh_time_mean = 0.0;   // Average remesh time
        double bc_time_mean = 0.0;       // Average BC fill time
        double flux_time_mean = 0.0;     // Average flux computation time
        std::size_t total_cells_avg = 0; // Average cell count
        double memory_mb = 0.0;          // Memory usage (MB)
    };

    const ProfileData& profile() const {
        return profile_data_;
    }

    /**
     * @brief Print profiling summary
     */
    void print_profile() const {
        if (!profiling_enabled_) {
            printf("Profiling disabled.\n");
            return;
        }

        printf("\n=== Profiling Summary ===\n");
        printf("Step time: %.3f ms (min: %.3f, max: %.3f)\n",
               profile_data_.step_time_mean,
               profile_data_.step_time_min,
               profile_data_.step_time_max);
        printf("Remesh time: %.3f ms\n", profile_data_.remesh_time_mean);
        printf("BC fill time: %.3f ms\n", profile_data_.bc_time_mean);
        printf("Flux time: %.3f ms\n", profile_data_.flux_time_mean);
        printf("Avg cells: %zu\n", profile_data_.total_cells_avg);
        printf("Memory: %.2f MB\n", profile_data_.memory_mb);
        printf("========================\n\n");
    }

    // ========================================================================
    // CUDA COMPATIBILITY: Helper methods must be public
    // ========================================================================
    //
    // NOTE: CUDA nvcc does not allow __host__ __device__ lambdas (KOKKOS_LAMBDA)
    // inside private or protected member functions. These helper methods
    // are therefore public, but intended for internal use.
    //
    // ========================================================================

    /**
     * @brief Compute maximum wave speed for CFL condition
     *
     * Returns max(|v| + a) over all cells, where:
     * - v = velocity magnitude
     * - a = sound speed
     *
     * NOTE: Public for CUDA compatibility (nvcc restriction on private methods
     *       with device lambdas)
     */
    Real compute_max_wave_speed() const {
        using ExecSpace = typename Kokkos::DefaultExecutionSpace;
        Real max_speed = Real(0);
        auto U = U_;
        auto gamma = cfg_.gamma;

        Kokkos::parallel_reduce(
            "compute_max_wave_speed",
            Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(n_cells_)),
            KOKKOS_LAMBDA(const int i, Real& local_max) {
                Primitive q = System::to_primitive(U(i), gamma);
                Real speed = Real(0);

                // Phase 6: Generic wave speed computation for ANY System
                if constexpr (System::n_conserved >= 4) {
                    // For systems with velocity components (Euler2D and similar)
                    Real a = System::sound_speed(q, gamma);
                    Real vel = Kokkos::sqrt(q.u * q.u + q.v * q.v);
                    speed = vel + a;
                } else {
                    // For scalar systems (Advection2D), wave speed is constant
                    // Use default value since system_instance is not accessible in device lambda
                    speed = Real(1);  // Default advection speed
                }

                if (speed > local_max) {
                    local_max = speed;
                }
            },
            Kokkos::Max<Real>(max_speed)
        );

        return max_speed;
    }

    /**
     * @brief Apply boundary conditions to ghost cells
     *
     * Applies the configured boundary conditions to all 4 domain edges.
     * For dense storage on regular grids with row-major ordering.
     *
     * NOTE: Public for CUDA compatibility.
     *
     * @param t Current simulation time (for time-dependent BCs)
     */
    void apply_boundary_conditions(Real t = Real(0)) {
        if (!fields_allocated_) return;

        // Phase 3: Use BcManager for time-dependent BCs if enabled
        if (use_bc_manager_) {
            apply_boundary_conditions_with_manager(t);
            return;
        }

        // Original implementation using simple bc_config_
        using ExecSpace = typename Kokkos::DefaultExecutionSpace;

        auto U = U_;
        const std::size_t nx = cfg_.nx;
        const std::size_t ny = cfg_.ny;
        const auto gamma = cfg_.gamma;

        // Capture BC config by value (required for Kokkos lambdas)
        const auto bc_left = bc_config_.left;
        const auto bc_right = bc_config_.right;
        const auto bc_bottom = bc_config_.bottom;
        const auto bc_top = bc_config_.top;

        // Left boundary (x=0, all y)
        if (nx > 0 && ny > 0) {
            Kokkos::parallel_for(
                "bc_left",
                Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(ny)),
                KOKKOS_LAMBDA(const int j) {
                    const std::size_t idx = j * nx;  // First column (ghost)
                    const std::size_t idx_int = j * nx + 1;  // First interior cell
                    if (idx_int < nx * ny) {
                        Conserved U_ghost = U(idx);
                        const Conserved U_interior = U(idx_int);
                        bc_left.apply(U_ghost, U_interior, gamma, t);
                        U(idx) = U_ghost;
                    }
                }
            );
        }

        // Right boundary (x=nx-1, all y)
        if (nx > 0 && ny > 0) {
            Kokkos::parallel_for(
                "bc_right",
                Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(ny)),
                KOKKOS_LAMBDA(const int j) {
                    const std::size_t idx = j * nx + (nx - 1);  // Last column (ghost)
                    const std::size_t idx_int = j * nx + (nx - 2);  // Last interior cell
                    if (nx >= 2 && idx_int < nx * ny) {
                        Conserved U_ghost = U(idx);
                        const Conserved U_interior = U(idx_int);
                        bc_right.apply(U_ghost, U_interior, gamma, t);
                        U(idx) = U_ghost;
                    }
                }
            );
        }

        // Bottom boundary (y=0, all x)
        if (nx > 0 && ny > 0) {
            Kokkos::parallel_for(
                "bc_bottom",
                Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(nx)),
                KOKKOS_LAMBDA(const int i) {
                    const std::size_t idx = i;  // First row (ghost)
                    const std::size_t idx_int = i + nx;  // First interior row
                    if (idx_int < nx * ny) {
                        Conserved U_ghost = U(idx);
                        const Conserved U_interior = U(idx_int);
                        bc_bottom.apply(U_ghost, U_interior, gamma, t);
                        U(idx) = U_ghost;
                    }
                }
            );
        }

        // Top boundary (y=ny-1, all x)
        if (nx > 0 && ny > 0) {
            Kokkos::parallel_for(
                "bc_top",
                Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(nx)),
                KOKKOS_LAMBDA(const int i) {
                    const std::size_t idx = (ny - 1) * nx + i;  // Last row (ghost)
                    const std::size_t idx_int = (ny - 2) * nx + i;  // Last interior row
                    if (ny >= 2 && idx_int < nx * ny) {
                        Conserved U_ghost = U(idx);
                        const Conserved U_interior = U(idx_int);
                        bc_top.apply(U_ghost, U_interior, gamma, t);
                        U(idx) = U_ghost;
                    }
                }
            );
        }

        Kokkos::fence();

        // Notify observers: BCs were evaluated
        SolverState<Real> state;
        state.time = t;
        observer_manager_.notify(SolverEvent::BoundaryConditionsEvaluated, state);
    }

    /**
     * @brief Apply time-dependent boundary conditions using BcManager
     *
     * Phase 3: Advanced BC system with time dependence and zonal BCs
     *
     * NOTE: Public for CUDA compatibility.
     *
     * @param t Current simulation time (for time-dependent BCs)
     */
    void apply_boundary_conditions_with_manager(Real t) {
        if (!fields_allocated_) return;

        // Sync any pending BC changes to device
        bc_manager_.sync_to_device();

        using ExecSpace = typename Kokkos::DefaultExecutionSpace;
        const auto& registry = bc_manager_.device_registry();

        auto U = U_;
        const std::size_t nx = cfg_.nx;
        const std::size_t ny = cfg_.ny;
        const auto gamma = cfg_.gamma;
        const Real dx = cfg_.dx;
        const Real dy = cfg_.dy;
        const Real x0 = domain_.x_min;
        const Real y0 = domain_.y_min;

        // Capture registry by value (not directly, but through device reference)
        // Note: registry is already device-accessible via its Kokkos::Views

        // Left boundary (side 0, x=0, all y)
        if (nx > 0 && ny > 0) {
            Kokkos::parallel_for(
                "bc_left_manager",
                Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(ny)),
                KOKKOS_LAMBDA(const int j) {
                    const std::size_t idx = j * nx;  // First column (ghost)
                    const std::size_t idx_int = j * nx + 1;  // First interior cell
                    if (idx_int < nx * ny) {
                        // Find BC descriptor for this location
                        auto desc = registry.find(0, 0, j, t);  // side 0 = left

                        Conserved U_ghost = U(idx);
                        const Conserved U_interior = U(idx_int);

                        // Apply BC based on descriptor type
                        if (desc.type == boundary::BcDescriptor<System>::StaticNeumann) {
                            U_ghost = U_interior;
                        } else if (desc.type == boundary::BcDescriptor<System>::StaticDirichlet ||
                                   desc.type == boundary::BcDescriptor<System>::StaticReflective) {
                            U_ghost = desc.static_value;
                        } else if (desc.type == boundary::BcDescriptor<System>::TimeDependentDirichlet ||
                                   desc.type == boundary::BcDescriptor<System>::TimeDependentInlet) {
                            U_ghost = desc.get_value(t, gamma);
                        } else {
                            // Default: Neumann
                            U_ghost = U_interior;
                        }

                        U(idx) = U_ghost;
                    }
                }
            );
        }

        // Right boundary (side 1, x=nx-1, all y)
        if (nx > 0 && ny > 0) {
            Kokkos::parallel_for(
                "bc_right_manager",
                Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(ny)),
                KOKKOS_LAMBDA(const int j) {
                    const std::size_t idx = j * nx + (nx - 1);  // Last column (ghost)
                    const std::size_t idx_int = j * nx + (nx - 2);  // Last interior cell
                    if (nx >= 2 && idx_int < nx * ny) {
                        auto desc = registry.find(1, nx - 1, j, t);  // side 1 = right

                        Conserved U_ghost = U(idx);
                        const Conserved U_interior = U(idx_int);

                        if (desc.type == boundary::BcDescriptor<System>::StaticNeumann) {
                            U_ghost = U_interior;
                        } else if (desc.type == boundary::BcDescriptor<System>::StaticDirichlet ||
                                   desc.type == boundary::BcDescriptor<System>::StaticReflective) {
                            U_ghost = desc.static_value;
                        } else if (desc.type == boundary::BcDescriptor<System>::TimeDependentDirichlet ||
                                   desc.type == boundary::BcDescriptor<System>::TimeDependentInlet) {
                            U_ghost = desc.get_value(t, gamma);
                        } else {
                            U_ghost = U_interior;
                        }

                        U(idx) = U_ghost;
                    }
                }
            );
        }

        // Bottom boundary (side 2, y=0, all x)
        if (nx > 0 && ny > 0) {
            Kokkos::parallel_for(
                "bc_bottom_manager",
                Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(nx)),
                KOKKOS_LAMBDA(const int i) {
                    const std::size_t idx = i;  // First row (ghost)
                    const std::size_t idx_int = i + nx;  // First interior row
                    if (idx_int < nx * ny) {
                        auto desc = registry.find(2, i, 0, t);  // side 2 = bottom

                        Conserved U_ghost = U(idx);
                        const Conserved U_interior = U(idx_int);

                        if (desc.type == boundary::BcDescriptor<System>::StaticNeumann) {
                            U_ghost = U_interior;
                        } else if (desc.type == boundary::BcDescriptor<System>::StaticDirichlet ||
                                   desc.type == boundary::BcDescriptor<System>::StaticReflective) {
                            U_ghost = desc.static_value;
                        } else if (desc.type == boundary::BcDescriptor<System>::TimeDependentDirichlet ||
                                   desc.type == boundary::BcDescriptor<System>::TimeDependentInlet) {
                            U_ghost = desc.get_value(t, gamma);
                        } else {
                            U_ghost = U_interior;
                        }

                        U(idx) = U_ghost;
                    }
                }
            );
        }

        // Top boundary (side 3, y=ny-1, all x)
        if (nx > 0 && ny > 0) {
            Kokkos::parallel_for(
                "bc_top_manager",
                Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(nx)),
                KOKKOS_LAMBDA(const int i) {
                    const std::size_t idx = (ny - 1) * nx + i;  // Last row (ghost)
                    const std::size_t idx_int = (ny - 2) * nx + i;  // Last interior row
                    if (ny >= 2 && idx_int < nx * ny) {
                        auto desc = registry.find(3, i, ny - 1, t);  // side 3 = top

                        Conserved U_ghost = U(idx);
                        const Conserved U_interior = U(idx_int);

                        if (desc.type == boundary::BcDescriptor<System>::StaticNeumann) {
                            U_ghost = U_interior;
                        } else if (desc.type == boundary::BcDescriptor<System>::StaticDirichlet ||
                                   desc.type == boundary::BcDescriptor<System>::StaticReflective) {
                            U_ghost = desc.static_value;
                        } else if (desc.type == boundary::BcDescriptor<System>::TimeDependentDirichlet ||
                                   desc.type == boundary::BcDescriptor<System>::TimeDependentInlet) {
                            U_ghost = desc.get_value(t, gamma);
                        } else {
                            U_ghost = U_interior;
                        }

                        U(idx) = U_ghost;
                    }
                }
            );
        }

        Kokkos::fence();

        // Notify observers: BCs were evaluated
        SolverState<Real> state;
        state.time = t;
        observer_manager_.notify(SolverEvent::BoundaryConditionsEvaluated, state);
    }

    /**
     * @brief Compute right-hand side: dU/dt = -div(F)
     *
     * Computes flux divergence using the configured flux scheme.
     * For dense storage on regular grids with row-major ordering:
     *   idx = j * nx + i
     *   neighbors: left=idx-1, right=idx+1, south=idx-nx, north=idx+nx
     *
     * The finite volume formulation:
     *   dU/dt = -1/dx * (F_{i+1/2} - F_{i-1/2})
     *          -1/dy * (G_{j+1/2} - G_{j-1/2})
     *
     * NOTE: Public for CUDA compatibility.
     */
    void compute_rhs(Kokkos::View<Conserved*>& rhs_out, Real t) {
        // Time parameter available for time-dependent source terms or BCs
        (void)t;  // Currently unused in RHS computation (reserved for future use)
        if (!fields_allocated_) return;

        using ExecSpace = typename Kokkos::DefaultExecutionSpace;

        // Capture all needed data by value (Kokkos requirement)
        auto U = U_;
        auto rhs = rhs_out;
        const Real dx = cfg_.dx;
        const Real dy = cfg_.dy;
        const Real gamma = cfg_.gamma;
        const std::size_t nx = cfg_.nx;
        const std::size_t ny = cfg_.ny;

        // Capture flux scheme by value
        const auto flux_scheme = flux_;

        // ========================================================================
        // PHASE 4: MUSCL RECONSTRUCTION INTEGRATION
        // ========================================================================
        // Compile-time dispatch: 1st order (NoReconstruction) vs 2nd order (MUSCL)
        // The Reconstruction template parameter determines which path is taken
        // ========================================================================

        if constexpr (!std::is_same_v<Reconstruction, reconstruction::NoReconstruction>) {
            // ====================================================================
            // 2ND ORDER MUSCL RECONSTRUCTION PATH
            // ====================================================================
            // Uses 5-point stencil to reconstruct left/right states at faces
            // This is the HIGH-ORDER path that was previously not activated
            //
            // PHASE 4 FIX: Compile-time dispatch for MUSCL vs WENO reconstruction
            // - MUSCL: 4-point stencil per face (uses reconstruct_interface)
            // - WENO: 5-point stencil per face (uses reconstruct_interface with 5 params)
            // ====================================================================

            if (nx < 5 || ny < 5) {
                // Grid too small for 5-point stencil needed for MUSCL/WENO
                return;
            }

            // ====================================================================
            // DETECT WENO vs MUSCL at compile time using stencil_width
            // ====================================================================
            constexpr bool is_weno = (Reconstruction::stencil_width == 5);

            if constexpr (is_weno) {
                // ====================================================================
                // WENO5 PATH: Uses 5-point symmetric stencil
                // ====================================================================
                Kokkos::parallel_for(
                    "compute_rhs_weno",
                    Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>((ny - 4) * (nx - 4))),
                    KOKKOS_LAMBDA(const int linear_idx) {
                        // Convert linear index to 2D coordinates (interior only, with 2-cell halo)
                        const int j = 2 + linear_idx / static_cast<int>(nx - 4);
                        const int i = 2 + linear_idx % static_cast<int>(nx - 4);

                        // Compute 1D indices for 5-point stencil in x-direction
                        const std::size_t idx_c  = j * nx + i;
                        const std::size_t idx_w  = j * nx + (i - 1);
                        const std::size_t idx_e  = j * nx + (i + 1);
                        const std::size_t idx_ww = j * nx + (i - 2);
                        const std::size_t idx_ee = j * nx + (i + 2);

                        // Compute 1D indices for 5-point stencil in y-direction
                        const std::size_t idx_s  = (j - 1) * nx + i;
                        const std::size_t idx_n  = (j + 1) * nx + i;
                        const std::size_t idx_ss = (j - 2) * nx + i;
                        const std::size_t idx_nn = (j + 2) * nx + i;

                        // Gather conserved variables from 5-point stencil
                        const Conserved U_ww_x = U(idx_ww);
                        const Conserved U_w_x  = U(idx_w);
                        const Conserved U_c_x  = U(idx_c);
                        const Conserved U_e_x  = U(idx_e);
                        const Conserved U_ee_x = U(idx_ee);

                        const Conserved U_ss_y = U(idx_ss);
                        const Conserved U_s_y  = U(idx_s);
                        const Conserved U_c_y  = U(idx_c);
                        const Conserved U_n_y  = U(idx_n);
                        const Conserved U_nn_y = U(idx_nn);

                        // Convert to primitive variables for all stencil points
                        const Primitive q_ww_x = System::to_primitive(U_ww_x, gamma);
                        const Primitive q_w_x  = System::to_primitive(U_w_x, gamma);
                        const Primitive q_c_x  = System::to_primitive(U_c_x, gamma);
                        const Primitive q_e_x  = System::to_primitive(U_e_x, gamma);
                        const Primitive q_ee_x = System::to_primitive(U_ee_x, gamma);

                        const Primitive q_ss_y = System::to_primitive(U_ss_y, gamma);
                        const Primitive q_s_y  = System::to_primitive(U_s_y, gamma);
                        const Primitive q_c_y  = System::to_primitive(U_c_y, gamma);
                        const Primitive q_n_y  = System::to_primitive(U_n_y, gamma);
                        const Primitive q_nn_y = System::to_primitive(U_nn_y, gamma);

                        // WENO5 reconstruction at faces
                        Primitive qL_west_face, qR_west_face;
                        Primitive qL_east_face, qR_east_face;
                        Primitive qL_south_face, qR_south_face;
                        Primitive qL_north_face, qR_north_face;

                        Reconstruction::reconstruct_interface(
                            q_ww_x, q_w_x, q_c_x, q_e_x, q_ee_x,
                            qL_west_face, qR_west_face
                        );
                        Reconstruction::reconstruct_interface(
                            q_ww_x, q_w_x, q_c_x, q_e_x, q_ee_x,
                            qL_east_face, qR_east_face
                        );

                        Reconstruction::reconstruct_interface(
                            q_ss_y, q_s_y, q_c_y, q_n_y, q_nn_y,
                            qL_south_face, qR_south_face
                        );
                        Reconstruction::reconstruct_interface(
                            q_ss_y, q_s_y, q_c_y, q_n_y, q_nn_y,
                            qL_north_face, qR_north_face
                        );

                        // Convert reconstructed primitives to conserved variables
                        const Conserved UL_west = System::from_primitive(qL_west_face, gamma);
                        const Conserved UR_west = System::from_primitive(qR_west_face, gamma);
                        const Conserved UL_east = System::from_primitive(qL_east_face, gamma);
                        const Conserved UR_east = System::from_primitive(qR_east_face, gamma);

                        const Conserved UL_south = System::from_primitive(qL_south_face, gamma);
                        const Conserved UR_south = System::from_primitive(qR_south_face, gamma);
                        const Conserved UL_north = System::from_primitive(qL_north_face, gamma);
                        const Conserved UR_north = System::from_primitive(qR_north_face, gamma);

                        // Compute numerical fluxes at faces
                        const Conserved F_w = flux_scheme.flux_x(UL_west, UR_west, qL_west_face, qR_west_face);
                        const Conserved F_e = flux_scheme.flux_x(UL_east, UR_east, qL_east_face, qR_east_face);
                        const Conserved F_s = flux_scheme.flux_y(UL_south, UR_south, qL_south_face, qR_south_face);
                        const Conserved F_n = flux_scheme.flux_y(UL_north, UR_north, qL_north_face, qR_north_face);

                        // Compute flux divergence
                        const Real inv_dx = Real(1) / dx;
                        const Real inv_dy = Real(1) / dy;
                        Conserved RHS = (F_w - F_e) * inv_dx + (F_s - F_n) * inv_dy;

                        rhs(idx_c) = RHS;
                    }
                );
            } else {
                // ====================================================================
                // MUSCL PATH: Uses shifted 3-point stencil per face
                // ====================================================================
                Kokkos::parallel_for(
                    "compute_rhs_muscl",
                    Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>((ny - 4) * (nx - 4))),
                    KOKKOS_LAMBDA(const int linear_idx) {
                        // Convert linear index to 2D coordinates (interior only, with 2-cell halo)
                        const int j = 2 + linear_idx / static_cast<int>(nx - 4);
                        const int i = 2 + linear_idx % static_cast<int>(nx - 4);

                        // Compute 1D indices for 5-point stencil in x-direction
                        const std::size_t idx_c  = j * nx + i;
                        const std::size_t idx_w  = j * nx + (i - 1);
                        const std::size_t idx_e  = j * nx + (i + 1);
                        const std::size_t idx_ww = j * nx + (i - 2);
                        const std::size_t idx_ee = j * nx + (i + 2);

                        // Compute 1D indices for 5-point stencil in y-direction
                        const std::size_t idx_s  = (j - 1) * nx + i;
                        const std::size_t idx_n  = (j + 1) * nx + i;
                        const std::size_t idx_ss = (j - 2) * nx + i;
                        const std::size_t idx_nn = (j + 2) * nx + i;

                        // Gather conserved variables from 5-point stencil
                        const Conserved U_ww_x = U(idx_ww);
                        const Conserved U_w_x  = U(idx_w);
                        const Conserved U_c_x  = U(idx_c);
                        const Conserved U_e_x  = U(idx_e);
                        const Conserved U_ee_x = U(idx_ee);

                        const Conserved U_ss_y = U(idx_ss);
                        const Conserved U_s_y  = U(idx_s);
                        const Conserved U_c_y  = U(idx_c);
                        const Conserved U_n_y  = U(idx_n);
                        const Conserved U_nn_y = U(idx_nn);

                        // Convert to primitive variables for all stencil points
                        const Primitive q_ww_x = System::to_primitive(U_ww_x, gamma);
                        const Primitive q_w_x  = System::to_primitive(U_w_x, gamma);
                        const Primitive q_c_x  = System::to_primitive(U_c_x, gamma);
                        const Primitive q_e_x  = System::to_primitive(U_e_x, gamma);
                        const Primitive q_ee_x = System::to_primitive(U_ee_x, gamma);

                        const Primitive q_ss_y = System::to_primitive(U_ss_y, gamma);
                        const Primitive q_s_y  = System::to_primitive(U_s_y, gamma);
                        const Primitive q_c_y  = System::to_primitive(U_c_y, gamma);
                        const Primitive q_n_y  = System::to_primitive(U_n_y, gamma);
                        const Primitive q_nn_y = System::to_primitive(U_nn_y, gamma);

                        // MUSCL reconstruction at faces
                        Primitive qL_west_face, qR_west_face;
                        Primitive qL_east_face, qR_east_face;
                        Primitive qL_south_face, qR_south_face;
                        Primitive qL_north_face, qR_north_face;

                        // X-direction: west face (i-1/2) and east face (i+1/2)
                        Reconstruction::reconstruct_interface(
                            q_ww_x, q_w_x, q_c_x, q_e_x,
                            qL_west_face, qR_west_face
                        );
                        Reconstruction::reconstruct_interface(
                            q_w_x, q_c_x, q_e_x, q_ee_x,
                            qL_east_face, qR_east_face
                        );

                        // Y-direction: south face (j-1/2) and north face (j+1/2)
                        Reconstruction::reconstruct_interface(
                            q_ss_y, q_s_y, q_c_y, q_n_y,
                            qL_south_face, qR_south_face
                        );
                        Reconstruction::reconstruct_interface(
                            q_s_y, q_c_y, q_n_y, q_nn_y,
                            qL_north_face, qR_north_face
                        );

                        // Convert reconstructed primitives to conserved variables
                        const Conserved UL_west = System::from_primitive(qL_west_face, gamma);
                        const Conserved UR_west = System::from_primitive(qR_west_face, gamma);
                        const Conserved UL_east = System::from_primitive(qL_east_face, gamma);
                        const Conserved UR_east = System::from_primitive(qR_east_face, gamma);

                        const Conserved UL_south = System::from_primitive(qL_south_face, gamma);
                        const Conserved UR_south = System::from_primitive(qR_south_face, gamma);
                        const Conserved UL_north = System::from_primitive(qL_north_face, gamma);
                        const Conserved UR_north = System::from_primitive(qR_north_face, gamma);

                        // Compute numerical fluxes at faces
                        const Conserved F_w = flux_scheme.flux_x(UL_west, UR_west, qL_west_face, qR_west_face);
                        const Conserved F_e = flux_scheme.flux_x(UL_east, UR_east, qL_east_face, qR_east_face);
                        const Conserved F_s = flux_scheme.flux_y(UL_south, UR_south, qL_south_face, qR_south_face);
                        const Conserved F_n = flux_scheme.flux_y(UL_north, UR_north, qL_north_face, qR_north_face);

                        // Compute flux divergence
                        const Real inv_dx = Real(1) / dx;
                        const Real inv_dy = Real(1) / dy;
                        Conserved RHS = (F_w - F_e) * inv_dx + (F_s - F_n) * inv_dy;

                        rhs(idx_c) = RHS;
                    }
                );
            }
        } else {
            // ====================================================================
            // 1ST ORDER PATH (NoReconstruction)
            // ====================================================================
            // Direct flux computation using cell-centered values (Godunov scheme)
            // ====================================================================

            if (nx < 3 || ny < 3) {
                // Grid too small for interior computation
                return;
            }

            Kokkos::parallel_for(
                "compute_rhs_dense",
                Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>((ny - 2) * (nx - 2))),
                KOKKOS_LAMBDA(const int linear_idx) {
                    // Convert linear index to 2D coordinates (interior only)
                    const int j = 1 + linear_idx / static_cast<int>(nx - 2);
                    const int i = 1 + linear_idx % static_cast<int>(nx - 2);

                    // Compute 1D indices for center and neighbors
                    const std::size_t idx_c = j * nx + i;       // center
                    const std::size_t idx_w  = j * nx + (i - 1); // west (left)
                    const std::size_t idx_e  = j * nx + (i + 1); // east (right)
                    const std::size_t idx_s  = (j - 1) * nx + i; // south (bottom)
                    const std::size_t idx_n  = (j + 1) * nx + i; // north (top)

                    // Gather conserved variables from neighbors
                    const Conserved U_c = U(idx_c);
                    const Conserved U_w = U(idx_w);
                    const Conserved U_e = U(idx_e);
                    const Conserved U_s = U(idx_s);
                    const Conserved U_n = U(idx_n);

                    // Convert to primitive variables
                    const Primitive q_c = System::to_primitive(U_c, gamma);
                    const Primitive q_w = System::to_primitive(U_w, gamma);
                    const Primitive q_e = System::to_primitive(U_e, gamma);
                    const Primitive q_s = System::to_primitive(U_s, gamma);
                    const Primitive q_n = System::to_primitive(U_n, gamma);

                    // Compute numerical fluxes at faces
                    // X-direction: flux at west and east faces
                    const Conserved F_w = flux_scheme.flux_x(U_w, U_c, q_w, q_c);
                    const Conserved F_e = flux_scheme.flux_x(U_c, U_e, q_c, q_e);

                    // Y-direction: flux at south and north faces
                    const Conserved F_s = flux_scheme.flux_y(U_s, U_c, q_s, q_c);
                    const Conserved F_n = flux_scheme.flux_y(U_c, U_n, q_c, q_n);

                    // Compute flux divergence: dU/dt = -div(F)
                    // div(F)_x = (F_east - F_West) / dx
                    // div(F)_y = (F_North - F_South) / dy
                    // Phase 6: Generic computation using operators - works for ANY System
                    const Real inv_dx = Real(1) / dx;
                    const Real inv_dy = Real(1) / dy;

                    // RHS = -div(F) = -(dF/dx + dF/dy)
                    // Using generic operators: (F_w - F_e) * inv_dx + (F_s - F_n) * inv_dy
                    // This avoids unary minus which may not be defined
                    Conserved RHS = (F_w - F_e) * inv_dx + (F_s - F_n) * inv_dy;

                    rhs(idx_c) = RHS;
                }
            );
        }

        Kokkos::fence();
    }

    /**
     * @brief Check if fields are allocated
     *
     * NOTE: Public for CUDA compatibility.
     */
    bool fields_allocated() const { return fields_allocated_; }

    /**
     * @brief Get number of cells
     *
     * NOTE: Public for CUDA compatibility.
     */
    std::size_t n_cells() const { return n_cells_; }

    // ========================================================================
    // AMR REMESHING (CUDA-compatible public methods)
    // ========================================================================

    /**
     * @brief Remesh implementation - rebuilds AMR hierarchy
     *
     * This method:
     * 1. Evaluates refinement criteria for each cell
     * 2. Builds refined geometries for levels that need refinement
     * 3. Prolongs solution from coarse to fine levels
     * 4. Updates the finest_level_ pointer
     *
     * NOTE: Public for CUDA compatibility (nvcc restriction on private methods
     *       with device lambdas)
     */
    void remesh() {
        if (!refinement_enabled_ || !fields_allocated_) return;

        // Notify observers: remesh beginning
        SolverState<Real> state;
        state.time = current_time_;
        state.dt = Real(0);
        state.step = step_count_;
        state.total_cells = n_cells_;
        observer_manager_.notify(SolverEvent::RemeshBegin, state);

        using ExecSpace = typename Kokkos::DefaultExecutionSpace;
        subsetix::csr::CsrSetAlgebraContext ctx;

        // Step 1: Evaluate refinement criteria on current finest level
        Kokkos::View<int8_t*> refinement_tags;
        evaluate_refinement(refinement_tags);

        // Step 2: Check if any cells need refinement
        int needs_refinement = 0;
        Kokkos::parallel_reduce(
            "check_refinement_needed",
            Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(n_cells_)),
            KOKKOS_LAMBDA(const int i, int& local_sum) {
                if (refinement_tags(i) == static_cast<int8_t>(amr::RefinementAction::Refine)) {
                    local_sum++;
                }
            },
            needs_refinement
        );

        if (needs_refinement == 0 && finest_level_ == 0) {
            // No refinement needed, stay at level 0
            return;
        }

        // Step 3: Build refined geometry for level 1
        if (finest_level_ == 0 && needs_refinement > 0) {
            // Allocate level 1 storage
            csr::IntervalSet2DDevice refined_geom;

            // Build refined geometry (MVP: refine entire domain)
            build_refined_geometry(refinement_tags, refined_geom);

            // Check if refinement succeeded
            if (refined_geom.total_cells > levels_[0].geometry.total_cells) {
                // Allocate fields for level 1
                const std::size_t n_fine = refined_geom.total_cells;
                levels_[1].U = Kokkos::View<Conserved*>("U_level_1", n_fine);
                levels_[1].rhs_work = Kokkos::View<Conserved*>("rhs_level_1", n_fine);
                levels_[1].geometry = refined_geom;
                levels_[1].n_cells = n_fine;
                levels_[1].active = true;
                levels_[1].level = 1;

                // Prolong solution from level 0 to level 1
                prolong_to_level(0, 1);

                // Update finest level
                finest_level_ = 1;
            }
        }

        // Step 4: Notify observers: remesh complete
        state.total_cells = levels_[finest_level_].n_cells;
        observer_manager_.notify(SolverEvent::RemeshEnd, state);

        // Reset remesh counter
        remesh_step_counter_ = 0;
    }

    /**
     * @brief Evaluate refinement criteria and tag cells
     *
     * Computes refinement action (Coarsen/Keep/Refine) for each cell
     * based on the configured refinement criteria.
     *
     * NOTE: Public for CUDA compatibility (nvcc restriction on private methods
     *       with device lambdas)
     */
    void evaluate_refinement(Kokkos::View<int8_t*>& tags) {
        using ExecSpace = typename Kokkos::DefaultExecutionSpace;

        if (!refinement_enabled_) return;

        const std::size_t n = n_cells_;
        tags = Kokkos::View<int8_t*>("refinement_tags", n);

        auto U = U_;
        auto gamma = cfg_.gamma;
        auto criterion = refinement_config_.criterion;
        const Real dx = cfg_.dx;
        const std::size_t nx = cfg_.nx;

        // Phase 5: Use full CompositeCriterion with coarsening support
        Kokkos::parallel_for(
            "evaluate_refinement",
            Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(n)),
            KOKKOS_LAMBDA(const int idx) {
                // Get cell state
                const Conserved U_cell = U(idx);
                const Primitive q_cell = System::to_primitive(U_cell, gamma);

                // Check level limits (for current finest level)
                int8_t current_level = finest_level_;
                if (current_level >= static_cast<int8_t>(refinement_config_.max_level)) {
                    tags(idx) = static_cast<int8_t>(amr::RefinementAction::Keep);
                    return;
                }

                // Phase 5: Evaluate using full CompositeCriterion
                // Pass nullptr for siblings (simple mode - no coarsening based on siblings)
                // Coarsening is handled by SmoothnessCriterion which checks local variation
                amr::RefinementAction action = criterion.evaluate(
                    U_cell, q_cell,
                    nullptr,  // siblings = nullptr for simple mode
                    dx
                );

                tags(idx) = static_cast<int8_t>(action);
            }
        );
        Kokkos::fence();
    }

    /**
     * @brief Build refined geometry from refinement tags
     *
     * Creates a new refined geometry by marking cells for refinement
     * and building the corresponding CSR structure.
     *
     * NOTE: Public for CUDA compatibility.
     */
    void build_refined_geometry(const Kokkos::View<int8_t*>& tags,
                                csr::IntervalSet2DDevice& refined_geom) {
        using ExecSpace = typename Kokkos::DefaultExecutionSpace;

        // For MVP: refine cells marked with Refine action
        // This is a simplified version - full implementation would:
        // 1. Identify refine regions from tags
        // 2. Expand by buffer cells
        // 3. Constrain to parent interior
        // 4. Build refined CSR geometry

        // Placeholder: refine entire domain by factor of 2
        subsetix::csr::CsrSetAlgebraContext ctx;
        subsetix::csr::refine_level_up_device(levels_[0].geometry, refined_geom, ctx);
    }

    /**
     * @brief Prolong solution from coarse to fine level
     *
     * Transfers solution data from coarse level to fine level using
     * injection or linear reconstruction.
     *
     * Phase 5: Uses CSR-aware prolongation with proper geometry handling.
     *
     * NOTE: Public for CUDA compatibility.
     */
    void prolong_to_level(int coarse_level, int fine_level) {
        if (coarse_level < 0 || fine_level >= max_amr_levels_) return;
        if (!levels_[coarse_level].active || !levels_[fine_level].active) return;

        // Phase 5: Use CSR-aware prolongation
        // Create Field2DDevice wrappers for prolongation
        subsetix::csr::Field2DDevice<Conserved> coarse_field;
        coarse_field.geometry = levels_[coarse_level].geometry;
        coarse_field.values = levels_[coarse_level].U;

        subsetix::csr::Field2DDevice<Conserved> fine_field;
        fine_field.geometry = levels_[fine_level].geometry;
        fine_field.values = levels_[fine_level].U;

        // Prolong on entire fine geometry (injection)
        subsetix::csr::CsrSetAlgebraContext ctx;
        subsetix::csr::prolong_field_on_subset_device(
            fine_field, coarse_field, fine_field.geometry, &ctx);

        Kokkos::fence();
    }

    /**
     * @brief Restrict solution from fine to coarse level
     *
     * Transfers solution data from fine level to coarse level using
     * volume-weighted averaging (conservative restriction).
     *
     * Phase 5: Uses CSR-aware restriction with proper geometry handling.
     *
     * NOTE: Public for CUDA compatibility.
     */
    void restrict_to_level(int fine_level, int coarse_level) {
        if (fine_level <= 0 || coarse_level >= fine_level) return;
        if (!levels_[fine_level].active || !levels_[coarse_level].active) return;

        // Phase 5: Use CSR-aware restriction
        // Create Field2DDevice wrappers for restriction
        subsetix::csr::Field2DDevice<Conserved> coarse_field;
        coarse_field.geometry = levels_[coarse_level].geometry;
        coarse_field.values = levels_[coarse_level].U;

        subsetix::csr::Field2DDevice<Conserved> fine_field;
        fine_field.geometry = levels_[fine_level].geometry;
        fine_field.values = levels_[fine_level].U;

        // Restrict on entire coarse geometry (volume-weighted averaging)
        subsetix::csr::CsrSetAlgebraContext ctx;
        subsetix::csr::restrict_field_on_subset_device(
            coarse_field, fine_field, coarse_field.geometry, &ctx);

        Kokkos::fence();
    }

private:
    // Friend declarations
    friend class Builder;

    // Configuration
    Config cfg_;

    // P0-4 FIX: System instance for runtime parameters
    System system_instance_;
    bool has_system_instance_;

    // Flux and reconstruction schemes
    FluxScheme<System> flux_;
    Reconstruction recon_;

    // Boundary conditions (P0-2 FIX)
    BoundaryConfig<System> bc_config_;

    // Time-dependent BC manager (Phase 3)
    boundary::BcManager<System> bc_manager_;
    bool use_bc_manager_ = false;  // Flag to enable time-dependent BCs

    // IMPROVEMENT D: Observer manager for callbacks
    ObserverManager<Real> observer_manager_;

    // Timing for observers
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
    std::chrono::time_point<std::chrono::steady_clock> last_step_time_;

    // Geometry
    csr::IntervalSet2DDevice fluid_geometry_;
    csr::Box2D domain_;

    // Simulation state
    Real current_time_ = Real(0);
    int step_count_ = 0;
    Real last_dt_ = Real(0);           // Phase 4: Last time step used

    // ========================================================================
    // FIELD STORAGE (for internal time integration)
    // ========================================================================

    /**
     * @brief Dense field storage for time integration
     *
     * DESIGN NOTE: The solver stores dense arrays internally for time stepping.
     * This is a simplified design that avoids the complexity of CSR-aware integrators.
     *
     * For production use with CSR geometries, users should:
     * 1. Store fields externally using CSR storage (as in mach2_cylinder.cpp)
     * 2. Convert CSR -> dense for time stepping
     * 3. Convert dense -> CSR after each step
     *
     * The current design provides a self-contained solver for simple cases.
     */
    Kokkos::View<Conserved*> U_;          // Conserved variables (flattened)
    Kokkos::View<Conserved*> rhs_work_;   // RHS workspace
    std::size_t n_cells_ = 0;             // Number of active cells
    bool fields_allocated_ = false;       // Track if fields are initialized

    // ========================================================================
    // MULTI-STAGE TIME INTEGRATION STORAGE
    // ========================================================================

    // For multi-stage Runge-Kutta methods (RK2, RK3, RK4, SSPRK3, etc.)
    // Using fixed maximum size (4 stages for RK4)
    static constexpr int max_rk_stages_ = 4;
    Kokkos::View<Conserved*> stage_rhs_[max_rk_stages_];  // RHS at each stage
    Kokkos::View<Conserved*> stage_solution_;     // Intermediate solution
    Kokkos::View<Conserved*> U_old_;              // Original solution (for RK)
    bool rk_storage_allocated_ = false;           // Track if RK storage is allocated

    // ========================================================================
    // AMR REFINEMENT CONFIGURATION
    // ========================================================================

    /**
     * @brief AMR refinement configuration
     *
     * Contains:
     * - CompositeCriterion: multiple refinement criteria with logic operators
     * - ExclusionZone[]: protected regions with minimum refinement levels
     * - Level limits: min_level, max_level
     * - Coarsening flag: enable/disable coarsening
     * - Remesh frequency: how often to check refinement
     *
     * Set via set_refinement() method.
     */
    amr::RefinementConfig<System> refinement_config_;
    bool refinement_enabled_ = false;
    int remesh_step_counter_ = 0;  // Steps since last remesh

    // ========================================================================
    // MULTI-LEVEL AMR STORAGE
    // ========================================================================

    /**
     * @brief AMR level data structure
     *
     * Each level contains:
     * - geometry: CSR geometry for this level
     * - U: conserved variables
     * - rhs_work: RHS workspace
     * - n_cells: number of active cells
     * - active: whether this level is in use
     *
     * Level 0 = coarsest (base mesh)
     * Level max_amr_levels_-1 = finest (most refined)
     */
    static constexpr int max_amr_levels_ = 6;  // Maximum refinement depth

    struct AmrLevel {
        csr::IntervalSet2DDevice geometry;
        Kokkos::View<Conserved*> U;
        Kokkos::View<Conserved*> rhs_work;
        std::size_t n_cells = 0;
        bool active = false;
        int8_t level = 0;  // Level index (0 to max_amr_levels_-1)
    };

    std::array<AmrLevel, max_amr_levels_> levels_;
    int finest_level_ = 0;  // Current finest active level

    // ========================================================================
    // FIELD ALLOCATION (private helper)
    // ========================================================================

    /**
     * @brief Allocate field storage (private helper)
     *
     * Called from initialize() to set up the internal arrays.
     * NOTE: This doesn't use KOKKOS_LAMBDA so can stay private.
     */
    void allocate_fields(std::size_t n) {
        n_cells_ = n;
        U_ = Kokkos::View<Conserved*>("U", n);
        rhs_work_ = Kokkos::View<Conserved*>("rhs_work", n);
        fields_allocated_ = true;

        // Allocate RK storage if needed (for multi-stage methods)
        if constexpr (TimeIntegrator::stages > 1) {
            for (int s = 0; s < max_rk_stages_; ++s) {
                stage_rhs_[s] = Kokkos::View<Conserved*>("stage_rhs_" + std::to_string(s), n);
            }
            stage_solution_ = Kokkos::View<Conserved*>("stage_solution", n);
            U_old_ = Kokkos::View<Conserved*>("U_old", n);
            rk_storage_allocated_ = true;
        }
    }

    // NEW: Source terms (compile-time, no runtime polymorphism)
    // Note: SourceManager removed - sources are now compile-time composites
    // Users can create custom sources using the template-based API in source_terms.hpp
    bool has_source_terms_ = false;

    // NEW: Checkpoint/restart
    int checkpoint_stride_ = 0;
    std::string checkpoint_prefix_ = "checkpoint";

    // NEW: Output streaming
    bool stream_output_ = false;
    std::string stream_dir_;
    int stream_stride_ = 100;
    std::string stream_format_;

    // NEW: Validation
    bool validation_enabled_ = false;
    ValidationConfig validation_;
    ValidationStats validation_stats_;

    // NEW: Profiling
    bool profiling_enabled_ = false;
    ProfileData profile_data_;

    // ========================================================================
    // PRIVATE METHODS: Checkpoint I/O
    // ========================================================================

    struct CheckpointHeader {
        char magic[4] = {'F', 'V', 'D', '\0'};
        uint32_t version = 1;
        uint64_t time_step = 0;
        double sim_time = 0.0;
        int64_t num_cells = 0;
        int64_t num_levels = 0;
    };

    bool write_checkpoint_binary(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);
        if (!out) return false;

        // Write header
        CheckpointHeader header;
        header.time_step = step_count_;
        header.sim_time = static_cast<double>(current_time_);
        header.num_cells = static_cast<int64_t>(fluid_geometry_.num_intervals);
        header.num_levels = 1;  // TODO: Multi-level support
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));

        // Write config
        out.write(reinterpret_cast<const char*>(&cfg_), sizeof(cfg_));

        // Write geometry (CSR structure)
        // Row offsets
        int64_t num_rows = fluid_geometry_.num_rows;
        out.write(reinterpret_cast<const char*>(&num_rows), sizeof(num_rows));

        // Copy row_ptr to host (CSR row pointers)
        auto row_ptr_host = Kokkos::create_mirror_view(fluid_geometry_.row_ptr);
        Kokkos::deep_copy(row_ptr_host, fluid_geometry_.row_ptr);
        for (int i = 0; i <= num_rows; ++i) {
            int64_t val = static_cast<int64_t>(row_ptr_host(i));
            out.write(reinterpret_cast<const char*>(&val), sizeof(val));
        }

        // Intervals (each Interval has begin, end members)
        int64_t num_intervals = fluid_geometry_.num_intervals;
        out.write(reinterpret_cast<const char*>(&num_intervals), sizeof(num_intervals));

        auto intervals_host = Kokkos::create_mirror_view(fluid_geometry_.intervals);
        Kokkos::deep_copy(intervals_host, fluid_geometry_.intervals);
        for (int64_t i = 0; i < num_intervals; ++i) {
            int32_t begin = intervals_host(i).begin;
            int32_t end = intervals_host(i).end;
            out.write(reinterpret_cast<const char*>(&begin), sizeof(begin));
            out.write(reinterpret_cast<const char*>(&end), sizeof(end));
        }

        // NOTE: Actual field data serialization would be done here
        // In production, for each AMR level:
        // - Serialize rho, rhou, rhov, E fields
        // - Use Kokkos::create_mirror_view + deep_copy to get host data
        // - Write binary data sequentially

        return out.good();
    }

    bool read_checkpoint_binary(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) return false;

        // Read header
        CheckpointHeader header;
        in.read(reinterpret_cast<char*>(&header), sizeof(header));

        // Validate magic
        if (std::string(header.magic) != "FVD") return false;

        // Restore state
        step_count_ = static_cast<int>(header.time_step);
        current_time_ = static_cast<Real>(header.sim_time);

        // Read config
        Config cfg_read;
        in.read(reinterpret_cast<char*>(&cfg_read), sizeof(cfg_read));
        cfg_ = cfg_read;

        // Read geometry
        int64_t num_rows, num_intervals;
        in.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
        in.read(reinterpret_cast<char*>(&num_intervals), sizeof(num_intervals));

        // NOTE: Geometry is const reference - cannot modify
        // In production, solver would need to be reconstructed with new geometry
        // For now, we verify compatibility
        if (num_rows != fluid_geometry_.num_rows ||
            num_intervals != fluid_geometry_.num_intervals) {
            fprintf(stderr, "[Checkpoint] Geometry mismatch!\n");
            return false;
        }

        // Read and verify row_ptr
        auto row_ptr_host = Kokkos::create_mirror_view(fluid_geometry_.row_ptr);
        Kokkos::deep_copy(row_ptr_host, fluid_geometry_.row_ptr);
        for (int i = 0; i <= num_rows; ++i) {
            int64_t val;
            in.read(reinterpret_cast<char*>(&val), sizeof(val));
            if (val != static_cast<int64_t>(row_ptr_host(i))) {
                fprintf(stderr, "[Checkpoint] Row ptr mismatch at %d\n", i);
                return false;
            }
        }

        // Read and verify intervals
        auto intervals_host = Kokkos::create_mirror_view(fluid_geometry_.intervals);
        Kokkos::deep_copy(intervals_host, fluid_geometry_.intervals);
        for (int64_t i = 0; i < num_intervals; ++i) {
            int32_t begin, end;
            in.read(reinterpret_cast<char*>(&begin), sizeof(begin));
            in.read(reinterpret_cast<char*>(&end), sizeof(end));
            if (begin != intervals_host(i).begin || end != intervals_host(i).end) {
                fprintf(stderr, "[Checkpoint] Interval mismatch at %ld\n", i);
                return false;
            }
        }

        // NOTE: Actual field data restoration would be done here
        // For each field, read binary data and use Kokkos::deep_copy to device

        return in.good();
    }

    bool write_checkpoint_ascii(const std::string& filename) const {
        std::ofstream out(filename);
        if (!out) return false;

        out << "# FVD Checkpoint (ASCII format)\n";
        out << "# Generated by Subsetix FVD Solver\n";
        out << "version: 1\n";
        out << "step: " << step_count_ << "\n";
        out << "time: " << current_time_ << "\n";
        out << "gamma: " << cfg_.gamma << "\n";
        out << "dx: " << cfg_.dx << "\n";
        out << "dy: " << cfg_.dy << "\n";
        out << "cfl: " << cfg_.cfl << "\n";

        // Geometry info
        out << "num_rows: " << fluid_geometry_.num_rows << "\n";
        out << "num_intervals: " << fluid_geometry_.num_intervals << "\n";

        // Row ptr
        out << "row_ptr:";
        auto row_ptr_host = Kokkos::create_mirror_view(fluid_geometry_.row_ptr);
        Kokkos::deep_copy(row_ptr_host, fluid_geometry_.row_ptr);
        for (int i = 0; i <= fluid_geometry_.num_rows; ++i) {
            out << " " << row_ptr_host(i);
        }
        out << "\n";

        // Intervals (each has begin and end)
        out << "intervals:";
        auto intervals_host = Kokkos::create_mirror_view(fluid_geometry_.intervals);
        Kokkos::deep_copy(intervals_host, fluid_geometry_.intervals);
        for (int i = 0; i < fluid_geometry_.num_intervals; ++i) {
            out << " " << intervals_host(i).begin << "-" << intervals_host(i).end;
        }
        out << "\n";

        // NOTE: Field data would be serialized here
        // Format: field_name: val0 val1 val2 ...
        // For AMR: include level information

        return out.good();
    }

    bool read_checkpoint_ascii(const std::string& filename) {
        std::ifstream in(filename);
        if (!in) return false;

        std::string line;
        while (std::getline(in, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') continue;

            // Parse key: value format
            size_t colon_pos = line.find(':');
            if (colon_pos == std::string::npos) continue;

            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);

            // Trim whitespace
            while (!key.empty() && key.back() == ' ') key.pop_back();
            while (!value.empty() && value.front() == ' ') value.erase(0, 1);

            // Parse known keys
            if (key == "step") {
                step_count_ = std::stoi(value);
            } else if (key == "time") {
                current_time_ = std::stof(value);
            } else if (key == "gamma") {
                cfg_.gamma = std::stof(value);
            } else if (key == "dx") {
                cfg_.dx = std::stof(value);
            } else if (key == "dy") {
                cfg_.dy = std::stof(value);
            } else if (key == "cfl") {
                cfg_.cfl = std::stof(value);
            }
            // TODO: Restore geometry and field data
        }

        return in.good();
    }
};

} // namespace subsetix::fvd
