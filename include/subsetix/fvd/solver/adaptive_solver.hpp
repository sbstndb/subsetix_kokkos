#pragma once

#include <Kokkos_Core.hpp>
#include <chrono>
#include <string>
#include <fstream>
#include <cstring>
#include <cstdint>
#include "../system/concepts_v2.hpp"
#include "../system/euler2d.hpp"
#include "../flux/flux_schemes.hpp"
#include "../reconstruction/reconstruction.hpp"
#include "boundary_generic.hpp"
#include "observer.hpp"
#include "../output/field_view.hpp"
#include "../geometry/csr_types.hpp"
#include "../sources/source_terms.hpp"

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
 *
 * C++20: Constrained with concepts for better error messages
 */
template<
    FiniteVolumeSystem System,
    typename Reconstruction = reconstruction::NoReconstruction,
    template<typename> class FluxScheme = flux::RusanovFlux
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
        // Initialize stub state
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
        // Initialize stub state
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

    // ========================================================================
    // INITIALIZATION
    // ========================================================================

    /**
     * @brief Initialize with uniform state
     */
    void initialize(const Primitive& initial) {
        // Stub: would initialize all fields with initial state
        current_time_ = Real(0);
        step_count_ = 0;
    }

    // ========================================================================
    // TIME STEPPING
    // ========================================================================

    /**
     * @brief Perform one global time step
     *
     * Returns the actual dt used (based on CFL condition)
     */
    Real step() {
        // Stub: would perform full AMR step
        // For now, just compute a fake dt
        Real dt = cfg_.cfl * Kokkos::min(cfg_.dx, cfg_.dy) / Real(2);
        current_time_ += dt;
        ++step_count_;
        return dt;
    }

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
     * @brief Checkpoint file format
     */
    enum class CheckpointFormat {
        Binary,    // Custom binary format (fast, portable)
        ASCII,     // Human-readable text format
        HDF5       // HDF5 format (if available)
    };

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
    // VALIDATION (NEW: Runtime stability checks)
    // ========================================================================

    /**
     * @brief Enable validation checks
     *
     * Checks for:
     * - Negative density/pressure
     * - NaN/Inf values
     * - CFL violation
     * - Mach number > specified limit
     */
    struct ValidationConfig {
        bool check_negative_density = true;
        bool check_negative_pressure = true;
        bool check_nan = true;
        bool check_cfl = true;
        Real max_mach = Real(100);  // Warn if Mach > this
        Real min_pressure = Real(1e-10);
        Real min_density = Real(1e-10);
        bool throw_on_error = false;  // Throw exception instead of warning
        bool abort_on_error = false;  // Abort simulation on error
    };

    void set_validation(const ValidationConfig& cfg) {
        validation_ = cfg;
        validation_enabled_ = true;
    }

    void disable_validation() {
        validation_enabled_ = false;
    }

    /**
     * @brief Get validation statistics
     */
    struct ValidationStats {
        int nan_count = 0;
        int negative_density_count = 0;
        int negative_pressure_count = 0;
        int cfl_violations = 0;
        Real max_mach_seen = Real(0);
        bool is_valid = true;
    };

    const ValidationStats& validation_stats() const {
        return validation_stats_;
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

private:
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
        // TODO: num_cells from actual data
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));

        // TODO: Write actual field data
        // In production, serialize:
        // - All conserved variables at each level
        // - Geometry (CSR structure)
        // - AMR hierarchy
        // - Config

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

        // TODO: Read actual field data

        return in.good();
    }

    bool write_checkpoint_ascii(const std::string& filename) const {
        std::ofstream out(filename);
        if (!out) return false;

        out << "# FVD Checkpoint (ASCII format)\n";
        out << "version: 1\n";
        out << "step: " << step_count_ << "\n";
        out << "time: " << current_time_ << "\n";
        out << "gamma: " << cfg_.gamma << "\n";
        // TODO: Write actual data

        return out.good();
    }

    bool read_checkpoint_ascii(const std::string& filename) {
        std::ifstream in(filename);
        if (!in) return false;

        std::string line, key;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            // Parse key: value
            // TODO: Proper parsing

            if (line.find("step:") == 0) {
                step_count_ = std::stoi(line.substr(6));
            } else if (line.find("time:") == 0) {
                current_time_ = std::stof(line.substr(6));
            }
        }

        return in.good();
    }
};

} // namespace subsetix::fvd
