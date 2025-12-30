#pragma once

#include <Kokkos_Core.hpp>
#include <functional>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

namespace subsetix::fvd {

// ============================================================================
// SOLVER STATE FOR OBSERVERS
// ============================================================================

/**
 * @brief Snapshot of solver state at a given time
 *
 * Passed to observers/callbacks so they can monitor the simulation.
 */
template<typename Real = float>
struct SolverState {
    // Time information
    Real time = Real(0);
    Real dt = Real(0);
    int step = 0;

    // AMR information
    int max_level = 0;
    std::size_t total_cells = 0;
    std::size_t cells_per_level[10] = {0};  // Up to 10 levels

    // Performance information
    double wall_time = 0.0;      // Total wall clock time (seconds)
    double step_time = 0.0;      // Time for last step (seconds)

    // Remesh information (only valid after remesh)
    bool did_remesh = false;
    std::size_t old_cells = 0;
    std::size_t new_cells = 0;

    // Residual information (if computed)
    Real residual_rho = Real(0);
    Real residual_momentum = Real(0);
    Real residual_energy = Real(0);
};

// ============================================================================
// OBSERVER EVENTS
// ============================================================================

/**
 * @brief Enum of all observable events in the solver
 */
enum class SolverEvent : int {
    SimulationStart = 0,
    SimulationEnd,
    StepBegin,
    StepEnd,
    RemeshBegin,
    RemeshEnd,
    OutputWritten,
    Error,
    CustomEvent
};

/**
 * @brief Convert event to string
 */
inline const char* to_string(SolverEvent event) {
    switch (event) {
        case SolverEvent::SimulationStart: return "SimulationStart";
        case SolverEvent::SimulationEnd: return "SimulationEnd";
        case SolverEvent::StepBegin: return "StepBegin";
        case SolverEvent::StepEnd: return "StepEnd";
        case SolverEvent::RemeshBegin: return "RemeshBegin";
        case SolverEvent::RemeshEnd: return "RemeshEnd";
        case SolverEvent::OutputWritten: return "OutputWritten";
        case SolverEvent::Error: return "Error";
        case SolverEvent::CustomEvent: return "CustomEvent";
        default: return "Unknown";
    }
}

// ============================================================================
// CALLBACK SIGNATURES
// ============================================================================

/**
 * @brief Generic callback function type
 *
 * Callbacks receive:
 * - The event that triggered them
 * - The current solver state
 * - Optional user data pointer
 */
template<typename Real>
using SolverCallback = std::function<void(SolverEvent, const SolverState<Real>&)>;

/**
 * @brief Progress callback: called after each step
 */
template<typename Real>
using ProgressCallback = std::function<void(const SolverState<Real>&)>;

/**
 * @brief Remesh callback: called before and after remeshing
 */
template<typename Real>
using RemeshCallback = std::function<void(const SolverState<Real>&, std::size_t old_cells, std::size_t new_cells)>;

/**
 * @brief Error callback: called when an error occurs
 */
using ErrorCallback = std::function<void(SolverEvent, const std::string& error_message)>;

// ============================================================================
// OBSERVER MANAGER
// ============================================================================

/**
 * @brief Manages multiple observers/callbacks for a solver
 *
 * Allows users to attach callbacks to specific events without
 * modifying the solver code.
 *
 * Thread-safe: callbacks are invoked sequentially.
 */
template<typename Real = float>
class ObserverManager {
public:
    // ========================================================================
    // CONSTRUCTORS
    // ========================================================================

    ObserverManager() = default;
    ~ObserverManager() = default;

    // Disable copy, enable move
    ObserverManager(const ObserverManager&) = delete;
    ObserverManager& operator=(const ObserverManager&) = delete;
    ObserverManager(ObserverManager&&) = default;
    ObserverManager& operator=(ObserverManager&&) = default;

    // ========================================================================
    // REGISTER CALLBACKS
    // ========================================================================

    /**
     * @brief Register a callback for a specific event
     *
     * @param event Event to listen for
     * @param callback Function to call when event occurs
     * @return ID of this callback (can be used to remove it later)
     */
    int add_callback(SolverEvent event, SolverCallback<Real> callback) {
        int id = next_id_++;
        callbacks_[event].push_back({id, std::move(callback)});
        return id;
    }

    /**
     * @brief Register a progress callback (called after each step)
     *
     * Convenience method for StepEnd event.
     */
    int on_progress(ProgressCallback<Real> callback) {
        return add_callback(SolverEvent::StepEnd, [cb = std::move(callback)](SolverEvent, const SolverState<Real>& state) {
            cb(state);
        });
    }

    /**
     * @brief Register a remesh callback
     *
     * Called after remeshing completes.
     */
    int on_remesh(RemeshCallback<Real> callback) {
        return add_callback(SolverEvent::RemeshEnd, [cb = std::move(callback)](SolverEvent, const SolverState<Real>& state) {
            cb(state, state.old_cells, state.new_cells);
        });
    }

    /**
     * @brief Register an error callback
     */
    int on_error(ErrorCallback callback) {
        return add_callback(SolverEvent::Error, [cb = std::move(callback)](SolverEvent event, const SolverState<Real>&) {
            std::string msg = "Error event: ";
            msg += to_string(event);
            cb(event, msg);
        });
    }

    /**
     * @brief Remove a callback by ID
     */
    bool remove_callback(int id) {
        for (auto& [event, cbs] : callbacks_) {
            auto it = std::remove_if(cbs.begin(), cbs.end(),
                [id](const CallbackInfo& cb) { return cb.id == id; });
            if (it != cbs.end()) {
                cbs.erase(it, cbs.end());
                return true;
            }
        }
        return false;
    }

    // ========================================================================
    // TRIGGER CALLBACKS
    // ========================================================================

    /**
     * @brief Notify all observers of an event
     *
     * @param event Event that occurred
     * @param state Current solver state
     */
    void notify(SolverEvent event, const SolverState<Real>& state) {
        auto it = callbacks_.find(event);
        if (it != callbacks_.end()) {
            for (const auto& [id, callback] : it->second) {
                try {
                    callback(event, state);
                } catch (const std::exception& e) {
                    // Don't let one bad callback break others
                    fprintf(stderr, "[Observer] Callback %d threw exception: %s\n",
                            id, e.what());
                } catch (...) {
                    fprintf(stderr, "[Observer] Callback %d threw unknown exception\n", id);
                }
            }
        }
    }

    /**
     * @brief Convenience: notify with default state
     */
    void notify(SolverEvent event) {
        SolverState<Real> state;
        notify(event, state);
    }

    // ========================================================================
    // UTILITY
    // ========================================================================

    /**
     * @brief Clear all callbacks
     */
    void clear() {
        callbacks_.clear();
    }

    /**
     * @brief Get number of registered callbacks
     */
    std::size_t count() const {
        std::size_t total = 0;
        for (const auto& [event, cbs] : callbacks_) {
            total += cbs.size();
        }
        return total;
    }

    /**
     * @brief Check if any callbacks are registered
     */
    bool empty() const {
        return count() == 0;
    }

private:
    struct CallbackInfo {
        int id;
        SolverCallback<Real> callback;
    };

    int next_id_ = 0;
    std::unordered_map<SolverEvent, std::vector<CallbackInfo>> callbacks_;
};

// ============================================================================
// PREDEFINED OBSERVERS (CONVENIENCE)
// ============================================================================

/**
 * @brief Built-in observers for common use cases
 */
class Observers {
public:
    /**
     * @brief Create a progress printer callback
     *
     * Prints: "Step 100: t=0.0500, dt=0.0005, 1000 cells, 2 levels"
     */
    template<typename Real = float>
    static SolverCallback<Real> progress_printer(int print_interval = 1) {
        return [print_interval](SolverEvent event, const SolverState<Real>& state) {
            if (event == SolverEvent::StepEnd && state.step % print_interval == 0) {
                printf("Step %d: t=%.5f, dt=%.5f, %zu cells, %d levels\n",
                       state.step, static_cast<double>(state.time),
                       static_cast<double>(state.dt),
                       state.total_cells, state.max_level + 1);
            }
        };
    }

    /**
     * @brief Create a simple time logger
     */
    template<typename Real = float>
    static SolverCallback<Real> time_logger() {
        return [](SolverEvent event, const SolverState<Real>& state) {
            if (event == SolverEvent::StepEnd) {
                printf("  Step time: %.3f ms, Total: %.2f s\n",
                       state.step_time * 1000.0, state.wall_time);
            }
        };
    }

    /**
     * @brief Create a residual monitor
     */
    template<typename Real = float>
    static SolverCallback<Real> residual_monitor(Real threshold = Real(1e-6)) {
        return [threshold](SolverEvent event, const SolverState<Real>& state) {
            if (event == SolverEvent::StepEnd) {
                Real max_res = Kokkos::max(state.residual_rho,
                                Kokkos::max(state.residual_momentum,
                                            state.residual_energy));
                if (max_res < threshold) {
                    printf("Converged! Max residual: %.2e < %.2e\n",
                           static_cast<double>(max_res),
                           static_cast<double>(threshold));
                }
            }
        };
    }

    /**
     * @brief Create a remesh reporter
     */
    template<typename Real = float>
    static SolverCallback<Real> remesh_reporter() {
        return [](SolverEvent event, const SolverState<Real>& state) {
            if (event == SolverEvent::RemeshEnd) {
                double change = 100.0 * (static_cast<double>(state.new_cells) -
                                        static_cast<double>(state.old_cells)) /
                                static_cast<double>(state.old_cells);
                printf("Remesh: %zu -> %zu cells (%.1f%% change)\n",
                       state.old_cells, state.new_cells, change);
            }
        };
    }

    /**
     * @brief Create a CSV logger for post-processing
     *
     * Writes: step,time,dt,cells,levels,residual_rho
     */
    template<typename Real = float>
    static std::function<void(SolverEvent, const SolverState<Real>&)>
    csv_logger(const std::string& filename) {
        // Open file and write header
        auto file = std::make_shared<FILE*>(fopen(filename.c_str(), "w"));
        if (*file) {
            fprintf(*file, "step,time,dt,cells,levels,residual_rho\n");
            fflush(*file);
        }

        return [file](SolverEvent event, const SolverState<Real>& state) {
            if (event == SolverEvent::StepEnd && *file) {
                fprintf(*file, "%d,%.6f,%.6f,%zu,%d,%.2e\n",
                        state.step,
                        static_cast<double>(state.time),
                        static_cast<double>(state.dt),
                        state.total_cells,
                        state.max_level + 1,
                        static_cast<double>(state.residual_rho));
                fflush(*file);
            } else if (event == SolverEvent::SimulationEnd && *file) {
                fclose(*file);
            }
        };
    }
};

// ============================================================================
// TYPE ALIASES
// ============================================================================

using ObserverManagerf = ObserverManager<float>;
using ObserverManagerd = ObserverManager<double>;
using SolverStatef = SolverState<float>;
using SolverStated = SolverState<double>;

} // namespace subsetix::fvd
