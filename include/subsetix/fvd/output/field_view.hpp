#pragma once

#include <Kokkos_Core.hpp>
#include <string>
#include <vector>
#include <memory>
#include "../geometry/csr_types.hpp"

namespace subsetix::fvd {

// ============================================================================
// FIELD VIEW WITH OWNERSHIP SEMANTICS
// ============================================================================

/**
 * @brief View of a single scalar field with ownership semantics
 *
 * Unlike the raw pointer approach in Views, FieldView owns its data
 * and manages the lifetime properly.
 *
 * This solves the dangling pointer problem and makes ownership explicit.
 */
template<typename Real = float>
class FieldView {
public:
    // ========================================================================
    // CONSTRUCTORS
    // ========================================================================

    /**
     * @brief Construct empty field view
     */
    FieldView() = default;

    /**
     * @brief Construct from existing Kokkos view (takes ownership)
     */
    explicit FieldView(Kokkos::View<Real*> data,
                       std::string name = "field",
                       int level = 0)
        : data_(data)
        , name_(std::move(name))
        , level_(level)
        , owns_data_(true)
    {}

    /**
     * @brief Construct with size allocation
     */
    static FieldView allocate(std::string name, std::size_t size, int level = 0) {
        return FieldView(Kokkos::View<Real*>(name + "_data", size),
                         std::move(name), level);
    }

    /**
     * @brief Non-owning view (reference to existing data)
     */
    static FieldView view(Kokkos::View<Real*> data,
                          std::string name,
                          int level = 0) {
        return FieldView(data, std::move(name), level);
    }

    // ========================================================================
    // ACCESSORS
    // ========================================================================

    /**
     * @brief Get underlying Kokkos view
     */
    const Kokkos::View<Real*>& data() const { return data_; }
    Kokkos::View<Real*>& data() { return data_; }

    /**
     * @brief Get field name
     */
    const std::string& name() const { return name_; }

    /**
     * @brief Get AMR level
     */
    int level() const { return level_; }

    /**
     * @brief Check if empty
     */
    bool is_empty() const {
        return data_.size() == 0;
    }

    /**
     * @brief Get data size
     */
    std::size_t size() const {
        return data_.size();
    }

    // ========================================================================
    // HOST ACCESS
    // ========================================================================

    /**
     * @brief Get host mirror (synchronizes device->host)
     */
    std::vector<Real> to_host() const {
        if (data_.size() == 0) return {};

        auto host_data = Kokkos::create_mirror_view(data_);
        Kokkos::deep_copy(host_data, data_);

        std::vector<Real> result(data_.size());
        for (std::size_t i = 0; i < data_.size(); ++i) {
            result[i] = host_data(i);
        }
        return result;
    }

    /**
     * @brief Set from host data (synchronizes host->device)
     */
    void from_host(const std::vector<Real>& host_data) {
        if (data_.size() != host_data.size()) {
            throw std::runtime_error("FieldView::from_host: size mismatch");
        }

        auto host_view = Kokkos::create_mirror_view(data_);
        for (std::size_t i = 0; i < host_data.size(); ++i) {
            host_view(i) = host_data[i];
        }
        Kokkos::deep_copy(data_, host_view);
    }

    // ========================================================================
    // DEVICE ACCESS (Kernels)
    // ========================================================================

    KOKKOS_INLINE_FUNCTION
    Real operator()(int i) const {
        return data_(i);
    }

    KOKKOS_INLINE_FUNCTION
    Real& operator()(int i) {
        return data_(i);
    }

private:
    Kokkos::View<Real*> data_;
    std::string name_;
    int level_ = 0;
    bool owns_data_ = true;
};

// ============================================================================
// MULTI-FIELD CONTAINER
// ============================================================================

/**
 * @brief Container for multiple fields (e.g., all conserved variables)
 *
 * Provides convenient access to all fields at a given AMR level.
 */
template<typename Real = float>
class FieldSet {
public:
    FieldSet() = default;

    /**
     * @brief Add a field to the set
     */
    void add(FieldView<Real> field) {
        fields_.push_back(std::move(field));
    }

    /**
     * @brief Get field by index
     */
    const FieldView<Real>& operator[](std::size_t i) const {
        return fields_.at(i);
    }

    FieldView<Real>& operator[](std::size_t i) {
        return fields_.at(i);
    }

    /**
     * @brief Get field by name
     */
    const FieldView<Real>* get(const std::string& name) const {
        for (const auto& f : fields_) {
            if (f.name() == name) return &f;
        }
        return nullptr;
    }

    FieldView<Real>* get(const std::string& name) {
        for (auto& f : fields_) {
            if (f.name() == name) return &f;
        }
        return nullptr;
    }

    /**
     * @brief Number of fields
     */
    std::size_t size() const { return fields_.size(); }

    /**
     * @brief Check if empty
     */
    bool is_empty() const { return fields_.empty(); }

    /**
     * @brief Iterator support
     */
    auto begin() const { return fields_.begin(); }
    auto end() const { return fields_.end(); }
    auto begin() { return fields_.begin(); }
    auto end() { return fields_.end(); }

private:
    std::vector<FieldView<Real>> fields_;
};

// ============================================================================
// OUTPUT STRUCTURE WITH GEOMETRY REFERENCE
// ============================================================================

/**
 * @brief Complete output including fields and geometry
 *
 * This replaces the old Views struct with proper ownership.
 */
template<typename Real = float>
struct SolverOutput {
    // Fields at finest level
    FieldSet<Real> fields;

    // Geometry reference (non-owning)
    const csr::IntervalSet2DDevice* geometry = nullptr;

    // AMR level
    int level = 0;

    // Simulation time
    Real time = Real(0);

    // ========================================================================
    // CONVENIENCE ACCESSORS
    // ========================================================================

    /**
     * @brief Get field by name
     */
    const FieldView<Real>* get_field(const std::string& name) const {
        return fields.get(name);
    }

    FieldView<Real>* get_field(const std::string& name) {
        return fields.get(name);
    }

    // ========================================================================
    // UTILITY METHODS
    // ========================================================================

    /**
     * @brief Check if output is valid
     */
    bool is_valid() const {
        return !fields.is_empty() && geometry != nullptr;
    }
};

// ============================================================================
// VTK EXPORT HELPER
// ============================================================================

/**
 * @brief VTK export functionality for SolverOutput
 *
 * Provides easy visualization of simulation results.
 */
class VTKExporter {
public:
    /**
     * @brief Write fields to VTK format
     *
     * @param output Solver output to write
     * @param filename Output file path
     * @param binary Use binary format (default: ASCII for compatibility)
     */
    template<typename Real>
    static void write_legacy(const SolverOutput<Real>& output,
                             const std::string& filename,
                             bool binary = false) {
        if (!output.is_valid()) {
            throw std::runtime_error("VTKExporter: invalid output");
        }

        // Stub: would write actual VTK file
        // In production:
        // 1. Open file
        // 2. Write VTK header
        // 3. Write geometry (from CSR intervals)
        // 4. Write cell data for each field
        // 5. Close file

        printf("[VTK] Would write to: %s (%zu fields, level %d)\n",
               filename.c_str(), output.fields.size(), output.level);
    }

    /**
     * @brief Write time series
     */
    template<typename Real>
    static void write_time_series(const std::vector<SolverOutput<Real>>& outputs,
                                  const std::string& prefix) {
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            std::string filename = prefix + "_step_" +
                                   std::to_string(i) + ".vtk";
            write_legacy(outputs[i], filename);
        }
    }
};

// ============================================================================
// TYPE ALIASES
// ============================================================================

using FieldViewf = FieldView<float>;
using FieldViewd = FieldView<double>;
using FieldSetf = FieldSet<float>;
using FieldSetd = FieldSet<double>;
using SolverOutputf = SolverOutput<float>;
using SolverOutputd = SolverOutput<double>;

} // namespace subsetix::fvd
