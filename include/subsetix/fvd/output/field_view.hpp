#pragma once

#include <Kokkos_Core.hpp>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <cstdint>
#include <cstdio>

#include "../geometry/csr_types.hpp"
#include <subsetix/multilevel/multilevel.hpp>

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
// VTK EXPORT HELPER - Full Implementation
// ============================================================================

/**
 * @brief VTK export functionality for SolverOutput
 *
 * Provides easy visualization of simulation results with multi-field support.
 *
 * This implementation works directly with FVD types (FieldView, CSR geometry)
 * and writes VTK files without depending on the core library's VTK export.
 */
class VTKExporter {
public:
    /**
     * @brief Write single field to VTK format
     *
     * @param output Solver output containing geometry and fields
     * @param filename Output file path
     * @param field_name Name of the field to write (default: first field)
     * @param binary Use binary format (default: true for performance)
     */
    template<typename Real>
    static void write_legacy(const SolverOutput<Real>& output,
                             const std::string& filename,
                             const std::string& field_name = "",
                             bool binary = true) {
        if (!output.is_valid()) {
            throw std::runtime_error("VTKExporter: invalid output - missing geometry or fields");
        }

        // Determine which field to write
        const FieldView<Real>* field_to_write = nullptr;
        std::string actual_name = field_name;

        if (field_name.empty()) {
            // Use first field if no name specified
            if (output.fields.size() > 0) {
                field_to_write = &output.fields[0];
                actual_name = field_to_write->name();
            } else {
                throw std::runtime_error("VTKExporter: no fields in output");
            }
        } else {
            field_to_write = output.fields.get(field_name);
            if (!field_to_write) {
                throw std::runtime_error("VTKExporter: field '" + field_name + "' not found");
            }
        }

        write_single_field(output, *field_to_write, filename, actual_name, binary);
    }

    /**
     * @brief Write ALL fields to a single VTK file
     *
     * This is the recommended method for CFD simulations where you want
     * to visualize all conserved variables (rho, rhou, rhov, E) together.
     *
     * Creates a VTK file with multiple scalar fields (one per field in FieldSet).
     *
     * @param output Solver output containing multiple fields
     * @param filename Output file path
     * @param binary Use binary format (default: true)
     */
    template<typename Real>
    static void write_all_fields(const SolverOutput<Real>& output,
                                  const std::string& filename,
                                  bool binary = true) {
        if (!output.is_valid()) {
            throw std::runtime_error("VTKExporter: invalid output - missing geometry or fields");
        }

        if (output.fields.is_empty()) {
            throw std::runtime_error("VTKExporter: no fields to write");
        }

        write_multi_field(output, filename, binary);
    }

    /**
     * @brief Write time series (multiple timesteps to separate files)
     *
     * @param outputs Vector of solver outputs (one per timestep)
     * @param prefix File prefix (files named: prefix_step_0000.vtk, etc.)
     * @param binary Use binary format (default: true)
     */
    template<typename Real>
    static void write_time_series(const std::vector<SolverOutput<Real>>& outputs,
                                  const std::string& prefix,
                                  bool binary = true) {
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            std::string filename = prefix + "_step_" +
                                   std::to_string(i) + ".vtk";
            write_all_fields(outputs[i], filename, binary);
        }
    }

    /**
     * @brief Write multi-level AMR output to VTK
     *
     * Exports all AMR levels to a single VTK file with proper physical
     * coordinates and level indicator field.
     *
     * @param outputs Vector of solver outputs (one per AMR level)
     * @param multilevel_geo Multi-level geometry with physical metadata
     * @param filename Output file path
     * @param binary Use binary format (default: true)
     */
    template<typename Real>
    static void write_multilevel(const std::vector<SolverOutput<Real>>& outputs,
                                  const subsetix::MultilevelGeoHost& multilevel_geo,
                                  const std::string& filename,
                                  bool binary = true) {
        if (outputs.empty()) {
            throw std::runtime_error("VTKExporter: no outputs provided for multilevel export");
        }

        write_multilevel_impl(outputs, multilevel_geo, filename, binary);
    }

private:
    // ========================================================================
    // INTERNAL DATA STRUCTURES FOR HOST GEOMETRY
    // ========================================================================

    /**
     * @brief Simple host-side geometry representation for VTK export
     *
     * Stores CSR geometry in std::vector format for easy serialization.
     */
    struct HostGeometry {
        std::vector<int> row_y;           // Y coordinate for each row
        std::vector<std::size_t> row_ptr; // CSR row pointers
        std::vector<int> interval_begin;  // Interval begin coordinates
        std::vector<int> interval_end;    // Interval end coordinates

        std::size_t num_cells() const {
            std::size_t count = 0;
            for (std::size_t i = 0; i + 1 < row_ptr.size(); ++i) {
                for (std::size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                    count += (interval_end[k] - interval_begin[k]);
                }
            }
            return count;
        }
    };

    // ========================================================================
    // CONVERSION HELPERS
    // ========================================================================

    /**
     * @brief Convert FVD stub geometry to host representation
     */
    static HostGeometry convert_geometry_to_host(
        const csr::IntervalSet2DDevice& device_geom) {
        HostGeometry host_geom;

        if (device_geom.num_rows == 0) {
            return host_geom;
        }

        // Deep copy Kokkos views to host
        auto row_ptr_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, device_geom.row_ptr);
        auto intervals_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, device_geom.intervals);

        // Build row_y array
        host_geom.row_y.resize(device_geom.num_rows);
        for (std::size_t i = 0; i < device_geom.num_rows; ++i) {
            host_geom.row_y[i] = static_cast<int>(i);
        }

        // Copy row_ptr
        host_geom.row_ptr.resize(device_geom.num_rows + 1);
        for (std::size_t i = 0; i <= device_geom.num_rows; ++i) {
            host_geom.row_ptr[i] = static_cast<std::size_t>(row_ptr_host(i));
        }

        // Copy intervals (each Interval has begin, end members)
        std::size_t num_intervals = device_geom.num_intervals;
        host_geom.interval_begin.resize(num_intervals);
        host_geom.interval_end.resize(num_intervals);

        for (std::size_t k = 0; k < num_intervals; ++k) {
            host_geom.interval_begin[k] = intervals_host(k).begin;
            host_geom.interval_end[k] = intervals_host(k).end;
        }

        return host_geom;
    }

    /**
     * @brief Convert FieldView data to host std::vector
     */
    template<typename Real>
    static std::vector<float> convert_field_to_host(const FieldView<Real>& field) {
        std::vector<float> result(field.size());
        auto host_data = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, field.data());

        for (std::size_t i = 0; i < field.size(); ++i) {
            result[i] = static_cast<float>(host_data(i));
        }
        return result;
    }

    // ========================================================================
    // VTK WRITING IMPLEMENTATIONS
    // ========================================================================

    /**
     * @brief Write single field to VTK file
     */
    template<typename Real>
    static void write_single_field(const SolverOutput<Real>& output,
                                    const FieldView<Real>& field,
                                    const std::string& filename,
                                    const std::string& field_name,
                                    bool binary) {
        auto host_geom = convert_geometry_to_host(*output.geometry);
        auto field_data = convert_field_to_host(field);
        std::size_t num_cells = host_geom.num_cells();

        if (num_cells == 0) {
            write_empty_vtk(filename);
            return;
        }

        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error("VTKExporter: cannot open file: " + filename);
        }

        write_vtk_header(ofs, "Subsetix Field Output");
        write_vtk_points(ofs, host_geom, binary);
        write_vtk_cells(ofs, num_cells, binary);
        write_vtk_single_field_data(ofs, host_geom, field_data, field_name, binary);

        ofs.close();
    }

    /**
     * @brief Write multiple fields to single VTK file
     */
    template<typename Real>
    static void write_multi_field(const SolverOutput<Real>& output,
                                   const std::string& filename,
                                   bool binary) {
        auto host_geom = convert_geometry_to_host(*output.geometry);
        std::size_t num_cells = host_geom.num_cells();

        if (num_cells == 0) {
            write_empty_vtk(filename);
            return;
        }

        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error("VTKExporter: cannot open file: " + filename);
        }

        write_vtk_header(ofs, "Subsetix Multi-Field Output");
        write_vtk_points(ofs, host_geom, binary);
        write_vtk_cells(ofs, num_cells, binary);

        // Write all fields
        ofs << "CELL_DATA " << num_cells << "\n";
        for (std::size_t f = 0; f < output.fields.size(); ++f) {
            const auto& field = output.fields[f];
            auto field_data = convert_field_to_host(field);
            write_vtk_scalar_field(ofs, host_geom, field_data, field.name(), binary);
        }

        ofs.close();
    }

    /**
     * @brief Write multi-level AMR data to VTK
     */
    template<typename Real>
    static void write_multilevel_impl(const std::vector<SolverOutput<Real>>& outputs,
                                       const subsetix::MultilevelGeoHost& multilevel_geo,
                                       const std::string& filename,
                                       bool binary) {
        // Convert all levels to host geometry
        std::vector<HostGeometry> host_geoms;
        std::size_t total_cells = 0;

        for (const auto& output : outputs) {
            auto host_geo = convert_geometry_to_host(*output.geometry);
            total_cells += host_geo.num_cells();
            host_geoms.push_back(std::move(host_geo));
        }

        if (total_cells == 0) {
            write_empty_vtk(filename);
            return;
        }

        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error("VTKExporter: cannot open file: " + filename);
        }

        write_vtk_header(ofs, "Subsetix Multi-Level AMR Output");

        // Write points with physical coordinates
        ofs << "POINTS " << (total_cells * 4) << " float\n";
        for (std::size_t level_idx = 0; level_idx < host_geoms.size(); ++level_idx) {
            double dx = multilevel_geo.dx_at(level_idx);
            double dy = multilevel_geo.dy_at(level_idx);
            double ox = multilevel_geo.origin_x;
            double oy = multilevel_geo.origin_y;
            write_vtk_points_phys(ofs, host_geoms[level_idx], ox, oy, dx, dy, binary);
        }

        // Write cells
        ofs << "\nCELLS " << total_cells << " " << (total_cells * 5) << "\n";
        std::size_t vertex_offset = 0;
        for (const auto& host_geo : host_geoms) {
            write_vtk_cell_connectivity(ofs, host_geo, vertex_offset, binary);
            vertex_offset += host_geo.num_cells() * 4;
        }

        // Write cell types
        ofs << "\nCELL_TYPES " << total_cells << "\n";
        for (std::size_t i = 0; i < total_cells; ++i) {
            write_binary_int(ofs, 9);
        }
        ofs << "\n";

        // Write level indicator
        ofs << "CELL_DATA " << total_cells << "\n";
        ofs << "SCALARS Level int 1\n";
        ofs << "LOOKUP_TABLE default\n";
        for (std::size_t level_idx = 0; level_idx < host_geoms.size(); ++level_idx) {
            std::size_t ncells = host_geoms[level_idx].num_cells();
            for (std::size_t i = 0; i < ncells; ++i) {
                write_binary_int(ofs, static_cast<int>(level_idx));
            }
        }
        ofs << "\n";

        // TODO: Write actual field values for each level

        ofs.close();
    }

    // ========================================================================
    // LOW-LEVEL VTK WRITING HELPERS
    // ========================================================================

    static void write_empty_vtk(const std::string& filename) {
        std::ofstream ofs(filename);
        ofs << "# vtk DataFile Version 3.0\n";
        ofs << "Empty Subsetix Output\n";
        ofs << "ASCII\n";
        ofs << "DATASET UNSTRUCTURED_GRID\n";
        ofs << "POINTS 0 float\n";
        ofs << "CELLS 0 0\n";
        ofs << "CELL_TYPES 0\n";
    }

    static void write_vtk_header(std::ofstream& ofs, const char* title) {
        ofs << "# vtk DataFile Version 3.0\n";
        ofs << title << "\n";
        ofs << "BINARY\n";
        ofs << "DATASET UNSTRUCTURED_GRID\n";
    }

    static void write_vtk_points(std::ofstream& ofs, const HostGeometry& geom, bool binary) {
        std::size_t num_cells = geom.num_cells();
        ofs << "POINTS " << (num_cells * 4) << " float\n";

        for (std::size_t i = 0; i < geom.row_y.size(); ++i) {
            int y = geom.row_y[i];
            for (std::size_t k = geom.row_ptr[i]; k < geom.row_ptr[i + 1]; ++k) {
                int x0 = geom.interval_begin[k];
                int x1 = geom.interval_end[k];
                for (int x = x0; x < x1; ++x) {
                    float pts[12] = {
                        static_cast<float>(x), static_cast<float>(y), 0.0f,
                        static_cast<float>(x + 1), static_cast<float>(y), 0.0f,
                        static_cast<float>(x + 1), static_cast<float>(y + 1), 0.0f,
                        static_cast<float>(x), static_cast<float>(y + 1), 0.0f
                    };
                    for (float v : pts) write_binary_float(ofs, v);
                }
            }
        }
    }

    static void write_vtk_points_phys(std::ofstream& ofs, const HostGeometry& geom,
                                       double ox, double oy, double dx, double dy, bool binary) {
        for (std::size_t i = 0; i < geom.row_y.size(); ++i) {
            double y = oy + geom.row_y[i] * dy;
            for (std::size_t k = geom.row_ptr[i]; k < geom.row_ptr[i + 1]; ++k) {
                int nx = geom.interval_end[k] - geom.interval_begin[k];
                for (int n = 0; n < nx; ++n) {
                    double x = ox + (geom.interval_begin[k] + n) * dx;
                    float pts[12] = {
                        static_cast<float>(x), static_cast<float>(y), 0.0f,
                        static_cast<float>(x + dx), static_cast<float>(y), 0.0f,
                        static_cast<float>(x + dx), static_cast<float>(y + dy), 0.0f,
                        static_cast<float>(x), static_cast<float>(y + dy), 0.0f
                    };
                    for (float v : pts) write_binary_float(ofs, v);
                }
            }
        }
    }

    static void write_vtk_cells(std::ofstream& ofs, std::size_t num_cells, bool binary) {
        ofs << "\nCELLS " << num_cells << " " << (num_cells * 5) << "\n";
        for (std::size_t c = 0; c < num_cells; ++c) {
            std::uint32_t conn[5] = {4,
                static_cast<std::uint32_t>(c * 4),
                static_cast<std::uint32_t>(c * 4 + 1),
                static_cast<std::uint32_t>(c * 4 + 2),
                static_cast<std::uint32_t>(c * 4 + 3)};
            for (std::uint32_t v : conn) write_binary_int(ofs, v);
        }

        ofs << "\nCELL_TYPES " << num_cells << "\n";
        for (std::size_t i = 0; i < num_cells; ++i) {
            write_binary_int(ofs, 9);
        }
        ofs << "\n";
    }

    static void write_vtk_cell_connectivity(std::ofstream& ofs, const HostGeometry& geom,
                                             std::size_t& vertex_offset, bool binary) {
        for (std::size_t i = 0; i < geom.row_y.size(); ++i) {
            for (std::size_t k = geom.row_ptr[i]; k < geom.row_ptr[i + 1]; ++k) {
                int nx = geom.interval_end[k] - geom.interval_begin[k];
                for (int n = 0; n < nx; ++n) {
                    std::uint32_t conn[5] = {4,
                        static_cast<std::uint32_t>(vertex_offset),
                        static_cast<std::uint32_t>(vertex_offset + 1),
                        static_cast<std::uint32_t>(vertex_offset + 2),
                        static_cast<std::uint32_t>(vertex_offset + 3)};
                    vertex_offset += 4;
                    for (std::uint32_t v : conn) write_binary_int(ofs, v);
                }
            }
        }
    }

    static void write_vtk_single_field_data(std::ofstream& ofs, const HostGeometry& geom,
                                             const std::vector<float>& field_data,
                                             const std::string& field_name, bool binary) {
        std::size_t num_cells = geom.num_cells();
        ofs << "CELL_DATA " << num_cells << "\n";
        write_vtk_scalar_field(ofs, geom, field_data, field_name, binary);
    }

    static void write_vtk_scalar_field(std::ofstream& ofs, const HostGeometry& geom,
                                        const std::vector<float>& field_data,
                                        const std::string& field_name, bool binary) {
        ofs << "SCALARS " << field_name << " float 1\n";
        ofs << "LOOKUP_TABLE default\n";

        std::size_t data_idx = 0;
        for (std::size_t i = 0; i < geom.row_y.size(); ++i) {
            for (std::size_t k = geom.row_ptr[i]; k < geom.row_ptr[i + 1]; ++k) {
                int nx = geom.interval_end[k] - geom.interval_begin[k];
                for (int n = 0; n < nx; ++n) {
                    float val = (data_idx < field_data.size()) ? field_data[data_idx++] : 0.0f;
                    write_binary_float(ofs, val);
                }
            }
        }
        ofs << "\n";
    }

    // ========================================================================
    // BINARY I/O HELPERS (big-endian for VTK)
    // ========================================================================

    static void write_binary_float(std::ofstream& ofs, float value) {
        std::uint32_t* iptr = reinterpret_cast<std::uint32_t*>(&value);
        std::uint32_t big_endian = to_big_endian_32(*iptr);
        ofs.write(reinterpret_cast<const char*>(&big_endian), sizeof(big_endian));
    }

    static void write_binary_int(std::ofstream& ofs, std::uint32_t value) {
        std::uint32_t big_endian = to_big_endian_32(value);
        ofs.write(reinterpret_cast<const char*>(&big_endian), sizeof(big_endian));
    }

    static std::uint32_t to_big_endian_32(std::uint32_t value) {
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
        return value;
#else
        return ((value >> 24) & 0xFF) |
               ((value >> 8) & 0xFF00) |
               ((value << 8) & 0xFF0000) |
               ((value << 24) & 0xFF000000);
#endif
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
