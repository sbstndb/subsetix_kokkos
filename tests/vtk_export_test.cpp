#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <array>

#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/geometry/csr_set_ops.hpp>
#include <subsetix/field/csr_field.hpp>
#include <subsetix/field/csr_field_ops.hpp>
#include <subsetix/io/vtk_export.hpp>

namespace {

using namespace subsetix::csr;

// Helper to read big-endian float from binary stream
inline float read_be_float(std::ifstream& ifs) {
    std::array<unsigned char, 4> bytes{};
    ifs.read(reinterpret_cast<char*>(bytes.data()), 4);
    std::reverse(bytes.begin(), bytes.end());
    float v;
    std::memcpy(&v, bytes.data(), 4);
    return v;
}

TEST(VTKExportTest, WriteLegacyQuadsBinaryContent) {
    // 1. Create a simple geometry: 2x2 square from (0,0) to (2,2)
    auto geom = make_interval_set_host(
        {{0}, {1}},           // row_keys
        {0, 1, 2},            // row_ptr
        {{0, 2}, {0, 2}}      // intervals
    );

    // 2. Create a field with values
    IntervalField2DHost<float> field = make_field_like_geometry<float>(geom, 0.0f);
    field.values[0] = 10.0f; // (0,0)
    field.values[1] = 20.0f; // (1,0)
    field.values[2] = 30.0f; // (0,1)
    field.values[3] = 40.0f; // (1,1)

    const std::string filename = "test_vtk_export.vtk";

    // 3. Export (now binary-only)
    subsetix::vtk::write_legacy_quads(field, filename, "test_scalar");

    // 4. Read back and verify header (ASCII portion)
    std::ifstream ifs(filename, std::ios::binary);
    ASSERT_TRUE(ifs.is_open()) << "Failed to open exported VTK file";

    std::string line;
    std::getline(ifs, line);
    EXPECT_EQ(line, "# vtk DataFile Version 3.0");

    std::getline(ifs, line);
    EXPECT_EQ(line, "Subsetix CSR field");

    std::getline(ifs, line);
    EXPECT_EQ(line, "BINARY");

    std::getline(ifs, line);
    EXPECT_EQ(line, "DATASET UNSTRUCTURED_GRID");

    std::getline(ifs, line);
    EXPECT_EQ(line, "POINTS 16 float");

    // Skip binary point data (16 points * 3 floats * 4 bytes = 192 bytes + newline)
    ifs.seekg(192, std::ios::cur);
    ifs.get(); // consume newline

    std::getline(ifs, line);
    EXPECT_EQ(line, "CELLS 4 20");

    // Skip binary cell data (4 cells * 5 uint32 * 4 bytes = 80 bytes + newline)
    ifs.seekg(80, std::ios::cur);
    ifs.get();

    std::getline(ifs, line);
    EXPECT_EQ(line, "CELL_TYPES 4");

    // Skip binary cell types (4 * 4 bytes = 16 bytes + newline)
    ifs.seekg(16, std::ios::cur);
    ifs.get();

    std::getline(ifs, line);
    EXPECT_EQ(line, "CELL_DATA 4");

    std::getline(ifs, line);
    EXPECT_EQ(line, "SCALARS test_scalar float 1");

    std::getline(ifs, line);
    EXPECT_EQ(line, "LOOKUP_TABLE default");

    // Read binary scalar values (4 floats in big-endian)
    std::vector<float> expected_values = {10.0f, 20.0f, 30.0f, 40.0f};
    for (float expected : expected_values) {
        float val = read_be_float(ifs);
        EXPECT_FLOAT_EQ(val, expected);
    }

    ifs.close();
    std::remove(filename.c_str());
}

} // namespace

