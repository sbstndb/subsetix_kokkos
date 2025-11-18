#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_field.hpp>
#include <subsetix/vtk_export.hpp>

namespace {

using namespace subsetix::csr;

TEST(VTKExportTest, WriteLegacyQuadsContent) {
    // 1. Create a simple geometry: 2x2 square from (0,0) to (2,2)
    // Row 0: [0, 2)
    // Row 1: [0, 2)
    IntervalSet2DHost geom;
    geom.row_keys = {RowKey2D{0}, RowKey2D{1}};
    geom.row_ptr = {0, 1, 2, 0}; // Corrected below
    // Actually row_ptr size is num_rows + 1.
    // interval 0 for row 0, interval 1 for row 1.
    geom.row_ptr = {0, 1, 2};
    geom.intervals = {Interval{0, 2}, Interval{0, 2}};

    // 2. Create a field with values
    // Interval 0 (y=0, x=[0,2)): values 1.0, 2.0
    // Interval 1 (y=1, x=[0,2)): values 3.0, 4.0
    IntervalField2DHost<float> field = make_field_like_geometry<float>(geom, 0.0f);
    
    // Fill values
    // FieldInterval 0 corresponds to geom interval 0
    field.values[0] = 10.0f; // (0,0)
    field.values[1] = 20.0f; // (1,0)
    // FieldInterval 1 corresponds to geom interval 1
    field.values[2] = 30.0f; // (0,1)
    field.values[3] = 40.0f; // (1,1)

    const std::string filename = "test_vtk_export.vtk";
    
    // 3. Export
    subsetix::vtk::write_legacy_quads(field, filename, "test_scalar");

    // 4. Read back and verify
    std::ifstream ifs(filename);
    ASSERT_TRUE(ifs.is_open()) << "Failed to open exported VTK file";

    std::string line;
    std::vector<std::string> lines;
    while (std::getline(ifs, line)) {
        lines.push_back(line);
    }
    ifs.close();
    std::remove(filename.c_str());

    // Expected structure:
    // # vtk DataFile Version 3.0
    // Subsetix CSR field
    // ASCII
    // DATASET UNSTRUCTURED_GRID
    // POINTS 16 float  <-- 4 cells, 4 points each = 16 points (naive implementation duplicates points)
    // ... points coords ...
    // CELLS 4 20      <-- 4 cells, each takes 1 + 4 integers = 5 per cell -> 20 integers
    // ... cell connectivity ...
    // CELL_TYPES 4
    // ... 9 9 9 9 ...
    // CELL_DATA 4
    // SCALARS test_scalar float 1
    // LOOKUP_TABLE default
    // 10
    // 20
    // 30
    // 40
    
    // Validate Header
    ASSERT_GE(lines.size(), 10);
    EXPECT_EQ(lines[0], "# vtk DataFile Version 3.0");
    EXPECT_EQ(lines[1], "Subsetix CSR field");
    EXPECT_EQ(lines[2], "ASCII");
    EXPECT_EQ(lines[3], "DATASET UNSTRUCTURED_GRID");

    // The implementation in vtk_export.hpp writes 4 points per cell.
    // Total cells = 4 (2x2)
    // Total points = 16
    EXPECT_EQ(lines[4], "POINTS 16 float");
    
    // Skip 16 lines of points
    size_t current_line = 5 + 16;
    
    // CELLS 4 20
    ASSERT_LT(current_line, lines.size());
    EXPECT_EQ(lines[current_line], "CELLS 4 20");
    current_line++;
    
    // Skip 4 lines of cells
    current_line += 4;

    // CELL_TYPES 4
    ASSERT_LT(current_line, lines.size());
    EXPECT_EQ(lines[current_line], "CELL_TYPES 4");
    current_line++;
    
    // Skip 4 lines of cell types (each is '9')
    current_line += 4;

    // CELL_DATA 4
    ASSERT_LT(current_line, lines.size());
    EXPECT_EQ(lines[current_line], "CELL_DATA 4");
    current_line++;

    // SCALARS ...
    ASSERT_LT(current_line, lines.size());
    EXPECT_EQ(lines[current_line], "SCALARS test_scalar float 1");
    current_line++;

    // LOOKUP_TABLE default
    ASSERT_LT(current_line, lines.size());
    EXPECT_EQ(lines[current_line], "LOOKUP_TABLE default");
    current_line++;

    // Values
    // We expect 4 values. The order depends on iteration order in write_legacy_quads.
    // It iterates rows, then intervals, then x.
    // So:
    // Row 0, x=0 -> 10.0
    // Row 0, x=1 -> 20.0
    // Row 1, x=0 -> 30.0
    // Row 1, x=1 -> 40.0
    
    std::vector<std::string> expected_values = {"10", "20", "30", "40"};
    for (const auto& val_str : expected_values) {
        ASSERT_LT(current_line, lines.size());
        // Float formatting might vary, but for integer-like floats typically it's simple.
        // However, the stream output might simply be "10" or "10.0".
        // Let's parse it to float to be safe.
        float val = std::stof(lines[current_line]);
        float expected = std::stof(val_str);
        EXPECT_FLOAT_EQ(val, expected);
        current_line++;
    }
}

} // namespace

