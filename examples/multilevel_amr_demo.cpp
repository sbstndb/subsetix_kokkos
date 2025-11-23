#include <cmath>
#include <iostream>
#include <string>

#include <Kokkos_Core.hpp>

#include <subsetix/geometry.hpp>
#include <subsetix/field.hpp>
#include <subsetix/multilevel/multilevel.hpp>
#include <subsetix/io.hpp>
#include <subsetix/csr_ops/amr.hpp>
#include <subsetix/csr_ops/field_amr.hpp>

using namespace subsetix;
using namespace subsetix::csr;

// Helper to allocate a field with the same geometry as an interval set
template <typename T>
Field2DDevice<T> make_field_like_device(const IntervalSet2DDevice& geom,
                                        T init_val = 0) {
  auto geom_host = build_host_from_device(geom);
  auto field_host = make_field_like_geometry<T>(geom_host, init_val);
  return build_device_field_from_host(field_host);
}

void run_demo() {
  std::cout << "Running Multilevel AMR Demo..." << std::endl;

  // 1. Setup Geometry: A base disk refined locally
  // Level 0: [-2, 2] x [-2, 2] covered by 40x40 cells => dx = 0.1
  MultilevelGeoDevice geo;
  geo.origin_x = -2.0;
  geo.origin_y = -2.0;
  geo.root_dx = 0.1;
  geo.root_dy = 0.1;
  geo.num_active_levels = 3;

  CsrSetAlgebraContext ctx;

  // --- Level 0 ---
  // Base Disk
  Disk2D disk0;
  disk0.cx = 20;
  disk0.cy = 20;
  disk0.radius = 15;
  
  std::cout << "Creating Level 0 (Base Disk)..." << std::endl;
  geo.levels[0] = make_disk_device(disk0);
  std::cout << "L0 rows: " << geo.levels[0].num_rows << std::endl;
  
  // --- Level 1 ---
  // Refine L0 -> L1_full, then intersect with a smaller central disk
  std::cout << "Creating Level 1 (Refining L0 + Intersection)..." << std::endl;
  IntervalSet2DDevice l1_full;
  refine_level_up_device(geo.levels[0], l1_full, ctx);
  
  // Center of L1 is at (40, 40) because coords doubled
  // Radius 15 refined becomes 30. We want a smaller radius, say 20 (in fine coords).
  Disk2D disk1;
  disk1.cx = 40;
  disk1.cy = 40;
  disk1.radius = 20;
  IntervalSet2DDevice mask_l1 = make_disk_device(disk1);
  
  // Allocate output with sufficient capacity (upper bound = l1_full)
  geo.levels[1] = allocate_interval_set_device(l1_full.num_rows, l1_full.num_intervals);
  set_intersection_device(l1_full, mask_l1, geo.levels[1], ctx);
  std::cout << "L1 rows: " << geo.levels[1].num_rows << std::endl;
  
  // --- Level 2 ---
  // Refine L1 -> L2_full, then intersect with an even smaller central disk
  std::cout << "Creating Level 2 (Refining L1 + Intersection)..." << std::endl;
  IntervalSet2DDevice l2_full;
  refine_level_up_device(geo.levels[1], l2_full, ctx);
  
  // Center of L2 is at (80, 80)
  // Radius 20 refined becomes 40. We want smaller, say 25.
  Disk2D disk2;
  disk2.cx = 80;
  disk2.cy = 80;
  disk2.radius = 25;
  IntervalSet2DDevice mask_l2 = make_disk_device(disk2);
  
  // Allocate output
  geo.levels[2] = allocate_interval_set_device(l2_full.num_rows, l2_full.num_intervals);
  set_intersection_device(l2_full, mask_l2, geo.levels[2], ctx);
  std::cout << "L2 rows: " << geo.levels[2].num_rows << std::endl;
  
  // 2. Setup Fields
  MultilevelFieldDevice<float> field;
  field.num_active_levels = 3;
  
  // Initialize fields with geometry
  field.levels[0] = make_field_like_device<float>(geo.levels[0]);
  field.levels[1] = make_field_like_device<float>(geo.levels[1]);
  field.levels[2] = make_field_like_device<float>(geo.levels[2]);
  
  // 3. Initialize Level 0 with a function
  // f(x,y) = exp(- (x^2 + y^2))
  std::cout << "Initializing Field on Level 0..." << std::endl;
  auto f0 = field.levels[0];
  double ox = geo.origin_x;
  double oy = geo.origin_y;
  double dx = geo.root_dx;

  auto row_keys = f0.geometry.row_keys;
  auto row_ptr = f0.geometry.row_ptr;
  auto intervals = f0.geometry.intervals;
  auto offsets = f0.geometry.cell_offsets;
  auto values = f0.values;
  
  Kokkos::parallel_for("init_field_l0", f0.geometry.num_rows,
    KOKKOS_LAMBDA(const int i) {
       Coord y_idx = row_keys(i).y;
       double y = oy + y_idx * dx;
       
       std::size_t begin = row_ptr(i);
       std::size_t end = row_ptr(i+1);
       
       for (std::size_t k=begin; k<end; ++k) {
         const auto iv = intervals(k);
         std::size_t offset = offsets(k);
         for (Coord x_idx = iv.begin; x_idx < iv.end; ++x_idx) {
           double x = ox + x_idx * dx;
           float val = static_cast<float>(std::exp(-(x*x + y*y)));
           values(offset + static_cast<std::size_t>(x_idx - iv.begin)) = val;
         }
       }
    });
  Kokkos::fence();

  // 4. Prolong to L1
  std::cout << "Prolonging to Level 1..." << std::endl;
  // We treat the geometry of L1 as the "mask" where we want values
  auto level1_sub = make_subview(field.levels[1], geo.levels[1], "level1");
  prolong_field_subview_device(level1_sub, field.levels[0], ctx);
  
  // 5. Prolong to L2
  std::cout << "Prolonging to Level 2..." << std::endl;
  auto level2_sub = make_subview(field.levels[2], geo.levels[2], "level2");
  prolong_field_subview_device(level2_sub, field.levels[1], ctx);
  
  // 6. Export
  std::cout << "Exporting to VTK..." << std::endl;
  
  auto h_geo = deep_copy_to_host(geo);
  auto h_field = deep_copy_to_host(field);
  
  // Export geometry with Level scalar
  subsetix::vtk::write_multilevel_vtk(h_geo, "demo_multilevel_geo.vtk");
  
  // Export field with physical coordinates and Temperature scalar
  subsetix::vtk::write_multilevel_field_vtk(h_field, h_geo, "demo_multilevel_field.vtk", "Temperature");
  
  std::cout << "Done. Output files: demo_multilevel_geo.vtk, demo_multilevel_field.vtk" << std::endl;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  try {
    run_demo();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
