#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <subsetix/multilevel.hpp>
#include <subsetix/csr_ops/amr.hpp>
#include <subsetix/io.hpp>

using namespace subsetix;
using namespace subsetix::csr;

// Test basic multilevel structure construction and refinement
TEST(Multilevel, SmokeTest) {
  // 1. Create a base geometry (Level 0) : A box [0, 10) x [0, 10)
  Box2D box_l0{0, 10, 0, 10};
  IntervalSet2DDevice mesh_l0 = make_box_device(box_l0);

  // 2. Refine to Level 1 using the real AMR operator
  // Note: This requires this test to be linked in an executable where 
  // ODR violations for refine_level_up_device kernels are avoided.
  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice mesh_l1;
  refine_level_up_device(mesh_l0, mesh_l1, ctx);
  Kokkos::fence();

  // 3. Setup MultilevelGeo on Device
  MultilevelGeoDevice multi_mesh;
  
  multi_mesh.origin_x = -5.0;
  multi_mesh.origin_y = -5.0;
  multi_mesh.root_dx = 1.0;
  multi_mesh.root_dy = 1.0;

  multi_mesh.levels[0] = mesh_l0;
  multi_mesh.levels[1] = mesh_l1;
  multi_mesh.num_active_levels = 2;

  // 4. Deep copy to Host for validation and export
  MultilevelGeoHost host_mesh = deep_copy_to_host(multi_mesh);

  ASSERT_EQ(host_mesh.num_active_levels, 2);
  ASSERT_EQ(host_mesh.levels[0].num_rows, 10);
  ASSERT_EQ(host_mesh.levels[1].num_rows, 20); // Refined 2x

  // 5. Export to VTK (Manual verification possible)
  subsetix::vtk::write_multilevel_vtk(host_mesh, "multilevel_smoke_test.vtk");
}

// Functor for kernel access test (standard Kokkos practice for tests)
struct AccessTestFunctor {
  MultilevelGeoDevice m;
  Kokkos::View<int*, DeviceMemorySpace> result;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int) const {
      if (m.num_active_levels == 1 && m.levels[0].num_rows > 0) {
        result(0) = 1;
      }
  }
};

TEST(Multilevel, KernelAccessTest) {
  MultilevelGeoDevice m;
  m.num_active_levels = 1;
  m.levels[0] = make_box_device(Box2D{0, 1, 0, 1});

  Kokkos::View<int*, DeviceMemorySpace> result("result", 1);

  AccessTestFunctor functor{m, result};

  Kokkos::parallel_for("test_multilevel_access", 
    Kokkos::RangePolicy<>(0, 1), functor);
  
  Kokkos::fence();
  auto h_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result);
  
  ASSERT_EQ(h_result(0), 1) << "Kernel failed to read MultilevelGeo structure";
}
