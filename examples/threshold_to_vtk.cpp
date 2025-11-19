#include <Kokkos_Core.hpp>
#include <cmath>
#include <string_view>

#include "example_output.hpp"

#include <subsetix/csr_field.hpp>
#include <subsetix/csr_field_ops.hpp>
#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_ops/threshold.hpp>
#include <subsetix/vtk_export.hpp>

namespace {
using namespace subsetix::csr;

// Function to evaluate: sin(x/10) * cos(y/10)
float field_function(Coord x, Coord y) {
  return std::sin(static_cast<float>(x) / 10.0f) * 
         std::cos(static_cast<float>(y) / 10.0f);
}

} // namespace

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  {
    using subsetix::vtk::write_legacy_quads;

    const auto output_dir =
        subsetix_examples::make_example_output_dir("threshold_to_vtk",
                                                   argc,
                                                   argv);
    const auto output_path = [&output_dir](std::string_view filename) {
      return subsetix_examples::output_file(output_dir, filename);
    };

    // 1. Create a dense field on a grid
    Box2D box;
    box.x_min = 0;
    box.x_max = 100;
    box.y_min = 0;
    box.y_max = 100;
    
    auto geom_dev = make_box_device(box);
    auto geom_host = build_host_from_device(geom_dev);

    // 2. Populate field values
    auto field_host = make_field_like_geometry<float>(geom_host, 0.0f);
    
    // Fill with function values
    for (std::size_t i = 0; i < field_host.num_rows(); ++i) {
      Coord y = field_host.row_keys[i].y;
      std::size_t start = field_host.row_ptr[i];
      std::size_t end = field_host.row_ptr[i + 1];
      
      for (std::size_t k = start; k < end; ++k) {
        auto& iv = field_host.intervals[k];
        std::size_t val_offset = iv.value_offset;
        for (Coord x = iv.begin; x < iv.end; ++x) {
          field_host.values[val_offset + (x - iv.begin)] = field_function(x, y);
        }
      }
    }

    // Export original field
    write_legacy_quads(field_host, output_path("original_field.vtk"), "value");

    auto field_dev = build_device_field_from_host(field_host);

    // 3. Apply thresholding
    // We want regions where |val| > 0.5
    float epsilon = 0.5f;
    auto threshold_set_dev = threshold_field(field_dev, static_cast<double>(epsilon));
    auto threshold_set_host = build_host_from_device(threshold_set_dev);

    // Export result geometry (mask)
    write_legacy_quads(threshold_set_host, output_path("threshold_mask.vtk"));

    // 4. Create a masked field to visualize only the kept values
    // We can reuse make_field_like_geometry but we need to copy values from original field
    // Or simpler: just re-evaluate or map. 
    // Let's create a field on the thresholded geometry and fill it with 1.0 for visualization
    auto mask_field_host = make_field_like_geometry<float>(threshold_set_host, 1.0f);
    write_legacy_quads(mask_field_host, output_path("threshold_mask_field.vtk"), "mask_val");
    
    // For better visualization, let's extract the actual values on the thresholded set.
    // Since we don't have a convenient "gather" function readily available in this context 
    // without writing a custom kernel or using apply_on_set, we can just show the mask geometry.
    // But wait, apply_on_set_device can copy values!
    
    // Create field on result geometry
    auto result_field_dev = build_device_field_from_host(
        make_field_like_geometry<float>(threshold_set_host, 0.0f));
        
    // Copy values from original field where mask is active
    auto threshold_src =
        make_subview(field_dev, threshold_set_dev, "threshold_src");
    auto threshold_dst =
        make_subview(result_field_dev, threshold_set_dev, "threshold_dst");
    copy_subview_device(threshold_dst, threshold_src);
    
    auto result_field_host = build_host_field_from_device(result_field_dev);
    write_legacy_quads(result_field_host, output_path("threshold_values.vtk"), "value");
  }

  Kokkos::finalize();
  return 0;
}
