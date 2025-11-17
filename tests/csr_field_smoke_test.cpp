#include <Kokkos_Core.hpp>

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_field.hpp>
#include <subsetix/vtk_export.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  int result = 0;
  {
    using namespace subsetix::csr;

    // Build a simple host field with two rows and a few values.
    IntervalField2DHost<float> host_field;
    host_field.append_interval(0, 0, std::vector<float>{1.f, 2.f, 3.f});
    host_field.append_interval(2, 5, std::vector<float>{4.f, 5.f});

    if (host_field.num_rows() != 2 ||
        host_field.num_intervals() != 2 ||
        host_field.value_count() != 5) {
      result = 1;
    }

    if (result == 0) {
      auto dev_field = build_device_field_from_host(host_field);

      if (dev_field.num_rows != host_field.num_rows() ||
          dev_field.num_intervals != host_field.num_intervals() ||
          dev_field.value_count != host_field.value_count()) {
        result = 1;
      } else {
        // Simple device-side operation: scale all values by 2.
        using ExecSpace = Kokkos::DefaultExecutionSpace;
        Kokkos::parallel_for(
            "subsetix_csr_field_scale_values",
            Kokkos::RangePolicy<ExecSpace>(0, dev_field.value_count),
            KOKKOS_LAMBDA(const std::size_t i) {
              dev_field.values(i) *= 2.0f;
            });
        ExecSpace().fence();

        auto roundtrip = build_host_field_from_device(dev_field);

        if (roundtrip.num_rows() != host_field.num_rows() ||
            roundtrip.num_intervals() != host_field.num_intervals() ||
            roundtrip.value_count() != host_field.value_count()) {
          result = 1;
        } else {
          for (std::size_t i = 0; i < host_field.value_count(); ++i) {
            const float expected = host_field.values[i] * 2.0f;
            if (roundtrip.values[i] != expected) {
              result = 1;
              break;
            }
          }
        }
      }
    }

    if (result == 0) {
      // Build a simple box geometry and a constant field on top of it,
      // then export to VTK to ensure the API is usable.
      Box2D box;
      box.x_min = 0;
      box.x_max = 4;
      box.y_min = 0;
      box.y_max = 2;

      auto geom_dev = make_box_device(box);
      auto geom_host = build_host_from_device(geom_dev);

      auto field_host =
          make_field_like_geometry<float>(geom_host, 1.0f);

      subsetix::vtk::write_legacy_quads(field_host,
                                        "csr_field_box.vtk",
                                        "value");
    }
  }

  Kokkos::finalize();
  return result;
}

