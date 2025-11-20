#include <Kokkos_Core.hpp>

#include "example_output.hpp"

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/vtk_export.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

using namespace subsetix::csr;

Kokkos::View<std::uint8_t**, HostMemorySpace>
load_pbm(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    return {};
  }

  std::string magic;
  in >> magic;
  if (magic != "P1") {
    return {};
  }

  auto skip_comments = [&]() {
    while (in.peek() == '#') {
      std::string tmp;
      std::getline(in, tmp);
    }
  };

  skip_comments();

  std::size_t width = 0;
  std::size_t height = 0;
  in >> width >> height;
  if (width == 0 || height == 0) {
    return {};
  }

  Kokkos::View<std::uint8_t**, HostMemorySpace> mask(
      "pbm_mask_host", height, width);

  for (std::size_t y = 0; y < height; ++y) {
    for (std::size_t x = 0; x < width; ++x) {
      skip_comments();
      int bit = 0;
      in >> bit;
      mask(y, x) = static_cast<std::uint8_t>(bit ? 1 : 0);
    }
  }

  return mask;
}

Kokkos::View<std::uint8_t**, HostMemorySpace>
fallback_smiley() {
  constexpr std::size_t width = 16;
  constexpr std::size_t height = 12;
  const std::uint8_t data[height][width] = {
      {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
      {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
      {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0},
      {0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0},
      {0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0},
      {1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1},
      {1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1},
      {0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0},
      {0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0},
      {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0},
      {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0}};

  Kokkos::View<std::uint8_t**, HostMemorySpace> mask(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "pbm_mask_fallback"),
      height, width);
  for (std::size_t y = 0; y < height; ++y) {
    for (std::size_t x = 0; x < width; ++x) {
      mask(y, x) = data[y][x];
    }
  }
  return mask;
}

} // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  {
    using subsetix::vtk::write_legacy_quads;

    const auto output_dir =
        subsetix_examples::make_example_output_dir("csr_bitmap_to_vtk",
                                                   argc,
                                                   argv);
    const auto output_path = [&output_dir](std::string_view filename) {
      return subsetix_examples::output_file(output_dir, filename);
    };

    const std::filesystem::path asset_path =
        std::filesystem::path(__FILE__).parent_path() / "output.pbm";

    auto h_mask = load_pbm(asset_path);
    if (h_mask.extent(0) == 0 || h_mask.extent(1) == 0) {
      // Fallback to embedded smiley if the user-supplied PBM is missing.
      h_mask = fallback_smiley();
    }

    // Flip Y so that PBM top row maps to highest Y visually (avoid upside-down render).
    Kokkos::View<std::uint8_t**, HostMemorySpace> h_mask_flipped(
        "pbm_mask_flipped", h_mask.extent(0), h_mask.extent(1));
    for (std::size_t y = 0; y < h_mask.extent(0); ++y) {
      const std::size_t src_y = h_mask.extent(0) - 1 - y;
      for (std::size_t x = 0; x < h_mask.extent(1); ++x) {
        h_mask_flipped(y, x) = h_mask(src_y, x);
      }
    }

    auto d_mask = Kokkos::create_mirror_view_and_copy(
        DeviceMemorySpace{}, h_mask_flipped);

    auto geom_dev = make_bitmap_device(d_mask, 0, 0, 1);
    auto geom_host = build_host_from_device(geom_dev);
    write_legacy_quads(geom_host, output_path("smiley.vtk"));
  }
  return 0;
}
