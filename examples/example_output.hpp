#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace subsetix_examples {

inline std::filesystem::path make_example_output_dir(std::string_view example_name,
                                                      int argc,
                                                      char* argv[]) {
  std::string name(example_name);
  std::filesystem::path output_dir = std::filesystem::path("examples_output") / name;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--output-dir" && i + 1 < argc) {
      output_dir = argv[++i];
    }
  }
  std::filesystem::create_directories(output_dir);
  return output_dir;
}

inline std::string output_file(const std::filesystem::path& dir,
                               std::string_view filename) {
  return (dir / std::string(filename)).string();
}

} // namespace subsetix_examples
