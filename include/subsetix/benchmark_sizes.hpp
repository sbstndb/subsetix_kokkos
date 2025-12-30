#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace subsetix {
namespace benchmark {

// Bring Coord type into scope
using Coord = std::int32_t;

/**
 * @brief Standardized benchmark sizes for Subsetix
 *
 * This header defines consistent size constants across all benchmarks.
 * All benchmarks should use these constants to ensure comparability.
 *
 * Size ranges:
 * - TINY:    Fast smoke tests, minimal data
 * - SMALL:   Quick validation, small datasets
 * - MEDIUM:  Standard benchmark size
 * - LARGE:   Stress test for larger datasets
 * - XLARGE:  Performance testing on large datasets
 * - XXLARGE: Maximum scale testing
 */

// Interval set sizes (for CSR operations)
constexpr Coord kIntervalsTiny   = 128;
constexpr Coord kIntervalsSmall  = 512;
constexpr Coord kIntervalsMedium = 1280;
constexpr Coord kIntervalsLarge  = 12800;
constexpr Coord kIntervalsXLarge = 128000;
constexpr Coord kIntervalsXXLarge = 1280000;

// Field/Geometry sizes (for field operations)
constexpr Coord kFieldTiny    = 64;
constexpr Coord kFieldSmall   = 256;
constexpr Coord kFieldMedium  = 1024;
constexpr Coord kFieldLarge   = 2048;
constexpr Coord kFieldXLarge  = 4096;
constexpr Coord kFieldXXLarge = 8192;

// Helper functions for creating benchmark configurations
inline Coord get_interval_size(const char* size_name) {
    if (size_name == std::string("Tiny"))   return kIntervalsTiny;
    if (size_name == std::string("Small"))  return kIntervalsSmall;
    if (size_name == std::string("Medium")) return kIntervalsMedium;
    if (size_name == std::string("Large"))  return kIntervalsLarge;
    if (size_name == std::string("XLarge")) return kIntervalsXLarge;
    if (size_name == std::string("XXLarge")) return kIntervalsXXLarge;
    return kIntervalsMedium; // Default
}

inline Coord get_field_size(const char* size_name) {
    if (size_name == std::string("Tiny"))   return kFieldTiny;
    if (size_name == std::string("Small"))  return kFieldSmall;
    if (size_name == std::string("Medium")) return kFieldMedium;
    if (size_name == std::string("Large"))  return kFieldLarge;
    if (size_name == std::string("XLarge")) return kFieldXLarge;
    if (size_name == std::string("XXLarge")) return kFieldXXLarge;
    return kFieldMedium; // Default
}

} // namespace benchmark
} // namespace subsetix
