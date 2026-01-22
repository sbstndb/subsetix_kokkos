//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

// PATCHED VERSION: Writes memory tracking data periodically to work around Google Benchmark's std::_Exit(0)

#include <cstdio>
#include <inttypes.h>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <cstdlib>
#include <fstream>

#include <sys/resource.h>
#include <unistd.h>
#include <chrono>

#include "kp_core.hpp"

namespace KokkosTools {
namespace MemoryUsage {

char space_name[16][64];

int num_spaces;
std::vector<std::tuple<double, uint64_t, double> > space_size_track[16];
uint64_t space_size[16];

static std::mutex m;
static std::atomic<bool> initialized{false};
static std::atomic<uint64_t> alloc_count{0};
static std::string log_filename;
static std::string hostname_str;

// Simple timer using chrono
static std::chrono::high_resolution_clock::time_point start_time;

double get_elapsed_seconds() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();
}

double max_mem_usage() {
  struct rusage app_info;
  getrusage(RUSAGE_SELF, &app_info);
  const double max_rssKB = app_info.ru_maxrss;
  return max_rssKB * 1024;
}

// Write memory usage report periodically
void write_memory_report() {
    char hostname[64];
    gethostname(hostname, 64);
    int pid = getpid();

    for (int s = 0; s < num_spaces; s++) {
        char fileOutput[256];
        snprintf(fileOutput, 256, "%s-%d-%s.memspace_usage", hostname, pid,
                 space_name[s]);

        FILE* ofile = fopen(fileOutput, "wb");
        if (ofile) {
            fprintf(ofile, "# Space %s\n", space_name[s]);
            fprintf(ofile,
                    "# Time(s)  Size(MB)   HighWater(MB)   HighWater-Process(MB)\n");
            uint64_t maxvalue = 0;
            for (unsigned int i = 0; i < space_size_track[s].size(); i++) {
                if (std::get<1>(space_size_track[s][i]) > maxvalue)
                    maxvalue = std::get<1>(space_size_track[s][i]);
                fprintf(ofile, "%lf %.1lf %.1lf %.1lf\n",
                        std::get<0>(space_size_track[s][i]),
                        1.0 * std::get<1>(space_size_track[s][i]) / 1024 / 1024,
                        1.0 * maxvalue / 1024 / 1024,
                        1.0 * std::get<2>(space_size_track[s][i]) / 1024 / 1024);
            }
            fclose(ofile);
        }
    }
}

void kokkosp_init_library(const int /*loadSeq*/,
                          const uint64_t /*interfaceVer*/,
                          const uint32_t /*devInfoCount*/,
                          Kokkos_Profiling_KokkosPDeviceInfo* /*deviceInfo*/) {
    if (initialized.exchange(true)) {
        return;  // Already initialized
    }

    char hostname[64];
    gethostname(hostname, 64);
    hostname_str = hostname;

    num_spaces = 0;
    for (int i = 0; i < 16; i++) space_size[i] = 0;

    start_time = std::chrono::high_resolution_clock::now();

    printf("KokkosP: Memory Usage Library Initialized (periodic writes enabled)\n");
}

void kokkosp_finalize_library() {
    printf("\nKokkosP: Finalization of memory usage profiling library.\n");
    write_memory_report();
    printf("KokkosP: Memory usage reports written to files\n");
}

void kokkosp_allocate_data(const SpaceHandle space, const char* /*label*/,
                           const void* const /*ptr*/, const uint64_t size) {
  std::lock_guard<std::mutex> lock(m);

  double time = get_elapsed_seconds();

  int space_i = num_spaces;
  for (int s = 0; s < num_spaces; s++)
    if (strcmp(space_name[s], space.name) == 0) space_i = s;

  if (space_i == num_spaces) {
    strncpy(space_name[num_spaces], space.name, 64);
    num_spaces++;
  }
  space_size[space_i] += size;
  space_size_track[space_i].push_back(
      std::make_tuple(time, space_size[space_i], max_mem_usage()));

  // Write report every 100 allocations to ensure data is saved
  alloc_count++;
  if (alloc_count % 100 == 0) {
      write_memory_report();
  }
}

void kokkosp_deallocate_data(const SpaceHandle space, const char* /*label*/,
                             const void* const /*ptr*/, const uint64_t size) {
  std::lock_guard<std::mutex> lock(m);

  double time = get_elapsed_seconds();

  int space_i = num_spaces;
  for (int s = 0; s < num_spaces; s++)
    if (strcmp(space_name[s], space.name) == 0) space_i = s;

  if (space_i == num_spaces) {
    strncpy(space_name[num_spaces], space.name, 64);
    num_spaces++;
  }
  if (space_size[space_i] >= size) {
    space_size[space_i] -= size;
    space_size_track[space_i].push_back(
        std::make_tuple(time, space_size[space_i], max_mem_usage()));
  }

  // Write report every 100 deallocations
  alloc_count++;
  if (alloc_count % 100 == 0) {
      write_memory_report();
  }
}

Kokkos::Tools::Experimental::EventSet get_event_set() {
  Kokkos::Tools::Experimental::EventSet my_event_set;
  memset(&my_event_set, 0,
         sizeof(my_event_set));  // zero any pointers not set here
  my_event_set.init            = kokkosp_init_library;
  my_event_set.finalize        = kokkosp_finalize_library;
  my_event_set.allocate_data   = kokkosp_allocate_data;
  my_event_set.deallocate_data = kokkosp_deallocate_data;
  return my_event_set;
}

}  // namespace MemoryUsage
}  // namespace KokkosTools

extern "C" {

namespace impl = KokkosTools::MemoryUsage;

EXPOSE_INIT(impl::kokkosp_init_library)
EXPOSE_FINALIZE(impl::kokkosp_finalize_library)
EXPOSE_ALLOCATE(impl::kokkosp_allocate_data)
EXPOSE_DEALLOCATE(impl::kokkosp_deallocate_data)

}  // extern "C"
