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

// PATCHED VERSION: Reports HWM periodically and at kernel end to work around Google Benchmark's std::_Exit(0)

#include <stdio.h>
#include <inttypes.h>
#include <execinfo.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <sys/time.h>
#include <cxxabi.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <atomic>
#include <mutex>
#include <iostream>
#include <fstream>

#include "kp_core.hpp"

namespace KokkosTools {
namespace HighwaterMark {

// darwin report rusage.ru_maxrss in bytes
#if defined(__APPLE__) || defined(__MACH__)
#define RU_MAXRSS_UNITS 1024
#else
#define RU_MAXRSS_UNITS 1
#endif

// Global state
static std::atomic<bool> initialized{false};
static std::atomic<uint64_t> kernel_count{0};
static std::atomic<long> current_hwm{0};
static std::mutex report_mutex;
static std::ofstream log_file;

// Get current HWM
long get_current_hwm() {
    struct rusage sys_resources;
    getrusage(RUSAGE_SELF, &sys_resources);
    return (long)sys_resources.ru_maxrss * RU_MAXRSS_UNITS;
}

// Print HWM report
void print_hwm_report(const char* context = "") {
    std::lock_guard<std::mutex> lock(report_mutex);
    long hwm = get_current_hwm();
    current_hwm.store(hwm);

    if (log_file.is_open()) {
        log_file << "[" << kernel_count.load() << "] " << context
                 << " HWM: " << hwm << " kB" << std::endl;
    }
}

void kokkosp_init_library(const int loadSeq, const uint64_t interfaceVer,
                          const uint32_t /*devInfoCount*/,
                          Kokkos_Profiling_KokkosPDeviceInfo* /*deviceInfo*/) {
    if (initialized.exchange(true)) {
        return;  // Already initialized
    }

    // Open log file
    char hostname[64];
    gethostname(hostname, 64);
    int pid = getpid();
    char filename[256];
    snprintf(filename, 256, "hwm_report_%s_%d.log", hostname, pid);
    log_file.open(filename);
    if (log_file.is_open()) {
        log_file << "KokkosP: High Water Mark Tracking Started" << std::endl;
        log_file << "KokkosP: Sequence: " << loadSeq << ", Version: " << interfaceVer << std::endl;
    }

    printf("KokkosP: High Water Mark Library Initialized (sequence is %d, version: %llu)\n",
           loadSeq, (unsigned long long)(interfaceVer));
    printf("KokkosP: HWM log file: %s\n", filename);

    // Initial HWM reading
    print_hwm_report("Initial");
}

void kokkosp_finalize_library() {
    printf("\n");
    printf("KokkosP: Finalization of profiling library.\n");

    long final_hwm = get_current_hwm();
    printf("\n");
    printf("KokkosP: High Water Mark Memory Report\n");
    printf("=====================================\n");
    printf("KokkosP: High water mark memory consumption: %li kB\n", final_hwm);
    printf("\n");

    if (log_file.is_open()) {
        log_file << "[FINAL] HWM: " << final_hwm << " kB" << std::endl;
        log_file << "Total kernels executed: " << kernel_count.load() << std::endl;
        log_file.close();
    }
}

// Track kernels to periodically report HWM
void kokkosp_begin_parallel_for(const char* name, const uint32_t dev_id,
                                 uint64_t* k_id) {
    kernel_count++;
    if (kernel_count % 10 == 0) {  // Report every 10 kernels
        print_hwm_report("");
    }
}

Kokkos::Tools::Experimental::EventSet get_event_set() {
    Kokkos::Tools::Experimental::EventSet my_event_set;
    memset(&my_event_set, 0, sizeof(my_event_set));
    my_event_set.init     = kokkosp_init_library;
    my_event_set.finalize = kokkosp_finalize_library;
    my_event_set.begin_parallel_for = kokkosp_begin_parallel_for;
    return my_event_set;
}

}  // namespace HighwaterMark
}  // namespace KokkosTools

extern "C" {

namespace impl = KokkosTools::HighwaterMark;

EXPOSE_INIT(impl::kokkosp_init_library)
EXPOSE_FINALIZE(impl::kokkosp_finalize_library)
EXPOSE_BEGIN_PARALLEL_FOR(impl::kokkosp_begin_parallel_for)

}  // extern "C"
