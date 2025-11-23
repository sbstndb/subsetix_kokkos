// Minimal Kokkos profiling tool: logs kernel durations to a JSONL file.
#include <impl/Kokkos_Profiling_C_Interface.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {
using Clock = std::chrono::steady_clock;

std::mutex g_mutex;
std::ofstream g_out;
struct KernelInfo {
  Clock::time_point start;
  std::string name;
};

std::unordered_map<uint64_t, KernelInfo> g_kernel_info;
std::atomic<uint64_t> g_kernel_counter{0};

std::string output_path() {
  const char* env = std::getenv("KOKKOS_PROFILE_OUTPUT");
  if (env && *env) {
    return std::string(env);
  }
  return "kp_timeline.jsonl";
}

void log_event(const std::string& kind,
               const std::string& name,
               double duration_us) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_out.is_open()) return;
  g_out << "{\"kind\":\"" << kind << "\","
        << "\"name\":\"" << name << "\","
        << "\"duration_us\":" << duration_us << "}\n";
}

uint64_t record_start(const char* name) {
  const uint64_t id = g_kernel_counter.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(g_mutex);
  KernelInfo info;
  info.start = Clock::now();
  info.name = name ? std::string(name) : std::string();
  g_kernel_info[id] = std::move(info);
  return id;
}

void record_end(const std::string& kind,
                uint64_t kID) {
  Clock::time_point t0;
  std::string kname;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_kernel_info.find(kID);
    if (it == g_kernel_info.end()) {
      return;
    }
    t0 = it->second.start;
    kname = it->second.name;
    g_kernel_info.erase(it);
  }
  const auto dt =
      std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0)
          .count();
  const std::string& final_name = kname.empty() ? kind : kname;
  log_event(kind, final_name, static_cast<double>(dt));
}
} // namespace

extern "C" {

void kokkosp_init_library(const int /*loadSeq*/,
                          const uint64_t /*interfaceVer*/,
                          const uint64_t /*devID*/,
                          uint64_t* /*outDevID*/) {
  g_out.open(output_path(), std::ios::out | std::ios::trunc);
}

void kokkosp_finalize_library() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_out.is_open()) {
    g_out.flush();
    g_out.close();
  }
  g_kernel_info.clear();
}

void kokkosp_begin_parallel_for(const char* name,
                                const uint32_t /*devID*/,
                                uint64_t* kID) {
  const uint64_t id = record_start(name);
  if (kID) {
    *kID = id;
  }
}

void kokkosp_end_parallel_for(const uint64_t kID) {
  record_end("parallel_for", kID);
}

void kokkosp_begin_parallel_scan(const char* name,
                                 const uint32_t /*devID*/,
                                 uint64_t* kID) {
  const uint64_t id = record_start(name);
  if (kID) {
    *kID = id;
  }
}

void kokkosp_end_parallel_scan(const uint64_t kID) {
  record_end("parallel_scan", kID);
}

void kokkosp_begin_parallel_reduce(const char* name,
                                   const uint32_t /*devID*/,
                                   uint64_t* kID) {
  const uint64_t id = record_start(name);
  if (kID) {
    *kID = id;
  }
}

void kokkosp_end_parallel_reduce(const uint64_t kID) {
  record_end("parallel_reduce", kID);
}

void kokkosp_push_profile_region(const char* name) {
  const uint64_t id = record_start(name);
  // Reuse the counter slot to mark region start; store under a synthetic ID.
  std::lock_guard<std::mutex> lock(g_mutex);
  g_kernel_info[id] = KernelInfo{Clock::now(), name ? std::string(name) : std::string()};
  g_kernel_counter.fetch_add(1, std::memory_order_relaxed);
  log_event("push_region", name ? name : "", 0.0);
}

void kokkosp_pop_profile_region() {
  // We do not track nested region durations here; emit a placeholder.
  log_event("pop_region", "", 0.0);
}

// Stub allocations/fences to satisfy the interface.
void kokkosp_begin_fence(const char* /*name*/,
                         const uint32_t /*devID*/,
                         uint64_t* kID) {
  if (kID) *kID = record_start("fence");
}
void kokkosp_end_fence(const uint64_t kID) {
  record_end("fence", kID);
}

void kokkosp_allocate_data(const Kokkos_Profiling_SpaceHandle /*handle*/,
                           const char* /*label*/,
                           const void* /*ptr*/,
                           const uint64_t /*size*/) {}
void kokkosp_deallocate_data(const Kokkos_Profiling_SpaceHandle /*handle*/,
                             const char* /*label*/,
                             const void* /*ptr*/,
                             const uint64_t /*size*/) {}
}
