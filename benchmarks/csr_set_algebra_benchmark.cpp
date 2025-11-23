#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>
#include <chrono>

#include <subsetix/geometry.hpp>

using namespace subsetix::csr;

namespace {

struct RectConfig {
  Coord x_min;
  Coord x_max;
  Coord y_min;
  Coord y_max;
};

IntervalSet2DDevice make_box(const RectConfig& cfg) {
  Box2D box;
  box.x_min = cfg.x_min;
  box.x_max = cfg.x_max;
  box.y_min = cfg.y_min;
  box.y_max = cfg.y_max;
  return make_box_device(box);
}

void make_intersection_configs(Coord extent,
                               RectConfig& a_cfg,
                               RectConfig& b_cfg) {
  a_cfg.x_min = 0;
  a_cfg.x_max = extent;
  a_cfg.y_min = 0;
  a_cfg.y_max = extent;
  const Coord offset = extent / 4;
  b_cfg.x_min = offset;
  b_cfg.x_max = offset + extent;
  b_cfg.y_min = offset;
  b_cfg.y_max = offset + extent;
}

std::size_t overlap_rows(const RectConfig& a_cfg,
                         const RectConfig& b_cfg) {
  const Coord y_begin =
      (a_cfg.y_min > b_cfg.y_min) ? a_cfg.y_min : b_cfg.y_min;
  const Coord y_end =
      (a_cfg.y_max < b_cfg.y_max) ? a_cfg.y_max : b_cfg.y_max;
  return (y_end > y_begin)
             ? static_cast<std::size_t>(y_end - y_begin)
             : 0;
}

constexpr Coord kSizeTiny = 128;
constexpr Coord kSizeMedium = 1280;
constexpr Coord kSizeLarge = 12800;
constexpr Coord kSizeXLarge = 128000;
constexpr Coord kSizeXXLarge = 1280000;

void bench_box_construction(benchmark::State& state,
                            const RectConfig& cfg) {
  const double intervals =
      static_cast<double>(cfg.y_max - cfg.y_min);
  double total_seconds = 0.0;

  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    auto box = make_box(cfg);
    const auto t1 = std::chrono::steady_clock::now();
    total_seconds += std::chrono::duration<double>(t1 - t0).count();

    benchmark::DoNotOptimize(box.row_keys.data());
    benchmark::DoNotOptimize(box.row_ptr.data());
    benchmark::DoNotOptimize(box.intervals.data());
  }

  if (intervals > 0.0) {
    const double total_intervals =
        intervals * static_cast<double>(state.iterations());
    state.counters["ns_per_interval"] =
        (total_seconds / total_intervals) * 1e9;
  }
}

template <typename Kernel>
void run_row_kernel(benchmark::State& state,
                    std::size_t rows,
                    Kernel&& kernel,
                    const char* counter_name) {
  if (rows == 0) {
    state.SkipWithMessage("No overlapping rows");
    return;
  }

  double total_seconds = 0.0;
  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    kernel();
    Kokkos::fence();
    const auto t1 = std::chrono::steady_clock::now();
    total_seconds += std::chrono::duration<double>(t1 - t0).count();
  }

  state.counters[counter_name] =
      (total_seconds /
       (static_cast<double>(rows) *
        static_cast<double>(state.iterations()))) *
      1e9;
}

template <class BinaryOp>
void bench_binary_op(benchmark::State& state,
                     const RectConfig& a_cfg,
                     const RectConfig& b_cfg,
                     BinaryOp op) {
  IntervalSet2DDevice A = make_box(a_cfg);
  IntervalSet2DDevice B = make_box(b_cfg);

  const std::size_t intervals_in =
      A.num_intervals + B.num_intervals;

  // Preallocate an output buffer large enough for union in the worst case.
  IntervalSet2DDevice out;
  const std::size_t rows_cap = A.num_rows + B.num_rows;
  const std::size_t intervals_cap =
      A.num_intervals + B.num_intervals;

  if (rows_cap > 0) {
    out.row_keys = IntervalSet2DDevice::RowKeyView(
        "subsetix_csr_bench_out_row_keys", rows_cap);
    out.row_ptr = IntervalSet2DDevice::IndexView(
        "subsetix_csr_bench_out_row_ptr", rows_cap + 1);
  }
  if (intervals_cap > 0) {
    out.intervals = IntervalSet2DDevice::IntervalView(
        "subsetix_csr_bench_out_intervals", intervals_cap);
  }

  out.num_rows = 0;
  out.num_intervals = 0;

  CsrSetAlgebraContext ctx;

  double total_seconds = 0.0;

  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    op(A, B, out, ctx);
    const auto t1 = std::chrono::steady_clock::now();

    const std::chrono::duration<double> dt = t1 - t0;
    total_seconds += dt.count();

    benchmark::DoNotOptimize(out.row_keys.data());
    benchmark::DoNotOptimize(out.row_ptr.data());
    benchmark::DoNotOptimize(out.intervals.data());
  }

  if (intervals_in > 0) {
    const double total_intervals =
        static_cast<double>(intervals_in) *
        static_cast<double>(state.iterations());
    const double seconds_per_interval =
        total_seconds / total_intervals;
    state.counters["ns_per_interval"] =
        seconds_per_interval * 1e9;
  }
}

void BM_CSRIntersection_Tiny(benchmark::State& state) {
  RectConfig a{0, 128, 0, 128};          // 128x128
  const Coord N = 128;
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N}; // carré 128x128
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_intersection_device(A, B, out, ctx);
  });
}

void BM_CSRIntersection_Medium(benchmark::State& state) {
  const Coord N = 1280;
  RectConfig a{0, N, 0, N};              // 1280x1280
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N}; // carré 1280x1280
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_intersection_device(A, B, out, ctx);
  });
}

void BM_CSRIntersection_Large(benchmark::State& state) {
  const Coord N = 12800;
  RectConfig a{0, N, 0, N};              // 12800x12800
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N}; // carré 12800x12800
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_intersection_device(A, B, out, ctx);
  });
}

void BM_CSRIntersection_XLarge(benchmark::State& state) {
  const Coord N = 128000;
  RectConfig a{0, N, 0, N};              // 128000x128000
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N}; // carré 128000x128000
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_intersection_device(A, B, out, ctx);
  });
}

void BM_CSRIntersection_XXLarge(benchmark::State& state) {
  const Coord N = kSizeXXLarge;
  RectConfig a{0, N, 0, N};
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N};
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_intersection_device(A, B, out, ctx);
  });
}

void BM_CSRUnion_Tiny(benchmark::State& state) {
  RectConfig a{0, 128, 0, 128};
  const Coord N = 128;
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N};
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_union_device(A, B, out, ctx);
  });
}

void BM_CSRUnion_Medium(benchmark::State& state) {
  const Coord N = 1280;
  RectConfig a{0, N, 0, N};
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N};
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_union_device(A, B, out, ctx);
  });
}

void BM_CSRUnion_Large(benchmark::State& state) {
  const Coord N = 12800;
  RectConfig a{0, N, 0, N};
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N};
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_union_device(A, B, out, ctx);
  });
}

void BM_CSRUnion_XLarge(benchmark::State& state) {
  const Coord N = 128000;
  RectConfig a{0, N, 0, N};
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N};
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_union_device(A, B, out, ctx);
  });
}

void BM_CSRUnion_XXLarge(benchmark::State& state) {
  const Coord N = kSizeXXLarge;
  RectConfig a{0, N, 0, N};
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N};
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_union_device(A, B, out, ctx);
  });
}

void BM_CSRDifference_Tiny(benchmark::State& state) {
  RectConfig a{0, 128, 0, 128};
  const Coord N = 128;
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N};
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_difference_device(A, B, out, ctx);
  });
}

void BM_CSRDifference_Medium(benchmark::State& state) {
  const Coord N = 1280;
  RectConfig a{0, N, 0, N};
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N};
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_difference_device(A, B, out, ctx);
  });
}

void BM_CSRDifference_Large(benchmark::State& state) {
  const Coord N = 12800;
  RectConfig a{0, N, 0, N};
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N};
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_difference_device(A, B, out, ctx);
  });
}

void BM_CSRDifference_XLarge(benchmark::State& state) {
  const Coord N = 128000;
  RectConfig a{0, N, 0, N};
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N};
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_difference_device(A, B, out, ctx);
  });
}

void BM_CSRDifference_XXLarge(benchmark::State& state) {
  const Coord N = kSizeXXLarge;
  RectConfig a{0, N, 0, N};
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N};
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_difference_device(A, B, out, ctx);
  });
}

void BM_CSRSymmetricDifference_Tiny(benchmark::State& state) {
  RectConfig a{0, 128, 0, 128};
  const Coord N = 128;
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N};
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_symmetric_difference_device(A, B, out, ctx);
  });
}

void BM_CSRSymmetricDifference_Medium(benchmark::State& state) {
  const Coord N = 1280;
  RectConfig a{0, N, 0, N};
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N};
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_symmetric_difference_device(A, B, out, ctx);
  });
}

void BM_CSRSymmetricDifference_Large(benchmark::State& state) {
  const Coord N = 12800;
  RectConfig a{0, N, 0, N};
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N};
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_symmetric_difference_device(A, B, out, ctx);
  });
}

void BM_CSRSymmetricDifference_XLarge(benchmark::State& state) {
  const Coord N = 128000;
  RectConfig a{0, N, 0, N};
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N};
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_symmetric_difference_device(A, B, out, ctx);
  });
}

void BM_CSRSymmetricDifference_XXLarge(benchmark::State& state) {
  const Coord N = kSizeXXLarge;
  RectConfig a{0, N, 0, N};
  const Coord offset = N / 4;
  RectConfig b{offset, offset + N, offset, offset + N};
  bench_binary_op(state, a, b, [](const IntervalSet2DDevice& A,
                                  const IntervalSet2DDevice& B,
                                  IntervalSet2DDevice& out,
                                  CsrSetAlgebraContext& ctx) {
    set_symmetric_difference_device(A, B, out, ctx);
  });
}

void BM_CSRMakeBox_Tiny(benchmark::State& state) {
  RectConfig cfg{0, 128, 0, 128};
  bench_box_construction(state, cfg);
}

void BM_CSRMakeBox_Medium(benchmark::State& state) {
  RectConfig cfg{0, 1280, 0, 1280};
  bench_box_construction(state, cfg);
}

void BM_CSRMakeBox_Large(benchmark::State& state) {
  RectConfig cfg{0, 12800, 0, 12800};
  bench_box_construction(state, cfg);
}

void BM_CSRMakeBox_XLarge(benchmark::State& state) {
  RectConfig cfg{0, 128000, 0, 128000};
  bench_box_construction(state, cfg);
}

void BM_CSRMakeBox_XXLarge(benchmark::State& state) {
  RectConfig cfg{0, kSizeXXLarge, 0, kSizeXXLarge};
  bench_box_construction(state, cfg);
}

void bench_map_rows(benchmark::State& state, Coord extent) {
  RectConfig a_cfg, b_cfg;
  make_intersection_configs(extent, a_cfg, b_cfg);
  IntervalSet2DDevice A = make_box(a_cfg);
  IntervalSet2DDevice B = make_box(b_cfg);
  const std::size_t rows = overlap_rows(a_cfg, b_cfg);

  CsrSetAlgebraContext ctx;

  run_row_kernel(state, rows, [&]() {
    auto mapping = detail::build_row_intersection_mapping(
        A, B, ctx.workspace);
    benchmark::DoNotOptimize(mapping.row_keys.data());
    benchmark::DoNotOptimize(mapping.row_index_a.data());
    benchmark::DoNotOptimize(mapping.row_index_b.data());
  }, "ns_per_row");
}

void bench_row_count(benchmark::State& state, Coord extent) {
  RectConfig a_cfg, b_cfg;
  make_intersection_configs(extent, a_cfg, b_cfg);
  IntervalSet2DDevice A = make_box(a_cfg);
  IntervalSet2DDevice B = make_box(b_cfg);
  detail::RowMergeResult mapping =
      detail::build_row_intersection_mapping(A, B);
  const std::size_t rows = mapping.num_rows;
  if (rows == 0) {
    state.SkipWithMessage("No overlapping rows");
    return;
  }

  IntervalSet2DDevice::IndexView row_counts(
      "subsetix_csr_intersection_bench_row_counts", rows);
  auto row_index_a = mapping.row_index_a;
  auto row_index_b = mapping.row_index_b;
  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  run_row_kernel(state, rows, [&]() {
    Kokkos::parallel_for(
        "subsetix_csr_intersection_bench_count",
        Kokkos::RangePolicy<ExecSpace>(0, rows),
        KOKKOS_LAMBDA(const std::size_t i) {
          const int ia = row_index_a(i);
          const int ib = row_index_b(i);
          if (ia < 0 || ib < 0) {
            row_counts(i) = 0;
            return;
          }
          const std::size_t row_a = static_cast<std::size_t>(ia);
          const std::size_t row_b = static_cast<std::size_t>(ib);
          const std::size_t begin_a = row_ptr_a(row_a);
          const std::size_t end_a = row_ptr_a(row_a + 1);
          const std::size_t begin_b = row_ptr_b(row_b);
          const std::size_t end_b = row_ptr_b(row_b + 1);
          if (begin_a == end_a || begin_b == end_b) {
            row_counts(i) = 0;
            return;
          }
          row_counts(i) = detail::row_intersection_count(
              intervals_a, begin_a, end_a,
              intervals_b, begin_b, end_b);
        });
  }, "ns_per_row");
}

void bench_scan(benchmark::State& state, Coord extent) {
  RectConfig a_cfg, b_cfg;
  make_intersection_configs(extent, a_cfg, b_cfg);
  IntervalSet2DDevice A = make_box(a_cfg);
  IntervalSet2DDevice B = make_box(b_cfg);
  detail::RowMergeResult mapping =
      detail::build_row_intersection_mapping(A, B);
  const std::size_t rows = mapping.num_rows;
  if (rows == 0) {
    state.SkipWithMessage("No overlapping rows");
    return;
  }

  IntervalSet2DDevice::IndexView row_counts(
      "subsetix_csr_intersection_bench_scan_counts", rows);
  auto row_index_a = mapping.row_index_a;
  auto row_index_b = mapping.row_index_b;
  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_intersection_bench_prep_count",
      Kokkos::RangePolicy<ExecSpace>(0, rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = row_index_a(i);
        const int ib = row_index_b(i);
        if (ia < 0 || ib < 0) {
          row_counts(i) = 0;
          return;
        }
        const std::size_t row_a = static_cast<std::size_t>(ia);
        const std::size_t row_b = static_cast<std::size_t>(ib);
        const std::size_t begin_a = row_ptr_a(row_a);
        const std::size_t end_a = row_ptr_a(row_a + 1);
        const std::size_t begin_b = row_ptr_b(row_b);
        const std::size_t end_b = row_ptr_b(row_b + 1);
        if (begin_a == end_a || begin_b == end_b) {
          row_counts(i) = 0;
          return;
        }
        row_counts(i) = detail::row_intersection_count(
            intervals_a, begin_a, end_a,
            intervals_b, begin_b, end_b);
      });
  Kokkos::fence();

  IntervalSet2DDevice::IndexView row_ptr_out(
      "subsetix_csr_intersection_bench_scan_row_ptr",
      rows + 1);
  Kokkos::View<std::size_t, DeviceMemorySpace> total_intervals(
      "subsetix_csr_intersection_bench_scan_total");

  run_row_kernel(state, rows, [&]() {
    Kokkos::parallel_scan(
        "subsetix_csr_intersection_bench_scan",
        Kokkos::RangePolicy<ExecSpace>(0, rows),
        KOKKOS_LAMBDA(const std::size_t i,
                      std::size_t& update,
                      const bool final_pass) {
          const std::size_t c = row_counts(i);
          if (final_pass) {
            row_ptr_out(i) = update;
            if (i + 1 == rows) {
              row_ptr_out(rows) = update + c;
              total_intervals() = update + c;
            }
          }
          update += c;
        });
  }, "ns_per_row");
}

void bench_fill(benchmark::State& state, Coord extent) {
  RectConfig a_cfg, b_cfg;
  make_intersection_configs(extent, a_cfg, b_cfg);
  IntervalSet2DDevice A = make_box(a_cfg);
  IntervalSet2DDevice B = make_box(b_cfg);
  detail::RowMergeResult mapping =
      detail::build_row_intersection_mapping(A, B);
  const std::size_t rows = mapping.num_rows;
  if (rows == 0) {
    state.SkipWithMessage("No overlapping rows");
    return;
  }

  IntervalSet2DDevice::IndexView row_counts(
      "subsetix_csr_intersection_bench_fill_counts", rows);
  auto row_index_a = mapping.row_index_a;
  auto row_index_b = mapping.row_index_b;
  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_intersection_bench_fill_prep_count",
      Kokkos::RangePolicy<ExecSpace>(0, rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = row_index_a(i);
        const int ib = row_index_b(i);
        if (ia < 0 || ib < 0) {
          row_counts(i) = 0;
          return;
        }
        const std::size_t row_a = static_cast<std::size_t>(ia);
        const std::size_t row_b = static_cast<std::size_t>(ib);
        const std::size_t begin_a = row_ptr_a(row_a);
        const std::size_t end_a = row_ptr_a(row_a + 1);
        const std::size_t begin_b = row_ptr_b(row_b);
        const std::size_t end_b = row_ptr_b(row_b + 1);
        if (begin_a == end_a || begin_b == end_b) {
          row_counts(i) = 0;
          return;
        }
        row_counts(i) = detail::row_intersection_count(
            intervals_a, begin_a, end_a,
            intervals_b, begin_b, end_b);
      });
  Kokkos::fence();

  IntervalSet2DDevice::IndexView row_ptr_out(
      "subsetix_csr_intersection_bench_fill_row_ptr",
      rows + 1);
  Kokkos::View<std::size_t, DeviceMemorySpace> total_intervals(
      "subsetix_csr_intersection_bench_fill_total");
  Kokkos::parallel_scan(
      "subsetix_csr_intersection_bench_fill_scan",
      Kokkos::RangePolicy<ExecSpace>(0, rows),
      KOKKOS_LAMBDA(const std::size_t i,
                    std::size_t& update,
                    const bool final_pass) {
        const std::size_t c = row_counts(i);
        if (final_pass) {
          row_ptr_out(i) = update;
          if (i + 1 == rows) {
            row_ptr_out(rows) = update + c;
            total_intervals() = update + c;
          }
        }
        update += c;
      });
  Kokkos::fence();

  std::size_t num_intervals_out = 0;
  Kokkos::deep_copy(num_intervals_out, total_intervals);
  if (num_intervals_out == 0) {
    state.SkipWithMessage("No intersection intervals");
    return;
  }

  IntervalSet2DDevice::IntervalView intervals_out(
      "subsetix_csr_intersection_bench_fill_intervals",
      num_intervals_out);

  double total_seconds = 0.0;
  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    Kokkos::parallel_for(
        "subsetix_csr_intersection_bench_fill",
        Kokkos::RangePolicy<ExecSpace>(0, rows),
        KOKKOS_LAMBDA(const std::size_t i) {
          const int ia = row_index_a(i);
          const int ib = row_index_b(i);
          if (ia < 0 || ib < 0) {
            return;
          }
          const std::size_t row_a = static_cast<std::size_t>(ia);
          const std::size_t row_b = static_cast<std::size_t>(ib);
          const std::size_t begin_a = row_ptr_a(row_a);
          const std::size_t end_a = row_ptr_a(row_a + 1);
          const std::size_t begin_b = row_ptr_b(row_b);
          const std::size_t end_b = row_ptr_b(row_b + 1);
          if (begin_a == end_a || begin_b == end_b) {
            return;
          }
          const std::size_t out_offset = row_ptr_out(i);
          detail::row_intersection_fill(
              intervals_a, begin_a, end_a,
              intervals_b, begin_b, end_b,
              intervals_out, out_offset);
        });
    Kokkos::fence();
    const auto t1 = std::chrono::steady_clock::now();
    total_seconds += std::chrono::duration<double>(t1 - t0).count();
  }

  state.counters["ns_per_interval"] =
      (total_seconds /
       (static_cast<double>(num_intervals_out) *
        static_cast<double>(state.iterations()))) *
      1e9;
}

// ============================================================================
// Union row-level benchmarks
// ============================================================================

void bench_union_map_rows(benchmark::State& state, Coord extent) {
  RectConfig a_cfg, b_cfg;
  make_intersection_configs(extent, a_cfg, b_cfg);
  IntervalSet2DDevice A = make_box(a_cfg);
  IntervalSet2DDevice B = make_box(b_cfg);

  const std::size_t rows = A.num_rows + B.num_rows;

  CsrSetAlgebraContext ctx;

  run_row_kernel(state, rows, [&]() {
    auto mapping = detail::build_row_union_mapping(
        A, B, ctx.workspace);
    benchmark::DoNotOptimize(mapping.row_keys.data());
    benchmark::DoNotOptimize(mapping.row_index_a.data());
    benchmark::DoNotOptimize(mapping.row_index_b.data());
  }, "ns_per_row");
}

void bench_union_row_count(benchmark::State& state, Coord extent) {
  RectConfig a_cfg, b_cfg;
  make_intersection_configs(extent, a_cfg, b_cfg);
  IntervalSet2DDevice A = make_box(a_cfg);
  IntervalSet2DDevice B = make_box(b_cfg);
  detail::RowMergeResult mapping =
      detail::build_row_union_mapping(A, B);
  const std::size_t rows = mapping.num_rows;
  if (rows == 0) {
    state.SkipWithMessage("No union rows");
    return;
  }

  IntervalSet2DDevice::IndexView row_counts(
      "subsetix_csr_union_bench_row_counts", rows);
  auto row_index_a = mapping.row_index_a;
  auto row_index_b = mapping.row_index_b;
  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  run_row_kernel(state, rows, [&]() {
    Kokkos::parallel_for(
        "subsetix_csr_union_bench_count",
        Kokkos::RangePolicy<ExecSpace>(0, rows),
        KOKKOS_LAMBDA(const std::size_t i) {
          const int ia = row_index_a(i);
          const int ib = row_index_b(i);

          std::size_t begin_a = 0;
          std::size_t end_a = 0;
          std::size_t begin_b = 0;
          std::size_t end_b = 0;

          if (ia >= 0) {
            const std::size_t row_a = static_cast<std::size_t>(ia);
            begin_a = row_ptr_a(row_a);
            end_a = row_ptr_a(row_a + 1);
          }
          if (ib >= 0) {
            const std::size_t row_b = static_cast<std::size_t>(ib);
            begin_b = row_ptr_b(row_b);
            end_b = row_ptr_b(row_b + 1);
          }

          if (begin_a == end_a && begin_b == end_b) {
            row_counts(i) = 0;
            return;
          }

          row_counts(i) = detail::row_union_count(
              intervals_a, begin_a, end_a,
              intervals_b, begin_b, end_b);
        });
  }, "ns_per_row");
}

void bench_union_scan(benchmark::State& state, Coord extent) {
  RectConfig a_cfg, b_cfg;
  make_intersection_configs(extent, a_cfg, b_cfg);
  IntervalSet2DDevice A = make_box(a_cfg);
  IntervalSet2DDevice B = make_box(b_cfg);
  detail::RowMergeResult mapping =
      detail::build_row_union_mapping(A, B);
  const std::size_t rows = mapping.num_rows;
  if (rows == 0) {
    state.SkipWithMessage("No union rows");
    return;
  }

  IntervalSet2DDevice::IndexView row_counts(
      "subsetix_csr_union_bench_scan_counts", rows);
  auto row_index_a = mapping.row_index_a;
  auto row_index_b = mapping.row_index_b;
  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_union_bench_scan_prep_count",
      Kokkos::RangePolicy<ExecSpace>(0, rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = row_index_a(i);
        const int ib = row_index_b(i);

        std::size_t begin_a = 0;
        std::size_t end_a = 0;
        std::size_t begin_b = 0;
        std::size_t end_b = 0;

        if (ia >= 0) {
          const std::size_t row_a = static_cast<std::size_t>(ia);
          begin_a = row_ptr_a(row_a);
          end_a = row_ptr_a(row_a + 1);
        }
        if (ib >= 0) {
          const std::size_t row_b = static_cast<std::size_t>(ib);
          begin_b = row_ptr_b(row_b);
          end_b = row_ptr_b(row_b + 1);
        }

        if (begin_a == end_a && begin_b == end_b) {
          row_counts(i) = 0;
          return;
        }

        row_counts(i) = detail::row_union_count(
            intervals_a, begin_a, end_a,
            intervals_b, begin_b, end_b);
      });
  Kokkos::fence();

  IntervalSet2DDevice::IndexView row_ptr_out(
      "subsetix_csr_union_bench_scan_row_ptr",
      rows + 1);
  Kokkos::View<std::size_t, DeviceMemorySpace> total_intervals(
      "subsetix_csr_union_bench_scan_total");

  run_row_kernel(state, rows, [&]() {
    Kokkos::parallel_scan(
        "subsetix_csr_union_bench_scan",
        Kokkos::RangePolicy<ExecSpace>(0, rows),
        KOKKOS_LAMBDA(const std::size_t i,
                      std::size_t& update,
                      const bool final_pass) {
          const std::size_t c = row_counts(i);
          if (final_pass) {
            row_ptr_out(i) = update;
            if (i + 1 == rows) {
              row_ptr_out(rows) = update + c;
              total_intervals() = update + c;
            }
          }
          update += c;
        });
  }, "ns_per_row");
}

void bench_union_fill(benchmark::State& state, Coord extent) {
  RectConfig a_cfg, b_cfg;
  make_intersection_configs(extent, a_cfg, b_cfg);
  IntervalSet2DDevice A = make_box(a_cfg);
  IntervalSet2DDevice B = make_box(b_cfg);
  detail::RowMergeResult mapping =
      detail::build_row_union_mapping(A, B);
  const std::size_t rows = mapping.num_rows;
  if (rows == 0) {
    state.SkipWithMessage("No union rows");
    return;
  }

  IntervalSet2DDevice::IndexView row_counts(
      "subsetix_csr_union_bench_fill_counts", rows);
  auto row_index_a = mapping.row_index_a;
  auto row_index_b = mapping.row_index_b;
  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_union_bench_fill_prep_count",
      Kokkos::RangePolicy<ExecSpace>(0, rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = row_index_a(i);
        const int ib = row_index_b(i);

        std::size_t begin_a = 0;
        std::size_t end_a = 0;
        std::size_t begin_b = 0;
        std::size_t end_b = 0;

        if (ia >= 0) {
          const std::size_t row_a = static_cast<std::size_t>(ia);
          begin_a = row_ptr_a(row_a);
          end_a = row_ptr_a(row_a + 1);
        }
        if (ib >= 0) {
          const std::size_t row_b = static_cast<std::size_t>(ib);
          begin_b = row_ptr_b(row_b);
          end_b = row_ptr_b(row_b + 1);
        }

        if (begin_a == end_a && begin_b == end_b) {
          row_counts(i) = 0;
          return;
        }

        row_counts(i) = detail::row_union_count(
            intervals_a, begin_a, end_a,
            intervals_b, begin_b, end_b);
      });
  Kokkos::fence();

  IntervalSet2DDevice::IndexView row_ptr_out(
      "subsetix_csr_union_bench_fill_row_ptr",
      rows + 1);
  Kokkos::View<std::size_t, DeviceMemorySpace> total_intervals(
      "subsetix_csr_union_bench_fill_total");
  Kokkos::parallel_scan(
      "subsetix_csr_union_bench_fill_scan",
      Kokkos::RangePolicy<ExecSpace>(0, rows),
      KOKKOS_LAMBDA(const std::size_t i,
                    std::size_t& update,
                    const bool final_pass) {
        const std::size_t c = row_counts(i);
        if (final_pass) {
          row_ptr_out(i) = update;
          if (i + 1 == rows) {
            row_ptr_out(rows) = update + c;
            total_intervals() = update + c;
          }
        }
        update += c;
      });
  Kokkos::fence();

  std::size_t num_intervals_out = 0;
  Kokkos::deep_copy(num_intervals_out, total_intervals);
  if (num_intervals_out == 0) {
    state.SkipWithMessage("No union intervals");
    return;
  }

  IntervalSet2DDevice::IntervalView intervals_out(
      "subsetix_csr_union_bench_fill_intervals",
      num_intervals_out);

  double total_seconds = 0.0;
  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    Kokkos::parallel_for(
        "subsetix_csr_union_bench_fill",
        Kokkos::RangePolicy<ExecSpace>(0, rows),
        KOKKOS_LAMBDA(const std::size_t i) {
          const int ia = row_index_a(i);
          const int ib = row_index_b(i);

          std::size_t begin_a = 0;
          std::size_t end_a = 0;
          std::size_t begin_b = 0;
          std::size_t end_b = 0;

          if (ia >= 0) {
            const std::size_t row_a = static_cast<std::size_t>(ia);
            begin_a = row_ptr_a(row_a);
            end_a = row_ptr_a(row_a + 1);
          }
          if (ib >= 0) {
            const std::size_t row_b = static_cast<std::size_t>(ib);
            begin_b = row_ptr_b(row_b);
            end_b = row_ptr_b(row_b + 1);
          }

          if (begin_a == end_a && begin_b == end_b) {
            return;
          }

          const std::size_t out_offset = row_ptr_out(i);
          detail::row_union_fill(
              intervals_a, begin_a, end_a,
              intervals_b, begin_b, end_b,
              intervals_out, out_offset);
        });
    Kokkos::fence();
    const auto t1 = std::chrono::steady_clock::now();
    total_seconds += std::chrono::duration<double>(t1 - t0).count();
  }

  state.counters["ns_per_interval"] =
      (total_seconds /
       (static_cast<double>(num_intervals_out) *
        static_cast<double>(state.iterations()))) *
      1e9;
}

// ============================================================================
// Difference row-level benchmarks
// ============================================================================

void bench_difference_map_rows(benchmark::State& state, Coord extent) {
  RectConfig a_cfg, b_cfg;
  make_intersection_configs(extent, a_cfg, b_cfg);
  IntervalSet2DDevice A = make_box(a_cfg);
  IntervalSet2DDevice B = make_box(b_cfg);

  const std::size_t rows = A.num_rows;

  CsrSetAlgebraContext ctx;

  run_row_kernel(state, rows, [&]() {
    auto mapping = detail::build_row_difference_mapping(
        A, B, ctx.workspace);
    benchmark::DoNotOptimize(mapping.row_keys.data());
    benchmark::DoNotOptimize(mapping.row_index_b.data());
  }, "ns_per_row");
}

void bench_difference_row_count(benchmark::State& state, Coord extent) {
  RectConfig a_cfg, b_cfg;
  make_intersection_configs(extent, a_cfg, b_cfg);
  IntervalSet2DDevice A = make_box(a_cfg);
  IntervalSet2DDevice B = make_box(b_cfg);
  detail::RowDifferenceResult mapping =
      detail::build_row_difference_mapping(A, B);
  const std::size_t rows = mapping.num_rows;
  if (rows == 0) {
    state.SkipWithMessage("No rows in A");
    return;
  }

  IntervalSet2DDevice::IndexView row_counts(
      "subsetix_csr_difference_bench_row_counts", rows);
  auto row_index_b = mapping.row_index_b;
  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  run_row_kernel(state, rows, [&]() {
    Kokkos::parallel_for(
        "subsetix_csr_difference_bench_count",
        Kokkos::RangePolicy<ExecSpace>(0, rows),
        KOKKOS_LAMBDA(const std::size_t i) {
          const std::size_t row_a = i;
          const std::size_t begin_a = row_ptr_a(row_a);
          const std::size_t end_a = row_ptr_a(row_a + 1);

          if (begin_a == end_a) {
            row_counts(i) = 0;
            return;
          }

          const int ib = row_index_b(i);
          std::size_t begin_b = 0;
          std::size_t end_b = 0;

          if (ib >= 0) {
            const std::size_t row_b = static_cast<std::size_t>(ib);
            begin_b = row_ptr_b(row_b);
            end_b = row_ptr_b(row_b + 1);
          }

          row_counts(i) = detail::row_difference_count(
              intervals_a, begin_a, end_a,
              intervals_b, begin_b, end_b);
        });
  }, "ns_per_row");
}

void bench_difference_scan(benchmark::State& state, Coord extent) {
  RectConfig a_cfg, b_cfg;
  make_intersection_configs(extent, a_cfg, b_cfg);
  IntervalSet2DDevice A = make_box(a_cfg);
  IntervalSet2DDevice B = make_box(b_cfg);
  detail::RowDifferenceResult mapping =
      detail::build_row_difference_mapping(A, B);
  const std::size_t rows = mapping.num_rows;
  if (rows == 0) {
    state.SkipWithMessage("No rows in A");
    return;
  }

  IntervalSet2DDevice::IndexView row_counts(
      "subsetix_csr_difference_bench_scan_counts", rows);
  auto row_index_b = mapping.row_index_b;
  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_difference_bench_scan_prep_count",
      Kokkos::RangePolicy<ExecSpace>(0, rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const std::size_t row_a = i;
        const std::size_t begin_a = row_ptr_a(row_a);
        const std::size_t end_a = row_ptr_a(row_a + 1);

        if (begin_a == end_a) {
          row_counts(i) = 0;
          return;
        }

        const int ib = row_index_b(i);
        std::size_t begin_b = 0;
        std::size_t end_b = 0;

        if (ib >= 0) {
          const std::size_t row_b = static_cast<std::size_t>(ib);
          begin_b = row_ptr_b(row_b);
          end_b = row_ptr_b(row_b + 1);
        }

        row_counts(i) = detail::row_difference_count(
            intervals_a, begin_a, end_a,
            intervals_b, begin_b, end_b);
      });
  Kokkos::fence();

  IntervalSet2DDevice::IndexView row_ptr_out(
      "subsetix_csr_difference_bench_scan_row_ptr",
      rows + 1);
  Kokkos::View<std::size_t, DeviceMemorySpace> total_intervals(
      "subsetix_csr_difference_bench_scan_total");

  run_row_kernel(state, rows, [&]() {
    Kokkos::parallel_scan(
        "subsetix_csr_difference_bench_scan",
        Kokkos::RangePolicy<ExecSpace>(0, rows),
        KOKKOS_LAMBDA(const std::size_t i,
                      std::size_t& update,
                      const bool final_pass) {
          const std::size_t c = row_counts(i);
          if (final_pass) {
            row_ptr_out(i) = update;
            if (i + 1 == rows) {
              row_ptr_out(rows) = update + c;
              total_intervals() = update + c;
            }
          }
          update += c;
        });
  }, "ns_per_row");
}

void bench_difference_fill(benchmark::State& state, Coord extent) {
  RectConfig a_cfg, b_cfg;
  make_intersection_configs(extent, a_cfg, b_cfg);
  IntervalSet2DDevice A = make_box(a_cfg);
  IntervalSet2DDevice B = make_box(b_cfg);
  detail::RowDifferenceResult mapping =
      detail::build_row_difference_mapping(A, B);
  const std::size_t rows = mapping.num_rows;
  if (rows == 0) {
    state.SkipWithMessage("No rows in A");
    return;
  }

  IntervalSet2DDevice::IndexView row_counts(
      "subsetix_csr_difference_bench_fill_counts", rows);
  auto row_index_b = mapping.row_index_b;
  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_difference_bench_fill_prep_count",
      Kokkos::RangePolicy<ExecSpace>(0, rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const std::size_t row_a = i;
        const std::size_t begin_a = row_ptr_a(row_a);
        const std::size_t end_a = row_ptr_a(row_a + 1);

        if (begin_a == end_a) {
          row_counts(i) = 0;
          return;
        }

        const int ib = row_index_b(i);
        std::size_t begin_b = 0;
        std::size_t end_b = 0;

        if (ib >= 0) {
          const std::size_t row_b = static_cast<std::size_t>(ib);
          begin_b = row_ptr_b(row_b);
          end_b = row_ptr_b(row_b + 1);
        }

        row_counts(i) = detail::row_difference_count(
            intervals_a, begin_a, end_a,
            intervals_b, begin_b, end_b);
      });
  Kokkos::fence();

  IntervalSet2DDevice::IndexView row_ptr_out(
      "subsetix_csr_difference_bench_fill_row_ptr",
      rows + 1);
  Kokkos::View<std::size_t, DeviceMemorySpace> total_intervals(
      "subsetix_csr_difference_bench_fill_total");
  Kokkos::parallel_scan(
      "subsetix_csr_difference_bench_fill_scan",
      Kokkos::RangePolicy<ExecSpace>(0, rows),
      KOKKOS_LAMBDA(const std::size_t i,
                    std::size_t& update,
                    const bool final_pass) {
        const std::size_t c = row_counts(i);
        if (final_pass) {
          row_ptr_out(i) = update;
          if (i + 1 == rows) {
            row_ptr_out(rows) = update + c;
            total_intervals() = update + c;
          }
        }
        update += c;
      });
  Kokkos::fence();

  std::size_t num_intervals_out = 0;
  Kokkos::deep_copy(num_intervals_out, total_intervals);
  if (num_intervals_out == 0) {
    state.SkipWithMessage("No difference intervals");
    return;
  }

  IntervalSet2DDevice::IntervalView intervals_out(
      "subsetix_csr_difference_bench_fill_intervals",
      num_intervals_out);

  double total_seconds = 0.0;
  for (auto _ : state) {
    const auto t0 = std::chrono::steady_clock::now();
    Kokkos::parallel_for(
        "subsetix_csr_difference_bench_fill",
        Kokkos::RangePolicy<ExecSpace>(0, rows),
        KOKKOS_LAMBDA(const std::size_t i) {
          const std::size_t row_a = i;
          const std::size_t begin_a = row_ptr_a(row_a);
          const std::size_t end_a = row_ptr_a(row_a + 1);

          if (begin_a == end_a) {
            return;
          }

          const int ib = row_index_b(i);
          std::size_t begin_b = 0;
          std::size_t end_b = 0;

          if (ib >= 0) {
            const std::size_t row_b = static_cast<std::size_t>(ib);
            begin_b = row_ptr_b(row_b);
            end_b = row_ptr_b(row_b + 1);
          }

          const std::size_t out_offset = row_ptr_out(i);
          detail::row_difference_fill(
              intervals_a, begin_a, end_a,
              intervals_b, begin_b, end_b,
              intervals_out, out_offset);
        });
    Kokkos::fence();
    const auto t1 = std::chrono::steady_clock::now();
    total_seconds += std::chrono::duration<double>(t1 - t0).count();
  }

  state.counters["ns_per_interval"] =
      (total_seconds /
       (static_cast<double>(num_intervals_out) *
        static_cast<double>(state.iterations()))) *
      1e9;
}

void BM_CSRIntersection_MapRows_Tiny(benchmark::State& state) {
  bench_map_rows(state, kSizeTiny);
}

void BM_CSRIntersection_MapRows_Medium(benchmark::State& state) {
  bench_map_rows(state, kSizeMedium);
}

void BM_CSRIntersection_MapRows_Large(benchmark::State& state) {
  bench_map_rows(state, kSizeLarge);
}

void BM_CSRIntersection_MapRows_XLarge(benchmark::State& state) {
  bench_map_rows(state, kSizeXLarge);
}

void BM_CSRIntersection_MapRows_XXLarge(benchmark::State& state) {
  bench_map_rows(state, kSizeXXLarge);
}

void BM_CSRIntersection_RowCount_Tiny(benchmark::State& state) {
  bench_row_count(state, kSizeTiny);
}

void BM_CSRIntersection_RowCount_Medium(benchmark::State& state) {
  bench_row_count(state, kSizeMedium);
}

void BM_CSRIntersection_RowCount_Large(benchmark::State& state) {
  bench_row_count(state, kSizeLarge);
}

void BM_CSRIntersection_RowCount_XLarge(benchmark::State& state) {
  bench_row_count(state, kSizeXLarge);
}

void BM_CSRIntersection_RowCount_XXLarge(benchmark::State& state) {
  bench_row_count(state, kSizeXXLarge);
}

void BM_CSRIntersection_Scan_Tiny(benchmark::State& state) {
  bench_scan(state, kSizeTiny);
}

void BM_CSRIntersection_Scan_Medium(benchmark::State& state) {
  bench_scan(state, kSizeMedium);
}

void BM_CSRIntersection_Scan_Large(benchmark::State& state) {
  bench_scan(state, kSizeLarge);
}

void BM_CSRIntersection_Scan_XLarge(benchmark::State& state) {
  bench_scan(state, kSizeXLarge);
}

void BM_CSRIntersection_Scan_XXLarge(benchmark::State& state) {
  bench_scan(state, kSizeXXLarge);
}

void BM_CSRIntersection_Fill_Tiny(benchmark::State& state) {
  bench_fill(state, kSizeTiny);
}

void BM_CSRIntersection_Fill_Medium(benchmark::State& state) {
  bench_fill(state, kSizeMedium);
}

void BM_CSRIntersection_Fill_Large(benchmark::State& state) {
  bench_fill(state, kSizeLarge);
}

void BM_CSRIntersection_Fill_XLarge(benchmark::State& state) {
  bench_fill(state, kSizeXLarge);
}

void BM_CSRIntersection_Fill_XXLarge(benchmark::State& state) {
  bench_fill(state, kSizeXXLarge);
}

void BM_CSRUnion_MapRows_Tiny(benchmark::State& state) {
  bench_union_map_rows(state, kSizeTiny);
}

void BM_CSRUnion_MapRows_Medium(benchmark::State& state) {
  bench_union_map_rows(state, kSizeMedium);
}

void BM_CSRUnion_MapRows_Large(benchmark::State& state) {
  bench_union_map_rows(state, kSizeLarge);
}

void BM_CSRUnion_MapRows_XLarge(benchmark::State& state) {
  bench_union_map_rows(state, kSizeXLarge);
}

void BM_CSRUnion_MapRows_XXLarge(benchmark::State& state) {
  bench_union_map_rows(state, kSizeXXLarge);
}

void BM_CSRUnion_RowCount_Tiny(benchmark::State& state) {
  bench_union_row_count(state, kSizeTiny);
}

void BM_CSRUnion_RowCount_Medium(benchmark::State& state) {
  bench_union_row_count(state, kSizeMedium);
}

void BM_CSRUnion_RowCount_Large(benchmark::State& state) {
  bench_union_row_count(state, kSizeLarge);
}

void BM_CSRUnion_RowCount_XLarge(benchmark::State& state) {
  bench_union_row_count(state, kSizeXLarge);
}

void BM_CSRUnion_RowCount_XXLarge(benchmark::State& state) {
  bench_union_row_count(state, kSizeXXLarge);
}

void BM_CSRUnion_Scan_Tiny(benchmark::State& state) {
  bench_union_scan(state, kSizeTiny);
}

void BM_CSRUnion_Scan_Medium(benchmark::State& state) {
  bench_union_scan(state, kSizeMedium);
}

void BM_CSRUnion_Scan_Large(benchmark::State& state) {
  bench_union_scan(state, kSizeLarge);
}

void BM_CSRUnion_Scan_XLarge(benchmark::State& state) {
  bench_union_scan(state, kSizeXLarge);
}

void BM_CSRUnion_Scan_XXLarge(benchmark::State& state) {
  bench_union_scan(state, kSizeXXLarge);
}

void BM_CSRUnion_Fill_Tiny(benchmark::State& state) {
  bench_union_fill(state, kSizeTiny);
}

void BM_CSRUnion_Fill_Medium(benchmark::State& state) {
  bench_union_fill(state, kSizeMedium);
}

void BM_CSRUnion_Fill_Large(benchmark::State& state) {
  bench_union_fill(state, kSizeLarge);
}

void BM_CSRUnion_Fill_XLarge(benchmark::State& state) {
  bench_union_fill(state, kSizeXLarge);
}

void BM_CSRUnion_Fill_XXLarge(benchmark::State& state) {
  bench_union_fill(state, kSizeXXLarge);
}

void BM_CSRDifference_MapRows_Tiny(benchmark::State& state) {
  bench_difference_map_rows(state, kSizeTiny);
}

void BM_CSRDifference_MapRows_Medium(benchmark::State& state) {
  bench_difference_map_rows(state, kSizeMedium);
}

void BM_CSRDifference_MapRows_Large(benchmark::State& state) {
  bench_difference_map_rows(state, kSizeLarge);
}

void BM_CSRDifference_MapRows_XLarge(benchmark::State& state) {
  bench_difference_map_rows(state, kSizeXLarge);
}

void BM_CSRDifference_MapRows_XXLarge(benchmark::State& state) {
  bench_difference_map_rows(state, kSizeXXLarge);
}

void BM_CSRDifference_RowCount_Tiny(benchmark::State& state) {
  bench_difference_row_count(state, kSizeTiny);
}

void BM_CSRDifference_RowCount_Medium(benchmark::State& state) {
  bench_difference_row_count(state, kSizeMedium);
}

void BM_CSRDifference_RowCount_Large(benchmark::State& state) {
  bench_difference_row_count(state, kSizeLarge);
}

void BM_CSRDifference_RowCount_XLarge(benchmark::State& state) {
  bench_difference_row_count(state, kSizeXLarge);
}

void BM_CSRDifference_RowCount_XXLarge(benchmark::State& state) {
  bench_difference_row_count(state, kSizeXXLarge);
}

void BM_CSRDifference_Scan_Tiny(benchmark::State& state) {
  bench_difference_scan(state, kSizeTiny);
}

void BM_CSRDifference_Scan_Medium(benchmark::State& state) {
  bench_difference_scan(state, kSizeMedium);
}

void BM_CSRDifference_Scan_Large(benchmark::State& state) {
  bench_difference_scan(state, kSizeLarge);
}

void BM_CSRDifference_Scan_XLarge(benchmark::State& state) {
  bench_difference_scan(state, kSizeXLarge);
}

void BM_CSRDifference_Scan_XXLarge(benchmark::State& state) {
  bench_difference_scan(state, kSizeXXLarge);
}

void BM_CSRDifference_Fill_Tiny(benchmark::State& state) {
  bench_difference_fill(state, kSizeTiny);
}

void BM_CSRDifference_Fill_Medium(benchmark::State& state) {
  bench_difference_fill(state, kSizeMedium);
}

void BM_CSRDifference_Fill_Large(benchmark::State& state) {
  bench_difference_fill(state, kSizeLarge);
}

void BM_CSRDifference_Fill_XLarge(benchmark::State& state) {
  bench_difference_fill(state, kSizeXLarge);
}

void BM_CSRDifference_Fill_XXLarge(benchmark::State& state) {
  bench_difference_fill(state, kSizeXXLarge);
}

} // namespace

BENCHMARK(BM_CSRIntersection_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_XLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_XXLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_XLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_XXLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_XLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_XXLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRSymmetricDifference_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRSymmetricDifference_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRSymmetricDifference_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRSymmetricDifference_XLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRSymmetricDifference_XXLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRMakeBox_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRMakeBox_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRMakeBox_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRMakeBox_XLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRMakeBox_XXLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_MapRows_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_MapRows_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_MapRows_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_MapRows_XLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_MapRows_XXLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_RowCount_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_RowCount_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_RowCount_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_RowCount_XLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_RowCount_XXLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_Scan_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_Scan_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_Scan_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_Scan_XLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_Scan_XXLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_Fill_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_Fill_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_Fill_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_Fill_XLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRIntersection_Fill_XXLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_MapRows_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_MapRows_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_MapRows_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_MapRows_XLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_MapRows_XXLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_RowCount_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_RowCount_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_RowCount_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_RowCount_XLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_RowCount_XXLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_Scan_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_Scan_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_Scan_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_Scan_XLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_Scan_XXLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_Fill_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_Fill_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_Fill_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_Fill_XLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRUnion_Fill_XXLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_MapRows_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_MapRows_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_MapRows_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_MapRows_XLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_MapRows_XXLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_RowCount_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_RowCount_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_RowCount_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_RowCount_XLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_RowCount_XXLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_Scan_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_Scan_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_Scan_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_Scan_XLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_Scan_XXLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_Fill_Tiny)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_Fill_Medium)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_Fill_Large)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_Fill_XLarge)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_CSRDifference_Fill_XXLarge)->Unit(benchmark::kNanosecond);

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  ::benchmark::RunSpecifiedBenchmarks();
  Kokkos::finalize();
  return 0;
}
