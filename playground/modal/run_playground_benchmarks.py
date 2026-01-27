# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique
"""
Modal benchmarks for subsetix_kokkos playground intersection module on NVIDIA GPU.

This script runs comprehensive performance benchmarks with 3s minimum time per benchmark
for reliable measurements.

Usage:
    # Run on specific GPU
    modal run playground/modal/run_playground_benchmarks.py::run_t4_entry
    modal run playground/modal/run_playground_benchmarks.py::run_a100_entry
    modal run playground/modal/run_playground_benchmarks.py::run_h100_entry

    # Run all GPUs sequentially with comprehensive report
    modal run playground/modal/run_playground_benchmarks.py::main
"""

from pathlib import Path
import shutil
import subprocess
import json
import re
from datetime import datetime

import modal

# -----------------------------------------------------------------------------
# Modal Configuration
# -----------------------------------------------------------------------------

# GPU architecture mapping for Kokkos
GPU_ARCH_MAP = {
    "T4": "TURING75",
    "A100": "AMPERE80",
    "H100": "HOPPER90",
    "L40S": "ADALATEST",  # Ada Lovelace
}

# Benchmark configuration
BENCHMARK_MIN_TIME = "3.0"  # 3 seconds per benchmark for reliable measurements

def create_image() -> modal.Image:
    """Create the Modal image with the project code included."""
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent

    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install(
            "cmake",
            "ninja-build",
            "gcc-12",
            "g++-12",
            "git",
            "libfmt-dev",
            "libmpfr-dev",
            "libbenchmark-dev",
        )
        .env({
            "CC": "gcc-12",
            "CXX": "g++-12",
        })
        .workdir("/workspace")
        .add_local_dir(project_root, remote_path="/workspace")
    )

IMAGE = create_image()

app = modal.App("subsetix-playground-benchmarks", image=IMAGE)


# -----------------------------------------------------------------------------
# Benchmark parsing utilities
# -----------------------------------------------------------------------------

def parse_benchmark_output(output: str, gpu_type: str) -> dict:
    """Parse Google Benchmark output into structured data."""
    benchmarks = []
    current_benchmark = None

    for line in output.split('\n'):
        # Parse benchmark header line
        # Example: BaselineRandomMeshBenchmark2D/Baseline_SmallConfig/mean
        match = re.match(r'^([^/]+(?:/[^/]+)*)\s+(mean|median|stddev)\s+(\d+)\s+ns\s+(\d+)\s+items/s', line)
        if match:
            name, stat, time_ns, items_per_sec = match.groups()
            if stat == "mean":
                current_benchmark = {
                    "name": name,
                    "time_ns": int(time_ns),
                    "time_us": int(time_ns) / 1000.0,
                    "time_ms": int(time_ns) / 1000000.0,
                    "items_per_second": int(items_per_sec),
                    "items_per_ms": int(items_per_sec) / 1000.0,
                }
                benchmarks.append(current_benchmark)

    return {
        "gpu": gpu_type,
        "benchmarks": benchmarks,
    }


def format_markdown_report(results: list[dict]) -> str:
    """Generate a comprehensive markdown report from benchmark results."""
    report = ["# Playground Intersection Benchmarks - Performance Report\n"]
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"Configuration: `benchmark_min_time={BENCHMARK_MIN_TIME}s`\n")

    # Summary table
    report.append("\n## Summary by GPU\n\n")
    report.append("| GPU | 2D Baseline (μs) | 2D Optimized (μs) | Speedup | 3D Baseline (μs) | 3D Optimized (μs) | Speedup |\n")
    report.append("|-----|------------------|-------------------|---------|------------------|-------------------|---------|\n")

    for result in results:
        gpu = result["gpu"]
        benchmarks = {b["name"]: b for b in result["benchmarks"]}

        # Extract representative benchmarks (MediumConfig)
        baseline_2d = benchmarks.get("BaselineRandomMeshBenchmark2D/Baseline_MediumConfig", {})
        optimized_2d = benchmarks.get("OptimizedRandomMeshBenchmark2D/Optimized_MediumConfig", {})
        baseline_3d = benchmarks.get("BaselineRandomMeshBenchmark3D/Baseline_3D_MediumConfig", {})
        optimized_3d = benchmarks.get("OptimizedRandomMeshBenchmark3D/Optimized_3D_MediumConfig", {})

        speedup_2d = baseline_2d.get("time_us", 0) / optimized_2d.get("time_us", 1) if optimized_2d.get("time_us", 0) > 0 else 0
        speedup_3d = baseline_3d.get("time_us", 0) / optimized_3d.get("time_us", 1) if optimized_3d.get("time_us", 0) > 0 else 0

        report.append(f"| {gpu} | {baseline_2d.get('time_us', 0):.1f} | {optimized_2d.get('time_us', 0):.1f} | {speedup_2d:.2f}x | "
                    f"{baseline_3d.get('time_us', 0):.1f} | {optimized_3d.get('time_us', 0):.1f} | {speedup_3d:.2f}x |\n")

    # Detailed results per GPU
    for result in results:
        gpu = result["gpu"]
        report.append(f"\n## {gpu} GPU - Detailed Results\n\n")

        # Group by dimension (2D vs 3D)
        benchmarks_2d = [b for b in result["benchmarks"] if "3D" not in b["name"]]
        benchmarks_3d = [b for b in result["benchmarks"] if "3D" in b["name"]]

        # 2D Results
        report.append("### 2D Benchmarks\n\n")
        report.append("| Benchmark | Time (μs) | Items/sec | Items/ms |\n")
        report.append("|-----------|-----------|-----------|---------|\n")
        for b in benchmarks_2d:
            report.append(f"| {b['name']} | {b['time_us']:.2f} | {b['items_per_second']:.0f} | {b['items_per_ms']:.2f} |\n")

        # 3D Results
        report.append("\n### 3D Benchmarks\n\n")
        report.append("| Benchmark | Time (μs) | Items/sec | Items/ms |\n")
        report.append("|-----------|-----------|-----------|---------|\n")
        for b in benchmarks_3d:
            report.append(f"| {b['name']} | {b['time_us']:.2f} | {b['items_per_second']:.0f} | {b['items_per_ms']:.2f} |\n")

    # Comparison table: baseline vs optimized across all GPUs
    report.append("\n## Cross-GPU Comparison: Baseline vs Optimized\n\n")
    report.append("### 2D Medium Config Performance\n\n")
    report.append("| GPU | Baseline (μs) | Optimized (μs) | Speedup | Baseline (items/ms) | Optimized (items/ms) | Throughput Gain |\n")
    report.append("|-----|---------------|----------------|---------|---------------------|---------------------|-----------------|\n")

    for result in results:
        gpu = result["gpu"]
        benchmarks = {b["name"]: b for b in result["benchmarks"]}

        baseline = benchmarks.get("BaselineRandomMeshBenchmark2D/Baseline_MediumConfig", {})
        optimized = benchmarks.get("OptimizedRandomMeshBenchmark2D/Optimized_MediumConfig", {})

        if baseline and optimized:
            speedup = baseline.get("time_us", 0) / optimized.get("time_us", 1)
            throughput_gain = optimized.get("items_per_ms", 0) / baseline.get("items_per_ms", 1)
            report.append(f"| {gpu} | {baseline.get('time_us', 0):.1f} | {optimized.get('time_us', 0):.1f} | {speedup:.2f}x | "
                        f"{baseline.get('items_per_ms', 0):.1f} | {optimized.get('items_per_ms', 0):.1f} | {throughput_gain:.2f}x |\n")

    report.append("\n### 3D Medium Config Performance\n\n")
    report.append("| GPU | Baseline (μs) | Optimized (μs) | Speedup | Baseline (items/ms) | Optimized (items/ms) | Throughput Gain |\n")
    report.append("|-----|---------------|----------------|---------|---------------------|---------------------|-----------------|\n")

    for result in results:
        gpu = result["gpu"]
        benchmarks = {b["name"]: b for b in result["benchmarks"]}

        baseline = benchmarks.get("BaselineRandomMeshBenchmark3D/Baseline_3D_MediumConfig", {})
        optimized = benchmarks.get("OptimizedRandomMeshBenchmark3D/Optimized_3D_MediumConfig", {})

        if baseline and optimized:
            speedup = baseline.get("time_us", 0) / optimized.get("time_us", 1)
            throughput_gain = optimized.get("items_per_ms", 0) / baseline.get("items_per_ms", 1)
            report.append(f"| {gpu} | {baseline.get('time_us', 0):.1f} | {optimized.get('time_us', 0):.1f} | {speedup:.2f}x | "
                        f"{baseline.get('items_per_ms', 0):.1f} | {optimized.get('items_per_ms', 0):.1f} | {throughput_gain:.2f}x |\n")

    # Config size scaling analysis
    report.append("\n## Scaling Analysis: Small → Medium → Large → ExtraLarge\n\n")
    report.append("### 2D Optimized - Time Scaling\n\n")

    configs = ["SmallConfig", "MediumConfig", "LargeConfig", "ExtraLargeConfig"]
    report.append("| GPU | Small (μs) | Medium (μs) | Large (μs) | ExtraLarge (μs) |\n")
    report.append("|-----|-----------|-------------|-----------|----------------|\n")

    for result in results:
        gpu = result["gpu"]
        benchmarks = {b["name"]: b for b in result["benchmarks"]}
        row = f"| {gpu} | "
        for cfg in configs:
            name = f"OptimizedRandomMeshBenchmark2D/Optimized_{cfg}"
            b = benchmarks.get(name, {})
            row += f"{b.get('time_us', 0):.1f} | "
        row += "\n"
        report.append(row)

    return ''.join(report)


# -----------------------------------------------------------------------------
# GPU-specific functions
# -----------------------------------------------------------------------------

def run_benchmarks(gpu_type: str, cuda_arch: str) -> dict:
    """Build and run playground benchmarks on specified GPU."""
    repo_root = Path("/workspace")
    build_dir = Path("/tmp/build-playground-cuda")

    # Clean build directory to avoid cache issues
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(exist_ok=True)

    print(f"🎯 GPU: {gpu_type} | CUDA ARCH: {cuda_arch}")

    # Check GPU status
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total,memory.free,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        print(f"🎮 GPU Status:\n{nvidia_smi.stdout}")
    except subprocess.TimeoutExpired:
        print("⚠️  nvidia-smi timeout")
    except Exception as e:
        print(f"⚠️  nvidia-smi failed: {e}")

    # Configure with playground enabled
    cmake_cmd = [
        "cmake", "-S", str(repo_root), "-B", str(build_dir), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DSUBSETIX_ENABLE_PLAYGROUND=ON",
        "-DSUBSETIX_BUILD_STABLE_LIBS=OFF",
        "-DSUBSETIX_BUILD_STABLE_TESTS=OFF",
        "-DSUBSETIX_BUILD_STABLE_BENCHMARKS=OFF",
        "-DSUBSETIX_KOKKOS_CUDA=ON",
        f"-DKokkos_ARCH_{cuda_arch}=ON",
        "-DCMAKE_CXX_COMPILER=g++-12",
    ]

    result = subprocess.run(cmake_cmd, capture_output=True, text=True)

    # Show CMake configuration
    print("\n🔍 CMake Configuration:")
    print(result.stdout)

    if result.returncode != 0:
        return {
            "gpu": gpu_type,
            "error": f"Configure failed:\n{result.stderr}\n{result.stdout}",
            "benchmarks": []
        }

    # Build
    print("🔨 Building...")
    result = subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {
            "gpu": gpu_type,
            "error": f"Build failed:\n{result.stderr}",
            "benchmarks": []
        }
    print("✅ Build successful")

    # Run benchmarks with benchmark_min_time=3s
    benchmark_exes = [
        build_dir / "playground" / "intersection" / "benchmarks" / "playground_intersection_regular_benchmark",
        build_dir / "playground" / "intersection" / "benchmarks" / "playground_intersection_comparison_benchmark",
        build_dir / "playground" / "intersection" / "benchmarks" / "playground_intersection_workspace_benchmark",
        build_dir / "playground" / "intersection" / "benchmarks" / "playground_intersection_phase_benchmark",
    ]

    all_benchmark_output = ""

    for bench_exe in benchmark_exes:
        if not bench_exe.exists():
            print(f"⚠️  Benchmark executable not found: {bench_exe.name}")
            continue

        print(f"\n📊 Running {bench_exe.name} with benchmark_min_time={BENCHMARK_MIN_TIME}s...")
        result = subprocess.run(
            [str(bench_exe), f"--benchmark_min_time={BENCHMARK_MIN_TIME}", "--benchmark_format=console"],
            capture_output=True, text=True, timeout=1800  # 30 min timeout
        )
        all_benchmark_output += f"\n{'='*60}\n{bench_exe.name}\n{'='*60}\n"
        all_benchmark_output += result.stdout
        if result.stderr:
            all_benchmark_output += result.stderr

    # Parse the benchmark output
    parsed = parse_benchmark_output(all_benchmark_output, gpu_type)

    # Also include raw output for reference
    parsed["raw_output"] = all_benchmark_output

    return parsed


@app.function(gpu="T4", cpu=16.0, timeout=2400)
def run_t4() -> dict:
    return run_benchmarks("T4", GPU_ARCH_MAP["T4"])

@app.function(gpu="A100", cpu=16.0, timeout=2400)
def run_a100() -> dict:
    return run_benchmarks("A100", GPU_ARCH_MAP["A100"])

@app.function(gpu="H100", cpu=16.0, timeout=2400)
def run_h100() -> dict:
    return run_benchmarks("H100", GPU_ARCH_MAP["H100"])

@app.function(gpu="L40S", cpu=16.0, timeout=2400)
def run_l40s() -> dict:
    return run_benchmarks("L40S", GPU_ARCH_MAP["L40S"])


# -----------------------------------------------------------------------------
# Entry points
# -----------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Run all GPU benchmarks sequentially and generate comprehensive report."""
    print("🚀 Starting Playground Benchmarks - Sequential GPU Execution")
    print(f"⏱️  Configuration: benchmark_min_time={BENCHMARK_MIN_TIME}s per benchmark")
    print("="*60)

    results = []

    # Run sequentially on all GPUs
    for gpu_name, gpu_arch in GPU_ARCH_MAP.items():
        print(f"\n{'='*60}")
        print(f"🚀 Running benchmarks on {gpu_name}...")
        print(f"{'='*60}")

        if gpu_name == "T4":
            result = run_t4.remote()
        elif gpu_name == "A100":
            result = run_a100.remote()
        elif gpu_name == "H100":
            result = run_h100.remote()
        elif gpu_name == "L40S":
            result = run_l40s.remote()
        else:
            print(f"⚠️  Unknown GPU: {gpu_name}")
            continue

        results.append(result)

        # Print summary for this GPU
        print(f"\n✅ {gpu_name} completed - {len(result.get('benchmarks', []))} benchmarks")

    # Generate and save comprehensive markdown report
    print("\n" + "="*60)
    print("📊 Generating comprehensive performance report...")
    print("="*60)

    report = format_markdown_report(results)

    # Save report to file
    report_path = Path("playground_benchmarks_report.md")
    report_path.write_text(report)

    print(f"\n✅ Report saved to: {report_path}")
    print("\n" + report)

    # Also save raw JSON for further analysis
    json_path = Path("playground_benchmarks_raw.json")
    json_path.write_text(json.dumps(results, indent=2))

    print(f"\n✅ Raw JSON data saved to: {json_path}")


@app.local_entrypoint()
def run_t4_entry():
    print(f"🚀 Running T4 benchmarks (min_time={BENCHMARK_MIN_TIME}s)...")
    result = run_t4.remote()
    print(result.get("raw_output", "No output"))


@app.local_entrypoint()
def run_a100_entry():
    print(f"🚀 Running A100 benchmarks (min_time={BENCHMARK_MIN_TIME}s)...")
    result = run_a100.remote()
    print(result.get("raw_output", "No output"))


@app.local_entrypoint()
def run_h100_entry():
    print(f"🚀 Running H100 benchmarks (min_time={BENCHMARK_MIN_TIME}s)...")
    result = run_h100.remote()
    print(result.get("raw_output", "No output"))


@app.local_entrypoint()
def run_l40s_entry():
    print(f"🚀 Running L40S benchmarks (min_time={BENCHMARK_MIN_TIME}s)...")
    result = run_l40s.remote()
    print(result.get("raw_output", "No output"))
