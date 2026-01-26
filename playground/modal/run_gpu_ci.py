# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique
"""
Modal CI for subsetix_kokkos playground module on NVIDIA GPU.

Usage:
    modal run playground/modal/run_gpu_ci.py::run_t4_entry
    modal run playground/modal/run_gpu_ci.py::run_a100_entry
    modal run playground/modal/run_gpu_ci.py::run_h100_entry
    modal run playground/modal/run_gpu_ci.py::run_l40s_entry
"""

from pathlib import Path
import shutil
import subprocess

import modal

# -----------------------------------------------------------------------------
# Modal Configuration
# -----------------------------------------------------------------------------

# GPU architecture mapping for Kokkos
GPU_ARCH_MAP = {
    "T4": "TURING75",
    "A100": "AMPERE80",
    "H100": "HOPPER90",
    "L40S": "ADALATEST",
}

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
# GPU-specific functions
# -----------------------------------------------------------------------------

def run_benchmarks(gpu_type: str, cuda_arch: str) -> str:
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

    # Configure - Use subsetix project options (not raw Kokkos)
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

    # Show CMake configuration for debugging
    print("\n🔍 CMake Configuration Output:")
    print(result.stdout)

    if result.returncode != 0:
        return f"❌ [{gpu_type}] Configure failed:\n{result.stderr}\n{result.stdout}"

    # Build
    result = subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return f"""❌ [{gpu_type}] Build failed:
STDERR:
{result.stderr}

STDOUT:
{result.stdout[:5000]}  # Truncate stdout to avoid overflow
"""

    # Run benchmarks with benchmark_min_time=3s
    benchmark_exes = [
        build_dir / "playground" / "intersection" / "benchmarks" / "playground_intersection_regular_benchmark",
        build_dir / "playground" / "intersection" / "benchmarks" / "playground_intersection_comparison_benchmark",
    ]

    bench_output = ""
    for bench_exe in benchmark_exes:
        if bench_exe.exists():
            bench_output += f"\n{'='*60}\nRunning {bench_exe.name} (benchmark_min_time=3s)\n{'='*60}\n"
            result = subprocess.run([str(bench_exe), "--benchmark_min_time=3s"], capture_output=True, text=True)
            bench_output += result.stdout + result.stderr
        else:
            bench_output += f"\n⚠️  Benchmark executable not found: {bench_exe.name}\n"

    return f"""
{'='*60}
🎯 GPU: {gpu_type} | CUDA ARCH: {cuda_arch}
{'='*60}

{'='*60}
BENCHMARK RESULTS (benchmark_min_time=3s):
{'='*60}
{bench_output}
"""


@app.function(gpu="T4", cpu=16.0, timeout=2400)
def run_t4() -> str:
    return run_benchmarks("T4", GPU_ARCH_MAP["T4"])

@app.function(gpu="A100", cpu=16.0, timeout=2400)
def run_a100() -> str:
    return run_benchmarks("A100", GPU_ARCH_MAP["A100"])

@app.function(gpu="H100", cpu=16.0, timeout=2400)
def run_h100() -> str:
    return run_benchmarks("H100", GPU_ARCH_MAP["H100"])

@app.function(gpu="L40S", cpu=16.0, timeout=2400)
def run_l40s() -> str:
    return run_benchmarks("L40S", GPU_ARCH_MAP["L40S"])


# -----------------------------------------------------------------------------
# Entry points
# -----------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Run all GPU benchmarks sequentially."""
    print("🚀 Running playground benchmarks on all GPUs (benchmark_min_time=3s)...")

    print("\n" + "="*60)
    print("🚀 Running benchmarks on T4...")
    t4_result = run_t4.remote()
    print(t4_result)

    print("\n" + "="*60)
    print("🚀 Running benchmarks on A100...")
    a100_result = run_a100.remote()
    print(a100_result)

    print("\n" + "="*60)
    print("🚀 Running benchmarks on H100...")
    h100_result = run_h100.remote()
    print(h100_result)

    print("\n" + "="*60)
    print("🚀 Running benchmarks on L40S...")
    l40s_result = run_l40s.remote()
    print(l40s_result)


@app.local_entrypoint()
def run_t4_entry():
    print("🚀 Running T4 playground benchmarks (benchmark_min_time=3s)...")
    print(run_t4.remote())

@app.local_entrypoint()
def run_a100_entry():
    print("🚀 Running A100 playground benchmarks (benchmark_min_time=3s)...")
    print(run_a100.remote())

@app.local_entrypoint()
def run_h100_entry():
    print("🚀 Running H100 playground benchmarks (benchmark_min_time=3s)...")
    print(run_h100.remote())

@app.local_entrypoint()
def run_l40s_entry():
    print("🚀 Running L40S playground benchmarks (benchmark_min_time=3s)...")
    print(run_l40s.remote())
