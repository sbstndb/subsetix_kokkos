"""
Modal CI for subsetix_kokkos experimental module on NVIDIA GPU.

Usage:
    modal run experimental/modal/run_gpu_ci.py::run_t4
    modal run experimental/modal/run_gpu_ci.py::run_a100
    modal run experimental/modal/run_gpu_ci.py::run_h100
    modal run experimental/modal/run_gpu_ci.py::run_b200
"""

from pathlib import Path
import subprocess
import sys

import modal

# -----------------------------------------------------------------------------
# Modal Configuration
# -----------------------------------------------------------------------------

# GPU architecture mapping for Kokkos
GPU_ARCH_MAP = {
    "T4": "TURING75",
    "A100": "AMPERE80",
    "H100": "HOPPER90",
    "B200": "BLACKWELL",
}

def create_image() -> modal.Image:
    """Create the Modal image with the project code included."""
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent

    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.3.2-devel-ubuntu22.04",
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
        )
        .env({
            "CC": "gcc-12",
            "CXX": "g++-12",
        })
        .workdir("/workspace")
        .add_local_dir(project_root, remote_path="/workspace")
    )

IMAGE = create_image()

app = modal.App("subsetix-experimental-gpu-ci", image=IMAGE)


# -----------------------------------------------------------------------------
# GPU-specific functions
# -----------------------------------------------------------------------------

def run_benchmarks(gpu_type: str, cuda_arch: str) -> str:
    """Build and run experimental tests on specified GPU."""
    repo_root = Path("/workspace")
    build_dir = Path("/tmp/build-experimental-cuda")
    build_dir.mkdir(exist_ok=True)

    print(f"🎯 GPU: {gpu_type} | CUDA ARCH: {cuda_arch}")

    # Configure
    cmake_cmd = [
        "cmake", "-S", str(repo_root), "-B", str(build_dir), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DKokkos_ENABLE_CUDA=ON",
        f"-DKokkos_ARCH_{cuda_arch}=ON",
        "-DSUBSETIX_ENABLE_EXPERIMENTAL=ON",
        "-DSUBSETIX_BUILD_STABLE_LIBS=OFF",
        "-DSUBSETIX_BUILD_STABLE_TESTS=OFF",
        "-DSUBSETIX_BUILD_STABLE_BENCHMARKS=OFF",
    ]

    result = subprocess.run(cmake_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"❌ [{gpu_type}] Configure failed:\n{result.stderr}\n{result.stdout}"

    # Build
    result = subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return f"❌ [{gpu_type}] Build failed:\n{result.stderr}"

    # Run tests
    result = subprocess.run(
        ["ctest", "--output-on-failure"],
        cwd=str(build_dir), capture_output=True, text=True,
    )
    test_output = result.stdout + result.stderr

    # Run benchmarks
    bench_exe = build_dir / "experimental" / "benchmarks" / "experimental_comparison_benchmark"
    bench_output = ""
    if bench_exe.exists():
        result = subprocess.run([str(bench_exe)], capture_output=True, text=True)
        bench_output = result.stdout + result.stderr
    else:
        bench_output = f"⚠️  Benchmark not found"

    # Extract key benchmark results
    lines = bench_output.split('\n')
    v3_large_2d = ""
    v3_large_3d = ""
    for line in lines:
        if "V3RandomMeshBenchmark2D<GetLargeConfig>" in line:
            v3_large_2d = line
        elif "V3RandomMeshBenchmark3D<GetLargeConfig>" in line:
            v3_large_3d = line

    return f"""
{'='*60}
🎯 GPU: {gpu_type} | CUDA ARCH: {cuda_arch}
{'='*60}

TESTS:
{test_output.split('Test project')[1] if 'Test project' in test_output else test_output}

KEY BENCHMARKS (V3 Large Config):
2D: {v3_large_2d}
3D: {v3_large_3d}

FULL BENCHMARK OUTPUT:
{bench_output}
"""


@app.function(gpu="T4", cpu=16.0, timeout=1200)
def run_t4() -> str:
    return run_benchmarks("T4", "TURING75")

@app.function(gpu="A100", cpu=16.0, timeout=1200)
def run_a100() -> str:
    return run_benchmarks("A100", "AMPERE80")

@app.function(gpu="H100", cpu=16.0, timeout=1200)
def run_h100() -> str:
    return run_benchmarks("H100", "HOPPER90")

@app.function(gpu="B200", cpu=16.0, timeout=1200)
def run_b200() -> str:
    return run_benchmarks("B200", "BLACKWELL")


# -----------------------------------------------------------------------------
# Entry points
# -----------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Run all GPU benchmarks."""
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
    print("🚀 Running benchmarks on B200...")
    b200_result = run_b200.remote()
    print(b200_result)


@app.local_entrypoint()
def run_t4_entry():
    print("🚀 Running T4 benchmarks...")
    print(run_t4.remote())

@app.local_entrypoint()
def run_a100_entry():
    print("🚀 Running A100 benchmarks...")
    print(run_a100.remote())

@app.local_entrypoint()
def run_h100_entry():
    print("🚀 Running H100 benchmarks...")
    print(run_h100.remote())

@app.local_entrypoint()
def run_b200_entry():
    print("🚀 Running B200 benchmarks...")
    print(run_b200.remote())
