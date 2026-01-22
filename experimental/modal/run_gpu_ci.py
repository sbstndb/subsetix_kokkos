"""
Modal CI for subsetix_kokkos experimental module on NVIDIA GPU.

Usage:
    modal run experimental/modal/run_gpu_ci.py::run_t4_entry
    modal run experimental/modal/run_gpu_ci.py::run_a100_entry
    modal run experimental/modal/run_gpu_ci.py::run_h100_entry
    modal run experimental/modal/run_gpu_ci.py::run_b200_entry
"""

from pathlib import Path
import shutil
import subprocess

import modal

# -----------------------------------------------------------------------------
# Modal Configuration
# -----------------------------------------------------------------------------

# GPU architecture mapping for Kokkos
# NOTE: This branch uses Kokkos 5.0.1 which supports Compute Capability 10.0 (BLACKWELL/B200)
GPU_ARCH_MAP = {
    "T4": "TURING75",
    "A100": "AMPERE80",
    "H100": "HOPPER90",
    "B200": "BLACKWELL",  # Supported with Kokkos 5.0.1+
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
        "-DSUBSETIX_ENABLE_EXPERIMENTAL=ON",
        "-DSUBSETIX_BUILD_STABLE_LIBS=OFF",
        "-DSUBSETIX_BUILD_STABLE_TESTS=OFF",
        "-DSUBSETIX_BUILD_STABLE_BENCHMARKS=OFF",
        "-DSUBSETIX_KOKKOS_CUDA=ON",      # Correct subsetix project option
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
        bench_output = "⚠️  Benchmark executable not found"

    # Parse test output - handle both full and partial output formats
    if "Test project" in test_output:
        test_section = test_output.split('Test project')[1]
    else:
        test_section = test_output

    return f"""
{'='*60}
🎯 GPU: {gpu_type} | CUDA ARCH: {cuda_arch}
{'='*60}

TESTS:
{test_section}

{'='*60}
ALL BENCHMARK RESULTS:
{'='*60}
{bench_output}
"""


@app.function(gpu="T4", cpu=16.0, timeout=1200)
def run_t4() -> str:
    return run_benchmarks("T4", GPU_ARCH_MAP["T4"])

@app.function(gpu="A100", cpu=16.0, timeout=1200)
def run_a100() -> str:
    return run_benchmarks("A100", GPU_ARCH_MAP["A100"])

@app.function(gpu="H100", cpu=16.0, timeout=1200)
def run_h100() -> str:
    return run_benchmarks("H100", GPU_ARCH_MAP["H100"])

@app.function(gpu="B200", cpu=16.0, timeout=1200)
def run_b200() -> str:
    return run_benchmarks("B200", GPU_ARCH_MAP["B200"])


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
