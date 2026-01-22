"""
Modal CI for subsetix_kokkos experimental module on NVIDIA GPU.

This script builds and runs the experimental tests/benchmarks on Modal GPU.

Usage:
    modal run experimental/modal/run_gpu_ci.py
"""

from pathlib import Path
import subprocess
import sys

import modal

# -----------------------------------------------------------------------------
# Modal Configuration
# -----------------------------------------------------------------------------

def create_image() -> modal.Image:
    """Create the Modal image with the project code included."""
    # Get project root relative to this script
    # The script is at: experimental/modal/run_gpu_ci.py
    # Project root is three levels up
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent

    print(f"📦 Building image with project root: {project_root}")

    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.3.2-devel-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install(
            # Build essentials
            "cmake",
            "ninja-build",
            "gcc-12",
            "g++-12",
            "git",
            # Dependencies
            "libfmt-dev",
            "libmpfr-dev",
        )
        .env({
            "CC": "gcc-12",
            "CXX": "g++-12",
            "CUDA_ARCHITECTURES": "75",  # Turing (T4), good default for Modal
        })
        .workdir("/workspace")
        # add_local_dir must be last (unless copy=True)
        .add_local_dir(project_root, remote_path="/workspace")
    )


# Create image - this will be called when the app is deployed
IMAGE = create_image()

GPU_CONFIG = "any"  # Any available NVIDIA GPU

app = modal.App("subsetix-experimental-gpu-ci", image=IMAGE)


# -----------------------------------------------------------------------------
# Build and run function
# -----------------------------------------------------------------------------

@app.function(gpu=GPU_CONFIG, timeout=1200)
def build_and_run_tests() -> str:
    """Build and run experimental tests on GPU."""

    import os

    # Working directory is /workspace because we set workdir on the image
    repo_root = Path("/workspace")

    print(f"📂 Working in: {repo_root}")
    print(f"📂 Contents: {[f.name for f in repo_root.iterdir()][:10]}")

    # Build in /tmp (writable)
    build_dir = Path("/tmp/build-experimental-cuda")
    build_dir.mkdir(exist_ok=True)

    # Configure
    print("\n🔨 Configuring...")
    cmake_cmd = [
        "cmake",
        "-S", str(repo_root),
        "-B", str(build_dir),
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DKokkos_ENABLE_CUDA=ON",
        "-DKokkos_ARCH_TURING75=ON",
        "-DSUBSETIX_ENABLE_EXPERIMENTAL=ON",
        "-DSUBSETIX_BUILD_STABLE_LIBS=OFF",
        "-DSUBSETIX_BUILD_STABLE_TESTS=OFF",
        "-DSUBSETIX_BUILD_STABLE_BENCHMARKS=OFF",
    ]

    result = subprocess.run(cmake_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"❌ Configure failed:\n{result.stderr}\n{result.stdout}"

    print("✅ Configure OK")

    # Build
    print("\n🔨 Building...")
    result = subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return f"❌ Build failed:\n{result.stderr}\n{result.stdout}"

    print("✅ Build OK")
    build_output = result.stdout

    # Run tests
    print("\n🧪 Running tests...")
    result = subprocess.run(
        ["ctest", "--output-on-failure"],
        cwd=str(build_dir),
        capture_output=True,
        text=True,
    )
    test_output = result.stdout + result.stderr

    # Run benchmarks
    print("\n📊 Running benchmarks...")
    bench_exe = build_dir / "experimental" / "benchmarks" / "experimental_comparison_benchmark"
    bench_output = ""
    if bench_exe.exists():
        result = subprocess.run([str(bench_exe)], capture_output=True, text=True)
        bench_output = result.stdout + result.stderr
    else:
        bench_output = f"⚠️  Benchmark not found at {bench_exe}"

    return f"""
{'='*60}
BUILD OUTPUT
{'='*60}
{build_output[-2000:] if len(build_output) > 2000 else build_output}

{'='*60}
TESTS OUTPUT
{'='*60}
{test_output[-2000:] if len(test_output) > 2000 else test_output}

{'='*60}
BENCHMARKS OUTPUT
{'='*60}
{bench_output}
"""


@app.local_entrypoint()
def main():
    """Entry point for running from local machine."""
    print("🚀 Running subsetix experimental GPU CI on Modal...")
    output = build_and_run_tests.remote()
    print(output)
