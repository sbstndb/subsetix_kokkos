#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

set -e

# =============================================================================
# GPU Optimization Wrapper Script
# Runs the complete optimization pipeline: orchestrator → benchmark → antitriche → combine
# =============================================================================

usage() {
    echo "Usage: $0 [N] [CHUNK_SIZE] [BUILD_JOBS] [TIMEOUT] [LOG_DIR] [REPEAT_COUNT] [COOLDOWN] [FILTER] [COMBINATIONS]"
    echo ""
    echo "Parameters:"
    echo "  N              Number of optimization agents (default: 24)"
    echo "  CHUNK_SIZE     Agents per chunk (default: 4)"
    echo "  BUILD_JOBS     Parallel build jobs (default: 4)"
    echo "  TIMEOUT        Per-agent timeout in seconds (default: 300)"
    echo "  LOG_DIR        Directory for logs (default: ./optim_logs)"
    echo "  REPEAT_COUNT    Benchmark repetitions (default: 10)"
    echo "  COOLDOWN       Seconds between benchmarks (default: 2)"
    echo "  FILTER         Benchmark filter (default: \"3D_Large\")"
    echo "  COMBINATIONS   Number of combinations to propose (default: 4)"
    echo ""
    echo "Example:"
    echo "  $0 40 4 8 600 ./logs 15 3 \"3D_Large,2D_Large\" 6"
    echo "  → 40 agents, chunks of 4, -j8, 10min timeout, custom logs,"
    echo "     15 reps, 3s cooldown, 3D+2D benchmarks, 6 combinations"
    echo ""
    echo "This script requires Claude Code with the optimization skills installed."
    exit 1
}

# =============================================================================
# Parse Arguments
# =============================================================================

N_AGENTS=${1:-24}
CHUNK_SIZE=${2:-4}
BUILD_JOBS=${3:-4}
TIMEOUT=${4:-300}
LOG_DIR=${5:-"./optim_logs"}
REPEAT_COUNT=${6:-10}
COOLDOWN=${7:-2}
FILTER=${8:-"3D_Large"}
COMBINATIONS=${9:-4}

# =============================================================================
# Setup
# =============================================================================

echo "=========================================="
echo "  GPU Optimization Pipeline"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Agents:           $N_AGENTS"
echo "  Chunk size:       $CHUNK_SIZE"
echo "  Build jobs:       $BUILD_JOBS"
echo "  Timeout:          ${TIMEOUT}s per agent"
echo "  Log directory:    $LOG_DIR"
echo "  Benchmark reps:   $REPEAT_COUNT"
echo "  Cooldown:         ${COOLDOWN}s"
echo "  Benchmark filter: $FILTER"
echo "  Combinations:     $COMBINATIONS"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Detect GPU
GPU_INFO=$(nvidia-smi -L 2>/dev/null | head -1)
GPU_ARCH=$(echo "$GPU_INFO" | grep -oP 'NVIDIA \K[^ ]+' | tr '[:lower:]' '[:upper:]')
GPU_NAME=$(echo "$GPU_INFO" | sed 's/GPU 0: //;s/(UUID.*//;s/ *$//')

echo "Detected GPU: $GPU_NAME ($GPU_ARCH)"
echo ""

# Check if in subsetix_kokkos directory
if [ ! -f "CMakeLists.txt" ] || [ ! -d "experimental/include/experimental/subsetix" ]; then
    echo "Error: Must be run from subsetix_kokkos root directory"
    exit 1
fi

# =============================================================================
# Phase 1: Orchestrator
# =============================================================================

echo "=========================================="
echo "  Phase 1: Orchestrator"
echo "=========================================="
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ORCHESTRATOR_LOG="$LOG_DIR/orchestrator_${TIMESTAMP}.log"

echo "Launching $N_AGENTS optimization agents..."
echo "Log: $ORCHESTRATOR_LOG"
echo ""

# The orchestrator is invoked as a Claude Code skill
# It will create worktrees and launch agents in chunks
# Results will be saved to $LOG_DIR/results_${TIMESTAMP}.json

echo "Note: The orchestrator runs as a Claude Code skill."
echo "Make sure you have the optim-orchestrator skill installed in .claude/skills/"
echo ""
echo "After the orchestrator completes, the following phases will run automatically:"
echo "  - Phase 2: Benchmark Specialist"
echo "  - Phase 3: Anti-Triche Specialist"
echo "  - Phase 4: Combination Specialist"
echo ""

# Check if we should auto-run subsequent phases
# For now, this script is a wrapper that invokes Claude Code skills
# The actual execution happens within Claude Code

# =============================================================================
# Instructions
# =============================================================================

echo "=========================================="
echo "  How to Run"
echo "=========================================="
echo ""
echo "This script is a wrapper that coordinates the optimization pipeline."
echo "To run the full pipeline, use the following commands in Claude Code:"
echo ""
echo "1. Phase 1 - Run optimization agents:"
echo "   /optim-orchestrator $N_AGENTS $CHUNK_SIZE $BUILD_JOBS $TIMEOUT $LOG_DIR"
echo ""
echo "2. Phase 2 - Run benchmarks:"
echo "   /optim-benchmark $N_AGENTS $REPEAT_COUNT $COOLDOWN \"$FILTER\""
echo ""
echo "3. Phase 3 - Run anti-triche:"
echo "   /optim-antitriche $N_AGENTS"
echo ""
echo "4. Phase 4 - Run combination:"
echo "   /optim-combine $N_AGENTS $COMBINATIONS"
echo ""
echo "Or use this script which will invoke all phases automatically:"
echo "   ./scripts/optimization_pipeline.sh $N_AGENTS $CHUNK_SIZE $BUILD_JOBS $TIMEOUT $LOG_DIR $REPEAT_COUNT $COOLDOWN \"$FILTER\" $COMBINATIONS"
echo ""

# =============================================================================
# Placeholder for automatic execution
# =============================================================================

# If we want to execute everything automatically from this script,
# we would need to call Claude Code CLI here.
# For now, this script provides instructions and parameter setup.

echo "=========================================="
echo "  Ready to Launch"
echo "=========================================="
echo ""
echo "Next step: Copy and paste the orchestrator command into Claude Code:"
echo ""
echo "  /optim-orchestrator $N_AGENTS $CHUNK_SIZE $BUILD_JOBS $TIMEOUT $LOG_DIR"
echo ""
echo "Or run all phases automatically (future enhancement):"
echo "  ./scripts/optimization_pipeline.sh $N_AGENTS $CHUNK_SIZE $BUILD_JOBS $TIMEOUT $LOG_DIR $REPEAT_COUNT $COOLDOWN \"$FILTER\" $COMBINATIONS"
