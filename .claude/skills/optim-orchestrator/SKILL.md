---
name: optim-orchestrator
description: Orchestrate N optimization agents with random personas for Kokkos CUDA GPU optimization. Use when optimizing GPU code, running parallel optimization experiments, or benchmarking different algorithm variants.
argument-hint: [N] [CHUNK_SIZE] [BUILD_JOBS] [TIMEOUT] [LOG_DIR]
disable-model-invocation: true
context: fork
agent: general-purpose
allowed-tools: Bash(git, cmake, ctest), Read, Edit, Task, Write
---

# GPU Optimization Orchestrator

You are orchestrating **N optimization agents** for Kokkos CUDA GPU optimization.

## Parameters

Extract parameters from $ARGUMENTS (space-separated):
- **N** = First argument (default: 24)
- **CHUNK_SIZE** = Second argument (default: 4)
- **BUILD_JOBS** = Third argument (default: 4) - Compile with `-j{BUILD_JOBS}`
- **TIMEOUT** = Fourth argument (default: 300) - Per-agent timeout in seconds
- **LOG_DIR** = Fifth argument (default: `/tmp/optim_logs`) - Directory for agent logs

Example: `/optim-orchestrator 24 4 8 600 ./logs` → 40 agents, chunks of 4, -j8, 10min timeout, custom logs

## Auto-detected

- **GPU_ARCH**: Auto-detected with `nvidia-smi -L`
- **GPU_NAME**: Auto-detected for reporting
- **TIMESTAMP**: For log filenames

## Context

- Repository: Current directory (must be subsetix_kokkos)
- Target file: `experimental/include/experimental/subsetix/csr/set_algebra/v2.hpp`
- Baseline: `v1.hpp` (NEVER modify)
- Benchmark target: 3D Large (~5M rows)

## Phase 0: Setup & Logging

```bash
# Get parameters
PARAMS=($ARGUMENTS)
N_AGENTS=${PARAMS[0]:-24}
CHUNK_SIZE=${PARAMS[1]:-4}
BUILD_JOBS=${PARAMS[2]:-4}
TIMEOUT=${PARAMS[3]:-300}
LOG_DIR=${PARAMS[4]:-"/tmp/optim_logs"}

# Create log directory
mkdir -p "$LOG_DIR"

# Detect GPU
GPU_INFO=$(nvidia-smi -L 2>/dev/null | head -1)
GPU_ARCH=$(echo "$GPU_INFO" | grep -oP 'NVIDIA \K[^ ]+' | tr '[:lower:]' '[:upper:]')
GPU_NAME=$(echo "$GPU_INFO" | sed 's/GPU 0: //;s/(UUID.*//;s/ *$//')

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Log file
LOG_FILE="$LOG_DIR/orchestrator_${TIMESTAMP}.log"

echo "=== GPU Optimization Orchestrator ===" | tee "$LOG_FILE"
echo "GPU: $GPU_NAME ($GPU_ARCH)" | tee -a "$LOG_FILE"
echo "Agents: $N_AGENTS" | tee -a "$LOG_FILE"
echo "Chunk size: $CHUNK_SIZE" | tee -a "$LOG_FILE"
echo "Build jobs: $BUILD_JOBS" | tee -a "$LOG_FILE"
echo "Timeout: ${TIMEOUT}s per agent" | tee -a "$LOG_FILE"
echo "Log dir: $LOG_DIR" | tee -a "$LOG_FILE"
echo "===================================" | tee -a "$LOG_FILE"
```

## Phase 1: Create Worktrees

```bash
cd /home/sbstndbs/subsetix_kokkos

for i in $(seq -f "%02g" 1 $N_AGENTS); do
  if ! git worktree list | grep -q "v2_opt${i}"; then
    echo "Creating worktree v2_opt${i}..." | tee -a "$LOG_FILE"
    git worktree add ../subsetix_kokkos_v2_opt${i} -b feature/v2-opt${i} >> "$LOG_FILE" 2>&1
  fi
done

WORKTREE_COUNT=$(git worktree list | grep v2_opt | wc -l)
echo "Created/verified $WORKTREE_COUNT worktrees" | tee -a "$LOG_FILE"
```

## Phase 2: Generate N Personas

For each agent, generate a random profile with 6 cursors. Generate a creative name based on risk + expertise.

See documentation in `docs/AGENT_PERSONA_SYSTEM.md` for cursor details.

## Phase 3: Launch Agents (Chunks)

Launch N agents in chunks of CHUNK_SIZE using the Task tool.

**CRITICAL COMPILE CONSTRAINT**: Use `-j${BUILD_JOBS}`
```bash
cmake --build --preset experimental-cuda -j${BUILD_JOBS}
```

**CONTINUE_ON_ERROR**: If an agent fails, log the error and continue with next agent/chunk.

**TIMEOUT**: Each agent has TIMEOUT seconds. If exceeded, mark as failed and continue.

## Phase 4: Collect Results

After all chunks complete, collect:
- Successful agents count
- Failed agents count
- Optimization summaries

Save results to `$LOG_DIR/results_${TIMESTAMP}.json`

## Phase 5: Next Steps

Return summary with:
- Parameters used
- Success/failure counts
- Log file location
- Recommended next commands:
  - `/optim-benchmark $N_AGENTS`
  - `/optim-antitriche $N_AGENTS`
  - `/optim-combine` (if enough successful agents)

## Log File Structure

```
$LOG_DIR/
├── orchestrator_20250123_123456.log      # Main orchestrator log
├── results_20250123_123456.json          # Collected results
├── agent_01.log                          # Individual agent logs
├── agent_02.log
└── ...
```

## Return Format

Return JSON:

```json
{
  "orchestrator": "v2",
  "gpu_arch": "$GPU_ARCH",
  "n_agents": $N_AGENTS,
  "chunk_size": $CHUNK_SIZE,
  "build_jobs": $BUILD_JOBS,
  "timeout": $TIMEOUT,
  "log_dir": "$LOG_DIR",
  "log_file": "$LOG_FILE",
  "timestamp": "$TIMESTAMP",
  "successful": 18,
  "failed": 6,
  "continue_on_error": true,
  "results_file": "$LOG_DIR/results_${TIMESTAMP}.json",
  "next_steps": [
    "Run /optim-benchmark $N_AGENTS to get reliable GPU measurements",
    "Run /optim-antitriche $N_AGENTS to verify integrity",
    "Run /optim-combine to merge compatible optimizations"
  ]
}
```

For detailed documentation, see: `docs/ORCHESTRATOR_SPECS_V2.md`
