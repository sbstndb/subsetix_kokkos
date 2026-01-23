# Subsetix Optim V3 - GPU Optimization Orchestrator

Système d'orchestration d'optimisation GPU pour Kokkos CUDA avec agents parallèles et personas aléatoires.

## Nouveautés V3

1. **Session ID unique** - Chaque run a un identifiant unique pour isoler les résultats
2. **Build from scratch** - Les agents partent toujours d'un build propre (nettoyage automatique)
3. **Worktrees gérés proprement** - Les worktrees sont reset à chaque session
4. **Monitoring intelligent** - Vérification de la progression toutes les 60s (pas de spam)

## Installation

```bash
# Installer les skills Claude Code
./scripts/install_skills.sh
```

## Quick Start

### 1. Lancer l'orchestrateur

```bash
# Lancer 24 agents avec configuration par défaut
/optim-orchestrator 24

# Configuration personnalisée
/optim-orchestrator 40 4 4 1800 ./my_logs
```

Cela crée une session avec un ID unique (ex: `20260123_143000`) et les worktrees sont automatiquement préparés.

### 2. Lancer les benchmarks

```bash
/optim-benchmark 24 20260123_143000 10 2 "3D_Large" json
```

### 3. Anti-triche

```bash
/optim-antitriche 24 20260123_143000
```

### 4. Générer le rapport

```bash
/optim-report 24 20260123_143000
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Session ID: 20260123_143000                   │
│  ./optim_logs/session_20260123_143000/                   │
└──────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Orchestrator│  │  Benchmark  │  │Anti-Triche  │
│             │  │  Specialist │  │  Specialist │
│  Creates    │  │            │  │             │
│  worktrees  │  │Sequential  │  │Git diff     │
│  + personas │  │benchmarks  │  │analysis     │
└─────────────┘  └─────────────┘  └─────────────┘
         │                │                │
         └────────────────┼────────────────┘
                          ▼
                 ┌─────────────────┐
                 │   Report Agent   │
                 │                 │
                 │ Markdown Report │
                 └─────────────────┘
```

## Session Directory Structure

```
./optim_logs/
└── session_20260123_143000/
    ├── orchestrator.log              # Log de l'orchestrateur
    ├── personas.json                 # Personas générés
    ├── results.json                  # Résultats finaux
    ├── benchmark_results.json         # Résultats benchmarks
    ├── antitriche_report.json        # Rapport anti-triche
    ├── optimization_report.md        # Rapport final
    ├── agent_01.log                  # Log agent 01
    ├── agent_01_result.json          # Résultat agent 01
    ├── agent_02.log
    ├── agent_02_result.json
    └── ...
```

## Skills Disponibles

| Skill | Description | Arguments |
|-------|-------------|------------|
| `optim-orchestrator` | Orchestrateur principal | N CHUNK_SIZE BUILD_JOBS TIMEOUT LOG_DIR |
| `optim-benchmark` | Benchmarks séquentiels GPU | N SESSION_ID REPEAT_COUNT COOLDOWN FILTER OUTPUT |
| `optim-antitriche` | Vérification intégrité | N SESSION_ID STRICT_MODE |
| `optim-report` | Génération rapport markdown | N SESSION_ID OUTPUT_FILE |
| `optim-profile` | Profiling GPU (Nsight) | N TOP_K PRESET |
| `optim-combine` | Combinaison optimisations | N MAX_COMBINATIONS MIN_SPEEDUP |

## Pipeline Complet

```bash
# 1. Orchestrator - Crée session et lance agents
/optim-orchestrator 24 4 4 1800 ./optim_logs
# → Session ID: 20260123_143000

# 2. Benchmark - Mesure performance (séquentiel)
/optim-benchmark 24 20260123_143000 10 2 "3D_Large"

# 3. Anti-triche - Vérifie intégrité
/optim-antitriche 24 20260123_143000

# 4. Report - Génère rapport final
/optim-report 24 20260123_143000
```

## Workflow avec Script Wrapper

```bash
./scripts/optimization_pipeline.sh 24 4 4 1800 ./optim_logs 10 2 "3D_Large" 4
```

## Dépannage

### Skills non reconnus

Si les skills ne sont pas reconnus:
```bash
./scripts/install_skills.sh
```

### Worktrees existants

Le système V3 gère automatiquement les worktrees existants en les supprimant et recréant à partir de HEAD.

### Build échoue

Vérifiez l'architecture GPU:
```bash
nvidia-smi -L
```

Assurez-vous que le flag Kokkos_ARCH correspond:
- RTX 40xx → `ADA89`
- RTX 30xx → `AMPERE86`
- RTX 20xx → `TURING75`

## Licence

Apache-2.0 - voir LICENSE dans le répertoire racine.
