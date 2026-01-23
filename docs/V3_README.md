# Subsetix Optim V3 - GPU Optimization Orchestrator

Système d'orchestration d'optimisation GPU pour Kokkos CUDA avec agents parallèles et personas aléatoires.

## 🚀 ONE COMMAND TO RULE THEM ALL

```bash
# UNE SEULE COMMANDE pour tout faire :
/optim-pipeline 24 4 4 1800 ./optim_logs
```

Cette commande unique exécute automatiquement :
1. ✅ Lance 24 agents d'optimisation avec personas aléatoires
2. ⏳ Monitor la progression avec détection intelligente des agents bloqués
3. 📊 Lance les benchmarks GPU sur les builds réussis
4. 🔍 Vérifie l'intégrité (anti-triche)
5. 📄 Génère le rapport markdown final

**Plus besoin de taper 4 commandes différentes !**

## Nouveautés V3

1. **Pipeline automatique** - Une seule commande pour tout le workflow
2. **Session ID unique** - Chaque run a un identifiant unique pour isoler les résultats
3. **Build from scratch** - Les agents partent toujours d'un build propre (nettoyage automatique)
4. **Worktrees gérés proprement** - Les worktrees sont reset à chaque session
5. **Monitoring intelligent** - Vérification de la progression toutes les 10s, affichage toutes les 60s

## Installation

```bash
# Installer les skills Claude Code
./scripts/install_skills.sh
```

## Quick Start

### Option 1: Pipeline automatique (RECOMMANDÉ)

```bash
# Lancer 24 agents avec tout le workflow automatique
/optim-pipeline 24

# Configuration personnalisée
/optim-pipeline 40 4 4 1800 ./my_logs
```

### Option 2: Workflow manuel (usage avancé)

Si tu veux contrôler chaque étape manuellement :

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

| Skill | Description | Arguments | Usage |
|-------|-------------|------------|-------|
| `optim-pipeline` ⭐ | **Pipeline complet automatique** | N CHUNK_SIZE BUILD_JOBS TIMEOUT LOG_DIR | `/optim-pipeline 24` |
| `optim-orchestrator` | Orchestrateur principal | N CHUNK_SIZE BUILD_JOBS TIMEOUT LOG_DIR | Workflow manuel |
| `optim-benchmark` | Benchmarks séquentiels GPU | N SESSION_ID REPEAT_COUNT COOLDOWN FILTER OUTPUT | Workflow manuel |
| `optim-antitriche` | Vérification intégrité | N SESSION_ID STRICT_MODE | Workflow manuel |
| `optim-report` | Génération rapport markdown | N SESSION_ID OUTPUT_FILE | Workflow manuel |
| `optim-profile` | Profiling GPU (Nsight) | N TOP_K PRESET | Usage avancé |
| `optim-combine` | Combinaison optimisations | N MAX_COMBINATIONS MIN_SPEEDUP | Usage avancé |

⭐ = **RECOMMANDÉ pour la majorité des cas d'usage**

## Pipeline Manuel (usage avancé)

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

**Note:** Utilise `/optim-pipeline` à la place si tu veux que tout se fasse automatiquement.

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
