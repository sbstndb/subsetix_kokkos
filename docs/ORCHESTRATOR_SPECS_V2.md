# ORCHESTRATEUR V2 - Spécifications Techniques

**Date**: 2025-01-23
**Version**: 2.0 (Chunking + Agents Spécialisés)
**Cible**: Optimiser v2.hpp pour GPU CUDA

---

## NOUVEAUTÉS V2

| Feature | V1 | V2 |
|---------|----|----|
| **Nombre d'agents** | Fixe (10) | Paramétrable (N) |
| **Lancement** | Tous en parallèle | Chunks de 4 |
| **Compilation** | Non contrainte | Forcée -j4 |
| **Benchmarks** | Par agent | Agent spécialisé séquentiel |
| **Anti-triche** | Confiance | Agent spécialisé |
| **Validation** | Top 2 | Tous les valides |

---

## ARCHITECTURE

```
Orchestrateur Principal
    │
    ├── Génère N personas uniques
    │
    ├── Lance les agents d'optimisation (chunks de 4)
    │   └── Chaque agent: compile (-j4) + test + retourne JSON
    │
    ├── Agent Benchmark Spécialisé
    │   └── Itère worktrees, bench séquentiels, tableau comparatif
    │
    ├── Agent Anti-Triche
    │   └── Analyse git diff, baseline stable, rapport confiance
    │
    └── Rapport final
        └── Tableau complet + top optimisations + recommandations
```

---

## WORKFLOW DÉTAILLÉ

### Phase 0: Setup (ONE-TIME)

```bash
# Paramètres
N_AGENTS=24  # Ou autre nombre
CHUNK_SIZE=4

# Détecter GPU
nvidia-smi -L | grep -oP 'NVIDIA \K[^ ]+' | tr '[:lower:]' '[:upper:]' > /tmp/gpu_arch.txt
GPU_ARCH=$(cat /tmp/gpu_arch.txt)

# Créer N worktrees
cd /home/sbstndbs/subsetix_kokkos
for i in $(seq -f "%02g" 1 $N_AGENTS); do
  git worktree add ../subsetix_kokkos_v2_opt${i} -b feature/v2-opt${i}
done

git worktree list
```

### Phase 1: Générer N Personas

Pour chaque agent (1 à N), générer un profil unique avec les 6 curseurs.

### Phase 2: Lancer les Agents (Chunks)

```python
import subprocess
import time

for chunk_start in range(1, N_AGENTS + 1, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE - 1, N_AGENTS)

    print(f"=== Chunk {chunk_start}-{chunk_end} ===")

    # Lancer CHUNK_SIZE agents en parallèle
    # (via Task tool Claude Code)

    # Attendre que tous terminent
    time.sleep(10)  # Ou attendre réellement

    print(f"=== Chunk {chunk_start}-{chunk_end} terminé ===")
```

**Chaque agent compile avec -j4 pour limiter la mémoire.**

### Phase 3: Benchmark Spécialisé

Lancer UN agent qui itère sur tous les worktrees:

```bash
for i in $(seq -f "%02g" 1 $N_AGENTS); do
  echo "=== Benchmark v2_opt${i} ==="
  cd /home/sbstndbs/subsetix_kokkos_v2_opt${i}

  # Benchmark séquentiel (un à la fois)
  ./build-experimental-cuda/experimental/benchmarks/experimental_unified_comparison_benchmark \
    --benchmark_filter="3D_Large" \
    --benchmark_repetitions=10 \
    --benchmark_report_aggregates_only=true \
    --benchmark_format=json > /tmp/v2_opt${i}_bench.json

  # Refroidir le GPU
  sleep 2
done
```

### Phase 4: Anti-Triche

Lancer UN agent qui analyse tous les worktrees:

```bash
for i in $(seq -f "%02g" 1 $N_AGENTS); do
  echo "=== Anti-triche v2_opt${i} ==="
  cd /home/sbstndbs/subsetix_kokkos_v2_opt${i}

  # Check v1 non modifié
  git diff experimental/include/experimental/subsetix/csr/set_algebra/v1.hpp | head -20

  # Check fichiers modifiés
  git status --short
done
```

### Phase 5: Rapport Final

Générer un markdown avec:
- Tableau des N agents
- Résultats benchmarks fiables
- Rapport anti-triche
- Top optimisations
- Recommandations

---

## FORMAT DE RÉSULTAT AGENT OPTIMISATION

```json
{
  "agent_id": "01",
  "persona_name": "Le Prudent Kokkosien",
  "risk": "Conservative",
  "expertise": "KokkosSpecialist",
  "opt_type": "QuickWin",
  "style": "Incremental",
  "scope": "Local",
  "innovation": "Proven",
  "optimization_proposed": "Description",
  "status": "success|partial|failed",
  "compile": true|false,
  "tests_pass": true|false,
  "notes": "Notes techniques"
}
```

**Note**: Les agents d'optimisation ne font PAS de benchmark (pour éviter interférence GPU).

---

## FORMAT DE RÉSULTAT AGENT BENCHMARK

```json
{
  "benchmark_agent": "specialized",
  "gpu_arch": "AMPERE86",
  "total_agents": 24,
  "valid_agents": 18,
  "results": [
    {
      "agent_id": "01",
      "v1_mean_ms": 19.5,
      "v2_mean_ms": 22.1,
      "speedup": 0.0,
      "status": "regression"
    },
    {
      "agent_id": "02",
      "v1_mean_ms": 19.5,
      "v2_mean_ms": 15.2,
      "speedup": 1.28,
      "status": "valid"
    }
  ],
  "top_optimizations": [
    {"agent_id": "15", "speedup": 1.45, "optimization": "..."},
    {"agent_id": "02", "speedup": 1.28, "optimization": "..."}
  ]
}
```

---

## FORMAT DE RÉSULTAT AGENT ANTI-TRICHE

```json
{
  "anti_triche_agent": "specialized",
  "total_agents": 24,
  "suspects": ["07"],
  "trusted": [1, 2, 3, 4, 5, 6, 8, ...],
  "report": [
    {
      "agent_id": "01",
      "v1_modified": false,
      "files_modified": ["v2.hpp"],
      "suspicious": false,
      "notes": "Modifications propres dans v2.hpp uniquement"
    },
    {
      "agent_id": "07",
      "v1_modified": false,
      "files_modified": ["v2.hpp", "test_cross_version.cpp"],
      "suspicious": true,
      "notes": "ATTENTION: Fichier de tests modifié!"
    }
  ]
}
```

---

## EXEMPLE DE RAPPORT FINAL

```markdown
# Rapport d'Optimisation v2 vs v1 - N=24 Agents

## Résumé

- **Agents lancés**: 24
- **Agents valides** (compilé + tests passent): 18
- **Agents suspects** (anti-triche): 1
- **Agents avec speedup > 1.05x**: 8

## Tableau Complet

| Agent | Persona | Status | Speedup | Trust |
|-------|---------|--------|---------|-------|
| 01 | Le Prudent Kokkosien | ✅ | 0.00x | ✅ |
| 02 | Le Visionnaire GPU | ✅ | 1.28x | ✅ |
| 07 | Le Fou Algorithmique | ⚠️ | N/A | ❌ Suspect |
| ... | ... | ... | ... | ... |

## Top Optimisations (valides et fiables)

### #1: Agent 15 - 1.45x
- **Persona**: Aggressive + GPUArchitect + GPUHwSpecific + Experimental
- **Optimisation**: Warp-shuffle reduction pour compaction
- **Speedup**: 1.45x
- **Confiance**: ✅ Baseline stable, pas de triche

### #2: Agent 02 - 1.28x
- **Persona**: Experimental + GPUArchitect + GPUHwSpecific + Hybrid
- **Optimisation**: Warp-aggregated binary search
- **Speedup**: 1.28x
- **Confiance**: ✅ Baseline stable, pas de triche

## Rapport Anti-Triche

- **Agents suspects**: 1 (agent 07 a modifié test_cross_version.cpp)
- **Baseline stable**: ✅ Tous les autres agents ont laissé v1.hpp intact

## Recommandations

1. Combiner les top 2 optimisations → Speedup estimé: 1.8x
2. Investiger pourquoi 6 agents ont échoué (compilation ou tests)
3. Exclure l'agent 07 des résultats futurs
```

---

## CONTRAINTES TECHNIQUES

### Compilation
- **TOUJOURS -j4**: `cmake --build --preset experimental-cuda -j4`
- Évite OOM quand plusieurs agents compilent en parallèle

### Benchmarks
- **Séquentiels uniquement**: Un à la fois via l'agent spécialisé
- Évite interférence GPU entre agents

### Git Worktrees
- **Isolation totale**: Chaque agent dans son worktree
- **Réutilisables**: Les worktrees persistent entre runs

---

## FICHIERS DU SYSTÈME V2

- **`docs/AGENT_PERSONA_SYSTEM.md`**: Système de curseurs (inchangé)
- **`docs/ORCHESTRATOR_PROMPT_V2.md`**: Nouveau prompt avec chunking + agents spécialisés
- **`docs/ORCHESTRATOR_SPECS_V2.md`**: Ce fichier (spécifications V2)
- **`docs/ORCHESTRATOR_PROMPT.md`**: Ancienne version (gardée pour référence)
- **`docs/ORCHESTRATOR_SPECS.md`**: Ancienne version (gardée pour référence)

---

## PROCHAINES ÉTAPES

Une fois V2 validée:
1. Créer un Skill Claude Code pour l'orchestrateur
2. Explorer des agents spécialisés supplémentaires (profiling, etc.)
3. Tester avec N=24 agents réels
