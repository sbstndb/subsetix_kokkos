# AGENTS SPÉCIALISÉS - Extensions Possibles

Document exploratoire sur les agents spécialisés qui pourraient enrichir le système d'optimisation.

---

## AGENTS DÉJÀ IMPLÉMENTÉS

### 1. Agent Benchmark Spécialisé
- **Rôle**: Lancer les benchmarks séquentiellement sur tous les worktrees
- **Pourquoi**: Évite interférence GPU entre agents
- **Sortie**: Tableau comparatif avec speedups fiables

### 2. Agent Anti-Triche
- **Rôle**: Analyser git diff, vérifier baseline stable
- **Pourquoi**: Garantir l'intégrité des résultats
- **Sortie**: Rapport de confiance par agent

### 3. Agent Report
- **Rôle**: Générer le rapport final agrégant tous les résultats
- **Pourquoi**: Synthétiser les données pour l'utilisateur final
- **Sortie**: Rapport markdown complet avec tableaux et recommandations

---

## NOUVEAUX AGENTS PROPOSÉS

### 4. Agent Profiling GPU

**Rôle**: Analyser les hotspots GPU pour les top optimisations

**Prompt**:
```text
# AGENT PROFILING GPU

## TON RÔLE

Tu es l'agent profiling. Tu analyses les hotspots GPU pour les optimisations les plus prometteuses.

## CONTEXTE

- Worktrees cibles: Top M optimisations (ex: M=5)
- GPU: {GPU_ARCH}
- Outils: Nsight Compute (ncu) pour analyse kernel

## WORKFLOW

1. Sélectionner les top M optimisations

2. Pour chaque optimisation:
   ```bash
   cd /home/sbstndbs/subsetix_kokkos_optimized_opt{XX}

   # Profiling avec Nsight Compute
   ncu --set full \
       --export profile_optimized_opt{XX} \
       ./build-experimental-cuda/experimental/benchmarks/experimental_unified_comparison_benchmark \
         --benchmark_filter="3D_Large"
   ```

3. Analyser les métriques:
   - Occupation GPU
   - Memory bandwidth utilization
   - Warp efficiency
   - Divergence branches

4. Retourner un rapport avec:
   - Top 3 goulots d'étranglement par optimisation
   - Suggestions d'améliorations

## FORMAT DE RETOUR

```json
{
  "profiling_agent": "specialized",
  "gpu_arch": "{GPU_ARCH}",
  "analyzed_agents": [2, 5, 15, ...],
  "findings": [
    {
      "agent_id": "02",
      "gpu_occupancy": "65%",
      "memory_bandwidth": "180 GB/s (33% de pic)",
      "warp_efficiency": "82%",
      "main_bottleneck": "Binary search memory access pattern",
      "suggestions": ["Consider shared memory caching", "Try warp-aggregated search"]
    }
  ]
}
```
```

**Quand le lancer**: Après l'agent benchmark, sur les top 5 optimisations.

---

### 5. Agent Mémoire

**Rôle**: Analyser les patterns d'accès mémoire et proposer optimisations

**Prompt**:
```text
# AGENT MÉMOIRE

## TON RÔLE

Tu es l'agent mémoire. Tu analyses les patterns d'accès mémoire pour identifier les goulots.

## CONTEXTE

- Fichier cible: optimized.hpp (et baseline.hpp pour comparaison)
- Focus: Accès mémoire non coalescés, cache misses

## WORKFLOW

1. Lire optimized.hpp et baseline.hpp

2. Analyser les patterns d'accès:
   - Accès aux row_keys (random ou séquentiel?)
   - Accès aux intervals (coalesced?)
   - Utilisation de scratch memory?
   - Structures de données (SoA vs AoS)

3. Identifier les problèmes:
   - Non-coalesced global memory access
   - Cache line inefficace
   - False sharing potentiel

4. Retourner un rapport avec:
   - Problèmes identifiés par phase
   - Suggestions d'optimisations mémoire

## FORMAT DE RETOUR

```json
{
  "memory_agent": "specialized",
  "analysis": {
    "phase1_row_mapping": {
      "access_pattern": "Random access à row_keys_b",
      "coalesced": false,
      "issue": "Chaque thread lit un élément différent, non coalesced",
      "suggestion": "Utiliser scratch memory ou warp-aggregated read"
    },
    "phase4_fill": {
      "access_pattern": "Sequential access à intervals",
      "coalesced": true,
      "issue": "None",
      "suggestion": "Optimal"
    }
  },
  "priority_optimizations": [
    "Phase 1: Warp-aggregated binary search (réduit transactions 32→1)",
    "Phase 5: Compact inline pour éviter allocation supplémentaire"
  ]
}
```
```

---

### 6. Agent Fusion

**Rôle**: Identifier les opportunités de fusion de kernels

**Prompt**:
```text
# AGENT FUSION

## TON RÔLE

Tu es l'agent fusion. Tu identifies les opportunités de fusionner des kernels pour réduire la latence.

## CONTEXTE

- optimized.hpp a 5 phases distinctes avec des kernels séparés
- Chaque kernel = overhead de lancement

## WORKFLOW

1. Lire optimized.hpp

2. Analyser les dépendances entre phases:
   - Phase 1 → Phase 2 (dépendance: rows communs)
   - Phase 2 → Phase 3 (dépendance: row_counts)
   - Phase 3 → Phase 4 (dépendance: row_ptr)
   - Phase 4 → Phase 5 (dépendance: result)

3. Identifier les fusions possibles:
   - Phase 2+3: Count+Scan en un kernel?
   - Phase 3+4: Scan+Fill?
   - Phase 4+5: Fill+Compact?

4. Retourner un rapport avec:
   - Fusions possibles
   - Complexité estimée
   - Gain potentiel

## FORMAT DE RETOUR

```json
{
  "fusion_agent": "specialized",
  "possible_fusions": [
    {
      "phases": "2+3",
      "description": "Count+Scan en un kernel avec exclusive_scan préfix",
      "complexity": "Medium",
      "potential_gain": "1.1-1.2x",
      "risk": "Moderate"
    },
    {
      "phases": "4+5",
      "description": "Fill+Compact inline avec marquage",
      "complexity": "Medium",
      "potential_gain": "1.15-1.3x",
      "risk": "Low"
    }
  ],
  "recommended": "Fusion 4+5 (Fill+Compact) - meilleur ratio gain/risque"
}
```
```

---

### 7. Agent Corrélation

**Rôle**: Analyser les corrélations entre personas et succès

**Prompt**:
```text
# AGENT CORRÉLATION

## TON RÔLE

Tu es l'agent corrélation. Tu analyses quels personas ont le plus de succès.

## CONTEXTE

- N agents avec des personas différents
- Résultats benchmarks disponibles
- Question: Quels profils réussissent le mieux?

## WORKFLOW

1. Collecter les données:
   - Persona de chaque agent (6 curseurs)
   - Speedup obtenu
   - Statut (success/failed)

2. Analyser les corrélations:
   - Risk vs Success rate
   - Expertise vs Speedup moyen
   - Opt_type vs Success rate
   - Style vs Success rate

3. Retourner un rapport avec:
   - Personas qui réussissent le mieux
   - Combinaisons gagnantes
   - Recommandations pour runs futurs

## FORMAT DE RETOUR

```json
{
  "correlation_agent": "specialized",
  "total_agents": 24,
  "successful_agents": 18,
  "findings": {
    "risk_vs_success": {
      "Conservative": "80% success, speedup moyen 1.02x",
      "Moderate": "85% success, speedup moyen 1.12x",
      "Aggressive": "70% success, speedup moyen 1.25x",
      "Experimental": "50% success, speedup moyen 1.35x"
    },
    "expertise_vs_speedup": {
      "KokkosSpecialist": "1.08x mean",
      "GPUArchitect": "1.22x mean",
      "AlgorithmExpert": "1.18x mean",
      "MemoryArchitect": "1.15x mean"
    },
    "winning_combinations": [
      "Experimental + GPUArchitect + GPUHwSpecific → 1.35x mean",
      "Aggressive + AlgorithmExpert + Algorithmic → 1.28x mean"
    ]
  },
  "recommendations": [
    "Pour run futur: favoriser Experimental+GPUArchitect pour les agents à haut potentiel",
    "Éviter Conservative+QuickWin pour les runs exploratoires (peu de gain)"
  ]
}
```
```

---

### 8. Agent Combinaison

**Rôle**: Proposer des combinaisons d'optimisations

**Prompt**:
```text
# AGENT COMBINAISON

## TON RÔLE

Tu es l'agent combinaison. Tu proposes de combiner les meilleures optimisations.

## CONTEXTE

- Top M optimisations identifiées
- Question: Peut-on les combiner pour un speedup cumulé?

## WORKFLOW

1. Analyser les top M optimisations

2. Vérifier la compatibilité:
   - Les optimisations sont-elles orthogonales?
   - Y a-t-il des conflits?
   - Peut-on les appliquer ensemble?

3. Pour les combinaisons viables:
   - Estimer le speedup cumulé
   - Évaluer la complexité
   - Identifier les risques

4. Retourner un rapport avec:
   - Combinaisons recommandées
   - Speedup estimé
   - Ordre d'application suggéré

## FORMAT DE RETOUR

```json
{
  "combination_agent": "specialized",
  "top_optimizations": [2, 5, 15],
  "viable_combinations": [
    {
      "agents": [2, 5],
      "description": "Warp-aggregated search + Fusion kernels",
      "compatibility": "Orthogonal - pas de conflit",
      "estimated_speedup": "1.5-1.7x",
      "complexity": "Medium",
      "recommended": true
    },
    {
      "agents": [2, 5, 15],
      "description": "Toutes les optimisations top 3",
      "compatibility": "Partiel - risque de conflit mémoire",
      "estimated_speedup": "1.6-2.0x",
      "complexity": "High",
      "recommended": false,
      "risk": "Trop complexe, difficile à debugger"
    }
  ],
  "best_combination": {
    "agents": [2, 15],
    "estimated_speedup": "1.8x",
    "order": "Appliquer 2 d'abord, puis 15"
  }
}
```
```

---

### 9. Agent Documentation

**Rôle**: Générer la documentation des optimisations

**Prompt**:
```text
# AGENT DOCUMENTATION

## TON RÔLE

Tu es l'agent documentation. Tu génères la documentation des optimisations réussies.

## CONTEXTE

- Top M optimisations à documenter
- Public: Développeurs du projet

## WORKFLOW

1. Pour chaque optimisation dans le top M:
   - Lire le code modifié (git diff)
   - Comprendre l'optimisation
   - Rédiger la documentation

2. Générer un markdown avec:
   - Description de l'optimisation
   - Code avant/après
   - Résultats benchmarks
   - Recommandations d'application

## FORMAT DE SORTIE

```markdown
# Optimisations GPU v2 - Documentation

## Optimisation #1: Warp-Aggregated Binary Search

**Agent**: 02 - Le Visionnaire GPU
**Speedup**: 1.28x

### Description

L'optimisation réduit les transactions mémoire pendant le binary search...

### Code Avant

```cpp
// Chaque lane lit le même élément
const auto mid_val = rows_b[mid];
```

### Code Après

```cpp
// Lane 0 lit et broadcast
Kokkos::pair<CoordType, bool> result;
if (team.team_rank() == 0) {
  result = rows_b[mid];
}
result = Kokkos::pair<CoordType, bool>(shfl(&result.first, 0), shfl(&result.second, 0));
```

### Résultats

| Benchmark | Baseline (ms) | Optimized (ms) | Speedup |
|-----------|---------------|----------------|---------|
| 3D Large  | 19.6    | 15.2    | 1.28x   |

### Recommandations

- Appliquer à toutes les phases de recherche binaire
- Fonctionne uniquement sur GPU CUDA
- Fallback nécessaire pour CPU/OpenMP
```
```

---

## ARCHITECTURE AVEC TOUS LES AGENTS

```
┌──────────────────────────────────────────────────────────┐
│              ORCHESTRATEUR PRINCIPAL                      │
└──────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Agents Opt  │  │ Agent Bench │  │Anti-Triche  │
│  (N chunks) │  │ Spécialisé  │  │  Spécialisé │
└─────────────┘  └─────────────┘  └─────────────┘
         │                │                │
         └────────────────┼────────────────┘
                          ▼
                 ┌─────────────────┐
                 │  TOP OPTIMS     │
                 └─────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│Agent Profile│  │Agent Fusion │  │ Agent Comb  │
│   GPU       │  │             │  │             │
└─────────────┘  └─────────────┘  └─────────────┘
                          │
                 ┌─────────────────┐
                 │Agent Documentation│
                 └─────────────────┘
```

---

## RECOMMANDATIONS D'UTILISATION

| Agent | Quand lancer? | Priorité |
|-------|----------------|----------|
| Benchmark | Après tous les agents opt | **Obligatoire** |
| Anti-Triche | Après tous les agents opt | **Obligatoire** |
| Report | Après benchmark + antitriche | **Obligatoire** |
| Profiling GPU | Après benchmark, top 5 | Optionnel |
| Mémoire | Au début, pour identifier goulots | Optionnel |
| Fusion | Après analyse des phases | Optionnel |
| Corrélation | Après plusieurs runs | Optionnel |
| Combinaison | Après validation top optimisations | Optionnel |
| Documentation | À la fin, pour top optimisations | Optionnel |

---

## PROCHAINES ÉTAPES

1. ~~Implémenter et tester les agents obligatoires (Benchmark, Anti-Triche, Report)~~ **FAIT**
2. Expérimenter avec les agents optionnels
3. Créer des templates de prompts réutilisables
4. Intégrer dans l'orchestrateur principal
