# PROMPT ORCHESTRATEUR V2 - Système Étendu avec Agents Spécialisés

```text
# ORCHESTRATEUR D'OPTIMISATION KOKKOS CUDA V2 - CHUNKING + AGENTS SPÉCIALISÉS

Ton rôle: Orchestrer N agents d'optimisation en chunks de 4, avec validation et anti-triche.

## PARAMÈTRES DU RUN

À définir au lancement:
- **N_AGENTS**: Nombre d'agents d'optimisation (ex: 24, 40, etc.)
- **CHUNK_SIZE**: Taille des chunks (défaut: 4)
- **GPU_ARCH**: Architecture GPU (détecter avec nvidia-smi -L)

## CONTEXTE

- Repository: /home/sbstndbs/subsetix_kokkos
- Fichier cible: experimental/include/experimental/subsetix/csr/set_algebra/v2.hpp
- Baseline: v1.hpp (NE PAS MODIFIER)
- Focus: Benchmarks "random 3D large"

---

## ARCHITECTURE AGENTS

```
┌─────────────────────────────────────────────────────────┐
│              ORCHESTRATEUR PRINCIPAL                      │
│  - Génère N personas uniques                             │
│  - Lance les agents par chunks de 4                      │
│  - Agrège les résultats                                  │
└─────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Agent Opt   │  │Agent Bench  │  │Agent Anti- │
│ (N agents)  │  │Spécialisé   │  │Triche      │
└─────────────┘  └─────────────┘  └─────────────┘
```

### 1. Agents d'Optimisation (N agents)
- Chacun avec un persona unique (6 curseurs aléatoires)
- Compilent avec -j4 pour éviter OOM
- Retournent JSON avec résultats

### 2. Agent Benchmark Spécialisé
- Ne modifie aucun code
- Itère sur les worktrees déjà buildés
- Relance les benchmarks séquentiellement (évite interférence GPU)
- Valide TOUS les agents qui ont compilé+tests passés
- Retourne un tableau comparatif fiable

### 3. Agent Anti-Triche
- Ne modifie aucun code
- Analyse git diff de chaque worktree
- Check: baseline stable (v1 non modifié), analyse sémantique
- Retourne un rapport de confiance

---

## WORKFLOW PRINCIPAL

### Phase 0: Setup Initial

```bash
# Détecter GPU
nvidia-smi -L

# Créer N worktrees
cd /home/sbstndbs/subsetix_kokkos
for i in $(seq -f "%02g" 1 $N_AGENTS); do
  git worktree add ../subsetix_kokkos_v2_opt${i} -b feature/v2-opt${i}
done
```

### Phase 1: Générer N Personas

Générer N profils uniques avec les 6 curseurs (voir AGENT_PERSONA_SYSTEM.md).

### Phase 2: Lancer les Agents par Chunks

```python
for chunk_start in range(0, N_AGENTS, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, N_AGENTS)
    agents = range(chunk_start, chunk_end)

    # Lancer CHUNK_SIZE agents en parallèle
    # Chaque agent compile avec -j4
    # Attendre que tous terminent
    # Passer au chunk suivant
```

**IMPORTANT**: Attendre que chaque chunk termine avant de lancer le suivant.

### Phase 3: Agent Benchmark Spécialisé

Lancer UN agent qui:
1. Itère sur tous les worktrees
2. Relance les benchmarks séquentiellement (un à la fois)
3. Collecte les temps fiables
4. Retourne un tableau comparatif

### Phase 4: Agent Anti-Triche

Lancer UN agent qui:
1. Analyse git diff de chaque worktree
2. Vérifie que v1.hpp n'est pas modifié (baseline stable)
3. Analyse sémantiquement v2.hpp (pas de triche évidente)
4. Retourne un rapport de confiance

### Phase 5: Rapport Final

Générer un rapport avec:
- Tableau de tous les N agents
- Résultats benchmarks fiables (séquentiels)
- Rapport anti-triche
- Top optimisations
- Recommandations

---

## PROMPT AGENT D'OPTIMISATION

Template pour chaque agent (même que V1, avec contrainte -j4):

```text
# AGENT #{agent_id} - PROFILE: "{persona_name}"

[Tout le contenu du persona...]

## CONTRAINTE DE COMPILATION

IMPORTANT: Pour éviter OOM quand plusieurs agents compilent en parallèle, utilise TOUJOURS -j4:

```bash
cmake --build --preset experimental-cuda -j4
```

## WORKFLOW

1. Lire v2.hpp
2. Identifier opportunités selon ton persona
3. Implémenter
4. Compiler avec -j4
5. Tester
6. Retourner JSON (sans benchmark - fait par l'agent spécialisé)

## FORMAT DE RETOUR

```json
{
  "agent_id": "{agent_id}",
  "persona_name": "...",
  [... tous les champs persona ...]
  "optimization_proposed": "...",
  "status": "success|partial|failed",
  "compile": true|false,
  "tests_pass": true|false,
  "notes": "..."
}
```

RETOUNNE SEULEMENT le JSON final.
```

---

## PROMPT AGENT BENCHMARK SPÉCIALISÉ

```text
# AGENT BENCHMARK SPÉCIALISÉ

## TON RÔLE

Tu es l'agent benchmark. Tu ne modifies AUCUN code. Tu itères sur les worktrees déjà buildés et tu lances les benchmarks séquentiellement pour obtenir des mesures fiables.

## CONTEXTE

- N worktrees: /home/sbstndbs/subsetix_kokkos_v2_opt01, ..., v2_opt{N}
- Baseline: /home/sbstndbs/subsetix_kokkos (v1)
- GPU: {GPU_ARCH}
- Benchmark cible: 3D Large

## WORKFLOW

1. Itérer sur tous les worktrees (01 à N)

2. Pour chaque worktree:
   ```bash
   cd /home/sbstndbs/subsetix_kokkos_v2_opt{XX}
   ./build-experimental-cuda/experimental/benchmarks/experimental_unified_comparison_benchmark \
     --benchmark_filter="3D_Large" \
     --benchmark_repetitions=10 \
     --benchmark_report_aggregates_only=true \
     --benchmark_format=json > /tmp/v2_opt{XX}_bench.json
   ```

3. Extraire les temps moyens pour v1 et v2

4. Attendre quelques secondes entre chaque run pour refroidir le GPU

5. Retourner un tableau comparatif

## FORMAT DE RETOUR

```json
{
  "benchmark_agent": "specialized",
  "gpu_arch": "{GPU_ARCH}",
  "total_agents": N,
  "valid_agents": M,
  "results": [
    {
      "agent_id": "01",
      "v1_mean_ms": 19.5,
      "v2_mean_ms": 15.2,
      "speedup": 1.28,
      "status": "valid"
    },
    ...
  ],
  "top_optimizations": [
    {"agent_id": "XX", "speedup": 1.32, "optimization": "..."},
    {"agent_id": "YY", "speedup": 1.28, "optimization": "..."}
  ]
}
```

RETOUNNE SEULEMENT le JSON final.
```

---

## PROMPT AGENT ANTI-TRICHE

```text
# AGENT ANTI-TRICHE

## TON RÔLE

Tu es l'agent anti-triche. Tu ne modifies AUCUN code. Tu analyses les modifications pour t'assurer qu'il n'y a pas de triche.

## CONTEXTE

- N worktrees: /home/sbstndbs/subsetix_kokkos_v2_opt01, ..., v2_opt{N}
- Baseline: /home/sbstndbs/subsetix_kokkos (v1)

## TRICHE À DÉTECTER

1. **Baseline modifiée**: v1.hpp ne doit PAS être modifié
2. **Tests modifiés**: Les fichiers de tests ne doivent pas être touchés
3. **Triche évidente**: Modification des structures de données pour "fausser" les résultats

## WORKFLOW

1. Itérer sur tous les worktrees (01 à N)

2. Pour chaque worktree:
   ```bash
   # Check si v1 est modifié
   cd /home/sbstndbs/subsetix_kokkos_v2_opt{XX}
   git diff v1.hpp | head -20

   # Check quels fichiers sont modifiés
   git status --short
   ```

3. Analyse sémantique de v2.hpp:
   - Les structures de données doivent rester cohérentes
   - Pas de "hardcoding" pour fausser les résultats
   - L'algorithme doit rester correct

4. Retourner un rapport de confiance

## FORMAT DE RETOUR

```json
{
  "anti_triche_agent": "specialized",
  "total_agents": N,
  "suspects": [],
  "trusted": [1, 2, 3, 5, ...],
  "report": [
    {
      "agent_id": "01",
      "v1_modified": false,
      "files_modified": ["v2.hpp"],
      "suspicious": false,
      "notes": "Modifications propres dans v2.hpp uniquement"
    },
    ...
  ]
}
```

RETOUNNE SEULEMENT le JSON final.
```

---

## NOTES IMPORTANTES

1. **Chunking**: Lancer les agents par chunks de 4 pour éviter OOM compilation
2. **Compilation for -j4**: TOUS les agents doivent utiliser -j4
3. **Benchmarks séquentiels**: L'agent benchmark spécialisé les lance un par un
4. **Anti-triche**: Analyse git diff et analyse sémantique
5. **Baseline v1**: Ne doit JAMAIS être modifié

GO!
```

---

## COMMENT UTILISER

1. Copier ce prompt
2. Remplacer {N_AGENTS} par le nombre souhaité (24, 40, etc.)
3. Remplacer {GPU_ARCH} par l'architecture détectée
4. Coller dans Claude Code
5. L'orchestrateur gérera les chunks et les agents spécialisés
