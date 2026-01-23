# ORCHESTRATEUR D'OPTIMISATION KOKKOS CUDA - Spécifications

**Date**: 2025-01-23
**Cible**: Optimiser v2.hpp (experimental module) pour GPU CUDA
**Baseline**: v1.hpp (ne pas modifier)
**Stratégie**: 10 worktrees parallèles, 10 personas uniques générés aléatoirement

---

## PRINCIPE CLÉ

**NE PAS pré-définir les optimisations**. Chaque agent reçoit un **profil unique** (persona) et doit trouver lui-même les opportunités d'optimisation qui correspondent à son profil.

---

## LES 6 CURSEURS

Chaque agent est défini par 6 attributs tirés aléatoirement:

| Curseur | Valeurs | Weights | Description |
|---------|---------|---------|-------------|
| **RISK_LEVEL** | Conservative, Moderate, Aggressive, Experimental | 25%, 40%, 25%, 10% | Niveau de risque accepté |
| **EXPERTISE** | KokkosSpecialist, GPUArchitect, AlgorithmExpert, MemoryArchitect, SystemsThinker, ParallelismExpert, DataStructureSpecialist | Uniform | Domaine d'expertise |
| **OPT_TYPE** | QuickWin, KokkosPattern, GPUHwSpecific, Algorithmic, Structural, MemoryLayout, LatencyHiding | Uniform | Type d'optimisation préféré |
| **STYLE** | Analytical, Experimental, Incremental, Hybrid | 20%, 30%, 20%, 30% | Style de recherche |
| **SCOPE** | Local, Regional, Global | Uniform | Portée des changements |
| **INNOVATION** | Proven, Novel, Wild | 40%, 40%, 20% | Niveau d'innovation |

**Nombre de personas possibles**: 4 × 7 × 7 × 4 × 3 × 3 = **7,056 combinaisons uniques**

---

## WORKFLOW PRINCIPAL

### Phase 0: Setup Initial (ONE-TIME)

```bash
# 1. Détecter le GPU
nvidia-smi -L

# 2. Créer les 10 worktrees depuis main
cd /home/sbstndbs/subsetix_kokkos
git worktree list | grep v2_opt || (
  for i in {01..10}; do
    git worktree add ../subsetix_kokkos_v2_opt${i} -b feature/v2-opt${i}
  done
)

# 3. Vérifier
git worktree list
```

### Phase 1: Générer les 10 Personas

Pour chaque agent (01 à 10), générer aléatoirement:

```python
import random

def generate_persona(agent_id: int) -> dict:
    """Génère un profil unique pour un agent"""

    # Curseur 1: Risk
    risk = random.choices(
        ["Conservative", "Moderate", "Aggressive", "Experimental"],
        weights=[0.25, 0.40, 0.25, 0.10]
    )[0]

    # Curseur 2: Expertise
    expertise = random.choice([
        "KokkosSpecialist", "GPUArchitect", "AlgorithmExpert",
        "MemoryArchitect", "SystemsThinker", "ParallelismExpert",
        "DataStructureSpecialist"
    ])

    # Curseur 3: Type d'optimisation
    opt_type = random.choice([
        "QuickWin", "KokkosPattern", "GPUHwSpecific",
        "Algorithmic", "Structural", "MemoryLayout", "LatencyHiding"
    ])

    # Curseur 4: Style
    style = random.choices(
        ["Analytical", "Experimental", "Incremental", "Hybrid"],
        weights=[0.2, 0.3, 0.2, 0.3]
    )[0]

    # Curseur 5: Scope
    scope = random.choice(["Local", "Regional", "Global"])

    # Curseur 6: Innovation
    innovation = random.choices(
        ["Proven", "Novel", "Wild"],
        weights=[0.4, 0.4, 0.2]
    )[0]

    # Générer un nom créatif
    persona_name = generate_persona_name(risk, expertise)

    return {
        "agent_id": agent_id,
        "persona_name": persona_name,
        "risk": risk,
        "expertise": expertise,
        "opt_type": opt_type,
        "style": style,
        "scope": scope,
        "innovation": innovation
    }

def generate_persona_name(risk: str, expertise: str) -> str:
    """Génère un nom mémorables pour le persona"""
    adjectives = {
        "Conservative": ["Prudent", "Méthodique", "Sage"],
        "Moderate": ["Équilibré", "Raisonnable", "Réfléchi"],
        "Aggressive": ["Audacieux", "Ambitieux", "Chercheur"],
        "Experimental": ["Visionnaire", "Créatif", "Explorateur"]
    }
    nouns = {
        "KokkosSpecialist": "des Patterns Kokkos",
        "GPUArchitect": "du GPU",
        "AlgorithmExpert": "des Algorithmes",
        "MemoryArchitect": "de la Mémoire",
        "SystemsThinker": "des Systèmes",
        "ParallelismExpert": "du Parallélisme",
        "DataStructureSpecialist": "des Structures de Données"
    }

    adj = random.choice(adjectives[risk])
    noun = random.choice(nouns[expertise])
    return f"{adj} {noun}"
```

### Phase 2: Lancer les 10 Agents (PARALLÈLE)

Utiliser le Task tool pour lancer 10 agents en une seule requête.

Chaque agent reçoit:
- Un worktree dédié
- Un persona unique (les 6 curseurs)
- Un prompt personnalisé basé sur son persona

**IMPORTANT**: L'agent doit TROUVER lui-même l'optimisation, pas la recevoir!

### Phase 3: Attendre et Collecter

Attendre que les 10 agents terminent. Collecter leurs résultats JSON.

### Phase 4: Agréger et Analyser

1. Créer un tableau comparatif des 10 personas
2. Analyser la diversité des approches
3. Sélectionner les 2 meilleures optimisations
4. Poser une question à l'utilisateur pour confirmation

### Phase 5: Rapport Final

Générer un rapport avec:
- Tableau complet des 10 personas et résultats
- Analyse de la diversité
- Top 2 optimisations
- Recommandations

---

## FORMAT DE PROMPT PAR AGENT

Chaque agent reçoit un prompt personnalisé. Voir `ORCHESTRATOR_PROMPT.md` pour le template complet.

---

## FORMAT DE RÉSULTAT PAR AGENT

Chaque agent retourne un JSON:

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
  "optimization_proposed": "Éliminer 2 fences inutiles dans Phase 1",
  "status": "success",
  "compile": true,
  "tests_pass": true,
  "benchmark_done": true,
  "v1_time_ms": 125.4,
  "v2_time_ms": 119.2,
  "speedup": 1.05,
  "notes": "Fences après parallel_scan étaient redondants"
}
```

---

## STRUCTURE DES WORKTREES

```
/home/sbstndbs/subsetix_kokkos/         # Main repo (v1 baseline)
/home/sbstndbs/subsetix_kokkos_v2_opt01/  # Worktree agent 01
/home/sbstndbs/subsetix_kokkos_v2_opt02/  # Worktree agent 02
...
/home/sbstndbs/subsetix_kokkos_v2_opt10/  # Worktree agent 10

/home/sbstndbs/builds/main-cuda/          # Build baseline
/home/sbstndbs/builds/v2_opt01-cuda/     # Build agent 01
...
/home/sbstndbs/builds/v2_opt10-cuda/     # Build agent 10
```

---

## EXEMPLES DE PERSONAS

### Persona #1: "Le Prudent Kokkosien"
```json
{
  "risk": "Conservative",
  "expertise": "KokkosSpecialist",
  "opt_type": "QuickWin",
  "style": "Incremental",
  "scope": "Local",
  "innovation": "Proven"
}
```
→ Cherche des changements simples dans une fonction, utilise des patterns Kokkos bien connus, procède pas à pas.

### Persona #2: "Le Visionnaire GPU"
```json
{
  "risk": "Experimental",
  "expertise": "GPUArchitect",
  "opt_type": "GPUHwSpecific",
  "style": "Hybrid",
  "scope": "Global",
  "innovation": "Wild"
}
```
→ Propose des changements majeurs exploitant le hardware GPU de manière créative, avec un raisonnement solide.

### Persona #3: "L'Audacieux des Algorithmes"
```json
{
  "risk": "Aggressive",
  "expertise": "AlgorithmExpert",
  "opt_type": "Algorithmic",
  "style": "Analytical",
  "scope": "Regional",
  "innovation": "Novel"
}
```
→ Analyse une phase de l'algorithme, propose un nouvel algo avec un raisonnement solide, accepte les risques.

---

## COMMANDES RAPIDES

### Tout setup en une fois
```bash
nvidia-smi -L | grep -oP 'GPU.*: \K[^ ]+' > /tmp/gpu_arch.txt

cd /home/sbstndbs/subsetix_kokkos
git worktree list | grep v2_opt || (
  for i in {01..10}; do
    git worktree add ../subsetix_kokkos_v2_opt${i} -b feature/v2-opt${i}
  done
)

git worktree list
```

### Lancer tous les benchmarks
```bash
GPU_ARCH=$(nvidia-smi -L | grep -oP 'NVIDIA \K[^ ]+' | head -1 | tr '[:lower:]' '[:upper:]')

for i in {01..10}; do
  echo "=== Testing v2_opt${i} ==="
  cd /home/sbstndbs/subsetix_kokkos_v2_opt${i}
  cmake --preset experimental-cuda -DKokkos_ARCH_${GPU_ARCH}=ON 2>&1 | tail -5
  cmake --build --preset experimental-cuda -j 2>&1 | tail -5
  ctest --preset experimental-cuda --output-on-failure 2>&1 | tail -20
  ./build-experimental-cuda/experimental/benchmarks/experimental_unified_comparison_benchmark \
    --benchmark_filter="3D_Large" \
    --benchmark_repetitions=10 \
    --benchmark_report_aggregates_only=true \
    --benchmark_format=json > /tmp/v2_opt${i}_bench.json
done
```

---

## CHECKLIST DE VALIDATION

Avant de considérer une optimisation comme "success":

- [ ] Code compile sans erreur
- [ ] `test_cross_version` passe (v1 vs v2 identiques mathématiquement)
- [ ] `test_large_mesh` passe (n=8192)
- [ ] Benchmark 3D Large termine sans crash
- [ ] Speedup > 1.02x (minimum 2% pour être considéré amélioration)
- [ ] Pas de régression sur d'autres tests

---

## FICHIERS DU SYSTÈME

- **`docs/AGENT_PERSONA_SYSTEM.md`**: Spécifications détaillées des curseurs et personas
- **`docs/ORCHESTRATOR_PROMPT.md`**: Prompt à copier/coller pour lancer l'orchestrateur
- **`docs/ORCHESTRATOR_SPECS.md`**: Ce fichier (spécifications techniques)

---

## NOTES IMPORTANTES

1. **Adaptabilité**: Le système est conçu pour être modifié entre deux runs
2. **Parallélisme**: Les 10 worktrees sont indépendants, les agents peuvent travailler en vrai parallèle
3. **Isolation**: Chaque worktree a son propre build directory
4. **Baseline v1**: Jamais modifiée, sert de référence
5. **Diversité**: Les personas sont générés aléatoirement pour garantir des approches variées

---

## PROCHAINES ÉTAPES

Une fois le système validé:
1. Copier le prompt depuis `ORCHESTRATOR_PROMPT.md`
2. Le coller dans Claude Code
3. L'orchestrateur générera 10 personas uniques et lancera 10 agents
4. Chaque agent explorera une optimisation différente
