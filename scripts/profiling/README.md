<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique
-->
# Kokkos Profiling Scripts

Scripts pour faciliter la génération et l'analyse des traces de profiling Kokkos.

## Scripts disponibles

### 1. `profile_benchmark.sh` - Script principal de profiling

Lance un benchmark expérimental avec un outil de profiling Kokkos.

```bash
./scripts/profile_benchmark.sh <preset> <tool> <benchmark_filter> [options]
```

**Arguments :**
- `preset` : Preset CMake (ex: `experimental-serial-profile`)
- `tool` : Outil de profiling (`kernel-timer`, `chrome-tracing`, `space-time-stack`)
- `benchmark_filter` : Filtre Google Benchmark (ex: `"3D.*LargeConfig"`)

**Options :**
- `-o, --output DIR` : Répertoire de sortie (défaut: `profiling_output/`)
- `-t, --threads N` : Threads OpenMP (défaut: 22)
- `-s, --sampling-prob N` : Probabilité de sampling 1-100% (défaut: 100 = pas de sampling)
- `-v, --sampler-verbose` : Afficher les kernels échantillonnés

**Exemples :**
```bash
# Profile 3D LargeConfig avec kernel-timer (Serial)
./scripts/profile_benchmark.sh experimental-serial-profile kernel-timer "3D.*LargeConfig"

# Profile 2D MediumConfig avec chrome-tracing (OpenMP), 10% sampling
./scripts/profile_benchmark.sh experimental-openmp-profile chrome-tracing "2D.*MediumConfig" -t 22 -s 10

# Profile SmallConfig avec space-time-stack (CUDA), 5% sampling
./scripts/profile_benchmark.sh experimental-cuda-gcc12-profile space-time-stack ".*SmallConfig" -s 5

# Profile avec verbose sampling pour voir quels kernels sont mesurés
./scripts/profile_benchmark.sh experimental-serial-profile kernel-timer ".*LargeConfig" -s 5 -v
```

**Exemples :**
```bash
# Profile 3D LargeConfig avec kernel-timer (Serial)
./scripts/profile_benchmark.sh experimental-serial-profile kernel-timer "3D.*LargeConfig"

# Profile 2D MediumConfig avec chrome-tracing (OpenMP)
./scripts/profile_benchmark.sh experimental-openmp-profile chrome-tracing "2D.*MediumConfig" -t 22

# Profile SmallConfig avec space-time-stack (CUDA)
./scripts/profile_benchmark.sh experimental-cuda-gcc12-profile space-time-stack ".*SmallConfig"
```

### 2. `profile_all_backends.sh` - Profiler tous les backends

Lance le profiling sur les 3 backends (Serial, OpenMP, CUDA) et compare les résultats.

```bash
./scripts/profiling/profile_all_backends.sh <tool> <benchmark_filter>
```

**Exemple :**
```bash
./scripts/profiling/profile_all_backends.sh kernel-timer "3D.*LargeConfig"
```

Génère des traces dans `profiling_output/<timestamp>-<tool>/Serial/`, `.../OpenMP/`, `.../CUDA/`.

### 3. `analyze_traces.sh` - Analyser les traces

Analyse les fichiers de tracing et génère des rapports lisibles.

```bash
./scripts/profiling/analyze_traces.sh <trace_directory>
```

**Fonctionnalités :**
- Convertit les fichiers `.dat` en JSON (kernel-timer)
- Extrait les kernels les plus coûteux (chrome-tracing)
- Résume les allocations mémoire (space-time-stack)

### 4. `compare_runs.sh` - Comparer plusieurs runs

Compare les résultats de plusieurs runs de profiling.

```bash
./scripts/profiling/compare_runs.sh <trace_dir1> <trace_dir2> ...
```

## Outils de profiling disponibles

| Outil | Sortie | Usage recommandé |
|--------|--------|-----------------|
| **kernel-timer** | `.dat` + JSON | Analyse quantitative des temps par kernel |
| **chrome-tracing** | `.json` | Visualisation timeline (chrome://tracing) |
| **space-time-stack** | stdout | Analyse complète (temps + mémoire) |
| **memory-hwm** | stdout | High water mark mémoire (fin de programme) |
| **memory-usage** | stdout | Suivi consommation mémoire (timestamps) |

## Sampling (KernelSampler)

Le **sampling** permet de réduire l'overhead de profiling en ne mesurant qu'un pourcentage des kernels.

**Quand l'utiliser ?**
- Pour réduire l'overhead massif de space-time-stack (surtout sur GPU)
- Pour des runs de profiling très longs
- Pour faire une analyse rapide avant un profiling complet

**Variables d'environnement :**
- `KOKKOS_TOOLS_SAMPLER_PROB=N` : Probabilité de sampling (1-100)
- `KOKKOS_TOOLS_SAMPLER_VERBOSE=1` : Affiche les kernels échantillonnés

**Recommandations :**
| Outil | Sampling suggéré | Overhead réduit |
|-------|-----------------|----------------|
| space-time-stack | 5-10% | 50-70% |
| chrome-tracing | 10-20% | 40-60% |
| kernel-timer | 1-5% | 20-40% |

**Note :** Le sampling ne réduit pas nécessairement le temps d'exécution, mais réduit la quantité de données collectées et les écritures dans les fichiers de sortie.

## Organisation des fichiers de sortie

```
profiling_output/
├── 20250122-145302-kernel-timer/
│   ├── Serial/
│   │   ├── sbstndbs-123456.dat
│   │   ├── sbstndbs-123456.json
│   │   ├── benchmark_output.txt
│   │   └── summary.txt
│   ├── OpenMP/
│   └── CUDA/
└── 20250122-150000-chrome-tracing/
    └── ...
```

## Workflow recommandé

1. **Lancer le profiling**
   ```bash
   ./scripts/profiling/profile_all_backends.sh chrome-tracing "3D.*LargeConfig"
   ```

2. **Analyser les traces**
   ```bash
   ./scripts/profiling/analyze_traces.sh profiling_output/20250122-150000-chrome-tracing/*
   ```

3. **Visualiser (chrome-tracing)**
   - Ouvrir `chrome://tracing`
   - Cliquer sur "Load"
   - Sélectionner les fichiers `.json`

4. **Comparer les backends**
   ```bash
   ./scripts/profiling/compare_runs.sh \
       profiling_output/20250122-150000-chrome-tracing/Serial \
       profiling_output/20250122-150000-chrome-tracing/OpenMP \
       profiling_output/20250122-150000-chrome-tracing/CUDA
   ```
