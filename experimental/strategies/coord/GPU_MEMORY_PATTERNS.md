# GPU Memory Access Patterns - Analyse Détaillée

Ce document analyse en détail les patterns d'accès mémoire GPU pour chaque stratégie de représentation 3D.

---

## Table des Matières

1. [Architecture Mémoire GPU - Rappels](#architecture-mémoire-gpu)
2. [Morton Encoding](#1-morton-encoding)
3. [Hash-Based Lookup](#2-hash-based-lookup)
4. [Merge-Path Set Algebra](#3-merge-path-set-algebra)
5. [Bitmap Representation](#4-bitmap-representation)
6. [Tiled Memory Layout](#5-tiled-memory-layout)
7. [Octree Hierarchical](#6-octree-hierarchical)
8. [Hybrid Adaptive](#7-hybrid-adaptive)
9. [Synthèse Comparative](#synthèse-comparative)

---

## Architecture Mémoire GPU - Rappels

### Hiérarchie Mémoire

```
┌─────────────────────────────────────────────────────────────┐
│                        GPU Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Registers  │───▶│  L1 Cache    │───▶│  L2 Cache    │   │
│  │  (per thread)│    │  (per SM)    │    │  (shared)    │   │
│  │  ~256 bytes  │    │  128 KB      │    │  40-80 MB    │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                   │                   │           │
│         └───────────────────┼───────────────────┘           │
│                             ▼                               │
│                    ┌──────────────┐                         │
│                    │ Global Memory│                         │
│                    │  (HBM2/GDDR6)│                        │
│                    │  16-80 GB    │                         │
│                    │  ~500 GB/s   │                         │
│                    └──────────────┘                         │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Shared Memory (per block)               │   │
│  │              48-164 KB (user-configurable)           │   │
│  │  ≈ 100× faster than global (user-managed)           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Concepts Clés

**Coalescing**: 32 threads d'un warp accèdent à 32 mots consécutifs → 1 transaction unique

```
✓ Coalesced (32 bytes in 1 transaction):
Thread 0: [0x1000]  Thread 1: [0x1004]  ...  Thread 31: [0x107C]
└────────────────────────────────────────────────────────────┘
         1 transaction de 128 bytes

✗ Scattered (32 threads = 32 transactions):
Thread 0: [0x1000]  Thread 1: [0x5A32]  ...  Thread 31: [0xFF84]
└────────────────────────────────────────────────────────────┘
         32 transactions de 128 bytes = catastrophique
```

**Warp Divergence**: Threads prennent des chemins différents → exécution sérialisée

```
┌─────────────────┐
│ Warp (32 threads)│
└────────┬────────┘
         │
    ┌────┴────┐
    │ if      │ else  ← 16 threads vont if, 16 threads vont else
    ▼         ▼
  [16 thr]  [16 thr]   ← Sérialisation! Les 16 autres attendent
    │         │
    └────┬────┘
         │
    Tous les threads finissent ensembles
```

---

## 1. Morton Encoding

### Concept de Base

L'encodage Morton (Z-order curve) intercale les bits de Y et Z pour créer un ordre spatial qui préserve la localité 3D.

```
Lexicographic (actuel):          Morton (Z-order):
┌───┬───┬───┬───┐              ┌───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │              │ 0 │ 2 │ 8 │10 │  ← Notez l'ordre Z
├───┼───┼───┼───┤              ├───┼───┼───┼───┤
│ 4 │ 5 │ 6 │ 7 │              │ 1 │ 3 │ 9 │11 │  ← Voisins 3D
├───┼───┼───┼───┤              ├───┼───┼───┼───┤
│ 8 │ 9 │10 │11 │              │ 4 │ 6 │12 │14 │     adjacents
├───┼───┼───┼───┤              ├───┼───┼───┼───┤
│12 │13 │14 │15 │              │ 5 │ 7 │13 │15 │
└───┴───┴───┴───┘              └───┴───┴───┴───┘
```

### Pattern d'Accès Mémoire

#### Comparaison Lexicographic vs Morton

```
Lexicographic (y-major then z):
┌────────────────────────────────────────────────────────────┐
│ row_keys array (sorted by y, then z):                      │
│                                                             │
│  [0]: (y=0, z=0) ────┐                                      │
│  [1]: (y=0, z=1)      │                                     │
│  [2]: (y=0, z=2)      │  Même y, contigu                   │
│  [3]: (y=0, z=3) ────┘                                     │
│  [4]: (y=1, z=0) ────┐  ← SAUT MASSIF dans row_keys        │
│  [5]: (y=1, z=1)      │  mais voisins 3D!                  │
│  [6]: (y=1, z=2)      │  Problème: (0,3) et (1,0) sont     │
│  [7]: (y=1, z=3) ────┘  voisins 3D mais loin en mémoire   │
└────────────────────────────────────────────────────────────┘

Morton (Z-order):
┌────────────────────────────────────────────────────────────┐
│ row_keys array (sorted by morton code):                     │
│                                                             │
│  [0]: morton=0b0000  (y=0, z=0)                            │
│  [1]: morton=0b0001  (y=0, z=1)  ← Voisins 3D              │
│  [2]: morton=0b0010  (y=1, z=0)  ← adjacents en mémoire   │
│  [3]: morton=0b0011  (y=1, z=1)                            │
│  [4]: morton=0b0100  (y=0, z=2)                            │
│  [5]: morton=0b0101  (y=0, z=3)  ← Localité 3D préservée  │
│  [6]: morton=0b0110  (y=1, z=2)                            │
│  [7]: morton=0b0111  (y=1, z=3)                            │
└────────────────────────────────────────────────────────────┘
```

#### Cache Behavior Visualisé

```
Cache Line (128 bytes) = 16 RowKey3D entries

Lexicographic - Cache thrashing:
┌──────────────────────────────────────────────────────────┐
│ Cache Line 0:  (0,0) (0,1) (0,2) (0,3) ... (0,7)        │
│ Cache Line 1:  (0,8) (0,9) (0,10)(0,11)... (0,15)       │
│ Cache Line 2:  (1,0) (1,1) (1,2) (1,3) ... (1,7)        │
│                                                           │
│ Problème: Pour accéder à tous les voisins 3D de (0,3):   │
│   - Besoin de (1,2), (1,3), (1,4)                        │
│   - Mais (1,2) est dans Cache Line 2                     │
│   - → Cache miss!                                         │
└──────────────────────────────────────────────────────────┘

Morton - Spatial locality:
┌──────────────────────────────────────────────────────────┐
│ Cache Line 0:  (0,0) (0,1) (1,0) (1,1)                  │
│                  (0,2) (0,3) (1,2) (1,3)  ← Voisins 3D!  │
│                  ...                                      │
│                                                           │
│ Avantage: Pour accéder aux voisins 3D de (0,3):           │
│   - (0,2), (0,4), (1,2), (1,3), (1,4)                   │
│   - Tous dans 1-2 cache lines contiguës                  │
│   - → Cache hit +80%                                      │
└──────────────────────────────────────────────────────────┘
```

### Impact sur les Transactions Mémoire

```
Binary Search - RowKey3D (2 comparaisons):
┌────────────────────────────────────────────────────────────┐
│ Thread: binary_search(row_keys, target={y, z})            │
│                                                             │
│  Itération 1:                                              │
│    mid = row_keys[N/2]                                     │
│    if (mid.y < target.y) ─┐  ← Branch 1                    │
│        return false        │                               │
│    else if (mid.y == target.y)                             │
│        if (mid.z < target.z) ─┐  ← Branch 2 (nested!)     │
│            return false        │                           │
│        else if (mid.z == target.z)                         │
│            return true                                     │
│                                                             │
│  Problème: 2 branches imbriquées = divergence sévère      │
│  Warp efficiency: ~60%                                      │
└────────────────────────────────────────────────────────────┘

Binary Search - Morton (1 comparaison):
┌────────────────────────────────────────────────────────────┐
│ Thread: binary_search(row_keys, target_morton)             │
│                                                             │
│  Itération 1:                                              │
│    mid = row_keys[N/2]                                     │
│    if (mid.morton < target_morton) ─┐  ← 1 seule branch!   │
│        return false                   │                     │
│    else if (mid.morton == target_morton)                   │
│        return true                                           │
│                                                             │
│  Avantage: 1 branch unique = divergence minimale           │
│  Warp efficiency: ~85%                                      │
└────────────────────────────────────────────────────────────┘
```

### Synthèse Morton

| Métrique | RowKey3D (actuel) | Morton | Amélioration |
|----------|-------------------|--------|--------------|
| Taille row key | 8 bytes | 8 bytes | identique |
| Comparaisons/itération | 2 | 1 | **2× moins** |
| Cache hit rate (voisins) | ~40% | ~60% | **+50%** |
| Warp efficiency | 60% | 85% | **+40% relatif** |
| Transactions (5M rows) | ~46M | ~23M | **2× moins** |

**Verdict**: Morton ne change pas la footprint mémoire mais améliore significativement l'utilisation du cache et réduit la divergence. Le break-even est autour de 10K rows.

---

## 2. Hash-Based Lookup

### Structure Mémoire

```
┌────────────────────────────────────────────────────────────┐
│                    Hash Table Layout                        │
│                                                             │
│  ┌──────────────────────────────────────────────────┐     │
│  │  keys[capacity]     RowKey3D ou Morton (8 bytes) │     │
│  ├──────────────────────────────────────────────────┤     │
│  │  values[capacity]   std::size_t interval index   │     │
│  ├──────────────────────────────────────────────────┤     │
│  │  occupied[capacity] bool (1 byte)               │     │
│  └──────────────────────────────────────────────────┘     │
│                                                             │
│  Problème: False sharing potentiel avec 3 arrays séparés   │
│                                                             │
│  Solution recommandée: Interleaved layout                   │
│                                                             │
│  ┌──────────────────────────────────────────────────┐     │
│  │  struct Entry {                                   │     │
│  │    RowKey3D key;      // 8 bytes                  │     │
│  │    std::size_t value; // 8 bytes                  │     │
│  │    bool occupied;     // 1 byte ( + 7 padding)   │     │
│  │  } // 16 bytes aligné pour coalescing parfait     │     │
│  │  Entry entries[capacity];                         │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────┘
```

### Pattern d'Accès - Linear Probing

```
Linear Probing (très GPU-friendly):
┌────────────────────────────────────────────────────────────┐
│ Thread i: hash(key) → idx = 42                             │
│                                                             │
│  Entrée 42: occupée? key match?                            │
│    ┌─Oui─→ Return value                                    │
│    │                                                        │
│    └─Non─→ Entrée 43: occupée? key match?                  │
│            ┌─Oui─→ Return value                            │
│            │                                                │
│            └─Non─→ Entrée 44: ...                          │
│                     │                                       │
│                     └─→ Séquence linéaire                  │
│                                                             │
│  AVANTAGE: Accès idx, idx+1, idx+2... = séquentiel         │
│  → Coalescing parfait entre threads du warp                │
│                                                             │
│  Thread 0: [42] [43] [44]                                  │
│  Thread 1: [45] [46] [47]                                  │
│  Thread 2: [48] [49] [50]  → 1 transaction continue!       │
└────────────────────────────────────────────────────────────┘
```

### Contention Atomique - Le Goulot d'Étranglement

```
Build Phase - Collision Handling:
┌────────────────────────────────────────────────────────────┐
│ Parallel insertion (threads N):                            │
│                                                             │
│  Thread A: hash(key_A) = 42                                │
│  Thread B: hash(key_B) = 42  ← COLLISION!                 │
│  Thread C: hash(key_C) = 42  ← COLLISION!                 │
│                                                             │
│  Tous exécutent:                                            │
│    while (atomic_compare_exchange(&entries[idx].occupied)) │
│      idx = (idx + 1) % capacity;                           │
│                                                             │
│  Problème:                                                  │
│    ┌─────────────────────────────────────────────┐         │
│    │ Thread A: opé [42] libre → SUCCESS          │         │
│    │ Thread B: opé [42] occupé → RETRY [43]      │         │
│    │ Thread C: opé [42] occupé → RETRY [43]      │         │
│    │          └─▶ [43] occupé → RETRY [44]       │         │
│    │                                             │         │
│    │ Sérialisation massive! Threads B et C       │         │
│    │ attendent que A libère [42]                 │         │
│    └─────────────────────────────────────────────┘         │
│                                                             │
│  Estimation: Avec load factor 0.7, ~30% des slots vides    │
│  mais distribution non-uniforme → hotspots de contention   │
│                                                             │
│  Speedup net: O(1) lookup mais ~10× plus lent au build     │
└────────────────────────────────────────────────────────────┘
```

### Warp Divergence dans Probing

```
Variable Probe Lengths:
┌────────────────────────────────────────────────────────────┐
│ Warp de 32 threads cherchant 32 keys différentes:          │
│                                                             │
│  Thread  0: [42] ✓ → 1 itération                           │
│  Thread  1: [55] ✓ → 1 itération                           │
│  Thread  2: [78] [79] [80] ✓ → 3 itérations               │
│  Thread  3: [91] [92] ... [101] ✓ → 11 itérations         │
│  ...                                                        │
│                                                             │
│  ┌──────────────────────────────────────────────┐         │
│  │ Itération 1-10: Threads 0-2 terminés         │         │
│  │                Thread 3 encore actif         │         │
│  │                                             │         │
│  │  Thread 0: IDLE (attends Thread 3)          │         │
│  │  Thread 1: IDLE                             │         │
│  │  Thread 2: IDLE                             │         │
│  │  Thread 3: WORKING (préfet hétérogène)      │         │
│  │                                             │         │
│  │ Warp efficiency: ~70% (vs 90% optimal)      │         │
│  └──────────────────────────────────────────────┘         │
│                                                             │
│  Divergence modérée car probing reste séquentiel           │
└────────────────────────────────────────────────────────────┘
```

### Optimisations Shared Memory

```
Per-Block Small Hash Table:
┌────────────────────────────────────────────────────────────┐
│ Chaque bloc GPU a sa propre hash table en shared memory:   │
│                                                             │
│  __shared__ Entry local_hash[1024];  // ~16 KB             │
│                                                             │
│  Stratégie:                                                │
│    1. Essayer lookup dans local_hash                       │
│    2. Si miss → fallback vers global_hash                  │
│                                                             │
│  Avantages:                                                 │
│    - Latence ~100× inférieure (shared vs global)           │
│    - Contention limitée au bloc                            │
│    - Parfait pour spatial locality                         │
│                                                             │
│  Hit rate estimé: ~40-60% pour données corrélées           │
└────────────────────────────────────────────────────────────┘
```

### Synthèse Hash

| Métrique | Impact |
|----------|--------|
| Lookup time | O(1) average |
| Build time | O(n) avec **contention atomique sévère** |
| Memory overhead | +16% (3 arrays vs 1 array) |
| Coalescing | ✓ Excellent (linear probing) |
| Warp efficiency | ~70% (variable probe lengths) |
| **Bottleneck** | Atomic contention au build |

**Verdict**: Hash table excelle en lookup mais souffre au build. À réserver pour workloads "build once, query many".

---

## 3. Merge-Path Set Algebra

### Concept Fondamental

Le merge-path transforme l'opération de recherche binaire O(log n) scattered en une fusion séquentielle O(n+m).

```
Binary Search (actuel) - Scattered Access:
┌────────────────────────────────────────────────────────────┐
│ Thread i: chercher row_key = (y=1234, z=5678)              │
│                                                             │
│  Binary search iterations:                                 │
│    lo=0, hi=5M → mid=2.5M                                  │
│    lo=2.5M, hi=5M → mid=3.75M  ← SAUT                     │
│    lo=2.5M, hi=3.75M → mid=3.125M  ← SAUT                 │
│    lo=3.125M, hi=3.75M → mid=3.437M  ← SAUT               │
│    ...                                                      │
│                                                             │
│  Chaque thread accède à des indices non-contigus            │
│  → 32 threads = 32 cache lines différentes                 │
│  → Cache thrashing                                         │
└────────────────────────────────────────────────────────────┘

Merge-Path - Sequential Access:
┌────────────────────────────────────────────────────────────┐
│ Thread i: fusionner A[a_start:a_end] et B[b_start:b_end]  │
│                                                             │
│  Two-pointer merge:                                        │
│    ia = a_start, ib = b_start                              │
│    while (ia < a_end && ib < b_end) {                      │
│      if (A[ia] < B[ib])                                    │
│        output[A[ia++]]  ← Accès séquentiel dans A         │
│      else                                                  │
│        output[B[ib++]]  ← Accès séquentiel dans B         │
│    }                                                       │
│                                                             │
│  Chaque thread lit A et B séquentiellement                 │
│  → 32 threads dans un warp lisent 32 blocs contigus        │
│  → Coalescing parfait                                      │
└────────────────────────────────────────────────────────────┘
```

### Partitionnement Merge-Path

```
Diagonal Search for Partitioning:
┌────────────────────────────────────────────────────────────┐
│              B[j]                                           │
│         0  1  2  3  4  5                                   │
│       ┌──┬──┬──┬──┬──┬──┐                                 │
│    0  │  │  │  │  │  │  │                                 │
│    1  │  │  │  │  │  │  │                                 │
│    2  │  │  │  │  │  │  │  Diagonale k = i + j            │
│ A[i]3  │  │  │  │■ │■ │  │  ┌─────────────────────────┐  │
│    4  │  │  │■ │■ │■ │  │  │ Thread i cherche où      │  │
│    5  │  │  │  │  │  │  │  │ k croise la diagonale    │  │
│       └──┴──┴──┴──┴──┴──┘  └─────────────────────────┘  │
│                             │                              │
│                             ▼                              │
│                    ┌──────────────┐                       │
│                    │ Partition i:  │                       │
│                    │   A[2:4]      │                       │
│                    │   B[1:3]      │                       │
│                    │   merge loc.  │                       │
│                    └──────────────┘                       │
│                                                             │
│  Avantage: Chaque thread a une partition indépendante      │
│  → Aucune synchronisation entre threads                    │
│  → Occupation maximale des warps                           │
└────────────────────────────────────────────────────────────┘
```

### Cache Efficiency

```
Binary Search - Cache Thrashing:
┌────────────────────────────────────────────────────────────┐
│ Cache Line 0: rows[0:15]    Cache Line 1: rows[16:31]     │
│ Cache Line 2: rows[32:47]   Cache Line 3: rows[48:63]     │
│ ...                                                        │
│ Cache Line 156250: rows[2.5M:2.5M+15] ← mid1              │
│ Cache Line 234375: rows[3.75M:3.75M+15] ← mid2            │
│ Cache Line 195312: rows[3.125M:3.125M+15] ← mid3          │
│                                                             │
│ 1 thread = ~23 itérations = ~23 cache lines différentes    │
│ 32 threads = ~736 cache lines → dépasse L2 cache           │
│                                                             │
│ Cache miss rate: ~60-80%                                   │
└────────────────────────────────────────────────────────────┘

Merge-Path - Cache Friendly:
┌────────────────────────────────────────────────────────────┐
│ Thread i: merge A[1000:1100] et B[800:900]                 │
│                                                             │
│ Cache Line X: A[1000:1015]  Cache Line Y: A[1016:1031]    │
│ Cache Line Z: A[1032:1047]  ...                            │
│                                                             │
│ Lecture séquentielle A[1000], A[1001], A[1002]...         │
│ → 1 cache line dure pour 16 itérations                     │
│ → Prefetcher hardware fonctionne parfaitement              │
│                                                             │
│ Cache hit rate: ~90%+                                      │
└────────────────────────────────────────────────────────────┘
```

### Transactions Mémoire - Calcul

```
Pour 5M rows, intersection mesh A × mesh B:

Binary Search:
┌────────────────────────────────────────────────────────────┐
│ 1. Row mapping:                                            │
│    - Pour chaque row de A (5M): binary search dans B       │
│    - 23 itérations × 5M = 115M comparaisons                │
│    - Chaque comparaison = 1 accès mémoire potentiel         │
│    → ~115M transactions                                    │
│                                                             │
│ 2. Phase 2-5: count, scan, fill, compact                   │
│    - ~50M transactions additionnelles                      │
│                                                             │
│  Total: ~165M transactions                                 │
└────────────────────────────────────────────────────────────┘

Merge-Path:
┌────────────────────────────────────────────────────────────┐
│ 1. Partitionnement:                                        │
│    - Binary search pour trouver diagonale                  │
│    - log(5M) ≈ 23 itérations (une seule fois!)             │
│    → ~23 transactions                                      │
│                                                             │
│ 2. Merge séquentiel:                                       │
│    - Chaque row de A et B lu une seule fois                │
│    - 5M + 5M = 10M accès                                   │
│    - Coalescing: 32 threads/warp = ~312K transactions      │
│    → ~10M transactions                                     │
│                                                             │
│ 3. Phase restantes: identiques                             │
│    - ~50M transactions                                     │
│                                                             │
│  Total: ~60M transactions                                   │
│                                                             │
│  Réduction: (165 - 60) / 165 = **64% moins**              │
└────────────────────────────────────────────────────────────┘
```

### Warp Efficiency

```
Binary Search - Divergence Sévère:
┌────────────────────────────────────────────────────────────┐
│ Warp threads: 32 threads cherchant 32 rows différentes     │
│                                                             │
│  Itération 1:                                              │
│    Tous: mid = 2.5M                                         │
│    Thread 0: target < mid ? OUI → lo = 2.5M                │
│    Thread 1: target < mid ? NON → hi = 2.5M                │
│    Thread 2: target < mid ? OUI → lo = 2.5M                │
│    ...                                                      │
│    ┌─────────────────────────────────┐                    │
│    │ 16 threads OUI, 16 threads NON  │                    │
│    │ → Warp exécute les 2 branches   │                    │
│    │ → Sérialisation!                │                    │
│    └─────────────────────────────────┘                    │
│                                                             │
│  Warp efficiency: ~60%                                     │
└────────────────────────────────────────────────────────────┘

Merge-Path - Divergence Minimale:
┌────────────────────────────────────────────────────────────┐
│ Warp threads: 32 threads avec 32 partitions indépendantes  │
│                                                             │
│  Tous exécutent:                                            │
│    while (ia < a_end && ib < b_end) {                      │
│      if (A[ia] < B[ib])                                    │
│        ia++;                                               │
│      else                                                  │
│        ib++;                                               │
│    }                                                       │
│                                                             │
│  ┌──────────────────────────────────────────┐             │
│  │ Chaque thread fait sa comparaison        │             │
│  │ indépendamment des autres                │             │
│  │ → Aucune synchronisation nécessaire      │             │
│  │ → Tous threads progressent ensemble      │             │
│  │                                          │             │
│  │ Warp efficiency: ~90%+                   │             │
│  └──────────────────────────────────────────┘             │
└────────────────────────────────────────────────────────────┘
```

### Synthèse Merge-Path

| Métrique | Binary Search | Merge-Path | Amélioration |
|----------|---------------|------------|--------------|
| Row mapping | O(n log m) scattered | O(n+m) sequential | **~11×** |
| Cache hit rate | ~40% | ~90% | **+125%** |
| Warp efficiency | ~60% | ~90%+ | **+50%** |
| Memory transactions | ~165M | ~60M | **64% moins** |
| Set operation time | Baseline | **4-5× faster** | |

**Verdict**: Merge-Path est la stratégie la plus GPU-friendly pour les opérations d'intersection. Il élimine le scattered access du binary search au profit d'un merge séquentiel parfaitement coalescé.

---

## 4. Bitmap Representation

### Concept de Base

Pour un domaine borné (ex: 4096×4096), on remplace la structure CSR par un bitmap où chaque bit représente une cellule.

```
CSR Representation:
┌────────────────────────────────────────────────────────────┐
│ struct Mesh<3> {                                           │
│   RowKey3D* row_keys;    // 5M rows × 8 bytes = 40 MB     │
│   size_t* row_ptr;       // 5M rows × 8 bytes = 40 MB     │
│   Interval* intervals;   // ~20M intervals × 8 bytes      │
│ };                                                         │
│                                                             │
│ Total: ~200 MB pour 5M cells (sparse)                     │
└────────────────────────────────────────────────────────────┘

Bitmap Representation:
┌────────────────────────────────────────────────────────────┐
│ struct BitmapMesh {                                        │
│   uint64_t* bitmap;  // 4096×4096 bits = 2 MB              │
│   int y_max, z_max;                                        │
│ };                                                         │
│                                                             │
│ Total: 2 MB pour 16.7M cells (dense bound)                │
│                                                             │
│ Réduction: 200 MB → 2 MB = **100× moins**                 │
└────────────────────────────────────────────────────────────┘
```

### Layout Mémoire

```
Bitmap Memory Layout:
┌────────────────────────────────────────────────────────────┐
│                                                             │
│  bitmap[0]:   bits 0-63    → (y=0, z=0)    à (y=0, z=63)  │
│  bitmap[1]:   bits 64-127  → (y=0, z=64)   à (y=0, z=127) │
│  bitmap[2]:   bits 128-191 → (y=0, z=128)  à (y=0, z=191) │
│  ...                                                        │
│  bitmap[64]:  bits 4096-...→ (y=1, z=0)    à (y=1, z=63)  │
│  ...                                                        │
│  bitmap[262144]: last word → (y=4095, z=4032)             │
│                                                             │
│  Index computation:                                         │
│    word_idx = y * Z_MAX / 64 + z / 64                      │
│    bit_offset = z % 64                                      │
│                                                             │
│  Coalescing parfait: threads adjacents lisent des bits     │
│  adjacents dans le même word                                │
└────────────────────────────────────────────────────────────┘
```

### Opérations Bitwise - Vectorization

```
Set Intersection - Bitwise AND:
┌────────────────────────────────────────────────────────────┐
│ CSR:                                                       │
│   for each row in A:                                      │
│     binary search row in B  ← scattered                   │
│     merge intervals                                       │
│   → 50 ms pour 5M rows                                   │
│                                                             │
│ Bitmap:                                                    │
│   // 1 kernel, 1 instruction par 64 rows!                  │
│   parallel_for(i, num_words) {                             │
│     result[i] = A[i] & B[i];  ← 64 rows simultanées        │
│   }                                                        │
│   → ~0.5 ms pour 16.7M cells                              │
│                                                             │
│  ┌──────────────────────────────────────────────┐         │
│  │ Vectorized: 1 thread traite 64 rows           │         │
│  │ Warp de 32 threads = 2048 rows/instruction   │         │
│  │ 262144 words → 8192 warp instructions         │         │
│  │                                               │         │
│  │ GPU parallelism: ~1000× plus de parallelisme │         │
│  └──────────────────────────────────────────────┘         │
└────────────────────────────────────────────────────────────┘

Set Union - Bitwise OR:
┌────────────────────────────────────────────────────────────┐
│   parallel_for(i, num_words) {                             │
│     result[i] = A[i] | B[i];  ← Union instantanée          │
│   }                                                        │
└────────────────────────────────────────────────────────────┘

Cell Count - Population Count:
┌────────────────────────────────────────────────────────────┐
│ CSR:                                                       │
│   sum += row.cell_count;  ← scatter-gather               │
│   → Reduction O(n)                                      │
│                                                             │
│ Bitmap:                                                    │
│   parallel_reduce(i, num_words) {                          │
│     count += __popcll(A[i]);  ← 1 instruction GPU          │
│   }                                                        │
│   → Réduction vectorized                                   │
│                                                             │
│  ┌──────────────────────────────────────────────┐         │
│  │ __popcll() = instruction GPU native          │         │
│  │ compte les bits à 1 en ~1 cycle              │         │
│  │                                               │         │
│  │ 262144 words → 262144 cycles (~0.1 ms)       │         │
│  └──────────────────────────────────────────────┘         │
└────────────────────────────────────────────────────────────┘
```

### Bandwidth Utilization

```
Traffic Mémoire - CSR vs Bitmap:

CSR Intersection (5M rows):
┌────────────────────────────────────────────────────────────┐
│ Phase 1 - Row mapping:                                     │
│   A: 40 MB read                                            │
│   B: 40 MB read (scattered, multiple passes)               │
│                                                             │
│ Phase 2-5 - Interval ops:                                  │
│   Intervals: ~160 MB read/write                            │
│   Row ptr: ~80 MB read/write                               │
│                                                             │
│ Total traffic: ~320 MB                                     │
│ Temps @ 500 GB/s: ~0.64 ms                                 │
│ Temps réel: ~50 ms (overhead algorithmique)               │
└────────────────────────────────────────────────────────────┘

Bitmap Intersection (16.7M cells):
┌────────────────────────────────────────────────────────────┐
│   A: 2 MB read                                             │
│   B: 2 MB read                                             │
│   Result: 2 MB write                                       │
│                                                             │
│ Total traffic: 6 MB                                        │
│ Temps @ 500 GB/s: ~0.012 ms                               │
│ Temps réel: ~0.5 ms (kernel overhead)                      │
│                                                             │
│  ┌────────────────────────────────────────────┐           │
│  │ Bandwidth utilization: 6 MB / 320 MB       │           │
│  │ = 1.9% du traffic CSR                      │           │
│  │                                             │           │
│  │ Speedup: 50 ms / 0.5 ms = 100×            │           │
│  └────────────────────────────────────────────┘           │
└────────────────────────────────────────────────────────────┘
```

### Tensor Core Utilization

```
Modern GPUs (A100/H100) - Tensor Cores:
┌────────────────────────────────────────────────────────────┐
│ Tensor cores optimisés pour:                               │
│   - Matrix multiplication (FP16/FP8)                       │
│   - Bitwise operations sur matrices                        │
│                                                             │
│ Bitmap intersection:                                        │
│   view A[i] & B[i] comme 1×64 matrix operation             │
│                                                             │
│  ┌──────────────────────────────────────────┐             │
│  │ A100: 312 TFLOPS tensor core             │             │
│  │ Bitwise ops: ~512 TOPS                   │             │
│  │                                           │             │
│  │ 262144 word operations × 64 ops/word     │             │
│  │ → ~16M operations @ 512 TOPS             │             │
│  │ → ~0.03 ms théorique                     │             │
│  └──────────────────────────────────────────┘             │
│                                                             │
│ Pratique: ~0.5 ms (kernel launch overhead dominant)        │
└────────────────────────────────────────────────────────────┘
```

### Limitations

```
Quand Bitmap EST PERFORMANT:
┌────────────────────────────────────────────────────────────┐
│ ✓ Domain borné (ex: 0-4096)                                │
│ ✓ Densité > 10% (sinon CSR plus compact)                  │
│ ✓ Set operations dominantes                                │
│ ✓ Mémoire limitée                                         │
│                                                             │
│ Exemple optimal:                                           │
│   Domain 4096×4096 = 16.7M cells                          │
│   5M cells actifs = 30% densité                           │
│   → Bitmap = 2 MB                                         │
│   → CSR = ~200 MB                                         │
│   → 100× speedup                                          │
└────────────────────────────────────────────────────────────┘

Quand Bitmap EST MOINS PERFORMANT:
┌────────────────────────────────────────────────────────────┐
│ ✗ Domain non-borné (coordonnées arbitraires)              │
│ ✗ Ultra-sparse (< 1% densité)                             │
│ ✗ Stencil operations (pas de voisinage direct)            │
│ ✗ AMR (résolution variable)                               │
│                                                             │
│ Exemple suboptimal:                                        │
│   Domain 1M×1M = 1T cells                                 │
│   5K cells actifs = 0.0005% densité                       │
│   → Bitmap = 16 GB (impraticable)                         │
│   → CSR = ~200 KB                                          │
│   → CSR 10000× plus compact                               │
└────────────────────────────────────────────────────────────┘
```

### Synthèse Bitmap

| Métrique | CSR | Bitmap | Ratio |
|----------|-----|--------|-------|
| Memory footprint | 200 MB | 2 MB | **100× moins** |
| Intersection time | 50 ms | 0.5 ms | **100× faster** |
| Union time | 50 ms | 0.5 ms | **100× faster** |
| Cell count | 5 ms | 0.1 ms | **50× faster** |
| Memory traffic | 320 MB | 6 MB | **53× moins** |
| Warp efficiency | 60% | 99% | **+65%** |

**Verdict**: Pour les domaines denses bornés, le bitmap est imbattable : 100× speedup avec 100× moins de mémoire. Limité aux domaines bornés (< ~1M cells par dimension).

---

## 5. Tiled Memory Layout

### Concept de Base

Le tiled layout regroupe les cellules en tiles de taille fixe (ex: 16×16) dans le plan YZ pour améliorer la localité spatiale.

```
CSR Layout (row-major by y, then z):
┌────────────────────────────────────────────────────────────┐
│ Row keys: (y,z) → intervals                                │
│                                                             │
│  (0,0): [0,100)     (0,1): [50,150)    (0,2): [0,80)     │
│  (1,0): [200,300)   (1,1): [0,50)      (1,2): [100,200)  │
│  ...                                                       │
│                                                             │
│  Problème: (0,1) et (1,0) sont voisins 3D mais séparés     │
│  dans le tableau row_keys → cache misses                   │
└────────────────────────────────────────────────────────────┘

Tiled Layout (16×16 tiles):
┌────────────────────────────────────────────────────────────┐
│ Tile (0,0): y∈[0,16), z∈[0,16)                             │
│   → Toutes les intervals pour ce bloc stockées ensemble    │
│                                                             │
│ Tile (0,1): y∈[0,16), z∈[16,32)                            │
│ Tile (1,0): y∈[16,32), z∈[0,16)                            │
│ ...                                                        │
│                                                             │
│ Avantage: Cellules voisines dans le même tile = contiguës  │
└────────────────────────────────────────────────────────────┘
```

### Structure de Données

```
Tiled Mesh Structure:
┌────────────────────────────────────────────────────────────┐
│ struct TiledMesh3D {                                       │
│   // Tile metadata (sorted by tile key)                    │
│   TileMeta* tiles;          // T tiles                     │
│   TileKey* tile_keys;        // (ty, tz) per tile          │
│                                                             │
│   // Interval storage (grouped by tile)                    │
│   Interval* intervals;        // Contiguous per tile       │
│                                                             │
│   int tile_size_y = 16;                                    │
│   int tile_size_z = 16;                                    │
│ };                                                         │
│                                                             │
│ struct TileMeta {                                          │
│   TileKey key;              // (ty, tz)                    │
│   size_t offset;           // Into intervals array        │
│   size_t count;            // Number of intervals         │
│   size_t cells;            // Total cells in tile         │
│ };                                                         │
└────────────────────────────────────────────────────────────┘
```

### Shared Memory Utilization

```
Tile Processing with Shared Memory:
┌────────────────────────────────────────────────────────────┐
│ GPU Block (16×16 threads):                                 │
│                                                             │
│  ┌────────────────────────────────────────────┐           │
│  │ __shared__ Interval tile_intervals[1024];   │           │
│  │                                            │           │
│  │ // 1. Cooperatively load tile              │           │
│  │ int tile_idx = blockIdx.x;                 │           │
│  │ int tid = threadIdx.y * 16 + threadIdx.x;  │           │
│  │                                            │           │
│  │ // Each thread loads 1-2 intervals         │           │
│  │ if (tid < tile.count) {                    │           │
│  │   tile_intervals[tid] =                    │           │
│  │     intervals[tile.offset + tid];          │           │
│  │ }                                          │           │
│  │ __syncthreads();                           │           │
│  │                                            │           │
│  │ // 2. All operations in shared memory      │           │
│  │ // (100× faster than global)               │           │
│  │ for (int iv = 0; iv < tile.count; ++iv) {  │           │
│  │   auto& interval = tile_intervals[iv];     │           │
│  │   // Process interval...                   │           │
│  │ }                                          │           │
│  └────────────────────────────────────────────┘           │
│                                                             │
│ Avantages:                                                 │
│   - 1 transaction globale pour charger le tile             │
│   - Opérations dans shared memory (~100× plus rapide)      │
│   - Aucune contention entre blocs                          │
└────────────────────────────────────────────────────────────┘
```

### Coalesced Global Memory Access

```
16×16 Tile - Warp Access Pattern:
┌────────────────────────────────────────────────────────────┐
│ Warp de 32 threads traite 2 rangées Z:                     │
│                                                             │
│  Thread 0-15:  (x, y=0..15, z=0)  ─┐                       │
│  Thread 16-31: (x, y=0..15, z=1)  ─┤ 1 transaction        │
│                                   ─┘ coalesced            │
│                                                             │
│  ┌──────────────────────────────────────────────┐         │
│  │ Threads adjacents = mots adjacents           │         │
│  │ → 1 transaction de 128 bytes pour 32 threads │         │
│  │                                               │         │
│  │ Avec 16×16 tile:                             │         │
│  │   256 threads = 8 warps                      │         │
│  │   8 transactions de 128 bytes = 1 KB total    │         │
│  └──────────────────────────────────────────────┘         │
│                                                             │
│ CSR comparison:                                            │
│   - Threads dispersés sur rows arbitraires                 │
│   - ~32 transactions pour le même warp                    │
│   - 32× plus de traffic                                    │
└────────────────────────────────────────────────────────────┘
```

### Stencil Operations - Le Cas d'Usage Optimal

```
7-Point 3D Stencil:

CSR - Scattered Access:
┌────────────────────────────────────────────────────────────┐
│ Thread i: compute(x, y, z)                                 │
│                                                             │
│   value =                                                 │
│     + 6 * f(x, y, z)      ← Current cell                  │
│     + 1 * f(x-1, y, z)    ← Row (y, z)                    │
│     + 1 * f(x+1, y, z)    ← Même row (OK)                 │
│     + 1 * f(x, y-1, z)    ← Row (y-1, z)   ← Scattered!  │
│     + 1 * f(x, y+1, z)    ← Row (y+1, z)   ← Scattered!  │
│     + 1 * f(x, y, z-1)    ← Row (y, z-1)   ← Scattered!  │
│     + 1 * f(x, y, z+1)    ← Row (y, z+1)   ← Scattered!  │
│                                                             │
│  → 6 row lookups potentiellement différentes              │
│  → 6 cache misses                                          │
│  → Cache hit rate: ~40%                                    │
└────────────────────────────────────────────────────────────┘

Tiled - Contiguous Access:
┌────────────────────────────────────────────────────────────┐
│ Thread i: compute(x, y, z) dans Tile(0,0)                  │
│                                                             │
│   // Intérieur du tile (pas de boundary checks)            │
│   if (y > 0 && y < 15 && z > 0 && z < 15) {               │
│     value =                                                │
│       + 6 * f(x, y, z)      ← Dans tile                   │
│       + 1 * f(x±1, y, z)    ← Dans tile                   │
│       + 1 * f(x, y±1, z)    ← Dans tile                   │
│       + 1 * f(x, y, z±1)    ← Dans tile                   │
│                                                             │
│     // Tous les voisins dans le même tile!                 │
│     // → 0-1 lookup tile                                   │
│     // → Données déjà en shared memory                     │
│   }                                                        │
│                                                             │
│  → 0-1 tile lookup                                         │
│  → Cache hit rate: ~80%                                    │
└────────────────────────────────────────────────────────────┘

Boundary Tiles - Halo Loading:
┌────────────────────────────────────────────────────────────┐
│ Tiles frontières nécessitent un halo:                      │
│                                                             │
│  ┌──────────────────────────────────────────┐             │
│  │ Tile (0,0)     │ Tile (0,1)              │             │
│  │ ┌───────────┐ │ ┌───────────┐            │             │
│  │ │ Interior  │ │ │ Interior  │            │             │
│  │ │ 0 lookup  │ │ │ 0 lookup  │            │             │
│  │ ├───────────┤ │ ├───────────┤            │             │
│  │ │ Boundary  │ │ │ Boundary  │            │             │
│  │ │ 1 lookup  │─┼─│ 1 lookup  │            │             │
│  │ │ (halo)    │ │ │ (halo)    │            │             │
│  │ └───────────┘ │ └───────────┘            │             │
│  └──────────────────────────────────────────┘             │
│                                                             │
│  → Chargement halo: +15% de overhead                       │
│  → Négociable pour le gain sur l'intérieur                │
└────────────────────────────────────────────────────────────┘
```

### Warp Efficiency

```
Divergence Analysis:

CSR:
┌────────────────────────────────────────────────────────────┐
│ Warp threads: 32 rows différentes                         │
│                                                             │
│  Thread 0: row (0, 0) found in 23 comparisons             │
│  Thread 1: row (0, 1) found in 20 comparisons             │
│  Thread 2: row (1, 0) found in 24 comparisons             │
│  ...                                                      │
│                                                             │
│  ┌────────────────────────────────────┐                  │
│  │ Binary search iterations variable  │                  │
│  │ → Threads terminent à des temps    │                  │
│  │   différents                        │                  │
│  │                                    │                  │
│  │ Warp efficiency: ~60%              │                  │
│  └────────────────────────────────────┘                  │
└────────────────────────────────────────────────────────────┘

Tiled:
┌────────────────────────────────────────────────────────────┐
│ Warp threads: 32 cells dans 1-2 rangées du même tile      │
│                                                             │
│  Tous les threads:                                         │
│    - Trouvent leur tile en 1 bsearch (tile count << rows)  │
│    - Accèdent aux intervals contiguës                     │
│    - Progressent ensemble                                 │
│                                                             │
│  ┌────────────────────────────────────┐                  │
│  │ Travail uniforme dans le tile      │                  │
│  │ → Threads synchronisés             │                  │
│  │                                    │                  │
│  │ Warp efficiency: ~85%              │                  │
│  └────────────────────────────────────┘                  │
│                                                             │
│  Exception: Tiles frontières avec halo (minoritaire)      │
└────────────────────────────────────────────────────────────┘
```

### Performance par Backend

```
Estimated Performance:

Serial CPU:
┌────────────────────────────────────────────────────────────┐
│ 5M rows, 16×16 tiles (20K tiles)                           │
│   Stencil operation:                                       │
│     CSR: ~100 ms                                           │
│     Tiled: ~60 ms (1.7×)                                   │
│                                                             │
│   Set operations:                                          │
│     CSR: ~50 ms                                            │
│     Tiled: ~40 ms (1.25×)                                  │
└────────────────────────────────────────────────────────────┘

OpenMP CPU:
┌────────────────────────────────────────────────────────────┐
│ 16 threads                                                 │
│   Stencil:                                                 │
│     CSR: ~6 ms/thread                                     │
│     Tiled: ~3.75 ms/thread (1.6×)                         │
│   → Near-linear scaling (tiles indépendants)              │
└────────────────────────────────────────────────────────────┘

CUDA GPU:
┌────────────────────────────────────────────────────────────┐
│ 5M rows, 16×16 tiles                                        │
│   Stencil:                                                 │
│     CSR: ~20 ms (scattered access)                        │
│     Tiled: ~8 ms (coalesced, shared memory)               │
│                                                             │
│   Breakdown:                                               │
│     - Coalescing: 1.5×                                     │
│     - Shared memory: 1.5×                                  │
│     - Better locality: 1.1×                               │
│     Total: 2.5×                                            │
│                                                             │
│   Set operations:                                          │
│     CSR: ~50 ms                                            │
│     Tiled: ~40 ms (1.25×)                                  │
└────────────────────────────────────────────────────────────┘
```

### Synthèse Tiled

| Métrique | CSR | Tiled | Amélioration |
|----------|-----|-------|--------------|
| Metadata overhead | 16×R | 8×T + 32×T | ~1% (T ≈ R/256) |
| Cache hit rate (stencil) | ~40% | ~80% | **+100%** |
| Warp efficiency | ~60% | ~85% | **+40%** |
| Stencil GPU | 20 ms | 8 ms | **2.5×** |
| Shared memory usage | None | 16-64 KB/block | **Optimal** |

**Verdict**: Tiled layout excelle pour les opérations de stencil grâce à la shared memory. Gain modéré pour les set operations. Break-even à ~10K rows.

---

## 6. Octree Hierarchical

### Concept de Base

L'octree organise les cellules dans une structure hiérarchique où chaque nœud représente un volume qui peut être subdivisé en 8 enfants (octants).

```
Octree Structure (2D simplifié pour illustration):
┌────────────────────────────────────────────────────────────┐
│ Level 0: Root node (entier domain)                        │
│                                                             │
│  ┌─────────────────────────────────────┐                  │
│  │                                     │                  │
│  │         Root (0,0,1024,1024)        │                  │
│  │                                     │                  │
│  └─────────────────────────────────────┘                  │
│                   │ subdivided                               │
│         ┌─────────┼─────────┐                              │
│         ▼         ▼         ▼                              │
│     ┌───────┐ ┌───────┐ ┌───────┐                          │
│     │ (0,0, │ │ (512, │ │       │                          │
│     │ 512,  │ │ 512,  │ │  ...  │  Level 1: 4 children    │
│     │ 512)  │ │ 256)  │ │       │  (8 en 3D)             │
│     └───────┘ └───────┘ └───────┘                          │
│         │ subdivided                                         │
│         └─▶ ...                                             │
│                                                             │
│ Chaque nœud feuille stocke les intervals X                 │
└────────────────────────────────────────────────────────────┘
```

### Structure Mémoire

```
Octree Node Layout:
┌────────────────────────────────────────────────────────────┐
│ struct Node {                                              │
│   uint64_t morton_code;    // Position du nœud            │
│   int8_t level;            // Profondeur dans l'arbre      │
│   uint8_t child_mask;      // Quels enfants existent       │
│   uint32_t first_child;    // Index du premier enfant     │
│   uint32_t parent_idx;     // Index du parent             │
│                                                             │
│   // Leaf data                                            │
│   size_t interval_offset;  // Offset dans intervals array  │
│   size_t interval_count;   // Nombre d'intervals          │
│ };                                                         │
│                                                             │
│ 8 nodes × 32 bytes = 256 bytes par niveau d'arbre         │
│                                                             │
│ Pour 5M leaf nodes:                                        │
│   - Nœuds internes: ~0.14 × 5M = 700K                     │
│   - Total nodes: ~5.7M                                     │
│   - Memory: 5.7M × 32 = 182 MB                             │
│   → +56% vs CSR (117 MB)                                   │
└────────────────────────────────────────────────────────────┘
```

### Pattern d'Accès - Pointer Chasing

```
Tree Traversal - Serial par Nature:
┌────────────────────────────────────────────────────────────┐
│ Thread: find_node(x, y, z)                                 │
│                                                             │
│   morton = encode_morton(x, y, z)  // 1 cycle              │
│   node_idx = 0  // Start at root                           │
│                                                             │
│   while (true) {                                           │
│     node = nodes[node_idx]    ─┐                            │
│                              │  ← Load depuis global mem  │
│     if (node.is_leaf())       │    latence ~100-300 cycles │
│       return node;            │                            │
│                              │                            │
│     child_bits = extract(morton, node.level)              │
│     if (!node.has_child(child_bits))                       │
│       return node;  // Pas d'enfant à ce niveau           │
│                                                             │
│     // Chasing pointer vers l'enfant                       │
│     node_idx = node.first_child +                         │
│                count_children_before(node, child_bits);   │
│   }                                                         │
│                                                             │
│  ┌──────────────────────────────────────────────┐         │
│  │ Chaque itération dépend de la précédente     │         │
│  │ → Pas de parallélisation possible            │         │
│  │ → 1 thread = 1 recherche à la fois           │         │
│  │                                               │         │
│  │ Pour un arbre équilibré (depth ~7):         │         │
│  │   - 7 indirections memory                   │         │
│  │   - 700-2100 cycles de latence              │         │
│  │                                               │         │
│  │ CSR binary search (23 itérations):          │         │
│  │   - 23 accès mémoire (peuvent prefetch)     │         │
│  │   - ~500 cycles                             │         │
│  └──────────────────────────────────────────────┘         │
│                                                             │
│  → Octree **plus lent** pour lookup simple!               │
└────────────────────────────────────────────────────────────┘
```

### Indirection Visualisée

```
Memory Access Pattern - CSR vs Octree:

CSR (Binary Search):
┌────────────────────────────────────────────────────────────┐
│ row_keys array:                                            │
│   [0] [1] [2] ... [2.5M] ... [3.75M] ... [5M]              │
│      │       │        │            │           │            │
│      └───────┴────────┴────────────┴───────────┘           │
│         Scattered mais PRÉDICTIBLE                          │
│         → Hardware prefetcher efficace                     │
└────────────────────────────────────────────────────────────┘

Octree (Tree Traversal):
┌────────────────────────────────────────────────────────────┐
│ nodes array:                                               │
│                                                             │
│   [0]: root                                              ─┐
│         │ first_child = 1                                │
│         └─────────────────────────────────────────────┐   │
│                                                      │   │
│   [1]: child 0                                       │   │
│         │ first_child = 8                            │   │
│         └───────────────────────────────────────┐    │   │
│                                                │    │   │
│   [8]: child 0 of child 0                     │    │   │
│         │ first_child = 15                    │    │   │
│         └─────────────────────────────┐       │    │   │
│                                         │       │    │   │
│   [15]: child 0 of child 0 of child 0  │       │    │   │
│         ...                             │       │    │   │
│                                          ▼       ▼    ▼   │
│                                    Chaîne de 7+ indirections │
│                                    (chaque accès dépend     │
│                                     du précédent)           │
│                                                             │
│  → IMPOSSIBLE à prefetch (dépendance data stricte)         │
│  → Chaque accès = cache miss potentiel                     │
└────────────────────────────────────────────────────────────┘
```

### Warp Divergence Sévère

```
Divergence Analysis:

Binary Search CSR:
┌────────────────────────────────────────────────────────────┐
│ 32 threads cherchent 32 rows différentes                   │
│                                                             │
│  Itération 1: tous lisent mid = 2.5M                       │
│    16 threads: target < mid (gauche)                       │
│    16 threads: target > mid (droite)                       │
│    → Warp divergence (2 paths)                             │
│                                                             │
│  Mais: tous lisent le même mid!                            │
│  → 1 transaction shared pour le warp                       │
│                                                             │
│  Warp efficiency: ~60%                                     │
└────────────────────────────────────────────────────────────┘

Octree Traversal:
┌────────────────────────────────────────────────────────────┐
│ 32 threads cherchent 32 positions différentes              │
│                                                             │
│  Itération 1: tous lisent root (même nœud)                 │
│    → Tous partent de root (sync)                           │
│                                                             │
│  Itération 2:                                               │
│    Thread 0: child 0 (depth 2)                             │
│    Thread 1: child 1 (depth 2)                             │
│    Thread 2: child 0 (depth 2)                             │
│    Thread 3: child 7 (depth 2)                             │
│    ...                                                      │
│    → 32 nodes différents!                                  │
│    → 32 transactions scattered!                            │
│                                                             │
│  Itération 3+:                                              │
│    Threads à depth 3, 4, 5...  mélés                       │
│    → Certains déjà leaf, d'autres depth 7                  │
│    → 32 paths différents!                                  │
│                                                             │
│  ┌────────────────────────────────────┐                   │
│  │ Warp exécute 32 chemins séquentiel │                   │
│  │ → Pire cas: 32× sérialisation      │                   │
│  │                                     │                   │
│  │ Warp efficiency: ~10-20%            │                   │
│  └────────────────────────────────────┘                   │
│                                                             │
│  → GPU massivement sous-utilisé                            │
└────────────────────────────────────────────────────────────┘
```

### Batch Processing - Tentative d'Optimisation

```
Parallel Tree Traversal (Batch):
┌────────────────────────────────────────────────────────────┐
│ Kokkos::parallel_for(N, KOKKOS_LAMBDA(int i) {             │
│   // Chaque thread cherche sa position                     │
│   morton = targets[i];                                     │
│   node = find_node_serial(morton);  // Toujours serial!    │
│ });                                                        │
│                                                             │
│  ┌────────────────────────────────────────┐               │
│  │ Meilleure occupation (N threads actifs)│               │
│  │ Mais chaque thread fait:               │               │
│  │   - 7+ indirections                    │               │
│  │   - 700-2100 cycles                    │               │
│  │                                         │               │
│  │ Total: N × 2000 cycles                 │               │
│  └────────────────────────────────────────┘               │
│                                                             │
│ CSR Binary Search:                                         │
│   N threads × 500 cycles = N × 500 cycles                  │
│                                                             │
│  → Octree **4× plus lent** même en batch!                  │
└────────────────────────────────────────────────────────────┘
```

### AMR Operations - Le Seul Cas d'Usage Valide

```
Refinement Operation - CSR vs Octree:

CSR (rebuild complet):
┌────────────────────────────────────────────────────────────┐
│ Refine 10% of cells (500K cells):                          │
│                                                             │
│   1. Créer nouveau mesh                                    │
│   2. Copier cells non-refined (4.5M)                       │
│   3. Subdiviser cells refined (500K → 4M)                  │
│   4. Rebuild row_keys, row_ptr                             │
│   5. Sort par (y, z)                                        │
│                                                             │
│  → O(R log R) où R = 5M → 4M = 9M rows                     │
│  → ~150 ms sur GPU                                         │
└────────────────────────────────────────────────────────────┘

Octree (refinement local):
┌────────────────────────────────────────────────────────────┐
│ Refine 10% of leaf nodes (500K nodes):                     │
│                                                             │
│   for each leaf_to_refine:                                 │
│     1. Allocate 8 children nodes                           │
│     2. Update parent.child_mask, first_child              │
│     3. Distribute parent intervals to children             │
│                                                             │
│  → O(0.1 × R) = O(500K)                                   │
│  → ~20 ms sur GPU                                          │
│                                                             │
│  ┌────────────────────────────────────────┐               │
│  │ Speedup: 150 ms / 20 ms = 7.5×        │               │
│  │                                         │               │
│  │ Mais:                                   │               │
│  │   - Memory overhead: +56%               │               │
│  │   - Lookup plus lent pour static data  │               │
│  │   - AMR uniquement                      │               │
│  └────────────────────────────────────────┘               │
└────────────────────────────────────────────────────────────┘
```

### Synthèse Octree

| Métrique | CSR | Octree | Note |
|----------|-----|--------|------|
| Memory footprint | 117 MB | 182 MB | **+56%** |
| Point lookup | ~500 cycles | ~2000 cycles | **4× slower** |
| Warp efficiency | ~60% | ~10-20% | **-70%** |
| Refinement | O(R log R) | O(k) | **7× faster** |
| GPU friendly | Oui | **Non** | Serial traversal |

**Verdict**: L'octree est GPU-hostile pour les opérations classiques. Le seul cas d'usage valide est AMR avec frequent refinement, où le gain de O(1) refinement compense la lenteur des lookups.

---

## 7. Hybrid Adaptive

### Concept de Base

La stratégie hybride sélectionne automatiquement la représentation optimale en fonction des caractéristiques du mesh à l'exécution.

```
Decision Flow:
┌────────────────────────────────────────────────────────────┐
│                                                             │
│   Input Mesh                                               │
│      │                                                     │
│      ▼                                                     │
│   ┌─────────────────┐                                      │
│   │ Compute Metrics │◀─ O(R) scan, ~0.1-0.5 ms           │
│   └────────┬────────┘                                      │
│            │                                               │
│            ▼                                               │
│   ┌─────────────────────┐                                  │
│   │ MeshMetrics:        │                                  │
│   │ - num_rows: 5M      │                                  │
│   │ - domain: 4096×4096 │                                  │
│   │ - density: 30%      │                                  │
│   │ - needs_amr: false  │                                  │
│   └────────┬────────────┘                                  │
│            │                                               │
│            ▼                                               │
│   ┌───────────────────────┐                                │
│   │ StrategySelector::    │                                │
│   │   select(metrics)     │                                │
│   └────────┬──────────────┘                                │
│            │                                               │
│     ┌──────┼──────┬──────┬──────┐                        │
│     ▼      ▼      ▼      ▼      ▼                         │
│  Classic Morton Hash Bitmap Tiled Octree                │
│     │      │      │      │      │                         │
│     └──────┴──────┴──────┴──────┘                         │
│            │                                               │
│            ▼                                               │
│      Optimized Mesh                                        │
│                                                             │
│  Overhead total: ~0.2-1 ms (une seule fois)                │
└────────────────────────────────────────────────────────────┘
```

### Structure Mémoire - Union

```
HybridMesh Memory Layout:
┌────────────────────────────────────────────────────────────┐
│ template <class MemorySpace>                              │
│ class HybridMesh3D {                                      │
│ public:                                                   │
│   Strategy strategy = Strategy::Auto;                     │
│                                                             │
│   union {                                                  │
│     struct {  // Classic CSR                              │
│       RowKey3D* row_keys;                                 │
│       size_t* row_ptr;                                    │
│     } classic;                                            │
│                                                             │
│     struct {  // Morton                                    │
│       uint64_t* morton_codes;                             │
│       size_t* row_ptr;                                    │
│     } morton;                                             │
│                                                             │
│     struct {  // Bitmap                                    │
│       uint64_t* bitmap;                                   │
│       int y_max, z_max;                                   │
│     } bitmap;                                             │
│                                                             │
│     struct {  // Tiled                                     │
│       TileMeta* tiles;                                    │
│       Interval* intervals;                                │
│     } tiled;                                              │
│                                                             │
│     struct {  // Octree                                    │
│       Node* nodes;                                        │
│       Interval* intervals;                                │
│     } octree;                                             │
│   };                                                       │
│                                                             │
│   Interval* intervals;  // Shared (pas dans union)         │
│ };                                                         │
│                                                             │
│  ┌──────────────────────────────────────────┐             │
│  │ Taille union = max(tailles membres)     │             │
│  │ = 25×R (octree)                         │             │
│  │                                          │             │
│  │ MAIS: 1 seul membre actif à la fois      │             │
│  │ → Coût réel: 0 overhead                 │             │
│  └──────────────────────────────────────────┘             │
└────────────────────────────────────────────────────────────┘
```

### Coûts de Conversion

```
Transformation Costs:

CSR → Morton:
┌────────────────────────────────────────────────────────────┐
│ Phase 1: Encoding (parallel)                              │
│   for each row:                                           │
│     morton[i] = encode(row_keys[i].y, row_keys[i].z)     │
│   → O(n), ~0.5-1 ms pour 100K rows                        │
│                                                             │
│ Phase 2: Sort (parallel)                                  │
│   sort_by_morton_code()                                   │
│   → O(n log n), ~0.5-1 ms pour 100K rows                  │
│                                                             │
│  Total: ~1-2 ms pour 100K rows                            │
└────────────────────────────────────────────────────────────┘

CSR → Bitmap:
┌────────────────────────────────────────────────────────────┐
│ Phase 1: Allocation                                        │
│   bitmap = uint64_t[domain_size / 64]                     │
│   → O(1) allocation                                        │
│                                                             │
│ Phase 2: Scatter (parallel with atomics)                  │
│   for each cell:                                          │
│     idx = y * Z_MAX + z                                   │
│     word = idx / 64                                       │
│     bit = idx % 64                                        │
│     atomic_or(&bitmap[word], 1ULL << bit)                │
│   → O(n), ~0.3-1 ms pour 100K rows                        │
│                                                             │
│  Total: ~0.3-1 ms pour 100K rows                          │
└────────────────────────────────────────────────────────────┘

CSR → Tiled:
┌────────────────────────────────────────────────────────────┐
│ Phase 1: Assign rows to tiles (parallel)                  │
│   for each row:                                           │
│     tile_key = {y/16, z/16}                               │
│   → O(n), ~0.2 ms                                         │
│                                                             │
│ Phase 2: Sort by tile key                                  │
│   → O(n log n), ~0.3 ms                                   │
│                                                             │
│ Phase 3: Count tiles and intervals                        │
│   → O(n), ~0.1 ms                                         │
│                                                             │
│ Phase 4: Allocate and fill                                 │
│   → O(n), ~0.3 ms                                         │
│                                                             │
│  Total: ~0.9 ms pour 100K rows                            │
└────────────────────────────────────────────────────────────┘

CSR → Octree:
┌────────────────────────────────────────────────────────────┐
│ Phase 1: Compute Morton codes                              │
│   → O(n), ~0.5 ms                                         │
│                                                             │
│ Phase 2: Build tree hierarchy (multi-pass)                │
│   → O(n log n), ~1-3 ms                                   │
│                                                             │
│ Phase 3: Distribute intervals to leaves                   │
│   → O(n), ~0.5 ms                                         │
│                                                             │
│  Total: ~2-5 ms pour 100K rows                            │
└────────────────────────────────────────────────────────────┘
```

### Break-Even Analysis

```
Cost/Benefit par Nombre d'Opérations:

Pour 100K rows, 1 opération set (50ms CSR baseline):
┌────────────────────────────────────────────────────────────┐
│ Stratégie    │ Conversion │ Opération │ Total │ Gain     │
│--------------│------------│-----------│-------│----------│
│ Classic CSR  │ 0 ms      │ 50 ms     │ 50 ms │ baseline │
│ Morton       │ 2 ms      │ 25 ms     │ 27 ms │ +46%     │
│ Bitmap       │ 1 ms      │ 0.5 ms    │ 1.5 ms│ +97%     │
│ Tiled        │ 1 ms      │ 40 ms     │ 41 ms │ -18%     │
└────────────────────────────────────────────────────────────┘

Pour 100K rows, 50 opérations set:
┌────────────────────────────────────────────────────────────┐
│ Stratégie    │ Conversion │ Opérations│ Total │ Gain     │
│--------------│------------│-----------│-------│----------│
│ Classic CSR  │ 0 ms      │ 2500 ms   │ 2500ms│ baseline │
│ Morton       │ 2 ms      │ 1250 ms   │ 1252ms│ +50%     │
│ Bitmap       │ 1 ms      │ 25 ms     │ 26 ms │ +99%     │
│ Tiled        │ 1 ms      │ 2000 ms   │ 2001ms│ +20%     │
└────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│ Conclusion:                                   │
│   - 1-10 opérations: Conversion non amortie  │
│   - 10+ opérations: Gain significatif       │
│   - 50+ opérations: Bitmap massivement gagnant│
└──────────────────────────────────────────────┘
```

### Selection Logic

```
StrategySelector::select(mesh_metrics):
┌────────────────────────────────────────────────────────────┐
│ if (metrics.needs_amr)                                    │
│   return Strategy::Octree_Morton;                         │
│                                                             │
│ if (metrics.stencil_heavy && metrics.num_rows > 10K)      │
│   return Strategy::Tiled_CSR;                             │
│                                                             │
│ // Domain-based selection                                 │
│ if (metrics.domain_size <= 1M &&                          │
│     metrics.cell_density > 0.5)                           │
│   return Strategy::Bitmap;  // Dense, bounded             │
│                                                             │
│ // Size + density matrix                                  │
│ if (metrics.num_rows < 10K) {                             │
│   if (metrics.row_density > 0.5)                          │
│     return Strategy::Packed_Hash;                        │
│   return Strategy::Classic_Binary;                        │
│ }                                                          │
│                                                             │
│ if (metrics.num_rows >= 1M) {                             │
│   if (metrics.row_density > 0.5) {                        │
│     if (metrics.domain_size <= 1M)                        │
│       return Strategy::Bitmap;                           │
│     return Strategy::Tiled_CSR;                           │
│   }                                                        │
│   if (metrics.set_operation_heavy)                        │
│     return Strategy::Morton_Merge;                        │
│   return Strategy::Classic_Binary;                        │
│ }                                                          │
│                                                             │
│ // Medium meshes (10K - 1M)                              │
│ if (metrics.set_operation_heavy)                          │
│   return Strategy::Morton_Merge;                          │
│ if (metrics.row_density > 0.3)                            │
│   return Strategy::Packed_Hash;                           │
│ return Strategy::Classic_Binary;                          │
└────────────────────────────────────────────────────────────┘
```

### Runtime Overhead

```
Branch Prediction Impact:
┌────────────────────────────────────────────────────────────┐
│ switch (mesh.strategy) {                                  │
│   case Strategy::Bitmap:                                  │
│     result = bitmap_intersect(mesh, other);               │
│     break;                                                │
│   case Strategy::Morton_Merge:                            │
│     result = morton_intersect(mesh, other);               │
│     break;                                                │
│   ...                                                     │
│ }                                                          │
│                                                             │
│  ┌────────────────────────────────────────────┐          │
│  │ Switch dispatch au niveau KERNEL           │          │
│  │ → Pas de divergence intra-warp             │          │
│  │ → Tous threads exécutent le même case      │          │
│  │                                             │          │
│  │ Overhead CPU: ~5-10 cycles (inline)        │          │
│  │ Overhead GPU: négligeable (kernel dispatch)│          │
│  └────────────────────────────────────────────┘          │
│                                                             │
│  Pas de branch prediction penalty!                        │
└────────────────────────────────────────────────────────────┘
```

### Synthèse Hybrid

| Métrique | Impact |
|----------|--------|
| Memory overhead | 0 (union) |
| Conversion cost | 0.3-5 ms (one-time) |
| Metrics computation | 0.1-0.5 ms (one-time) |
| Runtime selection | ~5-10 cycles (inline) |
| Break-even | 10+ operations |
| Optimal gain | 2-100× (context-dependent) |

**Verdict**: L'approche hybride est rentable pour les workloads réels avec de multiples opérations. Le coût de conversion (~1-5 ms) est rapidement amorti.

---

## Synthèse Comparative

### GPU-Friendliness Ranking

```
┌────────────────────────────────────────────────────────────┐
│                    GPU-Friendliness                        │
│                                                             │
│  1. Bitmap        ████████████████████ 100× (dense only)  │
│  2. Merge-Path    ████████████ 4-5× (universal)          │
│  3. Tiled         ████████ 2-3× (stencils)               │
│  4. Morton        ██████ 1.5-2× (lookup)                 │
│  5. Hybrid        ████ 1-100× (context)                  │
│  6. Hash          ███ 1-6× (atomic bottleneck)           │
│  7. Octree        ▬ 0.5× (GPU hostile)                   │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### Memory Transactions Comparison

```
5M Rows Intersection - Estimated Transactions:

┌────────────────────────────────────────────────────────────┐
│                    Binary Search                           │
│  ███████████████████████████████████ 165M transactions    │
│                                                             │
│                    Merge-Path                              │
│  ████████████ 60M transactions (64% less)                 │
│                                                             │
│                    Morton (baseline)                       │
│  ████████████████████ 100M transactions                   │
│                                                             │
│                    Tiled                                   │
│  ██████████████ 50M transactions (70% less)              │
│                                                             │
│                    Bitmap (dense domain)                   │
│  █ 6M transactions (96% less)                            │
└────────────────────────────────────────────────────────────┘
```

### Decision Matrix

```
┌────────────────────────────────────────────────────────────┐
│ Scenario                │ Recommended Strategy            │
├────────────────────────────────────────────────────────────┤
│ < 5K rows               │ Classic CSR (conversion overhead)│
│ 5K-100K sparse         │ Morton + Merge-Path              │
│ > 100K dense bounded   │ Bitmap (100× speedup)            │
│ > 100K sparse          │ Morton + Merge-Path              │
│ Stencil heavy          │ Tiled (2-3× speedup)             │
│ Set operation heavy    │ Merge-Path (4-5× speedup)        │
│ AMR workflows          │ Octree (despite GPU hostility)   │
│ Dynamic insertions     │ Hash table                       │
│ Variable workloads     │ Hybrid Auto                      │
│ Unknown / mixed        │ Hybrid Auto                      │
└────────────────────────────────────────────────────────────┘
```

### Memory Footprint Comparison

```
For 5M rows (~40M cells):

┌────────────────────────────────────────────────────────────┐
│ Strategy          │ Memory      │ vs CSR │ Notes          │
├────────────────────────────────────────────────────────────┤
│ CSR (baseline)    │ 117 MB      │ 1.0×   │                │
│ Morton            │ 117 MB      │ 1.0×   │ Identical      │
│ Hash table        │ 136 MB      │ 1.16×  │ +16% overhead  │
│ Tiled             │ 119 MB      │ 1.02×  │ +2% metadata   │
│ Octree            │ 182 MB      │ 1.56×  │ +56% tree      │
│ Bitmap (4K×4K)    │ 2 MB        │ 0.02×  │ 100× less!     │
│ Bitmap (1M×1M)    │ 128 MB      │ 1.1×   │ Still good     │
└────────────────────────────────────────────────────────────┘
```

### Performance Summary Table

```
All strategies compared to CSR baseline (100%):

┌────────────────────────────────────────────────────────────┐
│ Strategy  │ Lookup │ Set Ops │ Memory │ GPU-friendly │Note│
├────────────────────────────────────────────────────────────┤
│ CSR       │ 100%   │ 100%    │ 100%   │ ★★★☆☆        │Baseline│
│ Morton    │ 60%    │ 70%     │ 100%   │ ★★★★☆        │Best all│
│ Hash      │ 15%    │ 50%     │ 116%   │ ★★☆☆☆        │Atomic │
│ Merge-Path│ 100%   │ 25%     │ 100%   │ ★★★★★        │Set ops│
│ Bitmap    │ 0.5%   │ 1%      │ 2%     │ ★★★★★        │Dense  │
│ Tiled     │ 80%    │ 80%     │ 102%   │ ★★★★☆        │Stencil│
│ Octree    │ 400%   │ 200%    │ 156%   │ ★☆☆☆☆        │AMR    │
│ Hybrid    │ var    │ var     │ 100%   │ ★★★★☆        │Auto   │
└────────────────────────────────────────────────────────────┘

Lower is better (percentage of CSR baseline time)
```

### Implementation Recommendations

```
Phased Approach:

Phase 1 (3 weeks) - Merge-Path
┌────────────────────────────────────────────────────────────┐
│ Priority: HIGHEST                                           │
│ Risk: LOW                                                   │
│ Effort: 3 weeks                                             │
│ Gain: 4-5× on set operations                                │
│                                                             │
│ - Implement merge-path partitioning                         │
│ - Replace binary search in phase 1                          │
│ - Test intersection, union, difference                      │
│ - Benchmark all backends                                    │
└────────────────────────────────────────────────────────────┘

Phase 2 (2 weeks) - Morton Encoding
┌────────────────────────────────────────────────────────────┐
│ Priority: HIGH                                              │
│ Risk: LOW                                                   │
│ Effort: 2 weeks                                             │
│ Gain: 1.5-2× on lookup                                      │
│                                                             │
│ - Add morton_encode/decode functions                        │
│ - Replace RowKey3D with Morton code                         │
│ - Update comparison operators                               │
│ - Benchmark spatial locality                                │
└────────────────────────────────────────────────────────────┘

Phase 3 (3 weeks) - Bitmap (optional)
┌────────────────────────────────────────────────────────────┐
│ Priority: MEDIUM (dense workloads only)                     │
│ Risk: MEDIUM                                                │
│ Effort: 3 weeks                                             │
│ Gain: 100× on dense bounded domains                         │
│                                                             │
│ - Implement bitmap mesh type                                │
│ - Add CSR ↔ bitmap conversion                               │
│ - Bitwise AND/OR operations                                 │
│ - Auto-select for dense domains                            │
└────────────────────────────────────────────────────────────┘

Phase 4 (4 weeks) - Tiled (optional)
┌────────────────────────────────────────────────────────────┐
│ Priority: LOW (stencil workloads only)                      │
│ Risk: MEDIUM                                                │
│ Effort: 4 weeks                                             │
│ Gain: 2-3× on stencil operations                            │
│                                                             │
│ - Implement tile structure                                  │
│ - Add shared memory optimization                            │
│ - Tile-aware stencil kernels                                │
│ - Auto-select for stencil-heavy workloads                  │
└────────────────────────────────────────────────────────────┘
```

### Conclusion

L'analyse des patterns d'accès mémoire GPU révèle des opportunités significatives d'optimisation:

1. **Merge-Path** est la plus universelle : 4-5× speedup avec risque minimal
2. **Morton encoding** améliore la localité sans coût mémoire
3. **Bitmap** est imbattable pour les domaines denses bornés
4. **Tiled** excelle pour les opérations de stencil
5. **Octree** à éviter sauf pour AMR
6. **L'approche hybride** permet une sélection automatique optimale

**Recommandation**: Commencer par Merge-Path (Phase 1), puis ajouter Morton encoding (Phase 2). Les phases 3 et 4 sont optionnelles selon les cas d'usage.
