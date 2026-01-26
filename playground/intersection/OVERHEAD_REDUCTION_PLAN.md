# Overhead Reduction Plan - CUDA Benchmarks

**Date:** 2026-01-26
**Branch:** `reduce-overhead`
**Target:** CUDA backend with pre-allocated buffers

---

## Objectifs

1. **Éliminer les allocations mémoire** dans les benchmarks CUDA
2. **Permettre le benchmark de chaque phase** individuellement
3. **Pré-alllocation basée sur max(A, B)** - l'intersection ne dépasse jamais les entrées

---

## Architecture

### 1. Workspace Pattern

Un `IntersectionWorkspace` contient tous les buffers temporaires nécessaires:

```cpp
template <typename ExecSpace, typename IndexType = std::size_t>
struct IntersectionWorkspace {
    using MemorySpace = typename ExecSpace::memory_space;
    using IntView = Kokkos::View<int*, MemorySpace>;
    using SizeTView = Kokkos::View<IndexType*, MemorySpace>;

    // Phase 1: Row mapping buffers
    IntView flags;          // [max_rows] - marque les rows qui matchent
    IntView tmp_idx_a;      // [max_rows] - indices temporaires pour A
    IntView tmp_idx_b;      // [max_rows] - indices temporaires pour B
    SizeTView positions;    // [max_rows] - positions pour compaction

    // Phase 2: Row scan buffers
    SizeTView num_rows_out_view;   // [1] - scalaire pour réduction
    IntView out_rows;              // [max_rows] - rows compactées
    IntView out_idx_a;             // [max_rows] - indices A compactés
    IntView out_idx_b;             // [max_rows] - indices B compactés

    // Phase 3: Interval counting
    SizeTView row_counts;          // [max_rows] - compte d'intervalles par row

    // Phase 4: Scan buffers
    SizeTView total_view;          // [1] - scalaire pour scan

    // Phase 5: Final compaction
    IntView has_intervals;         // [max_rows] - marque les rows non-vides
    SizeTView new_positions;       // [max_rows] - positions pour re-compaction
    SizeTView final_num_rows_view; // [1] - scalaire final

    std::size_t capacity = 0;

    // Alloue/réalloue si nécessaire
    void ensure_capacity(std::size_t max_rows, std::size_t max_intervals);
};
```

**Taille de pré-allocation:**
- `max_rows = max(A.num_rows, B.num_rows)` - rows ne peuvent pas augmenter
- `max_intervals = max(A.num_intervals, B.num_intervals)` - intersection ≤ entrées

### 2. API In-Place

**Nouvelle fonction qui utilise des buffers pré-alloués:**

```cpp
namespace baseline {
    // API existante (alloue tout)
    Mesh2DDevice intersect_meshes_2d(const Mesh2DDevice& a, const Mesh2DDevice& b);

    // Nouvelle API (utilise buffers pré-alloués)
    void intersect_meshes_2d_in_place(
        const Mesh2DDevice& a,
        const Mesh2DDevice& b,
        Mesh2DDevice& result_out,           // Pré-alloué par l'appelant
        IntersectionWorkspace<ExecSpace>& ws // Workspace réutilisable
    );
}
```

**Allocation du result:**
```cpp
// L'appelant alloue le résultat une seule fois
Mesh2DDevice result;
result.row_keys = Kokkos::View<RowKey*, MemorySpace>("result_keys", max_rows);
result.row_ptr = Kokkos::View<IndexType*, MemorySpace>("result_ptr", max_rows + 1);
result.intervals = Kokkos::View<Interval*, MemorySpace>("result_intervals", max_intervals);

// Réutilisé à chaque itération du benchmark
for (auto _ : state) {
    intersect_meshes_2d_in_place(a, b, result, ws);
    // result est rempli, pas d'allocation
}
```

### 3. Benchmarks par Phase

Chaque phase de l'algorithme peut être benchmarkée individuellement:

```cpp
// Phase 1: Row mapping (binary search + scan)
BENCHMARK_F(IntersectionBenchmark, Phase1_RowMapping) {
    ws.phase1_row_mapping_only(a, b);
}

// Phase 2: Interval counting
BENCHMARK_F(IntersectionBenchmark, Phase2_CountIntervals) {
    // ... compte les intervalles pour chaque row ...
}

// Phase 3: Scan (row_ptr computation)
BENCHMARK_F(IntersectionBenchmark, Phase3_Scan) {
    ws.row_scan(intermediate_rows);
}

// Phase 4: Fill intervals
BENCHMARK_F(IntersectionBenchmark, Phase4_FillIntervals) {
    ws.fill_intervals(a, b, intermediate_result);
}

// Phase 5: Compaction
BENCHMARK_F(IntersectionBenchmark, Phase5_Compaction) {
    ws.compact_final(intermediate_result, final_result);
}
```

---

## Implémentation

### Étape 1: Créer le workspace

**Fichier:** `playground/intersection/include/playground/subsetix/csr/intersection/workspace.hpp`

```cpp
#pragma once
#ifdef SUBSETIX_ENABLE_PLAYGROUND

#include <Kokkos_Core.hpp>
#include <cstddef>

namespace playground::subsetix::csr::intersection {

template <typename ExecSpace, typename IndexType = std::size_t>
struct IntersectionWorkspace {
    using MemorySpace = typename ExecSpace::memory_space;
    using IntView = Kokkos::View<int*, MemorySpace>;
    using SizeTView = Kokkos::View<IndexType*, MemorySpace>;

    // Buffer views (uninitialized until ensure_capacity is called)
    IntView flags;
    IntView tmp_idx_a;
    IntView tmp_idx_b;
    SizeTView positions;
    SizeTView num_rows_out_view;
    IntView out_rows;
    IntView out_idx_a;
    IntView out_idx_b;
    SizeTView row_counts;
    SizeTView total_view;
    IntView has_intervals;
    SizeTView new_positions;
    SizeTView final_num_rows_view;

    std::size_t capacity_rows = 0;
    std::size_t capacity_intervals = 0;

    void ensure_capacity(std::size_t max_rows, std::size_t max_intervals) {
        if (max_rows <= capacity_rows && max_intervals <= capacity_intervals) {
            return;  // Déjà alloué
        }

        // Réallouer avec la nouvelle taille
        std::size_t new_capacity_rows = std::max(max_rows, capacity_rows);
        std::size_t new_capacity_intervals = std::max(max_intervals, capacity_intervals);

        flags = IntView("flags", new_capacity_rows);
        tmp_idx_a = IntView("tmp_idx_a", new_capacity_rows);
        tmp_idx_b = IntView("tmp_idx_b", new_capacity_rows);
        positions = SizeTView("positions", new_capacity_rows);
        out_rows = IntView("out_rows", new_capacity_rows);
        out_idx_a = IntView("out_idx_a", new_capacity_rows);
        out_idx_b = IntView("out_idx_b", new_capacity_rows);
        row_counts = SizeTView("row_counts", new_capacity_rows);
        has_intervals = IntView("has_intervals", new_capacity_rows);
        new_positions = SizeTView("new_positions", new_capacity_rows);

        num_rows_out_view = SizeTView("num_rows_out");
        total_view = SizeTView("total_intervals");
        final_num_rows_view = SizeTView("final_num_rows");

        capacity_rows = new_capacity_rows;
        capacity_intervals = new_capacity_intervals;
    }

    // Réinitialiser entre les utilisations (pas de réallocation)
    void reset() {
        // Les vues peuvent être réutilisées telles quelles
        // Les kernels écraseront les données
    }
};

} // namespace playground::subsetix::csr::intersection

#endif // SUBSETIX_ENABLE_PLAYGROUND
```

### Étape 2: Ajouter l'API in-place

**Fichier:** `playground/intersection/include/playground/subsetix/csr/intersection/algorithm/baseline.hpp`

```cpp
// Ajouter après intersect_meshes<2>()

template <typename ExecSpace = Kokkos::DefaultExecutionSpace>
void intersect_meshes_2d_in_place(
    const Mesh<2, typename ExecSpace::memory_space>& a,
    const Mesh<2, typename ExecSpace::memory_space>& b,
    Mesh<2, typename ExecSpace::memory_space>& result_out,
    IntersectionWorkspace<ExecSpace>& ws
);
```

### Étape 3: Benchmarks avec workspace

**Fichier:** `playground/intersection/benchmarks/intersection/workspace_benchmark.cpp` (NOUVEAU)

```cpp
template <typename GetConfigFunc>
class WorkspaceBenchmark : public benchmark::Fixture {
public:
    using baseline::Mesh2DDevice;
    using baseline::MeshConverter2D;
    using Workspace = IntersectionWorkspace<Kokkos::DefaultExecutionSpace>;

    void SetUp(const ::benchmark::State&) override {
        auto cfg = GetConfigFunc()();

        // Générer les meshes d'entrée (une seule fois)
        auto common_a = RegularMeshGenerator::generate_2d(cfg);
        auto common_b = RegularMeshGenerator::generate_2d(cfg);

        mesh_a_ = MeshConverter2D<baseline::Mesh, ..., int32_t, std::size_t>::from_common(common_a);
        mesh_b_ = MeshConverter2D<baseline::Mesh, ..., int32_t, std::size_t>::from_common(common_b);

        // Calculer la taille maximale nécessaire
        std::size_t max_rows = std::max(mesh_a_.num_rows, mesh_b_.num_rows);
        std::size_t max_intervals = std::max(mesh_a_.num_intervals, mesh_b_.num_intervals);

        // Allouer le workspace
        workspace_.ensure_capacity(max_rows, max_intervals);

        // Allouer le résultat (une seule fois!)
        result_.row_keys = Kokkos::View<RowKey*, ...>("result_keys", max_rows);
        result_.row_ptr = Kokkos::View<std::size_t*, ...>("result_ptr", max_rows + 1);
        result_.intervals = Kokkos::View<Interval*, ...>("result_intervals", max_intervals);
    }

    void TearDown(const ::benchmark::State&) override {
        workspace_.reset();
    }

protected:
    Mesh2DDevice mesh_a_, mesh_b_;
    Mesh2DDevice result_;
    Workspace workspace_;
};
```

---

## Fichiers à modifier/créer

| Fichier | Action | Description |
|---------|--------|-------------|
| `workspace.hpp` | CRÉER | Définition du IntersectionWorkspace |
| `baseline.hpp` | MODIFIER | Ajouter `intersect_meshes_2d_in_place()` |
| `baseline_impl.hpp` | MODIFIER | Implémenter la version in-place |
| `optimized.hpp` | MODIFIER | Ajouter version in-place (même chose) |
| `workspace_benchmark.cpp` | CRÉER | Benchmarks avec workspace |
| `phase_benchmark.cpp` | CRÉER | Benchmarks individuels par phase |
| `CMakeLists.txt` | MODIFIER | Ajouter nouveaux benchmarks |

---

## Ordre d'implémentation

1. ✅ Créer `workspace.hpp` avec la classe IntersectionWorkspace
2. ⬜ Ajouter la déclaration `intersect_meshes_2d_in_place` dans `baseline.hpp`
3. ⬜ Implémenter `intersect_meshes_2d_in_place` en utilisant le workspace
4. ⬜ Créer `workspace_benchmark.cpp` pour tester
5. ⬜ Créer `phase_benchmark.cpp` pour les micro-benchmarks
6. ⬜ Mettre à jour `CMakeLists.txt`
7. ⬜ Tester et vérifier la correction

---

## Résultats attendus

| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| Small mesh (19 rows) | ~500μs | ~50-100μs | **80-90%** |
| Allocations par itération | 33-40 | 0 (pré-alloué) | **100%** |
| Temps réel d'algorithme | caché | visible | ✅ |

---

## Notes

- **CUDA-specific:** Le workspace utilise `ExecSpace::memory_space` donc fonctionne automatiquement pour CUDA
- **Thread-safety:** Chaque benchmark fixture a son propre workspace
- **Validation:** Les résultats doivent être IDENTIQUES à l'API originale (tests croisés)
