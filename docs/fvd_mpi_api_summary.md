# Implémentation de l'API MPI FVD - Résumé

## Date: 31 Décembre 2024

---

## Fichiers Créés (Interface Haut Niveau avec Stubs)

Tous les fichiers sont dans `include/subsetix/fvd/mpi/` :

| Fichier | Description |
|---------|-------------|
| `mpi_concepts.hpp` | Concepts C++20 pour DecompositionPolicy, LoadBalancePolicy |
| `mpi_config.hpp` | Enums: CommMode, ObserverMode, MPICommMode, Boundary + structs de config |
| `decomposition.hpp` | Politiques: Cartesian1D, Cartesian2D, SpaceFillingCurve, Metis, Static |
| `topology.hpp` | TopologyQuery, TopologyInfo, HaloBuilder, HaloInfo |
| `comm_manager.hpp` | CommManager pour halos exchange, collectives, GPU-Direct |
| `observer_mpi.hpp` | MPISolverState, MPIObserverManager, callbacks MPI-aware |
| `load_balance.hpp` | Politiques: CellCount, LevelWeighted, PhysicsWeighted, Custom |
| `fvd_mpi.hpp` | **Header d'intégration** - inclut tous les composants MPI |

Extension du solveur dans `include/subsetix/fvd/solver/` :

| Fichier | Description |
|---------|-------------|
| `adaptive_solver_mpi.hpp` | Extension MPIAdaptiveSolver avec Builder étendu |

---

## Arborescence Complète

```
include/subsetix/fvd/
├── mpi/
│   ├── mpi_concepts.hpp           ✅ Concepts C++20
│   ├── mpi_config.hpp             ✅ Enums + Config structs
│   ├── decomposition.hpp          ✅ Politiques de décomposition
│   ├── topology.hpp               ✅ Topologie, voisins, halo
│   ├── comm_manager.hpp           ✅ Gestionnaire de communications
│   ├── observer_mpi.hpp           ✅ Observateurs MPI-aware
│   ├── load_balance.hpp           ✅ Politiques de load balancing
│   └── fvd_mpi.hpp                ✅ Header d'intégration
│
└── solver/
    └── adaptive_solver_mpi.hpp    ✅ Extension MPI du solveur
```

---

## API Rapide

### 1. Include Principal

```cpp
#include <subsetix/fvd/fvd_integrators.hpp>   // API FVD de base
#include <subsetix/fvd/mpi/fvd_mpi.hpp>        // API MPI

using namespace subsetix::fvd;
using namespace subsetix::fvd::solver;
using namespace subsetix::fvd::mpi;
```

### 2. Builder avec MPI

```cpp
// Méthode 1: Utiliser MPIAdaptiveSolver directement
using SolverMPI = EulerSolverRK3_MPI<float>;  // Type alias

auto solver = SolverMPI::Builder(1000, 500)
    .with_domain(0.0, 2.0, 0.0, 1.0)
    .with_initial_condition(my_condition)
    // Configuration MPI
    .with_decomposition<Cartesian2D>({
        .nx_global = 1000,
        .ny_global = 500,
        .px = 2,
        .py = 2
    })
    .with_halo_width(2)
    .with_auto_comm(true)
    .with_comm_mode(CommMode::Asynchronous)
    .with_observer_mode(ObserverMode::Reduced)
    .build();
```

### 3. Requêtes MPI

```cpp
// Informations de base
printf("Rank %d / %d\n", solver.rank(), solver.nranks());
printf("Neighbors: %d\n", solver.num_neighbors());

// Topologie
auto& neighbors = solver.neighbors();
for (int neighbor : neighbors) {
    printf("  Neighbor: %d\n", neighbor);
}

// Vérifier si sur une frontière
if (solver.mpi_info().is_on_boundary(Boundary::Left)) {
    printf("On left boundary\n");
}
```

### 4. Observateurs MPI-Aware

```cpp
// Observer avec stats globales (mode Reduced)
solver.mpi_observers().on_mpi_progress([](const MPISolverState<float>& state) {
    if (state.rank == 0) {  // Affichage seulement sur rank 0
        printf("Step %d: t=%.4f\n", state.step, state.time);
        printf("  Cells: %zu total, %.2f load balance\n",
               state.global_cells, state.load_balance_ratio);
        printf("  Density (global): min=%.4f, max=%.4f, avg=%.4f\n",
               state.global_min_rho, state.global_max_rho, state.global_avg_rho);
        printf("  Comm time: %.2f ms\n", state.last_comm_time * 1000.0);
    }
});

// Observer de load balancing
solver.mpi_observers().on_load_balance([](const auto& state, float old_r, float new_r) {
    if (state.rank == 0) {
        printf("Load balance: %.2f -> %.2f\n", old_r, new_r);
    }
});
```

### 5. Simulation

```cpp
// Les communications sont automatiques
while (solver.time() < t_final) {
    solver.step();  // Halo exchange automatique
}

// Ou contrôle manuel
solver.enable_auto_comm(false);
while (solver.time() < t_final) {
    for (int i = 0; i < 5; ++i) {
        solver.step_without_comm();
    }
    solver.exchange_halos();  // Communication manuelle
}
```

---

## Concepts Clés

### 1. Politiques de Décomposition

```cpp
// Disponibles:
mpi::Cartesian1D          // Bandes verticales
mpi::Cartesian2D          // Grille 2D
mpi::SpaceFillingCurve    // Courbe Hilbert/Morton
mpi::MetisDecomposition   // Nombre arbitraire de voisins
mpi::StaticDecomposition  // User-defined

// Toutes ont:
// - Config (struct POD)
// - DecompositionInfo (resultat de init)
// - init(cfg, comm) -> DecompositionInfo
// - find_neighbors(info) -> std::array<int, 4>
// - is_on_boundary(info, side) -> bool
```

### 2. Modes d'Observateurs

```cpp
enum class ObserverMode {
    Rank0Only,    // Seul rank 0 affiche
    AllRanks,     // Tous affichent avec préfixe [Rank X/Y]
    Reduced,      // Réductions automatiques
    Smart         // Hybride
};
```

### 3. Modes de Communication

```cpp
enum class CommMode {
    Synchronous,   // Bloquantes
    Asynchronous,  // Non-bloquantes avec recouvrement
    GPUDirect,     // GPU-Direct (auto-detect + fallback)
    Hybrid         // Automatique
};
```

### 4. Politiques de Load Balancing

```cpp
// Toutes device-friendly (KOKKOS_FUNCTION)

CellCountLoadBalance         // Par nombre de cellules
LevelWeightedLoadBalance    // Par niveau de raffinement
PhysicsWeightedLoadBalance  // Par activité physique
CustomLoadBalance<Func>      // User-defined (compile-time)
```

---

## Exemple Minimal

```cpp
#include <subsetix/fvd/mpi/fvd_mpi.hpp>

using namespace subsetix::fvd;
using namespace subsetix::fvd::solver;
using namespace subsetix::fvd::mpi;

int main() {
    Kokkos::initialize();

    // MPI auto-détecté
    auto solver = EulerSolverRK3_MPI<float>::Builder(1000, 500)
        .with_domain(0.0, 2.0, 0.0, 1.0)
        .with_initial_condition([](float x, float y) {
            return Euler2D<float>::from_primitive(
                {1.0f, 100.0f, 0.0f, 100000.0f}, 1.4f);
        })
        // Décomposition auto
        .with_decomposition<Cartesian2D>({.nx_global=1000, .ny_global=500})
        // Observateurs: stats globales
        .with_observer_mode(ObserverMode::Reduced)
        .build();

    // Observer
    solver.mpi_observers().on_mpi_progress([](const auto& state) {
        if (state.rank == 0 && state.step % 10 == 0) {
            printf("Step %d: t=%.4f, %zu cells\n",
                   state.step, state.time, state.global_cells);
        }
    });

    // Simulation
    while (solver.time() < 0.1f) {
        solver.step();  // Communications automatiques
    }

    Kokkos::finalize();
    return 0;
}
```

---

## Next Steps

Pour passer de l'interface (stubs) à l'implémentation complète:

### Phase 1: Implémentation Basique
1. `Cartesian1D::init()` - Décomposition 1D simple
2. `CommManager::exchange_halos_sync()` - Halo synchrone
3. `MPIObserverManager::on_mpi_progress()` - Observateurs basiques
4. `TopologyQuery` - Requêtes simples

### Phase 2: Fonctionnalités Avancées
1. `Cartesian2D::init()` - Grille 2D
2. `CommManager::exchange_halos_async()` - Asynchrone
3. `CommManager::allreduce()` - Collectives
4. GPU-Direct detection

### Phase 3: Metis + Load Balancing
1. `MetisDecomposition::init()` - Intégration Metis
2. `CellCountLoadBalance` - Load balancing simple
3. `LevelWeightedLoadBalance` - Avec poids
4. Redistribution automatique

### Phase 4: Optimisation
1. Recouvrement calcul/communication
2. GPU-Direct MPI
3. Profiling et benchmarking
4. Documentation complète

---

## Compilation Test (Single Rank)

Puisque ce sont des stubs, le code devrait compiler pour un seul rank:

```bash
# Test simple
cd examples
g++ -std=c++20 -I../include \
    fvd_mpi_simulation_examples.cpp \
    -o test_mpi \
    -lkokkos -lmpi

# Exécuter avec 1 rank
mpirun -np 1 ./test_mpi 1
```

---

## Validation de l'API

L'API proposée satisfait les exigences:

✅ **Implicit par défaut** - `with_auto_comm(true)`
✅ **Extensible** - Builder pattern avec methods optionnelles
✅ **Compile-time friendly** - Templates, concepts, pas de std::function
✅ **Kokkos CUDA ready** - KOKKOS_FUNCTION pour les device functions
✅ **Topologie queryable** - `TopologyQuery` avec méthodes de requête
✅ **Observers 3 modes** - Rank0Only, AllRanks, Reduced
✅ **Load balancing device-friendly** - Template policies avec lambdas
✅ **Stubs implémentables** - Interface claire pour implémentation future
