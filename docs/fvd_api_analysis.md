# Analyse de l'API FVD de Subsetix

## Vue d'ensemble

L'API **FVD (Finite Volume Dynamics)** de Subsetix est une couche d'abstraction haut niveau pour les simulations de dynamique des fluides par volumes finis, construite sur Kokkos pour GPU/CPU.

---

## Principes de Conception

### 1. **Genericité par Templates (Compile-Time)**
```cpp
template<
    FiniteVolumeSystem System,              // Le système d'équations (Euler, Navier-Stokes...)
    typename Reconstruction,                 // Schéma spatial (NoRecon, MUSCL...)
    template<typename> class FluxScheme,     // Flux numérique (Rusanov, HLLC...)
    typename TimeIntegrator                  // Intégrateur temporel (RK2, RK3, RK4...)
>
class AdaptiveSolver;
```

### 2. **Builder Pattern pour Configuration Fluent**
```cpp
auto solver = Solver::builder(nx, ny)
    .with_domain(x_min, x_max, y_min, y_max)
    .with_initial_condition(my_lambda)
    .with_cfl(0.5)
    .with_gamma(1.4)
    .build();
```

### 3. **Type Safety via Concepts C++20**
```cpp
template<typename T>
concept FiniteVolumeSystem = requires {
    typename T::RealType;
    typename T::Conserved;
    typename T::Primitive;
    T::to_primitive(Conserved, gamma);
    T::flux_phys_x(Conserved, Primitive);
};
```

### 4. **GPU-Safe (POD Types Only)**
```cpp
static_assert(std::is_trivially_copyable_v<TimeDependentBC<Real>>);
// Pas de virtual functions sur device
// Toutes les structures GPU doivent être POD
```

---

## Architecture en 4 Niveaux

```
┌─────────────────────────────────────────────────────────────────────┐
│ NIVEAU 4: High-Level Solver (User API)                              │
│                                                                     │
│ AdaptiveSolver::builder(nx, ny)                                     │
│     .with_domain(...)                                               │
│     .with_initial_condition(...)                                    │
│     .with_observers(...)                                            │
│     .build()                                                        │
│                                                                     │
│ while (t < t_final) solver.step();                                  │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│ NIVEAU 3: System Abstraction (Generic Interface)                    │
│                                                                     │
│ - Euler2D<float/double>: Équations d'Euler compressibles           │
│ - Advection2D: Advection linéaire                                   │
│ - NavierStokes2D: Avec viscosité (futur)                           │
│                                                                     │
│ Concept: FiniteVolumeSystem                                         │
│   + Conserved variables                                             │
│   + Primitive variables                                             │
│   + to_primitive(), from_primitive()                                │
│   + flux_phys_x/y(), sound_speed()                                  │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│ NIVEAU 2: Core Primitives (GPU-Safe)                                │
│                                                                     │
│ Time Integrators:                                                   │
│   - ForwardEuler (1st order)                                        │
│   - Heun2, Ralston3 (2nd order)                                     │
│   - Kutta3, SSPRK3 (3rd order)                                      │
│   - ClassicRK4 (4th order)                                          │
│                                                                     │
│ Reconstruction:                                                     │
│   - NoReconstruction (1st order)                                    │
│   - MUSCL_Reconstruction<Limiter> (2nd order)                       │
│                                                                     │
│ Flux Schemes:                                                       │
│   - RusanovFlux (diffusif, robuste)                                 │
│   - HLLCFlux (précis pour chocs)                                    │
│   - RoeFlux (haute résolution)                                      │
│                                                                     │
│ AMR:                                                                │
│   - RefinementManager                                               │
│   - GradientCriterion, ShockSensorCriterion                         │
│   - VorticityCriterion, ValueRangeCriterion                         │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│ NIVEAU 1: Subsetix Core (Existing)                                  │
│                                                                     │
│ - IntervalSet2D: Représentation AMR légère                          │
│ - Field2D: Champs de données sur grilles AMR                        │
│ - CSR Operations: Opérations géométriques GPU                       │
│ - VTK Output: Visualisation                                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Composants Clés de l'API

### 1. **Solveurs Pré-configurés (Type Aliases)**

```cpp
namespace subsetix::fvd::solver {

// Basiques (1er ordre espace, différents ordres temps)
using EulerSolver    = AdaptiveSolver<Euler, NoRecon, Rusanov, ForwardEuler>;
using EulerSolverRK2  = AdaptiveSolver<Euler, NoRecon, Rusanov, Heun2>;
using EulerSolverRK3  = AdaptiveSolver<Euler, NoRecon, Rusanov, Kutta3>;
using EulerSolverRK4  = AdaptiveSolver<Euler, NoRecon, Rusanov, ClassicRK4>;

// Convenience
using DefaultSolver   = EulerSolverRK3;   // Bon compromis
using FastSolver      = EulerSolver;      // Pour tests rapides
using HighOrderSolver = EulerSolverRK4;   // Pour écoulements lisses

} // namespace solver
```

### 2. **Observateurs (Monitoring/Callbacks)**

```cpp
enum class SolverEvent {
    SimulationStart, SimulationEnd,
    StepBegin, StepEnd,
    SubStepBegin, SubStepEnd,    // Pour sous-étapes RK
    RemeshBegin, RemeshEnd,
    OutputWritten, Error
};

// Enregistrement d'observers
solver.observers().on_progress([](const auto& state) {
    printf("Step %d: t=%.4f, dt=%.5f\n", state.step, state.time, state.dt);
});

solver.observers().on_remesh([](const auto& state, size_t old_c, size_t new_c) {
    printf("Remesh: %zu -> %zu cells\n", old_c, new_c);
});

// Observers prédéfinis
auto csv_log = Observers::csv_logger<Real>("data.csv");
solver.observers().add_callback(SolverEvent::StepEnd, csv_log);
```

### 3. **Conditions aux Limites**

```cpp
// BCs statiques
auto bc = BoundaryConfigBuilder<System>::inflow_outflow(inflow_state, gamma);
solver.set_boundary_conditions(bc);

// BCs dépendantes du temps
auto sinusoidal = boundary::sinusoidal_inlet<System>(rho0, u0, frequency);
solver.set_time_dependent_bc("left", sinusoidal);

// BCs zonales (différentes zones sur une frontière)
auto zone = ZonePredicate<Real>::interval_x(0.2, 0.4);
solver.add_zonal_bc("bottom", zone, primitive_state, gamma);
```

### 4. **AMR (Adaptive Mesh Refinement)**

```cpp
// Configuration standard AMR
auto amr_config = standard_amr<System>();
amr_config.config.max_level = 5;
solver.set_refinement_config(amr_config);

// Critères de raffinement
RefinementManager<System> mgr;
mgr.add_gradient_criterion(0.1f);                    // Gradient de densité
mgr.add_shock_sensor_criterion(ShockSensor::Ducros, 0.8f);
mgr.add_vorticity_criterion(1.0f);
mgr.add_value_range_criterion(ValueRange::Density, 0.5f, 1.5f);

// Zones d'exclusion (raffinement limité dans certaines régions)
mgr.add_exclusion_circle(0.5f, 0.5f, 0.1f, min_level=2);
mgr.add_exclusion_rectangle(0.0f, 0.3f, 0.0f, 0.5f, min_level=1);
```

### 5. **Termes Sources**

```cpp
// Gravité (builtin)
solver.add_gravity(-9.81f);

// Sources personnalisées
solver.add_source([](const auto& U, const auto& q,
                     Real x, Real y, Real t) {
    // U: variables conservatives, q: variables primitives
    return System::Conserved{
        0,               // Source de masse
        force_x,         // Source de quantité de mouvement X
        force_y,         // Source de quantité de mouvement Y
        heat_source      // Source d'énergie
    };
});
```

### 6. **Checkpoint/Restart**

```cpp
// Checkpoint manuel
solver.write_checkpoint("checkpoint.bin", CheckpointFormat::Binary);

// Auto-checkpoint toutes les N étapes
solver.set_auto_checkpoint(100, "checkpoint_prefix");

// Restart
auto new_solver = Solver::builder(nx, ny).build();
new_solver.read_checkpoint("checkpoint.bin", CheckpointFormat::Binary);
```

### 7. **Validation**

```cpp
ValidationConfig cfg;
cfg.check_negative_density = true;
cfg.check_negative_pressure = true;
cfg.check_nan = true;
cfg.check_inf = true;
cfg.min_density = 0.001f;
cfg.min_pressure = 0.001f;
cfg.max_mach = 100.0f;
cfg.max_temperature = 10000.0f;
solver.set_validation(cfg);
```

---

## Patterns de Code Courants

### Pattern 1: Simulation Minimaliste

```cpp
#include <subsetix/fvd/fvd_integrators.hpp>

using namespace subsetix::fvd;
using namespace subsetix::fvd::solver;

int main() {
    auto solver = EulerSolver::builder(100, 50)
        .with_domain(0.0, 1.0, 0.0, 0.5)
        .with_initial_condition([](float x, float y) {
            return Euler2D<float>::from_primitive(
                {1.0f, 0.0f, 0.0f, 1.0f/1.4f}, 1.4f);
        })
        .build();

    while (solver.time() < 0.1) {
        solver.step();
    }

    return 0;
}
```

### Pattern 2: Simulation avec Monitoring

```cpp
auto solver = EulerSolverRK3::builder(nx, ny)
    .with_domain(...)
    .with_initial_condition(...)
    .build();

// Observers
solver.observers().on_progress(progress_callback);
solver.observers().on_remesh(remesh_callback);
solver.observers().add_callback(SolverEvent::Error, error_callback);

while (solver.time() < t_final) {
    solver.step();
}
```

### Pattern 3: Solveur Personnalisé

```cpp
using MySolver = AdaptiveSolver<
    Euler2D<float>,
    MUSCL_Reconstruction<MinmodLimiter>,  // 2nd order space
    HLLCFlux,                               // HLLC flux
    ClassicRK4<float>                       // 4th order time
>;

auto solver = MySolver::builder(nx, ny)
    .with_domain(...)
    .build();
```

---

## Avantages de l'API

| Aspect | Bénéfice |
|--------|----------|
| **Simplicité** | Quelques lignes pour une simulation complète |
| **Genericité** | Fonctionne avec n'importe quel système (Euler, NS, advection...) |
| **Performance** | Compile-time template specialization, zéro overhead |
| **Extensibilité** | Ajoutez vos propres termes sources, observateurs, BCs |
| **GPU-Native** | Kokkos pour exécution CPU/GPU transparente |
| **Type-Safe** | Concepts C++20 pour erreurs de compilation claires |
| **Monitoring** | Observers pour logging, profiling, VTK output |
| **Robustesse** | Validation, checkpoints, error handling |

---

## Comparaison: Code Utilisateur vs MACH2

### MACH2 (Fortran classique)

```fortran
! Des centaines de lignes pour:
! - Initialiser les grilles AMR
! - Configurer les schémas numériques
! - Gérer les conditions aux limites
! - Boucle principale complexe
! - Sortie des résultats

call init_amr_grids(nlevels, ...)
call setup_flux_scheme(flux_type, ...)
call setup_reconstruction(recon_type, ...)
call setup_bc_array(...)
call setup_time_integrator(...)

do while (t < t_final)
    call fill_ghost_cells(...)
    call compute_fluxes(...)
    call update_solution(...)
    call apply_source_terms(...)
    call check_refinement(...)
    if (remesh_needed) call remesh(...)
end do
```

### Subsetix FVD

```cpp
auto solver = EulerSolverRK3::builder(nx, ny)
    .with_domain(x_min, x_max, y_min, y_max)
    .with_initial_condition(my_condition)
    .build();

auto amr = standard_amr<Euler2D<float>>();
solver.set_refinement_config(amr);

solver.observers().on_progress(my_callback);

while (solver.time() < t_final) {
    solver.step();
}
```

**Réduction de code: ~90% pour les cas standards !**

---

## Exemples d'Utilisation

Voir le fichier `examples/fvd_simulation_examples.cpp` pour 12 exemples complets couvrant:

1. **Advection simple** - La plus simple simulation possible
2. **Écoulement Mach 2** - Problème classique de cylindre
3. **AMR + Observers** - Raffinement adaptatif avec monitoring
4. **BCs dépendantes du temps** - Sinusoïdal, pulsation, rampe
5. **Termes sources** - Gravité, chauffage, traînée, réaction
6. **Checkpoint/Restart** - Persistence d'état
7. **Étude paramétrique** - Multi-runs systématiques
8. **Solveur personnalisé** - MUSCL + HLLC + RK4
9. **Sortie VTK** - Visualisation
10. **Multi-physique** - Combustion complexe
11. **Builder namespace** - API de commodité
12. **Error handling** - Validation robuste

---

## Roadmap Future

D'après `docs/fvd_api_next_steps_design.md`:

- [ ] Systèmes additionnels (Navier-Stokes, MHD)
- [ ] Limiteurs de reconstruction supplémentaires (MC, Superbee, van Leer)
- [ ] Schémas de flux d'ordre supérieur
- [ ] Intégrateurs temporels implicites (pour stiff problems)
- [ ] Parallélisme multi-GPU via MPI
- [ ] Interface Python pour prototypage rapide
- [ ] Benchmarks de performance systématiques

---

## Conclusion

L'API FVD de Subsetix représente une **modernisation significative** du code de dynamique des fluides par volumes finis:

- **Productivité:** 10x moins de code que MACH2
- **Performance:** Template compilation, GPU-native
- **Flexibilité:** Genericité totale via templates
- **Robustesse:** Observers, validation, checkpoints
- **Maintenabilité:** Architecture en couches claires

C'est une base solide pour le développement futur de codes CFD modernes.
