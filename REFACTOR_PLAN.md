# Plan de Refactor: MACH2 → FVD Abstraction

**Branche:** `feat/refactor-mach2-fvd-abstraction`
**Date:** 2026-01-20
**Version:** 2.0 (Auditée et validée)
**Approche:** Migration progressive hybride avec validation complète

---

## 📊 Résumé Exécutif

### Situation Actuelle

| Composant | État | Lignes | Fonctionnalité |
|-----------|------|--------|----------------|
| `mach2_cylinder.cpp` | ✅ Production | 2018 | Fonctionnel, testé, performant |
| `fvd_mach2_cylinder_example.cpp` | ❌ Stub | 178 | Compile mais ne calcule rien |
| `AdaptiveSolver` | ⚠️ 20% | ~1000 | API complète, implémentation vide |
| **CSR AMR Operations** | ✅ Production | ~865 | Prêtes à l'emploi |

### Audit de Faisabilité (5 agents)

| Phase | Faisabilité | Confiance | Lignes à écrire | Risques |
|-------|-------------|-----------|-----------------|---------|
| **Phase 0** | ✅ Certaine | 100% | ~200 | Aucun |
| **Phase 0.5** | ✅ Certaine | 100% | ~100 | Aucun |
| **Phase 1** | ✅ Certaine | 100% | ~50 | Aucun (binaire compatible) |
| **Phase 2a** | ✅ Certaine | 95% | ~100 | Aucun |
| **Phase 2b** | ⚠️ Possible | 85% | ~150 | Différences numériques |
| **Phase 3** | ⚠️ Moderée | 70% | ~250 | **Cylindre = obstacle interne** |
| **Phase 4** | ⚠️ Moderée | 75% | ~200 | CSR→Dense adapters requis |
| **Phase 5** | ✅ Certaine | 95% | ~200 | Pont actions→masque |
| **Phase 6** | ✅ Certaine | 95% | ~600 | Wrappers CSR existants |
| **Phase 7** | ⚠️ Complexe | 85% | ~1550 | Intégration massive |
| **Phase 8** | ✅ Certaine | 100% | ~100 | Aucun |

**Total estimé: ~3,500 lignes de nouveau code**

### Objectif

Migrer `mach2_cylinder.cpp` vers l'API FVD de manière **progressive**, en gardant les deux versions fonctionnelles et en validant à chaque étape.

---

## 🏗️ Architecture: 4 Niveaux FVD

```
LEVEL 4: AdaptiveSolver<T, System, Flux, Reconstruction, TimeIntegrator>
    ├── Builder pattern
    ├── Observer pattern
    └── Checkpoint/Restart
    ↓
LEVEL 3: System Abstraction
    ├── Euler2D<Real>
    ├── Conserved / Primitive variables
    └── flux_phys_x/y, to_primitive, from_primitive
    ↓
LEVEL 2: Core Primitives
    ├── Flux: Rusanov, HLLC, Roe
    ├── Reconstruction: NoRecon, MUSCL + Limiters
    ├── Time: RK1, RK2, RK3, RK4, SSPRK3
    ├── BC: Dirichlet, Neumann, Reflective, TimeDependent
    ├── AMR: Gradient, ShockSensor, Vorticity criteria
    └── Sources: Gravity, Custom, Zone
    ↓
LEVEL 1: Subsetix Core
    ├── CSR Geometry (IntervalSet2D)
    ├── Field2D (CSR storage)
    └── AMR operations (prolong, restrict, remesh) ← ✅ PRODUCTION READY
```

### Couverture Actuelle

| Level | Composant | Implémentation | Utilisé dans mach2 | Statut Audit |
|-------|-----------|----------------|-------------------|--------------|
| L4 | AdaptiveSolver | 20% (stub) | ❌ Non | ~1550 lignes à écrire |
| L3 | Euler2D | ✅ 100% | ❌ Non (inline) | **Binaire compatible** |
| L2 | Flux | ✅ 100% | ❌ Non (inline) | **GPU-safe** |
| L2 | Reconstruction | ✅ 95% | ❌ Non | Limiters OK |
| L2 | Time | ✅ 100% | ❌ Non | **Clean interface** |
| L2 | BC | ⚠️ 98% | ❌ Non | **Cylindre non supporté** |
| L2 | AMR Criteria | ✅ 100% | ❌ Non | **Pont CSR requis** |
| L1 | Subsetix Core | ✅ 100% | ✅ Oui | **Production-ready** |

---

## ⚠️ Risques Critiques Identifiés

### 1. Incompatibilité CSR/FVD (ÉLEVÉ)

**Problème:** FVD suppose des tableaux denses, CSR utilise un stockage sparse.

**Impact:** Ne peut pas utiliser directement les intégrateurs FVD.

**Solution:**
- Créer un adaptateur CSR→Dense dans **Phase 0** (pas Phase 1!)
- Option courte: conversions temporaires (acceptables pour MVP)
- Option longue: intégrateurs CSR-aware

```cpp
// mach2_fvd_bridge.hpp (NOUVEAU, Phase 0)
template<typename System>
class CSRFieldAdapter {
    // Convertit CSR Fields ↔ Dense Views pour intégrateurs
    Kokkos::View<typename System::Conserved*> to_dense(const ConservedFields& csr);
    ConservedFields from_dense(const Kokkos::View<Conserved*>& dense, const IntervalSet2D& geom);
};
```

### 2. Obstacle Cylindre (MOYEN)

**Problème:** Le système de BC FVD ne gère que les frontières **externes** (4 côtés du domaine).

**Impact:** Le cylindre est un obstacle **interne** - pas supporté.

**Solution:**
- **Hybride:** Utiliser `BcManager` pour les BCs externes uniquement
- **Garder l'approche mach2:** `fill_ghost_cells` avec `ghost_mask` pour le cylindre
- Ne pas forcer le cylindre dans le système BC générique

```cpp
// Phase 3: Approche hybride
void apply_bcs_mach2(ConservedFields& U, const IntervalSet2DDevice& ghost_mask,
                     const BcManager<Euler2D<float>>& bc_mgr,  // Externes
                     const IntervalSet2DDevice& cylinder, Real gamma);
```

### 3. Différences Numériques (FAIBLE)

**Problème:** `Euler2D::to_primitive` a un bug potentiel - utilise `U.rho` dans l'énergie cinétique au lieu de `q.rho`.

**Impact:** Pressure légèrement différente → propagation aux flux.

**Solution:**
- Vérifier l'équivalence bit-à-bit en Phase 1
- Si différence: corriger `euler2d.hpp` ou adapter `mach2_cylinder.cpp`

### 4. Template Bloat (MOYEN)

**Problème:** 360 combinaisons possibles (systèmes × flux × reconstruction × intégrateurs × BCs).

**Impact:** Temps de compilation > 60 secondes.

**Solution:**
- Instantiation explicite pour mach2 seulement (~6 combinaisons)
- `extern template` pour les types communs
- Prcompiled headers

---

## 📋 Plan de Migration Progressif (Révisé)

### Phase 0: Infrastructure de Validation + Adapteur CSR (1 semaine)

**Objectif:** Créer un framework de test **et** résoudre l'incompatibilité CSR.

#### Tâches:

1. **Créer `tests/mach2_validation/`**
   ```
   tests/mach2_validation/
   ├── CMakeLists.txt
   ├── validation_main.cpp
   ├── field_comparator.hpp      # L1, L2, Linf norms
   ├── diagnostics_comparator.hpp
   └── reference/
       └── mach2_baseline_*.vtk
   ```

2. **CRITIQUE: Créer `mach2_fvd_bridge.hpp`** (ADAPTEUR CSR)
   ```cpp
   // examples/mach2_cylinder/mach2_fvd_bridge.hpp
   template<typename System>
   class CSRFieldAdapter {
       // Conversion CSR ↔ Dense pour les intégrateurs
       auto to_dense_views(const ConservedFields& csr);
       ConservedFields from_dense_views(const auto& dense, const IntervalSet2D& geom);
   };

   // Vérification de compatibilité binaire
   static_assert(sizeof(Euler2D<Real>::Conserved) == sizeof(Conserved));
   static_assert(std::is_trivially_copyable_v<Euler2D<Real>::Conserved>);
   ```

3. **Extraction des utilitaires**
   - `examples/mach2_cylinder/mach2_utils.hpp`
   - Extraire: `compute_diagnostics`, `write_multilevel_outputs`

4. **Configuration commune**
   - `examples/mach2_cylinder/mach2_config.hpp`
   - Paramètres identiques pour les deux versions

5. **Test initial de régression**
   - Lancer mach2 original → sauvegarder baseline
   - Comparer chaque run contre la baseline

**Critère de succès:** Tests passent + adaptateur CSR compile.

---

### Phase 0.5: Validation Type Safety (2 jours) ⭐ NOUVEAU

**Objectif:** S'assurer que les types FVD sont compatibles avec CSR **avant** de les utiliser.

#### Tâches:

1. **Tests de compatibilité mémoire**
   ```cpp
   // tests/mach2_validation/type_safety_tests.cpp
   static_assert(sizeof(Euler2D<float>::Conserved) == sizeof(Conserved));
   static_assert(alignof(Euler2D<float>::Conserved) == alignof(Conserved));
   static_assert(std::is_trivially_copyable_v<Euler2D<float>::Conserved>);
   ```

2. **Tests GPU**
   - Compilation: CPU/GPU × Debug/Release
   - Vérifier que toutes les fonctions critiques sont `KOKKOS_INLINE_FUNCTION`

3. **Documentation des invariants**
   - Documenter les hypothèses sur les layouts mémoire
   - Spécifier les valeurs d'epsilon pour tous les calculs

**Critère de succès:** Tous les asserts compilent, tests GPU passent.

---

### Phase 1: Structures de Données (1 semaine)

**Objectif:** Remplacer les structures inline par `Euler2D<Real>`.

#### Tâches:

1. **Utiliser `Euler2D<Real>`**
   ```cpp
   using System = subsetix::fvd::Euler2D<Real>;
   using Conserved = System::Conserved;
   using Primitive = System::Primitive;

   // Remplacer les appels de fonction
   auto q = System::to_primitive(U, gamma);  // était: cons_to_prim
   ```

2. **Vérifier l'équivalence bit-à-bit**
   - Lancer la validation après cette phase
   - Toute différence indique un bug dans `euler2d.hpp`

3. **Validation**
   - Bit-identique à la Phase 0 (tolérance 0.0)
   - Performance identique (profilage)

**Risque:** BUG dans `Euler2D::to_primitive` - utiliser `U.rho` au lieu de `q.rho` dans l'énergie cinétique.

**Critère de succès:** `memcmp` passe entre original et FVD.

---

### Phase 2a: Flux Schemes (2 semaines)

**Objectif:** Remplacer `rusanov_flux_x/y` par `RusanovFlux<Euler2D<Real>>`.

#### Tâches:

1. **Intégrer `RusanovFlux`**
   ```cpp
   subsetix::fvd::flux::RusanovFlux<Euler2D<Real>> numerical_flux{system, gamma};

   // Dans EulerStencilSoA:
   auto flux_L = numerical_flux.flux_x(UL, UR, qL, qR);
   ```

2. **Remplacer les appels inline**
   - Lignes 325-367 dans `mach2_cylinder.cpp`
   - Garder la Structure-of-Arrays

3. **Validation**
   - Comparer les valeurs de flux
   - Vérifier l'égalité numérique

**Faisabilité:** ✅ 95% - fonctions `KOKKOS_INLINE_FUNCTION`, signatures compatibles.

**Critère de succès:** Valeurs de flux identiques (tolérance 1e-12).

---

### Phase 2b: Reconstruction (parallèle à 2a, 1 semaine)

**Objectif:** Ajouter MUSCL pour la précision 2ème ordre.

#### Tâches:

1. **Intégrer `MUSCL_Reconstruction`**
   ```cpp
   using Reconstruction = subsetix::fvd::reconstruction::MUSCL_Reconstruction<
       subsetix::fvd::reconstruction::MinmodLimiter<Real>>;
   ```

2. **Tests de convergence**
   - **Ne pas** comparer bit-à-bit (original = 1er ordre)
   - Vérifier l'ordre de convergence: erreur ∝ h²

3. **Comparer les limiteurs**
   - Minmod, MC, Superbee, VanLeer

**Faisabilité:** ⚠️ 85% - différents limiteurs donnent différents résultats.

**Critère de succès:** Ordre de convergence 2.0 ± 0.1.

---

### Phase 3: Conditions Limites (1 semaine)

**Objectif:** BCs externes avec `BcManager`, cylindre avec approche mach2.

#### Tâches:

1. **BCs externes avec `BoundaryConfigBuilder`**
   ```cpp
   auto bc = BoundaryConfigBuilder<Euler2D<Real>>::inflow_outflow(inflow_state, gamma);
   // Inflow (x=0): Dirichlet
   // Outflow (x=L): Neumann
   // Walls (y=0,H): Reflective
   ```

2. **Cylindre: GARDER l'approche mach2**
   ```cpp
   // CRITIQUE: Ne pas utiliser BcManager pour le cylindre
   // Garder fill_ghost_cells() avec ghost_mask

   IntervalSet2DDevice ghost_mask = set_difference_device(
       expanded_fluid, base_fluid, ctx);

   fill_obstacle_ghosts(U, ghost_mask, cylinder, gamma, no_slip);
   ```

3. **Kernel d'application BC côté device**
   - Créer `apply_bcs_to_csr_field()`
   - Utiliser `BcRegistry` pour le lookup

**Faisabilité:** ⚠️ 70% - **Cylindre = obstacle interne non supporté** par FVD BCs.

**Critère de succès:** Ghost cells identiques pour BCs externes, conservation OK.

---

### Phase 4: Intégration Temporelle (1 semaine)

**Objectif:** Remplacer Euler par des intégrateurs FVD.

#### Tâches:

1. **Utiliser l'adapteur CSR→Dense**
   ```cpp
   CSRFieldAdapter<Euler2D<Real>> adapter;

   // Pour chaque timestep:
   auto U_dense = adapter.to_dense(U);  // CSR → Dense
   rk_step<System, ClassicRK4<Real>>(U_dense, dt, t, rhs, ...);
   U = adapter.from_dense(U_dense, geometry);  // Dense → CSR
   ```

2. **Créer `compute_rhs`**
   - Boucle sur la géométrie
   - Applique les flux FVD
   - Retourne dU/dt

3. **Tester plusieurs intégrateurs**
   - ForwardEuler (original)
   - Heun2, Kutta3, ClassicRK4
   - Vérifier les ordres de convergence

**Faisabilité:** ⚠️ 75% - **CSR→Dense conversion requis** (approche hybride OK).

**Critère de succès:** Ordres de convergence corrects (1, 2, 3, 4).

---

### Phase 5: Critères AMR (1 semaine) ⭐ DÉPLACÉ (après Phase 1)

**Objectif:** Remplacer `build_refine_mask` par `GradientCriterion`.

**Pourquoi plus tôt?** Les critères AMR ne dépendent que de `Euler2D<Real>`.

#### Tâches:

1. **Utiliser `GradientCriterion`**
   ```cpp
   auto criterion = amr::GradientCriterion<Euler2D<Real>>::density(threshold);
   ```

2. **Pont actions→masque CSR**
   ```cpp
   // CRITIQUE: Besoin d'un pont
   Field2DDevice<int8_t> actions = evaluate_criterion(criterion, U, geometry);
   IntervalSet2DDevice refine_mask = actions_to_csr_mask(actions, geometry, ctx);
   ```

3. **Utiliser les opérations CSR existantes**
   ```cpp
   // Gradient avec stencil CSR
   apply_csr_stencil_on_set_device(gradient, U.rho, fluid_geom, stencil, false);

   // Threshold
   subsetix::csr::threshold(gradient, threshold, refine_mask, ctx);

   // Expand pour buffer
   subsetix::csr::morphology::expand(refine_mask, buffer, refined, ctx);
   ```

**Faisabilité:** ✅ 95% - opérations CSR existantes.

**Critère de succès:** Masques de raffinement équivalents.

---

### Phase 6: Opérations Multi-Niveaux (2 semaines)

**Objectif:** Créer des wrappers FVD autour des opérations CSR.

#### Tâches:

1. **Créer `fvd/amr/amr_operations.hpp`**
   ```cpp
   template<typename System>
   class AmrHierarchy {
       MultilevelGeoDevice geometries_;
       std::array<typename System::Views, MAX_LEVELS> U_levels_;

       void prolong_to_fine(int level);
       void restrict_to_coarse(int level);
       void remesh(const RefinementConfig& cfg);
   };
   ```

2. **Wrapper autour de CSR operations**
   ```cpp
   // Utiliser: field_amr.hpp de Subsetix core
   prolong_field_on_subset_device(...);  // Existe!
   restrict_field_on_subset_device(...);  // Existe!
   refine_level_up_device(...);           // Existe!
   ```

**Faisabilité:** ✅ 95% - **CSR operations = production-ready**.

**Critère de succès:** Conservation exacte lors de prolong/restrict.

---

### Phase 7: AdaptiveSolver (3 semaines)

**Objectif:** Implémenter le solver complet.

#### Tâches:

1. **Phase 7a: Single-Level (1 semaine)**
   ```cpp
   private:
       typename System::Views U_, U_next_, rhs_;
       IntervalSet2DDevice fluid_geometry_;

   public:
       Real step() {
           Real dt = compute_dt_cfl();
           fill_boundaries(U_);
           compute_rhs(U_, rhs_);
           euler_step(U_, dt, rhs_);
           return dt;
       }
   ```

2. **Phase 7b: Multi-Level V-Cycle (1 semaine)**
   ```cpp
   private:
       static constexpr int MAX_LEVELS = 16;
       std::array<typename System::Views, MAX_LEVELS> U_levels_;
       MultilevelGeoDevice geometries_;

   public:
       Real step() {
           Real dt = compute_dt_global();
           for (int lvl = 1; lvl < num_levels_; ++lvl)
               prolong_guard_from_coarse(U_levels_[lvl], U_levels_[lvl-1]);
           for (int lvl = 0; lvl < num_levels_; ++lvl)
               fill_boundaries(U_levels_[lvl]);
           for (int lvl = num_levels_ - 1; lvl >= 0; --lvl)
               apply_stencil_and_time_integrate(U_levels_[lvl], dt);
           for (int lvl = 0; lvl < num_levels_ - 1; ++lvl)
               restrict_to_coarse(U_levels_[lvl], U_levels_[lvl+1]);
           if (step_count_ % cfg_.remesh_stride == 0)
               remesh_hierarchy();
           return dt;
       }
   ```

3. **Phase 7c: Builder et Observers (1 semaine)**
   - Compléter le builder pattern
   - Intégrer les observers
   - VTK output

**Faisabilité:** ⚠️ 85% - ~1,550 lignes à écrire, intégration complexe.

**Critère de succès:** `fvd_mach2_cylinder_example` fonctionnel.

---

### Phase 8: Nettoyage et Documentation (1 semaine)

**Objectif:** Finaliser la migration.

#### Tâches:

1. **Déprécier `mach2_cylinder.cpp`**
2. **Documentation complète**
3. **Tests CI/CD**
4. **Benchmarking**

**Critère de succès:** Code review passée, CI vert.

---

## 🔗 Graphes de Dépendance

### Chemin Critique

```
Phase 0 (Validation + Adaptateur CSR) - 1 sem
  ↓
Phase 0.5 (Type Safety) - 2 jours
  ↓
Phase 1 (Structures) - 1 sem
  ↓
Phase 2a (Flux) - 2 sems
  ↓
Phase 3 (BCs) - 1 sem
  ↓
Phase 4 (Time Integration) - 1 sem
  ↓
Phase 7a (Single-Level) - 1 sem
  ↓
Phase 7b (Multi-Level) - 1 sem
  ↓
Phase 7c (Builder/Observers) - 1 sem
  ↓
Phase 8 (Cleanup) - 1 sem
```

**Total chemin critique: 10.5 semaines**

### Pistes Parallèles

```
Après Phase 1:
├── Track A: Phase 2a (Flux) ──────┐
├── Track B: Phase 2b (Recon) ─────┤
└── Track C: Phase 5 (AMR Criteria)┼──→ Merge → Phase 3/4/7
                                    ↑
                            Phase 6 (Multi-Level)
                            (peut commencer après Phase 5)
```

---

## 📊 Matrice de Validation

| Phase | Méthode de Validation | Tolérance | Critère de Succès |
|-------|----------------------|-----------|-------------------|
| **Phase 0** | Auto-consistance | N/A | Tests passent |
| **Phase 0.5** | Compile-time asserts | 0.0 | `static_assert` OK |
| **Phase 1** | Bit-exact `memcmp` | 0.0 | Structures identiques |
| **Phase 2a** | Bit-exact flux values | 1e-12 | Flux identiques |
| **Phase 2b** | Convergence rate | ±0.1 | Ordre 2.0 |
| **Phase 3** | Ghost cells compare | 1e-12 | BCs identiques |
| **Phase 4** | Convergence rate | ±0.1 | Ordres corrects |
| **Phase 5** | Mask XOR count | <1% cellules | Masques équivalents |
| **Phase 6** | Conservation | 1e-10 | Masse conservée |
| **Phase 7** | Integrated diagnostics | 1e-10 | Tous diagnostics OK |
| **Phase 8** | Performance benchmark | ≥90% | Target atteint |

---

## 🎯 Succès

Le refactor est réussi si:

1. ✅ `fvd_mach2_cylinder_example` compile et tourne
2. ✅ Résultats identiques à `mach2_cylinder` (tolérance 1e-10)
3. ✅ Performance ≥ 90% de l'original
4. ✅ Code utilisateur: ~200 lignes vs 2018
5. ✅ Tests de validation dans le CI
6. ✅ Documentation complète

---

## 📁 Arborescence Finale

```
examples/
├── mach2_cylinder/
│   ├── mach2_cylinder.cpp          # Original (déprécié)
│   ├── mach2_utils.hpp             # Utilitaires extraits
│   ├── mach2_config.hpp            # Configuration commune
│   └── mach2_fvd_bridge.hpp        # ⭐ Adaptateur CSR (Phase 0)
│
├── fvd_mach2_cylinder_example.cpp  # NOUVELLE implémentation
│
└── fvd_simulation_examples.cpp

tests/
└── mach2_validation/
    ├── CMakeLists.txt
    ├── validation_main.cpp
    ├── field_comparator.hpp
    ├── diagnostics_comparator.hpp
    ├── type_safety_tests.cpp        # ⭐ Phase 0.5
    ├── convergence_tests.cpp
    └── reference/
        └── mach2_baseline_*.vtk

include/subsetix/fvd/
└── amr/
    └── amr_operations.hpp           # ⭐ Phase 6 (wrapper CSR)
```

---

## 🚀 Prochaine Étape Immédiate

**Commencer par la Phase 0: Infrastructure de Validation + Adaptateur CSR**

1. Créer `tests/mach2_validation/`
2. **CRITIQUE:** Créer `mach2_fvd_bridge.hpp` avec `CSRFieldAdapter`
3. Extraire les utilitaires de `mach2_cylinder.cpp`
4. Tests de type safety (Phase 0.5)
