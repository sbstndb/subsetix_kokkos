# Proposition d'API MPI pour FVD Subsetix

## Version 1.0 - 31 Décembre 2024

---

## Résumé Exécutif

Cette proposition définit une API haut niveau pour la gestion MPI dans le framework FVD de Subsetix. Les principes directeurs sont:

1. **Implicit par défaut** - L'utilisateur n'a pas besoin de gérer MPI explicitement pour les cas standards
2. **Extensible** - L'utilisateur peut contrôler les détails si nécessaire
3. **Compile-time friendly** - C++20, templates, pas de `std::function`
4. **Kokkos CUDA ready** - Compatible device execution
5. **Topologie queryable** - L'utilisateur peut interroger les voisins et la structure

---

## Architecture Globale

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER API (High Level)                            │
│                                                                          │
│  auto solver = Solver::builder(nx, ny)                                   │
│      .with_decomposition<Cartesian2D>()                                  │
│      .with_mpi<MPIComm::Auto>()                                          │
│      .with_observer_mode(ObserverMode::Rank0Only)                        │
│      .with_auto_comm(true)                                               │
│      .build();                                                           │
│                                                                          │
│  while (solver.time() < t_final) {                                       │
│      solver.step();  // Communications automatiques                      │
│  }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         MPI ABSTRACTION LAYER                            │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │ Decomposition    │  │ HaloManager      │  │ TopologyQuery    │      │
│  │ Policy           │  │                  │  │                  │      │
│  │                  │  │ - Exchange       │  │ - Neighbors      │      │
│  │ - Cartesian1D    │  │ - Ghost sync     │  │ - Rank mapping   │      │
│  │ - Cartesian2D    │  │ - Async comm     │  │ - Boundary info  │      │
│  │ - Metis          │  │                  │  │                  │      │
│  │ - SpaceFilling   │  │                  │  │                  │      │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘      │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │ LoadBalance      │  │ ObserverManager  │  │ AMRDist          │      │
│  │ Policy           │  │ (MPI-aware)      │  │                  │      │
│  │                  │  │                  │  │                  │      │
│  │ - CellCount      │  │ - Rank0Only      │  │ - TreePerRank    │      │
│  │ - UserDefined    │  │ - AllRanks       │  │ - Distributed    │      │
│  │ - Static         │  │ - Reduced        │  │ - NeighborMesh   │      │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         IMPLEMENTATION LAYER                             │
│  (MPI calls, GPU-Direct, point-to-point, collectives, etc.)             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Politiques de Décomposition (Compile-Time)

### 1.1 Concept de Décomposition

```cpp
namespace subsetix::fvd::mpi {

/**
 * @brief Concept pour les politiques de décomposition de domaine
 *
 * Une politique de décomposition doit fournir:
 * - Type de configuration (POD, GPU-safe)
 * - Méthode statique pour initialiser la décomposition
 * - Méthode pour trouver les voisins d'un rank
 */
template<typename T>
concept DecompositionPolicy = requires {
    typename T::Config;
    T::init_decomposition;
    T::find_neighbors;
    T::is_on_boundary;
};

} // namespace mpi
```

### 1.2 Politiques de Décomposition Disponibles

```cpp
namespace subsetix::fvd::mpi {

// ============================================================================
// 1. Décomposition Cartésienne 1D
// ============================================================================

/**
 * @brief Décomposition selon l'axe X (bandes verticales)
 *
 * Exemple avec 4 ranks:
 *   Rank 0 | Rank 1 | Rank 2 | Rank 3
 *
 * Voisins: left (rank-1), right (rank+1), sauf aux frontières
 */
struct Cartesian1D {
    struct Config {
        int nx_global;           // Taille globale du domaine
        int ny_global;
        int padding = 1;         // Padding pour ghosts cells
    };

    struct DecompositionInfo {
        int rank;                // Mon rank
        int nranks;              // Nombre total de ranks
        int nx_local;            // Taille locale de mon domaine
        int ny_local;
        int x_offset;            // Position de mon domaine dans le global
        int y_offset;
        int left_neighbor;       // Rank de mon voisin gauche (-1 si bord)
        int right_neighbor;      // Rank de mon voisin droite (-1 si bord)
    };

    // Initialisation au démarrage
    static DecompositionInfo init(const Config& cfg, MPI_Comm comm);

    // Trouver les voisins
    static std::array<int, 4> find_neighbors(const DecompositionInfo& info);

    // Vérifier si un rank est sur la frontière globale
    static bool is_on_boundary(const DecompositionInfo& info, Boundary side);
};

// ============================================================================
// 2. Décomposition Cartésienne 2D
// ============================================================================

/**
 * @brief Décomposition en grille cartésienne 2D
 *
 * Exemple avec 4 ranks (grille 2x2):
 *   Rank 0 | Rank 1
 *   -------+-------
 *   Rank 2 | Rank 3
 *
 * Voisins: left, right, top, bottom
 */
struct Cartesian2D {
    struct Config {
        int nx_global;
        int ny_global;
        int px = -1;             // Nombre de ranks en X (-1 = auto)
        int py = -1;             // Nombre de ranks en Y (-1 = auto)
        int padding = 1;
    };

    struct DecompositionInfo {
        int rank, nranks;
        int nx_local, ny_local;
        int x_offset, y_offset;

        // Position dans la grille
        int grid_x, grid_y;
        int grid_nx, grid_ny;

        // Voisins
        int neighbors[4];        // {left, right, bottom, top}
    };

    static DecompositionInfo init(const Config& cfg, MPI_Comm comm);
    static std::array<int, 4> find_neighbors(const DecompositionInfo& info);
    static bool is_on_boundary(const DecompositionInfo& info, Boundary side);
};

// ============================================================================
// 3. Décomposition par Courbe de Remplissage (Space-Filling Curve)
// ============================================================================

/**
 * @brief Décomposition selon une courbe de Hilbert/Morton
 *
 * Garantit que les cellules proches dans l'espace sont sur le même rank
 * ou sur des ranks voisins.
 */
struct SpaceFillingCurve {
    struct Config {
        int nx_global;
        int ny_global;

        enum CurveType {
            Morton,      // Z-order curve (plus simple)
            Hilbert     // Hilbert curve (meilleure localité)
        };
        CurveType curve_type = Hilbert;

        int order = 0;           // Ordre de la courbe (-1 = auto)
    };

    struct DecompositionInfo {
        int rank, nranks;
        int nx_local, ny_local;
        int x_offset, y_offset;

        // Liste des voisins (nombre arbitraire)
        Kokkos::View<int*, DeviceMemorySpace> neighbors;
        int num_neighbors;

        // Mapping cellule -> rank
        static KOKKOS_FUNCTION int cell_to_rank(int ix, int iy, const Config& cfg);
    };

    static DecompositionInfo init(const Config& cfg, MPI_Comm comm);
    static bool is_on_boundary(const DecompositionInfo& info, Boundary side);
};

// ============================================================================
// 4. Décomposition par Metis (Nombre Arbitraire de Voisins)
// ============================================================================

/**
 * @brief Décomposition via Metis (graph partitioning)
 *
 * Permet un nombre arbitraire de voisins par rank.
 * Optimal pour des géométries complexes et AMR.
 */
struct MetisDecomposition {
    struct Config {
        // Soit on fournit un graphe explicite
        struct GraphInput {
            Kokkos::View<int*, DeviceMemorySpace> adjacency;  // Liste d'adjacence
            Kokkos::View<int*, DeviceMemorySpace> offsets;     // Offsets dans adjacency
            Kokkos::View<int*, DeviceMemorySpace> weights;     // Poids des arêtes (optionnel)
        };
        std::optional<GraphInput> graph;

        // Soit on laisse Metis construire le graphe depuis la géométrie
        struct GeometryInput {
            int nx, ny;
            int halo_width = 1;
        };
        std::optional<GeometryInput> geometry;

        // Options Metis
        int nparts = 0;           // Nombre de partitions (0 = nranks)
        float imbalance = 1.05f;  // Tolérance de déséquilibre
        int options[METIS_NOPTIONS];
    };

    struct DecompositionInfo {
        int rank, nranks;
        int nx_local, ny_local;
        int x_offset, y_offset;

        // Nombre arbitraire de voisins
        Kokkos::View<int*, DeviceMemorySpace> neighbors;
        int num_neighbors;

        // Pour chaque voisin: liste des cellules frontière
        Kokkos::View<int**, DeviceMemorySpace> boundary_cells; // [neighbor][local_cell]
        Kokkos::View<int*, DeviceMemorySpace> boundary_counts; // [neighbor]
    };

    static DecompositionInfo init(const Config& cfg, MPI_Comm comm);
    static bool is_on_boundary(const DecompositionInfo& info, Boundary side);
};

// ============================================================================
// 5. Décomposition Statique (User-Defined)
// ============================================================================

/**
 * @brief Décomposition statique définie par l'utilisateur
 *
 * Pour les cas où l'utilisateur veut un contrôle total.
 */
struct StaticDecomposition {
    struct Config {
        // Tableau rank -> [x_min, x_max, y_min, y_max]
        std::vector<std::array<int, 4>> rank_domains;
    };

    struct DecompositionInfo {
        // ... similaire aux autres
    };

    static DecompositionInfo init(const Config& cfg, MPI_Comm comm);
};

} // namespace mpi
```

---

## 2. API du Builder avec MPI

### 2.1 Signature Générale

```cpp
namespace subsetix::fvd::solver {

template<
    FiniteVolumeSystem System,
    typename Reconstruction = reconstruction::NoReconstruction,
    template<typename> class FluxScheme = flux::RusanovFlux,
    typename TimeIntegrator = time::ForwardEuler<typename System::RealType>,
    typename Decomposition = mpi::Cartesian1D    // NOUVEAU
>
class AdaptiveSolver {
public:
    // ...
};

} // namespace solver
```

### 2.2 Méthodes du Builder

```cpp
class AdaptiveSolver {
public:
    class Builder {
    public:
        // ... méthodes existantes ...

        // ====================================================================
        // MPI Configuration (NOUVEAU)
        // ====================================================================

        /**
         * @brief Spécifie le type de décomposition de domaine
         *
         * @tparam Decomp La politique de décomposition (Cartesian1D, Cartesian2D, Metis, etc.)
         * @param config Configuration de la décomposition
         *
         * Exemple:
         *   builder.with_decomposition<mpi::Cartesian2D>({
         *       .nx_global = 1000,
         *       .ny_global = 500,
         *       .px = 2,
         *       .py = 2
         *   });
         */
        template<typename Decomp>
        Builder& with_decomposition(typename Decomp::Config config);

        /**
         * @brief Configure la largeur des halo cells pour les communications MPI
         *
         * @param halo_width Nombre de couches de ghost cells (défaut: 1)
         *
         * Toutes les directions utilisent la même largeur.
         */
        Builder& with_halo_width(int halo_width = 1);

        /**
         * @brief Active/désactive les communications automatiques
         *
         * @param enable Si true, les halo exchanges sont automatiques après chaque step()
         *
         * Par défaut: true (communications implicites)
         */
        Builder& with_auto_comm(bool enable = true);

        /**
         * @brief Configure le mode de communication
         *
         * @param mode Mode de communication
         */
        Builder& with_comm_mode(mpi::CommMode mode);

        /**
         * @brief Configure le comportement des observateurs en multi-rank
         *
         * @param mode Mode des observateurs
         *
         * Voir mpi::ObserverMode ci-dessous pour les options.
         */
        Builder& with_observer_mode(mpi::ObserverMode mode);

        /**
         * @brief Configure le load balancing pour AMR
         *
         * @param policy Politique de load balancing
         */
        template<typename LoadBalancePolicy>
        Builder& with_load_balancing(LoadBalancePolicy policy);

        /**
         * @brief Configure le communicateur MPI
         *
         * @param comm_mode Mode d'initialisation MPI
         * @param custom_comm Communicateur custom (optionnel)
         *
         * Par défaut: MPI_COMM_WORLD avec auto-initialisation
         */
        Builder& with_mpi_comm(
            mpi::MPICommMode comm_mode = mpi::MPICommMode::Auto,
            MPI_Comm custom_comm = MPI_COMM_WORLD
        );

        // Construit le solveur
        AdaptiveSolver build();
    };
};
```

---

## 3. Modes et Configurations

### 3.1 Mode de Communication

```cpp
namespace subsetix::fvd::mpi {

/**
 * @brief Mode de communication entre ranks
 */
enum class CommMode {
    /**
     * Communications synchrones bloquantes après chaque step
     * - Simple et sûr
     * - Peut sous-utiliser GPU/CPU pendant les comm
     */
    Synchronous,

    /**
     * Communications asynchrones non-bloquantes
     * - Recouvrement calcul/communication
     * - Meilleure performance
     */
    Asynchronous,

    /**
     * Communications avec GPU-Direct (si disponible)
     * - Mémoire GPU vers GPU directe
     * - Requis CUDA-aware MPI
     */
    GPUDirect,

    /**
     * Mode hybride: automatique selon le runtime
     */
    Hybrid
};

} // namespace mpi
```

### 3.2 Mode des Observateurs

```cpp
namespace subsetix::fvd::mpi {

/**
 * @brief Comportement des observateurs en environnement multi-rank
 */
enum class ObserverMode {
    /**
     * Seul le rang 0 affiche/écrit les messages
     * - Autres ranks silencieux
     * - Sortie console/VTK uniquement depuis rank 0
     */
    Rank0Only,

    /**
     * Tous les ranks affichent avec préfixe
     * - Format: "[Rank 0/4] Step 100: t=0.5"
     * - Utile pour debugging
     */
    AllRanks,

    /**
     * Réductions automatiques pour statistiques globales
     * - Les observers font des MPI_Allreduce
     * - Affiche les min/max/moyenne global
     */
    Reduced,

    /**
     * Mode hybride: réduit pour les stats, rank 0 pour la sortie
     */
    Smart
};

} // namespace mpi
```

---

## 4. API MPI du Solveur

### 4.1 Informations MPI

```cpp
class AdaptiveSolver {
public:
    // ========================================================================
    // MPI Query API
    // ========================================================================

    /**
     * @brief Retourne les informations MPI de ce solveur
     */
    const mpi::DecompositionInfo& mpi_info() const;

    /**
     * @brief Retourne mon rank
     */
    int rank() const;

    /**
     * @brief Retourne le nombre total de ranks
     */
    int nranks() const;

    /**
     * @brief Retourne vrai si je suis le rang 0
     */
    bool is_rank0() const;

    /**
     * @brief Retourne la liste de mes voisins
     *
     * @return Kokkos::View des rangs voisins
     */
    Kokkos::View<int*, DeviceMemorySpace> neighbors() const;

    /**
     * @brief Retourne le nombre de voisins
     */
    int num_neighbors() const;

    /**
     * @brief Vérifie si un rank est mon voisin
     */
    KOKKOS_FUNCTION bool is_neighbor(int other_rank) const;

    /**
     * @brief Retourne les informations sur les cellules frontière avec un voisin
     */
    Kokkos::View<int*, DeviceMemorySpace> boundary_cells_with(int neighbor_rank) const;

    // ========================================================================
    // Communication Control
    // ========================================================================

    /**
     * @brief Active/désactive les communications automatiques
     */
    void enable_auto_comm(bool enable = true);

    /**
     * @brief Exécute manuellement un échange de halos
     *
     * Utile si auto_comm est désactivé.
     */
    void exchange_halos();

    /**
     * @brief Synchronise tous les ranks (barrière MPI)
     */
    void barrier();

    /**
     * @brief Réduit une valeur scalaire sur tous les ranks
     *
     * @param local_value Valeur locale
     * @param op Opération de réduction (MPI_SUM, MPI_MAX, etc.)
     * @return Valeur réduite (disponible sur tous les ranks)
     */
    template<typename T>
    T allreduce(T local_value, MPI_Op op = MPI_SUM) const;

    /**
     * @brief Broadcast une valeur depuis le rank 0
     */
    template<typename T>
    void broadcast(T& value, int root = 0) const;

    // ========================================================================
    // Load Balancing (AMR)
    // ========================================================================

    /**
     * @brief Active le load balancing automatique pour AMR
     *
     * @param enable Si true, redistribution automatique après remesh
     */
    void enable_auto_load_balance(bool enable = true);

    /**
     * @brief Force un load balancing immédiat
     */
    void load_balance();

    /**
     * @brief Retourne des statistiques sur le load balancing
     */
    LoadBalanceStats load_balance_stats() const;

    // ========================================================================
    // Ghost Cells / Halo
    // ========================================================================

    /**
     * @brief Retourne la largeur actuelle du halo
     */
    int halo_width() const;

    /**
     * @brief Modifie la largeur du halo (reconstruction nécessaire)
     */
    void set_halo_width(int width);

    /**
     * @brief Synchronise les ghost cells (synonyme de exchange_halos)
     */
    void sync_ghosts();
};
```

---

## 5. Observateurs MPI-Aware

### 5.1 Extension du SolverState

```cpp
namespace subsetix::fvd::mpi {

/**
 * @brief Snapshot du solveur avec informations MPI
 */
template<typename Real = float>
struct MPISolverState : public SolverState<Real> {
    // Hérite de SolverState<Real>...

    // Informations MPI
    int rank = 0;
    int nranks = 1;

    // Statistiques locales
    std::size_t local_cells = 0;
    Real local_min_rho = Real(0);
    Real local_max_rho = Real(0);
    Real local_avg_rho = Real(0);

    // Statistiques globales (réduites)
    std::size_t global_cells = 0;
    Real global_min_rho = Real(0);
    Real global_max_rho = Real(0);
    Real global_avg_rho = Real(0);

    // Load balancing
    float load_balance_ratio = 1.0f;  // max_cells / avg_cells
    int most_loaded_rank = 0;

    // Communication
    double last_comm_time = 0.0;      // Temps passé en comm MPI (secondes)
    double comm_overlap_ratio = 0.0;  // Recouvrement calcul/comm
};

} // namespace mpi
```

### 5.2 Observateurs MPI

```cpp
namespace subsetix::fvd::mpi {

/**
 * @brief Gestionnaire d'observateurs MPI-aware
 */
template<typename Real = float>
class MPIObserverManager : public ObserverManager<Real> {
public:
    /**
     * @brief Configure le mode des observateurs
     */
    void set_observer_mode(ObserverMode mode);

    /**
     * @brief Observateur de progression MPI-aware
     *
     * S'adapte au mode configuré:
     * - Rank0Only: affiche seulement sur rank 0
     * - AllRanks: affiche avec préfixe [Rank X/Y]
     * - Reduced: affiche les stats globales
     */
    int on_mpi_progress(std::function<void(const MPISolverState<Real>&)> callback);

    /**
     * @brief Observateur de load balancing
     */
    int on_load_balance(std::function<void(const MPISolverState<Real>&,
                                         float old_ratio,
                                         float new_ratio)> callback);

    // Observateurs prédéfinis
    template<typename RealType = Real>
    static SolverCallback<RealType> mpi_progress_printer(int print_interval = 1);

    template<typename RealType = Real>
    static SolverCallback<RealType> mpi_csv_logger(const std::string& filename);
};

} // namespace mpi
```

---

## 6. Politiques de Load Balancing

```cpp
namespace subsetix::fvd::mpi {

/**
 * @brief Concept pour les politiques de load balancing
 */
template<typename T>
concept LoadBalancePolicy = requires {
    T::compute_cell_cost;
    T::should_redistribute;
};

// ============================================================================
// Politiques de Load Balancing
// ============================================================================

/**
 * @brief Load balancing par nombre de cellules
 *
 * Stratégie la plus simple: chaque rank doit avoir ~N_cells_total / N_ranks
 */
struct CellCountLoadBalance {
    template<typename System>
    struct Config {
        float max_imbalance = 1.1f;  // Tolérance de déséquilibre
        int check_interval = 100;    // Vérifier tous les N steps
    };

    // Fonction device-friendly pour calculer le coût d'une cellule
    template<typename System>
    KOKKOS_FUNCTION static float compute_cell_cost(
        const typename System::Conserved& U,
        const typename System::Primitive& q,
        int level
    ) {
        return 1.0f;  // Toutes les cellules ont le même coût
    }
};

/**
 * @brief Load basé sur le niveau de raffinement
 *
 * Les cellules fines sont plus coûteuses (plus de travail par cellule)
 */
struct LevelWeightedLoadBalance {
    template<typename System>
    struct Config {
        float level_weight = 2.0f;  // Une cellule de niveau l+1 coûte level_weight fois plus
        float max_imbalance = 1.1f;
    };

    template<typename System>
    KOKKOS_FUNCTION static float compute_cell_cost(
        const typename System::Conserved& U,
        const typename System::Primitive& q,
        int level
    ) {
        return std::pow(level_weight, level);
    }
};

/**
 * @brief Load basé sur l'activité physique (gradient, etc.)
 *
 * Les régions actives (chocs, gradients forts) sont plus coûteuses
 */
struct PhysicsWeightedLoadBalance {
    template<typename System>
    struct Config {
        float gradient_weight = 1.0f;
        float shock_weight = 2.0f;
        float base_cost = 1.0f;
    };

    template<typename System>
    KOKKOS_FUNCTION static float compute_cell_cost(
        const typename System::Conserved& U,
        const typename System::Primitive& q,
        const typename System::RealType grad_rho_x,
        const typename System::RealType grad_rho_y,
        int level
    ) {
        float cost = base_cost;

        // Coût basé sur les gradients
        float grad_mag = Kokkos::sqrt(grad_rho_x * grad_rho_x + grad_rho_y * grad_rho_y);
        cost += gradient_weight * grad_mag;

        // Bonus pour les chocs
        float div_v = /* ... divergence de vitesse */;
        if (div_v < -0.5f) {  // Compression forte
            cost *= shock_weight;
        }

        return cost;
    }
};

/**
 * @brief Load personnalisé par l'utilisateur (compile-time)
 *
 * L'utilisateur fournit un lambda qui devient une fonction device via KOKKOS_LAMBDA
 */
template<typename System, typename CostFunc>
struct CustomLoadBalance {
    struct Config {
        CostFunc cost_func;
        float max_imbalance = 1.1f;
    };

    template<typename... Args>
    KOKKOS_FUNCTION static float compute_cell_cost(Args&&... args) {
        return CostFunc{}(std::forward<Args>(args)...);
    }
};

} // namespace mpi
```

---

## 7. Exemples d'Utilisation

### 7.1 Exemple 1: MPI Implicite (Minimal)

```cpp
#include <subsetix/fvd/fvd_integrators.hpp>

using namespace subsetix::fvd;
using namespace subsetix::fvd::solver;

int main(int argc, char** argv) {
    // Auto-initialisation MPI et Kokkos
    Kokkos::initialize(argc, argv);

    // Le solver détecte automatiquement MPI
    auto solver = EulerSolverRK3::builder(1000, 500)
        .with_domain(0.0, 2.0, 0.0, 1.0)
        .with_initial_condition(my_condition)
        // MPI automatique:
        // - Détection de MPI_COMM_WORLD
        // - Décomposition cartésienne 2D automatique
        // - Communications implicites activées
        // - Observateurs sur rank 0 seulement
        .build();

    // Simulation - les communications sont automatiques !
    while (solver.time() < t_final) {
        solver.step();  // Halo exchange automatique après chaque step
    }

    Kokkos::finalize();
    return 0;
}
```

### 7.2 Exemple 2: Décomposition Explicite

```cpp
auto solver = EulerSolverRK3::builder(1000, 500)
    .with_domain(0.0, 2.0, 0.0, 1.0)
    .with_initial_condition(my_condition)
    // Décomposition explicite
    .with_decomposition<mpi::Cartesian2D>({
        .nx_global = 1000,
        .ny_global = 500,
        .px = 2,    // 2 ranks en X
        .py = 2     // 2 ranks en Y (total: 4 ranks)
    })
    .with_halo_width(2)
    .build();

// Afficher les informations MPI
auto info = solver.mpi_info();
if (solver.is_rank0()) {
    printf("Décomposition: %d x %d ranks\n",
           info.grid_nx, info.grid_ny);
    printf("Mon domaine: [%d, %d] x [%d, %d]\n",
           info.x_offset, info.x_offset + info.nx_local,
           info.y_offset, info.y_offset + info.ny_local);
}
```

### 7.3 Exemple 3: Metis avec Voisins Arbitraires

```cpp
auto solver = EulerSolverRK3::builder(1000, 500)
    .with_domain(0.0, 2.0, 0.0, 1.0)
    .with_initial_condition(my_condition)
    // Décomposition Metis
    .with_decomposition<mpi::MetisDecomposition>({
        .geometry = mpi::MetisDecomposition::GeometryInput{
            .nx = 1000,
            .ny = 500,
            .halo_width = 1
        },
        .imbalance = 1.05f
    })
    .build();

// Interroger les voisins
auto neighbors = solver.neighbors();
printf("J'ai %d voisins:\n", solver.num_neighbors());
for (int i = 0; i < solver.num_neighbors(); ++i) {
    printf("  - Rank %d\n", neighbors(i));
}
```

### 7.4 Exemple 4: Observateurs MPI Multi-Mode

```cpp
auto solver = EulerSolverRK3::builder(1000, 500)
    .with_domain(0.0, 2.0, 0.0, 1.0)
    .with_initial_condition(my_condition)
    // Mode: tous les ranks affichent avec préfixe
    .with_observer_mode(mpi::ObserverMode::AllRanks)
    .build();

// Observer: progression
solver.observers().on_mpi_progress([](const auto& state) {
    if (state.step % 10 == 0) {
        // Affichage automatique avec préfixe [Rank X/Y]
        printf("Step %d: t=%.4f, %zu local cells, %zu global cells\n",
               state.step, state.time, state.local_cells, state.global_cells);
    }
});

// Observer: load balancing
solver.observers().on_load_balance([](const auto& state, float old_ratio, float new_ratio) {
    if (state.rank == 0) {
        printf("Load balance: %.2f -> %.2f (most loaded: rank %d)\n",
               old_ratio, new_ratio, state.most_loaded_rank);
    }
});
```

### 7.5 Exemple 5: Mode Réduit pour Stats Globales

```cpp
auto solver = EulerSolverRK3::builder(1000, 500)
    .with_domain(0.0, 2.0, 0.0, 1.0)
    .with_initial_condition(my_condition)
    // Mode: réductions automatiques
    .with_observer_mode(mpi::ObserverMode::Reduced)
    .build();

solver.observers().on_mpi_progress([](const auto& state) {
    if (state.rank == 0) {  // Affichage uniquement sur rank 0
        printf("Step %d: t=%.4f\n", state.step, state.time);
        printf("  Density: min=%.4f, max=%.4f, avg=%.4f (GLOBAL)\n",
               state.global_min_rho, state.global_max_rho, state.global_avg_rho);
        printf("  Load balance: %.2f (ratio max/avg)\n", state.load_balance_ratio);
        printf("  Comm time: %.2f ms\n", state.last_comm_time * 1000.0);
    }
});
```

### 7.6 Exemple 6: Communications Contrôlées

```cpp
auto solver = EulerSolverRK3::builder(1000, 500)
    .with_domain(0.0, 2.0, 0.0, 1.0)
    .with_initial_condition(my_condition)
    // Désactiver les communications automatiques
    .with_auto_comm(false)
    // Mode asynchrone pour performances
    .with_comm_mode(mpi::CommMode::Asynchronous)
    .build();

while (solver.time() < t_final) {
    // Faire plusieurs sous-étapes avant de communiquer
    for (int substep = 0; substep < 5; ++substep) {
        solver.step_without_comm();
    }

    // Communication manuelle
    solver.exchange_halos();
}
```

### 7.7 Exemple 7: Load Balancing avec AMR

```cpp
auto solver = EulerSolverRK3::builder(1000, 500)
    .with_domain(0.0, 2.0, 0.0, 1.0)
    .with_initial_condition(my_condition)
    // Load balancing automatique
    .with_load_balancing<mpi::LevelWeightedLoadBalance<Euler2D<float>>>({
        .level_weight = 2.0f,
        .max_imbalance = 1.1f
    })
    .build();

solver.enable_auto_load_balance(true);

// Observer pour le load balancing
solver.observers().on_load_balance([](const auto& state, float old_ratio, float new_ratio) {
    if (state.rank == 0) {
        printf("Load balance: %.2f -> %.2f\n", old_ratio, new_ratio);
    }
});

while (solver.time() < t_final) {
    solver.step();  // Load balancing automatique si nécessaire
}
```

### 7.8 Exemple 8: Custom Load Balance (Device-Friendly)

```cpp
// Fonction de coût personnalisée (Kokkos device-friendly)
auto my_cost_func = KOKKOS_LAMBDA(
    const Euler2D<float>::Conserved& U,
    const Euler2D<float>::Primitive& q,
    float grad_rho_x,
    float grad_rho_y,
    int level
) -> float {
    // Les cellules avec fort gradient coûtent plus cher
    float grad_mag = Kokkos::sqrt(grad_rho_x * grad_rho_x + grad_rho_y * grad_rho_y);
    float base_cost = 1.0f;

    if (grad_mag > 0.1f) {
        base_cost += 5.0f * grad_mag;  // Zones actives
    }

    // Bonus pour les cellules raffinées
    base_cost *= Kokkos::pow(2.0f, level);

    return base_cost;
};

auto solver = EulerSolverRK3::builder(1000, 500)
    .with_domain(0.0, 2.0, 0.0, 1.0)
    .with_initial_condition(my_condition)
    // Utiliser la fonction de coût personnalisée
    .with_load_balancing<mpi::CustomLoadBalance<Euler2D<float>, decltype(my_cost_func)>>({
        .cost_func = my_cost_func,
        .max_imbalance = 1.15f
    })
    .build();
```

### 7.9 Exemple 9: VTK Output avec MPI

```cpp
auto solver = EulerSolverRK3::builder(1000, 500)
    .with_domain(0.0, 2.0, 0.0, 1.0)
    .with_initial_condition(my_condition)
    // Seul rank 0 écrit les fichiers VTK
    .with_observer_mode(mpi::ObserverMode::Rank0Only)
    .build();

int frame = 0;
solver.observers().add_callback(SolverEvent::StepEnd,
    [&frame, &solver](SolverEvent, const auto& state) {
        if (state.step % 100 == 0 && solver.is_rank0()) {
            char filename[256];
            snprintf(filename, sizeof(filename), "output/frame_%04d.vtk", frame++);
            solver.write_vtk(filename);  // Gather automatique sur rank 0
        }
    });
```

### 7.10 Exemple 10: Full-Featured Simulation

```cpp
auto solver = EulerSolverRK3::builder(1000, 500)
    .with_domain(0.0, 2.0, 0.0, 1.0)
    .with_initial_condition(mach2_cylinder)
    // Décomposition Metis pour géométrie complexe
    .with_decomposition<mpi::MetisDecomposition>({
        .geometry = {.nx = 1000, .ny = 500, .halo_width = 2},
        .imbalance = 1.05f
    })
    .with_halo_width(2)
    // Communication asynchrone avec GPU-Direct
    .with_comm_mode(mpi::CommMode::GPUDirect)
    .with_auto_comm(true)
    // Observateurs: stats globales
    .with_observer_mode(mpi::ObserverMode::Reduced)
    // Load balancing par niveau de raffinement
    .with_load_balancing<mpi::LevelWeightedLoadBalance<Euler2D<float>>>({
        .level_weight = 2.0f,
        .max_imbalance = 1.1f
    })
    // AMR standard
    .with_refinement_config(standard_amr<Euler2D<float>>())
    .build();

// Observateurs complets
solver.observers().on_mpi_progress([](const auto& state) {
    if (state.rank == 0) {
        printf("Step %d: t=%.4f, dt=%.5f\n", state.step, state.time, state.dt);
        printf("  Cells: %zu total, %.2f load balance\n",
               state.global_cells, state.load_balance_ratio);
        printf("  Density (global): min=%.4f, max=%.4f, avg=%.4f\n",
               state.global_min_rho, state.global_max_rho, state.global_avg_rho);
        printf("  Comm: %.2f ms, overlap: %.1f%%\n",
               state.last_comm_time * 1000.0, state.comm_overlap_ratio * 100.0);
    }
});

solver.observers().on_remesh([](const auto& state, size_t old_local, size_t new_local) {
    if (state.rank == 0) {
        printf("REMESH: %zu -> %zu cells (local)\n", old_local, new_local);
    }
});

solver.observers().on_load_balance([](const auto& state, float old_r, float new_r) {
    if (state.rank == 0) {
        printf("LOAD BALANCE: %.2f -> %.2f\n", old_r, new_r);
    }
});

// Simulation principale
while (solver.time() < t_final) {
    solver.step();
}

// Afficher les statistiques finales
if (solver.is_rank0()) {
    auto stats = solver.load_balance_stats();
    printf("\n=== Final Statistics ===\n");
    printf("Total cells: %zu\n", stats.total_cells);
    printf("Load balance: %.2f (target: < %.2f)\n",
           stats.final_ratio, stats.target_ratio);
    printf("Total comm time: %.2f s (%.1f%% of runtime)\n",
           stats.total_comm_time, stats.comm_ratio * 100.0);
}
```

---

## 8. Résumé des Points Clés

### 8.1 Principes de Design

| Aspect | Choix de Design |
|--------|----------------|
| **Décomposition** | Template policies (compile-time) |
| **Communication** | Semi-implicite avec contrôle |
| **Observers** | 3 modes: Rank0Only, AllRanks, Reduced |
| **AMR+MPI** | Tree-per-rank avec load balancing optionnel |
| **Load Balance** | Device-friendly functions, pas de std::function |
| **Topologie** | Queryable par l'utilisateur |
| **Halo** | Largeur unique (toutes directions) |
| **Init MPI** | Auto par défaut (MPI_InitImplicit) |

### 8.2 Files à Ajouter

```
include/subsetix/fvd/mpi/
├── mpi_concepts.hpp           // Concepts C++20
├── decomposition.hpp          // Politiques de décomposition
├── topology.hpp               // Topologie et voisins
├── halo_manager.hpp           // Gestion des ghosts cells
├── comm_manager.hpp           // Gestion des communications
├── observer_mpi.hpp           // Observateurs MPI-aware
├── load_balance.hpp           // Politiques de load balancing
└── mpi_config.hpp             // Configurations et enums

include/subsetix/fvd/solver/
└── adaptive_solver_mpi.hpp   // Extension MPI du solveur

examples/
└── fvd_mpi_examples.cpp       // 10+ exemples MPI

docs/
└── fvd_mpi_api_proposal.md    // Ce document
```

---

## 9. Questions Restantes

1. **Serialisation**: Comment transférer les états AMR entre ranks ?
2. **Mesh voisins**: Format d'échange des meshs pour load balancing ?
3. **GPU-Direct MPI**: Détection et fallback automatique ?
4. **Performance profiling**: Comment mesurer le temps de comm proprement ?

---

## 10. Next Steps

1. Valider l'API avec l'utilisateur
2. Implémenter les politiques de décomposition (Cartesian1D/2D d'abord)
3. Implémenter HaloManager avec communications asynchrones
4. Étendre ObserverManager pour MPI-aware
5. Implémenter le load balancing de base
6. Écrire des tests unitaires
7. Benchmarks de performance scaling
