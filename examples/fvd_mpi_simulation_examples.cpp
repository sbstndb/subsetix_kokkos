/**
 * @file fvd_mpi_simulation_examples.cpp
 *
 * @brief Collection d'exemples de simulation FVD avec MPI
 *
 * Ce fichier montre l'utilisation de l'API MPI de Subsetix FVD,
 * de l'implicite total au contrôle fin.
 */

#include <subsetix/fvd/fvd_integrators.hpp>
#include <subsetix/fvd/mpi/mpi_decomposition.hpp>
#include <subsetix/fvd/mpi/mpi_observer.hpp>
#include <cstdio>
#include <cmath>

using namespace subsetix::fvd;
using namespace subsetix::fvd::solver;
using namespace subsetix::fvd::mpi;

// ============================================================================
// EXAMPLE 1: MPI IMPLICITE TOTAL (Minimal)
// ============================================================================

void example_1_mpi_minimal() {
    // MPI et Kokkos sont initialisés automatiquement
    // La détection de MPI se fait automatiquement au premier appel

    auto solver = EulerSolverRK3::builder(1000, 500)
        .with_domain(0.0, 2.0, 0.0, 1.0)
        .with_initial_condition([](float x, float y) {
            using System = Euler2D<float>;
            return System::from_primitive({1.0f, 100.0f, 0.0f, 100000.0f}, 1.4f);
        })
        // Tout est automatique:
        // - MPI auto-détecté
        // - Décomposition cartésienne 2D automatique
        // - Communications implicites après chaque step()
        // - Observateurs affichent seulement sur rank 0
        .build();

    // Simulation - les communications sont transparentes !
    const float t_final = 0.1f;
    while (solver.time() < t_final) {
        solver.step();  // Halo exchange automatique
    }

    // Affichage uniquement sur rank 0
    if (solver.is_rank0()) {
        printf("Simulation terminée: t = %.4f\n", solver.time());
    }
}

// ============================================================================
// EXAMPLE 2: DECOMPOSITION EXPLICITE CARTESIENNE 2D
// ============================================================================

void example_2_cartesian_2d() {
    // Décomposition explicite en grille 2x2 (pour 4 ranks)
    auto solver = EulerSolverRK3::builder(1000, 500)
        .with_domain(0.0, 2.0, 0.0, 1.0)
        .with_initial_condition([](float x, float y) {
            using System = Euler2D<float>;
            return System::from_primitive({1.0f, 100.0f, 0.0f, 100000.0f}, 1.4f);
        })
        // Décomposition explicite
        .with_decomposition<Cartesian2D>({
            .nx_global = 1000,
            .ny_global = 500,
            .px = 2,    // 2 ranks en X
            .py = 2,    // 2 ranks en Y
            .padding = 1  // Ghost cells
        })
        .build();

    // Afficher les informations de décomposition
    auto info = solver.mpi_info();
    printf("[Rank %d/%d] Position grille: (%d, %d) / (%d, %d)\n",
           solver.rank(), solver.nranks(),
           info.grid_x, info.grid_y, info.grid_nx, info.grid_ny);
    printf("[Rank %d/%d] Domaine local: x=[%d, %d], y=[%d, %d]\n",
           solver.rank(), solver.nranks(),
           info.x_offset, info.x_offset + info.nx_local,
           info.y_offset, info.y_offset + info.ny_local);

    const float t_final = 0.1f;
    while (solver.time() < t_final) {
        solver.step();
    }
}

// ============================================================================
// EXAMPLE 3: METIS AVEC VOISINS ARBITRAIRES
// ============================================================================

void example_3_metis_decomposition() {
    auto solver = EulerSolverRK3::builder(1000, 500)
        .with_domain(0.0, 2.0, 0.0, 1.0)
        .with_initial_condition([](float x, float y) {
            // Condition initiale complexe pour justifier Metis
            using System = Euler2D<float>;
            float rho = 1.0f + 0.5f * std::sin(10.0f * x) * std::cos(10.0f * y);
            return System::from_primitive({rho, 100.0f, 0.0f, 100000.0f}, 1.4f);
        })
        // Décomposition Metis (nombre arbitraire de voisins)
        .with_decomposition<MetisDecomposition>({
            .geometry = MetisDecomposition::GeometryInput{
                .nx = 1000,
                .ny = 500,
                .halo_width = 1
            },
            .imbalance = 1.05f  // Tolérance de déséquilibre
        })
        .build();

    // Interroger les voisins
    printf("[Rank %d/%d] J'ai %d voisins:\n",
           solver.rank(), solver.nranks(), solver.num_neighbors());

    auto neighbors = solver.neighbors();
    for (int i = 0; i < solver.num_neighbors(); ++i) {
        int neighbor_rank = neighbors(i);
        printf("  - Voisin rank %d\n", neighbor_rank);
    }

    const float t_final = 0.1f;
    while (solver.time() < t_final) {
        solver.step();
    }
}

// ============================================================================
// EXAMPLE 4: OBSERVERS MPI - MODE RANK0 ONLY
// ============================================================================

void example_4_observer_rank0_only() {
    auto solver = EulerSolverRK3::builder(1000, 500)
        .with_domain(0.0, 2.0, 0.0, 1.0)
        .with_initial_condition([](float x, float y) {
            using System = Euler2D<float>;
            return System::from_primitive({1.0f, 100.0f, 0.0f, 100000.0f}, 1.4f);
        })
        // Mode: seul rank 0 affiche (défaut)
        .with_observer_mode(ObserverMode::Rank0Only)
        .build();

    // Observer de progression
    solver.observers().on_mpi_progress([](const MPISolverState<float>& state) {
        if (state.step % 10 == 0) {
            printf("[Rank 0] Step %d: t=%.4f, dt=%.5f, %zu local cells\n",
                   state.step, state.time, state.dt, state.local_cells);
        }
    });

    const float t_final = 0.1f;
    while (solver.time() < t_final) {
        solver.step();
    }
}

// ============================================================================
// EXAMPLE 5: OBSERVERS MPI - MODE ALL RANKS
// ============================================================================

void example_5_observer_all_ranks() {
    auto solver = EulerSolverRK3::builder(1000, 500)
        .with_domain(0.0, 2.0, 0.0, 1.0)
        .with_initial_condition([](float x, float y) {
            using System = Euler2D<float>;
            return System::from_primitive({1.0f, 100.0f, 0.0f, 100000.0f}, 1.4f);
        })
        // Mode: tous les ranks affichent avec préfixe
        .with_observer_mode(ObserverMode::AllRanks)
        .build();

    // Observer de progression
    solver.observers().on_mpi_progress([](const MPISolverState<float>& state) {
        if (state.step % 10 == 0) {
            // Affichage automatique avec préfixe [Rank X/Y]
            printf("[Rank %d/%d] Step %d: t=%.4f, %zu cells\n",
                   state.rank, state.nranks, state.step, state.time, state.local_cells);
        }
    });

    const float t_final = 0.1f;
    while (solver.time() < t_final) {
        solver.step();
    }
}

// ============================================================================
// EXAMPLE 6: OBSERVERS MPI - MODE REDUCED
// ============================================================================

void example_6_observer_reduced() {
    auto solver = EulerSolverRK3::builder(1000, 500)
        .with_domain(0.0, 2.0, 0.0, 1.0)
        .with_initial_condition([](float x, float y) {
            // Shock tube pour variations significatives
            using System = Euler2D<float>;
            float rho = (x < 1.0f) ? 2.0f : 1.0f;
            float p = (x < 1.0f) ? 2.0f : 1.0f;
            return System::from_primitive({rho, 0.0f, 0.0f, p}, 1.4f);
        })
        // Mode: réductions automatiques
        .with_observer_mode(ObserverMode::Reduced)
        .build();

    // Observer avec réductions MPI
    solver.observers().on_mpi_progress([](const MPISolverState<float>& state) {
        if (state.step % 10 == 0 && state.rank == 0) {
            printf("Step %d: t=%.4f\n", state.step, state.time);
            printf("  Density (GLOBAL): min=%.4f, max=%.4f, avg=%.4f\n",
                   state.global_min_rho, state.global_max_rho, state.global_avg_rho);
            printf("  Cells: %zu total, %zu local (rank 0)\n",
                   state.global_cells, state.local_cells);
            printf("  Load balance: %.2f (ratio max/avg)\n", state.load_balance_ratio);
            printf("  Comm time: %.2f ms\n", state.last_comm_time * 1000.0);
        }
    });

    const float t_final = 0.2f;
    while (solver.time() < t_final) {
        solver.step();
    }
}

// ============================================================================
// EXAMPLE 7: COMMUNICATION ASYNCHRONE
// ============================================================================

void example_7_async_communication() {
    auto solver = EulerSolverRK3::builder(1000, 500)
        .with_domain(0.0, 2.0, 0.0, 1.0)
        .with_initial_condition([](float x, float y) {
            using System = Euler2D<float>;
            return System::from_primitive({1.0f, 100.0f, 0.0f, 100000.0f}, 1.4f);
        })
        // Mode asynchrone pour recouvrement calcul/communication
        .with_comm_mode(CommMode::Asynchronous)
        .with_auto_comm(true)
        .build();

    // Observer pour monitorer le recouvrement
    solver.observers().on_mpi_progress([](const MPISolverState<float>& state) {
        if (state.step % 20 == 0 && state.rank == 0) {
            printf("Step %d: comm overlap = %.1f%%\n",
                   state.step, state.comm_overlap_ratio * 100.0);
        }
    });

    const float t_final = 0.1f;
    while (solver.time() < t_final) {
        solver.step();
    }
}

// ============================================================================
// EXAMPLE 8: COMMUNICATION MANUELLE
// ============================================================================

void example_8_manual_communication() {
    auto solver = EulerSolverRK3::builder(1000, 500)
        .with_domain(0.0, 2.0, 0.0, 1.0)
        .with_initial_condition([](float x, float y) {
            using System = Euler2D<float>;
            return System::from_primitive({1.0f, 100.0f, 0.0f, 100000.0f}, 1.4f);
        })
        // Désactiver les communications automatiques
        .with_auto_comm(false)
        .build();

    const float t_final = 0.1f;

    // Faire plusieurs sous-étapes avant de communiquer
    int inner_steps = 0;
    while (solver.time() < t_final) {
        // 5 steps sans communication
        for (int i = 0; i < 5; ++i) {
            solver.step_without_comm();
            inner_steps++;
        }

        // Communication manuelle
        solver.exchange_halos();

        if (solver.is_rank0() && inner_steps % 50 == 0) {
            printf("Step %d: t=%.4f (manual comm every 5 steps)\n",
                   inner_steps, solver.time());
        }
    }
}

// ============================================================================
// EXAMPLE 9: LOAD BALANCING SIMPLE
// ============================================================================

void example_9_load_balancing() {
    auto solver = EulerSolverRK3::builder(1000, 500)
        .with_domain(0.0, 2.0, 0.0, 1.0)
        .with_initial_condition([](float x, float y) {
            using System = Euler2D<float>;
            return System::from_primitive({1.0f, 100.0f, 0.0f, 100000.0f}, 1.4f);
        })
        // Load balancing par nombre de cellules (simple)
        .with_load_balancing<CellCountLoadBalance<Euler2D<float>>>({
            .max_imbalance = 1.1f,
            .check_interval = 100
        })
        // AMR actif
        .with_refinement_config(standard_amr<Euler2D<float>>())
        .build();

    // Activer le load balancing automatique
    solver.enable_auto_load_balance(true);

    // Observer pour le load balancing
    solver.observers().on_load_balance([](const MPISolverState<float>& state,
                                         float old_ratio, float new_ratio) {
        if (state.rank == 0) {
            printf("Load balance: %.2f -> %.2f\n", old_ratio, new_ratio);
        }
    });

    solver.observers().on_mpi_progress([](const MPISolverState<float>& state) {
        if (state.step % 50 == 0 && state.rank == 0) {
            printf("Step %d: load balance = %.2f, most loaded = rank %d\n",
                   state.step, state.load_balance_ratio, state.most_loaded_rank);
        }
    });

    const float t_final = 0.2f;
    while (solver.time() < t_final) {
        solver.step();
    }

    // Statistiques finales
    if (solver.is_rank0()) {
        auto stats = solver.load_balance_stats();
        printf("\n=== Load Balance Statistics ===\n");
        printf("Final ratio: %.2f (target: < %.2f)\n",
               stats.final_ratio, stats.target_ratio);
        printf("Rebalances: %d\n", stats.num_rebalances);
    }
}

// ============================================================================
// EXAMPLE 10: LOAD BALANCING AVEC POIDS PAR NIVEAU
// ============================================================================

void example_10_level_weighted_lb() {
    auto solver = EulerSolverRK3::builder(1000, 500)
        .with_domain(0.0, 2.0, 0.0, 1.0)
        .with_initial_condition([](float x, float y) {
            using System = Euler2D<float>;
            float rho = 1.0f + 0.5f * std::exp(-50.0f * ((x-1.0)*(x-1.0) + (y-0.5)*(y-0.5)));
            return System::from_primitive({rho, 0.0f, 0.0f, 1.0f/1.4f}, 1.4f);
        })
        // Load balancng par niveau (cellules fines = plus coûteuses)
        .with_load_balancing<LevelWeightedLoadBalance<Euler2D<float>>>({
            .level_weight = 2.0f,  // Niveau l+1 coûte 2x plus
            .max_imbalance = 1.1f
        })
        .with_refinement_config(standard_amr<Euler2D<float>>())
        .build();

    solver.enable_auto_load_balance(true);

    solver.observers().on_mpi_progress([](const MPISolverState<float>& state) {
        if (state.step % 50 == 0 && state.rank == 0) {
            printf("Step %d: t=%.4f, %zu cells, load=%.2f\n",
                   state.step, state.time, state.global_cells, state.load_balance_ratio);
        }
    });

    const float t_final = 0.15f;
    while (solver.time() < t_final) {
        solver.step();
    }
}

// ============================================================================
// EXAMPLE 11: CUSTOM LOAD BALANCING (DEVICE-FRIENDLY)
// ============================================================================

void example_11_custom_load_balance() {
    using System = Euler2D<float>;
    using Real = float;

    // Fonction de coût personnalisée (Kokkos device-friendly)
    auto cost_lambda = KOKKOS_LAMBDA(
        const System::Conserved& U,
        const System::Primitive& q,
        Real grad_rho_x,
        Real grad_rho_y,
        int level
    ) -> Real {
        // Coût de base
        Real cost = 1.0f;

        // Bonus pour les gradients forts (zones actives)
        Real grad_mag = Kokkos::sqrt(grad_rho_x * grad_rho_x + grad_rho_y * grad_rho_y);
        if (grad_mag > 0.1f) {
            cost += 10.0f * grad_mag;
        }

        // Bonus pour le raffinement
        cost *= Kokkos::pow(2.0f, level);

        // Bonus pour les chocs (compression)
        Real div_v = 0.0f;  // TODO: calculer depuis q.u, q.v
        if (div_v < -0.5f) {
            cost *= 3.0f;
        }

        return cost;
    };

    auto solver = EulerSolverRK3::builder(1000, 500)
        .with_domain(0.0, 2.0, 0.0, 1.0)
        .with_initial_condition([](float x, float y) {
            return System::from_primitive({1.0f, 100.0f, 0.0f, 100000.0f}, 1.4f);
        })
        // Utiliser la fonction de coût personnalisée
        .with_load_balancing<CustomLoadBalance<System, decltype(cost_lambda)>>({
            .cost_func = cost_lambda,
            .max_imbalance = 1.15f
        })
        .with_refinement_config(standard_amr<System>())
        .build();

    solver.enable_auto_load_balance(true);

    const float t_final = 0.1f;
    while (solver.time() < t_final) {
        solver.step();
    }
}

// ============================================================================
// EXAMPLE 12: FULL-FEATURED MPI SIMULATION
// ============================================================================

void example_12_full_featured() {
    using System = Euler2D<float>;
    using Solver = EulerSolverRK3;

    auto solver = Solver::builder(1000, 500)
        .with_domain(0.0, 2.0, 0.0, 1.0)
        .with_initial_condition([](float x, float y) {
            // Écoulement Mach 2 avec obstacle cylindrique
            float mach = 2.0f;
            float gamma = 1.4f;
            float rho = 1.0f;
            float u = mach * std::sqrt(gamma);
            float v = 0.0f;
            float p = 1.0f / gamma;

            // Obstacle cylindrique au centre
            float dx = x - 0.5f;
            float dy = y - 0.5f;
            float r = std::sqrt(dx*dx + dy*dy);
            if (r < 0.1f) {
                u = 0.0f;
                v = 0.0f;
            }

            return System::from_primitive({rho, u, v, p}, gamma);
        })
        // Décomposition Metis pour géométrie complexe
        .with_decomposition<MetisDecomposition>({
            .geometry = MetisDecomposition::GeometryInput{
                .nx = 1000,
                .ny = 500,
                .halo_width = 2
            },
            .imbalance = 1.05f
        })
        .with_halo_width(2)
        // Communication asynchrone avec GPU-Direct si dispo
        .with_comm_mode(CommMode::GPUDirect)
        .with_auto_comm(true)
        // Observateurs: stats globales
        .with_observer_mode(ObserverMode::Reduced)
        // Load balancing par niveau
        .with_load_balancing<LevelWeightedLoadBalance<System>>({
            .level_weight = 2.0f,
            .max_imbalance = 1.1f
        })
        // AMR standard
        .with_refinement_config(standard_amr<System>())
        .build();

    // Activer le load balancing
    solver.enable_auto_load_balance(true);

    // Observateurs complets
    solver.observers().on_mpi_progress([](const MPISolverState<float>& state) {
        if (state.step % 10 == 0 && state.rank == 0) {
            printf("Step %d: t=%.4f, dt=%.5f\n", state.step, state.time, state.dt);
            printf("  Cells: %zu total, %.2f load balance\n",
                   state.global_cells, state.load_balance_ratio);
            printf("  Density (global): min=%.4f, max=%.4f, avg=%.4f\n",
                   state.global_min_rho, state.global_max_rho, state.global_avg_rho);
            printf("  Comm: %.2f ms, overlap: %.1f%%, most_loaded: rank %d\n",
                   state.last_comm_time * 1000.0,
                   state.comm_overlap_ratio * 100.0,
                   state.most_loaded_rank);
        }
    });

    solver.observers().on_remesh([](const MPISolverState<float>& state,
                                    size_t old_local, size_t new_local) {
        if (state.rank == 0) {
            printf("REMESH: %zu -> %zu cells (local), %zu total\n",
                   old_local, new_local, state.global_cells);
        }
    });

    solver.observers().on_load_balance([](const MPISolverState<float>& state,
                                         float old_ratio, float new_ratio) {
        if (state.rank == 0) {
            printf("LOAD BALANCE: %.2f -> %.2f (target: < 1.1)\n",
                   old_ratio, new_ratio);
        }
    });

    // Simulation principale
    const float t_final = 0.2f;
    while (solver.time() < t_final) {
        solver.step();
    }

    // Statistiques finales
    if (solver.is_rank0()) {
        auto stats = solver.load_balance_stats();
        printf("\n╔════════════════════════════════════════════════════════╗\n");
        printf("║           FINAL STATISTICS (Rank 0)                   ║\n");
        printf("╠════════════════════════════════════════════════════════╣\n");
        printf("║ Total cells:      %40zu ║\n", stats.total_cells);
        printf("║ Load balance:     %40.2f ║\n", stats.final_ratio);
        printf("║ Rebalances:       %40d ║\n", stats.num_rebalances);
        printf("║ Total comm time:  %40.2f s ║\n", stats.total_comm_time);
        printf("║ Comm ratio:       %40.1f%% ║\n", stats.comm_ratio * 100.0);
        printf("║ Steps:            %40d ║\n", static_cast<int>(t_final / 0.001f));
        printf("╚════════════════════════════════════════════════════════╝\n");
    }
}

// ============================================================================
// EXAMPLE 13: VTK OUTPUT AVEC MPI
// ============================================================================

void example_13_vtk_output() {
    auto solver = EulerSolverRK3::builder(1000, 500)
        .with_domain(0.0, 2.0, 0.0, 1.0)
        .with_initial_condition([](float x, float y) {
            using System = Euler2D<float>;
            return System::from_primitive({1.0f, 100.0f, 0.0f, 100000.0f}, 1.4f);
        })
        // Seul rank 0 écrit les fichiers VTK
        .with_observer_mode(ObserverMode::Rank0Only)
        .build();

    int frame = 0;
    solver.observers().add_callback(SolverEvent::StepEnd,
        [&frame, &solver](SolverEvent, const auto& state) {
            if (state.step % 100 == 0 && solver.is_rank0()) {
                char filename[256];
                snprintf(filename, sizeof(filename), "output_mpi/frame_%04d.vtk", frame++);
                // Gather automatique sur rank 0 + écriture VTK
                solver.write_vtk(filename);
                printf("VTK écrit: %s\n", filename);
            }
        });

    const float t_final = 0.5f;
    while (solver.time() < t_final) {
        solver.step();
    }

    if (solver.is_rank0()) {
        printf("Nombre de frames VTK: %d\n", frame);
    }
}

// ============================================================================
// EXAMPLE 14: CHECKPOINT/RESTART AVEC MPI
// ============================================================================

void example_14_checkpoint_restart() {
    auto solver = EulerSolverRK3::builder(1000, 500)
        .with_domain(0.0, 2.0, 0.0, 1.0)
        .with_initial_condition([](float x, float y) {
            using System = Euler2D<float>;
            return System::from_primitive({1.0f, 100.0f, 0.0f, 100000.0f}, 1.4f);
        })
        .build();

    // Auto-checkpoint: chaque rank écrit sa portion
    solver.set_auto_checkpoint(100, "mpi_checkpoint");

    // Observer pour les checkpoints
    solver.observers().add_callback(SolverEvent::OutputWritten,
        [](SolverEvent, const auto& state) {
            if (state.rank == 0) {
                printf("Checkpoint écrit à t=%.4f (tous les ranks)\n", state.time);
            }
        });

    // Simulation partielle
    const float t_final = 0.1f;
    while (solver.time() < t_final) {
        solver.step();
    }

    // Checkpoint manuel
    solver.write_checkpoint("final_mpi.bin", CheckpointFormat::Binary);

    // Restart: chaque rank lit sa portion
    auto solver2 = EulerSolverRK3::builder(1000, 500)
        .with_domain(0.0, 2.0, 0.0, 1.0)
        .build();

    solver2.read_checkpoint("final_mpi.bin", CheckpointFormat::Binary);

    if (solver2.is_rank0()) {
        printf("Restart réussi à t=%.4f\n", solver2.time());
    }

    // Continuer
    const float t_final2 = 0.15f;
    while (solver2.time() < t_final2) {
        solver2.step();
    }
}

// ============================================================================
// EXAMPLE 15: RÉDUCTIONS MPI PERSONNALISÉES
// ============================================================================

void example_15_custom_reductions() {
    auto solver = EulerSolverRK3::builder(1000, 500)
        .with_domain(0.0, 2.0, 0.0, 1.0)
        .with_initial_condition([](float x, float y) {
            using System = Euler2D<float>;
            float rho = 1.0f + 0.5f * std::sin(10.0f * x);
            return System::from_primitive({rho, 100.0f, 0.0f, 100000.0f}, 1.4f);
        })
        .build();

    solver.observers().on_mpi_progress([](const MPISolverState<float>& state) {
        if (state.step % 50 == 0) {
            // Réductions personnalisées
            float local_min_rho = state.local_min_rho;
            float global_min_rho;
            MPI_Allreduce(&local_min_rho, &global_min_rho, 1, MPI_FLOAT, MPI_MIN,
                         MPI_COMM_WORLD);

            if (state.rank == 0) {
                printf("Step %d: global min density = %.4f\n",
                       state.step, global_min_rho);
            }
        }
    });

    const float t_final = 0.1f;
    while (solver.time() < t_final) {
        solver.step();
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    // Initialisation Kokkos (MPI sera initialisé automatiquement par Subsetix)
    Kokkos::initialize(argc, argv);

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║     COLLECTION D'EXEMPLES FVD AVEC MPI                         ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");

    // Choisir l'exemple à exécuter
    int example = 1;
    if (argc > 1) {
        example = std::atoi(argv[1]);
    }

    switch (example) {
        case 1:  example_1_mpi_minimal(); break;
        case 2:  example_2_cartesian_2d(); break;
        case 3:  example_3_metis_decomposition(); break;
        case 4:  example_4_observer_rank0_only(); break;
        case 5:  example_5_observer_all_ranks(); break;
        case 6:  example_6_observer_reduced(); break;
        case 7:  example_7_async_communication(); break;
        case 8:  example_8_manual_communication(); break;
        case 9:  example_9_load_balancing(); break;
        case 10: example_10_level_weighted_lb(); break;
        case 11: example_11_custom_load_balance(); break;
        case 12: example_12_full_featured(); break;
        case 13: example_13_vtk_output(); break;
        case 14: example_14_checkpoint_restart(); break;
        case 15: example_15_custom_reductions(); break;
        default:
            printf("Exemple inconnu: %d\n", example);
            printf("Usage: %s [example_number 1-15]\n", argv[0]);
            break;
    }

    Kokkos::finalize();
    return 0;
}
