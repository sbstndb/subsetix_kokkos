/**
 * @file fvd_simulation_examples.cpp
 *
 * @brief Collection d'exemples de simulation utilisant l'API FVD haut niveau
 *
 * Ce fichier montre la simplicité, la généricité et la puissance de l'API
 * FVD (Finite Volume Dynamics) de Subsetix.
 */

#include <subsetix/fvd/fvd_integrators.hpp>
#include <subsetix/fvd/output/field_view.hpp>
#include <cstdio>
#include <cmath>
#include <iostream>

using namespace subsetix::fvd;
using namespace subsetix::fvd::solver;
using namespace subsetix::fvd::time;
using namespace subsetix::fvd::boundary;
using namespace subsetix::fvd::amr;

// ============================================================================
// EXAMPLE 1: SIMULATION LA PLUS SIMPLE - Advection 2D
// ============================================================================

void example_1_simplest_advection() {
    printf("\n=== EXEMPLE 1: Advection 2D la plus simple ===\n\n");

    // Un seul include, quelques lignes de code
    using Real = float;
    using Solver = AdaptiveSolver<
        Advection2D<Real>,                    // Système: advection
        reconstruction::NoReconstruction,      // 1er ordre espace
        flux::RusanovFlux,                     // Flux Rusanov
        time::ForwardEuler<Real>               // 1er ordre temps
    >;

    // Construction avec builder pattern
    auto solver = Solver::builder(100, 50)
        .with_domain(0.0, 1.0, 0.0, 0.5)
        .with_initial_condition([](Real x, Real y) {
            // Condition initiale: gaussienne au centre
            Real r2 = (x-0.5)*(x-0.5) + (y-0.25)*(y-0.25);
            return typename Advection2D<Real>::Conserved{
                std::exp(-10.0 * r2)  // u(x,y,0)
            };
        })
        .build();

    // Simulation
    while (solver.time() < 0.1) {
        solver.step();
    }

    printf("Advection terminée: t = %.5f\n", solver.time());
}

// ============================================================================
// EXAMPLE 2: ÉCOULEMENT EULER AUTOUR D'UN CYLINDRE (Mach 2)
// ============================================================================

void example_2_euler_cylinder() {
    printf("\n=== EXEMPLE 2: Écoulement Mach 2 autour d'un cylindre ===\n\n");

    using Real = float;
    using System = Euler2D<Real>;
    using Solver = EulerSolverRK3;  // Type alias: RK3 + Rusanov + 1st order

    // Paramètres du problème
    const int nx = 200, ny = 80;
    const Real x_min = 0.0, x_max = 2.0;
    const Real y_min = 0.0, y_max = 0.8;
    const Real gamma = 1.4f;
    const Real mach = 2.0f;

    // Condition initiale: écoulement uniforme Mach 2
    auto mach2_cylinder = [mach, gamma](Real x, Real y) {
        Real rho = 1.0f;
        Real u = mach * std::sqrt(gamma);  // Mach number
        Real v = 0.0f;
        Real p = 1.0f / gamma;

        return System::from_primitive({rho, u, v, p}, gamma);
    };

    // Construction du solveur
    auto solver = Solver::builder(nx, ny)
        .with_domain(x_min, x_max, y_min, y_max)
        .with_initial_condition(mach2_cylinder)
        .with_gamma(gamma)
        .build();

    // Conditions aux limites: inflow gauche, outflow droite
    auto inflow_state = System::Primitive{1.0f, mach*std::sqrt(gamma), 0.0f, 1.0f/gamma};
    auto bc = BoundaryConfigBuilder<System>::inflow_outflow(inflow_state, gamma);
    solver.set_boundary_conditions(bc);

    // Ajout d'un obstacle (cylindre) - zone d'exclusion du raffinement
    // (serait implémenté via un masque géométrique)

    // Boucle de simulation
    const Real t_final = 0.2f;
    int step = 0;
    while (solver.time() < t_final) {
        solver.step();
        step++;
        if (step % 10 == 0) {
            printf("Step %d: t = %.4f, dt = %.5f\n", step, solver.time(), solver.dt());
        }
    }

    printf("Simulation terminée: %d steps, t_final = %.4f\n", step, solver.time());
}

// ============================================================================
// EXAMPLE 3: AVEC AMR ADAPTATIF ET OBSERVERS
// ============================================================================

void example_3_with_amr_and_observers() {
    printf("\n=== EXEMPLE 3: Simulation avec AMR et Observers ===\n\n");

    using Real = float;
    using System = Euler2D<Real>;
    using Solver = EulerSolverRK3;

    auto solver = Solver::builder(100, 50)
        .with_domain(0.0, 2.0, 0.0, 0.8)
        .with_initial_condition([](Real x, Real y) {
            // Condition de shock tube initiale
            Real rho = (x < 1.0) ? 1.0f : 0.125f;
            Real p = (x < 1.0) ? 1.0f : 0.1f;
            return System::from_primitive({rho, 0.0f, 0.0f, p}, 1.4f);
        })
        .build();

    // --- Configuration AMR standard ---
    auto amr_config = standard_amr<System>();
    solver.set_refinement_config(amr_config);

    // --- Observers: monitoring de la simulation ---

    // Observer 1: Progression simple
    solver.observers().on_progress([](const auto& state) {
        if (state.step % 10 == 0) {
            printf("Step %d: t=%.4f, dt=%.5f, %zu cells, %d levels\n",
                   state.step, state.time, state.dt, state.total_cells, state.max_level + 1);
        }
    });

    // Observer 2: Remeshing events
    solver.observers().on_remesh([](const auto& state, size_t old_c, size_t new_c) {
        double change = 100.0 * (static_cast<double>(new_c) - static_cast<double>(old_c))
                        / static_cast<double>(old_c);
        printf("  REMESH: %zu -> %zu cells (%.1f%% change)\n", old_c, new_c, change);
    });

    // Observer 3: Logging CSV pour post-traitement
    auto csv_logger = Observers::csv_logger<Real>("simulation_data.csv");
    solver.observers().add_callback(SolverEvent::StepEnd, csv_logger);

    // Observer 4: Temps d'exécution
    solver.observers().add_callback(SolverEvent::StepEnd,
        Observers::time_logger<Real>()
    );

    // Observer 5: Fin de simulation
    solver.observers().add_callback(SolverEvent::SimulationEnd,
        [](SolverEvent, const SolverState<Real>& state) {
            printf("\n=== Simulation terminée ===\n");
            printf("Temps final: %.4f\n", state.time);
            printf("Steps totaux: %d\n", state.step);
            printf("Temps wall: %.2f s\n", state.wall_time);
        }
    );

    // Simulation
    const Real t_final = 0.15f;
    while (solver.time() < t_final) {
        solver.step();
    }
}

// ============================================================================
// EXAMPLE 4: CONDITIONS AUX LIMITES DÉPENDANTES DU TEMPS
// ============================================================================

void example_4_time_dependent_bcs() {
    printf("\n=== EXEMPLE 4: Conditions aux limites dépendantes du temps ===\n\n");

    using Real = float;
    using System = Euler2D<Real>;
    using Solver = EulerSolverRK3;

    auto solver = Solver::builder(100, 50)
        .with_domain(0.0, 2.0, 0.0, 0.8)
        .with_initial_condition([](Real x, Real y) {
            return System::from_primitive({1.0f, 0.0f, 0.0f, 1.0f/1.4f}, 1.4f);
        })
        .build();

    // --- BCs dépendantes du temps ---

    // 1. Inlet sinusoïdal à gauche
    auto sinusoidal_inlet = boundary::sinusoidal_inlet<System>(
        1.0f,    // rho0
        100.0f,  // u0
        2.0f * 3.14159f  // frequency = 1 Hz
    );
    solver.set_time_dependent_bc("left", sinusoidal_inlet);

    // 2. Outlet avec onde pulsative
    auto pulsating_outlet = boundary::pulsating_inlet<System>(
        1.0f,    // rho0
        -50.0f,  // u0 (outflow)
        5.0f     // frequency
    );
    solver.set_time_dependent_bc("right", pulsating_outlet);

    // 3. Inlet avec rampe linéaire en bas
    auto ramp_inlet = boundary::linear_ramp<System>(
        1.5f,    // rho0
        0.0f,    // u0
        0.5f     // amplitude
    );
    solver.set_time_dependent_bc("bottom", ramp_inlet);

    // Monitoring de l'entrée
    solver.observers().on_progress([](const auto& state) {
        if (state.step % 20 == 0) {
            printf("t=%.4f: Inlet rho = %.4f (sinusoidal)\n",
                   state.time, 1.0f * (1.0f + 0.1f * std::sin(2.0f * 3.14159f * state.time)));
        }
    });

    // Simulation
    while (solver.time() < 1.0) {
        solver.step();
    }
}

// ============================================================================
// EXAMPLE 5: TERMES SOURCES PERSONNALISÉS
// ============================================================================

void example_5_custom_source_terms() {
    printf("\n=== EXEMPLE 5: Termes sources personnalisés ===\n\n");

    using Real = float;
    using System = Euler2D<Real>;
    using Solver = EulerSolverRK3;

    auto solver = Solver::builder(100, 50)
        .with_domain(0.0, 1.0, 0.0, 0.5)
        .with_initial_condition([](Real x, Real y) {
            return System::from_primitive({1.0f, 0.0f, 0.0f, 1.0f/1.4f}, 1.4f);
        })
        .build();

    // --- Source 1: Gravité ---
    solver.add_gravity(-9.81f);  // Gravité vers le bas (y)

    // --- Source 2: Chauffage dans une zone circulaire ---
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    solver.add_source([gamma = System::default_gamma](const Conserved& U,
                                                       const Primitive& q,
                                                       Real x, Real y, Real t) {
        // Zone de chauffage: cercle centré en (0.5, 0.25)
        Real dx = x - 0.5f;
        Real dy = y - 0.25f;
        Real r2 = dx*dx + dy*dy;

        // Source d'énergie: chauffage pulsatoire
        Real heat_source = 0.0f;
        if (r2 < 0.01f) {  // Rayon = 0.1
            heat_source = 1000.0f * (1.0f + 0.5f * std::sin(10.0f * t));
        }

        return Conserved{0, 0, 0, heat_source};  // Seulement l'énergie
    });

    // --- Source 3: Force de traînée ---
    solver.add_source([](const Conserved& U, const Primitive& q,
                         Real x, Real y, Real t) {
        Real Cd = 0.1f;  // Coefficient de traînée
        Real drag_x = -Cd * q.rho * q.u * std::sqrt(q.u*q.u + q.v*q.v);
        Real drag_y = -Cd * q.rho * q.v * std::sqrt(q.u*q.u + q.v*q.v);

        return Conserved{0, drag_x, drag_y, 0};
    });

    // --- Source 4: Réaction chimique (modèle simplifié) ---
    solver.add_source([](const Conserved& U, const Primitive& q,
                         Real x, Real y, Real t) {
        // Modèle Arrhenius simplifié: A * rho^2 * exp(-E/RT)
        Real A = 1e6f;
        Real E_RT = 10.0f;  // Température adimensionnelle inverse
        Real reaction_rate = A * q.rho * q.rho * std::exp(-E_RT * 1.4f / (q.p/q.rho));

        return Conserved{0, 0, 0, reaction_rate};  // Libération d'énergie
    });

    // Monitoring des termes sources
    solver.observers().on_progress([](const auto& state) {
        if (state.step % 50 == 0) {
            printf("t=%.4f: Gravité + Chauffage + Traînée + Réaction actifs\n", state.time);
        }
    });

    // Simulation
    while (solver.time() < 0.5) {
        solver.step();
    }
}

// ============================================================================
// EXAMPLE 6: CHECKPOINT/RESTART
// ============================================================================

void example_6_checkpoint_restart() {
    printf("\n=== EXEMPLE 6: Checkpoint/Restart ===\n\n");

    using Real = float;
    using System = Euler2D<Real>;
    using Solver = EulerSolverRK3;

    // --- Première partie: Simulation avec checkpoint ---

    auto solver = Solver::builder(100, 50)
        .with_domain(0.0, 2.0, 0.0, 0.8)
        .with_initial_condition([](Real x, Real y) {
            return System::from_primitive({1.0f, 100.0f, 0.0f, 100000.0f}, 1.4f);
        })
        .build();

    // Auto-checkpoint toutes les 100 étapes
    solver.set_auto_checkpoint(100, "sim_checkpoint");

    // Observer pour noter les checkpoints
    solver.observers().add_callback(SolverEvent::OutputWritten,
        [](SolverEvent, const SolverState<Real>& state) {
            printf("Checkpoint écrit à t=%.4f\n", state.time);
        }
    );

    // Simulation partielle
    printf("--- Première partie: t = 0 à 0.1 ---\n");
    while (solver.time() < 0.1) {
        solver.step();
    }

    // Checkpoint manuel final
    solver.write_checkpoint("final_checkpoint.bin", CheckpointFormat::Binary);
    printf("Checkpoint final écrit\n");

    // --- Deuxième partie: Restart ---

    printf("\n--- Deuxième partie: Restart ---\n");

    // Créer un nouveau solveur et charger le checkpoint
    auto solver2 = Solver::builder(100, 50)
        .with_domain(0.0, 2.0, 0.0, 0.8)
        .build();

    solver2.read_checkpoint("final_checkpoint.bin", CheckpointFormat::Binary);
    printf("Restart à t=%.4f\n", solver2.time());

    // Continuer la simulation
    while (solver2.time() < 0.2) {
        solver2.step();
    }

    printf("Simulation terminée: t=%.4f\n", solver2.time());
}

// ============================================================================
// EXAMPLE 7: ÉTUDE PARAMÉTRIQUE (MULTIPLE RUNS)
// ============================================================================

void example_7_parametric_study() {
    printf("\n=== EXEMPLE 7: Étude paramétrique ===\n\n");

    using Real = float;
    using System = Euler2D<Real>;
    using Solver = EulerSolverRK3;

    // Paramètres à varier
    std::vector<Real> mach_numbers = {0.5f, 1.0f, 2.0f, 3.0f};
    std::vector<Real> cfl_numbers = {0.3f, 0.5f, 0.8f};

    printf("Nombre de runs: %zu\n", mach_numbers.size() * cfl_numbers.size());

    // Boucle sur les paramètres
    for (Real mach : mach_numbers) {
        for (Real cfl : cfl_numbers) {
            printf("\n--- Run: Mach=%.1f, CFL=%.2f ---\n", mach, cfl);

            // Créer un solveur pour ce run
            auto solver = Solver::builder(100, 50)
                .with_domain(0.0, 2.0, 0.0, 0.8)
                .with_cfl(cfl)
                .with_initial_condition([mach](Real x, Real y) {
                    Real gamma = 1.4f;
                    Real rho = 1.0f;
                    Real u = mach * std::sqrt(gamma);
                    Real v = 0.0f;
                    Real p = 1.0f / gamma;
                    return System::from_primitive({rho, u, v, p}, gamma);
                })
                .build();

            // Mesure de performance
            auto start = std::chrono::high_resolution_clock::now();

            // Simulation
            const Real t_final = 0.05f;
            int steps = 0;
            while (solver.time() < t_final) {
                solver.step();
                steps++;
            }

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();

            printf("Résultat: %d steps, %.3f s, %.1f steps/s\n",
                   steps, elapsed, steps / elapsed);

            // Sauvegarder les résultats pour analyse
            char filename[256];
            snprintf(filename, sizeof(filename),
                     "results_mach_%.1f_cfl_%.2f.csv", mach, cfl);
            // Écrire les résultats...
        }
    }

    printf("\nÉtude paramétrique terminée\n");
}

// ============================================================================
// EXAMPLE 8: SOLVEUR PERSONNALISÉ (MUSCL + HLLC + RK4)
// ============================================================================

void example_8_custom_solver() {
    printf("\n=== EXEMPLE 8: Solveur personnalisé (MUSCL + HLLC + RK4) ===\n\n");

    using Real = float;
    using System = Euler2D<Real>;

    // Définir un solveur personnalisé avec:
    // - Reconstruction MUSCL (2ème ordre espace)
    // - Limiteur Minmod
    // - Flux HLLC (plus précis que Rusanov)
    // - Intégrateur RK4 (4ème ordre temps)
    using MySolver = AdaptiveSolver<
        System,
        reconstruction::MUSCL_Reconstruction<reconstruction::MinmodLimiter>,
        flux::HLLCFlux,
        time::ClassicRK4<Real>
    >;

    // Construction
    auto solver = MySolver::builder(200, 100)
        .with_domain(0.0, 1.0, 0.0, 0.5)
        .with_initial_condition([](Real x, Real y) {
            // Problème de Shu-Osher (interaction onde de choc / turbulence)
            Real rho = (x < 0.1f) ? 3.857143f : 1.0f + 0.2f * std::sin(20.0f * 3.14159f * x);
            Real p = (x < 0.1f) ? 10.33333f : 1.0f;
            return System::from_primitive({rho, 0.0f, 0.0f, p}, 1.4f);
        })
        .build();

    // Configuration AMR pour capturer les structures fines
    auto amr_config = standard_amr<System>();
    amr_config.config.max_level = 6;  // Plus de niveaux pour MUSCL+HLLC
    solver.set_refinement_config(amr_config);

    // Monitoring détaillé
    solver.observers().on_progress([](const auto& state) {
        printf("Step %d: t=%.5f, dt=%.6f, %zu cells (lev=%d)\n",
               state.step, state.time, state.dt, state.total_cells, state.max_level + 1);
    });

    // Validation stricte pour schéma de haut ordre
    ValidationConfig val_cfg;
    val_cfg.check_negative_density = true;
    val_cfg.check_negative_pressure = true;
    val_cfg.check_nan = true;
    val_cfg.max_mach = 50.0f;
    solver.set_validation(val_cfg);

    // Profiling activé
    solver.enable_profiling(true);

    // Simulation
    const Real t_final = 0.2f;
    while (solver.time() < t_final) {
        solver.step();
    }

    // Afficher le profil
    solver.print_profile();

    printf("Simulation MUSCL+HLLC+RK4 terminée\n");
}

// ============================================================================
// EXAMPLE 9: SIMULATION AVEC SORTIE VTK (Multi-Field Support)
// ============================================================================

void example_9_with_vtk_output() {
    printf("\n=== EXEMPLE 9: Simulation avec sortie VTK (Multi-Field) ===\n\n");

    using Real = float;
    using System = Euler2D<Real>;
    using Solver = EulerSolverRK3;

    auto solver = Solver::builder(100, 50)
        .with_domain(0.0, 2.0, 0.0, 0.8)
        .with_initial_condition([](Real x, Real y) {
            // Explosion ponctuelle au centre
            Real dx = x - 1.0f;
            Real dy = y - 0.4f;
            Real r = std::sqrt(dx*dx + dy*dy);
            Real rho = 1.0f + 2.0f * std::exp(-20.0f * r*r);
            Real p = 1.0f + 10.0f * std::exp(-20.0f * r*r);
            return System::from_primitive({rho, 0.0f, 0.0f, p}, 1.4f);
        })
        .build();

    // Observer pour sortie VTK périodique avec NOUVELLE API write_all_fields()
    // C'est la méthode recommandée car elle exporte TOUTES les variables conservatives
    // (rho, rhou, rhov, E) dans un seul fichier VTK
    int vtk_frame = 0;
    solver.observers().add_callback(SolverEvent::StepEnd,
        [&vtk_frame, &solver](SolverEvent, const SolverState<Real>& state) {
            if (state.step % 50 == 0) {
                // NOUVEAU: write_all_fields() exporte toutes les variables
                auto output = solver.get_output();

                char filename[256];
                snprintf(filename, sizeof(filename), "output/frame_%04d.vtk", vtk_frame++);

                // Méthode recommandée: toutes les variables dans un fichier
                VTKExporter::write_all_fields(output, filename, true);
                printf("VTK multi-champ écrit: %s (rho, rhou, rhov, E)\n", filename);
            }
        }
    );

    // Observer alternatif: utiliser write_legacy() pour un seul champ
    solver.observers().add_callback(SolverEvent::StepEnd,
        [&solver](SolverEvent, const SolverState<Real>& state) {
            if (state.step == 100) {
                // Exemple: exporter seulement la densité
                auto output = solver.get_output();
                VTKExporter::write_legacy(output, "output/rho_only.vtk", "rho", true);
                printf("VTK single-champ écrit: rho_only.vtk\n");
            }
        }
    );

    // Simulation
    while (solver.time() < 0.3) {
        solver.step();
    }

    printf("Nombre de frames VTK: %d\n", vtk_frame);
    printf("\nNOTE: write_all_fields() est recommandé pour la visualisation CFD complète\n");
    printf("      write_legacy() est utile pour exporter une seule variable\n");
}

// ============================================================================
// EXAMPLE 10: SIMULATION MULTI-PHYSIQUE COMPLEXE
// ============================================================================

void example_10_complex_multiphysics() {
    printf("\n=== EXEMPLE 10: Simulation multi-physique complexe ===\n\n");

    using Real = float;
    using System = Euler2D<Real>;
    using Solver = EulerSolverRK3;

    // Configuration: Chambre de combustion avec injection, gravité, AMR
    auto solver = Solver::builder(150, 75)
        .with_domain(0.0, 1.5, 0.0, 0.75)
        .with_initial_condition([](Real x, Real y) {
            Real rho = 1.0f;
            Real u = 10.0f * x;  // Écoulement initial
            Real v = 0.0f;
            Real p = 101325.0f;
            return System::from_primitive({rho, u, v, p}, 1.4f);
        })
        .build();

    // --- Conditions aux limites complexes ---

    // Injection pulsatoire à gauche
    auto injection = boundary::pulsating_inlet<System>(2.5f, 150.0f, 10.0f);
    solver.set_time_dependent_bc("left", injection);

    // Paroi réfléchissante en bas
    auto wall_bc = BoundaryConfigBuilder<System>::reflective_wall();
    solver.set_boundary_condition("bottom", wall_bc);

    // Conditions de sortie non-réfléchissante en haut et à droite
    auto outflow = BoundaryConfigBuilder<System>::non_reflecting_outflow();
    solver.set_boundary_condition("top", outflow);
    solver.set_boundary_condition("right", outflow);

    // --- Termes sources ---

    // Gravité (chambre verticale)
    solver.add_gravity(-9.81f);

    // Source de combustion (zone réactive)
    solver.add_source([](const auto& U, const auto& q, Real x, Real y, Real t) {
        // Zone de combustion: x > 0.5 et y < 0.3
        if (x > 0.5f && y < 0.3f) {
            Real T = 1.4f * q.p / q.rho;  // Température
            Real rate = 1e5f * q.rho * q.rho * std::exp(-5000.0f / T);
            return System::Conserved{0, 0, 0, rate};
        }
        return System::Conserved{0, 0, 0, 0};
    });

    // Refroidissement par les parois
    solver.add_source([](const auto& U, const auto& q, Real x, Real y, Real t) {
        Real cooling_rate = 0.0f;
        if (y < 0.01f || y > 0.74f) {  // Parois haut/bas
            Real h = 100.0f;  // Coefficient de transfert
            Real T_wall = 300.0f;
            Real T = 1.4f * q.p / q.rho;
            cooling_rate = -h * (T - T_wall);
        }
        return System::Conserved{0, 0, 0, cooling_rate};
    });

    // --- AMR adaptatif ---

    auto amr_config = standard_amr<System>();
    // Affiner la zone de combustion
    amr_config.add_exclusion_rectangle(0.5f, 1.5f, 0.0f, 0.3f, 3);  // Min level 3
    solver.set_refinement_config(amr_config);

    // --- Observers multi-capacités ---

    // Progression
    solver.observers().on_progress([](const auto& state) {
        printf("Step %d: t=%.4f, %zu cells, %d levels\n",
               state.step, state.time, state.total_cells, state.max_level + 1);
    });

    // Logging CSV complet
    auto csv_log = Observers::csv_logger<Real>("combustion_chamber.csv");
    solver.observers().add_callback(SolverEvent::StepEnd, csv_log);

    // Sortie VTK pour visualisation
    int vtk_frame = 0;
    solver.observers().add_callback(SolverEvent::StepEnd,
        [&vtk_frame, &solver](SolverEvent, const SolverState<Real>& state) {
            if (state.step % 100 == 0) {
                char filename[256];
                snprintf(filename, sizeof(filename), "combustion_%04d.vtk", vtk_frame++);
                solver.write_vtk(filename);
            }
        }
    );

    // Remesh reporting
    solver.observers().add_callback(SolverEvent::RemeshEnd,
        Observers::remesh_reporter<Real>()
    );

    // Checkpoint auto
    solver.set_auto_checkpoint(500, "combustion_checkpoint");

    // --- Validation stricte ---
    ValidationConfig val_cfg;
    val_cfg.check_negative_density = true;
    val_cfg.check_negative_pressure = true;
    val_cfg.check_nan = true;
    val_cfg.min_density = 0.01f;
    val_cfg.min_pressure = 1.0f;
    val_cfg.max_temperature = 5000.0f;
    solver.set_validation(val_cfg);

    // --- Simulation ---
    const Real t_final = 0.01f;  // 10 ms de temps physique
    while (solver.time() < t_final) {
        solver.step();
    }

    printf("\nSimulation multi-physique terminée:\n");
    printf("  - Durée: %.4f s\n", solver.time());
    printf("  - Steps: %d\n", static_cast<int>(solver.time() / solver.dt()));
    printf("  - Frames VTK: %d\n", vtk_frame);
}

// ============================================================================
// EXAMPLE 11: UTILISATION DU BUILDER NAMESPACE
// ============================================================================

void example_11_builder_namespace() {
    printf("\n=== EXEMPLE 11: Utilisation du namespace builder ===\n\n");

    using Real = float;
    using System = Euler2D<Real>;

    // Utiliser le namespace builder pour créer un solveur
    auto solver_builder = builder::create<System,
                                          reconstruction::NoReconstruction,
                                          flux::RusanovFlux,
                                          time::Kutta3<Real>>(100, 50);

    auto solver = solver_builder
        .with_domain(0.0, 1.0, 0.0, 0.5)
        .with_initial_condition([](Real x, Real y) {
            return System::from_primitive({1.0f, 0.0f, 0.0f, 1.0f/1.4f}, 1.4f);
        })
        .build();

    // Utiliser les fonctions standard
    auto amr = standard_amr<System>();
    solver.set_refinement_config(amr);

    auto dt_config = standard_adaptive_dt<Real>();
    solver.set_adaptive_time_stepping(dt_config);

    // Simulation
    while (solver.time() < 0.1) {
        solver.step();
    }

    printf("Simulation terminée avec builder namespace\n");
}

// ============================================================================
// EXAMPLE 12: ERROR HANDLING ET VALIDATION
// ============================================================================

void example_12_error_handling() {
    printf("\n=== EXEMPLE 12: Error handling et validation ===\n\n");

    using Real = float;
    using System = Euler2D<Real>;
    using Solver = EulerSolverRK3;

    auto solver = Solver::builder(100, 50)
        .with_domain(0.0, 1.0, 0.0, 0.5)
        .with_initial_condition([](Real x, Real y) {
            return System::from_primitive({1.0f, 0.0f, 0.0f, 1.0f/1.4f}, 1.4f);
        })
        .build();

    // Configuration de validation stricte
    ValidationConfig val_cfg;
    val_cfg.check_negative_density = true;
    val_cfg.check_negative_pressure = true;
    val_cfg.check_nan = true;
    val_cfg.check_inf = true;
    val_cfg.min_density = 0.001f;
    val_cfg.min_pressure = 0.001f;
    val_cfg.max_mach = 100.0f;
    val_cfg.max_temperature = 10000.0f;
    solver.set_validation(val_cfg);

    // Observer pour les erreurs
    solver.observers().add_callback(SolverEvent::Error,
        [](SolverEvent event, const SolverState<Real>& state) {
            printf("ERREUR détectée à t=%.4f, step=%d\n", state.time, state.step);
        }
    );

    // Observer pour les avertissements
    solver.observers().add_callback(SolverEvent::StepEnd,
        [](SolverEvent event, const SolverState<Real>& state) {
            if (state.residual_rho > 1.0f) {
                printf("AVERTISSEMENT: Residual élevé = %.2e\n", state.residual_rho);
            }
        }
    );

    // Simulation avec gestion des erreurs
    try {
        while (solver.time() < 0.1) {
            solver.step();
        }
    } catch (const std::exception& e) {
        printf("Exception capturée: %s\n", e.what());
        // Écrire checkpoint pour diagnostic
        solver.write_checkpoint("error_checkpoint.bin", CheckpointFormat::Binary);
        printf("Checkpoint de diagnostic écrit\n");
    }

    printf("Simulation terminée (avec ou sans erreur)\n");
}

// ============================================================================
// EXAMPLE 13: COMPARAISON DES SCHÉMAS DE FLUX (Rusanov vs HLLC vs Roe)
// ============================================================================

void example_13_flux_scheme_comparison() {
    printf("\n=== EXEMPLE 13: Comparaison des schémas de flux ===\n\n");

    using Real = float;
    using System = Euler2D<Real>;

    // Problème de test: tube à choc (Sod problem)
    auto sod_initial = [](Real x, Real y) {
        Real rho = (x < 0.5) ? 1.0f : 0.125f;
        Real p = (x < 0.5) ? 1.0f : 0.1f;
        return System::from_primitive({rho, 0.0f, 0.0f, p}, 1.4f);
    };

    // --- Solveur 1: Rusanov (diffusif, robuste) ---
    using SolverRusanov = AdaptiveSolver<System,
        reconstruction::NoReconstruction,
        flux::RusanovFlux,
        time::ForwardEuler<Real>>;

    auto solver_rusanov = SolverRusanov::builder(200, 50)
        .with_domain(0.0, 1.0, 0.0, 0.25)
        .with_initial_condition(sod_initial)
        .build();

    // --- Solveur 2: HLLC (moins diffusif, résout l'onde de contact) ---
    using SolverHLLC = AdaptiveSolver<System,
        reconstruction::NoReconstruction,
        flux::HLLCFlux,
        time::ForwardEuler<Real>>;

    auto solver_hllc = SolverHLLC::builder(200, 50)
        .with_domain(0.0, 1.0, 0.0, 0.25)
        .with_initial_condition(sod_initial)
        .build();

    // --- Solveur 3: Roe (le plus précis, avec fix d'entropie) ---
    using SolverRoe = AdaptiveSolver<System,
        reconstruction::NoReconstruction,
        flux::RoeFlux,
        time::ForwardEuler<Real>>;

    auto solver_roe = SolverRoe::builder(200, 50)
        .with_domain(0.0, 1.0, 0.0, 0.25)
        .with_initial_condition(sod_initial)
        .build();

    printf("Comparaison des schémas de flux:\n");
    printf("  - Rusanov: très diffusif, robuste\n");
    printf("  - HLLC: moins diffusif, résout l'onde de contact\n");
    printf("  - Roe: le plus précis, fix d'entropie pour les ondes de choc\n\n");

    // Simulation et export des résultats
    const Real t_final = 0.2f;

    while (solver_rusanov.time() < t_final) solver_rusanov.step();
    while (solver_hllc.time() < t_final) solver_hllc.step();
    while (solver_roe.time() < t_final) solver_roe.step();

    // Export pour comparaison
    auto out_rusanov = solver_rusanov.get_output();
    auto out_hllc = solver_hllc.get_output();
    auto out_roe = solver_roe.get_output();

    VTKExporter::write_all_fields(out_rusanov, "output/sod_rusanov.vtk", true);
    VTKExporter::write_all_fields(out_hllc, "output/sod_hllc.vtk", true);
    VTKExporter::write_all_fields(out_roe, "output/sod_roe.vtk", true);

    printf("Résultats exportés:\n");
    printf("  - output/sod_rusanov.vtk\n");
    printf("  - output/sod_hllc.vtk\n");
    printf("  - output/sod_roe.vtk\n");
    printf("\nRecommandation: Pour les simulations CFD, utiliser HLLC ou Roe\n");
    printf("                 pour une meilleure résolution des discontinuités.\n");
}

// ============================================================================
// EXAMPLE 14: COMPARAISON DES LIMITEURS MUSCL
// ============================================================================

void example_14_muscl_limiter_comparison() {
    printf("\n=== EXEMPLE 14: Comparaison des limiteurs MUSCL ===\n\n");

    using Real = float;
    using System = Euler2D<Real>;

    // Problème de Shu-Osher: interaction onde de choc / ondes d'entropie
    auto shu_osher = [](Real x, Real y) {
        Real rho = (x < 0.1) ? 3.857143f : 1.0f + 0.2f * std::sin(20.0f * 3.14159f * x);
        Real p = (x < 0.1) ? 10.33333f : 1.0f;
        return System::from_primitive({rho, 2.629369f, 0.0f, p}, 1.4f);
    };

    printf("Comparaison des limiteurs MUSCL (du plus au moins diffusif):\n");
    printf("  - Minmod: très robuste, mais trop diffusif\n");
    printf("  - MC (Monotonized Central): bon compromis\n");
    printf("  - Van Leer: symétrique, moins diffusif\n");
    printf("  - Superbee: le moins diffusif (peut générer des oscillations)\n\n");

    // --- Minmod ---
    using SolverMinmod = AdaptiveSolver<System,
        reconstruction::MUSCL_Reconstruction<reconstruction::MinmodLimiter>,
        flux::HLLCFlux,
        time::Kutta3<Real>>;

    auto solver_minmod = SolverMinmod::builder(200, 50)
        .with_domain(0.0, 1.0, 0.0, 0.125)
        .with_initial_condition(shu_osher)
        .build();

    // --- MC ---
    using SolverMC = AdaptiveSolver<System,
        reconstruction::MUSCL_Reconstruction<reconstruction::MCLimiter>,
        flux::HLLCFlux,
        time::Kutta3<Real>>;

    auto solver_mc = SolverMC::builder(200, 50)
        .with_domain(0.0, 1.0, 0.0, 0.125)
        .with_initial_condition(shu_osher)
        .build();

    // --- Van Leer ---
    using SolverVanLeer = AdaptiveSolver<System,
        reconstruction::MUSCL_Reconstruction<reconstruction::VanLeerLimiter>,
        flux::HLLCFlux,
        time::Kutta3<Real>>;

    auto solver_vl = SolverVanLeer::builder(200, 50)
        .with_domain(0.0, 1.0, 0.0, 0.125)
        .with_initial_condition(shu_osher)
        .build();

    // --- Superbee ---
    using SolverSuperbee = AdaptiveSolver<System,
        reconstruction::MUSCL_Reconstruction<reconstruction::SuperbeeLimiter>,
        flux::HLLCFlux,
        time::Kutta3<Real>>;

    auto solver_sb = SolverSuperbee::builder(200, 50)
        .with_domain(0.0, 1.0, 0.0, 0.125)
        .with_initial_condition(shu_osher)
        .build();

    // Simulation
    const Real t_final = 0.18f;

    while (solver_minmod.time() < t_final) solver_minmod.step();
    while (solver_mc.time() < t_final) solver_mc.step();
    while (solver_vl.time() < t_final) solver_vl.step();
    while (solver_sb.time() < t_final) solver_sb.step();

    // Export
    VTKExporter::write_all_fields(solver_minmod.get_output(), "output/shuosher_minmod.vtk");
    VTKExporter::write_all_fields(solver_mc.get_output(), "output/shuosher_mc.vtk");
    VTKExporter::write_all_fields(solver_vl.get_output(), "output/shuosher_vanleer.vtk");
    VTKExporter::write_all_fields(solver_sb.get_output(), "output/shuosher_superbee.vtk");

    printf("Résultats exportés pour comparaison des limiteurs MUSCL\n");
    printf("\nRecommandation: MC ou Van Leer pour la plupart des cas CFD\n");
    printf("                 Minmod pour les cas très robustes\n");
    printf("                 Superbee pour résoudre les structures fines (attention aux oscillations)\n");
}

// ============================================================================
// EXAMPLE 15: COMPARAISON DES INTÉGRATEURS DE TEMPS
// ============================================================================

void example_15_time_integrator_comparison() {
    printf("\n=== EXEMPLE 15: Comparaison des intégrateurs de temps ===\n\n");

    using Real = float;
    using System = Euler2D<Real>;

    // Test sur un problème advection d'un pulse gaussien
    auto gaussian_pulse = [](Real x, Real y) {
        Real dx = x - 0.5;
        Real dy = y - 0.25;
        Real r2 = dx*dx + dy*dy;
        Real rho = std::exp(-20.0 * r2);
        return System::from_primitive({rho, 1.0f, 0.0f, 1.0f/1.4f}, 1.4f);
    };

    printf("Comparaison des intégrateurs de temps:\n");
    printf("  - Forward Euler (1er ordre, 1 stage)\n");
    printf("  - Heun2 / RK2 (2ème ordre, 2 stages)\n");
    printf("  - Kutta3 / SSPRK3 (3ème ordre, 3 stages)\n");
    printf("  - ClassicRK4 (4ème ordre, 4 stages)\n\n");

    using Flux = flux::RusanovFlux;
    using Recon = reconstruction::NoReconstruction;

    // Create solvers with different time integrators
    using SolverEuler = AdaptiveSolver<System, Recon, Flux, time::ForwardEuler<Real>>;
    using SolverRK2 = AdaptiveSolver<System, Recon, Flux, time::Heun2<Real>>;
    using SolverRK3 = AdaptiveSolver<System, Recon, Flux, time::Kutta3<Real>>;
    using SolverSSPRK3 = AdaptiveSolver<System, Recon, Flux, time::SSPRK3<Real>>;
    using SolverRK4 = AdaptiveSolver<System, Recon, Flux, time::ClassicRK4<Real>>;

    auto solver_euler = SolverEuler::builder(100, 50)
        .with_domain(0.0, 2.0, 0.0, 0.5).with_initial_condition(gaussian_pulse).build();
    auto solver_rk2 = SolverRK2::builder(100, 50)
        .with_domain(0.0, 2.0, 0.0, 0.5).with_initial_condition(gaussian_pulse).build();
    auto solver_rk3 = SolverRK3::builder(100, 50)
        .with_domain(0.0, 2.0, 0.0, 0.5).with_initial_condition(gaussian_pulse).build();
    auto solver_ssprk3 = SolverSSPRK3::builder(100, 50)
        .with_domain(0.0, 2.0, 0.0, 0.5).with_initial_condition(gaussian_pulse).build();
    auto solver_rk4 = SolverRK4::builder(100, 50)
        .with_domain(0.0, 2.0, 0.0, 0.5).with_initial_condition(gaussian_pulse).build();

    const Real t_final = 0.5f;

    while (solver_euler.time() < t_final) solver_euler.step();
    while (solver_rk2.time() < t_final) solver_rk2.step();
    while (solver_rk3.time() < t_final) solver_rk3.step();
    while (solver_ssprk3.time() < t_final) solver_ssprk3.step();
    while (solver_rk4.time() < t_final) solver_rk4.step();

    // Export results
    VTKExporter::write_all_fields(solver_euler.get_output(), "output/integrator_euler.vtk");
    VTKExporter::write_all_fields(solver_rk2.get_output(), "output/integrator_rk2.vtk");
    VTKExporter::write_all_fields(solver_rk3.get_output(), "output/integrator_rk3.vtk");
    VTKExporter::write_all_fields(solver_ssprk3.get_output(), "output/integrator_ssprk3.vtk");
    VTKExporter::write_all_fields(solver_rk4.get_output(), "output/integrator_rk4.vtk");

    printf("Résultats exportés:\n");
    printf("  - output/integrator_euler.vtk (1st order)\n");
    printf("  - output/integrator_rk2.vtk (2nd order)\n");
    printf("  - output/integrator_rk3.vtk (3rd order)\n");
    printf("  - output/integrator_ssprk3.vtk (3rd order, SSP)\n");
    printf("  - output/integrator_rk4.vtk (4th order)\n");
    printf("\nRecommandation: SSPRK3 pour la stabilité forte, RK4 pour la précision\n");
}

// ============================================================================
// EXAMPLE 16: VTK MULTI-LEVEL AMR EXPORT
// ============================================================================

void example_16_multilevel_amr_vtk() {
    printf("\n=== EXEMPLE 16: Export VTK multi-niveau AMR ===\n\n");

    using Real = float;
    using System = Euler2D<Real>;
    using Solver = EulerSolverRK3;

    auto solver = Solver::builder(100, 50)
        .with_domain(0.0, 2.0, 0.0, 0.8)
        .with_initial_condition([](Real x, Real y) {
            // Deux explosions pour tester le multi-niveau
            Real dx1 = x - 0.5, dy1 = y - 0.4;
            Real r1 = std::sqrt(dx1*dx1 + dy1*dy1);
            Real dx2 = x - 1.5, dy2 = y - 0.4;
            Real r2 = std::sqrt(dx2*dx2 + dy2*dy2);
            Real rho = 1.0f + std::exp(-20.0 * r1*r1) + 0.5f * std::exp(-20.0 * r2*r2);
            Real p = 1.0f + 5.0f * std::exp(-20.0 * r1*r1);
            return System::from_primitive({rho, 0.0f, 0.0f, p}, 1.4f);
        })
        .build();

    // Configuration AMR pour générer plusieurs niveaux
    auto amr_config = standard_amr<System>();
    amr_config.config.max_level = 4;
    solver.set_refinement_config(amr_config);

    // Simulation courte pour générer de l'AMR
    for (int i = 0; i < 20; ++i) {
        solver.step();
    }

    // Récupérer tous les niveaux AMR
    auto all_levels = solver.get_all_levels();
    printf("Nombre de niveaux AMR: %zu\n", all_levels.size());

    // Récupérer la géométrie multi-niveau
    // Note: Ceci nécessite que AdaptiveSolver expose multilevel_geometry
    // Pour cet exemple, nous montrons l'API conceptuelle

    printf("\nExport multi-niveau AMR:\n");
    printf("  - write_multilevel() exporte tous les niveaux dans un seul VTK\n");
    printf("  - Les cellules ont des tailles physiques différentes (dx, dy)\n");
    printf("  - Un champ scalaire 'Level' indique le niveau de raffinement\n");

    // Pour l'export effectif, utiliser:
    // VTKExporter::write_multilevel(all_levels, multilevel_geo, "output/multilevel.vtk", true);

    // Export chaque niveau séparément pour comparaison
    for (std::size_t level = 0; level < all_levels.size(); ++level) {
        char filename[256];
        snprintf(filename, sizeof(filename), "output/level_%zu.vtk", level);
        VTKExporter::write_all_fields(all_levels[level], filename, true);
        printf("  Niveau %zu: %s\n", level, filename);
    }

    printf("\nNOTE: Pour visualiser correctement l'AMR dans ParaView:\n");
    printf("      - Utilisez le champ 'Level' pour colorer les cellules par niveau\n");
    printf("      - Ou utilisez 'Threshold' pour isoler un niveau spécifique\n");
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  COLLECTION D'EXEMPLES FVD - Subsetix Finite Volume API       ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");

    // Lancer les exemples (commentez/décommentez selon vos besoins)
    example_1_simplest_advection();
    // example_2_euler_cylinder();
    // example_3_with_amr_and_observers();
    // example_4_time_dependent_bcs();
    // example_5_custom_source_terms();
    // example_6_checkpoint_restart();
    // example_7_parametric_study();
    // example_8_custom_solver();
    // example_9_with_vtk_output();
    // example_10_complex_multiphysics();
    // example_11_builder_namespace();
    // example_12_error_handling();

    // NOUVEAUX EXEMPLES (VTK multi-champ, comparaisons de schémas)
    // example_13_flux_scheme_comparison();   // Rusanov vs HLLC vs Roe
    // example_14_muscl_limiter_comparison(); // Minmod vs MC vs Van Leer vs Superbee
    // example_15_time_integrator_comparison(); // Euler vs RK2 vs RK3 vs SSPRK3 vs RK4
    // example_16_multilevel_amr_vtk();       // Export VTK multi-niveau AMR

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  Tous les exemples terminés                                    ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    Kokkos::finalize();
    return 0;
}
