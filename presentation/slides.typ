// ============================================
// SUBSETIX KOKKOS - PRÉSENTATION
// ============================================

#set page(
  paper: "presentation-16-9",
  margin: (x: 1.2cm, y: 1cm),
  numbering: "1 / 1",
  footer: context [
    #set text(size: 10pt, fill: rgb("#7f8c8d"))
    #h(1fr)
    #counter(page).display("1 / 1", both: true)
  ]
)

#set text(
  font: "DejaVu Sans",
  size: 16pt,
)

// Couleurs HPC@Maths
#let hpc-dark = rgb("#003957")      // Bleu foncé (H, P, C)
#let hpc-medium = rgb("#046D98")    // Bleu moyen (Maths)
#let hpc-light = rgb("#5AA0BB")     // Bleu clair (spirale)

#let accent = hpc-medium
#let dark = hpc-dark
#let light-gray = rgb("#ecf0f1")
#let green = rgb("#27ae60")
#let orange = rgb("#e67e22")

// Slide helper
#let slide(title: none, body) = {
  pagebreak(weak: true)
  if title != none {
    align(left, text(size: 24pt, weight: "bold", fill: dark, title))
    line(length: 100%, stroke: 2pt + accent)
    v(0.3em)
  }
  body
}

// Title slide helper
#let title-slide(title, subtitle: none, author: none, affiliation: none, date: none, logo: none) = {
  set page(
    fill: dark,
    footer: context [
      #set text(size: 10pt, fill: white.transparentize(50%))
      #h(1fr)
      #counter(page).display("1 / 1", both: true)
    ]
  )
  set text(fill: white)

  // Logo en bas à droite
  if logo != none {
    place(bottom + right, dx: -0.3cm, dy: -0.8cm)[
      #box(
        fill: white,
        inset: 0.3em,
        radius: 4pt,
        image(logo, width: 3.5cm)
      )
    ]
  }

  align(center + horizon)[
    #text(size: 36pt, weight: "bold", title)
    #if subtitle != none {
      v(0.3em)
      text(size: 20pt, fill: hpc-light, subtitle)
    }
    #if author != none {
      v(1.5em)
      text(size: 16pt, author)
    }
    #if affiliation != none {
      v(0.3em)
      text(size: 14pt, style: "italic", affiliation)
    }
    #if date != none {
      v(0.5em)
      text(size: 14pt, date)
    }
  ]
}

// Section slide
#let section-slide(title) = {
  pagebreak(weak: true)
  set page(
    fill: accent,
    footer: context [
      #set text(size: 10pt, fill: white.transparentize(30%))
      #h(1fr)
      #counter(page).display("1 / 1", both: true)
    ]
  )
  set text(fill: white)
  align(center + horizon)[
    #text(size: 36pt, weight: "bold", title)
  ]
}

// Code block style
#show raw.where(block: true): block.with(
  fill: light-gray,
  inset: 8pt,
  radius: 4pt,
  width: 100%,
)

// ============================================
// SLIDE 1: TITRE
// ============================================
#title-slide(
  "Algèbre d'Ensembles sur Représentation CSR d'Intervalles",
  subtitle: "Implémentation Haute Performance avec Kokkos",
  author: "Sébastien DUBOIS",
  affiliation: "Équipe HPC@Maths",
  date: "Décembre 2025",
  logo: "logo_hpc.png",
)

// ============================================
// SLIDE 2: PLAN
// ============================================
#slide(title: "Plan")[
  #set text(size: 15pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      == I. Contexte
      1. Calcul GPU & Kokkos

      #v(0.5em)
      == II. Représentation Sparse
      2. Intervalles et CSR
      3. Exemple de maillage 2D sparse

      #v(0.5em)
      == III. Structures de Données
      4. IntervalSet2D
      5. Field2D
      6. Workspace & AMR
    ],
    [
      == IV. Algorithmes
      7. Constructeurs de géométrie
      8. Algèbre ensembliste
      9. Opérations sur champs
      10. Morphologie & AMR

      #v(0.5em)
      == V. Démonstration
      11. Mach2 Cylinder (AMR multi-niveaux)

      #v(0.5em)
      == VI. Annexes
      - Évolution du projet
      - Pourquoi Kokkos ?
      - Méthodologie de développement
    ]
  )
]

// ============================================
// SECTION: GPU & KOKKOS
// ============================================
#section-slide("I. Contexte : GPU & Kokkos")

// ============================================
// SLIDE 3: GPU & CUDA Essentials (condensé)
// ============================================
#slide(title: "Calcul GPU — L'Essentiel")[
  #set text(size: 12pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == Pourquoi le GPU ?
      #table(
        columns: (auto, auto, auto),
        inset: 5pt,
        align: center,
        [*Aspect*], [*CPU*], [*GPU*],
        [Cœurs], [4-64], [*milliers*],
        [Bande passante], [~100 GB/s], [*~1-2 TB/s*],
        [Paradigme], [Séquentiel], [*SIMT*],
      )

      == Modèle CUDA
      ```
      Grid ──► Blocks ──► Threads (32 = 1 warp)
                │
                └── Shared Memory (rapide)
      ```

      ```cpp
      // Lancement kernel
      kernel<<<numBlocks, threadsPerBlock>>>(args);
      ```
    ],
    [
      == Hiérarchie mémoire
      #set text(size: 11pt)
      ```
      ┌─────────────────────────────────┐
      │         HOST (CPU + RAM)        │
      └───────────────┬─────────────────┘
                      │ PCIe 5.0: ~64 GB/s
                      │ Latence: ~1-2 µs
      ┌───────────────▼─────────────────┐
      │         DEVICE (GPU)            │
      │  Global Memory ──► L2 ──► L1    │
      │      (GB)         (MB)   (KB)   │
      │           ▼                     │
      │    Shared Memory (par block)    │
      │    Registres (par thread)       │
      └─────────────────────────────────┘
      ```

      #box(fill: rgb("#fff3cd"), inset: 0.4em, radius: 4pt)[
        *Clés performance* :
        - Minimiser transferts CPU↔GPU
        - Maximiser occupancy (threads actifs)
        - Accès mémoire *coalesced*
      ]
    ]
  )
]

// ============================================
// SLIDE 4: Kokkos Introduction
// ============================================
#slide(title: "Kokkos — Portabilité Performance")[
  #set text(size: 12pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == Le problème
      - CUDA = NVIDIA only
      - OpenMP = CPU only (GPU limité)
      - HIP = AMD only
      - Réécrire pour chaque plateforme ?

      #v(0.3em)
      == La solution : Kokkos
      ```cpp
      // Même code pour CPU et GPU !
      Kokkos::parallel_for(n,
        KOKKOS_LAMBDA(int i) {
          data[i] = compute(i);
        });
      ```

      #box(fill: rgb("#d4edda"), inset: 0.4em, radius: 4pt)[
        *Un code source* → compile pour : \
        OpenMP, CUDA, HIP, SYCL, Serial
      ]
    ],
    [
      == Abstractions clés
      ```cpp
      // Views (tableaux portables)
      Kokkos::View<double*> data("data", n);

      // Parallel patterns
      parallel_for(n, lambda);      // map
      parallel_reduce(n, lambda, r); // reduce
      parallel_scan(n, lambda);      // scan

      // Memory spaces
      HostSpace, CudaSpace, HIPSpace...

      // Execution spaces
      OpenMP, Cuda, HIP, Serial...
      ```

      == Pourquoi pour ce projet ?
      - *Fiabilité* : code testé sur CPU et GPU
      - *Maintenance* : un seul code à maintenir
      - *Écosystème* : Trilinos, Sandia Labs
      - *Std algorithms* : transform, copy, scan...
    ]
  )
]

// ============================================
// SECTION: SPARSE REPRESENTATION
// ============================================
#section-slide("II. Représentation Sparse")

// ============================================
// SLIDE 5: Exemple Mesh 2D Sparse
// ============================================
#slide(title: "Exemple : Maillage 2D Sparse avec Intervalles")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == Géométrie "Smiley" :-)
      #align(center)[
        #box(stroke: 1pt + dark, inset: 0.5em, radius: 4pt)[
          ```
          Y
          9│ . . . . . . . . . .    (vide)
          8│ . . . . . . . . . .    (vide)
          7│ . . ▓ ▓ . . ▓ ▓ . .    YEUX
          6│ . . ▓ ▓ . . ▓ ▓ . .    YEUX
          5│ . . . . . . . . . .    (TROU)
          4│ . . . . . . . . . .    (TROU)
          3│ . ▓ ▓ . . . . ▓ ▓ .    SOURIRE
          2│ . . ▓ ▓ . . ▓ ▓ . .    SOURIRE
          1│ . . . ▓ ▓ ▓ ▓ . . .    SOURIRE
          0│ . . . . . . . . . .    (vide)
           └──────────────────── X
             0 1 2 3 4 5 6 7 8 9

          ▓ = actif    TROU Y=4,5
          ```
        ]
      ]
    ],
    [
      == Représentation CSR
      ```cpp
      // 5 lignes, TROU Y=4,5
      row_keys = [1, 2, 3, 6, 7]  // saute 4,5!
      num_rows = 5

      // Lignes avec 1 ou 2 intervalles
      row_ptr = [0, 1, 3, 5, 7, 9]

      intervals = [
        {3, 7},        // Y=1: sourire bas
        {2, 4}, {6, 8},// Y=2: sourire épais
        {1, 3}, {7, 9},// Y=3: sourire coins
        {2, 4}, {6, 8},// Y=6: YEUX bas
        {2, 4}, {6, 8},// Y=7: YEUX haut
      ]
      num_intervals = 9

      cell_offsets = [0,4,6,8,10,12,14,16,18,20]
      total_cells = 20
      ```

      == Complexité mémoire
      *O(R + I)* où :
      - R = nb lignes Y occupées
      - I = nb intervalles

      #box(fill: rgb("#d4edda"), inset: 0.4em, radius: 4pt)[
        Dense : O(W × H) \
        CSR : O(R + I) ≪ O(W × H)
      ]
    ]
  )
]

// ============================================
// SECTION: DATA STRUCTURES
// ============================================
#section-slide("III. Structures de Données")

// ============================================
// SLIDE 7: IntervalSet2D Structure
// ============================================
#slide(title: "IntervalSet2D — Structure CSR Complète")[
  #set text(size: 12pt)
  #grid(
    columns: (1.1fr, 1fr),
    gutter: 1em,
    [
      == Définition C++
      ```cpp
      template<class MemorySpace>
      struct IntervalSet2D {
        // Coordonnées Y des lignes non-vides
        View<RowKey2D*> row_keys;  // [num_rows]

        // Index dans intervals[] pour chaque ligne
        View<size_t*> row_ptr;     // [num_rows + 1]

        // Tous les intervalles (contigus)
        View<Interval*> intervals; // [num_intervals]

        // Offset linéaire des cellules
        View<size_t*> cell_offsets;// [num_intervals]

        size_t total_cells;
        int num_rows;
        int num_intervals;
      };
      ```
    ],
    [
      == Invariants
      - `row_keys` trié par Y croissant
      - Intervalles triés par X dans chaque ligne
      - Pas de chevauchement entre intervalles
      - `row_ptr[r+1] - row_ptr[r]` = nb intervalles ligne r

      #v(0.3em)
      == Accès aux intervalles d'une ligne
      ```cpp
      // Intervalles de la ligne r
      int begin = row_ptr[r];
      int end   = row_ptr[r + 1];
      for (int i = begin; i < end; i++) {
        Interval iv = intervals[i];
        // Traiter [iv.begin, iv.end)
      }
      ```

      #box(fill: rgb("#e8f4f8"), inset: 0.4em, radius: 4pt)[
        *Template MemorySpace* : Device ou Host
      ]
    ]
  )
]

// ============================================
// SLIDE 8: Field2D
// ============================================
#slide(title: "Field2D — Champ sur Géométrie Creuse")[
  #set text(size: 12pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      == Définition
      Associe une *valeur* à chaque cellule

      ```cpp
      template<class T, class MemorySpace>
      struct Field2D {
        IntervalSet2DView geometry;
        View<T*> values;  // [total_cells]

        // Accès à une valeur
        T& at(int interval_idx, Coord x) {
          size_t off = geometry
            .cell_offsets[interval_idx];
          Coord x0 = geometry
            .intervals[interval_idx].begin;
          return values[off + (x - x0)];
        }
      };
      ```
    ],
    [
      == Organisation mémoire linéaire
      #set text(size: 10pt)
      ```
      Géométrie (sparse Y=0,2):
      Y=2: ░░████░░  intervals: [(2,6)]
      Y=0: ████░░██  intervals: [(0,4), (6,8)]

      values[] (stockage contigu):
      ┌───────────────┬───────────┬─────────────┐
      │   row0/int0   │ row0/int1 │   row2/int0 │
      │    [0,4)      │   [6,8)   │    [2,6)    │
      │ v0  v1  v2  v3│  v4   v5  │v6  v7  v8  v9│
      └───────────────┴───────────┴─────────────┘
       idx: 0   1   2   3    4   5   6   7   8   9

      cell_offsets = [0, 4, 6, 10]
      ```

      == Avantages
      - Accès *coalesced* sur GPU
      - Cache-friendly sur CPU
      - Opérations vectorisées
    ]
  )
]

// ============================================
// SLIDE 9: Workspace & AMR
// ============================================
#slide(title: "Workspace & Support AMR")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      == UnifiedCsrWorkspace
      Pool de buffers réutilisables

      ```cpp
      struct UnifiedCsrWorkspace {
        View<int*> int_bufs_[5];
        View<size_t*> size_t_bufs_[2];
        View<RowKey2D*> row_key_bufs_[2];
        View<Interval*> interval_buf_0;

        auto get_int_buf(int id, size_t n) {
          if (n > int_bufs_[id].extent(0))
            Kokkos::resize(int_bufs_[id], n*1.5);
          return subview(int_bufs_[id], {0,n});
        }
      };
      ```

      #box(fill: rgb("#d4edda"), inset: 0.4em, radius: 4pt)[
        *Évite* allocations répétées GPU \
        Crucial pour chaînage d'opérations
      ]
    ],
    [
      == MultilevelGeo (AMR)
      Grilles multi-résolution

      ```cpp
      template<class MemorySpace>
      struct MultilevelGeo {
        double origin_x, origin_y;
        double root_dx, root_dy;
        int num_active_levels;
        Array<GeoView, 16> levels;

        double dx_at(int level) {
          return root_dx / (1 << level);
        }
      };
      ```

      #set text(size: 9pt)
      ```
      Level 0: ┌───┬───┬───┬───┐  dx=1.0
               └───┴───┴───┴───┘
      Level 1: ┌─┬─┬─┬─┬─┬─┬─┬─┐  dx=0.5
               └─┴─┴█┴█┴█┴█┴─┴─┘  (raffiné)
      Level 2:     ┌┬┬┬┬┬┬┬┐      dx=0.25
                   └┴┴█┴█┴┴┴┘     (très fin)
      ```
    ]
  )
]

// ============================================
// SECTION: ALGORITHMES
// ============================================
#section-slide("IV. Algorithmes")

// ============================================
// SLIDE 10: Geometry Builders
// ============================================
#slide(title: "Constructeurs de Géométrie")[
  #set text(size: 12pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == Formes primitives
      ```cpp
      // Rectangle
      auto box = make_box_device(
        Box2D{0, 100, 0, 50});

      // Disque
      auto disk = make_disk_device(
        Disk2D{cx, cy, radius});

      // Damier
      auto checker = make_checkerboard_device(
        domain, cell_size);
      ```

      == À partir de données
      ```cpp
      // Depuis bitmap (masque binaire)
      auto geo = make_bitmap_device(
        bitmap_view, corner_x, corner_y);

      // Aléatoire (test/benchmark)
      auto rand = make_random_device(
        domain, fill_prob, seed);
      ```
    ],
    [
      == Pattern : build_interval_set_from_rows
      #set text(size: 10pt)
      ```cpp
      template<class Compute, class Fill>
      IntervalSet2D build_interval_set_from_rows(
        int num_rows, Compute compute, Fill fill)
      {
        // 1. COUNT (parallel)
        parallel_for(num_rows, [&](int r) {
          counts[r] = compute(r).num_intervals;
        });

        // 2. SCAN → row_ptr
        exclusive_scan(counts, row_ptr);

        // 3. FILL (parallel)
        parallel_for(num_rows, [&](int r) {
          fill(r, &intervals[row_ptr[r]]);
        });

        // 4. SCAN → cell_offsets
        compute_cell_offsets(intervals, offsets);
      }
      ```

      #box(fill: rgb("#e8f4f8"), inset: 0.3em, radius: 4pt)[
        Pattern *Count-Scan-Fill* : allocation exacte
      ]
    ]
  )
]

// ============================================
// SLIDE 11: Set Algebra
// ============================================
#slide(title: "Algèbre Ensembliste — Opérations Binaires")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == Opérations supportées
      ```cpp
      set_union_device(A, B, out, ctx);
      set_intersection_device(A, B, out, ctx);
      set_difference_device(A, B, out, ctx);
      set_symmetric_difference_device(...);
      ```

      #box(fill: light-gray, inset: 0.5em, radius: 4pt)[
        #set text(size: 10pt)
        ```
        A:       ████████████░░░░░░░░░░░░░░░░
        B:       ░░░░░░░░████████████░░░░░░░░

        A ∪ B:   ████████████████████░░░░░░░░
        A ∩ B:   ░░░░░░░░████░░░░░░░░░░░░░░░░
        A \ B:   ████████░░░░░░░░░░░░░░░░░░░░
        A △ B:   ████████░░░░████████░░░░░░░░
        ```
      ]

      == Algorithme Two-Pointer
      - *O(n + m)* par ligne
      - Parallélisme sur les *lignes*
      - Template `CountOnly` : même code count/fill
    ],
    [
      == Architecture unifiée
      ```cpp
      template<class RowOp>
      void apply_binary_csr_operation(
        A, B, mapping, output, row_op)
      {
        int num_out_rows = mapping.num_rows;

        // Phase 1: COUNT
        parallel_for(num_out_rows, [&](int r) {
          counts[r] = row_op.count(r, A, B);
        });

        // Phase 2: SCAN
        exclusive_scan(counts, row_ptr);

        // Phase 3: FILL
        parallel_for(num_out_rows, [&](int r) {
          row_op.fill(r, A, B, out_intervals);
        });
      }
      ```

      `RowOp` : UnionRowOp, IntersectionRowOp...
    ]
  )
]

// ============================================
// SLIDE 12: Field Operations
// ============================================
#slide(title: "Opérations sur Champs")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == Algèbre élément par élément
      ```cpp
      // Binaires
      field_add_device(a, b, result);
      field_sub_device(a, b, result);
      field_mul_device(a, b, result);

      // Combinaison linéaire: α·a + β·b
      field_axpby_device(alpha, a, beta, b, result);

      // Réductions
      T dot  = field_dot_device(a, b);
      T norm = field_norm_l2_device(a);
      ```

      == Implémentation (Kokkos std algorithms)
      ```cpp
      void field_add_device(a, b, result) {
        Kokkos::Experimental::transform(
          exec_space,
          a.values, b.values, result.values,
          KOKKOS_LAMBDA(T x, T y) {
            return x + y;
          });
      }
      ```
    ],
    [
      == Opérations sur sous-ensembles
      ```cpp
      // Remplir avec une valeur
      fill_on_subset_device(field, subset, val);

      // Multiplier par un scalaire
      scale_on_subset_device(field, subset, k);

      // Copier
      copy_on_subset_device(src, dst, subset);
      ```

      == Implémentation TeamPolicy
      ```cpp
      TeamPolicy policy(num_entries, AUTO);
      parallel_for(policy, KOKKOS_LAMBDA(team) {
        int e = team.league_rank();
        Coord x0 = subset.x_begin[e];
        Coord x1 = subset.x_end[e];

        TeamThreadRange(team, x0, x1,
          [&](Coord x) {
            field.at(e, x) = value;
          });
      });
      ```
    ]
  )
]

// ============================================
// SLIDE 13: Morphology & AMR
// ============================================
#slide(title: "Morphologie Mathématique & AMR")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == Dilatation / Érosion
      ```cpp
      // Union N-way avec décalage ±radius
      row_n_way_union_impl(rows[], radius, out)

      // Intersection N-way avec shrink
      row_n_way_intersection_impl(rows[], r, out)
      ```

      #set text(size: 9pt)
      ```
      Original:  ░░░░████████░░░░░░░░
      Dilate(1): ░░░█████████░░░░░░░░  (+1 côtés)
      Erode(1):  ░░░░░██████░░░░░░░░░  (-1 côtés)
      ```

      == Extension 2D
      - Considérer lignes y-r à y+r
      - Fusionner avec opération N-way
      - Structuring element implicite (carré)
    ],
    [
      == Opérations AMR
      ```cpp
      // Coarsening: fin → grossier
      build_row_coarsen_mapping(fine, ws)
      // y_coarse = y_fine / 2, fusionner X

      // Refinement: grossier → fin
      refine_level_up_device(coarse, ws)
      // [a,b) → [2a, 2b), doubler Y
      ```

      #set text(size: 9pt)
      ```
      Fine (level 1):        Coarse (level 0):
      Y=3: ████████          Y=1: ████████
      Y=2: ████████    →          (fusion Y=2,3)
      Y=1: ░░░░████          Y=0: ░░░░████
      Y=0: ░░░░████               (fusion Y=0,1)
      ```

      == Transfert de champs
      ```cpp
      // Projection fine → coarse (moyenne)
      // Prolongation coarse → fine (interp)
      build_amr_interval_mapping(coarse, fine)
      ```
    ]
  )
]

// ============================================
// SECTION: DEMO
// ============================================
#section-slide("V. Démonstration")

// ============================================
// SLIDE 14: Mach2 Cylinder Overview
// ============================================
#slide(title: "Mach2 Cylinder — Simulation AMR Multi-Niveaux")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == Description
      Simulation d'écoulement compressible 2D :
      - *Mach 2* supersonique autour d'un cylindre
      - Schéma Godunov 1er ordre + flux Rusanov
      - *AMR dynamique* : jusqu'à 6 niveaux

      #v(0.3em)
      == Utilisation de Subsetix
      ```cpp
      // Géométrie fluide = domaine - obstacle
      auto fluid = set_difference_device(
        make_box_device(domain),
        make_disk_device(cylinder),
        ctx);

      // Champs conservés (ρ, ρu, ρv, E)
      Field2DDevice<Real> rho(fluid);
      Field2DDevice<Real> rhou(fluid);
      // ...
      ```
    ],
    [
      == Architecture AMR
      #set text(size: 10pt)
      ```
      ┌─────────────────────────────────────┐
      │  Level 0 (coarse)                   │
      │  ┌─────────────────────────────┐    │
      │  │ Flux + Update (dt_coarse)  │    │
      │  └─────────────────────────────┘    │
      │         ▲ restrict    │ prolong     │
      │  ┌──────┴─────────────▼────────┐    │
      │  │  Level 1 (finer)            │    │
      │  │  ┌─────────────────────┐    │    │
      │  │  │ Flux + Update (dt)  │    │    │
      │  │  └─────────────────────┘    │    │
      │  │       ▲         │               │
      │  │  Level 2 (finest around shock)  │
      │  └─────────────────────────────┘    │
      └─────────────────────────────────────┘
      ```

      == Raffinement dynamique
      - Indicateur : gradient de densité
      - `expand_device()` pour zones de garde
      - Remaillage tous les N pas de temps
    ]
  )
]

// ============================================
// SLIDE 15: Mach2 Results
// ============================================
#slide(title: "Mach2 Cylinder — Résultats & Visualisation")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == Outputs générés
      ```
      output/
      ├── fluid_geometry.vtk
      ├── obstacle_geometry.vtk
      ├── level_0_density_0000.vtk
      ├── level_0_density_0050.vtk
      ├── level_1_density_0050.vtk
      ├── level_2_density_0050.vtk
      └── ...
      ```

      == Commande d'exécution
      ```bash
      ./mach2_cylinder \
        --nx 400 --ny 160 \
        --radius 20 \
        --mach-inlet 2.0 \
        --max-steps 1000 \
        --output-stride 50 \
        --amr
      ```
    ],
    [
      == Phénomènes observés
      #box(fill: light-gray, inset: 0.5em, radius: 4pt)[
        - *Choc d'étrave* (bow shock) devant le cylindre
        - Zone subsonique dans le sillage
        - Allée de *Von Kármán* (vortex)
        - Raffinement automatique près du choc
      ]

      #v(0.3em)
      == Points techniques clés
      - Stencil CSR : `apply_csr_stencil_on_set_device()`
      - Struct-of-Arrays pour cache efficiency
      - `prolong_guard_from_coarse()` : interpolation
      - `restrict_fine_to_coarse()` : conservation
      - Export VTK multi-niveau pour ParaView

      #v(0.3em)
      #box(fill: rgb("#d4edda"), inset: 0.4em, radius: 4pt)[
        *Sparse* : calcul uniquement sur cellules fluides !
      ]
    ]
  )
]

// ============================================
// SLIDE 16: Live Demo
// ============================================
#slide(title: "Démonstration Live")[
  #set text(size: 14pt)
  #align(center)[
    #v(1em)
    #text(size: 28pt, weight: "bold", fill: accent)[
      Live Demo
    ]

    #v(1em)

    #grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 1.5em,
      [
        == Construction
        - Box, Disk, Bitmap
        - Différence (obstacle)
        - Affichage CSR
      ],
      [
        == Opérations
        - Union / Intersection
        - Field algebra
        - Stencil
      ],
      [
        == Mach2
        - Lancement simulation
        - Visualisation ParaView
        - AMR en action
      ]
    )

    #v(2em)
    #box(fill: light-gray, inset: 1em, radius: 4pt)[
      _Démonstration en direct..._
    ]
  ]
]

// ============================================
// SLIDE 17: FIN
// ============================================
#pagebreak()
#set page(
  fill: dark,
  footer: context [
    #set text(size: 10pt, fill: white.transparentize(50%))
    #h(1fr)
    #counter(page).display("1 / 1", both: true)
  ]
)
#set text(fill: white)

#align(center + horizon)[
  #text(size: 38pt, weight: "bold")[Merci !]

  #v(0.8em)

  #text(size: 22pt, fill: accent)[Questions ?]

  #v(1.5em)

  #set text(size: 13pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      == Points clés
      - Représentation CSR d'intervalles
      - Pattern Count-Scan-Fill
      - Parallélisme Kokkos (CPU/GPU)
      - Workspace pour réutilisation mémoire
      - AMR multi-niveaux (Mach2)
    ],
    [
      == Contact
      Sébastien DUBOIS \
      Équipe HPC\@Maths

      #v(0.5em)
      Code : `include/subsetix/` \
      Demo : `examples/mach2_cylinder/`
    ]
  )
]

// ============================================
// SECTION: ANNEXES
// ============================================
#section-slide("Annexes")

// ============================================
// ANNEXE A: Évolution du projet
// ============================================
#slide(title: "Annexe A : Évolution du Projet")[
  #set text(size: 11pt)

  == Historique des implémentations

  #table(
    columns: (auto, 1fr, auto, auto),
    inset: 5pt,
    align: (center, left, center, center),
    fill: (x, y) => if y == 0 { rgb("#3498db").lighten(70%) } else if calc.odd(y) { rgb("#ecf0f1") } else { white },
    [*Version*], [*Description*], [*Performance*], [*Statut*],

    [*v1*],
    [CPU only, Sparse CSR + Workspaces \
     Première implémentation séquentielle],
    [Faster than baseline],
    [✓ Stable],

    [*v2*],
    [Multithreaded *Tiled* Sparse CSR \
     Backends OpenMP et TBB \
     Découpage en tuiles pour localité],
    [Excellent sur gros mesh],
    [⚠ Complexe \
     Bugs probables],

    [*v3*],
    [CUDA only \
     Algèbre d'ensembles GPU \
     Proof of concept],
    [*Plus rapide*],
    [✓ PoC validé],

    [*v4*],
    [*Kokkos* (version actuelle) \
     Non-tiled Sparse CSR \
     Portabilité OpenMP + CUDA],
    [Plus lent que v2/v3],
    [✓✓ *Fiable* \
     Vérifié],
  )

  #v(0.3em)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == Leçons apprises
      - Le *tiling* améliore la localité mais complexifie énormément
      - CUDA natif plus rapide mais moins portable
      - Kokkos = meilleur compromis *fiabilité/portabilité*
    ],
    [
      == Choix final : Kokkos
      - Code *unique* pour CPU et GPU
      - Maintenance simplifiée
      - Facilité de test et vérification
      - Écosystème actif (Sandia, Trilinos)
    ]
  )
]

// ============================================
// ANNEXE B: Pourquoi Kokkos
// ============================================
#slide(title: "Annexe B : Pourquoi Kokkos ?")[
  #set text(size: 11pt)

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      == Comparaison avec CUDA natif

      #table(
        columns: (auto, 1fr, 1fr),
        inset: 5pt,
        align: (left, center, center),
        fill: (x, y) => if y == 0 { accent.lighten(70%) } else { white },
        [*Aspect*], [*CUDA*], [*Kokkos*],
        [Portabilité], [NVIDIA only], [*Multi-vendor*],
        [Syntaxe], [`<<<>>>`], [*C++ standard*],
        [Memory], [cudaMalloc], [*View\<T\*\>*],
        [Debug CPU], [Difficile], [*Facile (Serial)*],
        [Maintenance], [Code dupliqué], [*Code unique*],
        [Performance], [*Optimal*], [~90-95%],
      )

      #v(0.3em)
      == Backends supportés
      - *OpenMP* : CPU multi-thread
      - *CUDA* : NVIDIA GPU
      - *HIP* : AMD GPU
      - *SYCL* : Intel GPU
      - *Serial* : debug et tests
    ],
    [
      == Avantages pour ce projet

      #box(fill: rgb("#d4edda"), inset: 0.5em, radius: 4pt)[
        *1. Développement plus rapide* \
        Debug sur CPU (Serial/OpenMP), deploy sur GPU
      ]

      #v(0.3em)
      #box(fill: rgb("#e8f4f8"), inset: 0.5em, radius: 4pt)[
        *2. Tests fiables* \
        Même code testé sur CPU et GPU \
        Pas de bugs "GPU-only" cachés
      ]

      #v(0.3em)
      #box(fill: rgb("#fff3cd"), inset: 0.5em, radius: 4pt)[
        *3. Std Algorithms* \
        `transform`, `reduce`, `scan`, `copy`... \
        API familière, optimisée par plateforme
      ]

      #v(0.3em)
      #box(fill: light-gray, inset: 0.5em, radius: 4pt)[
        *4. Écosystème* \
        Trilinos, ArborX, Cabana... \
        Support Sandia National Labs
      ]
    ]
  )
]

// ============================================
// ANNEXE C: Méthodologie
// ============================================
#slide(title: "Annexe C : Méthodologie de Développement")[
  #set text(size: 12pt)

  == Utilisation Intensive de LLMs

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      === Modèles utilisés
      - *Claude Opus 4* (Anthropic)
      - *Claude Sonnet 4* (Anthropic)

      #v(0.5em)
      === Pattern de travail
      #box(fill: light-gray, inset: 0.5em, radius: 4pt)[
        ```
        1. PLAN
           Architecture et interfaces
           Discussion des alternatives

        2. QUESTION
           Détails d'implémentation
           Edge cases

        3. IMPLEMENTATION
           Génération du code
           Revue et itération
        ```
      ]
    ],
    [
      === Avantages observés
      - *Exploration rapide* des designs
      - Documentation inline générée
      - Tests suggérés automatiquement
      - Refactoring assisté

      #v(0.5em)
      === Points d'attention
      - Vérification systématique du code
      - LLMs peuvent halluciner des APIs
      - Toujours compiler et tester
      - Garder le *contrôle architectural*

      #v(0.5em)
      #box(fill: rgb("#fff3cd"), inset: 0.4em, radius: 4pt)[
        LLM = *accélérateur*, pas remplacement \
        L'expertise humaine reste essentielle
      ]
    ]
  )
]

// ============================================
// ANNEXE D: Références
// ============================================
#slide(title: "Annexe D : Références & Ressources")[
  #set text(size: 12pt)

  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      == Kokkos
      - Site : kokkos.org
      - GitHub : github.com/kokkos/kokkos
      - Wiki : kokkos.org/kokkos-core-wiki

      #v(0.5em)
      == CUDA
      - CUDA Toolkit Documentation
      - CUDA C++ Programming Guide

      #v(0.5em)
      == Visualisation
      - VTK : vtk.org
      - ParaView : paraview.org
    ],
    [
      == Morphologie Mathématique
      - Serra, J. "Image Analysis and \
        Mathematical Morphology" (1982)
      - Soille, P. "Morphological Image \
        Analysis" (2003)

      #v(0.5em)
      == Code source
      ```
      include/subsetix/
      ├── geometry/      # IntervalSet2D
      ├── field/         # Field2D
      ├── csr_ops/       # Algorithmes
      ├── multilevel/    # AMR
      └── detail/        # Utilitaires

      examples/mach2_cylinder/
      └── mach2_cylinder.cpp  # Demo AMR
      ```
    ]
  )
]
