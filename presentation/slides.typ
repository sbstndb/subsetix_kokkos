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
  align(horizon, body)
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

// Fletcher pour diagrammes
#import "@preview/fletcher:0.4.5" as fletcher: diagram, node, edge

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
      4. *Vue d'ensemble Device*
      5. IntervalSet2D, Field2D, SubSet
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
#slide(title: "Architecture GPU — Massively Parallel")[
  #set text(size: 12pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      == Hiérarchie d'exécution
      #set text(size: 11pt)
      ```
      GPU
      └── SM (Streaming Multiprocessor) ×N
          └── Warps ×64 par SM
              └── Threads ×32 par warp (SIMT)
      ```

      #v(0.3em)
      - *Warp* = 32 threads exécutés *en lockstep*
      - *SM* = unité de calcul autonome
      - Plusieurs warps actifs par SM (latency hiding)

      #v(0.3em)
      == Comparaison B200 vs EPYC 9965
      #set text(size: 10pt)
      #table(
        columns: (auto, auto, auto),
        inset: 4pt,
        align: (left, center, center),
        fill: (x, y) => if y == 0 { accent.lighten(70%) } else { white },
        [], [*GPU B200*], [*CPU EPYC 9965*],
        [Cœurs], [148 SM], [192 cores],
        [Mémoire], [192 GB HBM3e], [jusqu'à 6 TB DDR5],
        [Bandwidth], [*8 TB/s*], [576 GB/s],
        [FP32], [*80 TFlops*], [~14 TFlops],
      )
    ],
    [
      == Modèle d'exécution
      #set text(size: 9pt)
      #align(center)[
        #box(stroke: 1.5pt + dark, fill: light-gray.lighten(50%), radius: 4pt, inset: 0.4em)[
          #align(center)[
            #text(weight: "bold", size: 10pt)[GRID]
            #v(0.2em)
            #diagram(
              node-stroke: 1pt + dark,
              edge-stroke: 1pt + accent,
              spacing: (8mm, 5mm),

              // Blocks row
              node((0, 0), [*Block 0* \ #text(size: 7pt)[32-1024 th]], corner-radius: 2pt, fill: white, name: <b0>),
              node((1, 0), [*Block 1* \ #text(size: 7pt)[threads]], corner-radius: 2pt, fill: white, name: <b1>),
              node((2, 0), [*Block N* \ #text(size: 7pt)[...]], corner-radius: 2pt, fill: white, name: <bn>),

              // Arrows to SMs
              edge(<b0>, <sm0>, "->"),
              edge(<b1>, <sm1>, "->"),
              edge(<bn>, <smk>, "->"),

              // SMs row
              node((0, 1), [SM 0], corner-radius: 2pt, fill: rgb("#d4edda"), name: <sm0>),
              node((1, 1), [SM 1], corner-radius: 2pt, fill: rgb("#d4edda"), name: <sm1>),
              node((2, 1), [SM k], corner-radius: 2pt, fill: rgb("#d4edda"), name: <smk>),
            )
          ]
        ]
      ]

      #v(0.2em)
      == Pour notre projet
      - *1 thread* = traite 1 ligne Y (ou 1 cellule)
      - Des milliers de lignes → *saturent le GPU*

      #box(fill: rgb("#d4edda"), inset: 0.4em, radius: 4pt)[
        GPU : *14× plus de bandwidth* que CPU \
        → idéal pour grands maillages
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
      #set text(size: 11pt)
      - CUDA = NVIDIA only
      - OpenMP = CPU only (GPU limité)
      - HIP = AMD only
      - Réécrire pour chaque plateforme ?

      == La solution : Kokkos
      #set text(size: 10pt)
      ```cpp
      // 1. COUNT — taille résultat inconnue
      parallel_for(num_rows, KOKKOS_LAMBDA(int r) {
        counts[r] = count_intervals(r);
      });
      // 2. SCAN — calcul des offsets
      exclusive_scan(counts, row_ptr);
      // 3. FILL — écriture parallèle
      parallel_for(num_rows, KOKKOS_LAMBDA(int r) {
        fill_intervals(r, &out[row_ptr[r]]);
      });
      ```
    ],
    [
      == CUDA vs Kokkos
      #set text(size: 9pt)
      #grid(
        columns: (1fr, 1fr),
        gutter: 0.5em,
        [
          *CUDA natif*
          ```cpp
          // Allocation
          double* d_data;
          cudaMalloc(&d_data, n*8);

          // Copie Host → Device
          cudaMemcpy(d_data, h_data,
            n*8, HostToDevice);

          // Kernel
          kernel<<<B,T>>>(d_data, n);

          // Copie Device → Host
          cudaMemcpy(h_data, d_data,
            n*8, DeviceToHost);

          // Libération
          cudaFree(d_data);
          ```
        ],
        [
          *Kokkos*
          ```cpp
          // Allocation + miroir auto
          View<double*> data("d", n);
          auto h_data = create_mirror_view(data);

          // Copie Host → Device
          deep_copy(data, h_data);

          // Parallel (CPU ou GPU)
          parallel_for(n, KOKKOS_LAMBDA(int i){
            data(i) = compute(i);
          });

          // Copie Device → Host
          deep_copy(h_data, data);

          // Libération automatique (RAII)
          ```
        ]
      )
    ]
  )

  #align(center)[
    #box(fill: rgb("#d4edda"), inset: 0.5em, radius: 4pt)[
      *Un code source unique* → compile pour OpenMP, CUDA, HIP, SYCL, Serial — spécialisable si nécessaire
    ]
  ]
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
          ```
        ]
      ]

      == Complexité mémoire
      *O(R + I)* — R = lignes Y, I = intervalles

      #box(fill: rgb("#d4edda"), inset: 0.3em, radius: 4pt)[
        Dense O(W×H) vs CSR O(R+I) ≪
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

      #box(fill: rgb("#e8f4f8"), inset: 0.3em, radius: 4pt)[
        *Trou Y=4,5* : row_keys saute de 3 à 6
      ]
    ]
  )
]

// ============================================
// SECTION: DATA STRUCTURES
// ============================================
#section-slide("III. Structures de Données")

// ============================================
// SLIDE 7: Vue d'Ensemble — Structures Device
// ============================================
#slide(title: "Vue d'Ensemble — Structures Device")[
  #set text(size: 10pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == Architecture des types GPU
      #set text(size: 8pt)
      #align(center)[
        #diagram(
          node-stroke: 1pt + dark,
          node-fill: light-gray,
          edge-stroke: 1pt + accent,
          spacing: (8mm, 6mm),

          node((0, 0), align(left)[
            *IntervalSet2D* \
            #set text(size: 7pt)
            row_keys[], row_ptr[] \
            intervals[], cell_offsets[]
          ], corner-radius: 3pt, width: 38mm, name: <set>),

          edge(<set>, <field>, "->", [référence], label-side: right),

          node((0, 1), align(left)[
            *Field2D\<T\>* \
            #set text(size: 7pt)
            geometry: IntervalSet2D \
            values: View\<T\*\>
          ], corner-radius: 3pt, width: 38mm, fill: rgb("#d4edda"), name: <field>),

          edge(<field>, <subview>, "->", [make_subview()], label-side: right),

          node((0, 2), align(left)[
            *Field2DSubView\<T\>* \
            #set text(size: 7pt)
            parent: Field2D& \
            region: IntervalSet2D \
            subset: SubSet (lazy)
          ], corner-radius: 3pt, width: 38mm, fill: rgb("#e8f4f8"), name: <subview>),
        )
      ]

      #v(0.2em)
      == IntervalSubSet2D (intersection)
      #set text(size: 9pt)
      ```cpp
      // Décrit l'intersection parent ∩ region
      struct IntervalSubSet2D {
        interval_indices[];  // → parent
        x_begin[], x_end[];  // sous-plages
        row_indices[];       // index lignes
      };
      ```
    ],
    [
      == Workflow GPU typique
      #set text(size: 9pt)
      ```cpp
      // Champs sur même géométrie
      Field2D<Real> rho(fluid_geom);
      Field2D<Real> rhou(fluid_geom);

      // Zone d'intérêt (overlap, guard, etc.)
      auto overlap = set_intersection_device(
          field.geometry, other_level, ctx);

      // SubView = référence légère
      auto sub_rho = make_subview(rho, overlap);
      auto sub_rhou = make_subview(rhou, overlap);

      // Opérations sur la zone uniquement
      fill_subview_device(sub_rho, 0.0, &ctx);
      copy_subview_device(dst, src, &ctx);

      // Stencil, AMR restrict/prolong...
      restrict_field_subview_device(coarse, fine);
      prolong_field_subview_device(fine, coarse);
      ```

      == Pourquoi SubView ?
      #table(
        columns: (auto, 1fr),
        inset: 3pt,
        align: (left, left),
        fill: (x, y) => if y == 0 { accent.lighten(70%) } else { white },
        [*Avantage*], [*Description*],
        [Non-owning], [Pas de copie, juste des références],
        [Lazy subset], [Intersection calculée à la demande],
        [Cache ctx], [SubSet réutilisé entre opérations],
        [API unifiée], [fill, copy, stencil, AMR...],
      )
    ]
  )
]

// ============================================
// SLIDE 8: IntervalSet2D Structure
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
// SLIDE 9: Field2D
// ============================================
#slide(title: "Field2D — Champ sur Géométrie Creuse")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      == Définition
      Associe une *valeur* à chaque cellule sparse

      ```cpp
      template<class T, class MemorySpace>
      struct Field2D {
        IntervalSet2D geometry;  // Réf géométrie
        View<T*> values;         // [total_cells]

        // Accès à une valeur
        T& at(interval_idx, x) {
          offset = cell_offsets(interval_idx);
          x0 = intervals(interval_idx).begin;
          return values(offset + x - x0);
        }
      };
      ```

      #v(0.3em)
      == Stockage mémoire
      #set text(size: 10pt)
      ```
      Géométrie: ████ ░░ ████ ░░ ██████
      values[]:  [v0 v1 | v2 v3 | v4 v5 v6]
                  ↑       ↑       ↑
      offsets:    0       2       4
      ```
      Valeurs *contiguës* → cache-friendly
    ],
    [
      == Opérations sur Fields
      ```cpp
      // Algèbre élément par élément
      field_add(a, b, result);       // a + b
      field_sub(a, b, result);       // a - b
      field_mul(a, b, result);       // a * b
      field_axpby(α, a, β, b, r);    // αa + βb

      // Réductions globales
      T sum  = field_reduce_sum(f);
      T dot  = field_dot(a, b);      // Σ aᵢbᵢ
      T norm = field_norm_l2(f);     // √(Σ fᵢ²)
      T min  = field_min(f);
      T max  = field_max(f);
      ```

      #v(0.3em)
      == Implémentation (Kokkos std-like)
      #set text(size: 10pt)
      ```cpp
      // Utilise transform, reduce, etc.
      Kokkos::Experimental::transform(
        exec, a.values, b.values, result.values,
        KOKKOS_LAMBDA(T x, T y) { return x + y; }
      );
      ```

      #box(fill: rgb("#e8f4f8"), inset: 0.3em, radius: 4pt)[
        Pour opérer sur une *zone spécifique* → SubSet
      ]
    ]
  )
]

// ============================================
// SLIDE 10: SubSet — Opérations sur Zones
// ============================================
#slide(title: "SubSet — Opérations sur Zones Ciblées")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == Problème
      Comment appliquer une opération *uniquement* sur une zone ?
      - Conditions aux limites (bords)
      - Zone source (injection d'énergie)
      - Mise à jour partielle

      #v(0.3em)
      == Solution : SubSet = Intersection
      #set text(size: 9pt)
      #align(center)[
        #box(stroke: 1pt + dark, inset: 0.4em, radius: 4pt)[
          ```
          Géométrie Field:       Masque:
          ████████████████       ░░░░████████░░░░
          ████████████████   ∩   ░░░░████████░░░░
          ████████████████       ░░░░░░░░░░░░░░░░

                    ↓ build_interval_subset()

                 SubSet:
          ░░░░████████░░░░  ← référence parent
          ░░░░████████░░░░  ← [x_begin, x_end)
          ░░░░░░░░░░░░░░░░    par intervalle
          ```
        ]
      ]

      #box(fill: rgb("#d4edda"), inset: 0.3em, radius: 4pt)[
        *Pas de copie* — référence la géométrie parente
      ]
    ],
    [
      == Structure
      ```cpp
      struct IntervalSubSet2D {
        IntervalSet2D parent;  // Géométrie ref
        interval_indices[];    // Index intervalles
        x_begin[], x_end[];    // Sous-plages X
        row_indices[];         // Index lignes
        num_entries;
        total_cells;
      };
      ```

      #v(0.3em)
      == Utilisation
      ```cpp
      // Construire le subset (intersection)
      build_interval_subset(
        field.geometry, mask, subset);

      // Opérations sur la zone uniquement
      fill_on_subset(field, subset, 0.0);
      scale_on_subset(field, subset, 2.0);

      // Functor personnalisé
      apply_on_subset(field, subset,
        KOKKOS_LAMBDA(x, y, val, idx) {
          val = source(x, y);
        });
      ```

      #box(fill: rgb("#e8f4f8"), inset: 0.3em, radius: 4pt)[
        *GPU-friendly* : parallélisme sur les entries
      ]
    ]
  )
]

// ============================================
// SLIDE 11: Workspace & AMR
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
// SLIDE 12: Geometry Builders
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
// SLIDE 13: Set Algebra
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
// SLIDE 14: Field Operations
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
// SLIDE 15: Morphology & AMR
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
// SLIDE 16: Mach2 Cylinder Overview
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
// SLIDE 17: Mach2 Results
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
// SLIDE 18: Live Demo
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
// SLIDE 19: FIN
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
