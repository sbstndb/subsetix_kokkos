// ============================================
// SUBSETIX KOKKOS - PRESENTATION
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

// HPC@Maths Colors
#let hpc-dark = rgb("#003957")      // Dark blue (H, P, C)
#let hpc-medium = rgb("#046D98")    // Medium blue (Maths)
#let hpc-light = rgb("#5AA0BB")     // Light blue (spiral)

#let accent = hpc-medium
#let dark = hpc-dark
#let light-gray = rgb("#ecf0f1")
#let green = rgb("#27ae60")
#let orange = rgb("#e67e22")

// Slide helper function
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

  // Logo at bottom right
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

// Fletcher for diagrams
#import "@preview/fletcher:0.4.5" as fletcher: diagram, node, edge

// ============================================
// SLIDE 1: TITLE
// ============================================
#title-slide(
  "Subsetix: Sparse 2D Geometry on GPU",
  subtitle: "From Set Algebra to AMR Simulation",
  author: "Sébastien DUBOIS",
  affiliation: "HPC@Maths Team",
  date: "December 2025",
  logo: "logo_hpc.png",
)

// ============================================
// SLIDE 2: OUTLINE
// ============================================
#slide(title: "Outline")[
  #set text(size: 15pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      == I. Context
      1. GPU Computing & Kokkos

      #v(0.5em)
      == II. Sparse Representation
      2. Intervals and CSR
      3. 2D Sparse Mesh Example

      #v(0.5em)
      == III. Data Structures
      4. *Device-Side Overview*
      5. IntervalSet2D, Field2D, SubSet
      6. Workspace & AMR
    ],
    [
      == IV. Algorithms
      7. Geometry Constructors
      8. Set Algebra
      9. Field Operations
      10. Morphology & AMR

      #v(0.5em)
      == V. Demo
      11. Mach2 Cylinder (Multi-level AMR)

      #v(0.5em)
      == VI. Appendices
      - Project Evolution
      - Why Kokkos?
      - Development Methodology
    ]
  )
]

// ============================================
// SECTION: GPU & KOKKOS
// ============================================
#section-slide("I. Context: GPU & Kokkos")

// ============================================
// SLIDE 3: GPU & CUDA Essentials (condensed)
// ============================================
#slide(title: "GPU Architecture — Massively Parallel")[
  #set text(size: 13pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      == Execution Hierarchy
      #v(0.5em)
      #align(center)[
        #diagram(
          node-stroke: 1.5pt + dark,
          edge-stroke: 2pt + accent,
          spacing: (8mm, 5mm),

          node((0, 0), text(size: 10pt, fill: white)[*GPU*], corner-radius: 3pt, fill: hpc-dark, stroke: none, name: <gpu>, inset: 6pt),
          edge(<gpu>, <sm>, "->"),
          node((1, 0), text(size: 10pt, fill: white)[*SM* ×N], corner-radius: 3pt, fill: hpc-medium, stroke: none, name: <sm>, inset: 6pt),
          edge(<sm>, <warp>, "->"),
          node((2, 0), text(size: 10pt, fill: white)[*Warp* ×64], corner-radius: 3pt, fill: hpc-light, stroke: none, name: <warp>, inset: 6pt),
          edge(<warp>, <thread>, "->"),
          node((3, 0), text(size: 10pt)[*Thread* ×32], corner-radius: 3pt, fill: rgb("#d4edda"), name: <thread>, inset: 6pt),
        )
      ]

      #v(0.8em)
      - *Warp* = 32 threads in *lockstep* (SIMT)
      - *SM* = autonomous compute unit
      - Multiple warps/SM → latency hiding

      #v(1em)
      == For Our Project
      - *1 thread* = processes 1 Y row (or 1 cell)
      - Thousands of rows → *saturate the GPU*
    ],
    [
      == Execution Model
      #v(0.3em)
      #align(center)[
        #box(stroke: 2pt + dark, fill: light-gray.lighten(50%), radius: 6pt, inset: 0.8em)[
          #text(weight: "bold", size: 13pt)[GRID]
          #v(0.5em)
          #diagram(
            node-stroke: 1.5pt + dark,
            edge-stroke: 1.5pt + accent,
            spacing: (14mm, 10mm),

            // Blocks row
            node((0, 0), [*Block 0* \ #text(size: 9pt)[32-1024 th]], corner-radius: 3pt, fill: white, name: <b0>, inset: 5pt),
            node((1, 0), [*Block 1* \ #text(size: 9pt)[threads]], corner-radius: 3pt, fill: white, name: <b1>, inset: 5pt),
            node((2, 0), [*Block N* \ #text(size: 9pt)[...]], corner-radius: 3pt, fill: white, name: <bn>, inset: 5pt),

            // Arrows to SMs
            edge(<b0>, <sm0>, "->"),
            edge(<b1>, <sm1>, "->"),
            edge(<bn>, <smk>, "->"),

            // SMs row
            node((0, 1), text(size: 10pt)[SM 0], corner-radius: 3pt, fill: rgb("#d4edda"), name: <sm0>, inset: 5pt),
            node((1, 1), text(size: 10pt)[SM 1], corner-radius: 3pt, fill: rgb("#d4edda"), name: <sm1>, inset: 5pt),
            node((2, 1), text(size: 10pt)[SM k], corner-radius: 3pt, fill: rgb("#d4edda"), name: <smk>, inset: 5pt),
          )
        ]
      ]

      #v(1em)
      == B200 vs EPYC 9965
      #align(center)[
        #table(
          columns: (auto, auto, auto),
          inset: 6pt,
          align: (left, center, center),
          fill: (x, y) => if y == 0 { accent.lighten(70%) } else { white },
          [], [*GPU B200*], [*CPU EPYC 9965*],
          [Bandwidth], [*8 TB/s*], [576 GB/s],
          [FP32], [*80 TFlops*], [~14 TFlops],
        )
      ]
    ]
  )

  #v(0.5em)
  #align(center)[
    #box(fill: rgb("#d4edda"), inset: 0.6em, radius: 6pt)[
      #text(size: 14pt)[GPU: *14× more bandwidth* than CPU → ideal for large sparse meshes]
    ]
  ]
]

// ============================================
// SLIDE 4: Kokkos Introduction
// ============================================
#slide(title: "Kokkos — Performance Portability")[
  #set text(size: 12pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == The Problem
      #set text(size: 11pt)
      - CUDA = NVIDIA only
      - OpenMP = CPU only (limited GPU)
      - HIP = AMD only
      - Rewrite for each platform?

      == The Solution: Kokkos
      #set text(size: 10pt)
      ```cpp
      // 1. COUNT — unknown result size
      parallel_for(num_rows, KOKKOS_LAMBDA(int r) {
        counts[r] = count_intervals(r);
      });
      // 2. SCAN — compute offsets
      exclusive_scan(counts, row_ptr);
      // 3. FILL — parallel write
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
          *Native CUDA*
          ```cpp
          // Allocation
          double* d_data;
          cudaMalloc(&d_data, n*8);

          // Copy Host → Device
          cudaMemcpy(d_data, h_data,
            n*8, HostToDevice);

          // Kernel
          kernel<<<B,T>>>(d_data, n);

          // Copy Device → Host
          cudaMemcpy(h_data, d_data,
            n*8, DeviceToHost);

          // Free
          cudaFree(d_data);
          ```
        ],
        [
          *Kokkos*
          ```cpp
          // Allocation + auto mirror
          View<double*> data("d", n);
          auto h_data = create_mirror_view(data);

          // Copy Host → Device
          deep_copy(data, h_data);

          // Parallel (CPU or GPU)
          parallel_for(n, KOKKOS_LAMBDA(int i){
            data(i) = compute(i);
          });

          // Copy Device → Host
          deep_copy(h_data, data);

          // Automatic cleanup (RAII)
          ```
        ]
      )
    ]
  )

  #align(center)[
    #box(fill: rgb("#d4edda"), inset: 0.5em, radius: 4pt)[
      *Single source code* → compiles for OpenMP, CUDA, HIP, SYCL, Serial — specializable if needed
    ]
  ]
]

// ============================================
// SECTION: SPARSE REPRESENTATION
// ============================================
#section-slide("II. Sparse Representation")

// ============================================
// SLIDE 5: 2D Sparse Mesh Example
// ============================================
#slide(title: "Example: 2D Sparse Mesh with Intervals")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == "Smiley" Geometry :-)
      #align(center)[
        #box(stroke: 1pt + dark, inset: 0.5em, radius: 4pt)[
          ```
          Y
          9│ . . . . . . . . . .    (empty)
          8│ . . . . . . . . . .    (empty)
          7│ . . ▓ ▓ . . ▓ ▓ . .    EYES
          6│ . . ▓ ▓ . . ▓ ▓ . .    EYES
          5│ . . . . . . . . . .    (HOLE)
          4│ . . . . . . . . . .    (HOLE)
          3│ . ▓ ▓ . . . . ▓ ▓ .    SMILE
          2│ . . ▓ ▓ . . ▓ ▓ . .    SMILE
          1│ . . . ▓ ▓ ▓ ▓ . . .    SMILE
          0│ . . . . . . . . . .    (empty)
           └──────────────────── X
             0 1 2 3 4 5 6 7 8 9
          ```
        ]
      ]

    ],
    [
      == CSR Representation
      ```cpp
      // 5 rows, HOLE Y=4,5
      row_keys = [1, 2, 3, 6, 7]  // skips 4,5!
      num_rows = 5

      // Rows with 1 or 2 intervals
      row_ptr = [0, 1, 3, 5, 7, 9]

      intervals = [
        {3, 7},        // Y=1: smile bottom
        {2, 4}, {6, 8},// Y=2: smile thick
        {1, 3}, {7, 9},// Y=3: smile corners
        {2, 4}, {6, 8},// Y=6: EYES bottom
        {2, 4}, {6, 8},// Y=7: EYES top
      ]
      num_intervals = 9

      cell_offsets = [0,4,6,8,10,12,14,16,18,20]
      total_cells = 20
      ```

      #align(center)[
        #box(fill: rgb("#e8f4f8"), inset: 0.3em, radius: 4pt)[
          *Hole Y=4,5*: row_keys jumps from 3 to 6
        ]
      ]
    ]
  )
]

// ============================================
// SECTION: DATA STRUCTURES
// ============================================
#section-slide("III. Data Structures")

// ============================================
// SLIDE 7: Overview — Device Structures
// ============================================
#slide(title: "Overview — Device Structures")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      == Core Types
      #v(0.3em)
      #align(center)[
        #diagram(
          node-stroke: 1.5pt + dark,
          node-fill: light-gray,
          edge-stroke: 1.5pt + accent,
          spacing: (10mm, 10mm),

          node((0, 0), align(left)[
            #text(size: 10pt, weight: "bold")[IntervalSet2D] \
            #text(size: 8pt)[row_keys[], row_ptr[] \
            intervals[], cell_offsets[]]
          ], corner-radius: 4pt, width: 42mm, inset: 6pt, name: <set>),

          edge(<set>, <field>, "->"),

          node((0, 1), align(left)[
            #text(size: 10pt, weight: "bold")[Field2D\<T\>] \
            #text(size: 8pt)[geometry: IntervalSet2D \
            values: View\<T\*\>]
          ], corner-radius: 4pt, width: 42mm, inset: 6pt, fill: rgb("#d4edda"), name: <field>),

          // Region input
          node((1, 1), align(center)[
            #text(size: 10pt, weight: "bold")[Region] \
            #text(size: 8pt)[(IntervalSet2D)]
          ], corner-radius: 4pt, width: 26mm, inset: 6pt, fill: rgb("#fff3cd"), name: <region>),

          edge(<field>, <subview>, "->"),
          edge(<region>, <subview>, "->"),

          node((0.5, 2), align(left)[
            #text(size: 10pt, weight: "bold")[Field2DSubView\<T\>] \
            #text(size: 8pt)[parent: Field2D& \
            region: IntervalSet2D \
            subset: IntervalSubSet2D]
          ], corner-radius: 4pt, width: 50mm, inset: 6pt, fill: rgb("#e8f4f8"), name: <subview>),
        )
      ]

      #v(0.3em)
      == IntervalSubSet2D
      #set text(size: 9pt)
      ```cpp
      struct IntervalSubSet2D {
        IntervalSet2D parent;      // ref
        interval_indices[];        // which intervals
        x_begin[], x_end[];        // restricted range
        row_indices[];             // Y coords
      };
      ```
    ],
    [
      == SubView: Lazy Intersection
      #set text(size: 8pt)
      ```cpp
      // Region = any IntervalSet2D
      IntervalSet2DDevice left_bc = make_box_device({0,2,0,ny});
      Field2DSubViewDevice<T> sub = make_subview(field, left_bc);

      // First op: computes field.geo ∩ region
      fill_subview_device(sub, T_inlet, &ctx);

      // Time loop: reuses cached intersection
      for (int step = 0; step < nsteps; ++step) {
        fill_subview_device(sub, T_inlet);  // fast
      }
      ```

      #v(0.3em)
      == SubView Operations
      - `fill_subview_device(sub, val)`
      - `scale_subview_device(sub, alpha)`
      - `copy_subview_device(dst, src)`
      - `apply_stencil_on_subview_device(...)`

      #v(0.3em)
      #box(fill: rgb("#d4edda"), inset: 0.4em, radius: 4pt, width: 100%)[
        #set text(size: 10pt)
        *Lazy*: intersection computed on first use \
        *Cached*: reused for subsequent operations
      ]
    ]
  )
]

// ============================================
// SLIDE 8: IntervalSet2D Structure
// ============================================
#slide(title: "IntervalSet2D — Complete CSR Structure")[
  #set text(size: 12pt)
  #grid(
    columns: (1.1fr, 1fr),
    gutter: 1em,
    [
      == C++ Definition
      ```cpp
      template<class MemorySpace>
      struct IntervalSet2D {
        // Y coordinates of non-empty rows
        View<RowKey2D*> row_keys;  // [num_rows]

        // Index into intervals[] for each row
        View<size_t*> row_ptr;     // [num_rows + 1]

        // All intervals (contiguous)
        View<Interval*> intervals; // [num_intervals]

        // Linear cell offsets
        View<size_t*> cell_offsets;// [num_intervals]

        size_t total_cells;
        int num_rows;
        int num_intervals;
      };
      ```
    ],
    [
      == Invariants
      - `row_keys` sorted by increasing Y
      - Intervals sorted by X within each row
      - No overlap between intervals
      - `row_ptr[r+1] - row_ptr[r]` = nb intervals row r

      #v(0.5em)
      #align(center)[
        #box(fill: rgb("#e8f4f8"), inset: 0.4em, radius: 4pt)[
          *Template MemorySpace*: Device or Host
        ]
      ]
    ]
  )
]

// ============================================
// SLIDE 9: Field2D
// ============================================
#slide(title: "Field2D — Field on Sparse Geometry")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      == Definition
      Associates a *value* with each sparse cell

      ```cpp
      template<class T, class MemorySpace>
      struct Field2D {
        IntervalSet2D geometry;  // Geometry ref
        View<T*> values;         // [total_cells]
      };
      ```

      #v(0.3em)
      == Memory Layout
      #set text(size: 10pt)
      #align(center)[
        ```
        Geometry:  ████ ░░ ████ ░░ ██████
        values[]:  [v0 v1 | v2 v3 | v4 v5 v6]
                    ↑       ↑       ↑
        offsets:    0       2       4
        ```
        *Contiguous* values → cache-friendly
      ]
    ],
    [
      == Cell Access
      #set text(size: 10pt)
      ```cpp
      // O(1) - when interval index known
      T val = field.at(interval_idx, x);

      // O(log R + log I) - by coordinates
      // (binary search on Y, then X)
      bool ok = accessor.try_get(x, y, val);
      ```

      #v(0.5em)
      == Usage
      #set text(size: 10pt)
      ```cpp
      Field2DDevice<double> rho(fluid_geo);
      fill_field_device(rho, 1.0);
      auto rho_host = to_host(rho);  // I/O
      ```
    ]
  )
]

// ============================================
// SLIDE 10: SubSet — Targeted Region Operations
// ============================================
#slide(title: "SubSet — Targeted Region Operations")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      == Structure
      #set text(size: 10pt)
      ```cpp
      struct IntervalSubSet2D {
        IntervalSet2D parent;  // ref to Field geo
        interval_indices[];    // which intervals
        x_begin[], x_end[];    // restricted range
        row_indices[];         // Y row in parent
        num_entries;
      };
      ```

      #v(0.3em)
      == Usage
      #set text(size: 10pt)
      ```cpp
      // Build subset (intersection)
      build_interval_subset(
        field.geometry, mask, subset, &ctx);

      // Operations on subset only
      fill_on_subset(field, subset, 0.0);

      // Iteration: O(1) access per entry
      for (e = 0; e < num_entries; ++e) {
        int iv = interval_indices[e];
        for (x = x_begin[e]; x < x_end[e]; ++x)
          field.at(iv, x) = ...;  // O(1)
      }
      ```
    ],
    [
      == 1D Example: Intersection
      #set text(size: 8pt)
      #align(center)[
        #box(stroke: 1pt + dark, inset: 0.5em, radius: 4pt, fill: light-gray.lighten(50%))[
          ```
          Parent:  [==A==]   [==B==]   [==C==]
            idx:      0         1         2
                   0     8  12    18  22    30

          Mask:        [=======M=======]
                       5              25

          SubSet:      [=]   [==B==]   [=]
                       5 8  12    18  22 25
                        ↑       ↑       ↑
          entry:        0       1       2
          ```
        ]
      ]

      #v(0.2em)
      == SubSet = references to Parent
      #set text(size: 8pt)
      #align(center)[
        #box(stroke: 1pt + accent, inset: 0.4em, radius: 4pt)[
          ```
          entry | interval_idx | x_begin | x_end
          ------+--------------+---------+------
            0   |      0 (A)   |    5    |   8
            1   |      1 (B)   |   12    |  18
            2   |      2 (C)   |   22    |  25
          ```
        ]
      ]

      #align(center)[
        #box(fill: rgb("#d4edda"), inset: 0.3em, radius: 4pt)[
          *No data copy* — just indices + bounds
        ]
      ]
    ]
  )
]

// ============================================
// SLIDE 11: Field2DSubView
// ============================================
#slide(title: "Field2DSubView — View on Field + Region")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      == Structure
      #set text(size: 10pt)
      ```cpp
      struct Field2DSubView<T> {
        Field2D<T> parent;        // ref to field
        IntervalSet2D region;     // where to operate
        IntervalSubSet2D subset;  // lazy intersection
      };
      ```

      #v(0.3em)
      == Lazy Pattern
      #set text(size: 9pt)
      ```cpp
      // 1. Create (no computation)
      auto sub = make_subview(field, region);
      // sub.subset is empty

      // 2. First op with ctx → triggers build
      fill_subview_device(sub, 0.0, &ctx);
      // sub.subset = field.geo ∩ region

      // 3. Next ops reuse cached subset
      scale_subview_device(sub, 2.0);  // fast!
      fill_subview_device(sub, 1.0);   // fast!
      ```
    ],
    [
      == Memory Mapping
      #set text(size: 8pt)
      #align(center)[
        #box(stroke: 1pt + dark, inset: 0.5em, radius: 4pt, fill: light-gray.lighten(50%))[
          ```
          Geometry:   [==A==]    [==B==]    [==C==]
                         ↓          ↓          ↓
          values[]:  [░░███|░░░░░████░|██░░░░░░]
                        ↑          ↑       ↑
                     entry 0    entry 1  entry 2
          ```
        ]
      ]
      #align(center)[
        #text(size: 7pt)[░ = skipped #h(1em) █ = accessed by SubSet]
      ]

      #v(0.4em)
      == Access Formula
      #set text(size: 9pt)
      #align(center)[
        #box(fill: rgb("#fff3cd"), inset: 0.4em, radius: 4pt)[
          `values[ offset[idx] + (x - interval.begin) ]`
        ]
      ]

      #v(0.3em)
      #align(center)[
        #box(fill: rgb("#d4edda"), inset: 0.3em, radius: 4pt)[
          #set text(size: 9pt)
          *O(1)* per cell — no coordinate lookup
        ]
      ]
    ]
  )
]

// ============================================
// SLIDE 12: Workspace & AMR
// ============================================
#slide(title: "Workspace & AMR Support")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      == UnifiedCsrWorkspace
      Pool of reusable buffers

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

      #align(center)[
        #box(fill: rgb("#d4edda"), inset: 0.4em, radius: 4pt)[
          *Avoids* repeated GPU allocations \
          Crucial for chained operations
        ]
      ]
    ],
    [
      == MultilevelGeo (AMR)
      Multi-resolution grids

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
      #align(center)[
        ```
        Level 0: ┌───┬───┬───┬───┐  dx=1.0
                 └───┴───┴───┴───┘
        Level 1: ┌─┬─┬─┬─┬─┬─┬─┬─┐  dx=0.5
                 └─┴─┴█┴█┴█┴█┴─┴─┘  (refined)
        Level 2:     ┌┬┬┬┬┬┬┬┐      dx=0.25
                     └┴┴█┴█┴┴┴┘     (very fine)
        ```
      ]
    ]
  )
]

// ============================================
// SECTION: ALGORITHMS
// ============================================
#section-slide("IV. Algorithms")

// ============================================
// SLIDE 12: Set Algebra
// ============================================
#slide(title: "Set Algebra — Binary Operations")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      == CsrSetAlgebraContext
      #set text(size: 10pt)
      ```cpp
      struct CsrSetAlgebraContext {
        UnifiedCsrWorkspace workspace;
        // Pool of reusable GPU buffers:
        // - int_bufs_[5], size_t_bufs_[2]
        // - row_key_bufs_[2], interval_buf_
        // Auto-grows on demand, never shrinks
      };
      ```

      #align(center)[
        #diagram(
          node-stroke: 1pt + dark,
          edge-stroke: 1.5pt + accent,
          spacing: (6mm, 8mm),

          node((0, 0), text(size: 9pt)[*ctx*], fill: rgb("#fff3cd"), corner-radius: 3pt, inset: 5pt, name: <ctx>),
          edge(<ctx>, <op1>, "->"),
          node((1, 0), text(size: 8pt)[op 1], fill: rgb("#d4edda"), corner-radius: 3pt, inset: 4pt, name: <op1>),
          edge(<op1>, <op2>, "->"),
          node((2, 0), text(size: 8pt)[op 2], fill: rgb("#d4edda"), corner-radius: 3pt, inset: 4pt, name: <op2>),
          edge(<op2>, <op3>, "->"),
          node((3, 0), text(size: 8pt)[op N], fill: rgb("#d4edda"), corner-radius: 3pt, inset: 4pt, name: <op3>),
        )
        #text(size: 9pt)[Same ctx reused → *zero allocations* after warmup]
      ]

      #v(0.3em)
      == Complete Example
      #set text(size: 9pt)
      ```cpp
      CsrSetAlgebraContext ctx;  // create once

      auto domain = make_box_device({0,400,0,160});
      auto obstacle = make_disk_device({80,80,20});

      auto fluid = allocate_interval_set_device(
          domain.num_rows,
          domain.num_intervals + obstacle.num_intervals);

      set_difference_device(domain, obstacle, fluid, ctx);
      ```
    ],
    [
      == Chaining with Buffer Reuse
      #set text(size: 9pt)
      ```cpp
      CsrSetAlgebraContext ctx;

      // Pre-allocate output buffers ONCE
      auto set1 = allocate_interval_set_device(512, 2048);
      auto set2 = allocate_interval_set_device(512, 2048);

      // Compute: set1 = A ∪ B
      set_union_device(A, B, set1, ctx);

      // Compute: set2 = set1 \ C
      set_difference_device(set1, C, set2, ctx);

      // ... use set2 (e.g., create Field2D on it) ...

      // Later: reuse same buffers!
      set_intersection_device(D, E, set1, ctx);  // set1 reused
      set_union_device(set1, F, set2, ctx);      // set2 reused
      ```

      #v(0.2em)
      #box(fill: rgb("#d4edda"), inset: 0.4em, radius: 4pt, width: 100%)[
        #set text(size: 10pt)
        *Allocate once* → reuse for entire simulation \
        *ctx + set1 + set2*: zero GPU malloc in hot loop
      ]
    ]
  )
]

// ============================================
// SLIDE 13b: Intersection Internals
// ============================================
#slide(title: "Intersection — How It Works")[
  #set text(size: 10pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.2em,
    [
      == Phase 1: Row Mapping
      #set text(size: 8pt)
      ```
      A.row_keys:  [y=2, y=5, y=8]
      B.row_keys:  [y=3, y=5, y=7, y=8]
                         ↓         ↓
      Binary search: A[1] ↔ B[1], A[2] ↔ B[3]
      ```
      #v(0.2em)
      #box(stroke: 1pt + dark, inset: 0.3em, radius: 3pt, fill: light-gray.lighten(50%))[
        ```
        Output rows:  [y=5, y=8]
        row_index_a:  [ 1,   2 ]
        row_index_b:  [ 1,   3 ]
        ```
      ]

      #v(0.4em)
      == Phase 2: Interval Merge (per row)
      #set text(size: 8pt)
      ```
      A: [===]     [=======]
            2   6       10    18

      B:     [=====] [===]
              4    9  12  16

      Sweep → max(begin), min(end):
        [4,6] ∩  → output [4,6]
        [10,18] ∩ [12,16] → output [12,16]
      ```
      #align(center)[
        #box(fill: rgb("#d4edda"), inset: 0.2em, radius: 3pt)[
          *O(n+m)* per row — linear merge
        ]
      ]
    ],
    [
      == GPU Pattern: Count → Scan → Fill
      #set text(size: 8pt)

      #v(0.2em)
      *1. COUNT* (parallel per row)
      #box(stroke: 1pt + gray, inset: 0.3em, radius: 3pt, width: 100%)[
        ```cpp
        row_counts[i] = count_intersection(
            A.intervals[begin_a..end_a],
            B.intervals[begin_b..end_b]);
        ```
      ]

      #v(0.2em)
      *2. SCAN* (exclusive prefix sum)
      #box(stroke: 1pt + gray, inset: 0.3em, radius: 3pt, width: 100%)[
        ```cpp
        row_ptr_out = exclusive_scan(row_counts)
        // row_ptr_out[i] = where row i starts
        ```
      ]

      #v(0.2em)
      *3. FILL* (parallel per row)
      #box(stroke: 1pt + gray, inset: 0.3em, radius: 3pt, width: 100%)[
        ```cpp
        fill_intersection(
            A.intervals, B.intervals,
            out.intervals, row_ptr_out[i]);
        ```
      ]

      #v(0.3em)
      #align(center)[
        #box(fill: rgb("#fff3cd"), inset: 0.3em, radius: 3pt)[
          Same pattern for ∪, \, ⊕
        ]
      ]
    ]
  )
]

// ============================================
// SLIDE 14: Field Operations
// ============================================
#slide(title: "Field Operations")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      == Basic Operations
      #set text(size: 10pt)
      ```cpp
      // Algebra & reductions
      field_add_device(a, b, result);
      T dot = field_dot_device(a, b);

      // 5-point stencil (W, C, E, S, N)
      apply_csr_stencil_on_set_device(
        dst, src, region,
        KOKKOS_LAMBDA(CsrStencilPoint p) {
          return 0.25 * (p.west + p.east
                       + p.south + p.north);
        });
      ```

      #v(0.5em)
      == AMR: Restrict & Prolong
      #set text(size: 10pt)
      ```cpp
      // Fine → Coarse (average 4 cells)
      restrict_field_device(fine, coarse);

      // Coarse → Fine (interpolation)
      prolong_field_device(coarse, fine);
      ```
    ],
    [
      == Threshold: Field → Geometry
      #set text(size: 10pt)
      ```cpp
      // Select cells where |value| > epsilon
      IntervalSet2DDevice active =
          threshold_field(field, epsilon);
      // Use case: detect shock, refine there
      ```

      #v(0.5em)
      == Remap: Change Geometry
      #set text(size: 10pt)
      ```cpp
      // Project src onto dst geometry
      // (overlap → copy, else → default)
      remap_field_device(src, dst, default_val);
      ```

      #v(0.3em)
      #align(center)[
        #box(fill: light-gray, inset: 0.4em, radius: 4pt)[
          #set text(size: 9pt)
          ```
          src geo:  ████████░░░░░░░░
          dst geo:  ░░░░████████████
          result:   ░░░░████░░░░░░░░
                    copy ↑   ↑ default
          ```
        ]
      ]
    ]
  )
]

// ============================================
// SLIDE 15: Morphology & AMR
// ============================================
#slide(title: "Mathematical Morphology & AMR")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == Dilation / Erosion
      ```cpp
      // N-way union with ±radius offset
      row_n_way_union_impl(rows[], radius, out)

      // N-way intersection with shrink
      row_n_way_intersection_impl(rows[], r, out)
      ```

      #set text(size: 9pt)
      #align(center)[
        ```
        Original:  ░░░░████████░░░░░░░░
        Dilate(1): ░░░█████████░░░░░░░░  (+1 sides)
        Erode(1):  ░░░░░██████░░░░░░░░░  (-1 sides)
        ```
      ]

      == 2D Extension
      - Consider rows y-r to y+r
      - Merge with N-way operation
      - Implicit structuring element (square)
    ],
    [
      == AMR Operations
      ```cpp
      // Coarsening: fine → coarse
      build_row_coarsen_mapping(fine, ws)
      // y_coarse = y_fine / 2, merge X

      // Refinement: coarse → fine
      refine_level_up_device(coarse, ws)
      // [a,b) → [2a, 2b), double Y
      ```

      #set text(size: 9pt)
      #align(center)[
        ```
        Fine (level 1):        Coarse (level 0):
        Y=3: ████████          Y=1: ████████
        Y=2: ████████    →          (merge Y=2,3)
        Y=1: ░░░░████          Y=0: ░░░░████
        Y=0: ░░░░████               (merge Y=0,1)
        ```
      ]

      == Field Transfer
      ```cpp
      // Projection fine → coarse (average)
      // Prolongation coarse → fine (interp)
      build_amr_interval_mapping(coarse, fine)
      ```
    ]
  )
]

// ============================================
// SECTION: DEMO
// ============================================
#section-slide("V. Demo")

// ============================================
// SLIDE 16: Mach2 Cylinder Overview
// ============================================
#slide(title: "Mach2 Cylinder — Multi-Level AMR Simulation")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == Description
      2D compressible flow simulation:
      - *Mach 2* supersonic around a cylinder
      - 1st order Godunov scheme + Rusanov flux
      - *Dynamic AMR*: up to 6 levels

      #v(0.3em)
      == Subsetix Usage
      ```cpp
      // Fluid geometry = domain - obstacle
      auto fluid = set_difference_device(
        make_box_device(domain),
        make_disk_device(cylinder),
        ctx);

      // Conserved fields (ρ, ρu, ρv, E)
      Field2DDevice<Real> rho(fluid);
      Field2DDevice<Real> rhou(fluid);
      // ...
      ```
    ],
    [
      == AMR Architecture
      #set text(size: 10pt)
      #align(center)[
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
      ]

      == Dynamic Refinement
      - Indicator: density gradient
      - `expand_device()` for guard zones
      - Remeshing every N time steps
    ]
  )
]

// ============================================
// SLIDE 17: Mach2 Results
// ============================================
#slide(title: "Mach2 Cylinder — Results & Visualization")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == Generated Outputs
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

      == Execution Command
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
      == Observed Phenomena
      #align(center)[
        #box(fill: light-gray, inset: 0.5em, radius: 4pt)[
          - *Bow shock* in front of the cylinder
          - Subsonic zone in the wake
          - *Von Kármán* vortex street
          - Automatic refinement near the shock
        ]
      ]

      #v(0.3em)
      == Key Technical Points
      - CSR stencil: `apply_csr_stencil_on_set_device()`
      - Struct-of-Arrays for cache efficiency
      - `prolong_guard_from_coarse()`: interpolation
      - `restrict_fine_to_coarse()`: conservation
      - Multi-level VTK export for ParaView

      #v(0.3em)
      #align(center)[
        #box(fill: rgb("#d4edda"), inset: 0.4em, radius: 4pt)[
          *Sparse*: computation only on fluid cells!
        ]
      ]
    ]
  )
]

// ============================================
// SLIDE 18: Live Demo
// ============================================
#slide(title: "Live Demo")[
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
        - Difference (obstacle)
        - CSR display
      ],
      [
        == Operations
        - Union / Intersection
        - Field algebra
        - Stencil
      ],
      [
        == Mach2
        - Launch simulation
        - ParaView visualization
        - AMR in action
      ]
    )

    #v(2em)
    #box(fill: light-gray, inset: 1em, radius: 4pt)[
      _Live demonstration..._
    ]
  ]
]

// ============================================
// SLIDE 19: END
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
  #text(size: 38pt, weight: "bold")[Thank You!]

  #v(0.8em)

  #text(size: 22pt, fill: accent)[Questions?]

  #v(1.5em)

  #set text(size: 13pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      == Key Points
      - CSR interval representation
      - Count-Scan-Fill pattern
      - Kokkos parallelism (CPU/GPU)
      - Workspace for memory reuse
      - Multi-level AMR (Mach2)
    ],
    [
      == Contact
      Sébastien DUBOIS \
      HPC\@Maths Team

      #v(0.5em)
      Code: `include/subsetix/` \
      Demo: `examples/mach2_cylinder/`
    ]
  )
]

// ============================================
// SECTION: APPENDICES
// ============================================
#section-slide("Appendices")

// ============================================
// APPENDIX A: Project Evolution
// ============================================
#slide(title: "Appendix A: Project Evolution")[
  #set text(size: 11pt)

  == Implementation History

  #table(
    columns: (auto, 1fr, auto, auto),
    inset: 5pt,
    align: (center, left, center, center),
    fill: (x, y) => if y == 0 { rgb("#3498db").lighten(70%) } else if calc.odd(y) { rgb("#ecf0f1") } else { white },
    [*Version*], [*Description*], [*Performance*], [*Status*],

    [*v1*],
    [CPU only, Sparse CSR + Workspaces \
     First sequential implementation],
    [Faster than baseline],
    [✓ Stable],

    [*v2*],
    [Multithreaded *Tiled* Sparse CSR \
     OpenMP and TBB backends \
     Tiling for locality],
    [Excellent on large mesh],
    [⚠ Complex \
     Likely bugs],

    [*v3*],
    [CUDA only \
     GPU set algebra \
     Proof of concept],
    [*Fastest*],
    [✓ PoC validated],

    [*v4*],
    [*Kokkos* (current version) \
     Non-tiled Sparse CSR \
     OpenMP + CUDA portability],
    [Slower than v2/v3],
    [✓✓ *Reliable* \
     Verified],
  )

  #v(0.3em)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == Lessons Learned
      - *Tiling* improves locality but greatly increases complexity
      - Native CUDA faster but less portable
      - Kokkos = best *reliability/portability* tradeoff
    ],
    [
      == Final Choice: Kokkos
      - *Single* code for CPU and GPU
      - Simplified maintenance
      - Easy testing and verification
      - Active ecosystem (Sandia, Trilinos)
    ]
  )
]

// ============================================
// APPENDIX B: Why Kokkos
// ============================================
#slide(title: "Appendix B: Why Kokkos?")[
  #set text(size: 11pt)

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      == Comparison with Native CUDA

      #table(
        columns: (auto, 1fr, 1fr),
        inset: 5pt,
        align: (left, center, center),
        fill: (x, y) => if y == 0 { accent.lighten(70%) } else { white },
        [*Aspect*], [*CUDA*], [*Kokkos*],
        [Portability], [NVIDIA only], [*Multi-vendor*],
        [Syntax], [`<<<>>>`], [*C++ standard*],
        [Memory], [cudaMalloc], [*View\<T\*\>*],
        [CPU Debug], [Difficult], [*Easy (Serial)*],
        [Maintenance], [Duplicated code], [*Single code*],
        [Performance], [*Optimal*], [~90-95%],
      )

      #v(0.3em)
      == Supported Backends
      - *OpenMP*: CPU multi-thread
      - *CUDA*: NVIDIA GPU
      - *HIP*: AMD GPU
      - *SYCL*: Intel GPU
      - *Serial*: debug and tests
    ],
    [
      == Benefits for This Project

      #box(fill: rgb("#d4edda"), inset: 0.5em, radius: 4pt)[
        *1. Faster Development* \
        Debug on CPU (Serial/OpenMP), deploy on GPU
      ]

      #v(0.3em)
      #box(fill: rgb("#e8f4f8"), inset: 0.5em, radius: 4pt)[
        *2. Reliable Tests* \
        Same code tested on CPU and GPU \
        No hidden "GPU-only" bugs
      ]

      #v(0.3em)
      #box(fill: rgb("#fff3cd"), inset: 0.5em, radius: 4pt)[
        *3. Std Algorithms* \
        `transform`, `reduce`, `scan`, `copy`... \
        Familiar API, platform-optimized
      ]

      #v(0.3em)
      #box(fill: light-gray, inset: 0.5em, radius: 4pt)[
        *4. Ecosystem* \
        Trilinos, ArborX, Cabana... \
        Sandia National Labs support
      ]
    ]
  )
]

// ============================================
// APPENDIX C: Methodology
// ============================================
#slide(title: "Appendix C: Development Methodology")[
  #set text(size: 12pt)

  == Intensive Use of LLMs

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      === Models Used
      - *Claude Opus 4* (Anthropic)
      - *Claude Sonnet 4* (Anthropic)

      #v(0.5em)
      === Work Pattern
      #box(fill: light-gray, inset: 0.5em, radius: 4pt)[
        ```
        1. PLAN
           Architecture and interfaces
           Discussion of alternatives

        2. QUESTION
           Implementation details
           Edge cases

        3. IMPLEMENTATION
           Code generation
           Review and iteration
        ```
      ]
    ],
    [
      === Observed Benefits
      - *Rapid exploration* of designs
      - Generated inline documentation
      - Automatically suggested tests
      - Assisted refactoring

      #v(0.5em)
      === Points of Attention
      - Systematic code verification
      - LLMs can hallucinate APIs
      - Always compile and test
      - Maintain *architectural control*

      #v(0.5em)
      #box(fill: rgb("#fff3cd"), inset: 0.4em, radius: 4pt)[
        LLM = *accelerator*, not replacement \
        Human expertise remains essential
      ]
    ]
  )
]

// ============================================
// APPENDIX D: References
// ============================================
#slide(title: "Appendix D: References & Resources")[
  #set text(size: 12pt)

  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      == Kokkos
      - Website: kokkos.org
      - GitHub: github.com/kokkos/kokkos
      - Wiki: kokkos.org/kokkos-core-wiki

      #v(0.5em)
      == CUDA
      - CUDA Toolkit Documentation
      - CUDA C++ Programming Guide

      #v(0.5em)
      == Visualization
      - VTK: vtk.org
      - ParaView: paraview.org
    ],
    [
      == Mathematical Morphology
      - Serra, J. "Image Analysis and \
        Mathematical Morphology" (1982)
      - Soille, P. "Morphological Image \
        Analysis" (2003)

      #v(0.5em)
      == Source Code
      ```
      include/subsetix/
      ├── geometry/      # IntervalSet2D
      ├── field/         # Field2D
      ├── csr_ops/       # Algorithms
      ├── multilevel/    # AMR
      └── detail/        # Utilities

      examples/mach2_cylinder/
      └── mach2_cylinder.cpp  # AMR Demo
      ```
    ]
  )
]
