// Shared theme for slides
#import "theme.typ": slide, title-slide, section-slide, hpc-dark, hpc-medium, hpc-light, accent, dark, light-gray, green, orange, diagram, node, edge, slide-page-config, slide-text-config

// Ensure slide page format (16:9) when compiling this file directly
#set page(..slide-page-config)
#set text(..slide-text-config)

// ============================================
// SECTION: ALGORITHMS
// ============================================
#section-slide("IV. Algorithms")

// ============================================
// SLIDE: Binary Search Lookups
// ============================================
#slide(title: "Binary Search — O(log n) Lookups Everywhere")[
  #set text(size: 10pt)
  CSR structure requires binary search for row and interval lookups — efficient but suboptimal on GPU.

  #v(0.2em)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.2em,
    [
      == CSR Requires Sorted Data
      All lookups rely on binary search:

      #v(0.3em)
      *1. Find row by Y coordinate*
      #box(stroke: 1pt + gray, inset: 0.3em, radius: 3pt, width: 100%, fill: light-gray.lighten(70%))[
        #set text(size: 9pt)
        ```cpp
        int find_row_by_y(row_keys, num_rows, y) {
          // Binary search in row_keys[]
          return lower_bound(row_keys, y);
        }
        ```
      ]
      #align(center)[#text(size: 8pt)[*O(log R)* — R = number of rows]]

      #v(0.3em)
      *2. Find interval by X coordinate*
      #box(stroke: 1pt + gray, inset: 0.3em, radius: 3pt, width: 100%, fill: light-gray.lighten(70%))[
        #set text(size: 9pt)
        ```cpp
        int find_interval_by_x(intervals, begin, end, x) {
          // Binary search in intervals[begin..end]
          return lower_bound(intervals, x);
        }
        ```
      ]
      #align(center)[#text(size: 8pt)[*O(log I#sub[row])* — I = intervals in row]]
    ],
    [
      == Combined: Cell Lookup
      #set text(size: 9pt)
      ```cpp
      T& get(Coord x, Coord y) {
        // Step 1: find row
        int row = find_row_by_y(row_keys, y);
        // Step 2: find interval in row
        int iv = find_interval_by_x(
          intervals, row_ptr[row], row_ptr[row+1], x);
        // Step 3: compute offset
        return values[offsets[iv] + (x - intervals[iv].begin)];
      }
      ```

      #v(0.3em)
      #align(center)[
        #box(fill: rgb("#fff3cd"), inset: 0.4em, radius: 4pt)[
          Total: *O(log R + log I)*
        ]
      ]

      #v(0.3em)
      #align(center)[
        #box(fill: rgb("#ffe4b5"), inset: 0.3em, radius: 4pt)[
          #set text(size: 9pt)
          *GPU*: Binary search = suboptimal \
          (future work)
        ]
      ]
    ]
  )
]


// ============================================
// SLIDE: Set Algebra
// ============================================
#slide(title: "Set Algebra — Binary Operations")[
  #set text(size: 11pt)
  Binary set operations (∪, ∩, \\) combine geometries using a shared workspace to avoid repeated GPU allocations.

  #v(0.2em)
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
// SLIDE: Intersection Internals
// ============================================
#slide(title: "Intersection — How It Works")[
  #set text(size: 10pt)
  Intersection uses the *Count-Scan-Fill* pattern: count output intervals, compute offsets via prefix sum, then fill in parallel.

  #v(0.2em)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.2em,
    [
      == Phase 1: Row Mapping
      #align(center)[
        #diagram(
          node-stroke: 1pt + dark,
          edge-stroke: 1.5pt + accent,
          spacing: (8mm, 12mm),

          // Labels
          node((-0.8, 0), text(size: 8pt, weight: "bold")[A:], stroke: none, fill: none),
          node((-0.8, 1), text(size: 8pt, weight: "bold")[B:], stroke: none, fill: none),

          // A row_keys
          node((0, 0), text(size: 8pt)[y=2], corner-radius: 3pt, fill: rgb("#f8d7da"), inset: 4pt, name: <a0>),
          node((1, 0), text(size: 8pt)[y=5], corner-radius: 3pt, fill: rgb("#d4edda"), inset: 4pt, name: <a1>),
          node((2, 0), text(size: 8pt)[y=8], corner-radius: 3pt, fill: rgb("#d4edda"), inset: 4pt, name: <a2>),

          // B row_keys
          node((0, 1), text(size: 8pt)[y=3], corner-radius: 3pt, fill: rgb("#f8d7da"), inset: 4pt, name: <b0>),
          node((1, 1), text(size: 8pt)[y=5], corner-radius: 3pt, fill: rgb("#d4edda"), inset: 4pt, name: <b1>),
          node((2, 1), text(size: 8pt)[y=7], corner-radius: 3pt, fill: rgb("#f8d7da"), inset: 4pt, name: <b2>),
          node((3, 1), text(size: 8pt)[y=8], corner-radius: 3pt, fill: rgb("#d4edda"), inset: 4pt, name: <b3>),

          // Matching arrows
          edge(<a1>, <b1>, "<->", stroke: 2pt + rgb("#28a745")),
          edge(<a2>, <b3>, "<->", stroke: 2pt + rgb("#28a745")),
        )
      ]
      #set text(size: 8pt)
      #align(center)[
        #box(fill: light-gray.lighten(50%), inset: 0.3em, radius: 3pt)[
          Binary search: O(log n) per row
        ]
      ]

      #v(0.3em)
      == Phase 2: Interval Merge (per row)
      #set text(size: 8pt)
      #align(center)[
        #box(stroke: 1pt + gray, inset: 0.4em, radius: 4pt, fill: light-gray.lighten(70%))[
          #text(weight: "bold")[Row y=5] — 2 intervals each
        ]
      ]
      #v(0.2em)
      #align(center)[
        #diagram(
          node-stroke: 1pt + dark,
          edge-stroke: 1.5pt + rgb("#28a745"),
          spacing: (5mm, 9mm),

          // Row labels
          node((-1.5, 0), text(size: 8pt, weight: "bold")[A:], stroke: none, fill: none),
          node((-1.5, 1), text(size: 8pt, weight: "bold")[B:], stroke: none, fill: none),
          node((-1.5, 2), text(size: 8pt, weight: "bold")[∩:], stroke: none, fill: none),

          // A intervals (2 intervals on same row)
          node((0, 0), text(size: 7pt)[[2,6]], corner-radius: 3pt, fill: hpc-light, inset: 3pt, name: <a1>),
          node((1.8, 0), text(size: 7pt)[[10,18]], corner-radius: 3pt, fill: hpc-light, inset: 3pt, name: <a2>),

          // B intervals (2 intervals on same row)
          node((0, 1), text(size: 7pt)[[4,9]], corner-radius: 3pt, fill: rgb("#fff3cd"), inset: 3pt, name: <b1>),
          node((1.8, 1), text(size: 7pt)[[12,16]], corner-radius: 3pt, fill: rgb("#fff3cd"), inset: 3pt, name: <b2>),

          // Result intervals (2 intervals)
          node((0, 2), text(size: 7pt, weight: "bold")[[4,6]], corner-radius: 3pt, fill: rgb("#d4edda"), inset: 3pt, name: <r1>),
          node((1.8, 2), text(size: 7pt, weight: "bold")[[12,16]], corner-radius: 3pt, fill: rgb("#d4edda"), inset: 3pt, name: <r2>),

          // Merge arrows
          edge(<a1>, <r1>, "->", bend: -20deg),
          edge(<b1>, <r1>, "->", bend: 20deg),
          edge(<a2>, <r2>, "->", bend: -20deg),
          edge(<b2>, <r2>, "->", bend: 20deg),
        )
      ]
      #align(center)[
        #box(fill: rgb("#d4edda"), inset: 0.2em, radius: 3pt)[
          *O(n+m)* sweep — `max(begin)`, `min(end)`
        ]
      ]
    ],
    [
      == GPU Pattern: Count-Scan-Fill
      #set text(size: 8pt)
      *Why?* GPU threads can't dynamically allocate — output size must be known before parallel write.

      #v(0.2em)
      #align(center)[
        #diagram(
          node-stroke: 1.5pt + dark,
          edge-stroke: 2pt + accent,
          spacing: (12mm, 5mm),

          node((0, 0), text(size: 9pt, weight: "bold")[COUNT], corner-radius: 4pt, fill: rgb("#e3f2fd"), inset: 6pt, name: <count>),
          edge(<count>, <scan>, "->"),
          node((1, 0), text(size: 9pt, weight: "bold")[SCAN], corner-radius: 4pt, fill: rgb("#fff3cd"), inset: 6pt, name: <scan>),
          edge(<scan>, <fill>, "->"),
          node((2, 0), text(size: 9pt, weight: "bold")[FILL], corner-radius: 4pt, fill: rgb("#d4edda"), inset: 6pt, name: <fill>),
        )
      ]

      #v(0.3em)
      *1. COUNT* — how many intervals per row?
      #box(stroke: 1pt + gray, inset: 0.2em, radius: 3pt, width: 100%, fill: rgb("#e3f2fd").lighten(70%))[
        ```cpp
        row_counts[i] = count_intersect(A[i], B[i])
        ```
      ]
      #text(size: 7pt, fill: gray)[Parallel per row — don't write yet]

      *2. SCAN* — where does each row start?
      #box(stroke: 1pt + gray, inset: 0.2em, radius: 3pt, width: 100%, fill: rgb("#fff3cd").lighten(70%))[
        ```cpp
        row_ptr = exclusive_scan(row_counts)
        ```
      ]
      #text(size: 7pt, fill: gray)[Prefix sum → row_ptr[i] = write offset]

      *3. FILL* — write results at known offsets
      #box(stroke: 1pt + gray, inset: 0.2em, radius: 3pt, width: 100%, fill: rgb("#d4edda").lighten(70%))[
        ```cpp
        fill_intersect(A[i], B[i], out, row_ptr[i])
        ```
      ]
      #text(size: 7pt, fill: gray)[Parallel per row — no conflicts!]

      #v(0.2em)
      #align(center)[
        #box(fill: rgb("#e8f4f8"), inset: 0.25em, radius: 3pt)[
          Same pattern for *∪*, *\\*, *⊕*
        ]
      ]
    ]
  )
]

// ============================================
// SLIDE: Row Mapping (Prerequisite)
// ============================================
#slide(title: "Row Mapping — Why and How")[
  #set text(size: 9pt)
  GPU parallelization requires knowing output rows before processing — row mapping creates a correspondence table between output and input rows.

  #v(0.2em)
  #grid(
    columns: (1fr, 1.2fr),
    gutter: 1em,
    [
      == GPU Constraint
      #align(center)[
        #box(fill: rgb("#fff3cd"), inset: 0.4em, radius: 4pt)[
          *1 thread = 1 output row*
        ]
      ]
      We need to know output rows *before* parallel processing.

      #v(0.3em)
      == The Mapping Structure
      #set text(size: 8pt)
      #box(stroke: 1pt + gray, inset: 0.3em, radius: 3pt, width: 100%, fill: light-gray.lighten(70%))[
        ```cpp
        struct RowMergeResult {
          row_keys[];    // Y coords of output rows
          row_index_a[]; // index in A (-1 if absent)
          row_index_b[]; // index in B (-1 if absent)
        };
        ```
      ]

      #v(0.3em)
      == Usage in Parallel
      #set text(size: 8pt)
      #box(stroke: 1pt + accent, inset: 0.3em, radius: 3pt, width: 100%, fill: rgb("#d4edda").lighten(70%))[
        ```cpp
        parallel_for(num_rows_out, [&](int i) {
          int ia = row_index_a[i]; // -1 or valid
          int ib = row_index_b[i]; // -1 or valid

          intervals_a = (ia >= 0) ? A.row(ia) : ∅;
          intervals_b = (ib >= 0) ? B.row(ib) : ∅;

          merge(intervals_a, intervals_b, out[i]);
        });
        ```
      ]
      #align(center)[
        #text(size: 7pt)[Each thread knows exactly what to read → *no conflicts*]
      ]
    ],
    [
      == Concrete Example: A ∪ B
      #set text(size: 8pt)

      #align(center)[
        #diagram(
          node-stroke: 1pt + dark,
          spacing: (5mm, 6mm),

          // Labels
          node((-0.8, 0), text(size: 7pt, weight: "bold")[A:], stroke: none, fill: none),
          node((-0.8, 1), text(size: 7pt, weight: "bold")[B:], stroke: none, fill: none),

          // A rows (y = 2, 5, 8)
          node((0, 0), text(size: 6pt)[y=2], corner-radius: 2pt, fill: hpc-light, inset: 3pt),
          node((1, 0), text(size: 6pt)[y=5], corner-radius: 2pt, fill: hpc-light, inset: 3pt),
          node((2, 0), text(size: 6pt)[y=8], corner-radius: 2pt, fill: hpc-light, inset: 3pt),

          // B rows (y = 3, 5, 8, 9)
          node((0, 1), text(size: 6pt)[y=3], corner-radius: 2pt, fill: rgb("#fff3cd"), inset: 3pt),
          node((1, 1), text(size: 6pt)[y=5], corner-radius: 2pt, fill: rgb("#fff3cd"), inset: 3pt),
          node((2, 1), text(size: 6pt)[y=8], corner-radius: 2pt, fill: rgb("#fff3cd"), inset: 3pt),
          node((3, 1), text(size: 6pt)[y=9], corner-radius: 2pt, fill: rgb("#fff3cd"), inset: 3pt),
        )
      ]

      #v(0.3em)
      #align(center)[
        #table(
          columns: (auto, auto, auto, auto, auto),
          inset: 3pt,
          align: center,
          fill: (x, y) => if y == 0 { accent.lighten(70%) } else if calc.rem(y, 2) == 0 { light-gray.lighten(70%) } else { white },
          stroke: 0.5pt + gray,
          [*i*], [*y*], [*idx\_a*], [*idx\_b*], [*Signification*],
          [0], [2], [0], [-1], [A\[0\] seul],
          [1], [3], [-1], [0], [B\[0\] seul],
          [2], [5], [1], [1], [A\[1\] ∪ B\[1\]],
          [3], [8], [2], [2], [A\[2\] ∪ B\[2\]],
          [4], [9], [-1], [3], [B\[3\] seul],
        )
      ]

      #v(0.3em)
      #align(center)[
        #box(fill: rgb("#e8f4f8"), inset: 0.3em, radius: 4pt)[
          #set text(size: 8pt)
          *-1* = ligne absente dans ce set \
          Le mapping est construit par recherche binaire
        ]
      ]

      #v(0.2em)
      #align(center)[
        #table(
          columns: (auto, auto),
          inset: 3pt,
          align: center,
          fill: (x, y) => if y == 0 { accent.lighten(70%) } else { white },
          stroke: 0.5pt + gray,
          [*Op*], [*Mapping*],
          [A ∩ B], [Garde les y communs],
          [A ∪ B], [Fusionne tous les y],
          [A \\ B], [Garde les y de A],
        )
      ]
    ]
  )
]

// ============================================
// SLIDE: AMR Restrict & Prolong
// ============================================
#slide(title: "AMR — Restrict & Prolong")[
  #set text(size: 10pt)
  Inter-level transfers use 2:1 coordinate mapping — restrict averages 4 fine cells, prolong interpolates with gradient correction.

  #v(0.2em)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == Coordinate Mapping (2:1)
      #align(center)[
        #diagram(
          node-stroke: 1pt + dark,
          edge-stroke: 1.5pt + rgb("#28a745"),
          spacing: (8mm, 8mm),

          // Coarse cell
          node((0, 0), text(size: 9pt, weight: "bold")[C], corner-radius: 3pt, fill: rgb("#fff3cd"), inset: 8pt, name: <coarse>),
          node((0, 0.8), text(size: 7pt)[coarse(x,y)], stroke: none, fill: none),

          // Arrow
          node((1.5, 0), text(size: 10pt)[↔], stroke: none, fill: none),

          // Fine cells (2x2 grid)
          node((2.5, -0.4), text(size: 7pt)[00], corner-radius: 2pt, fill: rgb("#d4edda"), inset: 5pt, name: <f00>),
          node((3.3, -0.4), text(size: 7pt)[01], corner-radius: 2pt, fill: rgb("#d4edda"), inset: 5pt, name: <f01>),
          node((2.5, 0.4), text(size: 7pt)[10], corner-radius: 2pt, fill: rgb("#d4edda"), inset: 5pt, name: <f10>),
          node((3.3, 0.4), text(size: 7pt)[11], corner-radius: 2pt, fill: rgb("#d4edda"), inset: 5pt, name: <f11>),
          node((2.9, 1), text(size: 7pt)[fine(2x+i, 2y+j)], stroke: none, fill: none),
        )
      ]

      #v(0.3em)
      == Restrict (Fine → Coarse)
      #box(stroke: 1pt + gray, inset: 0.3em, radius: 3pt, width: 100%, fill: rgb("#fff3cd").lighten(70%))[
        ```cpp
        coarse[x,y] = 0.25 * (
          fine[2x,   2y  ] + fine[2x+1, 2y  ] +
          fine[2x,   2y+1] + fine[2x+1, 2y+1]
        );
        ```
      ]
      #align(center)[
        #text(size: 8pt)[Average of 4 fine cells → 1 coarse cell]
      ]

      #v(0.3em)
      == Floor/Ceil for Negative Coords
      #set text(size: 8pt)
      #box(stroke: 1pt + gray, inset: 0.2em, radius: 3pt, width: 100%, fill: rgb("#e8f4f8"))[
        ```cpp
        // Handles negative coordinates correctly
        floor_div2(x) = (x>=0) ? x/2 : (x-1)/2;
        ceil_div2(x)  = (x>=0) ? (x+1)/2 : x/2;
        // Example: floor_div2(-3) = -2, not -1
        ```
      ]
    ],
    [
      == Prolong (Coarse → Fine)

      *1. Injection* (simple copy)
      #box(stroke: 1pt + gray, inset: 0.2em, radius: 3pt, width: 100%, fill: rgb("#d4edda").lighten(70%))[
        ```cpp
        fine[x,y] = coarse[x/2, y/2];
        ```
      ]

      #v(0.2em)
      *2. Linear Prediction* (with gradients)
      #align(center)[
        #diagram(
          node-stroke: 1pt + dark,
          spacing: (6mm, 6mm),

          // Coarse stencil
          node((1, 0), text(size: 7pt)[U], corner-radius: 2pt, fill: hpc-light.lighten(50%), inset: 3pt),
          node((0, 1), text(size: 7pt)[L], corner-radius: 2pt, fill: hpc-light.lighten(50%), inset: 3pt),
          node((1, 1), text(size: 7pt, weight: "bold")[C], corner-radius: 2pt, fill: rgb("#fff3cd"), inset: 3pt, name: <cc>),
          node((2, 1), text(size: 7pt)[R], corner-radius: 2pt, fill: hpc-light.lighten(50%), inset: 3pt),
          node((1, 2), text(size: 7pt)[D], corner-radius: 2pt, fill: hpc-light.lighten(50%), inset: 3pt),

          // Fine output
          node((4, 0.7), text(size: 6pt)[−−], corner-radius: 2pt, fill: rgb("#d4edda"), inset: 3pt),
          node((4.6, 0.7), text(size: 6pt)[+−], corner-radius: 2pt, fill: rgb("#d4edda"), inset: 3pt),
          node((4, 1.3), text(size: 6pt)[−+], corner-radius: 2pt, fill: rgb("#d4edda"), inset: 3pt),
          node((4.6, 1.3), text(size: 6pt)[++], corner-radius: 2pt, fill: rgb("#d4edda"), inset: 3pt),
        )
      ]

      #box(stroke: 1pt + gray, inset: 0.2em, radius: 3pt, width: 100%, fill: rgb("#e8f4f8"))[
        #set text(size: 8pt)
        ```cpp
        grad_x = 0.125 * (R - L);
        grad_y = 0.125 * (U - D);
        sign_x = (x%2==0) ? -1 : +1;
        sign_y = (y%2==0) ? -1 : +1;
        fine = C + sign_x*grad_x + sign_y*grad_y;
        ```
      ]

      #v(0.3em)
      #align(center)[
        #box(fill: rgb("#d4edda"), inset: 0.3em, radius: 3pt)[
          *O(1)* per cell with pre-computed mapping
        ]
      ]
    ]
  )
]

// ============================================
// SLIDE: Field Operations
// ============================================
#slide(title: "Field Operations")[
  #set text(size: 11pt)
  Field operations include algebra, stencil application, and geometry-aware transforms (threshold, remap) for AMR workflows.

  #v(0.2em)
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

