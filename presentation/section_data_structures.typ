// Shared theme for slides
#import "theme.typ": slide, title-slide, section-slide, hpc-dark, hpc-medium, hpc-light, accent, dark, light-gray, green, orange, diagram, node, edge

// Ensure slide page format (16:9) when compiling this file directly
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

      #v(0.2em)
      #align(center)[
        #box(fill: rgb("#fff3cd"), inset: 0.3em, radius: 4pt)[
          #set text(size: 9pt)
          *Note*: Outputs must be pre-allocated \
          Use `allocate_interval_set_device()`
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
