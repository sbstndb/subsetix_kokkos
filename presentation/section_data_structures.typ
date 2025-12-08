// Shared theme for slides
#import "theme.typ": slide, title-slide, section-slide, hpc-dark, hpc-medium, hpc-light, accent, dark, light-gray, green, orange, diagram, node, edge, slide-page-config, slide-text-config

// Ensure slide page format (16:9) when compiling this file directly
#set page(..slide-page-config)
#set text(..slide-text-config)

// ============================================
// SECTION: DATA STRUCTURES
// ============================================
#section-slide("III. Data Structures")

// ============================================
// SLIDE: Overview — Device Structures
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

          // Top row: Field and Region
          node((0, 0), align(center)[
            #text(size: 10pt, weight: "bold")[Field2DDevice\<T\>] \
            #text(size: 8pt)[values[] + geometry]
          ], corner-radius: 4pt, width: 45mm, inset: 6pt, fill: rgb("#d4edda"), name: <field>),

          node((1, 0), align(center)[
            #text(size: 10pt, weight: "bold")[IntervalSet2DDevice] \
            #text(size: 8pt)[Region / mask]
          ], corner-radius: 4pt, width: 40mm, inset: 6pt, fill: rgb("#fff3cd"), name: <region>),

          // Middle: geometric subset = intersection(Field.geometry, Region)
          node((0.5, 1), align(center)[
            #text(size: 10pt, weight: "bold")[IntervalSubSet2DDevice] \
            #text(size: 8pt)[Geo subset \
            intersection(Field.geometry, Region)]
          ], corner-radius: 4pt, width: 60mm, inset: 6pt, name: <subset>),

          // Bottom: SubField = view using subset to access values
          node((0.5, 2), align(center)[
            #text(size: 10pt, weight: "bold")[Field2DSubViewDevice\<T\>] \
            #text(size: 8pt)["SubField" = Field + Geo subset \
            no extra values]
          ], corner-radius: 4pt, width: 60mm, inset: 6pt, fill: rgb("#e8f4f8"), name: <subview>),

          // Data flow
          edge(<field>, <subset>, "->"),
          edge(<region>, <subset>, "->"),
          edge(<field>, <subview>, "->"),
          edge(<subset>, <subview>, "->"),
        )
      ]

      #v(0.3em)
      == Device-Side Grammar
      #set text(size: 9pt)
      - `Field2DDevice<T>` = field values + `IntervalSet2DDevice geometry`
      - `IntervalSet2DDevice` (region) = mask / target cells
      - `IntervalSubSet2DDevice` = geo subset = `geometry ∩ region`
      - `Field2DSubViewDevice<T>` = "SubField" = `Field + IntervalSubSet2DDevice`

      #v(0.2em)
      #set text(size: 8pt)
      `Field2DSubViewDevice<T>` internally uses `IntervalSubSet2DDevice` \
      to iterate only on active cells.
    ],
    [
      == SubField: Usage Example
      #set text(size: 8pt)
      ```cpp
      Field2DDevice<Real> rho(fluid_geo);      // Field

      // Region = any IntervalSet2DDevice (BC, AMR, overlap)
      IntervalSet2DDevice left_bc = make_box_device({0,2,0,ny});

      // SubField = rho restricted to left_bc
      Field2DSubViewDevice<Real> sub = make_subview(rho, left_bc);

      // Apply operations only on this region
      fill_subview_device(sub, rho_inlet);
      apply_stencil_on_subview_device(sub, bc_stencil);
      ```

      #v(0.3em)
      == SubView Operations
      - `fill_subview_device(sub, val)`
      - `scale_subview_device(sub, alpha)`
      - `copy_subview_device(dst, src)`
      - `apply_stencil_on_subview_device(...)`

      #v(0.3em)
      #set text(size: 8pt)
      In `mach2_cylinder`, overlap/guard regions and AMR masks \
      use exactly this SubField (SubView) + region mask design.
    ]
  )
]

// ============================================
// SLIDE: IntervalSet2D Structure
// ============================================
#slide(title: "IntervalSet2D — Complete CSR Structure")[
  #set text(size: 11pt)
  Sparse 2D geometry stored as *rows of X-intervals*, indexed by Y coordinate — similar to CSR matrix format.

  #v(0.3em)
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
      == Basic Types
      #set text(size: 10pt)
      ```cpp
      using Coord = std::int32_t;

      struct Interval {
        Coord begin = 0;  // Inclusive
        Coord end = 0;    // Exclusive
      };

      struct RowKey2D {
        Coord y = 0;
      };
      ```

      #v(0.3em)
      == Invariants
      #set text(size: 10pt)
      - `row_keys` sorted by increasing Y
      - Intervals sorted by X within each row
      - No overlap between intervals
      - `row_ptr[r+1] - row_ptr[r]` = nb intervals row r
    ]
  )
]

// ============================================
// SLIDE: Field2D
// ============================================
#slide(title: "Field2D — Field on Sparse Geometry")[
  #set text(size: 11pt)
  Associates a *contiguous array of values* with each cell of an IntervalSet2D geometry.

  #v(0.3em)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      == Definition

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
      // O(1) - interval index + x coordinate
      T val = field.at(interval_idx, x);
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
// SLIDE: SubSet — Targeted Region Operations
// ============================================
#slide(title: "SubSet — Targeted Region Operations")[
  #set text(size: 11pt)
  Represents a *subset of the parent geometry* (intersection with a mask) — used by SubFields to restrict operations to specific cells.

  #v(0.3em)
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

      #v(0.3em)
      #align(center)[
        #box(fill: rgb("#fff3cd"), inset: 0.3em, radius: 4pt)[
          #text(size: 8pt)[⚠️ Structure too complex — needs simplification]
        ]
      ]
    ]
  )
]

// ============================================
// SLIDE: Field2DSubView
// ============================================
#slide(title: "Field2DSubView — View on Field + Region")[
  #set text(size: 11pt)
  Combines a Field with a target region for *localized operations* — lazy intersection, cached for reuse.

  #v(0.3em)
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
      Field2DSubViewDevice<T> sub = make_subview(field, region);
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
// SLIDE: Workspace & AMR
// ============================================
#slide(title: "Workspace & AMR Support")[
  #set text(size: 11pt)
  Reusable buffer pool to avoid repeated GPU allocations, and multi-resolution grid structure for AMR.

  #v(0.3em)
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
