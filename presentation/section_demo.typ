// Shared theme for slides
#import "theme.typ": slide, title-slide, section-slide, hpc-dark, hpc-medium, hpc-light, accent, dark, light-gray, green, orange, diagram, node, edge, slide-page-config, slide-text-config

// Ensure slide page format (16:9) when compiling this file directly
#set page(..slide-page-config)
#set text(..slide-text-config)

// ============================================
// SECTION: DEMO
// ============================================
#section-slide("V. Demo")

// ============================================
// SLIDE: Mach2 Problem & Setup
// ============================================
#slide(title: "Mach2 Cylinder — Problem & Setup")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      == Physical Problem
      *Supersonic flow around a cylinder* — classic CFD benchmark for shock capturing.

      #v(0.3em)
      #box(stroke: 1pt + dark, inset: 0.5em, radius: 4pt, width: 100%, fill: light-gray.lighten(70%))[
        #align(center)[
          #diagram(
            node-stroke: 1pt + dark,
            edge-stroke: 1.5pt + accent,
            spacing: (8mm, 8mm),

            // Domain box
            node((0, 0), box(width: 80mm, height: 35mm, stroke: 1.5pt + dark, fill: rgb("#e8f4f8").lighten(50%))[
              #place(center + horizon)[
                #circle(radius: 8mm, fill: gray.lighten(30%), stroke: 1.5pt + dark)
              ]
              #place(left + horizon, dx: 2mm)[
                #text(size: 14pt, fill: accent)[→ → →]
              ]
              #place(right + horizon, dx: -6mm)[
                #text(size: 14pt, fill: accent)[→ →]
              ]
            ], stroke: none, inset: 0pt),
          )
        ]
        #align(center)[
          #text(size: 8pt)[Mach 2 inlet #h(2em) Cylinder obstacle #h(2em) Outflow]
        ]
      ]

      #v(0.3em)
      == Boundary Conditions
      #set text(size: 10pt)
      - *Left*: Supersonic inlet (Mach 2, fixed state)
      - *Right*: Supersonic outlet (extrapolation)
      - *Top/Bottom*: Reflective walls (slip)
      - *Cylinder*: Solid wall (reflective)
    ],
    [
      == Numerical Method
      #set text(size: 10pt)
      - *Equations*: 2D Euler (compressible, inviscid)
      - *Variables*: ρ, ρu, ρv, E (density, momentum, energy)
      - *Scheme*: 1st order finite volume, Rusanov flux
      - *Gas*: Ideal gas, γ = 1.4

      #v(0.5em)
      == Adaptive Mesh Refinement
      #set text(size: 10pt)
      - *4 levels* of refinement (factor 2 per level)
      - *Criterion*: Density gradient magnitude
      - *Dynamic*: Regrid every step
      - *Guard zones*: Smooth transitions between levels

      #v(0.5em)
      #align(center)[
        #box(fill: rgb("#d4edda"), inset: 0.5em, radius: 4pt)[
          *Sparse geometry*: only fluid cells stored & computed
        ]
      ]

      #v(0.3em)
      #align(center)[
        #box(fill: rgb("#fff3cd"), inset: 0.4em, radius: 4pt)[
          #set text(size: 9pt)
          *Bow shock* forms in front of cylinder \
          AMR refines automatically near discontinuity
        ]
      ]
    ]
  )
]

// ============================================
// SLIDE: Mach2 AMR Operations
// ============================================
#slide(title: "Mach2 Cylinder — Subsetix Usage")[
  #set text(size: 7pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == 1. Fluid Geometry
      #box(fill: rgb("#e8f4f8"), inset: 0.4em, radius: 4pt, width: 100%)[
        ```cpp
        auto domain = make_box_device({0, nx, 0, ny});
        auto obstacle = make_disk_device({cx, cy, radius});
        auto fluid = allocate_interval_set_device(...);
        set_difference_device(domain, obstacle, fluid, ctx);
        ```
      ]

      #v(0.2em)
      == 2. Refinement Mask (detect shock)
      #box(fill: rgb("#fff3cd"), inset: 0.4em, radius: 4pt, width: 100%)[
        ```cpp
        IntervalSet2DDevice interior;
        shrink_device(fluid, 1, 1, interior, ctx);

        Field2DDevice<Real> indicator(interior);
        apply_csr_stencil_on_set_device(
            indicator, rho, interior, GradientStencil{});

        auto mask = threshold_field(indicator, thresh);
        ```
      ]

      #v(0.2em)
      == 3. Guard Zone Extraction
      #box(fill: rgb("#d4edda"), inset: 0.4em, radius: 4pt, width: 100%)[
        ```cpp
        IntervalSet2DDevice expanded;
        expand_device(mask, guard, guard, expanded, ctx);

        auto with_guard = allocate_interval_set_device(...);
        set_intersection_device(expanded, fluid, with_guard, ctx);

        auto guard_only = allocate_interval_set_device(...);
        set_difference_device(with_guard, mask, guard_only, ctx);
        ```
      ]
    ],
    [
      == 4. Coarse Active (exclude fine level)
      #box(fill: rgb("#f8d7da"), inset: 0.4em, radius: 4pt, width: 100%)[
        ```cpp
        IntervalSet2DDevice fine_proj;
        project_level_down_device(fine_geo, fine_proj, ctx);

        auto coarse_active = allocate_interval_set_device(...);
        set_difference_device(coarse_geo, fine_proj, coarse_active, ctx);
        ```
      ]

      #v(0.2em)
      == 5. Field Geometry (fluid + obstacle boundary)
      #box(fill: rgb("#e2d6f5"), inset: 0.4em, radius: 4pt, width: 100%)[
        ```cpp
        IntervalSet2DDevice expanded;
        expand_device(with_guard, 2, 2, expanded, ctx);

        auto obs_clip = allocate_interval_set_device(...);
        set_intersection_device(expanded, obstacle, obs_clip, ctx);

        auto field_geo = allocate_interval_set_device(...);
        set_union_device(with_guard, obs_clip, field_geo, ctx);
        ```
      ]

      #v(0.3em)
      #align(center)[
        #box(fill: rgb("#d4edda"), inset: 0.5em, radius: 4pt)[
          #set text(size: 9pt)
          *Même `ctx`* pour toutes les opérations \
          → *zéro allocation GPU* après warmup
        ]
      ]
    ]
  )
]

// ============================================
// SLIDE: Mach2 Visual Results
// ============================================
#slide(title: "Mach2 Cylinder — Visual Results")[
  #set text(size: 10pt)
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 0.8em,
    [
      == Density Field
      #align(center)[
        #image("mach2_field.png", width: 100%)
      ]
      #align(center)[
        #text(size: 8pt)[Bow shock in front of cylinder \ Colormap: blue (low) → red (high)]
      ]
    ],
    [
      == AMR Levels
      #align(center)[
        #image("mach2_levels.png", width: 100%)
      ]
      #align(center)[
        #text(size: 8pt)[Automatic refinement zones \ near the shock front]
      ]
    ],
    [
      == Mesh Zoom
      #align(center)[
        #image("mach2_zoom.png", width: 100%)
      ]
      #align(center)[
        #text(size: 8pt)[Multi-level AMR resolution \ near the bow shock]
      ]
    ]
  )

  #v(0.5em)
  #align(center)[
    #box(fill: rgb("#d4edda"), inset: 0.5em, radius: 4pt)[
      #text(size: 11pt)[*4 AMR levels* (9–12) — Automatic refinement based on density gradient]
    ]
  ]
]
