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
      == Description
      2D compressible flow simulation:
      - *Mach 2* supersonic around a cylinder
      - 1st order Godunov scheme + Rusanov flux
      - *Dynamic AMR*: 4 levels

      #v(0.5em)
      #align(center)[
        #box(fill: rgb("#d4edda"), inset: 0.5em, radius: 4pt)[
          *Sparse*: computation only on fluid cells!
        ]
      ]
    ],
    [
      == Subsetix Setup
      #set text(size: 8pt)

      #box(fill: rgb("#fff3cd"), inset: 0.4em, radius: 4pt, width: 100%)[
        *Geometry Construction*
        ```cpp
        auto fluid = set_difference_device(
          make_box_device(domain),
          make_disk_device(cylinder), ctx);
        ```
      ]

      #v(0.3em)
      #box(fill: rgb("#d4edda"), inset: 0.4em, radius: 4pt, width: 100%)[
        *Conserved Fields* (ρ, ρu, ρv, E)
        ```cpp
        ConservedFields U_coarse(coarse_geo);
        ConservedFields U_fine(fine_geo);
        ```
      ]

      #v(0.3em)
      #box(fill: rgb("#e8f4f8"), inset: 0.4em, radius: 4pt, width: 100%)[
        *Flux on Coarse Level* (excluding fine)
        ```cpp
        auto coarse_active = set_difference_device(
          coarse_geo, fine_projection, ctx);
        apply_stencil_on_set_device(
          flux, U_coarse, coarse_active, Flux{});
        ```
      ]
    ]
  )
]

// ============================================
// SLIDE: Mach2 AMR Operations
// ============================================
#slide(title: "Mach2 Cylinder — AMR Operations")[
  #set text(size: 8pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      #box(fill: rgb("#fff3cd"), inset: 0.4em, radius: 4pt, width: 100%)[
        *1. Gradient Indicator* (detect shock)
        ```cpp
        apply_csr_stencil_on_set_device(
          indicator, U_fine.rho, interior, GradStencil{});
        ```
      ]

      #v(0.3em)
      #box(fill: rgb("#e8f4f8"), inset: 0.4em, radius: 4pt, width: 100%)[
        *2. Threshold → Refinement Mask*
        ```cpp
        auto mask = threshold_field(indicator, thresh);
        ```
      ]

      #v(0.3em)
      #box(fill: rgb("#d4edda"), inset: 0.4em, radius: 4pt, width: 100%)[
        *3. Expand → Guard Zones*
        ```cpp
        expand_device(mask, guard_size, guard_size, fine_geo, ctx);
        ```
      ]
    ],
    [
      #box(fill: rgb("#f8d7da"), inset: 0.4em, radius: 4pt, width: 100%)[
        *4. Inter-level Transfers*
        ```cpp
        restrict_fine_to_coarse(U_coarse, U_fine, overlap);
        prolong_guard_from_coarse(U_fine, guard_region, U_coarse);
        ```
      ]

      #v(0.3em)
      #box(fill: rgb("#e2d6f5"), inset: 0.4em, radius: 4pt, width: 100%)[
        *5. Intersect → Overlap Region*
        ```cpp
        auto overlap = intersect_device(
          coarse_geo, fine_geo, ctx);
        ```
      ]

      #v(0.3em)
      #box(fill: rgb("#fce4ec"), inset: 0.4em, radius: 4pt, width: 100%)[
        *6. Union → Merge Geometries*
        ```cpp
        auto all_refined = union_device(
          mask_level1, mask_level2, ctx);
        ```
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
