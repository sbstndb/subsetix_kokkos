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
// SLIDE: Mach2 Cylinder Overview
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
      - *Dynamic AMR*: 4 levels

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
// SLIDE: Mach2 Results
// ============================================
#slide(title: "Mach2 Cylinder — Results & Visualization")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == Generated Outputs
      #set text(size: 9pt)
      ```
      mach2_cylinder/
      ├── fluid_geometry.vtk
      ├── obstacle_geometry.vtk
      ├── refine_mask_lvl{1,2,3}.vtk
      ├── fine_geometry_lvl{1,2,3}.vtk
      ├── step_00001_density.vtk
      ├── step_00001_l0_density.vtk
      ├── step_00001_l1_density.vtk
      ├── step_00001_mach.vtk
      ├── step_00001_pressure.vtk
      └── ...
      ```

      == Execution Command
      #set text(size: 9pt)
      ```bash
      ./mach2_cylinder \
        --nx 400 --ny 160 \
        --radius 20 \
        --mach-inlet 2.0 \
        --max-steps 5000 \
        --output-stride 50 \
        --amr --amr-levels 4
      ```
    ],
    [
      == Observed Phenomena
      #align(center)[
        #box(fill: light-gray, inset: 0.5em, radius: 4pt)[
          - *Bow shock* in front of the cylinder
          - Density/pressure gradient captured
          - AMR refinement follows the shock
        ]
      ]

      #v(0.3em)
      == Key Technical Points
      #set text(size: 10pt)
      - 1st order Godunov + Rusanov flux
      - Struct-of-Arrays: `ConservedFields` (ρ, ρu, ρv, E)
      - `threshold_field()` → detect shock gradient
      - `expand_device()` → guard cells around refined zone
      - `restrict_fine_to_coarse()` / `prolong_guard_from_coarse()`
      - `write_multilevel_field_vtk()` for ParaView

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
