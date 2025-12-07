// ============================================
// SUBSETIX KOKKOS - PRESENTATION
// ============================================

// Shared theme (styles, colors, helpers)
#import "theme.typ": slide, title-slide, section-slide, hpc-dark, hpc-medium, hpc-light, accent, dark, light-gray, green, orange, diagram, node, edge, slide-page-config, slide-text-config

// Base page/text styles for the deck
#set page(..slide-page-config)
#set text(..slide-text-config)

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
#include "section_context.typ"

// ============================================
// SECTION: SPARSE REPRESENTATION
// ============================================
#include "section_sparse.typ"

// ============================================
// SECTION: DATA STRUCTURES
// ============================================
#include "section_data_structures.typ"

// ============================================
// SECTION: ALGORITHMS
// ============================================
#include "section_algorithms.typ"

// ============================================
// SECTION: DEMO
// ============================================
#include "section_demo.typ"

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
#include "section_appendices.typ"
