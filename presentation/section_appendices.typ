// Shared theme for slides
#import "theme.typ": slide, title-slide, section-slide, hpc-dark, hpc-medium, hpc-light, accent, dark, light-gray, green, orange, diagram, node, edge, slide-page-config, slide-text-config

// Ensure slide page format (16:9) when compiling this file directly
#set page(..slide-page-config)
#set text(..slide-text-config)

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
