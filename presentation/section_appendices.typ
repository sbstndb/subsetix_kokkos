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

  #align(center + horizon)[
    #block[
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
  ]
]

// ============================================
// APPENDIX B: Why Kokkos
// ============================================
#slide(title: "Appendix B: Why Kokkos?")[
  #set text(size: 11pt)

  #align(center + horizon)[
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
]

// ============================================
// APPENDIX C: Methodology
// ============================================
#slide(title: "Appendix C: Development Methodology")[
  #set text(size: 12pt)

  #align(center + horizon)[
    #grid(
      columns: (1fr, 1fr),
      gutter: 1.5em,
      [
        === Models Used
        - *Claude 4.5 Opus* (Anthropic)
        - *Claude 4.5 Sonnet* (Anthropic)
        - *gpt-5.1-codex-max* (OpenAI)

        #v(0.5em)
        === Work Pattern
        #align(center)[
          #diagram(
            node-stroke: 1.5pt + dark,
            edge-stroke: 1.5pt + accent,
            spacing: (0mm, 10mm),

            // Top: PLAN
            node((0, 0), align(center)[
              #text(size: 10pt, weight: "bold")[PLAN] \
              #text(size: 8pt)[Architecture & interfaces \
              Alternatives]
            ], corner-radius: 4pt, inset: 6pt, fill: light-gray, name: <plan>),

            // Middle: QUESTION
            node((0, 1), align(center)[
              #text(size: 10pt, weight: "bold")[QUESTION] \
              #text(size: 8pt)[Impl details \
              Edge cases]
            ], corner-radius: 4pt, inset: 6pt, fill: light-gray, name: <question>),

            // Bottom: IMPLEMENTATION
            node((0, 2), align(center)[
              #text(size: 10pt, weight: "bold")[IMPLEMENTATION] \
              #text(size: 8pt)[Code generation \
              Review & iteration]
            ], corner-radius: 4pt, inset: 6pt, fill: light-gray, name: <impl>),

            edge(<plan>, <question>, "->"),
            edge(<question>, <impl>, "->"),
          )
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
]
