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
          [FP32], [*90 TFlops*], [~14 TFlops],
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
