// Shared theme for slides
#import "theme.typ": slide, title-slide, section-slide, hpc-dark, hpc-medium, hpc-light, accent, dark, light-gray, green, orange, diagram, node, edge, slide-page-config, slide-text-config

// Ensure slide page format (16:9) when compiling this file directly
#set page(..slide-page-config)
#set text(..slide-text-config)

// ============================================
// SECTION: SPARSE REPRESENTATION
// ============================================
#section-slide("II. Sparse Representation")

// ============================================
// SLIDE: 2D Sparse Mesh Example
// ============================================
#slide(title: "Example: 2D Sparse Mesh with Intervals")[
  #set text(size: 11pt)
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      == "Smiley" Geometry :-)
      #align(center)[
        #box(stroke: 1pt + dark, inset: 0.5em, radius: 4pt)[
          ```
          Y
          9│ . . . . . . . . . .    (empty)
          8│ . . . . . . . . . .    (empty)
          7│ . . ▓ ▓ . . ▓ ▓ . .    EYES
          6│ . . ▓ ▓ . . ▓ ▓ . .    EYES
          5│ . . . . . . . . . .    (HOLE)
          4│ . . . . . . . . . .    (HOLE)
          3│ . ▓ ▓ . . . . ▓ ▓ .    SMILE
          2│ . . ▓ ▓ . . ▓ ▓ . .    SMILE
          1│ . . . ▓ ▓ ▓ ▓ . . .    SMILE
          0│ . . . . . . . . . .    (empty)
           └──────────────────── X
             0 1 2 3 4 5 6 7 8 9
          ```
        ]
      ]

    ],
    [
      == Sparse-CSR-like Representation
      ```cpp
      // 5 rows, HOLE Y=4,5
      row_keys = [1, 2, 3, 6, 7]  // skips 4,5!
      num_rows = 5

      // Rows with 1 or 2 intervals
      row_ptr = [0, 1, 3, 5, 7, 9]

      intervals = [
        {3, 7},        // Y=1: smile bottom
        {2, 4}, {6, 8},// Y=2: smile thick
        {1, 3}, {7, 9},// Y=3: smile corners
        {2, 4}, {6, 8},// Y=6: EYES bottom
        {2, 4}, {6, 8},// Y=7: EYES top
      ]
      num_intervals = 9

      cell_offsets = [0,4,6,8,10,12,14,16,18,20]
      total_cells = 20
      ```

      #align(center)[
        #box(fill: rgb("#e8f4f8"), inset: 0.3em, radius: 4pt)[
          *Hole Y=4,5*: row_keys jumps from 3 to 6
        ]
      ]
    ]
  )
]
