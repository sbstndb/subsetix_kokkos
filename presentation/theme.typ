// ============================================
// SUBSETIX KOKKOS - PRESENTATION THEME
// Shared styles and helpers for slides
// ============================================

// HPC@Maths Colors
#let hpc-dark = rgb("#003957")      // Dark blue (H, P, C)
#let hpc-medium = rgb("#046D98")    // Medium blue (Maths)
#let hpc-light = rgb("#5AA0BB")     // Light blue (spiral)

#let accent = hpc-medium
#let dark = hpc-dark
#let light-gray = rgb("#ecf0f1")
#let green = rgb("#27ae60")
#let orange = rgb("#e67e22")

// Shared page and text configuration for slides
#let slide-page-config = (
  paper: "presentation-16-9",
  margin: (x: 1.2cm, y: 1cm),
  numbering: "1 / 1",
  footer: context [
    #set text(size: 10pt, fill: rgb("#7f8c8d"))
    #h(1fr)
    #counter(page).display("1 / 1", both: true)
  ],
)

#let slide-text-config = (
  font: "DejaVu Sans",
  size: 16pt,
)

// Slide helper function
#let slide(title: none, body) = {
  pagebreak(weak: true)
  if title != none {
    align(left, text(size: 24pt, weight: "bold", fill: dark, title))
    line(length: 100%, stroke: 2pt + accent)
    v(0.3em)
  }
  align(horizon, body)
}

// Title slide helper
#let title-slide(title, subtitle: none, author: none, affiliation: none, date: none, logo: none) = {
  set page(
    fill: dark,
    footer: context [
      #set text(size: 10pt, fill: white.transparentize(50%))
      #h(1fr)
      #counter(page).display("1 / 1", both: true)
    ]
  )
  set text(fill: white)

  // Logo at bottom right
  if logo != none {
    place(bottom + right, dx: -0.3cm, dy: -0.8cm)[
      #box(
        fill: white,
        inset: 0.3em,
        radius: 4pt,
        image(logo, width: 3.5cm)
      )
    ]
  }

  align(center + horizon)[
    #text(size: 36pt, weight: "bold", title)
    #if subtitle != none {
      v(0.3em)
      text(size: 20pt, fill: hpc-light, subtitle)
    }
    #if author != none {
      v(1.5em)
      text(size: 16pt, author)
    }
    #if affiliation != none {
      v(0.3em)
      text(size: 14pt, style: "italic", affiliation)
    }
    #if date != none {
      v(0.5em)
      text(size: 14pt, date)
    }
  ]
}

// Section slide
#let section-slide(title) = {
  pagebreak(weak: true)
  set page(
    fill: accent,
    footer: context [
      #set text(size: 10pt, fill: white.transparentize(30%))
      #h(1fr)
      #counter(page).display("1 / 1", both: true)
    ]
  )
  set text(fill: white)
  align(center + horizon)[
    #text(size: 36pt, weight: "bold", title)
  ]
}

// Code block style
#show raw.where(block: true): block.with(
  fill: light-gray,
  inset: 8pt,
  radius: 4pt,
  width: 100%,
)

// Fletcher for diagrams (re-export helpers)
#import "@preview/fletcher:0.4.5" as fletcher: diagram, node, edge
