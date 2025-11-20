# Mach 2 Cylinder (Pixel Mask) – Plan & Impl Snapshot

## Impl snapshot (current code)
- New target `mach2_cylinder` (`examples/mach2_cylinder/mach2_cylinder.cpp`) wired in `examples/CMakeLists.txt`; CUDA preset builds and runs.
- Geometry: box minus disk via `set_difference_device`; VTK dumps for `fluid_geometry.vtk` and `obstacle_geometry.vtk`.
- Scheme: first-order Godunov with Rusanov (HLL) flux, ideal gas EOS; slip walls by default, `--no-slip` zeroes (u,v) at walls/obstacle ghosts.
- BCs: supersonic inlet (Mach 2 default), supersonic outflow (extrap), slip top/bottom, obstacle as pixel mask; steady CFL timestep with global reduce.
- Outputs (every `--output-stride`): density, pressure, Mach; files named `step_<N>_{density,pressure,mach}.vtk` under `examples_output/mach2_cylinder`.
- CLI: `--nx/--ny/--cx/--cy/--radius/--mach-inlet/--rho/--p/--gamma/--cfl/--t-final/--max-steps/--output-stride/--no-slip`.

## Goals
- Add a new example next to `examples/amr_advection`: a uniform-grid FV solver for 2D compressible flow (Mach 2 inlet, cylinder obstacle represented as a pixel mask).
- Leverage existing subsetix CSR geometry/field utilities, keep the obstacle as a bitmap (no smoothing), and export ParaView-friendly VTK outputs.
- Keep the implementation device-compatible (Kokkos) and runnable via the existing CMake `examples` target presets.

## Subsetix Building Blocks to Reuse
- Geometry builders: `make_box_device`, `make_disk_device`, `build_device_from_host` from `include/subsetix/csr_interval_set.hpp`; set logic `set_difference_device` from `include/subsetix/csr_ops/core.hpp` to carve the obstacle out of the domain.
- Fields and accessors: `Field2DDevice<T>` and `detail::FieldReadAccessor` from `include/subsetix/csr_field.hpp` and `include/subsetix/csr_ops/field_stencil.hpp` for neighbor reads on sparse rows.
- Stencil application: `apply_stencil_on_set_device` from `include/subsetix/csr_ops/field_stencil.hpp` to update only fluid cells.
- Mask/subviews: `make_subview` and related helpers in `include/subsetix/csr_ops/field_subview.hpp` for targeted updates if we need to touch only boundary bands.
- VTK export: `subsetix::vtk::write_legacy_quads` for geometry/fields in `include/subsetix/vtk_export.hpp`; output directory helper `examples/example_output.hpp`.

## Directory and Build Targets
- New folder: `examples/mach2_cylinder/`.
- Main file: `examples/mach2_cylinder/mach2_cylinder.cpp`.
- Hook into `examples/CMakeLists.txt` with an executable `mach2_cylinder` linking to `subsetix_core`.
- Outputs land in `examples_output/mach2_cylinder` by default (override via `--output-dir`).

## Geometry and Masks
- Domain: rectangular box `[0, nx) × [0, ny)` built with `make_box_device(Box2D{0, nx, 0, ny})`; `nx`, `ny` configurable CLI flags (defaults e.g. 400 × 160).
- Obstacle: disk mask via `make_disk_device(Disk2D{cx, cy, radius})`, center default at mid-height/lower-third; radius configurable.
- Fluid geometry: `set_difference_device(domain, obstacle, fluid, ctx)` where `ctx` is a reusable `csr::Context` accumulator (follow existing examples).
- Store obstacle mask as a geometry/field for diagnostics and to distinguish domain edges vs. internal solids during boundary handling.

## State Layout
- Conserved variables per cell: struct `Conserved { double rho, rhou, rhov, E; };` held in `Field2DDevice<Conserved> U`.
- Optional scratch fields: `Field2DDevice<Conserved> U_next`, `Field2DDevice<double> pressure`, `Field2DDevice<double> mach`, `Field2DDevice<double> cfl_local` for diagnostics.
- Accessors: build `detail::FieldReadAccessor` for `U` and a `Field2DDevice<std::uint8_t>`/boolean flag for the obstacle to query neighbors (fluid/solid/out-of-domain).

## Numerics and Update Loop
- Fluxes: HLLC (preferred) or HLL Riemann solver on each face; MUSCL-Hancock reconstruction optional after baseline first-order is stable. Current implementation uses first-order Rusanov (HLL) only.
- Time step:
  1. Compute local wave speeds and CFL-limited `dt` over fluid cells (`apply_on_set_device` or `apply_stencil_on_set_device`).
  2. For each fluid cell, compute fluxes on four faces:
     - Fluid–fluid: standard Riemann between left/right states.
     - Fluid–solid: build ghost/image state reflecting normal velocity (slip/no-slip toggle), zero mass flux expected; reuse face normal (+/-x, +/-y).
     - Solid–solid: flux zero (skip).
     - Domain exterior: inlet/outlet/top/bottom BCs as ghost states (see below).
  3. Update `U_next = U - dt * (flux_x + flux_y) / cell_area`; swap.
- Equation of state: ideal gas (`gamma` configurable, default 1.4). Functions to convert conserved <-> primitive for flux calculations.

## Boundary Conditions
- Inlet (x=0): supersonic fixed state with Mach 2 (set `rho`, `u`, `v=0`, `p`); ghost state equals inflow.
- Outlet (x=nx-1): supersonic outflow via zero-gradient/extrapolated ghost.
- Top/bottom (y=0/ny-1): slip wall (reflect normal velocity) by default; allow toggling to far-field copy if desired.
- Obstacle faces: same slip/no-slip handling as above using the obstacle mask instead of domain bounds.

## Output and Diagnostics
- VTK dumps every `output_stride` steps: density, pressure, Mach, and optionally `|u|` magnitude; also export `fluid` and `obstacle` geometries for debugging stair-cased walls.
- Naming: `step_%05d_density.vtk`, `step_%05d_mach.vtk`, `geometry.vtk` (one-off).
- Lightweight ASCII log with step, time, `dt_min`, and max Mach to detect blow-ups.

## CLI Parameters (suggested)
- `--nx`, `--ny` (ints); `--radius`, `--cx`, `--cy` (ints, pixel space).
- `--mach-inlet`, `--rho`, `--p`, `--gamma`.
- `--cfl`, `--t-final`, `--max-steps`, `--output-stride`, `--output-dir`.
- `--no-slip` toggle for obstacle walls (default slip).

## Validation and Checks
- Smoke runs: small grid (e.g., 128 × 64) for 200–500 steps to verify stable fields and zero mass flux across obstacle faces.
- Diagnostics: compute integrated mass in fluid region each output; warn if drifting.
- Visual checks in ParaView: initial geometry, density contours showing bow shock and wake; ensure obstacle cells remain empty.
- Add a tiny unit test (if feasible) that builds a domain minus disk and checks `fluid.geometry.total_cells` and set difference invariants.

## Milestones
- Phase 1: geometry/mask creation, I/O wiring, and constant state VTK dump.
- Phase 2: first-order Godunov with CFL control, inlet/outlet/wall BCs, obstacle ghost handling.
- Phase 3: optional MUSCL-Hancock reconstruction + gradient limiter, additional diagnostics, and polishing CLI/help text.
