# Examples

Each example writes its VTK files into `examples_output/<example name>` by default.
You can override the folder with `--output-dir <path>` when launching the executable.

Most examples only emit `.vtk` files, so a single folder per example keeps the tree tidy.

The stencil smoothing example (`field_stencil_smoothing_to_vtk`) also understands:

- `--iterations <n>`: number of smoothing passes (default 10).
- `--inner-margin <m>`: how far from the domain edge the stencil mask stops; the mask built from `[m, 64-m)` in both dimensions (clamped to `[0,31]`).

Run the executables from a build directory (e.g., `build-serial/examples/<name>`) to generate the outputs.
