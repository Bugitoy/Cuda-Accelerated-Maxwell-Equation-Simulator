### Maxwell Equation Simulator (CUDA-Accelerated)

This module accelerates a 2D Maxwell's Equations simulator that predicts electromagnetic wave propagation. The original CPU implementation took ~15 s for 4096^2 cells and ~4 min for 65,536^2 cells (~197,402,344 cells/s). The CUDA implementation achieves ~4,887,778,050 cells/s, exceeding the expected ~4,146,082,804 cells/s.

Key techniques:
- Port core field updates to the GPU with Thrust and device lambdas in `maxwell.cu`.
- Coarsen the grid by tiling and computing per-tile averages using `cub::BlockReduce` in `coarse.cu` for faster, grainier visualization.
- Assign one thread block per coarse tile; compute the average of thread values and write to the coarse grid.

Files:
- `maxwell.cu`: GPU implementation of field updates and the `simulate` loop.
- `coarse.cu`: Coarse-grid kernel using `cub::BlockReduce` to average tiles.

Assumptions:
- The shared support header `dli.h` provides types, constants like `dli::C0`, dimensions, and utilities used by these modules.

Build notes:
- Requires CUDA toolkit and CUB (modern CUDA ships CUB).
- Include paths must resolve `dli.h` and link with any project scaffolding using these functions.

Performance summary:
- Baseline CPU: ~197,402,344 cells/s.
- CUDA port: ~4,887,778,050 cells/s.
- With coarse tiling and `cub::BlockReduce`: ~4,902,636,317 cells/s.
