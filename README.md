## CUDA-Accelerated Maxwell’s Equation Simulator

### Overview
This project accelerates a 2D Maxwell’s Equations simulator (electromagnetic wave propagation) using CUDA. The original CPU implementation required significant execution time at higher resolutions; the CUDA version achieves multi‑billion cells/second throughput while preserving correctness, and an optional coarse‑grid path trades visual fidelity for additional speed.

### Key Results

| Configuration | Throughput (cells/s) | Notes |
|---|---:|---|
| CPU (baseline) | 197,402,344 | 4096^2 ≈ 15 s; 65,536^2 ≈ 4 min |
| GPU (target) | 4,146,082,804 | Expected target |
| GPU (achieved) | 4,887,778,050 | Surpassed target |
| GPU (coarse tiling) | 4,902,636,317 | Grainier but faster animation |

### What Was Accelerated
- Ported field updates of the 2D Yee‑style grid to the GPU.
- Replaced scalar CPU loops with parallel primitives:
  - `thrust::transform` for updating `hx`/`hy` using zipped/shifted `ez` values.
  - `thrust::for_each` over a counting range for updating `dz` using neighbor differences from `hx`/`hy`.
  - `thrust::transform` to update `ez` from `dz`.
- Implemented a coarse‑grid pathway using one thread block per coarse tile and `cub::BlockReduce` to average per‑thread tile values into a single coarse cell.

### Method Details
- Fine grid update (Step 1 & 2):
  - `update_hx` / `update_hy`: compute spatial derivatives from `ez` using zip/transform iterators; update magnetic field components.
  - `update_dz`: iterate cells with a counting iterator; compute divergence from neighbor differences of `hx` / `hy` and accumulate into `dz`.
  - `update_ez`: transform `dz` to `ez` (material coefficient application).
- Coarse grid (Step 3):
  - Assign one CUDA thread block to each coarse tile.
  - Each thread reads one fine cell from the tile; `cub::BlockReduce<float, block_threads>` reduces to a sum.
  - Thread 0 writes the averaged value to the coarse grid; visualizes at reduced resolution for better throughput.

### Files
- `Assessment.cu`: contains the accelerated simulator logic (Thrust‑based updates and the coarse‑grid kernel using CUB).

### Build & Run (example)
This repository provides the core kernels and transformations in `Assessment.cu`. Integration into a host driver/application depends on your environment. A minimal approach is to compile your host code together with the kernels in `Assessment.cu` using `nvcc` and link against CUB (header‑only) and Thrust (bundled with CUDA).

Example (adjust include paths as needed):

```bash
nvcc -O3 -arch=sm_80 -std=c++17 Assessment.cu -o maxwell_sim
```

Notes:
- Ensure your CUDA Toolkit version supports Thrust and includes CUB headers.
- Match `-arch` to your GPU’s compute capability.

### Performance Notes
- Throughput measurements depend on GPU model, clock, memory bandwidth, and compiler flags.
- Coarse tiling improves speed by reducing the number of displayed cells; compute is focused on tile averages rather than full‑resolution visualization.

### Repository
This work is hosted at `https://github.com/Bugitoy/Cuda-Accelerated-Maxwell-Equation-Simulator`.


