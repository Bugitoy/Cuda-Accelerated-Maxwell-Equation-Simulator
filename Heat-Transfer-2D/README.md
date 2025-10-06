### Heat Transfer 2D (CUDA)

CUDA-accelerated 2D heat diffusion using a 5-point stencil. The final `simulate` uses device lambdas and `thrust::tabulate` plus CUDA libcu++ vocabulary types for clean indexing.

Files:
- `heat-2D.cu`: Driver using `dli::init`, `dli::store`, and `dli::simulate` for timing and I/O.
- `simulate.cu`: Final GPU `simulate` using `cuda::std::pair` and `thrust::tabulate`.

Notes:
- Assumes `dli.h` provides scaffolding (`dli::init`, `dli::store`) and integrates your `simulate`.
- Build with CUDA toolkit; ensure include paths resolve `dli.h`.
