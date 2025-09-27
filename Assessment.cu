// Assessment: Accelerate and Optimise Maxwell's Equations Simulator

// The Maxwell's Equations simulator predicts how electromagnetic waves propagate. 
// For this assessment, you'll begin with simple, throughly working, 2D Maxwell's
// Equations simulator. In its current CPU-only form, this application takes about 
// 15 seconds to run on 4096^2 cells, and 4 minutes to run on 65536^2 cells. Your task 
// is to GPU-accelerate the program, retaining the correctness of the simulation.

// The Problem:

// The provided code simulates the propagation of electromagnetic waves in a 2D grid 
// using Maxwell's equations. It performs iterative updates of the electric and magnetic 
// fields, accounting for spatial and temporal changes. The use of CPU-based computation 
// limits the performance as it is scaled up. Your job is to accelerate the computation 
// using techniques learned in this class.

// Scoring:

// You will be assessed on your ability to accelerate the simulator in three steps, from 
// basic parallelization to employing fancy iterators, and finally using a course grid kernel. 
// This coding assessment is worth 100 points, divided as follows:

// Rubric:

// 1. Port CPU Simulator to GPU 
    // - FIXMEs? 2, Points? 50
// 2. Accelerate the Simulator with Fancy Iterators 
    // - FIXMEs? 3, Points? 25
// 3. Use Cooperative Algorithms to Coarse the Grid
    // - FIXMEs? 3, Points? 25

// Step 1: Port CPU Simulator to GPU & Step 2: Accelerate the Simulator with Fancy Iterators

%%writefile Sources/maxwell.cu
#include "dli.h"


void update_hx(int n, float dx, float dy, float dt, thrust::device_vector<float> &hx,
               thrust::device_vector<float> &ez) {

  auto ez_zip = thrust::make_zip_iterator(ez.begin() + n, ez.begin());
  auto ez_transform = thrust::make_transform_iterator(ez_zip, []__host__ __device__(thrust::tuple<float, float> t){
    return thrust::get<0>(t) - thrust::get<1>(t);
  });

  thrust::transform(thrust::device, hx.begin(), hx.end() - n, ez_transform, hx.begin(),
  [dt, dx, dy] __host__ __device__ (float h, float cex) {
    return h - dli::C0 * dt / 1.3f * cex / dy;
  });
}

void update_hy(int n, float dx, float dy, float dt, thrust::device_vector<float> &hy,
               thrust::device_vector<float> &ez) {
 
  auto ez_zip = thrust::make_zip_iterator(ez.begin(), ez.begin() + 1);
  auto ez_transform = thrust::make_transform_iterator(ez_zip, []__host__ __device__(thrust::tuple<float, float> t){
    return thrust::get<0>(t) - thrust::get<1>(t);
  });

  thrust::transform(thrust::device, hy.begin(), hy.end() - 1, ez_transform, hy.begin(),
  [dt, dx, dy] __host__ __device__ (float h, float cex) {
    return h - dli::C0 * dt / 1.3f * cex / dx;
  });
}

void update_dz(int n, float dx, float dy, float dt, 
               thrust::device_vector<float> &hx_vec,
               thrust::device_vector<float> &hy_vec, 
               thrust::device_vector<float> &dz_vec,
               thrust::counting_iterator<int> cells_begin, thrust::counting_iterator<int> cells_end) {

  auto hx = hx_vec.begin();
  auto hy = hy_vec.begin();
  auto dz = dz_vec.begin();

  thrust::for_each(thrust::device, cells_begin, cells_end,
                [n, dx, dy, dt, hx, hy, dz] __host__ __device__ (int cell_id) {
                  if (cell_id > n) {
                    float hx_diff = hx[cell_id - n] - hx[cell_id];
                    float hy_diff = hy[cell_id] - hy[cell_id - 1];
                    dz[cell_id] += dli::C0 * dt * (hx_diff / dx + hy_diff / dy);
                  }
                });
}

void update_ez(thrust::device_vector<float> &ez, thrust::device_vector<float> &dz) {
  thrust::transform(thrust::device, dz.begin(), dz.end(), ez.begin(),
                 [] __host__ __device__ (float d) { return d / 1.3f; });
}

// Do not change the signature of this function
void simulate(int cells_along_dimension, float dx, float dy, float dt,
              thrust::device_vector<float> &d_hx,
              thrust::device_vector<float> &d_hy,
              thrust::device_vector<float> &d_dz,
              thrust::device_vector<float> &d_ez) {

  int cells = cells_along_dimension * cells_along_dimension;
  auto cells_begin = thrust::make_counting_iterator(0);
  auto cells_end = thrust::make_counting_iterator(cells);

  for (int step = 0; step < dli::steps; step++) {
    update_hx(cells_along_dimension, dx, dy, dt, d_hx, d_ez);
    update_hy(cells_along_dimension, dx, dy, dt, d_hy, d_ez);
    update_dz(cells_along_dimension, dx, dy, dt, d_hx, d_hy, d_dz, cells_begin, cells_end);
    update_ez(d_ez, d_dz);
  }
}

// Step 3: Use Cooperative Algorithms to Coarse the Grid

// Original code:

%%writefile Sources/coarse.cu
#include "dli.h"

__global__ void kernel(dli::temperature_grid_f fine,
                       dli::temperature_grid_f coarse) {
  int coarse_row = blockIdx.x / coarse.extent(1);
  int coarse_col = blockIdx.x % coarse.extent(1);
  int row = threadIdx.x / dli::tile_size;
  int col = threadIdx.x % dli::tile_size;
  int fine_row = coarse_row * dli::tile_size + row;
  int fine_col = coarse_col * dli::tile_size + col;

  float thread_value = fine(fine_row, fine_col);

  // FIXME(Step 3):
  // Compute the sum of `thread_value` across threads of a thread block
  // using `cub::BlockReduce`
  float block_sum = ...;

  // FIXME(Step 3):
  // `cub::BlockReduce` returns block sum in thread 0, make sure to write
  // result only from the first thread of the block
  coarse(coarse_row, coarse_col) = block_average;
}

// Don't change the signature of this function
void coarse(dli::temperature_grid_f fine, dli::temperature_grid_f coarse) {
  kernel<<<coarse.size(), dli::block_threads>>>(fine, coarse);
}

// Updated code:

%%writefile Sources/coarse.cu
#include "dli.h"

__global__ void kernel(dli::temperature_grid_f fine,
                       dli::temperature_grid_f coarse) {
  int coarse_row = blockIdx.x / coarse.extent(1);
  int coarse_col = blockIdx.x % coarse.extent(1);
  int row = threadIdx.x / dli::tile_size;
  int col = threadIdx.x % dli::tile_size;
  int fine_row = coarse_row * dli::tile_size + row;
  int fine_col = coarse_col * dli::tile_size + col;

  float thread_value = fine(fine_row, fine_col);

  using block_reduce_t = cub::BlockReduce<float, dli::block_threads>;
  using storage_t = block_reduce_t::TempStorage;
  __shared__ storage_t storage;
  float block_sum = block_reduce_t(storage).Sum(thread_value);
  float block_average = block_sum / (dli::tile_size * dli::tile_size);

  if (threadIdx.x == 0) {
    coarse(coarse_row, coarse_col) = block_average;
  }
}

// Don't change the signature of this function
void coarse(dli::temperature_grid_f fine, dli::temperature_grid_f coarse) {
  kernel<<<coarse.size(), dli::block_threads>>>(fine, coarse);
  cudaDeviceSynchronize();
}



