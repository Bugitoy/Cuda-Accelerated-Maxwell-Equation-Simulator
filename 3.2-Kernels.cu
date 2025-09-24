// Kernels

/*
In the previous section, we learned how to use asynchrony to improve the performance of a 
heterogeneous program by overlapping computation with I/O. We switched from a synchronous 
Thrust algorithm to the asynchronous CUB interface, which allowed the computational part 
of our program to look like this:
*/

%%writefile Sources/cub.cu
#include "dli.h"

void simulate(dli::temperature_grid_f temp_in, float *temp_out, cudaStream_t stream)
{
  auto cell_ids = thrust::make_counting_iterator(0);
  cub::DeviceTransform::Transform(
    cell_ids, temp_out, temp_in.size(), 
    [temp_in] __host__ __device__ (int cell_id) { 
      return dli::compute(cell_id, temp_in); 
    }, stream);
}

// Launching a Cuda Kernel with __global__

/*
However, sometimes the algorithm you need is not available in existing accelerated libraries. 
What can you do when you cannot simply extend these existing algorithms (as we did in the first 
section) to fit your unique use case? At this point, it helps to understand the “magic” behind 
these accelerated libraries—specifically, how to launch a function on the GPU from the CPU.

So far, we have only used the __host__ and __device__ function specifiers, where host functions 
run on the CPU and device functions run on the GPU. To launch a function on the GPU from the CPU, 
we need a different specifier. That is where __global__ comes in.
*/

/*
A function annotated with __global__ is called a CUDA kernel. It is launched from the CPU but 
runs on the GPU. To launch a kernel, we use the specialized “triple chevrons” syntax:
*/

kernel<<<1, 1, 0, stream>>>(...)

/*
The first two numbers in the triple chevrons will be explained in more detail soon, but for now, 
note that CUDA kernels are asynchronous. In fact, CUB achieves its asynchrony by launching multiple 
CUDA kernels. Because kernels themselves are asynchronous, CUB can provide asynchronous functionality.

Let’s try to reimplement the functionality of cub::DeviceTransform directly as a CUDA kernel. We'll 
start with the code below which runs the algorithm with a single thread.
*/

%%writefile Sources/simple-kernel.cu
#include "dli.h"

__global__ void single_thread_kernel(dli::temperature_grid_f in, float *out)
{
  for (int id = 0; id < in.size(); id++) 
  {
    out[id] = dli::compute(id, in);
  }
}

void simulate(dli::temperature_grid_f temp_in, float *temp_out, cudaStream_t stream)
{
  single_thread_kernel<<<1, 1, 0, stream>>>(temp_in, temp_out);
}

// Parallelising

/*
Notice that we specify the CUDA Stream (stream) in the triple chevrons <<<1, 1, 0, stream>>>. 
However, as you might guess, this kernel is significantly slower than the CUB version because 
it processes the loop in a serial fashion. As we've learned already, the GPU does not automatically 
parallelize serial code.

We want to avoid serialization whenever possible. To parallelize this kernel, we need to launch 
more threads. The second parameter in the triple chevrons kernel<<<1, NUMBER-OF-THREADS, 0, stream>>> 
represents the number of threads. By increasing this number, we can launch more threads on the GPU. 
Of course, we also need to ensure that each thread processes a different subset of the data.

CUDA provides the built-in variable threadIdx.x, the value of which is used inside a kernel and 
stores the index of the current thread within a thread block, starting from 0. If we launch more 
threads, we can use threadIdx.x to split the work across them:
*/

const int number_of_threads = 2;

__global__ void block_kernel(dli::temperature_grid_f in, float *out)
{
  int thread_index = threadIdx.x;

  for (int id = thread_index; id < in.size(); id += number_of_threads) 
  {
    out[id] = dli::compute(id, in);
  }
}

/*
In this example, two threads run with indices threadIdx.x = 0 and threadIdx.x = 1. Each thread 
starts processing from its own index and increments by number_of_threads to avoid overlapping.
*/

/*
This change will evenly distribute work between threads, which should result in a speedup. Let's 
take a look if this is the case. When you run the next two cells you should observe a speedup over 
the previous iteration of the code.
*/

%%writefile Sources/block-kernel.cu
#include "dli.h"

const int number_of_threads = 2;

__global__ void block_kernel(dli::temperature_grid_f in, float *out)
{
  int thread_index = threadIdx.x;

  for (int id = thread_index; id < in.size(); id += number_of_threads) 
  {
    out[id] = dli::compute(id, in);
  }
}

void simulate(dli::temperature_grid_f temp_in, float *temp_out, cudaStream_t stream)
{
  block_kernel<<<1, number_of_threads, 0, stream>>>(temp_in, temp_out);
}

// Adding More Threads

// While this provides some speedup, it may still be far from the performance of the CUB 
// implementation. Increasing the number of threads further should help. Run the next two cells 
// and observe how performance changes when the number of threads is increased.

%%writefile Sources/block-256-kernel.cu
#include "dli.h"

const int number_of_threads = 256;

__global__ void block_kernel(dli::temperature_grid_f in, float *out)
{
  int thread_index = threadIdx.x;

  for (int id = thread_index; id < in.size(); id += number_of_threads) 
  {
    out[id] = dli::compute(id, in);
  }
}

void simulate(dli::temperature_grid_f temp_in, float *temp_out, cudaStream_t stream)
{
  block_kernel<<<1, number_of_threads, 0, stream>>>(temp_in, temp_out);
}

// This works well, but if you try to go too high (for example, number_of_threads = 2048), you 
// might see an error regarding invalid configuration. Run the following two cells to observe 
// this error.

%%writefile Sources/failed-block-kernel.cu
#include "dli.h"

const int number_of_threads = 2048;

__global__ void block_kernel(dli::temperature_grid_f in, float *out)
{
  int thread_index = threadIdx.x;

  for (int id = thread_index; id < in.size(); id += number_of_threads) 
  {
    out[id] = dli::compute(id, in);
  }
}

void simulate(dli::temperature_grid_f temp_in, float *temp_out, cudaStream_t stream)
{
  block_kernel<<<1, number_of_threads, 0, stream>>>(temp_in, temp_out);
}

// This error happens because there is a limit on the number of threads in a single block... 
// So, what is a thread block?

// Threads in a CUDA kernel are organized into a hierarchical structure. This structure consists 
// of equally-sized blocks of threads. All thread blocks together form a grid.

// The second parameter of the triple chevron specifies the number of threads in a block, and 
// this number can't exceed 1024. (There's nothing magic about 1024, it's simply a limit enforced 
// by NVIDIA based on HW resources.) To launch more than 1024 threads, we need to launch more 
// blocks. The first parameter in the triple chevrons 
// kernel<<<NUMBER-OF-BLOCKS, NUMBER-OF-THREADS, 0, stream>>> specifies the number of blocks.

// The thread indexing we saw earlier is local to a block, so threadIdx.x will always be in the 
// range [0, NUMBER-OF-THREADS).

// To uniquely identify each thread across blocks, we need to combine both the block index and 
// the thread index. To do that, we can combine the blockIdx.x variable, which stores the index 
// of the current block, with blockDim.x, which stores the number of threads in each block:

int thread_index = blockDim.x * blockIdx.x + threadIdx.x

// For more details on these built-in variables, refer to the CUDA Programming Guide: 
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy

// Here are a few examples of how thread_index is calculted for a few selected threads in 
// different thread blocks.

// Image

// Note that blockDim.x is a constant and is the same for every thread, while blockIdx.x and 
// threadIdx.x vary depending on which thread and which block are running.

// Besides that, we'll also have to update the stride calculation in the loop. To do this, we'll 
// need to compute the total number of threads in the grid which we can do using another built-in 
// variable called gridDim.x. This variable stores the number of blocks in the grid, so the total 
// number of threads in the grid can be computed as:

int number_of_threads = blockDim.x * gridDim.x

// Choosing how many threads go in each block is often independent of problem size. A common rule 
// of thumb is to use a multiple of 32 (a warp size), with 256 being a reasonable starting choice. 
// The number of blocks, by contrast, is usually derived from the problem size so that all elements 
// can be covered.

// If you attempt to do something like this:

int problem_size = 6;
int block_size = 4;
int grid_size = 6 / 4; // results in 1 block, but we need 2

// you would not launch enough blocks because of the integer division. To fix this, you can use a 
// helper function that performs a ceiling division:

int ceil_div(int a, int b) 
{
  return (a + b - 1) / b;
}

// This ensures enough blocks are launched to cover every element in the data. Putting it all 
// together, we can write:

%%writefile Sources/grid-kernel.cu
#include "dli.h"

__global__ void grid_kernel(dli::temperature_grid_f in, float *out)
{
  int thread_index = blockDim.x * blockIdx.x + threadIdx.x;
  int number_of_threads = blockDim.x * gridDim.x;

  for (int id = thread_index; id < in.size(); id += number_of_threads) 
  {
    out[id] = dli::compute(id, in);
  }
}

int ceil_div(int a, int b) 
{
  return (a + b - 1) / b;
}

void simulate(dli::temperature_grid_f temp_in, float *temp_out, cudaStream_t stream)
{
  int block_size = 1024;
  int grid_size = ceil_div(temp_in.size(), block_size);

  grid_kernel<<<grid_size, block_size, 0, stream>>>(temp_in, temp_out);
}

// You should observe a significant speedup of this code compared to versions earlier in this notebook. 
// This makes sense intuitively as with each now kernel we are launching more threads. We'd expect 
// launching more threads to result in a faster execution time.

// With this approach, our kernel more effectively utilizes the GPU. While it may still not be as fast 
// as the CUB implementation,which uses additional optimizations beyond our current scope, 
// understanding how to write and launch CUDA kernels directly is crucial for creating high-performance 
// custom algorithms.

// Exercise: Symmetry 

// For your reference, the following is an example of launching a simple kernel. Execute the next two 
// cells to view the results.

%%writefile Sources/kernel-launch.cu
#include <cstdio>

__global__ void kernel(int value) {
  std::printf("value on device = %d\n", value);
}

int main() {
  int blocks_in_grid = 1;
  int threads_in_block = 1;
  cudaStream_t stream = 0;
  kernel<<<blocks_in_grid, threads_in_block, 0, stream>>>(42);
  cudaStreamSynchronize(stream);
}

// Change the code below the image to launch a kernel with a single thread that checks if the input 
// array is symmetric. If the code does NOT obtain the correct answer, an error message will be printed.

%%writefile Sources/symmetry-check.cu
#include "dli.cuh"

// 1. convert the function below from a CPU function into a CUDA kernel
void symmetry_check_kernel(dli::temperature_grid_f temp, int row)
{
  int column = 0;

  if (abs(temp(row, column) - temp(temp.extent(0) - 1 - row, column)) > 0.1)
  {
    printf("Error: asymmetry in %d / %d\n", column, temp.extent(1));
  }
}

void symmetry_check(dli::temperature_grid_f temp, cudaStream_t stream)
{
  int target_row = 0;
  // 2. use triple chevron to launch the kernel
  symmetry_check_kernel(temp, target_row);
}

// Solution

// Key points:

// - Annotate the kernel with __global__
// - Launch the kernel with triple chevrons <<<1, 1, 0, stream>>>

// Solution:

__global__ void symmetry_check_kernel(dli::temperature_grid_f temp, int row)
{
  int column = 0;

  if (abs(temp(row, column) - temp(temp.extent(0) - 1 - row, column)) > 0.1)
  {
    printf("Error: asymmetry in %d / %d\n", column, temp.extent(1));
  }
}

void symmetry_check(dli::temperature_grid_f temp, cudaStream_t stream)
{
  int target_row = 0;
  symmetry_check_kernel<<<1, 1, 0, stream>>>(temp, target_row);
}

// Exercise: Row Symmetry 

// Threads are grouped into blocks, and each thread in a block has a unique ID threadIdx.x 
// ranging from 0 to blockDim.x - 1. Blocks themselves are indexed by blockIdx.x within a grid, 
// which ranges from 0 to gridDim.x - 1.

// The global (grid-level) thread index is:

int thread_index = blockIdx.x * blockDim.x + threadIdx.x

// And the total number of threads in the entire grid is:

int num_threads = gridDim.x * blockDim.x

// Given a problem size N and thread block size, we can compute the number of blocks we need 
// in a grid as: For a problem of size N, if block size is threads_per_block threads, you can
// compute the number of blocks as:

int threads_per_block = 256;
int num_blocks = cuda::ceil_div(N, threads_per_block)

// This ensures you launch enough threads to cover all N elements in the problem. Using this 
// information, modify the code below to launch a grid of threads, checking for symmetry of 
// a given row:

// Assign each thread to a unique column index. An error will be printed if your code does not 
// obtain the correct answer.

%%writefile Sources/row-symmetry-check.cu
#include "dli.cuh"

__global__ void symmetry_check_kernel(dli::temperature_grid_f temp, int row)
{
  // 1. change the line below so that each thread in a grid 
  //    checks exactly one column
  int column = 0;

  if (abs(temp(row, column) - temp(temp.extent(0) - 1 - row, column)) > 0.1)
  {
    printf("Error: asymmetry in %d / %d\n", column, temp.extent(1));
  }
}

void symmetry_check(dli::temperature_grid_f temp, cudaStream_t stream)
{
  int width      = temp.extent(1);
  // 2. launch sufficient number of threads to assign one thread per column

  int target_row = 0;
  symmetry_check_kernel<<<1, 1, 0, stream>>>(temp, target_row);
}

// Key points:

// - Launch a grid of threads
// - Use thread index as column index

// Solution:

__global__ void symmetry_check_kernel(dli::temperature_grid_f temp, int row)
{
  int column = blockIdx.x * blockDim.x + threadIdx.x;

  if (abs(temp(row, column) - temp(temp.extent(0) - 1 - row, column)) > 0.1)
  {
    printf("Error: asymmetry in %d / %d\n", column, temp.extent(1));
  }
}

void symmetry_check(dli::temperature_grid_f temp, cudaStream_t stream)
{
  int width      = temp.extent(1);
  int block_size = 1024;
  int grid_size  = cuda::ceil_div(width, block_size);

  int target_row = 0;
  symmetry_check_kernel<<<grid_size, block_size, 0, stream>>>(temp, target_row);
}

// Dev Tools

// Let's start with the simplest CUDA kernel. Observe the error messages that occur when executing 
// the following cell. Notice that we are building the executable and then running it with 
// compute-sanitizer.

!nvcc --extended-lambda -g -G -o /tmp/a.out Solutions/row-symmetry-check.cu # build executable
!/tmp/a.out # run executable
!compute-sanitizer /tmp/a.out # run sanitizer

// There are a lot of error messages printed, and the instructive errors are at the very top. Note 
// one of the errors printed Invalid __global__ read of size 4 bytes at symmetry_check_kernel. This 
// tells you exactly where to look to find the memory access error.

Invalid __global__ read of size 4 bytes
=========     at symmetry_check_kernel(cuda::std::__4::mdspan<float, cuda::std::__4::extents<int, (unsigned long)18446744073709551615, (unsigned long)18446744073709551615>, cuda::std::__4::layout_right, cuda::std::__4::default_accessor<float>>, int)+0x2490 in /home/jbentz/dli/c-ac-01-v2/task1/task/03.02-Kernels/Solutions/row-symmetry-check.cu:6
=========     by thread (928,0,0) in block (4,0,0)
=========     Address 0x709388060 is out of bounds
=========     and is 97 bytes after the nearest allocation at 0x708000000 of size 20480000 bytes

// The code below fixes the error. Note the use of the if (column < temp.extent(1)) statement, which 
// guards the execution of the thread. Each thread checks whether its column is less than the size 
// of the array, temp. If it is less, it executes the symmetry check, but if it is NOT, then it just 
// returns. This type of simple fix is very common in CUDA kernel programming to ensure that threads 
// don't access out-of-bounds memory.

// Execute the next two cells and verify that compute-sanitizer does not report any further errors.

%%writefile Sources/row-symmetry-check-fixed.cu
#include "dli.h"

__global__ void symmetry_check_kernel(dli::temperature_grid_f temp, int row)
{
  int column = blockIdx.x * blockDim.x + threadIdx.x;

  if (column < temp.extent(1))
  {
    if (abs(temp(row, column) - temp(temp.extent(0) - 1 - row, column)) > 0.1)
    {
        printf("Error: asymmetry in %d / %d\n", column, temp.extent(1));
    }
  }
}

void symmetry_check(dli::temperature_grid_f temp_in, cudaStream_t stream)
{
  int width      = temp_in.extent(1);
  int block_size = 1024;
  int grid_size  = cuda::ceil_div(width, block_size);

  int target_row = 0;
  symmetry_check_kernel<<<grid_size, block_size, 0, stream>>>(temp_in, target_row);
}

void simulate(dli::temperature_grid_f temp_in, float *temp_out, cudaStream_t stream)
{
  symmetry_check(temp_in, stream);
  dli::simulate(temp_in, temp_out, stream);
}

