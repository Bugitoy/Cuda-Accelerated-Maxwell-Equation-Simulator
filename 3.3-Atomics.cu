// Atomics

// We recently fixed a bug caused by our thread hierarchy, which might prompt the question: 
// why did we need that hierarchy in the first place? To illustrate its value, let’s look at a 
// related problem: computing a histogram of our temperature grid.

// Histogram of Temperature Grid

// A histogram helps visualize the distribution of temperatures by grouping values into "bins". 
// In this example, each bin covers a 10-degree range, so the first bin represents temperatures 
// in [0, 10), the second in [10, 20), and so on.

// Given a cell’s temperature, how do we determine the bin it belongs to? We can simply use integer 
// division:

int bin = static_cast<int>(temperatures[cell] / bin_width)

// So, a temperature of 14 falls into bin 1, while 4 maps to bin 0. Next, we’ll implement this logic 
// in a CUDA kernel, assigning one thread per cell to calculate its bin.

%%writefile Sources/histogram-bug.cu
#include "dli.cuh"

constexpr float bin_width = 10;

__global__ void histogram_kernel(cuda::std::span<float> temperatures, 
                                 cuda::std::span<int> histogram)
{
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  int bin = static_cast<int>(temperatures[cell] / bin_width);
  int old_count = histogram[bin];
  int new_count = old_count + 1;
  histogram[bin] = new_count;
}

void histogram(cuda::std::span<float> temperatures, 
               cuda::std::span<int> histogram, 
               cudaStream_t stream)
{
  int block_size = 256;
  int grid_size = cuda::ceil_div(temperatures.size(), block_size);
  histogram_kernel<<<grid_size, block_size, 0, stream>>>(
    temperatures, histogram);
}

// Data Race

// Something went wrong. Despite having four million cells, our histogram comes out nearly empty. 
// The culprit is in this kernel code:

int old_count = histogram[bin];
int new_count = old_count + 1;
histogram[bin] = new_count;

// Because this code runs simultaneously on millions of threads while attempting to read/write a 
// single copy of the histogram span, it introduces a data race.

// For example, if two threads increment the same bin at the same time, both read the same initial 
// value and overwrite one another’s updates, causing the bin to increment only once instead of twice. 
// Multiplied by millions of cells, this leads to a nearly empty histogram.

// To fix this, we need to make the read, modify, and write steps a single, indivisible operation. 
// CUDA provides atomic operations that handle concurrency safely, ensuring we don’t lose any increments
// in our histogram.

%%writefile Sources/atomic.cu
#include <cuda/std/span>
#include <cuda/std/atomic>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void kernel(cuda::std::span<int> count)
{
    // Wrap data in atomic_ref
    cuda::std::atomic_ref<int> ref(count[0]);

    // Atomically increment the underlying value
    ref.fetch_add(1);
}

int main()
{
    thrust::device_vector<int> count(1);

    int threads_in_block = 256;
    int blocks_in_grid = 42;

    kernel<<<blocks_in_grid, threads_in_block>>>(
        cuda::std::span<int>{thrust::raw_pointer_cast(count.data()), 1});

    cudaDeviceSynchronize();

    thrust::host_vector<int> count_host = count;
    std::cout << "expected: " << threads_in_block * blocks_in_grid << std::endl;
    std::cout << "observed: " << count_host[0] << std::endl;
}

// In the example above, we reproduce our histogram kernel’s structure, where multiple threads attempt 
// to increment the same memory location. This time, however, we wrap the memory reference in a 
// cuda::std::atomic_ref<int>:

cuda::std::atomic_ref<int> ref(count[0])

// Here, int indicates the type of the underlying value, and the constructor accepts a reference to 
// the memory we want to modify. The resulting atomic_ref object offers atomic operations, such as:

ref.fetch_add(1)

// This call performs an indivisible read-modify-write operation: it reads the current value of count[0],
// adds one, and writes the result back atomically. You can think of atomics as writing an instruction 
// rather than a direct value.

// Image

// The "?" is replaced by the current value of count[0], incremented by one, and stored in a single step. It doesn’t matter how many threads do this concurrently - the result remains correct.

// Exercise: Fix Histogram

// The code below has a data race in it. Multiple threads concurrently increment the same element of the histogram array. Use cuda::std::atomic_ref to fix this bug.

// Interface of cuda::std::atomic_ref is equivalent to std::atomic_ref:

__global__ void kernel(int *count)
{
  // Wrap data in atomic_ref
  cuda::std::atomic_ref<int> ref(count[0]);

  // Atomically increment the underlying value
  ref.fetch_add(1);
}

// Original code in case you need to refer to it:

%%writefile Sources/histogram.cu
#include "dli.cuh"

constexpr float bin_width = 10;

__global__ void histogram_kernel(cuda::std::span<float> temperatures, 
                                 cuda::std::span<int> histogram)
{
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell < temperatures.size()) {
    int bin = static_cast<int>(temperatures[cell] / bin_width);

    // fix data race in incrementing histogram bins by using `cuda::std::atomic_ref`
    int old_count = histogram[bin];
    int new_count = old_count + 1;
    histogram[bin] = new_count;
  }
}

void histogram(cuda::std::span<float> temperatures, 
               cuda::std::span<int> histogram,
               cudaStream_t stream)
{
  int block_size = 256;
  int grid_size = cuda::ceil_div(temperatures.size(), block_size);
  histogram_kernel<<<grid_size, block_size, 0, stream>>>(
    temperatures, histogram);
}

// Solution

// Key points:

// - Wrap selected bin in cuda::std::atomic_ref<int> for atomic operations
// - Use fetch_add to increment the bin value atomically

// Solution:

__global__ void histogram_kernel(cuda::std::span<float> temperatures,
    cuda::std::span<int> histogram) 
{
int cell = blockIdx.x * blockDim.x + threadIdx.x;
int bin = static_cast<int>(temperatures[cell] / 10);

cuda::std::atomic_ref<int> ref(histogram[bin]);
ref.fetch_add(1);
}