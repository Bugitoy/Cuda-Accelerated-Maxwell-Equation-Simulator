// Shared Memory

// With our previous optimizations, the kernel now performs significantly better. However, some 
// inefficiencies remain. Currently, each block’s histogram is stored in global GPU memory, even 
// though it’s never used outside the kernel. This approach not only consumes unnecessary bandwidth 
// but also increases the overall memory footprint.

// Cache Memory

// As shown in the figure above, there’s a much closer memory resource: each Streaming Multiprocessor 
// (SM) has its own L1 cache. Ideally, we want to store each block’s histogram right there in L1. 
// Fortunately, CUDA makes this possible through software-controlled shared memory. By allocating 
// the block histogram in shared memory, we can take full advantage of the SM’s L1 cache and reduce 
// unnecessary memory traffic.

%%writefile Sources/simple-shmem.cu
#include <cstdio>

__global__ void kernel()
{
  __shared__ int shared[4];
  shared[threadIdx.x] = threadIdx.x;
  __syncthreads();

  if (threadIdx.x == 0)
  {
    for (int i = 0; i < 4; i++) {
      std::printf("shared[%d] = %d\n", i, shared[i]);
    }
  }
}

int main() {
  kernel<<<1, 4>>>();
  cudaDeviceSynchronize();
  return 0;
}

// To allocate shared memory, simply annotate a variable with the __shared__ keyword. This puts the 
// variable into shared memory that coresides with the L1 cache. Since shared memory isn't automatically 
// initialized, we begin our kernel by having each thread write its own index into a corresponding 
// shared memory location:

shared[threadIdx.x] = threadIdx.x;
__syncthreads()

// The __syncthreads() call ensures that all threads have finished writing to the shared array before 
// any thread reads from it. Afterwards, the first thread prints out the contents of the shared memory:

// As you can see, each thread successfully stored its index in the shared array, and the first thread 
// can read back those values.

// Exercise: Optimise Histogram

// You can allocate shared memory with __shared__ memory space specifier.

// Use shared memory to optimize the performance of the histogram. You will do this algorithm 
// in two stages:

// - Compute a privatized histogram for each thread block.
// - Contribute the privatized histogram to the global histogram.

// Original code in case you need it.

%%writefile Sources/shmem.cu
#include "dli.cuh"

constexpr int num_bins = 10;
constexpr float bin_width = 10;

// 1. Remove `block_histograms` from kernel parameters
__global__ void histogram_kernel(cuda::std::span<float> temperatures,
                                 cuda::std::span<int> block_histograms,
                                 cuda::std::span<int> histogram) 
{
  // 2. Allocate `block_histogram` in shared memory and initialize it to 0
  cuda::std::span<int> block_histogram =
      block_histograms.subspan(blockIdx.x * histogram.size(), histogram.size());

  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  int bin = static_cast<int>(temperatures[cell] / bin_width);

  cuda::atomic_ref<int, cuda::thread_scope_block> 
    block_ref(block_histogram[bin]);
  block_ref.fetch_add(1);
  __syncthreads();

  if (threadIdx.x < num_bins) 
  {
    cuda::atomic_ref<int, cuda::thread_scope_device> ref(histogram[threadIdx.x]);
    ref.fetch_add(block_histogram[threadIdx.x]);
  }
}

void histogram(cuda::std::span<float> temperatures,
               cuda::std::span<int> block_histograms,
               cuda::std::span<int> histogram, cudaStream_t stream) {
  int block_size = 256;
  int grid_size = cuda::ceil_div(temperatures.size(), block_size);
  histogram_kernel<<<grid_size, block_size, 0, stream>>>(
      temperatures, block_histograms, histogram);
}

// Solution

// Key points:

// - Allocate a shared memory array

// Solution:

__shared__ int block_histogram[num_bins];

if (threadIdx.x < num_bins) 
{
  block_histogram[threadIdx.x] = 0;
}
__syncthreads();

int cell = blockIdx.x * blockDim.x + threadIdx.x;
int bin = static_cast<int>(temperatures[cell] / bin_width);

cuda::atomic_ref<int, cuda::thread_scope_block> 
  block_ref(block_histogram[bin]);
block_ref.fetch_add(1, cuda::memory_order_relaxed);
__syncthreads();

if (threadIdx.x < num_bins) 
{
  cuda::atomic_ref<int, cuda::thread_scope_device> ref(histogram[threadIdx.x]);
  ref.fetch_add(block_histogram[threadIdx.x], cuda::memory_order_relaxed);
}
