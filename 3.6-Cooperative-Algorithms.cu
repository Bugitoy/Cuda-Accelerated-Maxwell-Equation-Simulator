// Cooperative Algorithms

// Our kernel performance is now at acceptable levels. However, we’ve made a significant oversight in 
// our strategy. While writing host-side code, we used accelerated libraries. However, once we started 
// writing CUDA kernels, we ended up implementing everything from scratch. This isn’t the best approach.

// Device-Side Libraries

// CUDA offers many device-side libraries that can help you write more efficient kernels more quickly. 
// For example:

// - cuBLASDx: provides cooperative linear algebra functions inside CUDA kernels
// - cuFFTDx: provides fast cooperative Fourier Transform inside CUDA kernels
// - CUB: provides cooperative general-purpose algorithms inside CUDA kernels

// ... and there are more. But they all share a common trait. They're cooperative. What does that actually mean?

// To answer this question, let's revisit the algorithm types we’ve seen so far:

// 1. Sequential algorithms (e.g., std::transform):

// - Invoked and executed by a single thread.
// - Even if multiple threads call the same algorithm, there's no way the input of one thread can affect the output of another.

// 2. Cooperative algorithms:

// - Invoked and executed by multiple threads working together to achieve a common goal.
// - It’s as if they accept a single input that is divided among multiple threads.

// 3. Parallel algorithms (e.g., those in CUB and Thrust):

// - Invoked by a single thread, but internally executed by many threads.
// - From the user’s perspective, they often look similar to sequential algorithms.
// - Under the hood, they often rely on cooperative algorithms to perform the actual parallel work.

// CUB Cooperative Algorithms

// CUB provides both parallel and cooperative algorithms. To learn the basics of cooperative libraries, 
// let’s look at a block-level reduction example. The CUB block-level interface can be summarized as 
// follows:

template <typename T, int BlockDimX>
struct cub::BlockReduce
{
  struct TempStorage { ... };

  __device__ BlockReduce(TempStorage& temp_storage) { ... }

  __device__ T Sum(T input) { return ...; }
}

// Unlike traditional function-oriented interfaces, CUB exposes its cooperative algorithms as 
// templated structs. Template parameters are used to specialize algorithms for the problem at hand:

// - data type (int, float, etc.)
// - number of threads in the thread block
// - grain size (number of items per thread)
// - and so on

// The nested TempStorage type provides a type of temporary storage needed by cooperative algorithms 
// for thread communication. An instance of this type has to be allocated in shared memory. TempStorage 
// allocation must be provided to construct an instance of the algorithm.

// Finally, member functions represent different flavors of a given cooperative algorithms. The simple 
// usage of block-level reduction is as follows:

__shared__ cub::BlockReduce<int, 4>::TempStorage storage;
int block_sum = cub::BlockReduce<int, 4>(storage).Sum(threadIdx.x)

// Although the following diagram is only a conceptual model, it illustrate what’s happening under the 
// hood in a cooperative algorithm such as block-level reduction:

// Image

// The code starts by copying algorithm input that likely comes from registers into shared memory. 
// Then, the cooperative algorithm has to synchronize the thread block to make sure all stores were 
// completed. This means that if any of the threads don't call the algorithm, the entire thread block 
// will deadlock. Besides that, the algorithm uses shared memory to communicate intermediate results 
// between threads. Finally, the algorithm returns the result. Reduction is a bit specific in this 
// regard, because it returns a valid result only to the first thread in the thread block.

%%writefile Sources/block-sum.cu
#include <cub/block/block_reduce.cuh>

constexpr int block_threads = 128;

__global__ void block_sum()
{
  using block_reduce_t = cub::BlockReduce<int, block_threads>;
  using storage_t = block_reduce_t::TempStorage;
  
  __shared__ storage_t storage;

  int block_sum = block_reduce_t(storage).Sum(threadIdx.x);

  if (threadIdx.x == 0)
  {
    printf("block sum = %d\n", block_sum);
  }
}

int main() {
  block_sum<<<1, block_threads>>>();
  cudaDeviceSynchronize();
  return 0;
}

// Exercise: Coarsening

// You can compute the block histogram using CUB. from the block size, number of bins, and elements 
// per thread:

    // block size has to be known at compile time
    constexpr int block_size = 256;
    constexpr int items_per_thread = 1;
    constexpr int num_bins = 10

// The block histogram type would be:

    using histogram_t =
    cub::BlockHistogram<
        int,              // type of histogram counters
        block_size,       // number of threads in a block
        items_per_thread, // number of bin indices that each thread contributes
        num_bins>         // number of bins in the histogram

// With that, you can allocate temporary storage in shared memory:

    __shared__ typename histogram_t::TempStorage temp_storage;

// The histogram member function has the following signature:

int thread_bins[items_per_thread] = {...};
histogram_t{temp_storage}.Histogram(thread_bins, block_histogram)

// Modify the kernel below to compute the average temperature in the tile and write it to the output 
// array:

// Original code in case you need it

%%writefile Sources/cooperative.cu
#include "dli.cuh"

constexpr int block_size = 256;
constexpr int items_per_thread = 1;
constexpr int num_bins = 10;
constexpr float bin_width = 10;

__global__ void histogram_kernel(cuda::std::span<float> temperatures,
                                 cuda::std::span<int> histogram) 
{
  __shared__ int block_histogram[num_bins];

  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  int bins[items_per_thread] = {static_cast<int>(temperatures[cell] / bin_width)};

  ??? // 1. Use `cub::BlockHistogram` to compute the block histogram

  __syncthreads();
  if (threadIdx.x < num_bins) {
    cuda::atomic_ref<int, cuda::thread_scope_device> ref(histogram[threadIdx.x]);
    ref.fetch_add(block_histogram[threadIdx.x]);
  }
}

void histogram(cuda::std::span<float> temperatures, 
               cuda::std::span<int> histogram, 
               cudaStream_t stream) 
{
  int grid_size = cuda::ceil_div(temperatures.size(), block_size);
  histogram_kernel<<<grid_size, block_size, 0, stream>>>(
    temperatures, histogram);
}

// Solution

// Key points:

// - Use cub::BlockHistogram to compute the block histogram
// - Allocate temporary storage in shared memory
// - Make sure to synchronize before reading the block histogram

// Solution:

using histogram_t = cub::BlockHistogram<int, block_size, 1, 10>;
__shared__ typename histogram_t::TempStorage temp_storage;
histogram_t(temp_storage).Histogram(bins, block_histogram);
__syncthreads()