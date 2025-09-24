// Synchronisation

// Memory Contention

// With the fix from the previous exercise, our histogram kernel finally produces correct results, but 
// performance remains suboptimal. Why? Because using a single shared histogram forces millions of 
// atomic operations on the same memory location. This causes significant contention and implicit 
// serialization.

// In the worst case, all threads map their data to a single bin. With around 16 thousand blocks and 
// 256 threads each, that’s roughly 4 million atomic operations contending for the same location. So 
// while we have launched a few million threads, the atomic operation serializes the write to the 
// histogram span, and in effect our parallel code now runs partly in serial.

// Private Histogram

// To reduce this overhead, we can introduce a "private" histogram for each thread block. Each block 
// would accumulate its own local copy of histogram, then merge it into the global histogram after all 
// local updates are complete.

// Now, in the worst case, up to 256 atomic operations occur within a block’s private histogram, plus 
// about 16k merges (one per block). That’s 256 + 16k total atomic operations, a big improvement over 
// 4 million.

// Let’s see how to implement this optimization:

%%writefile Sources/bug.cu
#include "dli.cuh"

constexpr float bin_width = 10;

__global__ void histogram_kernel(
  cuda::std::span<float> temperatures, 
  cuda::std::span<int> block_histograms, 
  cuda::std::span<int> histogram) 
{
  cuda::std::span<int> block_histogram = 
    block_histograms.subspan(blockIdx.x * histogram.size(), histogram.size());

  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  int bin = static_cast<int>(temperatures[cell] / bin_width);

  cuda::std::atomic_ref<int> block_ref(block_histogram[bin]);
  block_ref.fetch_add(1);

  if (threadIdx.x < 10) {
    cuda::std::atomic_ref<int> ref(histogram[threadIdx.x]);
    ref.fetch_add(block_histogram[threadIdx.x]);
  }
}

void histogram(
  cuda::std::span<float> temperatures, 
  cuda::std::span<int> block_histograms, 
  cuda::std::span<int> histogram,
  cudaStream_t stream) 
{
  int block_size = 256;
  int grid_size = cuda::ceil_div(temperatures.size(), block_size);
  histogram_kernel<<<grid_size, block_size, 0, stream>>>(
    temperatures, block_histograms, histogram);
}

// Our updated kernel now accepts an additional argument for storing per-block histograms. Its 
// size is the number of bins times the number of thread blocks. Within the kernel, we use subspan 
// to focus on the portion of this buffer corresponding to the current block’s histogram. However, 
// if you run the code below, you’ll see that the result is still incorrect.

// Data Race

// The following code contains a data race:

cuda::std::atomic_ref<int> block_ref(block_histogram[bin]);
block_ref.fetch_add(1);

if (threadIdx.x < 10) {
  cuda::std::atomic_ref<int> ref(histogram[threadIdx.x]);
  ref.fetch_add(block_histogram[threadIdx.x]);
}

// We assumed all threads in the same thread block would finish updating the block histogram before 
// any threads started reading from it, but CUDA threads can progress independently, even within the 
// same thread block. To state it more clearly, there is no guarantee that threads in the same thread 
// block are synchonized with each other. Some threads maybe be finished executing the entire kernel 
// before other threads even start. This is a very important concept to internalize as you write 
// parallel algorithms and CUDA kernels.

// As a result, some threads may read the histogram before it’s fully updated. Here, we assumed that 
// all threads in the block finished upating block histogram before other threads start reading it.

// To fix this issue, we must force all threads to complete their updates before allowing any thread 
// to read the block histogram. CUDA provides __syncthreads() function for this exact purpose. 
// The __syncthreads() function is a barrier which all threads in the thread block must reach before
// any thread is permitted to proceed to the next part of the code.

// Add Synchronization

// In the next exercise, you'll fix the issue by adding synchronization in the appropriate place. 
// Besides the correctness issue, we have some performance inefficiencies in the current implementation. 
// To figure out what's wrong, let's return to what's available in the cuda:: namespace. We've seen 
// cuda::std::atomic_ref already, but there's also cuda::atomic_ref type. These two types share the 
// same interface except for one important difference. cuda::atomic_ref has one extra template 
// parameter, representing a thread scope.

cuda::std::atomic_ref<int> ref(/* ... */);
cuda::atomic_ref<int, thread_scope> ref(/* ... */)

// Thread scope represents the set of threads that can synchronize using a given atomic. Thread scope 
// can be system, device, or block.

// For instance, all threads of a given system are related to each other thread by 
// cuda::thread_scope_system. This means that a thread from any GPU (in a multi-GPU system) can 
// synchronize with any other GPU thread, or any CPU thread. The cuda::std::atomic_ref is actually 
// the same thing as cuda::atomic_ref<int, cuda::thread_scope_system>.

// In addition to the system scope, there are also device and block scopes. The device scope allows 
// threads from the same device to synchronize with each other. The block scope allows threads from 
// the same block to synchronize with each other.

// Since our histogram kernel is limited to a single GPU, we don't need to use the system scope. 
// Besides that, only threads of a single block are issuing atomics to the same block histogram. 
// This means that we can leverage the block scope to improve performance.

// Exercise: Fix Data Race

// You can use __syncthreads() to synchronize threads within a block:

// Fix the data race using thread-block synchronization. Optionally, switch to cuda::atomic_ref 
// to reduce the scope of communication:

// Original code if you need to refer back to it.

%%writefile Sources/sync.cu
#include "dli.cuh"

constexpr float bin_width = 10;

// 1. Use `__syncthreads()` to synchronize threads within a block and avoid data race
__global__ void histogram_kernel(
  cuda::std::span<float> temperatures, 
  cuda::std::span<int> block_histograms, 
  cuda::std::span<int> histogram) 
{
  cuda::std::span<int> block_histogram = 
    block_histograms.subspan(blockIdx.x * histogram.size(), 
                             histogram.size());

  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  int bin = static_cast<int>(temperatures[cell] / bin_width);

  cuda::std::atomic_ref<int> block_ref(block_histogram[bin]);
  block_ref.fetch_add(1);

  if (threadIdx.x < histogram.size()) {
    // 2. Reduce scope of atomic operation using `cuda::atomic_ref`
    cuda::std::atomic_ref<int> ref(histogram[threadIdx.x]);
    ref.fetch_add(block_histogram[threadIdx.x]);
  }
}


void histogram(
  cuda::std::span<float> temperatures, 
  cuda::std::span<int> block_histograms, 
  cuda::std::span<int> histogram,
  cudaStream_t stream) 
{
  int block_size = 256;
  int grid_size = cuda::ceil_div(temperatures.size(), block_size);
  histogram_kernel<<<grid_size, block_size, 0, stream>>>(
    temperatures, block_histograms, histogram);
}

// Solution

// Key points:

// - Synchronize before reading the block histogram

// Solution:

cuda::std::span<int> block_histogram =
  block_histograms.subspan(blockIdx.x * histogram.size(), histogram.size());

int cell = blockIdx.x * blockDim.x + threadIdx.x;
int bin = static_cast<int>(temperatures[cell] / bin_width);

cuda::atomic_ref<int, cuda::thread_scope_block> 
  block_ref(block_histogram[bin]);
block_ref.fetch_add(1);
__syncthreads();

if (threadIdx.x < histogram.size()) 
{
  cuda::atomic_ref<int, cuda::thread_scope_device> 
    ref(histogram[threadIdx.x]);
  ref.fetch_add(block_histogram[threadIdx.x]);
}


