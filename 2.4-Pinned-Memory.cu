// Pinned Memory

// Let’s take another look at our current simulator state:

cudaStream_t compute_stream;
cudaStreamCreate(&compute_stream);

cudaStream_t copy_stream;
cudaStreamCreate(&copy_stream);

for (int write_step = 0; write_step < write_steps; write_step++) 
{
  thrust::copy(d_prev.begin(), d_prev.end(), d_buffer.begin());
  cudaMemcpyAsync(thrust::raw_pointer_cast(h_prev.data()),
                  thrust::raw_pointer_cast(d_buffer.data()),
                  height * width * sizeof(float), cudaMemcpyDeviceToHost,
                  copy_stream);

  for (int compute_step = 0; compute_step < compute_steps; compute_step++) 
  {
    simulate(width, height, d_prev, d_next, compute_stream);
    d_prev.swap(d_next);
  }

  cudaStreamSynchronize(copy_stream);
  dli::store(write_step, height, width, h_prev);

  cudaStreamSynchronize(compute_stream);
}

cudaStreamDestroy(compute_stream);
cudaStreamDestroy(copy_stream)

// We use two CUDA streams to overlap the expensive device-to-host copy (copy_stream) 
// with ongoing computations (compute_stream). However, if you profile this code (for 
// instance, using Nsight Systems), you will see that the copy and compute still run 
// sequentially. This indicates we’re missing a key concept about how the hardware works. 
// To understand why, we need to step back and look at how memory operates.

// Swap Memory

// Operating systems do not provide direct access to physical memory. Instead, programs 
// use virtual memory, which is mapped to physical memory. Virtual memory is organized 
// into pages, enabling the operating system to manage them flexibly, such as swapping 
// pages to disk when physical memory runs low.

// So any given page can be in physical memory, on disk, or in some other place, and the 
// operating system keeps track of that. When the page can be relocated to disk, it's 
// called pageable. But memory can also be page-locked, or "pinned" to physical memory.

// GPU Access

// What does this have to do with CUDA? GPU can only copy data from physical memory. This 
// means that when copying data between host and device, memory has to be pinned.

// But this cannot be right. We just copied data between host and device without doing 
// anything special like pinning memory. How did that work? Under the covers, when moving 
// memory from host to device the CUDA Runtime utilizes a staging buffer in pinned memory. 
// When you copy data from host to device, the CUDA Runtime first copies data to the 
// staging buffer, and then copies it to device.

// This should explain why our copy wasn't overlapped with compute. It was actually synchronous, 
// because under the covers the data was copied to a staging buffer. Unbeknownst to us at the 
// time, the code first copied a chunk of data into pinned staging buffer, waited till the copy 
// is done, and then proceeded to copy the next chunk of data that fit into the staging buffer.

// The good news is that we can pin memory ourselves via an explicit function call. In this case, 
// there'll be no need to stream data through staging buffer, enabling asynchrony.

// To allocate pinned memory, it's sufficient to use another container from Thrust:

thrust::universal_host_pinned_vector<float> pinned_memory(size)

// Exercise: Async Copy and Pinned Memory

// For this exercise, we'll attempt to fix our program to make the copy actually asynchronous. 
// To do this, you are expected to:

// - Use thrust::universal_host_pinned_vector to allocate pinned memory for the host buffer
// - Profile the program to see if the copy becomes asynchronous

// Copy of the original code if you need to refer to it.

%%writefile Sources/copy-overlap.cu
#include "dli.h"

int main() 
{
  int height = 2048;
  int width = 8192;

  cudaStream_t compute_stream;
  cudaStreamCreate(&compute_stream);

  cudaStream_t copy_stream;
  cudaStreamCreate(&copy_stream);

  thrust::device_vector<float> d_prev = dli::init(height, width);
  thrust::device_vector<float> d_next(height * width);
  thrust::device_vector<float> d_buffer(height * width);

  // 1. Change code below to allocate host vector in pinned memory
  thrust::host_vector<float> h_prev(height * width);

  const int compute_steps = 750;
  const int write_steps = 3;
  for (int write_step = 0; write_step < write_steps; write_step++) 
  {
    cudaMemcpy(thrust::raw_pointer_cast(d_buffer.data()),
               thrust::raw_pointer_cast(d_prev.data()),
               height * width * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(thrust::raw_pointer_cast(h_prev.data()),
                    thrust::raw_pointer_cast(d_buffer.data()),
                    height * width * sizeof(float), cudaMemcpyDeviceToHost,
                    copy_stream);

    for (int compute_step = 0; compute_step < compute_steps; compute_step++) 
    {
      dli::simulate(width, height, d_prev, d_next, compute_stream);
      d_prev.swap(d_next);
    }

    cudaStreamSynchronize(copy_stream);
    dli::store(write_step, height, width, h_prev);

    cudaStreamSynchronize(compute_stream);
  }

  cudaStreamDestroy(compute_stream);
  cudaStreamDestroy(copy_stream);
}

// Solution

// Key points:

// - Use thrust::universal_host_pinned_vector to allocate pinned memory for the host buffer

// Solution:

thrust::universal_host_pinned_vector<float> h_prev(height * width);

const int compute_steps = 750;
const int write_steps = 3;
for (int write_step = 0; write_step < write_steps; write_step++) {
  cudaMemcpy(thrust::raw_pointer_cast(d_buffer.data()),
             thrust::raw_pointer_cast(d_prev.data()),
             height * width * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(thrust::raw_pointer_cast(h_prev.data()),
                  thrust::raw_pointer_cast(d_buffer.data()),
                  height * width * sizeof(float), cudaMemcpyDeviceToHost,
                  copy_stream);
  // ...
}
