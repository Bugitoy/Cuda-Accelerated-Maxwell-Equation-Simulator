// Memory Spaces

// At the beginning of this section, we covered execution spaces but left one 
// change without explanation. We replaced std::vector with thrust::universal_vector. 
// By the end of this lab, you'll understand why this change was necessary.

// But before we start, let's try to figure out why GPUs are so good at massive 
// parallelism. Many benefits of GPUs result focusing on high throughput. To support 
// massive compute that GPUs are able of sustaining, we have to provide memory speed 
// that matches these capabilities. This essentially means that memory also has to 
// be throughput-oriented. That's why GPUs often come with built-in high-bandwidth 
// memory rather than relying on system memory. Let's return to our code to see how 
// it's affected by this fact.

%%writefile Sources/heat-2D.cu
#include "dli.h"

int main()
{
  int height = 4096;
  int width  = 4096;

  thrust::universal_vector<float> prev = dli::init(height, width);
  thrust::universal_vector<float> next(height * width);

  for (int write_step = 0; write_step < 3; write_step++) {
    std::printf("   write step %d\n", write_step);
    dli::store(write_step, height, width, prev);
    
    for (int compute_step = 0; compute_step < 3; compute_step++) {
      auto begin = std::chrono::high_resolution_clock::now();
      dli::simulate(height, width, prev, next);
      auto end = std::chrono::high_resolution_clock::now();
      auto seconds = std::chrono::duration<double>(end - begin).count();
      std::printf("computed step %d in %g s\n", compute_step, seconds);
      prev.swap(next);
    }
  }
}

// In the code above, we allocate data in thrust::universal_vector. Then, dli::store 
// accesses content of this vector on CPU to store results on disk. After that, the 
// data is repeatedly accessed by the GPU in the dli::simulate function. This is a bit 
// suspicious. We just said that CPU and GPU have distinct memory spaces, but we are 
// not seeing anything that'd reflect this in the code. Maybe performance can reveal 
// something?

// Running this code provides the following stats:

// write step 0
// computed step 0 in 0.0245061 s
// computed step 1 in 0.000579675 s
// computed step 2 in 0.000575186 s
//   write step 1
// computed step 0 in 0.0248845 s
// computed step 1 in 0.000585406 s
// computed step 2 in 0.000574885 s
//   write step 2
// computed step 0 in 0.0241217 s
// computed step 1 in 0.000579654 s
// computed step 2 in 0.000575306 s

// There's a strange pattern in the execution times. Every time we write data, the next 
// compute step takes 100 times longer to compute. This happens because the data is 
// being implicitly copied between CPU and GPU memory spaces.

// Let's say our data resides in the GPU memory. When dli::store accesses it, the data 
// has to be copied to the CPU memory. Next, when we call dli::simulate, the data is 
// being accessed by the GPU, so the data has to be copied back. So thrust::universal_vector 
// works as a vector that lives in both CPU and GPU memory spaces and automatically 
// migrates between them. The problem is that we know that dli::store is not modifying 
// the data, so the copy back to the GPU is unnecessary. Fortunately, we can avoid this 
// extra copy by using explicit memory spaces.

// Host and Device Memory Spaces

// Presense of distinct host and device memory spaces is a fundamental concept in GPU programming. 
// For you, as a software engineer, this means that in addition to thinking about where code runs, 
// you also have to keep in mind where the bytes that this code accesses live. On a high level, 
// we have a host memory space and a device memory space. Thrust provides container types that manage 
// memory in the associated memory spaces. Let's take a look at a program that allocates vectors 
// in corresponding memory spaces:

thrust::host_vector<int> h_vec{ 11, 12 };
thrust::device_vector<int> d_vec{ 21, 22 };
thrust::copy_n(h_vec.begin(), 1, d_vec.begin())

// Let's take a look at this code step by step. We started by allocating a vector with two element 
// in host memory. We initialized these two elements with 11 and 12:

thrust::host_vector<int> h_vec{ 11, 12 }

// Functionally, there's little difference between std::vector and thrust::host_vector. As you learn, 
// we suggest using thrust::host_vector just to make memory space more pronounced. Besides host vector, 
// we also allocated device one:

thrust::device_vector<int> d_vec{ 21, 22 }

// We then copied one element from host memory space to device memory space using Thrust copy algorithm. 
// In general, copy is one of the few algorithms that you can provide mixed memory spaces.

thrust::copy_n(h_vec.begin(), 1, d_vec.begin())

// Exercise example:

// Original code:

%%writefile Sources/heat-2D-explicit-memory-spaces.cu
#include "dli.h"

int main()
{
  int height = 4096;
  int width  = 4096;

  thrust::universal_vector<float> prev = dli::init(height, width);
  thrust::universal_vector<float> next(height * width);

  for (int write_step = 0; write_step < 3; write_step++) {
    std::printf("   write step %d\n", write_step);
    dli::store(write_step, height, width, prev);
    
    for (int compute_step = 0; compute_step < 3; compute_step++) {
      auto begin = std::chrono::high_resolution_clock::now();
      dli::simulate(height, width, prev, next);
      auto end = std::chrono::high_resolution_clock::now();
      auto seconds = std::chrono::duration<double>(end - begin).count();
      std::printf("computed step %d in %g s\n", compute_step, seconds);
      prev.swap(next);
    }
  }
}

// Modified code:

thrust::device_vector<float> d_prev = dli::init(height, width);
thrust::device_vector<float> d_next(height * width);
thrust::host_vector<float> h_prev(height * width);

for (int write_step = 0; write_step < 3; write_step++) {
  std::printf("   write step %d\n", write_step);
  thrust::copy(d_prev.begin(), d_prev.end(), h_prev.begin());
  dli::store(write_step, height, width, h_prev);

  for (int compute_step = 0; compute_step < 3; compute_step++) {
    auto begin = std::chrono::high_resolution_clock::now();
    dli::simulate(height, width, d_prev, d_next);
    auto end = std::chrono::high_resolution_clock::now();
    auto seconds = std::chrono::duration<double>(end - begin).count();
    std::printf("computed step %d in %g s\n", compute_step, seconds);
    d_prev.swap(d_next);
  }
}