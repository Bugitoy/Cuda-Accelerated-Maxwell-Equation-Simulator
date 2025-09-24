// Asynchrony

// Notable developer tools mentioned:

// Nsight Systems:

  /*
  - Provides system-wide view of GPU and CPU activities
  - Visually represents asynchronous compute and memory transfers
  - Allows users to visually identify optimisation opportunities
  */

// Nvidia Tools Extension (NVTX)

  /*
  - Allows users to create custom ranges in Nsight Systems from code
  */

// In the previous sections, we learned about:

  // - Execution spaces (where code runs: CPU vs. GPU)
  // - Memory spaces (where data is stored: host vs. device)
  // - Parallel algorithms (how to run operations in parallel using Thrust)
  
// By combining these concepts, we improved our simulator. Here’s what the updated simulator code looks like:
  
  void simulate(int height, int width, 
                thrust::device_vector<float> &in, 
                thrust::device_vector<float> &out) 
  {
    cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);
    thrust::tabulate(
      thrust::device, out.begin(), out.end(), 
      [=] __host__ __device__(int id) { /* ... */ }
    );
  }
  
  for (int write_step = 0; write_step < 3; write_step++) 
  {
    thrust::copy(d_prev.begin(), d_prev.end(), h_prev.begin());
    dli::store(write_step, height, width, h_prev);
  
    for (int compute_step = 0; compute_step < 3; compute_step++) {
      simulate(height, width, d_prev, d_next);
      d_prev.swap(d_next);
    }
  }

//  In this loop we do the following:
  
  // 1. Copy data from the device (GPU) to the host (CPU).
  // 2. Write the host data to disk.
  // 3. Compute the next state on the GPU.

// Overlapping

// We see that Thrust launches work on the GPU for each simulation step (thrust::tabulate). 
// However, it then waits for the GPU to finish before returning control to the CPU. 
// Because Thrust calls are synchronous, the CPU remains idle whenever the GPU is working. 
// Writing efficient heterogeneous code means utilizing all available resources, including 
// the CPU. In many real-world applications, we can keep the CPU busy at the same time the 
// GPU is computing. This is called overlapping. Instead of waiting idly, the CPU could do 
// something useful, like write data while the GPU is crunching numbers.

// While the GPU is computing the next simulation step, the CPU can be writing out the 
// previous results to disk. This overlap uses both CPU and GPU resources efficiently, 
// reducing the total runtime. Unfortunately, Thrust’s interface doesn’t provide a direct 
// way to separate launching GPU work from waiting for its completion. Under the hood, 
// Thrust calls another library called CUB (CUDA UnBound) to implement its GPU algorithms. 
// If you look at the software stack, you'll see CUB us underneath Thrust. CUB is also a 
// library in it's own right.

// CUB

// If we want finer control to use the CPU while GPU kernels are still running, we need 
// more flexible tools. That’s where direct libraries like CUB come into play.

// Let's take a closer look at CUB:

%%writefile Sources/cub-perf.cu
#include "dli.h"

float simulate(int width,
               int height,
               const thrust::device_vector<float> &in,
                     thrust::device_vector<float> &out,
               bool use_cub) 
{
  cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);
  auto compute = [=] __host__ __device__(int id) {
    const int column = id % width;
    const int row    = id / width;

    // loop over all points in domain (except boundary)
    if (row > 0 && column > 0 && row < height - 1 && column < width - 1)
    {
      // evaluate derivatives
      float d2tdx2 = temp_in(row, column - 1) - 2 * temp_in(row, column) + temp_in(row, column + 1);
      float d2tdy2 = temp_in(row - 1, column) - 2 * temp_in(row, column) + temp_in(row + 1, column);

      // update temperatures
      return temp_in(row, column) + 0.2f * (d2tdx2 + d2tdy2);
    }
    else
    {
      return temp_in(row, column);
    }
  };

  auto begin = std::chrono::high_resolution_clock::now();

  if (use_cub) 
  {
    auto cell_ids = thrust::make_counting_iterator(0);
    cub::DeviceTransform::Transform(cell_ids, out.begin(), width * height, compute);
  }
  else 
  {
    thrust::tabulate(thrust::device, out.begin(), out.end(), compute);
  }
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<float>(end - begin).count();
}

int main()
{
  std::cout << "size, thrust, cub\n";
  for (int size = 1024; size <= 16384; size *= 2)
  {
    int width = size;
    int height = size;
    thrust::device_vector<float> current_temp(height * width, 15.0f);
    thrust::device_vector<float> next_temp(height * width);

    std::cout << size << ", "
              << simulate(width, height, current_temp, next_temp, false) << ", "
              << simulate(width, height, current_temp, next_temp, true) << "\n";
  }
}

// Running this code provides the following stats:

/* 
size, thrust, cub
1024, 7.1588e-05, 1.6872e-05
2048, 0.000160482, 3.868e-06
4096, 0.000592078, 4.418e-06
8192, 0.00232587, 4.188e-06
16384, 0.00954266, 1.3897e-05
*/

// When you run the example cell, you might see some unexpected performance results:

// - When using Thrust, the runtime increases as the number of cells increases. This 
//   makes sense because the GPU is doing more work.
// - When using CUB, the runtime seems almost constant, regardless of how many cells you use.

// Why does this happen?

// 1. Thrust is synchronous, meaning it waits for the GPU to finish all work before giving 
//    control back to the CPU. Naturally, as we scale the workload, the GPU takes longer, 
//    so you see longer total run times.
// 2. CUB, on the other hand, is asynchronous. It launches the GPU kernels and then immediately 
//    returns control to the CPU. That means your CPU timer stops quickly, and it looks like 
//    the GPU work is instantaneous, even though the GPU may still be processing in the background.

// In other words, CUB’s asynchronous behavior explains why the measured runtime doesn’t grow 
// as expected with the problem size. The GPU is still doing the work, but the CPU measurements 
// aren’t accounting for its actual duration.

// This answers the question of how Thrust launches work on the GPU, but what causes Thrust to 
// wait? Thrust uses a function from the CUDA Runtime, cudaDeviceSynchronize(), to wait for the 
// GPU to finish. If we insert this function when using CUB, we should see the same behavior:

%%writefile Sources/cub-perf-sync.cu
#include "dli.h"

float simulate(int width,
               int height,
               const thrust::device_vector<float> &in,
                     thrust::device_vector<float> &out,
               bool use_cub) 
{
  cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);
  auto compute = [=] __host__ __device__(int id) {
    const int column = id % width;
    const int row    = id / width;

    // loop over all points in domain (except boundary)
    if (row > 0 && column > 0 && row < height - 1 && column < width - 1)
    {
      // evaluate derivatives
      float d2tdx2 = temp_in(row, column - 1) - 2 * temp_in(row, column) + temp_in(row, column + 1);
      float d2tdy2 = temp_in(row - 1, column) - 2 * temp_in(row, column) + temp_in(row + 1, column);

      // update temperatures
      return temp_in(row, column) + 0.2f * (d2tdx2 + d2tdy2);
    }
    else
    {
      return temp_in(row, column);
    }
  };

  auto begin = std::chrono::high_resolution_clock::now();

  if (use_cub) 
  {
    auto cell_ids = thrust::make_counting_iterator(0);
    cub::DeviceTransform::Transform(cell_ids, out.begin(), width * height, compute);
    cudaDeviceSynchronize();
  }
  else 
  {
    thrust::tabulate(thrust::device, out.begin(), out.end(), compute);
  }
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<float>(end - begin).count();
}

int main()
{
  std::cout << "size, thrust, cub\n";
  for (int size = 1024; size <= 16384; size *= 2)
  {
    int width = size;
    int height = size;
    thrust::device_vector<float> current_temp(height * width, 15.0f);
    thrust::device_vector<float> next_temp(height * width);

    std::cout << size << ", "
              << simulate(width, height, current_temp, next_temp, false) << ", "
              << simulate(width, height, current_temp, next_temp, true) << "\n";
  }
}

/*
size, thrust, cub
1024, 6.8463e-05, 6.0768e-05
2048, 0.000153217, 0.000151624
4096, 0.000572479, 0.000571006
8192, 0.00224998, 0.00224859
16384, 0.0091184, 0.00886591
*/

// The code above is similar to the previous example, but with the addition of cudaDeviceSynchronize() 
// after the calls to CUB. cudaDeviceSynchronize() is a CUDA Runtime function that causes the CPU to 
// wait until the GPU has finished all work. With cudaDeviceSynchronize(), you can see that it takes 
// the same time for both Thrust and CUB to complete the work.

// We can now use CUB and cudaDeviceSynchronize() to control overlap computation and I/O. This change 
// should result in a significant speedup, as the CPU can now write data to disk while the GPU is 
// computing the next simulation step.

// Exercise: Compute-IO Overlap

// Usage of cub::DeviceTransform for your reference:

// cub::DeviceTransform::Transform(input_iterator, output_iterator, num_items, op)

// In the code below, replace thrust::tabulate with cub::DeviceTransform and use 
// cudaDeviceSynchronize appropriately:

// Original code:

%%writefile Sources/compute-io-overlap.cu
#include "dli.h"

void simulate(int width,
              int height,
              const thrust::device_vector<float> &in,
                    thrust::device_vector<float> &out)
{
  cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);
  thrust::tabulate(out.begin(), out.end(), [=] __device__(int id) {
    return dli::compute(id, temp_in);
  });
}

int main()
{
  int height = 2048;
  int width  = 8192;

  thrust::device_vector<float> d_prev = dli::init(height, width);
  thrust::device_vector<float> d_next(height * width);
  thrust::host_vector<float> h_prev(height * width);

  const int compute_steps = 500;
  const int write_steps = 3;
  for (int write_step = 0; write_step < write_steps; write_step++)
  {
    auto step_begin = std::chrono::high_resolution_clock::now();
    thrust::copy(d_prev.begin(), d_prev.end(), h_prev.begin());

    for (int compute_step = 0; compute_step < compute_steps; compute_step++)
    {
      simulate(width, height, d_prev, d_next);
      d_prev.swap(d_next);
    }

    auto write_begin = std::chrono::high_resolution_clock::now();
    dli::store(write_step, height, width, h_prev);
    auto write_end = std::chrono::high_resolution_clock::now();
    auto write_seconds = std::chrono::duration<double>(write_end - write_begin).count();

    auto step_end = std::chrono::high_resolution_clock::now();
    auto step_seconds = std::chrono::duration<double>(step_end - step_begin).count();
    std::printf("compute + write %d in %g s\n", write_step, step_seconds);
    std::printf("          write %d in %g s\n", write_step, write_seconds);
  }
}

// Solution:

#include "dli.h"

void simulate(int width, int height, const thrust::device_vector<float> &in,
              thrust::device_vector<float> &out) {
  cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);
  cub::DeviceTransform::Transform(
      thrust::make_counting_iterator(0), out.begin(), width * height,
      [=] __host__ __device__(int id) { return dli::compute(id, temp_in); });
}

int main() {
  int height = 2048;
  int width = 8192;

  thrust::device_vector<float> d_prev = dli::init(height, width);
  thrust::device_vector<float> d_next(height * width);
  thrust::host_vector<float> h_prev(height * width);

  const int compute_steps = 750;
  const int write_steps = 3;
  for (int write_step = 0; write_step < write_steps; write_step++) {
    auto step_begin = std::chrono::high_resolution_clock::now();
    thrust::copy(d_prev.begin(), d_prev.end(), h_prev.begin());

    for (int compute_step = 0; compute_step < compute_steps; compute_step++) {
      simulate(width, height, d_prev, d_next);
      d_prev.swap(d_next);
    }

    auto write_begin = std::chrono::high_resolution_clock::now();
    dli::store(write_step, height, width, h_prev);
    auto write_end = std::chrono::high_resolution_clock::now();
    auto write_seconds =
        std::chrono::duration<double>(write_end - write_begin).count();

    cudaDeviceSynchronize();
    auto step_end = std::chrono::high_resolution_clock::now();
    auto step_seconds =
        std::chrono::duration<double>(step_end - step_begin).count();
    std::printf("compute + write %d in %g s\n", write_step, step_seconds);
    std::printf("          write %d in %g s\n", write_step, write_seconds);
  }
}

// Exercise: Use NVTX to annotate your code.

%%writefile Sources/nvtx.cu
#include "dli.h"

void simulate(int width, int height, const thrust::device_vector<float> &in,
              thrust::device_vector<float> &out) 
{
  cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);
  cub::DeviceTransform::Transform(
      thrust::make_counting_iterator(0), out.begin(), width * height,
      [=] __host__ __device__(int id) { return dli::compute(id, temp_in); });
}

int main() 
{
  int height = 2048;
  int width = 8192;

  thrust::device_vector<float> d_prev = dli::init(height, width);
  thrust::device_vector<float> d_next(height * width);
  thrust::host_vector<float> h_prev(height * width);

  const int compute_steps = 750;
  const int write_steps = 3;
  for (int write_step = 0; write_step < write_steps; write_step++) 
  {
    nvtx3::scoped_range r{std::string("write step ") + std::to_string(write_step)};

    {
        // 1. Annotate the "copy" step using nvtx range
        thrust::copy(d_prev.begin(), d_prev.end(), h_prev.begin());
    }

    {
        // 2. Annotate the "compute" step using nvtx range
        for (int compute_step = 0; compute_step < compute_steps; compute_step++) 
        {
        simulate(width, height, d_prev, d_next);
        d_prev.swap(d_next);
        }
    }

    {
        // 3. Annotate the "write" step using nvtx range
        dli::store(write_step, height, width, h_prev);
    }

    {
      // 4. Annotate the "wait" step using nvtx range
      cudaDeviceSynchronize();
    }
  }
}

// The code above stores the output in a file called nvtx in nsight-reports directory.

// If you just completed the Nsight exercise, your UI interface should still be open. 
// If not, review the steps provided in the Nsight exercise.

// Open the new nvtx report and navigate to see the timeline of your application. Identify:

/* 
- when GPU compute is launched
- when CPU writes data on disk
- when CPU waits for GPU
- when data is transferred between CPU and GPU
*/

// Solution:

{
    nvtx3::scoped_range r{"copy"};
    thrust::copy(d_prev.begin(), d_prev.end(), h_prev.begin());
  }
  
  {
    nvtx3::scoped_range r{"compute"};
    for (int compute_step = 0; compute_step < compute_steps; compute_step++) {
      simulate(width, height, d_prev, d_next);
      d_prev.swap(d_next);
    }
  }
  
  {
    nvtx3::scoped_range r{"write"};
    dli::store(write_step, height, width, h_prev);
  }
  
  {
    nvtx3::scoped_range r{"wait"};
    cudaDeviceSynchronize();
  }