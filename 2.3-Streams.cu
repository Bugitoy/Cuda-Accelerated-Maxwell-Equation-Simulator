// Streams

// So far, you’ve learned how to use asynchronous APIs to overlap computation (on the GPU) 
// and I/O (on the CPU). Here’s what our simulator code looks like when we overlap compute 
// and I/O:

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

for (int write_step = 0; write_step < 3; write_step++) 
{
thrust::copy(d_prev.begin(), d_prev.end(), h_prev.begin());

for (int compute_step = 0; compute_step < 750; compute_step++) 
{
simulate(width, height, d_prev, d_next);
d_prev.swap(d_next);
}

dli::store(write_step, height, width, h_prev);

cudaDeviceSynchronize(); 
}
}

// This code is already fast, but there are still further optimizations we can make.
// Right now, the simulator:

// Synchronously copies data from GPU to CPU memory.
// Overlaps computation and I/O to some extent.
// Waits for the copy to finish before proceeding with the computation.
// To improve performance even more, we can also overlap the data copy with the GPU computation, 
// just as we did with the disk I/O.

// Copying Memory Asynchronously

// To achieve this, we need an asynchronous version of thrust::copy. Because Thrust itself doesn’t 
// have direct “magical” powers to copy between the CPU and GPU, it relies on the CUDA Runtime API. 
// The CUDA Runtime API provides asynchronous memory copy functions such as cudaMemcpyAsync, which 
// has the following interface:

cudaError_t cudaMemcpyAsync(
    void*           dst,  // destination pointer
    const void*     src,  // source pointer
    size_t        count,  // number of bytes to copy
    cudaMemcpyKind kind   // direction of copy
  )

/*
Unlike Thrust, cudaMemcpyAsync works on raw pointers and operates in terms of bytes rather than elements. 
This means that we need to calculate the size of the data we want to copy in bytes. Besides that, 
cudaMemcpyAsync also requires an explicit copy direction, which can be one of the following:

cudaMemcpyHostToDevice: instructs cudaMemcpyAsync to copy data from CPU to GPU
cudaMemcpyDeviceToHost: instructs cudaMemcpyAsync to copy data from GPU to CPU
cudaMemcpyDeviceToDevice: instructs cudaMemcpyAsync to copy data from GPU to GPU

You might have also noticed that cudaMemcpyAsync returns a cudaError_t. What kind of error can it be? 
Well, it can actually be any error from previous asynchronous operations.

In the diagram above, we have two asynchronous operations: A and B followed by a cudaMemcpyAsync. Since 
both A and B are computed asynchronously, A can start executing after B was launched. This means that if A 
fails, the error can be caught by cudaMemcpyAsync.

Unfortunately, if we just use cudaMemcpyAsync in our code, we won't get any performance improvement. 

The problem is that all asynchronous operations are ordered on the GPU. Just as when we launched multiple
asynchronous CUB calls, we expected the next invocation to start after the previous one finished, but 
the same thing happened with cudaMemcpyAsync. Subsequent CUB computations wait for cudaMemcpyAsync to 
finish, even though the copy operation is asynchronous.

CUDA Stream

This is an appropriate time to introduce a new concept called a CUDA stream. You can think of a CUDA 
stream as an in-order work queue of things (commands, functions, etc.) that will be executed on the 
GPU. In all the code we've been writing, we've been executing our GPU work in a stream - we just didn't 
know it. If the programmer doesn't specify a stream (which we haven't up to this point), then the work 
is issued to something called the default CUDA stream.

Very importantly, the work issued to a specific CUDA stream is executed synchronously and in-order 
with respect to that stream. This makes sense intuitively as a typical GPU programming flow is to do
something like the following:

1. Copy data from host to device
2. Compute on the device
3. Copy data from device to host

For example, one would not want the compute in step 2 to begin before all the data from step 1 is copied 
to the device. So again, work in the same stream is executed synchronously with respect to that stream. 
However, work in different streams is not synchronized. This is how we can achieve proper concurrency 
among all the parts of the application that can be executed asynchronously. In our example, specifically,
we can use different streams to allow computation and data transfer to be executed concurrently.

On the language level, a CUDA stream is represented by a specific type:

cudaStream_t copy_stream, compute_stream;

To construct a stream, we use the following function:

cudaStreamCreate(&compute_stream);
cudaStreamCreate(&copy_stream)

We can also synchronize the CPU with a given stream, instead of synchronizing with the entire GPU using 
cudaDeviceSynchronize:

cudaStreamSynchronize(compute_stream);
cudaStreamSynchronize(copy_stream)

This is a recommended way to synchronize CPU with GPU, as it allows for more fine-grained control over the 
synchronization.

Finally, you can destroy a stream using the following function:

cudaStreamDestroy(compute_stream);
cudaStreamDestroy(copy_stream)

cudaMemcpyAsync actually has an additional parameter that allows you to specify a stream in which the copy 
operation should be executed:

cudaError_t 
cudaMemcpyAsync(
  void*           dst, 
  const void*     src, 
  size_t        count, 
  cudaMemcpyKind kind,
  cudaStream_t stream = 0 // <- 
);

CUB also allows you to specify which stream to use.

```c++
cudaError_t 
cub::DeviceTransform::Transform(
  IteratorIn 	  input, 
  IteratorOut  output,
  int       num_items,
  TransformOp 	   	op, 
  cudaStream_t stream = 0 // <-
)

It's very common for accelerated libraries to provide an optional stream parameter. The idea is that you as 
a user of these libraries will likely want to overlap their operations with data transfers, CPU computations, 
or even other library calls.

Returning to our simulator, if we just use cudaMemcpyAsync and cub::DeviceTransform::Transform with different 
streams, we'll end up with a data race. If you take a look at each iteration, you'll notice how the second 
iteration step overwrites d_prev while it's being copied to the CPU.

cudaMemcpyAsync(
  thrust::raw_pointer_cast(h_prev.data()),
  thrust::raw_pointer_cast(d_prev.data()), // reads d_prev
  height * width * sizeof(float),
  cudaMemcpyDeviceToHost,
  copy_stream);

simulate(width, height, d_prev, d_next, compute_stream); // reads d_prev, writes d_next
simulate(width, height, d_next, d_prev, compute_stream); // reads d_next, writes d_prev

We can fix this with another level of indirection. We can allocate a staging buffer on the GPU, copy the data 
from d_prev to the staging buffer synchronously, and then copy the data from the staging buffer to the CPU.

thrust::copy(d_prev.begin(), d_prev.end(), d_buffer.begin()); // reads d_prev synchronously

cudaMemcpyAsync(
  thrust::raw_pointer_cast(h_prev.data()),
  thrust::raw_pointer_cast(d_buffer.data()), // reads d_buffer asynchronously
  height * width * sizeof(float),
  cudaMemcpyDeviceToHost,
  copy_stream);

simulate(width, height, d_prev, d_next, compute_stream); // reads d_prev, writes d_next
simulate(width, height, d_next, d_prev, compute_stream); // reads d_next, writes d_prev

But doesn't this defeat the purpose of overlapping computation and IO? We just made the copy synchronous again! 
To answer this question, let's return to our high-level overview of bandwidth provided by different subsystems:

/ Image

Here you can see how the bandwidth of CPU-GPU interconnect is much lower than the bandwidth of GPU memory. 
This means that copying data from GPU to GPU should be significantly faster than copying data from GPU to CPU. 
So this change can still lead to a performance improvement, at the small expense of having a small temporary 
buffer in memory.
*/



