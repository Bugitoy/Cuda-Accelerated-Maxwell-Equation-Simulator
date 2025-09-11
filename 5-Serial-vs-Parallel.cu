// Serial vs Parallel

// Segmented Sum

// A segmented sum is defined as taking a single input array and, given a 
// segment size, calculating the sum of each segment. 

// We could build a segmented sum on top of thrust::tabulate. The tabulate 
// algorithm receives a sequence and a function. It then applies this 
// function to index of each element in the sequence, and stores the 
// result into the provided sequence. For example, after the following invocation:

thrust::universal_vector<int> vec(4);
thrust::tabulate(
   thrust::device, vec.begin(), vec.end(), 
   []__host__ __device__(int index) -> int { 
      return index * 2; 
   })

// vec would store {0, 2, 4, 6}. We can use this algorithm to implement our 
// segmented sum as follows:

// vec would store {0, 2, 4, 6}. We can use this algorithm to implement our segmented sum as follows:

thrust::universal_vector<float> sums(num_segments);
thrust::tabulate(
   thrust::device, sums.begin(), sums.end(), 
   []__host__ __device__(int segment_id) -> float {
      return compute_sum_for(segment_id);
   })

// Reduction is a memory-bound algorithm. This means that instead of analyzing its performance in 
// terms of elapsed time, we could take a look at how many bytes does our implementation process 
// in a second. This metric is called achieved throughput. By contrasting it with the peak theoretical 
// bandwidth of our GPU, we'll understand if our implementation is efficient or not.

%%writefile Sources/naive-segmented-sum.cu
#include <cstdio>
#include <chrono>

#include <thrust/tabulate.h>
#include <thrust/execution_policy.h>
#include <thrust/universal_vector.h>

thrust::universal_vector<float> row_temperataures(
    int height, int width,
    const thrust::universal_vector<float>& temp) 
{
    // allocate vector to store sums
    thrust::universal_vector<float> sums(height);

    // take raw pointer to `temp`
    const float *d_temp_ptr = thrust::raw_pointer_cast(temp.data());

    // compute row sum
    thrust::tabulate(thrust::device, sums.begin(), sums.end(), [=]__host__ __device__(int row_id) {
        float sum = 0;
        for (int i = 0; i < width; i++) {
            sum += d_temp_ptr[row_id * width + i];
        }
        return sum; 
    });

    return sums;
}

thrust::universal_vector<float> init(int height, int width) {
  const float low = 15.0;
  const float high = 90.0;
  thrust::universal_vector<float> temp(height * width, low);
  thrust::fill(thrust::device, temp.begin(), temp.begin() + width, high);
  return temp;
}

int main() 
{
    int height = 16;
    int width = 16777216;
    thrust::universal_vector<float> temp = init(height, width);

    auto begin = std::chrono::high_resolution_clock::now();
    thrust::universal_vector<float> sums = row_temperataures(height, width, temp);
    auto end = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(end - begin).count();
    const double gigabytes = static_cast<double>(temp.size() * sizeof(float)) / 1024 / 1024 / 1024;
    const double throughput = gigabytes / seconds;

    std::printf("computed in %g s\n", seconds);
    std::printf("achieved throughput: %g GB/s\n", throughput);
}