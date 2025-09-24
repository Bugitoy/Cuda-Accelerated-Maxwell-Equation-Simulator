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

// Running this code provides the following stats:

// computed in 0.597472 s
// achieved throughput: 1.67372 GB/s

// Let's take a look at the achieved throughput and contrast it 
// with maximal bandwidth. Out implementation achieves less than 
// a percent of what GPU can provide. The reason our implementation 
// underperforms is due to the way we used thrust::tabulate:

thrust::tabulate(thrust::device, sums.begin(), sums.end(), [=]__host__ __device__(int segment_id) {
    float sum = 0;
    for (int i = 0; i < segment_size; i++) {
        sum += d_values_ptr[segment_id * segment_size + i];
    }
    return sum; 
})

// Reduce by Key

// GPUs are massively parallel processors. That said, code that ends up being 
// executed by GPU doesn't get magically parallelized. The for loop in the operator 
// we provided to thrust::tabulate is executed sequentially. Tabulate could process 
// each of the 16 elements in parallel, while the operator processes over 16 million 
// elements. To fix performance, let's try increasing parallelism.

// To do that, we can try the thrust::reduce_by_key algorithm, which is a generalization 
// of the thrust::reduce algorithm. Instead of reducing the sequence into a single value, 
// it allows you to reduce segments of values. To distinguish these segments, you have to 
// provide keys. Consecutive keys that are equal form a segment. As the output, reduce_by_key
//  returns one value per segment.

For example:

int in_keys[] = {1, 1, 1, 3, 3};
int in_vals[] = {1, 2, 3, 4, 5};
int out_keys[2];
int out_vals[2];

thrust::reduce_by_key(in_keys, in_keys + 5, in_vals, out_keys, out_vals);
// out_keys = {1, 3}
// out_vals = {6, 9}

// Lets try to frame our segmented sum in terms of reduce by key:

%%writefile Sources/reduce-by-key.cu
#include "dli.h"

thrust::universal_vector<float> row_temperatures(
    int height, int width,
    thrust::universal_vector<int>& row_ids,
    thrust::universal_vector<float>& temp)
{
    thrust::universal_vector<float> sums(height);
    thrust::reduce_by_key(
        thrust::device, 
        row_ids.begin(), row_ids.end(),   // input keys 
        temp.begin(),                     // input values
        thrust::make_discard_iterator(),  // output keys
        sums.begin());                    // output values

    return sums;
}

// Running this code provides the following stats:

// computed in 2.33905 s
// achieved throughput: 0.427524 GB/s
// maximal bandwidth: 298.083 GB/s
// row 0: { 90, 90, ..., 90 } = 1.50995e+09
// row 1: { 15, 15, ..., 15 } = 2.51658e+08
// row 2: { 15, 15, ..., 15 } = 2.51658e+08

// We are not interested in output keys, so we made a discard iterator. 
// This technique often helps you save memory bandwidth when you don't need certain parts 
// of the algorithm's output. Speaking of bandwidth, we've got much better results now. 
// That's because we eliminated the serialization that was dominating execution time. 
// However, there's still an issue: Now we are reading keys.