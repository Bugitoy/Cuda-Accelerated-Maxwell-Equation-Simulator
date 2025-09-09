// Naive Heat Transfer

// Below is the code for the naive simulate function. Execute the following 
// two cells to run the code and observe the heat transfer via a visualization. 
// Note the use of thrust::transform and also thrust::make_counting_iterator(0).

%%writefile Sources/naive-heat-2D.cu
#include "dli.h"

void simulate(int height, int width, 
              const thrust::universal_vector<float> &in,
                    thrust::universal_vector<float> &out) 
{
  const float *in_ptr = thrust::raw_pointer_cast(in.data());

  auto cell_indices = thrust::make_counting_iterator(0);
  thrust::transform(
      thrust::device, cell_indices, cell_indices + in.size(), out.begin(),
      [in_ptr, height, width] __host__ __device__(int id) {
        int column = id % width;
        int row = id / width;

        if (row > 0 && column > 0 && row < height - 1 && column < width - 1) {
          float d2tdx2 = in_ptr[(row) * width + column - 1] - 2 * in_ptr[row * width + column] + in_ptr[(row) * width + column + 1];
          float d2tdy2 = in_ptr[(row - 1) * width + column] - 2 * in_ptr[row * width + column] + in_ptr[(row + 1) * width + column];

          return in_ptr[row * width + column] + 0.2f * (d2tdx2 + d2tdy2);
        } else {
          return in_ptr[row * width + column];
        }
      });
}

// Thrust Tabulate

// The Thrust library has another function called tabulate that applies an 
// operator to element indices and stores the result at the same index. It is 
// essentially the equivalent of the above example of transformation of the counting 
// iterator.

%%writefile Sources/tabulate.cu
#include "dli.h"

void simulate(int height, int width,
              const thrust::universal_vector<float> &in,
                    thrust::universal_vector<float> &out)
{
  const float *in_ptr = thrust::raw_pointer_cast(in.data());

  thrust::tabulate(
    thrust::device, out.begin(), out.end(), 
    [in_ptr, height, width] __host__ __device__(int id) {
      int column = id % width;
      int row = id / width;

      if (row > 0 && column > 0 && row < height - 1 && column < width - 1) {
        float d2tdx2 = in_ptr[(row) * width + column - 1] - 2 * in_ptr[row * width + column] + in_ptr[(row) * width + column + 1];
        float d2tdy2 = in_ptr[(row - 1) * width + column] - 2 * in_ptr[row * width + column] + in_ptr[(row + 1) * width + column];

        return in_ptr[row * width + column] + 0.2f * (d2tdx2 + d2tdy2);
      } else {
        return in_ptr[row * width + column];
      }
    });
}

// Code Reuse

// You may have noticed that in the body of the function we are doing some involved 
// offset arithmetic to index into the correct values of the array that we're working 
// with. Since we are in C++, we are doing these offset calculations in row major order 
// and we can use the make_pair function to do this arithmetic for us once, instead of 
// calculating row and column repeatedly.

// Execute the following two cells to illustrate the use of make_pair to create the 
// function row_col.

%%writefile Sources/std-pair.cu
#include "dli.h"

__host__ __device__
std::pair<int, int> row_col(int id, int width) {
    return std::make_pair(id / width, id % width);
}

void simulate(int height, int width,
              const thrust::universal_vector<float> &in,
                    thrust::universal_vector<float> &out)
{
  const float *in_ptr = thrust::raw_pointer_cast(in.data());

  thrust::tabulate(
    thrust::device, out.begin(), out.end(), 
    [in_ptr, height, width] __host__ __device__(int id) {
      auto [row, column] = row_col(id, width);

      if (row > 0 && column > 0 && row < height - 1 && column < width - 1) {
        float d2tdx2 = in_ptr[(row) * width + column - 1] - 2 * in_ptr[row * width + column] + in_ptr[(row) * width + column + 1];
        float d2tdy2 = in_ptr[(row - 1) * width + column] - 2 * in_ptr[row * width + column] + in_ptr[(row + 1) * width + column];

        return in_ptr[row * width + column] + 0.2f * (d2tdx2 + d2tdy2);
      } else {
        return in_ptr[row * width + column];
      }
    });
}

// If you didn't change any code, you'll have encountered a warning message similar to the following:

// calling a __host__ function("make_pair") from a __host__ __device__ function("row_col") is not allowed. 
//
// std::make_pair is a host function, not a device function. Thinking back to the earlier part of our 
// course, a host function is compiled to run on the host, and if there is no equivalent device function, 
// that code will NOT run on the device. That is what the error message is telling us, i.e., we don't 
// have any device function called std::make_pair and therefore it can't run on the device.

//Fortunately NVIDIA has implemented many of these standard types in CUDA via the libcu++ library, 
// and for the std:: types implemented in libcu++, it's as simple as prepending cuda:: in front of 
// the std:: types you're using.

//Notice below the code is identical to the previous example, with the two small changes of prepending 
// cuda:: in front of both std::pair and std::make_pair.

%%writefile Sources/pair.cu
#include "dli.h"

__host__ __device__
cuda::std::pair<int, int> row_col(int id, int width) {
    return cuda::std::make_pair(id / width, id % width);
}

void simulate(int height, int width,
              const thrust::universal_vector<float> &in,
                    thrust::universal_vector<float> &out)
{
  const float *in_ptr = thrust::raw_pointer_cast(in.data());

  thrust::tabulate(
    thrust::device, out.begin(), out.end(), 
    [in_ptr, height, width] __host__ __device__(int id) {
      auto [row, column] = row_col(id, width);

      if (row > 0 && column > 0 && row < height - 1 && column < width - 1) {
        float d2tdx2 = in_ptr[(row) * width + column - 1] - 2 * in_ptr[row * width + column] + in_ptr[(row) * width + column + 1];
        float d2tdy2 = in_ptr[(row - 1) * width + column] - 2 * in_ptr[row * width + column] + in_ptr[(row + 1) * width + column];

        return in_ptr[row * width + column] + 0.2f * (d2tdx2 + d2tdy2);
      } else {
        return in_ptr[row * width + column];
      }
    });
}

// mdspan

// pair is not the only vocabulary type that will improve our code. Consider 
// how we are manually flattening the 2D coordinates to access the 1D array. 
// This approach is error-prone and makes the code difficult to read and 
// understand. Additionally, consider situations where more than 2D coordinates 
// are being used; the complexity of the pointer indexing will rapidly increase!

// For the section exercise, we'll explore the use of mdspan, which is a 
// vocabulary type used to make this type of indexing much easier. See a 
// simple use of mdspan below.

%%writefile Sources/mdspan-intro.cu

#include <cuda/std/mdspan>
#include <cuda/std/array>
#include <cstdio>

int main() {
  cuda::std::array<int, 6> sd {0, 1, 2, 3, 4, 5};
  cuda::std::mdspan md(sd.data(), 2, 3);

  std::printf("md(0, 0) = %d\n", md(0, 0)); // 0
  std::printf("md(1, 2) = %d\n", md(1, 2)); // 5

  std::printf("size   = %zu\n", md.size());    // 6
  std::printf("height = %zu\n", md.extent(0)); // 2
  std::printf("width  = %zu\n", md.extent(1)); // 3
}

