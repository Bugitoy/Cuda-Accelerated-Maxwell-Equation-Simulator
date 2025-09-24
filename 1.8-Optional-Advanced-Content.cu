// Optional Advanced Content

// Exercise: Computing Run Length Encode

// Limitations

// Following the previous recipes may occasionally yield unexpected results, 
// due to certain limitations imposed by CUDA on C++. Let’s examine one of these 
// limitations.

// In C++, standard algorithms don’t require the use of lambdas. For example, you might 
// want to extract a lambda into a named function for reuse. Let’s attempt to do that 
// with our transformation:

%%writefile Sources/host-function-pointers.cu
#include <thrust/transform.h>
#include <thrust/universal_vector.h>
#include <cstdio>

__host__ __device__ float transformation(float x) {
  return 2 * x + 1;
}

int main() {
  thrust::universal_vector<float> vec{ 1, 2, 3 };

  thrust::transform(vec.begin(), vec.end(), vec.begin(), transformation);

  std::printf("%g %g %g\n", vec[0], vec[1], vec[2]);
}

// Unfortunately, if you run this code, you'll likely see an exception saying something about 
// invalid program counter. However, invoking this named function from within a lambda works 
// just fine:

%%writefile Sources/host-function-pointers-fix.cu
#include <thrust/transform.h>
#include <thrust/universal_vector.h>
#include <cstdio>

__host__ __device__ float transformation(float x) {
  return 2 * x + 1;
}

int main() {
  thrust::universal_vector<float> vec{ 1, 2, 3 };

  thrust::transform(vec.begin(), vec.end(), vec.begin(), [] __host__ __device__ (float x) { 
    return transformation(x); 
  });

  std::printf("%g %g %g\n", vec[0], vec[1], vec[2]);
}

// So, what's going on? This issue is related to one of the CUDA limitations:

// It is not allowed to take the address of a device function in host code.

// We invoke thrust::transform on the host (CPU code). When we pass transformation 
// function to thrust::transform, C++ implicitly takes its address. But as we just 
// learned, taking address of __device__ function is not allowed on the host.

// That should shed some light on why the version with lambda works. Lambda is not 
// a function, it's a function object. Code below illustrates what lambda actually 
// looks like:

%%writefile Sources/function-objects.cu

#include <thrust/execution_policy.h>
#include <thrust/universal_vector.h>
#include <thrust/transform.h>
#include <cstdio>

struct transformation {
  __host__ __device__ float operator()(float x) {
    return 2 * x + 1;
  }
};

int main() {
  thrust::universal_vector<float> vec{ 1, 2, 3 };

  thrust::transform(thrust::device, vec.begin(), vec.end(), vec.begin(), transformation{});

  std::printf("%g %g %g\n", vec[0], vec[1], vec[2]);
}

This code passes an object, and the __device__ operator is not referenced on the host.