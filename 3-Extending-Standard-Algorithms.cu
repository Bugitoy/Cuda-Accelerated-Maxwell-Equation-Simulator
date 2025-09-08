// Inefficient implementation of max difference:

%%writefile Sources/naive-max-diff.cu
#include "dli.h"

float naive_max_change(const thrust::universal_vector<float>& a, 
                       const thrust::universal_vector<float>& b) 
{
    // allocate vector to store `a - b`
    thrust::universal_vector<float> unnecessarily_materialized_diff(a.size());

    // compute products
    thrust::transform(thrust::device, 
                      a.begin(), a.end(),                       // first input sequence
                      b.begin(),                                // second input sequence
                      unnecessarily_materialized_diff.begin(),  // result
                      []__host__ __device__(float x, float y) { // transformation (abs diff)
                         return abs(x - y); 
                      });

    // compute max difference
    return thrust::reduce(thrust::device, 
                          unnecessarily_materialized_diff.begin(), 
                          unnecessarily_materialized_diff.end(), 
                          0.0f, thrust::maximum<float>{});
}

int main() 
{
    float k = 0.5;
    float ambient_temp = 20;
    thrust::universal_vector<float> temp[] = {{ 42, 24, 50 }, { 0, 0, 0}};
    auto transformation = [=] __host__ __device__ (float temp) { return temp + k * (ambient_temp - temp); };

    std::printf("step  max-change\n");
    for (int step = 0; step < 3; step++) {
        thrust::universal_vector<float> &current = temp[step % 2];
        thrust::universal_vector<float> &next = temp[(step + 1) % 2];

        thrust::transform(thrust::device, current.begin(), current.end(), next.begin(), transformation);
        std::printf("%d     %.2f\n", step, naive_max_change(current, next));
    }
}

//*Iterators*//

// The following code demonstrates using a pointer to access data in an array.

%%writefile Sources/pointer.cu
#include "dli.h"

int main() 
{
    std::array<int, 3> a{ 0, 1, 2 };

    int *pointer = a.data();

    std::printf("pointer[0]: %d\n", pointer[0]); // prints 0
    std::printf("pointer[1]: %d\n", pointer[1]); // prints 1
}

// Simple Counting Iterator

// C++ allows operator overloading. This means that we can define what operators 
// such as * or ++ do. The concept of an iterator builds on top of this idea. 
// With this, we don't even need an underlying container. Here's an example of 
// how we can create an infinite sequence without allocating a single byte. Note 
// the redefinition of the square brackets [] operator.

%%writefile Sources/counting.cu
#include "dli.h"

struct counting_iterator 
{
  int operator[](int i) 
  {
    return i;
  }
};

int main() 
{
  counting_iterator it;

  std::printf("it[0]: %d\n", it[0]); // prints 0
  std::printf("it[1]: %d\n", it[1]); // prints 1
}

// Simple Transform Iterator

// Below we again redefine the square brackets [] operator, but this time instead of 
// simple counting, we multiple each input value times 2. This is an example of 
// applying a simple function to the input before returning a value.

%%writefile Sources/transform.cu
#include "dli.h"

struct transform_iterator 
{
  int *a;

  int operator[](int i) 
  {
    return a[i] * 2;
  }
};

int main() 
{
  std::array<int, 3> a{ 0, 1, 2 };

  transform_iterator it{a.data()};

  std::printf("it[0]: %d\n", it[0]); // prints 0 (0 * 2)
  std::printf("it[1]: %d\n", it[1]); // prints 2 (1 * 2)
}

// Simple Zip Iterator

// We can continue to redefine the square brackets [] operator, to combine 
// multiple sequences. The zip iterator below takes two arrays and combines 
// them into a sequence of tuples.

%%writefile Sources/zip.cu
#include "dli.h"

struct zip_iterator 
{
  int *a;
  int *b;

  std::tuple<int, int> operator[](int i) 
  {
    return {a[i], b[i]};
  }
};

int main() 
{
  std::array<int, 3> a{ 0, 1, 2 };
  std::array<int, 3> b{ 5, 4, 2 };

  zip_iterator it{a.data(), b.data()};

  std::printf("it[0]: (%d, %d)\n", std::get<0>(it[0]), std::get<1>(it[0])); // prints (0, 5)
  std::printf("it[0]: (%d, %d)\n", std::get<0>(it[1]), std::get<1>(it[1])); // prints (1, 4)
}

// Combining Input Iterators

// One very powerful feature of iterators is that you can combine them with 
// each other. If we think about our original code above where we computed 
// the absolute value of the difference between each element in two arrays, 
// you can see below that we can combine, or nest, the zip_iterator with the 
// transform_iterator to first combine the two arrays a and b with zip, and 
// then transform them via the transform iterator with our custom operation 
// to compute the absolute value of the differences of each successive element 
// in the original arrays a and b.

%%writefile Sources/transform-zip.cu
#include "dli.h"

struct zip_iterator 
{
  int *a;
  int *b;

  std::tuple<int, int> operator[](int i) 
  {
    return {a[i], b[i]};
  }
};

struct transform_iterator 
{
  zip_iterator zip;

  int operator[](int i) 
  {
    auto [a, b] = zip[i];
    return abs(a - b);
  }
};

int main() 
{
  std::array<int, 3> a{ 0, 1, 2 };
  std::array<int, 3> b{ 5, 4, 2 };

  zip_iterator zip{a.data(), b.data()};
  transform_iterator it{zip};

  std::printf("it[0]: %d\n", it[0]); // prints 5
  std::printf("it[0]: %d\n", it[1]); // prints 3
}

// Transform Output Iterator

// The concept of iterators is not limited to inputs alone. With another 
// level of indirection one can transform values that are written into a 
// transform output iterator. Note in the code below, both = and [] operators 
// are being redefined.

%%writefile Sources/transform-output.cu
#include "dli.h"

struct wrapper
{
   int *ptr; 

   void operator=(int value) {
      *ptr = value / 2;
   }
};

struct transform_output_iterator 
{
  int *a;

  wrapper operator[](int i) 
  {
    return {a + i};
  }
};

int main() 
{
  std::array<int, 3> a{ 0, 1, 2 };
  transform_output_iterator it{a.data()};

  it[0] = 10;
  it[1] = 20;

  std::printf("a[0]: %d\n", a[0]); // prints 5
  std::printf("a[1]: %d\n", a[1]); // prints 10
}

// Discard Iterator

%%writefile Sources/discard.cu
#include "dli.h"

struct wrapper
{
   void operator=(int value) {
      // discard value
   }
};

struct discard_iterator 
{
  wrapper operator[](int i) 
  {
    return {};
  }
};

int main() 
{
  discard_iterator it{};

  it[0] = 10;
  it[1] = 20;
}

// CUDA Fancy Iterators

//CUDA Core Libraries provide a variety of iterators. 
// Let's take a look at some of them as we try to improve the performance 
// of our inner product implementation. The first step is computing the absolute 
// differences of corresponding vector components. To do that, we have to somehow 
// make operator `*` return a pair of values, one taken from `a` and another taken from `b`.
// This functionality is covered by `thrust::zip_iterator`.

%%writefile Sources/zip.cu
#include "dli.h"

int main() 
{
    // allocate and initialize input vectors
    thrust::universal_vector<float> a{ 31, 22, 35 };
    thrust::universal_vector<float> b{ 25, 21, 27 };

    // zip two vectors into a single iterator
    auto zip = thrust::make_zip_iterator(a.begin(), b.begin());

    thrust::tuple<float, float> first = *zip;
    std::printf("first: (%g, %g)\n", thrust::get<0>(first), thrust::get<1>(first));

    zip++;

    thrust::tuple<float, float> second = *zip;
    std::printf("second: (%g, %g)\n", thrust::get<0>(second), thrust::get<1>(second));
}

// However, we don't need just pairs of vector components. We need their absolute 
// differences. A thrust::transform_iterator allows us to attach a function to the 
// dereferencing of an iterator. When combined with the zip iterator, it allows us 
// to compute absolute differences without materializing them in memory.

%%writefile Sources/transform.cu
#include "dli.h"

int main() 
{
    thrust::universal_vector<float> a{ 31, 22, 35 };
    thrust::universal_vector<float> b{ 25, 21, 27 };

    auto zip = thrust::make_zip_iterator(a.begin(), b.begin());
    auto transform = thrust::make_transform_iterator(zip, []__host__ __device__(thrust::tuple<float, float> t) {
        return abs(thrust::get<0>(t) - thrust::get<1>(t));
    });

    std::printf("first: %g\n", *transform); // absolute difference of `a[0] = 31` and `b[0] = 25`

    transform++;

    std::printf("second: %g\n", *transform); // absolute difference of `a[1] = 22` and `b[1] = 21`
}

// The only remaining part is computing a maximum value. We already know how to do 
// that using thrust::reduce. Now, the code below is functionally equivalent to our starting 
// code example at the top of this notebook. Notice we have eliminated the need for the 
// temporary array to store the differences.

%%writefile Sources/optimized-max-diff.cu
#include "dli.h"

float max_change(const thrust::universal_vector<float>& a, 
                 const thrust::universal_vector<float>& b) 
{
    auto zip = thrust::make_zip_iterator(a.begin(), b.begin());
    auto transform = thrust::make_transform_iterator(zip, []__host__ __device__(thrust::tuple<float, float> t) {
        return abs(thrust::get<0>(t) - thrust::get<1>(t));
    });

    // compute max difference
    return thrust::reduce(thrust::device, transform, transform + a.size(), 0.0f, thrust::maximum<float>{});
}

int main() 
{
    float k = 0.5;
    float ambient_temp = 20;
    thrust::universal_vector<float> temp[] = {{ 42, 24, 50 }, { 0, 0, 0}};
    auto transformation = [=] __host__ __device__ (float temp) { return temp + k * (ambient_temp - temp); };

    std::printf("step  max-change\n");
    for (int step = 0; step < 3; step++) {
        thrust::universal_vector<float> &current = temp[step % 2];
        thrust::universal_vector<float> &next = temp[(step + 1) % 2];

        thrust::transform(thrust::device, current.begin(), current.end(), next.begin(), transformation);
        std::printf("%d     %.2f\n", step, max_change(current, next));
    }
}

// Recall that this code is memory bound, so we'd expect that the elimination of unnecessary 
// memory usage (in this case, temporary storage to hold the differences) should improve our 
// performance. Let's evaluate performance of this implementation to see if it matches our intuition. 
// To do that, we'll allocate much larger vectors.

%%writefile Sources/naive-vs-iterators.cu
#include "dli.h"

float naive_max_change(const thrust::universal_vector<float>& a, 
                       const thrust::universal_vector<float>& b) 
{
    thrust::universal_vector<float> diff(a.size());
    thrust::transform(thrust::device, a.begin(), a.end(), b.begin(), diff.begin(),
                      []__host__ __device__(float x, float y) {
                         return abs(x - y); 
                      });
    return thrust::reduce(thrust::device, diff.begin(), diff.end(), 0.0f, thrust::maximum<float>{});
}

float max_change(const thrust::universal_vector<float>& a, 
                 const thrust::universal_vector<float>& b) 
{
    auto zip = thrust::make_zip_iterator(a.begin(), b.begin());
    auto transform = thrust::make_transform_iterator(zip, []__host__ __device__(thrust::tuple<float, float> t) {
        return abs(thrust::get<0>(t) - thrust::get<1>(t));
    });
    return thrust::reduce(thrust::device, transform, transform + a.size(), 0.0f, thrust::maximum<float>{});
}

int main() 
{
    // allocate vectors containing 2^28 elements
    thrust::universal_vector<float> a(1 << 28);
    thrust::universal_vector<float> b(1 << 28);

    thrust::sequence(a.begin(), a.end());
    thrust::sequence(b.rbegin(), b.rend());

    auto start_naive = std::chrono::high_resolution_clock::now();
    naive_max_change(a, b);
    auto end_naive = std::chrono::high_resolution_clock::now();
    const double naive_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_naive - start_naive).count();

    auto start = std::chrono::high_resolution_clock::now();
    max_change(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    const double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::printf("iterators are %g times faster than naive approach\n", naive_duration / duration);
}