#include "fields/id.h"
// #define FIELD_ID 1
#define CURVE_ID 3
#include "curves/curve_config.cuh"
// #include "fields/field_config.cuh"

#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <cub/device/device_radix_sort.cuh>

#include "fields/field.cuh"
#include "curves/projective.cuh"
#include "gpu-utils/device_context.cuh"

#include "kernels.cu"

class Dummy_Scalar
{
public:
  static constexpr unsigned NBITS = 32;

  unsigned x;
  unsigned p = 10;
  // unsigned p = 1<<30;

  static HOST_DEVICE_INLINE Dummy_Scalar zero() { return {0}; }

  static HOST_DEVICE_INLINE Dummy_Scalar one() { return {1}; }

  friend HOST_INLINE std::ostream& operator<<(std::ostream& os, const Dummy_Scalar& scalar)
  {
    os << scalar.x;
    return os;
  }

  HOST_DEVICE_INLINE unsigned get_scalar_digit(unsigned digit_num, unsigned digit_width) const
  {
    return (x >> (digit_num * digit_width)) & ((1 << digit_width) - 1);
  }

  friend HOST_DEVICE_INLINE Dummy_Scalar operator+(Dummy_Scalar p1, const Dummy_Scalar& p2)
  {
    return {(p1.x + p2.x) % p1.p};
  }

  friend HOST_DEVICE_INLINE bool operator==(const Dummy_Scalar& p1, const Dummy_Scalar& p2) { return (p1.x == p2.x); }

  friend HOST_DEVICE_INLINE bool operator==(const Dummy_Scalar& p1, const unsigned p2) { return (p1.x == p2); }

  static HOST_DEVICE_INLINE Dummy_Scalar neg(const Dummy_Scalar& scalar) { return {scalar.p - scalar.x}; }
  static HOST_INLINE Dummy_Scalar rand_host()
  {
    return {(unsigned)rand() % 10};
    // return {(unsigned)rand()};
  }
};


// typedef field_config::scalar_t test_scalar;
typedef curve_config::scalar_t test_scalar;
typedef curve_config::projective_t test_projective;
typedef curve_config::affine_t test_affine;

typedef int test_t;
// typedef int4 test_t;
// typedef Dummy_Scalar test_t;
// typedef test_projective test_t;
// typedef test_scalar test_t;

int main()
{

  cudaEvent_t start, stop;
  float kernel_time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int N = 1<<25;
  
  void *arr1, *arr2;
  
  cudaMalloc(&arr1, N);
  cudaMalloc(&arr2, N);

  int THREADS = 256;
  int BLOCKS = (N/sizeof(test_t) + THREADS - 1)/THREADS;
  
  //warm up
  device_memory_copy<test_t, sizeof(test_t)><<<BLOCKS, THREADS>>>(arr1, arr2, N);
  segmented_memory_copy<test_t, sizeof(test_t)><<<BLOCKS, THREADS>>>(arr1, arr2, N, 32, 1024);
  cudaDeviceSynchronize();
  std::cout << "cuda err: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

  cudaEventRecord(start, 0);

  device_memory_copy<test_t, sizeof(test_t)><<<BLOCKS, THREADS>>>(arr1, arr2, N);
  // segmented_memory_copy<test_t, sizeof(test_t)><<<BLOCKS, THREADS>>>(arr1, arr2, N, 2, 1024);
  // int elements_per_thread = 8;
  // BLOCKS = (N/sizeof(test_t)/elements_per_thread + THREADS - 1)/THREADS;
  // multi_memory_copy1<test_t, sizeof(test_t)><<<BLOCKS, THREADS>>>(arr1, arr2, N, elements_per_thread);
  // multi_memory_copy2<test_t, sizeof(test_t)><<<BLOCKS, THREADS>>>(arr1, arr2, N, elements_per_thread);
  
  cudaDeviceSynchronize();
  std::cout << "cuda err: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
  cudaEventRecord(stop, 0);
  cudaStreamSynchronize(0);
  cudaEventElapsedTime(&kernel_time, start, stop);
  printf("kernel_time : %.3f ms.\n", kernel_time);

  return 0;
}