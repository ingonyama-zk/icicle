#include "fields/id.h"
#define FIELD_ID 1001
// #define CURVE_ID 3
// #include "curves/curve_config.cuh"
#include "fields/field_config.cuh"

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


typedef field_config::scalar_t test_scalar;
// typedef curve_config::scalar_t test_scalar;
// typedef curve_config::projective_t test_projective;
// typedef curve_config::affine_t test_affine;

// typedef int test_t;
// typedef int4 test_t;
// typedef Dummy_Scalar test_t;
// typedef test_projective test_t;
typedef test_scalar test_t;

int main()
{

  cudaEvent_t start, stop;
  float kernel_time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int N = 1<<11;
  int N2 = N*N;
  
  test_t* arr1_h = new test_t[N2];
  test_t* arr2_h = new test_t[N2];

  test_t *arr1_d, *arr2_d;
  
  cudaMalloc(&arr1_d, N2*sizeof(test_t));
  cudaMalloc(&arr2_d, N2*sizeof(test_t));

  for (int i = 0; i < N2; i++)
  {
    arr1_h[i] = i > 100? arr1_h[i-100] : test_t::rand_host();
  }
  
  cudaMemcpy(arr1_d, arr1_h, sizeof(test_t) * N2, cudaMemcpyHostToDevice);

  int THREADS = 256;
  int BLOCKS = (N2 + THREADS - 1)/THREADS;
  
  //warm up
  simple_memory_copy<<<BLOCKS, THREADS>>>(arr1_d, arr2_d, N2);
  shmem_transpose<<<BLOCKS, THREADS>>>(arr1_d, arr2_d, N);
  cudaDeviceSynchronize();
  std::cout << "cuda err: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

  cudaEventRecord(start, 0);

  simple_memory_copy<<<BLOCKS, THREADS>>>(arr1_d, arr2_d, N2);
  // naive_transpose_write<<<BLOCKS, THREADS>>>(arr1_d, arr2_d, N);
  // naive_transpose_read<<<BLOCKS, THREADS>>>(arr1_d, arr2_d, N);
  // shmem_transpose<<<BLOCKS, THREADS>>>(arr1_d, arr2_d, N);
  
  cudaDeviceSynchronize();
  std::cout << "cuda err: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
  cudaEventRecord(stop, 0);
  cudaStreamSynchronize(0);
  cudaEventElapsedTime(&kernel_time, start, stop);
  printf("kernel_time : %.3f ms.\n", kernel_time);

  return 0;
}