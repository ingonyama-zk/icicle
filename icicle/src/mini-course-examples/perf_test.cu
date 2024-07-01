#include "fields/id.h"
// #define FIELD_ID 2
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


typedef curve_config::scalar_t test_scalar;
typedef curve_config::projective_t test_projective;
typedef curve_config::affine_t test_affine;

typedef Dummy_Scalar test_t;
// typedef test_projective test_t;
// typedef test_scalar test_t;

int main()
{

  cudaEvent_t start, stop;
  float kernel_time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int N = 1<<20;
  // int N = 1<<3;

  test_t* buckets_h = new test_t[N];
  unsigned* indices_h = new unsigned[N];
  unsigned* sizes_h = new unsigned[N];

  for (int i = 0; i < N; i++)
  {
    indices_h[i] = static_cast<unsigned>(i);
    sizes_h[i] = static_cast<unsigned>(std::rand())%20;
    // sizes_h[i] = 10;
    buckets_h[i] = i<100? test_t::rand_host() : buckets_h[i-100];
    if (i<10) std::cout << indices_h[i] << " " << sizes_h[i] << " " << buckets_h[i] << std::endl;
  }
  
  test_t *buckets_d, *buckets2_d;
  unsigned *sizes_d, *indices_d;

  cudaMalloc(&buckets_d, sizeof(test_t) * N);
  cudaMalloc(&buckets2_d, sizeof(test_t) * N);
  cudaMalloc(&sizes_d, sizeof(unsigned) * N);
  cudaMalloc(&indices_d, sizeof(unsigned) * N);
  
  cudaMemcpy(buckets_d, buckets_h, sizeof(test_t) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(sizes_d, sizes_h, sizeof(unsigned) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(indices_d, indices_h, sizeof(unsigned) * N, cudaMemcpyHostToDevice);

  int THREADS = 256;
  int BLOCKS = (N + THREADS - 1)/THREADS;
  
  //warm up
  bucket_acc_naive<<<BLOCKS, THREADS>>>(buckets_d, indices_d, sizes_d, N);
  cudaDeviceSynchronize();
  std::cout << "cuda err: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

  // cudaEventRecord(start, 0);

  
  unsigned* sorted_sizes;
  cudaMalloc(&sorted_sizes, sizeof(unsigned) * N);

  unsigned* sorted_indices;
  cudaMalloc(&sorted_indices, sizeof(unsigned) * N);
  unsigned* sort_indices_temp_storage{};
  size_t sort_indices_temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(
    sort_indices_temp_storage, sort_indices_temp_storage_bytes, sizes_d,
    sorted_sizes, indices_d, sorted_indices, N, 0);
  cudaMalloc(&sort_indices_temp_storage, sort_indices_temp_storage_bytes);
  cub::DeviceRadixSort::SortPairsDescending(
    sort_indices_temp_storage, sort_indices_temp_storage_bytes, sizes_d,
    sorted_sizes, indices_d, sorted_indices, N, 0);
  cudaFree(sort_indices_temp_storage);
  
  test_t* sorted_buckets;
  cudaMalloc(&sorted_buckets, sizeof(test_t) * N);
  unsigned* sort_buckets_temp_storage{};
  size_t sort_buckets_temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(
    sort_buckets_temp_storage, sort_buckets_temp_storage_bytes, sizes_d,
    sorted_sizes, buckets_d, sorted_buckets, N, 0);
  cudaMalloc(&sort_buckets_temp_storage, sort_buckets_temp_storage_bytes);
  cub::DeviceRadixSort::SortPairsDescending(
    sort_buckets_temp_storage, sort_buckets_temp_storage_bytes, sizes_d,
    sorted_sizes, buckets_d, sorted_buckets, N, 0);
  cudaFree(sort_buckets_temp_storage);

  cudaEventRecord(start, 0);

  // bucket_acc_naive<<<BLOCKS, THREADS>>>(buckets_d, indices_d, sizes_d, N);
  // bucket_acc_reg<<<BLOCKS, THREADS>>>(buckets_d, indices_d, sizes_d, N);
  // bucket_acc_reg<<<BLOCKS, THREADS>>>(buckets_d, sorted_indices, sorted_sizes, N);
  // bucket_acc_reg<<<BLOCKS, THREADS>>>(sorted_buckets, indices_d, sorted_sizes, N);
  // bucket_acc_compute_baseline<<<BLOCKS, THREADS>>>(buckets_d, indices_d, sizes_d, N);
  // bucket_acc_memory_baseline<<<BLOCKS, THREADS>>>(buckets_d, buckets2_d, indices_d, N);

  simple_memory_copy<<<BLOCKS, THREADS>>>(buckets_d, buckets2_d, N);
  
  cudaDeviceSynchronize();
  std::cout << "cuda err: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
  cudaEventRecord(stop, 0);
  cudaStreamSynchronize(0);
  cudaEventElapsedTime(&kernel_time, start, stop);
  printf("kernel_time : %.3f ms.\n", kernel_time);

  cudaMemcpy(buckets_h, buckets_d, sizeof(test_t) * N, cudaMemcpyDeviceToHost);
  // cudaMemcpy(buckets_h, sorted_buckets, sizeof(test_t) * N, cudaMemcpyDeviceToHost);
  // cudaMemcpy(sizes_h, sorted_indices, sizeof(unsigned) * N, cudaMemcpyDeviceToHost);

  // printf("res:\n");
  // for (size_t i = 0; i < 8; i++)
  // {
  //   std::cout << buckets_h[i] << "\n";
  //   // std::cout << sizes_h[i] << "\n";
  // }
  // printf("\n");
  // printf("C test: ");
  // for (size_t i = 0; i < 8; i++)
  // {
  //   std::cout << Cb_h[i] << ", ";
  // }
  // printf("\n");
  // printf("C ref: ");
  // for (size_t i = 0; i < 8; i++)
  // {
  //   std::cout << C_d[i] << ", ";
  //   // std::cout << C_h[i] << ", ";
  // }
  // printf("\n");

  return 0;
}