#include "fields/id.h"
// #define FIELD_ID 2
#define CURVE_ID 1
#include "curves/curve_config.cuh"
// #include "fields/field_config.cuh"

#include "msm.cu"

#include <chrono>
#include <iostream>
#include <vector>

#include "curves/params/bn254.cuh"
#include "fields/field.cuh"
#include "curves/projective.cuh"
#include "gpu-utils/device_context.cuh"

// using namespace bn254;

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

class Dummy_Projective
{
public:
  Dummy_Scalar x;

  static HOST_DEVICE_INLINE Dummy_Projective zero() { return {0}; }

  static HOST_DEVICE_INLINE Dummy_Projective one() { return {1}; }

  static HOST_DEVICE_INLINE Dummy_Projective to_affine(const Dummy_Projective& point) { return {point.x}; }

  static HOST_DEVICE_INLINE Dummy_Projective from_affine(const Dummy_Projective& point) { return {point.x}; }

  static HOST_DEVICE_INLINE Dummy_Projective neg(const Dummy_Projective& point) { return {Dummy_Scalar::neg(point.x)}; }

  friend HOST_DEVICE_INLINE Dummy_Projective operator+(Dummy_Projective p1, const Dummy_Projective& p2)
  {
    return {p1.x + p2.x};
  }

  // friend HOST_DEVICE_INLINE Dummy_Projective operator-(Dummy_Projective p1, const Dummy_Projective& p2) {
  //   return p1 + neg(p2);
  // }

  friend HOST_INLINE std::ostream& operator<<(std::ostream& os, const Dummy_Projective& point)
  {
    os << point.x;
    return os;
  }

  friend HOST_DEVICE_INLINE Dummy_Projective operator*(Dummy_Scalar scalar, const Dummy_Projective& point)
  {
    Dummy_Projective res = zero();
#ifdef CUDA_ARCH
    UNROLL
#endif
    for (int i = 0; i < Dummy_Scalar::NBITS; i++) {
      if (i > 0) { res = res + res; }
      if (scalar.get_scalar_digit(Dummy_Scalar::NBITS - i - 1, 1)) { res = res + point; }
    }
    return res;
  }

  friend HOST_DEVICE_INLINE bool operator==(const Dummy_Projective& p1, const Dummy_Projective& p2)
  {
    return (p1.x == p2.x);
  }

  static HOST_DEVICE_INLINE bool is_zero(const Dummy_Projective& point) { return point.x == 0; }

  static HOST_INLINE Dummy_Projective rand_host()
  {
    return {(unsigned)rand() % 10};
    // return {(unsigned)rand()};
  }
};

// switch between dummy and real:

// typedef scalar_t test_scalar;
// typedef projective_t test_projective;
// typedef affine_t test_affine;

typedef curve_config::scalar_t test_scalar;
typedef curve_config::projective_t test_projective;
typedef curve_config::affine_t test_affine;

// typedef Dummy_Scalar test_scalar;
// typedef Dummy_Projective test_projective;
// typedef Dummy_Projective test_affine;

int main(int argc, char** argv)
{
  cudaEvent_t start, stop;
  float msm_time;

  int msm_log_size = (argc > 1) ? atoi(argv[1]) : 3;
  int msm_size = 1 << msm_log_size;
  int batch_size = (argc > 2) ? atoi(argv[2]) : 1;
  //   unsigned msm_size = 1<<21;
  int N = batch_size * msm_size;
  int precomp_factor = (argc > 3) ? atoi(argv[3]) : 1;
  int user_c = (argc > 4) ? atoi(argv[4]) : 11;

  printf(
    "running msm curve=%d, 2^%d, batch_size=%d, precomp_factor=%d, c=%d\n", CURVE_ID, msm_log_size, batch_size,
    precomp_factor, user_c);

  test_scalar* scalars = new test_scalar[N];
  test_affine* points = new test_affine[N];

  test_scalar::rand_host_many(scalars, N);
  test_projective::rand_host_many_affine(points, N);
  // for (int i = 0; i < N; i++)
  // {
  //   // scalars[i] = i? scalars[i-1] + test_scalar::one() : test_scalar::zero();
  //   scalars[i] = i? scalars[i-1] : test_scalar::rand_host();
  //   points[i] = test_projective::to_affine(test_projective::generator());
  //   std::cout << i << ": "<< scalars[i] << "\n";
  // }
  

  std::cout << "finished generating" << std::endl;

  // projective_t *short_res = (projective_t*)malloc(sizeof(projective_t));
  // test_projective *large_res = (test_projective*)malloc(sizeof(test_projective));
  test_projective res[batch_size];
  test_projective ref[batch_size];
  // test_projective batched_large_res[batch_size];
  // fake_point *large_res = (fake_point*)malloc(sizeof(fake_point));
  // fake_point batched_large_res[256];

  // short_msm<scalar_t, projective_t, affine_t>(scalars, points, N, short_res);
  // for (unsigned i=0;i<batch_size;i++){
  // large_msm<test_scalar, test_projective, test_affine>(scalars+msm_size*i, points+msm_size*i, msm_size, large_res+i,
  // false); std::cout<<"final result large"<<std::endl; std::cout<<test_projective::to_affine(*large_res)<<std::endl;
  // }

  test_scalar* scalars_d;
  test_affine* points_d;
  test_affine* precomp_points_d;
  test_projective* res_d;
  test_projective* buckets_d;
  test_projective* ref_d;

  cudaMalloc(&scalars_d, sizeof(test_scalar) * N);
  cudaMalloc(&points_d, sizeof(test_affine) * N);
  cudaMalloc(&precomp_points_d, sizeof(test_affine) * N * precomp_factor);
  cudaMalloc(&res_d, sizeof(test_projective) * batch_size);
  cudaMalloc(&ref_d, sizeof(test_projective) * batch_size);
  cudaMemcpy(scalars_d, scalars, sizeof(test_scalar) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(points_d, points, sizeof(test_affine) * N, cudaMemcpyHostToDevice);

  std::cout << "finished copying" << std::endl;

  // batched_large_msm<test_scalar, test_projective, test_affine>(scalars, points, batch_size, msm_size,
  // batched_large_res, false);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  device_context::DeviceContext ctx = {
    stream, // stream
    0,      // device_id
    0,      // mempool
  };
  msm::MSMConfig config = {
    ctx,            // DeviceContext
    N,              // points_size
    precomp_factor, // precompute_factor
    user_c,         // c
    0,              // bitsize
    10,             // large_bucket_factor
    batch_size,     // batch_size
    false,          // are_scalars_on_device
    false,          // are_scalars_montgomery_form
    true,           // are_points_on_device
    false,          // are_points_montgomery_form
    true,           // are_results_on_device
    false,          // is_big_triangle
    true,           // is_async
    true,           // init_buckets
    false,          // return_buckets
    // false,  // segments_reduction
  };

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if (precomp_factor > 1)
    msm::precompute_msm_points<test_affine, test_projective>(points_d, msm_size, config, precomp_points_d);

  // warm up
  msm::msm<test_scalar, test_affine, test_projective>(
    scalars, precomp_factor > 1 ? precomp_points_d : points_d, msm_size, config, res_d, &buckets_d);
  cudaDeviceSynchronize();

  // auto begin1 = std::chrono::high_resolution_clock::now();
  cudaEventRecord(start, stream);
  config.points_size = config.points_size/2;
  config.return_buckets = true;
  msm::msm<test_scalar, test_affine, test_projective>(
    scalars, precomp_factor > 1 ? precomp_points_d : points_d, msm_size/2, config, res_d, &buckets_d);
  // cudaDeviceSynchronize();
  // std::cout << "cuda err: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
  // printf("first half done\n");
  // test_projective* buckets_h = new test_projective[N];
  // cudaMemcpy(buckets_h, buckets_d, sizeof(test_projective) * N, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < N; i++)
  // {
  //   std::cout << buckets_h[i] << "\n";
  // }
  
  // cudaMalloc(&buckets_d, sizeof(test_projective) * (17<<15));
  config.init_buckets = false;
  config.return_buckets = false;
  msm::msm<test_scalar, test_affine, test_projective>(
    scalars + msm_size/2, precomp_factor > 1 ? precomp_points_d + (msm_size/2)*precomp_factor: points_d + msm_size/2, msm_size/2, config, res_d, &buckets_d);
  // cudaDeviceSynchronize();
  // std::cout << "cuda err: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
  // printf("second half done\n");
  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);
    // test_projective* buckets_h = new test_projective[N];
  // cudaMemcpy(buckets_h, buckets_d, sizeof(test_projective) * N, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < N; i++)
  // {
  //   std::cout << test_projective::to_affine(buckets_h[i]) << "\n";
  // }
  cudaEventElapsedTime(&msm_time, start, stop);
  // cudaEvent_t msm_end_event;
  // cudaEventCreate(&msm_end_event);
  // auto end1 = std::chrono::high_resolution_clock::now();
  // auto elapsed1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - begin1);
  printf("msm time : %.3f ms.\n", msm_time);

  // reference
  config.c = 16;
  config.precompute_factor = 1;
  config.is_big_triangle = true;
  config.batch_size = 1;
  config.points_size = msm_size;
  config.init_buckets = true;
  config.return_buckets = false;
  // config.segments_reduction = false;
  for (int i = 0; i < batch_size; i++) {
    msm::msm<test_scalar, test_affine, test_projective>(
      scalars + i * msm_size, points_d + i * msm_size, msm_size, config, ref_d + i, &buckets_d);
  }

  // config.are_results_on_device = false;
  // std::cout << test_projective::to_affine(large_res[0]) << std::endl;
  // auto begin = std::chrono::high_resolution_clock::now();
  // msm::MSM<test_scalar, test_affine, test_projective>(scalars_d, points_d, msm_size, config, large_res);
  // test_reduce_triangle(scalars);
  // test_reduce_rectangle(scalars);
  // test_reduce_single(scalars);
  // test_reduce_var(scalars);
  // auto end = std::chrono::high_resolution_clock::now();
  // auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  // printf("Big Triangle: %.3f seconds.\n", elapsed.count() * 1e-9);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  // std::cout << test_projective::to_affine(large_res[0]) << std::endl;

  cudaMemcpy(res, res_d, sizeof(test_projective) * batch_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(ref, ref_d, sizeof(test_projective) * batch_size, cudaMemcpyDeviceToHost);

  //   reference_msm<test_affine, test_scalar, test_projective>(scalars, points, msm_size);

  // std::cout<<"final results batched large"<<std::endl;
  bool success = true;
  for (unsigned i = 0; i < batch_size; i++) {
    std::cout << test_projective::to_affine(res[i]) << std::endl;
    if (test_projective::to_affine(res[i]) == test_projective::to_affine(ref[i])) {
      std::cout << "good" << std::endl;
    } else {
      std::cout << "miss" << std::endl;
      std::cout << test_projective::to_affine(ref[i]) << std::endl;
      success = false;
    }
  }
  if (success) { std::cout << "success!" << std::endl; }

  // std::cout<<batched_large_res[0]<<std::endl;
  // std::cout<<batched_large_res[1]<<std::endl;
  // std::cout<<projective_t::to_affine(batched_large_res[0])<<std::endl;
  // std::cout<<projective_t::to_affine(batched_large_res[1])<<std::endl;

  // std::cout<<"final result short"<<std::endl;
  // std::cout<<pr<<std::endl;

  return 0;
}